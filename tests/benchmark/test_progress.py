"""Checkpoint-linked continuation without a learner format or live simulator state."""

import json
import unittest
from copy import deepcopy
from types import SimpleNamespace as NS

import numpy as np

from parkour_lab.benchmark.progress import capture_progress, restore_progress
from parkour_lab.environments.randomization import TaskRandomization
from parkour_lab.runtime.accounting import InteractionCounter


def environment(seed=42):
    sampler = TaskRandomization(dict(num_envs=2, seed=seed, dynamics="randomized"))
    sampler.dynamics()
    command = NS(
        sampling_report=dict(
            version="flat_uniform_twist_v1", period_steps=200, samples=2, stops=1
        )
    )
    return NS(
        cfg=NS(
            benchmark_version="go2-blind-v1",
            benchmark_task_id="Go2-Flat-v1",
            benchmark_training=True,
            seed=seed,
        ),
        num_envs=2,
        scene=NS(terrain=NS()),
        parkour_randomization=sampler,
        interactions=InteractionCounter(2),
        command_manager=NS(get_term=lambda name: command),
    )


def checkpoint(env, name="full-learner-A", milestones=()):
    return capture_progress(
        env,
        run_id="method-seed42",
        checkpoint_id=name,
        configuration_id="fixed-physical-configuration",
        milestones=milestones,
    )


def restore(env, record, **kwargs):
    return restore_progress(
        env,
        record,
        checkpoint_id=record["checkpoint_id"],
        configuration_id="fixed-physical-configuration",
        **kwargs,
    )


class ProgressTests(unittest.TestCase):
    def test_json_recreation_continues_next_stream_draws_and_keeps_milestones(self):
        source = environment()
        source.interactions.sync(6, 24)
        source.parkour_randomization.sample_start(None)
        source.parkour_randomization.uniform([1], 30, "observation-noise")
        record = json.loads(json.dumps(checkpoint(source, milestones=[4, 8])))
        expected = source.parkour_randomization.sample_start(None)
        expected_commands = source.parkour_randomization.uniform(None, 4, "commands")
        recreated = environment()
        recreated.parkour_randomization.sample_start(None)
        recreated.parkour_randomization.uniform(None, 4, "commands")
        result = restore(recreated, record)
        self.assertEqual(result["run_id"], "method-seed42")
        self.assertEqual(result["milestones"], [4, 8])
        self.assertEqual(
            recreated.interactions.state_dict(),
            dict(session=0, cumulative=12, uncertain=0),
        )
        self.assertTrue(
            all(
                row["start"] is None
                for row in recreated.parkour_randomization.manifest()["attempts"]
            )
        )
        for name, value in recreated.parkour_randomization.sample_start(None).items():
            np.testing.assert_array_equal(value, expected[name])
        np.testing.assert_array_equal(
            recreated.parkour_randomization.uniform(None, 4, "commands"),
            expected_commands,
        )
        self.assertEqual(recreated.interactions.cumulative, 12)
        record["sampling_report"]["samples"] = 999
        self.assertEqual(
            recreated.command_manager.get_term("base_velocity").sampling_report[
                "samples"
            ],
            2,
        )

    def test_known_rollback_is_a_total_floor_and_never_added_twice(self):
        source = environment()
        source.interactions.sync(6, 24)
        record = checkpoint(source, milestones=[4, 8])
        target = environment()
        restored = restore(
            target,
            record,
            charged_interactions=14,
            reason="discarded collection after checkpoint",
        )
        self.assertEqual(target.interactions.cumulative, 14)
        self.assertEqual(restored["checkpoint_interactions"], 12)
        self.assertEqual(restored["restart"]["discarded_interactions"], 2)
        target.interactions.sync(1, 4)
        self.assertEqual(target.interactions.cumulative, 16)
        next_record = checkpoint(target, "full-learner-B", milestones=[16])
        self.assertEqual(next_record["checkpoint_interactions"], 16)
        self.assertEqual(next_record["milestones"], [4, 8, 16])
        later = environment()
        restore(later, next_record, charged_interactions=14)
        self.assertEqual(later.interactions.cumulative, 16)
        self.assertEqual(
            later._benchmark_progress["restart"]["discarded_interactions"], 2
        )

    def test_uncertain_collection_is_explicit_and_preserved(self):
        source = environment()
        source.interactions.sync(0, 1, failed=True)
        record = checkpoint(source)
        target = environment()
        result = restore(
            target,
            record,
            charged_interactions=5,
            uncertain_interactions=3,
            reason="bounded collection lost after process failure",
        )
        self.assertEqual(
            result["interactions"], dict(session=0, cumulative=8, uncertain=5)
        )
        self.assertEqual(result["restart"]["discarded_interactions"], 3)
        self.assertEqual(result["checkpoint_interactions"], 2)

    def test_rejections_are_atomic_for_identity_sampler_reports_and_accounting(self):
        source = environment()
        source.interactions.sync(6, 24)
        record = checkpoint(source)
        variants = []
        for name, value in (
            ("version", "wrong"),
            ("protocol", "wrong"),
            ("task_id", "Go2-Hills-v1"),
            ("seed", 43),
            ("num_envs", 3),
            ("checkpoint_id", "other-checkpoint"),
            ("configuration_id", "changed-reward"),
        ):
            bad = deepcopy(record)
            bad[name] = value
            variants.append(bad)
        bad = deepcopy(record)
        bad["sampler"]["generators"][-1]["commands"]["state"]["inc"] = 0
        variants.append(bad)
        bad = deepcopy(record)
        bad["sampling_report"]["samples"] = -1
        variants.append(bad)
        bad = deepcopy(record)
        bad["sampling_report"]["stops"] = bad["sampling_report"]["samples"] + 1
        variants.append(bad)
        bad = deepcopy(record)
        bad["sampling_report"]["period_steps"] = 100
        variants.append(bad)
        bad = deepcopy(record)
        bad["milestones"] = [13]
        variants.append(bad)
        target = environment()
        initial = target.parkour_randomization.state_dict()
        for bad in variants:
            with self.subTest(record=bad), self.assertRaises(ValueError):
                restore_progress(
                    target,
                    bad,
                    checkpoint_id=record["checkpoint_id"],
                    configuration_id="fixed-physical-configuration",
                )
            self.assertEqual(target.parkour_randomization.state_dict(), initial)
            self.assertEqual(target.interactions.cumulative, 0)
            self.assertFalse(hasattr(target, "_benchmark_progress"))
        for options in (
            dict(charged_interactions=14),
            dict(uncertain_interactions=2),
            dict(uncertain_interactions=-1),
            dict(charged_interactions=14, reason=""),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                restore(target, record, **options)
            self.assertEqual(target.parkour_randomization.state_dict(), initial)

    def test_warmup_cannot_be_erased_and_run_identity_cannot_be_reassigned(self):
        source = environment()
        source.interactions.sync(6, 24)
        record = checkpoint(source)
        target = environment()
        target.interactions.sync(1, 4)
        with self.assertRaisesRegex(ValueError, "before.*collects"):
            restore(target, record)
        self.assertEqual(target.interactions.cumulative, 2)
        with self.assertRaisesRegex(ValueError, "run identity"):
            capture_progress(
                source,
                run_id="other-seed",
                checkpoint_id="B",
                configuration_id="fixed-physical-configuration",
            )
        source.cfg.benchmark_training = False
        with self.assertRaisesRegex(ValueError, "training environment"):
            checkpoint(source)

    def test_flat_record_rejects_nonflat_curriculum_without_mutation(self):
        record = checkpoint(environment())
        record["curriculum"] = dict(levels=[2, 3], types=[0, 1])
        target = environment()
        initial = target.parkour_randomization.state_dict()
        with self.assertRaisesRegex(ValueError, "Curriculum continuation"):
            restore(target, record)
        self.assertEqual(target.interactions.cumulative, 0)
        self.assertEqual(target.parkour_randomization.state_dict(), initial)


if __name__ == "__main__":
    unittest.main()
