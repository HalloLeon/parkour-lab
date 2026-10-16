"""Pinned native origin updates, prescribed eligibility and atomic continuation."""

import json
import unittest
from copy import deepcopy

import numpy as np
import torch

from parkour_lab.benchmark.progress import capture_progress, restore_progress
from parkour_lab.environments.terrains.families.curriculum import (
    curriculum_state,
    initialize_levels,
    terrain_levels,
)
from tests.support.curriculum import curriculum_environment


def checkpoint(env):
    return capture_progress(
        env,
        run_id="method-seed42",
        checkpoint_id="learner-A",
        configuration_id="mixed-recipe",
        milestones=[16],
    )


def restore(env, record):
    return restore_progress(
        env, record, checkpoint_id="learner-A", configuration_id="mixed-recipe"
    )


class CurriculumTests(unittest.TestCase):
    def test_native_promote_demote_caps_and_nontranslation_holds_preserve_rng_types(
        self,
    ):
        env = curriculum_environment()
        term, terrain = env.command_manager.get_term("base_velocity"), env.scene.terrain
        env.scene["robot"].data.root_pos_w[:, 0] += torch.tensor(
            [100, 6.01, 0, 100, 100, 0, 100, 100]
        )
        term.curriculum_eligible[3:5] = False
        term.command[7, 2] = 0.3
        types, rng = terrain.terrain_types.clone(), torch.get_rng_state().clone()
        terrain_levels(env, torch.arange(8))
        self.assertEqual(terrain.terrain_levels.tolist(), [0, 3, 2, 4, 5, 0, 7, 4])
        torch.testing.assert_close(terrain.terrain_types, types)
        torch.testing.assert_close(
            terrain.env_origins, terrain.terrain_origins[terrain.terrain_levels, types]
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_strict_distance_thresholds_and_initial_reset_hold(self):
        env = curriculum_environment()
        terrain = env.scene.terrain
        env.scene["robot"].data.root_pos_w[1:3, 0] += torch.tensor([6.0, 4.0])
        terrain_levels(env, torch.tensor([1, 2]))
        self.assertEqual(terrain.terrain_levels[1:3].tolist(), [2, 3])
        env.episode_length_buf.zero_()
        env.scene["robot"].data.root_pos_w[:, 0] += 100
        before = terrain.terrain_levels.clone()
        terrain_levels(env, torch.arange(8))
        torch.testing.assert_close(terrain.terrain_levels, before)

    def test_private_initial_range_is_reproducible_and_flat_stays_zero(self):
        first, second = curriculum_environment(), curriculum_environment()
        expected = deepcopy(first.parkour_randomization).uniform(None, 1, "geometry")[
            :, 0
        ]
        rng = torch.get_rng_state().clone()
        initialize_levels(first)
        initialize_levels(second)
        levels = first.scene.terrain.terrain_levels
        np.testing.assert_array_equal(
            levels[1:].numpy(), (expected[1:] * 2).astype(int)
        )
        self.assertEqual(levels[0], 0)
        torch.testing.assert_close(levels, second.scene.terrain.terrain_levels)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_fixed_evaluation_row_zero_represents_its_declared_physical_level(self):
        env = curriculum_environment()
        env.cfg.benchmark_training = False
        env.cfg.benchmark_terrain.update(
            num_rows=1, fixed_level=3, physical_levels_by_row=[3]
        )
        terrain = env.scene.terrain
        terrain.terrain_origins = terrain.terrain_origins[:1]
        terrain.terrain_levels.zero_()
        before = env.parkour_randomization.state_dict()
        initialize_levels(env)
        self.assertEqual(env.parkour_randomization.state_dict(), before)
        self.assertEqual(curriculum_state(env)["levels"], [0] * 8)

    def test_json_restore_keeps_physical_assignments_then_starts_fresh_episodes(self):
        source = curriculum_environment()
        source.interactions.sync(6, 24)
        report = source.command_manager.get_term("base_velocity").sampling_report
        report.update(samples=8, translations=4, stops=1, pivots=2, arcs=1)
        record = json.loads(json.dumps(checkpoint(source)))
        target = curriculum_environment()
        initialize_levels(target)
        restore(target, record)
        self.assertEqual(curriculum_state(target), record["curriculum"])
        torch.testing.assert_close(
            target.scene.terrain.env_origins, source.scene.terrain.env_origins
        )
        self.assertEqual(
            target.interactions.state_dict(),
            dict(session=0, cumulative=48, uncertain=0),
        )
        self.assertEqual(
            target.command_manager.get_term("base_velocity").sampling_report, report
        )
        np.testing.assert_array_equal(
            target.parkour_randomization.uniform(None, 4, "commands"),
            source.parkour_randomization.uniform(None, 4, "commands"),
        )
        target.episode_length_buf.zero_()
        target.scene["robot"].data.root_pos_w[:, 0] += 1000
        terrain_levels(target, torch.arange(8))
        self.assertEqual(curriculum_state(target), record["curriculum"])

    def test_all_identity_levels_and_late_report_rejections_are_atomic(self):
        source = curriculum_environment()
        source.interactions.sync(6, 24)
        original = checkpoint(source)
        original["sampling_report"].update(samples=2, translations=2)
        variants = []
        for mutate in (
            lambda r: r["curriculum"].update(version="other"),
            lambda r: r["curriculum"]["definition"]["columns"][1].update(family="Flat"),
            lambda r: r["curriculum"]["types"].__setitem__(1, 2),
            lambda r: r["curriculum"]["types"].__setitem__(1, True),
            lambda r: r["curriculum"]["levels"].__setitem__(1, -1),
            lambda r: r["curriculum"]["levels"].__setitem__(1, 8),
            lambda r: r["curriculum"]["levels"].__setitem__(0, 1),
            lambda r: r["sampling_report"].update(period_steps=4),
            lambda r: r["sampling_report"].update(stops=1),
        ):
            bad = deepcopy(original)
            mutate(bad)
            variants.append(bad)
        for bad in variants:
            target = curriculum_environment()
            initialize_levels(target)
            state, sampler = (
                curriculum_state(target),
                target.parkour_randomization.state_dict(),
            )
            origins = target.scene.terrain.env_origins.clone()
            with self.subTest(record=bad), self.assertRaises(ValueError):
                restore(target, bad)
            self.assertEqual(curriculum_state(target), state)
            self.assertEqual(target.parkour_randomization.state_dict(), sampler)
            torch.testing.assert_close(target.scene.terrain.env_origins, origins)
            self.assertEqual(target.interactions.cumulative, 0)
