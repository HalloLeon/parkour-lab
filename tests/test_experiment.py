"""Current train/continue/export/evaluate integration on native API doubles."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from parkour_lab import artifacts, experiment
from parkour_lab.config import ExperimentConfig, MethodConfig, TaskConfig
from parkour_lab.evaluation.flat import profile_commands, score_flat_profile
from parkour_lab.evaluation.runner import command_sequence, evaluate
from parkour_lab.methods import get_backend
from tests.support.native import TrainingEnvironmentFixture


class ExperimentIntegrationTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.config = ExperimentConfig(
            task=TaskConfig(num_envs=8, device="cpu", seed=97),
            method=MethodConfig(
                options=dict(
                    rollout_steps=2,
                    history_steps=2,
                    history_interval=2,
                    adaptation_epochs=1,
                    adaptation_batches=2,
                    num_learning_epochs=1,
                    num_mini_batches=2,
                    contact_conditioned=False,
                )
            ),
            updates=2,
            save_interval=1,
        )

    def native(self):
        fixture = TrainingEnvironmentFixture()
        fixture.env.cfg.seed = self.config.task.seed
        return fixture

    def initial_state(self, output, report):
        receipt = report["initial_learning_state"]
        self.assertEqual(receipt["path"], "initial_learning_state.payload")
        self.assertEqual(receipt["updates"], report["initial_updates"])
        path = output / receipt["path"]
        self.assertEqual(receipt["sha256"], artifacts.file_sha256(path))
        self.assertEqual(report["evidence_sha256"][path.name], receipt["sha256"])
        return get_backend(self.config.method.name).load(
            path.read_bytes(),
            "training",
            self.config.method.options,
            receipt["updates"],
        )

    def test_fresh_train_continue_export_and_frozen_evaluation_round_trip(
        self,
    ):
        native = self.native()
        output = self.root / "train"
        output.mkdir()
        report = {}
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.train(native.env, native.app, self.config, output, report)
        self.assertEqual(report["status"], "TRAINING_COMPLETE_NOT_QUALIFIED")
        self.assertEqual(report["initial_updates"], 0)
        self.assertEqual(report["updates"], 2)
        self.assertEqual(report["environment_transitions"], (2 * 2 + 2) * 8)
        self.assertEqual(report["method_metrics"]["adaptation_optimizer_steps"], 2)
        initial = self.initial_state(output, report)
        self.assertEqual(initial["updates"], 0)
        self.assertEqual(initial["adaptation_optimizer_steps"], 0)
        self.assertTrue(
            all(
                torch.isfinite(value).all()
                for value in initial["policy_state"].values()
            )
        )
        for name in ("ppo_optimizer", "adaptation_optimizer"):
            self.assertEqual(initial[name]["state"], {})
            self.assertTrue(initial[name]["param_groups"])
        self.assertEqual(native.reset_seeds, [97])
        records = [
            json.loads(line)
            for line in (output / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertEqual([record["update"] for record in records], [1, 2])
        self.assertIsNone(records[0]["adaptation"])
        self.assertIsNotNone(records[1]["adaptation"])
        for row in records:
            task = row["task_metrics"]
            self.assertIn("privileged_ppo", task)
            for phase in task.values():
                for regime in phase["regimes"].values():
                    self.assertAlmostEqual(
                        sum(
                            term["contribution_sum"]
                            for term in regime["reward_terms"].values()
                        ),
                        regime["reward_contribution_sum"],
                        places=5,
                    )
        checkpoint = Path(report["checkpoint"])
        source_bytes = checkpoint.read_bytes()
        state = artifacts.load_artifact(checkpoint, kind="training")
        self.assertEqual(state["state"]["updates"], 2)
        self.assertTrue(
            any(
                not torch.equal(value, state["state"]["policy_state"][name])
                for name, value in initial["policy_state"].items()
            )
        )

        continued_native = self.native()
        continued_output = self.root / "continued"
        continued_output.mkdir()
        continued_report = {}
        continued_config = replace(
            self.config,
            task=replace(
                self.config.task, linear_tracking_std=0.25, angular_tracking_std=0.3
            ),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.train(
                continued_native.env,
                continued_native.app,
                continued_config,
                continued_output,
                continued_report,
                checkpoint=checkpoint,
            )
        self.assertEqual(continued_report["initial_updates"], 2)
        torch.testing.assert_close(
            self.initial_state(continued_output, continued_report),
            state["state"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(continued_report["updates"], 4)
        self.assertEqual(
            continued_report["method_metrics"]["adaptation_optimizer_steps"], 4
        )
        self.assertIn("not exact resume", continued_report["continuation"])
        self.assertEqual(
            continued_report["source_checkpoint"],
            {
                "path": str(checkpoint.resolve()),
                "sha256": artifacts.file_sha256(checkpoint),
            },
        )
        self.assertEqual(checkpoint.read_bytes(), source_bytes)
        current = Path(continued_report["checkpoint"])
        actor = self.root / "actor.pt"
        receipt = artifacts.export_actor(current, actor)
        self.assertFalse(receipt["qualified"])
        loaded = artifacts.load_actor(actor)
        self.assertEqual(loaded.updates, 4)
        self.assertEqual(
            ExperimentConfig.from_dict(
                artifacts.load_artifact(actor, kind="actor")["config"]
            ),
            continued_config,
        )

        evaluation_native = self.native()
        evaluation_native.env.scene["contact_forces"].data.net_forces_w.fill_(2)
        evaluation_output = self.root / "evaluation"
        evaluation_output.mkdir()
        evaluation_report = {}
        commands = command_sequence(steps=55)
        evaluate(
            evaluation_native.env,
            evaluation_native.app,
            loaded,
            commands,
            evaluation_output,
            evaluation_report,
        )
        self.assertEqual(
            evaluation_report["status"], "EVALUATION_COMPLETE_NOT_QUALIFIED"
        )
        self.assertTrue(evaluation_report["policy_unchanged"])
        self.assertEqual(evaluation_report["control_steps"], 55)
        self.assertEqual(evaluation_report["environment_transitions"], 55 * 8)
        self.assertEqual(len(evaluation_report["tracking_rmse"]), 3)
        self.assertEqual(
            sum(
                row["samples"]
                for row in evaluation_report["task_metrics"]["evaluation"][
                    "regimes"
                ].values()
            ),
            55 * 8,
        )
        self.assertFalse(evaluation_native.env.capture_diagnostics)
        self.assertTrue((evaluation_output / "commands.json").is_file())
        self.assertTrue((evaluation_output / "tracking.npz").is_file())
        with np.load(evaluation_output / "motion.npz", allow_pickle=False) as motion:
            self.assertEqual(motion["position_w"].shape, (55, 8, 3))
            self.assertEqual(motion["initial_position_w"].shape, (8, 3))
            self.assertEqual(motion["terminated"].dtype, np.bool_)
            self.assertTrue(motion["terminated"].any())
            self.assertFalse(evaluation_native.env.capture_motion)
        with np.load(evaluation_output / "tracking.npz", allow_pickle=False) as trace:
            self.assertEqual(trace["foot_contacts"].shape, (55, 8, 4))
            expected = (
                np.linalg.norm(trace["foot_net_forces_w"], axis=-1) > 1.5
            ).astype(np.float32)
            expected[trace["reset_mask"]] = 0
            np.testing.assert_array_equal(trace["foot_contacts"], expected)
            self.assertTrue(trace["reset_mask"][0].all())
            self.assertTrue(trace["reset_mask"][1:].any())

        profile_output = self.root / "profile"
        profile_output.mkdir()
        profile_report = {}
        fixture = self.native()
        evaluate(
            fixture.env,
            fixture.app,
            loaded,
            profile_commands("stand"),
            profile_output,
            profile_report,
            profile="stand",
        )
        scored = profile_report["flat_diagnostic"]
        self.assertFalse(scored["qualification_eligible"])
        self.assertFalse(scored["trials"][0]["diagnostic_passed"])
        self.assertEqual(scored["trials"][0]["first_attempt_steps"], 4)
        with np.load(profile_output / "motion.npz", allow_pickle=False) as motion:
            self.assertEqual(score_flat_profile(motion, "stand"), scored)
        self.assertFalse(fixture.env.capture_motion)

        interrupted = self.root / "interrupted"
        interrupted.mkdir()
        fixture = self.native()
        fixture.running = False
        with self.assertRaisesRegex(RuntimeError, "stopped before the tape"):
            evaluate(fixture.env, fixture.app, loaded, commands, interrupted, {})
        self.assertFalse(fixture.env.capture_motion)
        self.assertFalse(
            json.loads((interrupted / "commands.json").read_text())["complete"]
        )
        self.assertFalse((interrupted / "motion.npz").exists())

    def test_same_seed_reward_variants_preserve_equal_initial_learning_payloads(self):
        initial_states, hashes = [], []
        for index, rewards in enumerate(({}, {"action_rate_l2": {"weight": -0.02}})):
            config = replace(
                self.config, task=replace(self.config.task, rewards=rewards), updates=1
            )
            native, report = self.native(), {}
            output = self.root / f"recipe_{index}"
            output.mkdir()
            with contextlib.redirect_stdout(io.StringIO()):
                experiment.train(native.env, native.app, config, output, report)
            initial_states.append(self.initial_state(output, report))
            hashes.append(report["initial_learning_state"]["sha256"])
        torch.testing.assert_close(initial_states[0], initial_states[1], rtol=0, atol=0)
        self.assertEqual(hashes[0], hashes[1])

    def test_existing_initial_learning_payload_is_not_overwritten_or_trained_past(self):
        output = self.root / "occupied"
        output.mkdir()
        path = output / "initial_learning_state.payload"
        path.write_bytes(b"existing evidence")
        native = self.native()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(FileExistsError),
        ):
            experiment.train(native.env, native.app, self.config, output, {})
        self.assertEqual(path.read_bytes(), b"existing evidence")
        self.assertEqual(native.deliveries, [])
        self.assertFalse((output / "metrics.jsonl").exists())
        self.assertEqual(list(output.iterdir()), [path])

    def test_continuation_rejects_changed_method_configuration(self):
        native = self.native()
        output = self.root / "source"
        output.mkdir()
        report = {}
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.train(native.env, native.app, self.config, output, report)
        changed = replace(
            self.config,
            method=MethodConfig(
                options={**self.config.method.options, "learning_rate": 0.002}
            ),
        )
        destination = self.root / "changed"
        destination.mkdir()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "unchanged method"),
        ):
            fixture = self.native()
            experiment.train(
                fixture.env,
                fixture.app,
                changed,
                destination,
                {},
                checkpoint=report["checkpoint"],
            )
        self.assertFalse((destination / "metrics.jsonl").exists())

    def test_cross_rate_continuation_rejects_before_loading_or_advancing(self):
        source = self.root / "source"
        source.mkdir()
        fixture, report = self.native(), {}
        with contextlib.redirect_stdout(io.StringIO()):
            experiment.train(fixture.env, fixture.app, self.config, source, report)
        checkpoint = Path(report["checkpoint"])
        source_bytes = checkpoint.read_bytes()
        candidate = replace(self.config, task=replace(self.config.task, physics_hz=400))
        native = self.native()
        native.env.physics_dt = 0.0025
        native.env.cfg.decimation = 8
        output = self.root / "changed_rate"
        output.mkdir()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "motor"),
        ):
            experiment.train(
                native.env, native.app, candidate, output, {}, checkpoint=checkpoint
            )
        self.assertFalse((output / "initial_learning_state.payload").exists())
        self.assertFalse((output / "metrics.jsonl").exists())
        self.assertEqual(native.deliveries, [])
        self.assertEqual(checkpoint.read_bytes(), source_bytes)


if __name__ == "__main__":
    unittest.main()
