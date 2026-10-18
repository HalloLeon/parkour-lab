"""Full optimizer barriers, live save neutrality and intermediate continuation."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from examples.sb3 import snapshots, training
from examples.sb3.checkpoint import load_model
from examples.sb3.policy import SB3FramePolicy
from examples.sb3.state import (
    capture_rng,
    model_state_sha256,
    replay_state_sha256,
    state_sha256,
)
from examples.sb3.tests.fixtures import FakeTask
from examples.sb3.training import run_training


class SnapshotScheduleTests(unittest.TestCase):
    def test_default_update_arithmetic_includes_sac_warmup_and_collision(self):
        ppo = NS(_n_updates=0, num_timesteps=0, n_epochs=2, n_envs=20, n_steps=32)
        self.assertEqual(
            snapshots.next_snapshot(
                ppo, "ppo", interval=5000, milestones=(), target=3_200_000
            ),
            (1_600_000, ["periodic"]),
        )
        ppo._n_updates, ppo.num_timesteps = 5000, 1_600_000
        self.assertEqual(
            snapshots.next_snapshot(
                ppo, "ppo", interval=5000, milestones=(3_200_000,), target=3_200_000
            ),
            (3_200_000, ["periodic", "milestone", "terminal"]),
        )
        for timesteps, updates in ((0, 0), (60, 0), (80, 1)):
            sac = NS(
                _n_updates=updates,
                num_timesteps=timesteps,
                n_envs=20,
                learning_starts=64,
            )
            self.assertEqual(
                snapshots.next_snapshot(
                    sac, "sac", interval=5000, milestones=(), target=200_060
                ),
                (100_060, ["periodic"]),
            )


class RealSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_periodic_milestone_saves_preserve_live_training_and_rng(self):
        for algorithm, interval in (("ppo", 4), ("sac", 40)):
            with (
                self.subTest(algorithm=algorithm),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                baseline_task = FakeTask(horizon=10_000)
                baseline = run_training(
                    baseline_task,
                    algorithm=algorithm,
                    total_interactions=512,
                    output=root / "baseline",
                )
                task, captures = FakeTask(horizon=10_000), []
                original = snapshots.save_checkpoint

                def save(model, live_task, *args, **kwargs):
                    before = dict(
                        physical=live_task.state.clone(),
                        age=live_task.age.clone(),
                        reset_calls=live_task.reset_calls,
                        counts=dict(live_task.interactions),
                        observation=model._last_obs.copy(),
                        episode_starts=model._last_episode_starts.copy(),
                        model=model_state_sha256(model),
                        rng=state_sha256(capture_rng(model.action_space)),
                        replay=(
                            None if algorithm == "ppo" else replay_state_sha256(model)
                        ),
                    )
                    receipt = original(model, live_task, *args, **kwargs)
                    torch.testing.assert_close(
                        before["physical"], live_task.state, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        before["age"], live_task.age, rtol=0, atol=0
                    )
                    self.assertEqual(before["reset_calls"], live_task.reset_calls)
                    self.assertEqual(before["counts"], live_task.interactions)
                    np.testing.assert_array_equal(
                        before["observation"], model._last_obs
                    )
                    np.testing.assert_array_equal(
                        before["episode_starts"], model._last_episode_starts
                    )
                    self.assertEqual(before["model"], model_state_sha256(model))
                    self.assertEqual(
                        before["rng"], state_sha256(capture_rng(model.action_space))
                    )
                    if algorithm == "sac":
                        self.assertEqual(before["replay"], replay_state_sha256(model))
                    captures.append(receipt)
                    return receipt

                with patch.object(snapshots, "save_checkpoint", side_effect=save):
                    result = run_training(
                        task,
                        algorithm=algorithm,
                        total_interactions=512,
                        output=root / "segmented",
                        save_every_updates=interval,
                        milestones=(128, 256),
                    )
                self.assertEqual(task.reset_calls, 1)
                torch.testing.assert_close(
                    task.state, baseline_task.state, rtol=0, atol=0
                )
                self.assertEqual(
                    result["checkpoint"]["model_state_sha256"],
                    baseline["checkpoint"]["model_state_sha256"],
                )
                self.assertEqual(
                    result["checkpoint"]["rng_state_sha256"],
                    baseline["checkpoint"]["rng_state_sha256"],
                )
                self.assertEqual(
                    result["checkpoint"]["replay_state_sha256"],
                    baseline["checkpoint"]["replay_state_sha256"],
                )
                self.assertEqual(result["learner_recipe"], baseline["learner_recipe"])
                self.assertNotIn("seed", result["learner_recipe"])
                index = json.loads(
                    (root / "segmented/checkpoint_index.json").read_text()
                )
                entries = index["checkpoints"]
                self.assertTrue(index["complete"])
                expected_points = (
                    [128, 256, 512] if algorithm == "ppo" else [128, 224, 256, 384, 512]
                )
                self.assertEqual(
                    [entry["training_interactions"] for entry in entries],
                    expected_points,
                )
                self.assertEqual(len(captures), len(expected_points))
                if algorithm == "ppo":
                    self.assertEqual(entries[1]["reasons"], ["periodic", "milestone"])
                    self.assertEqual(entries[-1]["reasons"], ["periodic", "terminal"])
                self.assertEqual(entries[-1]["path"], "checkpoint")
                self.assertEqual(entries[-1]["reached_milestones"], [128, 256, 512])
                times = [entry["training_loop_wall_time_s"] for entry in entries]
                self.assertEqual(times, sorted(times))
                self.assertGreater(times[0], 0)
                for entry in entries:
                    folder = root / "segmented" / entry["path"]
                    receipt = json.loads((folder / "checkpoint.json").read_text())
                    self.assertEqual(receipt["checkpoint_id"], entry["checkpoint_id"])
                    self.assertEqual(
                        receipt["training_interactions"], entry["learner_timesteps"]
                    )
                    restored, _ = load_model(folder)
                    self.assertEqual(
                        restored._n_updates, entry["external_update_counter"]
                    )

    def test_intermediate_full_bundle_resumes_in_a_fresh_process(self):
        program = """
import json, sys, torch
from pathlib import Path
from examples.sb3.tests.fixtures import FakeTask
from examples.sb3.training import run_training
torch.set_num_threads(1)
task = FakeTask()
report = run_training(task, algorithm=sys.argv[1], total_interactions=512,
                      output=Path(sys.argv[3]), resume=Path(sys.argv[2]),
                      save_every_updates=int(sys.argv[4]), milestones=(384,))
assert task.restore_calls == task.reset_calls == 1
print(json.dumps(report))
"""
        for algorithm, interval in (("ppo", 4), ("sac", 40)):
            with (
                self.subTest(algorithm=algorithm),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                run_training(
                    FakeTask(),
                    algorithm=algorithm,
                    total_interactions=512,
                    output=root / "first",
                    save_every_updates=interval,
                    milestones=(256,),
                )
                intermediate = root / "first/checkpoints/step_256"
                model, receipt = load_model(intermediate)
                fingerprint = SB3FramePolicy(model).state_sha256()
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        program,
                        algorithm,
                        str(intermediate),
                        str(root / "resumed"),
                        str(interval),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(report["initial_interactions"]["cumulative"], 256)
                self.assertEqual(
                    report["training_interactions"],
                    dict(session=256, cumulative=512, uncertain=0),
                )
                self.assertEqual(
                    report["external_update_counter_before"], model._n_updates
                )
                self.assertEqual(report["policy_state_before"], fingerprint)
                self.assertEqual(report["checkpoint"]["run_id"], receipt["run_id"])
                progress = json.loads(
                    (root / "resumed/checkpoint/progress.json").read_text()
                )
                self.assertEqual(progress["milestones"], [256, 384, 512])

    def test_invalid_schedules_fail_before_collection(self):
        for algorithm, kwargs in (
            ("ppo", {"save_every_updates": 3}),
            ("sac", {"save_every_updates": True}),
            ("ppo", {"milestones": (129,)}),
            ("ppo", {"milestones": (256, 128)}),
            ("sac", {"milestones": (64,)}),
        ):
            with (
                self.subTest(algorithm=algorithm, kwargs=kwargs),
                tempfile.TemporaryDirectory() as temporary,
            ):
                task = FakeTask()
                with self.assertRaises(ValueError):
                    run_training(
                        task,
                        algorithm=algorithm,
                        total_interactions=512,
                        output=Path(temporary),
                        **kwargs,
                    )
                self.assertEqual(task.reset_calls, 0)
                self.assertEqual(task.interactions["cumulative"], 0)
                self.assertFalse((Path(temporary) / "checkpoint_index.json").exists())

    def test_unsupported_optimizer_schedules_fail_before_collection(self):
        original = training._model
        for algorithm, changed in (
            ("ppo", {"target_kl": 0.02}),
            ("ppo", {"clip_range": lambda fraction: 0.1 + 0.1 * fraction}),
            ("sac", {"gradient_steps": 2}),
        ):
            with (
                self.subTest(algorithm=algorithm, changed=changed),
                tempfile.TemporaryDirectory() as temporary,
            ):

                def build(*args, **kwargs):
                    model = original(*args, **kwargs)
                    for name, value in changed.items():
                        setattr(model, name, value)
                    return model

                task = FakeTask()
                with patch.object(training, "_model", side_effect=build):
                    with self.assertRaises(ValueError):
                        run_training(
                            task,
                            algorithm=algorithm,
                            total_interactions=512,
                            output=Path(temporary),
                        )
                self.assertEqual(task.reset_calls, 0)
                self.assertEqual(task.interactions["cumulative"], 0)

    def test_charged_or_uncertain_accounting_remains_explicitly_unsupported(self):
        for uncertain in (0, 4):
            with (
                self.subTest(uncertain=uncertain),
                tempfile.TemporaryDirectory() as temporary,
            ):
                task = FakeTask()
                task.interactions.update(cumulative=128, uncertain=uncertain)
                with self.assertRaisesRegex(ValueError, "exact checkpoint accounting"):
                    run_training(
                        task,
                        algorithm="ppo",
                        total_interactions=512,
                        output=Path(temporary),
                    )
                self.assertEqual(
                    task.interactions,
                    dict(session=0, cumulative=128, uncertain=uncertain),
                )
                self.assertEqual(task.reset_calls, 0)

    def test_failed_later_save_leaves_previous_full_snapshot_and_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, calls = snapshots.save_checkpoint, []

            def save(model, task, folder, **kwargs):
                calls.append(folder)
                if len(calls) == 2:
                    folder.mkdir(parents=True)
                    (folder / "model.zip").write_bytes(b"incomplete")
                    raise OSError("interrupted snapshot")
                return original(model, task, folder, **kwargs)

            task = FakeTask()
            with patch.object(snapshots, "save_checkpoint", side_effect=save):
                with self.assertRaisesRegex(OSError, "interrupted snapshot"):
                    run_training(
                        task,
                        algorithm="ppo",
                        total_interactions=512,
                        output=root,
                        save_every_updates=2,
                    )
            index = json.loads((root / "checkpoint_index.json").read_text())
            self.assertFalse(index["complete"])
            self.assertEqual(
                [entry["training_interactions"] for entry in index["checkpoints"]],
                [128],
            )
            load_model(root / index["checkpoints"][0]["path"])
            self.assertFalse((root / "checkpoints/step_256/checkpoint.json").exists())
            self.assertFalse((root / "checkpoint").exists())
            self.assertEqual(task.interactions["cumulative"], 256)


if __name__ == "__main__":
    unittest.main()
