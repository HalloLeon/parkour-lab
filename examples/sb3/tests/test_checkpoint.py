"""Exercise genuine PPO/SAC archives, replay, RNG and fresh-process continuation."""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from examples.sb3.checkpoint import (
    capture_model_state,
    load_checkpoint,
    load_model,
    model_state_sha256,
    save_checkpoint,
)
from examples.sb3.environment import Go2VecEnv
from examples.sb3.state import capture_rng, replay_state_sha256, state_sha256
from examples.sb3.tests.fixtures import FakeTask


def trained(algorithm):
    task = FakeTask()
    env = Go2VecEnv(task)
    common = dict(
        seed=42, learning_rate=3e-4, device="cpu", policy_kwargs=dict(net_arch=[8])
    )
    if algorithm == "PPO":
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, **common)
    else:
        model = SAC(
            "MlpPolicy",
            env,
            buffer_size=32,
            batch_size=4,
            learning_starts=0,
            train_freq=1,
            gradient_steps=1,
            **common,
        )
    model.learn(64)
    return task, model


def next_random(model):
    return (
        random.random(),
        np.random.random(),
        torch.rand(4).tolist(),
        model.action_space.sample().tolist(),
    )


class CheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_optimizer_entropy_replay_rng_and_fresh_episode_resume(self):
        for algorithm in ("PPO", "SAC"):
            with (
                self.subTest(algorithm=algorithm),
                tempfile.TemporaryDirectory() as temporary,
            ):
                task, model = trained(algorithm)
                before = capture_model_state(model)
                self.assertGreater(model._n_updates, 0)
                self.assertTrue(any("optimizer" in key for key in before["parameters"]))
                folder = Path(temporary) / "checkpoint"
                receipt = save_checkpoint(
                    model, task, folder, run_id="smoke", milestones=[64]
                )
                self.assertEqual(
                    receipt["checkpoint_id"], "sb3:" + receipt["checkpoint_sha256"]
                )
                expected = next_random(model)
                restored_task = FakeTask()
                random.seed(500)
                np.random.seed(500)
                torch.manual_seed(500)
                restored, loaded = load_checkpoint(folder, restored_task)
                self.assertEqual(loaded, receipt)
                self.assertEqual(state_sha256(before), model_state_sha256(restored))
                self.assertEqual(
                    state_sha256(capture_rng(restored.action_space)),
                    receipt["rng_state_sha256"],
                )
                self.assertEqual(next_random(restored), expected)
                self.assertIsNone(restored._last_obs)
                self.assertIsNone(restored.get_env())
                self.assertEqual(restored_task.reset_calls, 0)
                self.assertEqual(restored_task.restore_calls, 1)
                self.assertEqual(
                    restored_task.interactions,
                    dict(session=0, cumulative=64, uncertain=0),
                )
                if algorithm == "SAC":
                    self.assertTrue(model.replay_buffer.full)
                    self.assertEqual(
                        replay_state_sha256(restored), replay_state_sha256(model)
                    )
                    self.assertIn("log_ent_coef", before["variables"])
                    self.assertIn("ent_coef_optimizer", before["parameters"])
                fingerprint = model_state_sha256(restored)
                rng = state_sha256(capture_rng(restored.action_space))
                restored.set_env(Go2VecEnv(restored_task), force_reset=True)
                self.assertEqual(state_sha256(capture_rng(restored.action_space)), rng)
                restored.learn(32, reset_num_timesteps=False)
                self.assertEqual(restored.num_timesteps, 96)
                self.assertEqual(restored_task.interactions["cumulative"], 96)
                self.assertGreater(restored._n_updates, model._n_updates)
                self.assertNotEqual(model_state_sha256(restored), fingerprint)

    def test_inference_preserves_caller_rng_and_does_not_unpickle_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            task, model = trained("SAC")
            folder = Path(temporary) / "checkpoint"
            receipt = save_checkpoint(model, task, folder, run_id="smoke")
            before = state_sha256(capture_rng())
            with patch.object(
                SAC,
                "load_replay_buffer",
                side_effect=AssertionError("inference replay"),
            ):
                loaded, actual = load_model(folder)
            self.assertEqual(state_sha256(capture_rng()), before)
            self.assertEqual(actual, receipt)
            self.assertEqual(model_state_sha256(loaded), receipt["model_state_sha256"])
            observation = np.ones((2, 49), dtype=np.float32)
            np.testing.assert_array_equal(
                model.predict(observation, deterministic=True)[0],
                loaded.predict(observation, deterministic=True)[0],
            )

    def test_expanded_evaluator_loads_original_policy_without_relaxing_resume(self):
        from examples.sb3 import checkpoint

        with tempfile.TemporaryDirectory() as temporary:
            task, model = trained("PPO")
            folder = Path(temporary) / "checkpoint"
            receipt = save_checkpoint(model, task, folder, run_id="smoke")
            changed = dict(receipt["source_sha256"], parkour_lab="new-evaluator-source")
            with patch.object(checkpoint, "_source_identity", return_value=changed):
                loaded, actual = load_model(folder)
                self.assertEqual(actual, receipt)
                self.assertEqual(
                    model_state_sha256(loaded), receipt["model_state_sha256"]
                )
                with self.assertRaisesRegex(ValueError, "version or source"):
                    load_checkpoint(folder, FakeTask())
            changed["stable_baselines3"] = "different-learner-source"
            with patch.object(checkpoint, "_source_identity", return_value=changed):
                with self.assertRaisesRegex(ValueError, "version or source"):
                    load_model(folder)

    def test_bundle_corruption_identity_and_overwrite_rejected_before_deserialization(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            task, model = trained("SAC")
            folder = Path(temporary) / "checkpoint"
            receipt = save_checkpoint(model, task, folder, run_id="smoke")
            with self.assertRaises(FileExistsError):
                save_checkpoint(model, task, folder, run_id="smoke")
            wrong_task = FakeTask(seed=43)
            with patch.object(SAC, "load", side_effect=AssertionError("must not load")):
                with self.assertRaisesRegex(ValueError, "task protocol"):
                    load_checkpoint(folder, wrong_task)
                replay = folder / "replay.pkl"
                original = replay.read_bytes()
                replay.write_bytes(original + b"corrupted")
                with self.assertRaisesRegex(ValueError, "hash differs"):
                    load_model(folder)
                replay.write_bytes(original)
                receipt["library_version"] = "unsupported"
                (folder / "checkpoint.json").write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "version or source"):
                    load_model(folder)
            self.assertEqual(wrong_task.restore_calls, 0)

    def test_fresh_process_restores_before_any_update_and_collects_more(self):
        code = """
import json, sys, torch
from examples.sb3.checkpoint import load_checkpoint, model_state_sha256
from examples.sb3.environment import Go2VecEnv
from examples.sb3.state import capture_rng, state_sha256
from examples.sb3.tests.fixtures import FakeTask
torch.set_num_threads(1)
task = FakeTask()
model, receipt = load_checkpoint(sys.argv[1], task)
assert model_state_sha256(model) == receipt['model_state_sha256']
assert state_sha256(capture_rng(model.action_space)) == receipt['rng_state_sha256']
assert model.num_timesteps == 64 and model._last_obs is None
assert task.reset_calls == 0 and task.restore_calls == 1
updates = model._n_updates
model.set_env(Go2VecEnv(task), force_reset=True)
model.learn(32, reset_num_timesteps=False)
assert model.num_timesteps == task.interactions['cumulative'] == 96
assert model._n_updates > updates
print(json.dumps({'algorithm': receipt['algorithm'], 'cumulative': 96}))
"""
        for algorithm in ("PPO", "SAC"):
            with (
                self.subTest(algorithm=algorithm),
                tempfile.TemporaryDirectory() as temporary,
            ):
                task, model = trained(algorithm)
                folder = Path(temporary) / "checkpoint"
                save_checkpoint(model, task, folder, run_id="smoke")
                result = subprocess.run(
                    [sys.executable, "-c", code, str(folder)],
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('"cumulative": 96', result.stdout)


if __name__ == "__main__":
    unittest.main()
