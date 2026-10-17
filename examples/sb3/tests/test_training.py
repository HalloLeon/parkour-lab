"""Exercise real external optimizers, fresh-process resume and frozen inference."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from examples.sb3.checkpoint import load_model
from examples.sb3.policy import SB3FramePolicy, load_policy
from examples.sb3.state import capture_rng, state_sha256
from examples.sb3.tests.fixtures import FakeTask
from examples.sb3.training import run_training


class RealLearnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_genuine_updates_fresh_process_resume_and_loaded_inference(self):
        for algorithm in ("ppo", "sac"):
            with (
                self.subTest(algorithm=algorithm),
                tempfile.TemporaryDirectory() as tmp,
            ):
                folder = Path(tmp)
                task = FakeTask()
                first = run_training(
                    task,
                    algorithm=algorithm,
                    total_interactions=256,
                    output=folder / "first",
                )
                task.close()
                self.assertTrue(first["parameters_changed"])
                self.assertGreater(first["external_update_counter_after"], 0)
                self.assertEqual(first["training_information_used"], [])
                self.assertEqual(first["learner_timesteps"], 256)
                checkpoint = folder / "first/checkpoint"
                program = """
import json, sys, torch
from pathlib import Path
from examples.sb3.tests.fixtures import FakeTask
from examples.sb3.training import run_training
torch.set_num_threads(1)
task = FakeTask()
report = run_training(task, algorithm=sys.argv[1], total_interactions=512,
                      output=Path(sys.argv[3]), resume=Path(sys.argv[2]))
assert task.restore_calls == 1 and task.reset_calls == 1
task.close()
print(json.dumps(report))
"""
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        program,
                        algorithm,
                        str(checkpoint),
                        str(folder / "resumed"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=False,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                )
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                resumed = json.loads(result.stdout)
                self.assertTrue(resumed["learner_checkpoint_restored"])
                self.assertEqual(resumed["initial_interactions"]["cumulative"], 256)
                self.assertEqual(resumed["training_interactions"]["cumulative"], 512)
                self.assertEqual(resumed["training_interactions"]["session"], 256)
                self.assertEqual(
                    resumed["external_update_counter_before"],
                    first["external_update_counter_after"],
                )
                self.assertNotEqual(
                    resumed["policy_state_before"], resumed["policy_state_after"]
                )
                self.assertEqual(
                    resumed["policy_state_before"], first["policy_state_after"]
                )
                self.assertEqual(
                    resumed["checkpoint"]["run_id"], first["checkpoint"]["run_id"]
                )
                before_rng = state_sha256(capture_rng())
                loaded, _ = load_policy(folder / "resumed/checkpoint")
                model, _ = load_model(folder / "resumed/checkpoint")
                self.assertEqual(state_sha256(capture_rng()), before_rng)
                frame = torch.linspace(-1, 1, 49).repeat(3, 1)
                before_policy = loaded.state_sha256()
                loaded.reset(torch.ones(3, dtype=torch.bool))
                action = loaded.act(frame)
                expected, _ = model.predict(frame.numpy(), deterministic=True)
                np.testing.assert_array_equal(action.numpy(), expected)
                self.assertTrue(torch.isfinite(action).all())
                self.assertTrue((action.numpy() >= model.action_space.low).all())
                self.assertTrue((action.numpy() <= model.action_space.high).all())
                self.assertEqual(loaded.state_sha256(), before_policy)
                self.assertEqual(state_sha256(capture_rng()), before_rng)
                self.assertEqual(SB3FramePolicy(model).state_sha256(), before_policy)

    def test_invalid_allowances_do_not_collect_or_publish_a_checkpoint(self):
        for algorithm, allowance in (("ppo", 129), ("sac", 257)):
            with (
                self.subTest(algorithm=algorithm),
                tempfile.TemporaryDirectory() as tmp,
            ):
                task = FakeTask()
                with self.assertRaisesRegex(ValueError, "multiple"):
                    run_training(
                        task,
                        algorithm=algorithm,
                        total_interactions=allowance,
                        output=Path(tmp),
                    )
                self.assertEqual(task.interactions["cumulative"], 0)
                self.assertEqual(task.reset_calls, 0)
                self.assertFalse((Path(tmp) / "checkpoint").exists())


if __name__ == "__main__":
    unittest.main()
