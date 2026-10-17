"""Exercise collector checks with real SB3 and corrupt their evidence deliberately."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from examples.sb3.environment import Go2VecEnv
from examples.sb3.tests.fixtures import FakeTask
from scripts import _learning_checks as native_checks
from scripts import check_learning_contracts as check


class ActionFrameTask(FakeTask):
    def step(self, actions):
        self.state[:, 33:45] = actions
        return super().step(actions)


@contextmanager
def fixture_steps(task, *, force_timeout=False):
    original, records = task.step, []

    def step(actions):
        if force_timeout and not records:
            task.next_terminated = torch.zeros(task.num_envs, dtype=torch.bool)
            task.next_truncated = torch.arange(task.num_envs) == 0
        transition = original(actions)
        observation, reward, terminated, truncated, info = transition
        final = info["final_observation"]
        final = None if final is None else final["actor"]
        records.append(
            dict(
                reward=reward.clone(),
                terminated=terminated.clone(),
                truncated=truncated.clone(),
                final_actor=None if final is None else final.clone(),
                next_actor=observation["actor"].clone(),
            )
        )
        return transition

    task.step = step
    try:
        yield records
    finally:
        task.step = original


class ContractCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_real_collectors_without_optimizer_and_wrong_reset_frame_rejected(self):
        app = SimpleNamespace(is_running=lambda: True)
        task = ActionFrameTask()
        with patch.object(native_checks, "record_native_steps", fixture_steps):
            ppo = check._check_ppo(task, app)
            sac = check._check_sac(task, app)
            self.assertNotEqual(ppo["terminal_value"], ppo["replacement_value"])
            self.assertEqual(ppo["optimizer_updates"], 0)
            self.assertEqual(sac["optimizer_updates"], 0)
            self.assertEqual(sac["replay_timeouts"], [1, 0, 0, 0])
            self.assertEqual(sac["sampled_done_values"], [0.0])
            self.assertEqual(task.interactions["session"], 4 * task.num_envs)
            self.assertEqual(task.reset_calls, 2)
            outgoing = Go2VecEnv._outgoing

            def wrong_final(env, current, *args):
                result = outgoing(env, current, *args)
                for row, info in enumerate(result[-1]):
                    if "terminal_observation" in info:
                        info["terminal_observation"] = current[row].copy()
                return result

            for collector in (check._check_ppo, check._check_sac):
                with self.subTest(collector=collector.__name__):
                    with patch.object(Go2VecEnv, "_outgoing", wrong_final):
                        with self.assertRaises(AssertionError):
                            collector(ActionFrameTask(), app)

    def test_failed_check_keeps_receipt_and_closes_both_resources(self):
        import parkour_lab.benchmark as benchmark

        task = ActionFrameTask()
        app = SimpleNamespace(close=lambda: setattr(app, "closed", True), closed=False)
        launcher = SimpleNamespace(AppLauncher=lambda **_: SimpleNamespace(app=app))
        cfg = SimpleNamespace(to_dict=lambda: {})
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(sys.modules, {"isaaclab.app": launcher}),
            patch.object(benchmark, "build_config", return_value=cfg),
            patch.object(benchmark, "make", return_value=task),
            patch.object(native_checks, "dependency_evidence", return_value={}),
            patch.object(
                check,
                "_check_ppo",
                side_effect=RuntimeError("injected collection failure"),
            ),
            patch.object(check, "exit_native_process") as exit_process,
        ):
            output = Path(tmp) / "new"
            code = check.main(["--device", "cpu", "--output", str(output)])
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(code, 2)
            self.assertIn("injected collection failure", report["error"])
            self.assertEqual(
                report["cleanup"], dict(environment="complete", application="complete")
            )
            self.assertEqual(report["interactions"]["session"], 0)
            self.assertTrue(task.closed and app.closed)
            exit_process.assert_called_once_with(2)

    def test_noise_guard_rejects_an_extra_terminal_draw(self):
        generator = np.random.Generator(np.random.PCG64(1077))

        def state():
            return {
                "generators": [{"observation-noise": generator.bit_generator.state}]
            }

        before = state()
        generator.uniform(-1, 1, size=30)
        native_checks._check_noise(before, state())
        generator.uniform(-1, 1, size=30)
        with self.assertRaises(AssertionError):
            native_checks._check_noise(before, state())

    def test_delivery_guard_reads_physical_targets_and_ending_witness(self):
        task = ActionFrameTask(num_envs=2)
        raw = torch.tensor([[1e6] * 12, [-1e6] * 12])
        delivered = raw.clamp(
            torch.as_tensor(task.single_action_space.low),
            torch.as_tensor(task.single_action_space.high),
        )
        target = 0.25 * delivered
        action, processed, actual = delivered.clone(), target.clone(), target.clone()
        limits = torch.stack(
            (
                0.25 * torch.as_tensor(task.single_action_space.low),
                0.25 * torch.as_tensor(task.single_action_space.high),
            ),
            dim=-1,
        )
        data = SimpleNamespace(
            default_joint_pos=torch.zeros(2, 12),
            joint_pos_target=actual,
            soft_joint_pos_limits=limits.expand(2, -1, -1),
        )
        term = SimpleNamespace(processed_actions=processed)
        task._env = SimpleNamespace(
            scene={"robot": SimpleNamespace(data=data)},
            action_manager=SimpleNamespace(action=action, get_term=lambda _: term),
        )
        actor = torch.zeros(2, 49)
        actor[:, 33:45] = delivered
        final = actor[:1].clone()
        transition = (
            {"actor": actor},
            None,
            torch.tensor([False, False]),
            torch.tensor([True, False]),
            {"final_env_ids": torch.tensor([0]), "final_observation": {"actor": final}},
        )
        witness = dict(
            ids=torch.tensor([0]),
            actor=final.clone(),
            buffers={
                "action": action[:1].clone(),
                "processed_target": processed[:1].clone(),
                "joint_target": actual[:1].clone(),
            },
        )
        action[0] = processed[0] = actual[0] = (
            0  # Replacement buffers are deliberately ignored.
        )
        result = native_checks._check_delivery(task, raw, transition, witness)
        self.assertEqual(result["saturated_joints"], 24)
        self.assertEqual(result["saturated_rows"], 2)
        actual[1, 0] += 0.001
        with self.assertRaisesRegex(AssertionError, "joint_target"):
            native_checks._check_delivery(task, raw, transition, witness)

    def test_help_imports_no_native_or_learning_runtime(self):
        program = """
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in
               ('torch', 'stable_baselines3', 'isaaclab', 'isaacsim', 'omni', 'pxr')):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.argv = ['check_learning_contracts', '--help']
runpy.run_module('scripts.check_learning_contracts', run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--num-envs", result.stdout)


if __name__ == "__main__":
    unittest.main()
