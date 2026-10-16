"""Public task configuration and native reward customization without PhysX."""

import importlib.util
import json
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as NS

import torch

from parkour_lab.benchmark.config import TASK_IDS, Go2Config, build_config
from parkour_lab.config import TaskConfig as WorkflowTaskConfig
from parkour_lab.environments.configuration import build_environment_config
from parkour_lab.environments.rewards import reward_recipe
from parkour_lab.environments.task_config import TaskConfig
from tests.support.isaaclab import (
    NativeConfig,
    NativeRewardTerm,
    modules,
    native_config_modules,
    pinned_source_root,
    stock_rough_config,
)


def rough_recipe():
    cfg = stock_rough_config()
    cfg.rewards.feet_air_time.weight = 0.01
    return cfg


def native_reward_manager(cfg):
    """Run native term discovery and arithmetic, doubling only its base/Sim layer."""
    name = "isaaclab.managers.reward_manager"
    spec = importlib.util.spec_from_file_location(
        name, pinned_source_root() / "managers/reward_manager.py"
    )
    module = importlib.util.module_from_spec(spec)
    with modules(
        {
            "prettytable": NS(PrettyTable=None),
            "isaaclab.managers.manager_base": NS(
                ManagerBase=object, ManagerTermBase=type("ManagerTermBase", (), {})
            ),
            "isaaclab.managers.manager_term_cfg": NS(RewardTermCfg=NativeConfig),
        }
    ):
        spec.loader.exec_module(module)
    manager = module.RewardManager.__new__(module.RewardManager)
    # Configuration doubles use namespaces; present native term types to the SDK.
    manager.cfg = {
        name: NativeRewardTerm(func=term.func, weight=term.weight, params=term.params)
        for name, term in vars(cfg).items()
        if term is not None
    }
    manager._env = NS(num_envs=2)
    manager._term_names, manager._term_cfgs, manager._class_term_cfgs = [], [], []
    manager._resolve_common_term_cfg = lambda *args, **kwargs: None
    manager._prepare_terms()
    manager._episode_sums = {name: torch.zeros(2) for name in manager._term_names}
    manager._reward_buf = torch.zeros(2)
    manager._step_reward = torch.zeros((2, len(manager._term_names)))
    return manager


class PublicConfigTests(unittest.TestCase):
    def test_pure_configuration_needs_no_method_or_simulator_import(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """import sys
class Guard:
    def find_spec(self, fullname, *args):
        if fullname == 'parkour_lab.config' or fullname.startswith(('parkour_lab.methods', 'isaaclab', 'rsl_rl')):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from parkour_lab.benchmark.config import Go2Config, TASK_IDS, build_config
assert Go2Config().task_id == 'Go2-Flat-v1'
for name, (_, supported) in TASK_IDS.items():
    if not supported:
        try:
            build_config(Go2Config(task_id=name))
        except NotImplementedError:
            pass
        else:
            raise AssertionError(name)
""",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIs(WorkflowTaskConfig, TaskConfig)
        self.assertEqual(TaskConfig().terrain, "procedural")

    def test_options_are_frozen_and_unsupported_definitions_are_honest(self):
        options = Go2Config()
        with self.assertRaises(FrozenInstanceError):
            options.seed = 1
        self.assertEqual(len(TASK_IDS), 6)
        self.assertEqual(
            [name for name, (_, ready) in TASK_IDS.items() if ready], [options.task_id]
        )
        for kwargs in (
            {"task_id": "missing"},
            {"task_id": []},
            {"num_envs": 0},
            {"num_envs": True},
            {"seed": -1},
            {"seed": 2**32},
            {"seed": True},
            {"device": " "},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Go2Config(**kwargs)
        with self.assertRaises(ValueError):
            build_config(options, training=1)

    def test_flat_uses_rough_native_rewards_and_only_causal_observation_group(self):
        rough = rough_recipe()
        with native_config_modules(rough=rough):
            cfg = build_config(Go2Config(num_envs=7, seed=13, device="cpu"))
        self.assertEqual(cfg.sim.dt, 0.005)
        self.assertEqual(
            (cfg.decimation, cfg.episode_length_s, cfg.scene.num_envs), (4, 20, 7)
        )
        self.assertEqual((cfg.seed, cfg.sim.device), (13, "cpu"))
        self.assertEqual(cfg.benchmark_version, "go2-blind-v1")
        self.assertEqual(cfg.benchmark_task_id, "Go2-Flat-v1")
        self.assertTrue(cfg.benchmark_training)
        self.assertIsNone(cfg.observations.policy)
        self.assertIsNone(cfg.observations.terrain)
        self.assertIsNotNone(cfg.observations.proprio)
        self.assertIsNone(cfg.terminations.procedural_physical_failure)
        self.assertFalse(cfg.terminations.base_contact.time_out)
        self.assertTrue(cfg.terminations.time_out.time_out)
        for name, expected in vars(rough.rewards).items():
            with self.subTest(term=name):
                actual = getattr(cfg.rewards, name)
                if expected is None:
                    self.assertIsNone(actual)
                else:
                    self.assertEqual(actual.weight, expected.weight)
                    self.assertIs(actual.func, expected.func)
                    self.assertEqual(actual.params, expected.params)
                    self.assertIsNot(actual, expected)
        self.assertFalse(hasattr(cfg.rewards, "joint_posture"))
        cfg.rewards.feet_air_time.params["threshold"] = 0.37
        self.assertEqual(rough.rewards.feet_air_time.params["threshold"], 0.5)

    def test_current_workflow_defaults_stay_unchanged(self):
        with native_config_modules(rough=rough_recipe()):
            cfg = build_environment_config(TaskConfig(terrain="flat"))
            evaluation = build_config(Go2Config(device="cpu"), training=False)
        self.assertEqual(cfg.rewards.flat_orientation_l2.weight, -2.5)
        self.assertEqual(cfg.rewards.dof_pos_limits.weight, -10)
        self.assertEqual(cfg.rewards.feet_air_time.weight, 0.01)
        self.assertIsNotNone(cfg.terminations.procedural_physical_failure)
        self.assertIsNotNone(cfg.observations.policy)
        self.assertIsNotNone(cfg.observations.terrain)
        self.assertFalse(evaluation.benchmark_training)
        self.assertEqual(
            evaluation.commands.base_velocity.resampling_time_range, (1e9, 1e9)
        )

    def test_native_configuration_accepts_added_reward_and_zero_disables_computation(
        self,
    ):
        with native_config_modules(rough=rough_recipe()):
            cfg = build_config(Go2Config(device="cpu"))
        for term in vars(cfg.rewards).values():
            if term is not None:
                term.weight = 0
        calls = []

        def extra(env, target):
            calls.append(target)
            return torch.full((env.num_envs,), target)

        def disabled(env):
            self.fail("A zero-weight custom reward was evaluated")

        cfg.rewards.extra = NativeRewardTerm(
            func=extra, weight=2, params={"target": 3.0}
        )
        cfg.rewards.disabled = NativeRewardTerm(func=disabled, weight=0)
        manager = native_reward_manager(cfg.rewards)
        torch.testing.assert_close(manager.compute(0.02), torch.full((2,), 0.12))
        self.assertEqual(calls, [3.0])
        recipe = reward_recipe(NS(reward_manager=manager, step_dt=0.02))
        self.assertIsNone(recipe["terms"]["extra"]["formula"])
        self.assertEqual(recipe["terms"]["extra"]["params"], {"target": 3.0})
        self.assertTrue(
            recipe["terms"]["extra"]["function"].endswith(
                ":PublicConfigTests.test_native_configuration_accepts_added_reward_and_zero_disables_computation.<locals>.extra"
            )
        )
        json.dumps(recipe, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
