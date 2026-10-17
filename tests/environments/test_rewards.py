"""Actual reward metadata and pinned native arithmetic, without custom recipes."""

import importlib.util
import json
import unittest
from types import SimpleNamespace as NS

import torch

from parkour_lab.environments.rewards import reward_recipe
from tests.support.isaaclab import NativeConfig, modules, pinned_source_root


class RewardMetadataTests(unittest.TestCase):
    def test_known_formula_tracks_callable_identity_and_actual_parameters(self):
        def function(env):
            return torch.zeros(env.num_envs)

        function.__module__ = "isaaclab.envs.mdp.rewards"
        function.__qualname__ = "track_lin_vel_xy_exp"
        term = NativeConfig(func=function, weight=0, params={"std": 0.37})
        env = NS(
            step_dt=0.02,
            reward_manager=NS(
                active_terms=["track_lin_vel_xy_exp"], get_term_cfg=lambda name: term
            ),
        )
        original = reward_recipe(env)["terms"]["track_lin_vel_xy_exp"]
        self.assertIsNotNone(original["formula"])
        self.assertEqual(original["weight"], 0)
        self.assertEqual(original["params"], {"std": 0.37})
        term.func = lambda env: torch.zeros(env.num_envs)
        changed = reward_recipe(env)["terms"]["track_lin_vel_xy_exp"]
        self.assertIsNone(changed["formula"])
        self.assertIsNotNone(original["formula"])

    def test_added_stateful_callable_and_native_selector_parameters_are_json_safe(self):
        class Reward:
            def __call__(self, env, **kwargs):
                return torch.zeros(env.num_envs)

        selector = NativeConfig(
            "robot", body_names=["base"], body_ids=slice(None), joint_ids=[0, 1]
        )
        vector = torch.tensor([1.0, 2.0])
        params = {"asset_cfg": selector, "vector": vector, "nested": ({"gain": 2},)}
        term = NativeConfig(func=Reward(), weight=-0.1, params=params)
        manager = NS(active_terms=["custom"], get_term_cfg=lambda name: term)
        recipe = reward_recipe(NS(reward_manager=manager, step_dt=0.02))
        report = recipe["terms"]["custom"]
        self.assertTrue(report["function"].endswith(".<locals>.Reward"))
        self.assertIsNone(report["formula"])
        self.assertEqual(report["params"]["vector"], [1.0, 2.0])
        self.assertEqual(
            report["params"]["asset_cfg"]["body_ids"],
            {"start": None, "stop": None, "step": None},
        )
        json.dumps(recipe, allow_nan=False)
        report["params"]["asset_cfg"]["body_names"].append("other")
        report["params"]["nested"][0]["gain"] = 99
        self.assertEqual(selector.body_names, ["base"])
        self.assertEqual(params["nested"], ({"gain": 2},))
        torch.testing.assert_close(params["vector"], vector)


class NativeRewardTests(unittest.TestCase):
    def test_native_reward_manager_applies_weight_and_dt_once_and_skips_zero(self):
        path = pinned_source_root() / "managers/reward_manager.py"
        name = "isaaclab.managers.reward_manager"
        spec = importlib.util.spec_from_file_location(name, path)
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
        manager._env = object()
        manager._term_names = ["active", "disabled"]

        def disabled(env):
            self.fail("A zero-weight reward must not be evaluated")

        manager._term_cfgs = [
            NativeConfig(
                func=lambda env: torch.tensor([2.0, 3.0]), weight=-0.5, params={}
            ),
            NativeConfig(func=disabled, weight=0, params={}),
        ]
        manager._episode_sums = {name: torch.zeros(2) for name in manager._term_names}
        manager._reward_buf = torch.zeros(2)
        manager._step_reward = torch.ones((2, 2))
        torch.testing.assert_close(manager.compute(0.02), torch.tensor([-0.02, -0.03]))
        torch.testing.assert_close(
            manager._step_reward, torch.tensor([[-1.0, 0.0], [-1.5, 0.0]])
        )
        torch.testing.assert_close(manager._episode_sums["disabled"], torch.zeros(2))

    def test_native_air_time_requires_first_contact_and_translation_above_threshold(
        self,
    ):
        path = (
            pinned_source_root().parents[1]
            / "isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py"
        )
        spec = importlib.util.spec_from_file_location("_pinned_velocity_rewards", path)
        native = importlib.util.module_from_spec(spec)
        with modules(
            {
                "isaaclab.envs": NS(mdp=NS()),
                "isaaclab.managers": NS(SceneEntityCfg=NativeConfig),
                "isaaclab.sensors": NS(ContactSensor=object),
                "isaaclab.utils.math": NS(quat_apply_inverse=None, yaw_quat=None),
            }
        ):
            spec.loader.exec_module(native)
        commands = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        )
        contact = torch.ones((6, 4))
        contact[3:] = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
        )
        periods = []

        def first_contact(dt):
            periods.append(dt)
            return contact

        sensor = NS(
            data=NS(last_air_time=torch.tensor([[0.2, 0.7, 0.1, 0.8]]).repeat(6, 1)),
            compute_first_contact=first_contact,
        )
        env = NS(
            step_dt=0.02,
            scene=NS(sensors={"contact_forces": sensor}),
            command_manager=NS(get_command=lambda name: commands),
        )
        result = native.feet_air_time(
            env,
            "base_velocity",
            NativeConfig("contact_forces", body_ids=slice(None)),
            0.5,
        )
        torch.testing.assert_close(
            result, torch.tensor([0.0, 0.0, 0.0, -0.3, 0.5, 0.0])
        )
        self.assertEqual(periods, [0.02])
