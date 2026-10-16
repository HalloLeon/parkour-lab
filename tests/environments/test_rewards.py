"""Reward configuration, posture arithmetic and native manager weighting contracts."""

import importlib.util
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace as NS

import torch

from parkour_lab.config import ExperimentConfig, TaskConfig
from parkour_lab.environments.rewards import (
    configure_rewards,
    joint_posture,
    reward_recipe,
    validate_rewards,
)
from tests.support.isaaclab import (
    NativeConfig,
    modules,
    native_config_modules,
    pinned_source_root,
    stock_config,
)


class RewardConfigurationTests(unittest.TestCase):
    def test_default_experiments_have_independent_reward_overrides(self):
        first, second = ExperimentConfig(), ExperimentConfig()
        first.task.rewards["action_rate_l2"] = {"weight": -0.02}
        self.assertEqual(second.task.rewards, {})
        self.assertIsNot(first.task.rewards, second.task.rewards)

    def test_json_round_trip_and_nested_defensive_copy(self):
        overrides = {
            "feet_air_time": {"weight": 0.02, "params": {"threshold": 0.4}},
            "joint_posture": {"weight": -0.1, "params": {"stand_still_scale": 4.0}},
        }
        validated = validate_rewards(overrides)
        task = TaskConfig(rewards=overrides)
        self.assertEqual(validated, overrides)
        config = ExperimentConfig(task=task)
        self.assertEqual(
            ExperimentConfig.from_dict(json.loads(json.dumps(config.to_dict()))), config
        )
        overrides["feet_air_time"]["params"]["threshold"] = 99
        self.assertEqual(validated["feet_air_time"]["params"]["threshold"], 0.4)
        self.assertEqual(task.rewards["feet_air_time"]["params"]["threshold"], 0.4)
        external = config.to_dict()
        external["task"]["rewards"]["joint_posture"]["weight"] = -99
        self.assertEqual(task.rewards["joint_posture"]["weight"], -0.1)

    def test_invalid_names_shapes_weights_and_parameters_fail_early(self):
        for value in (
            None,
            [],
            "native",
            {"unknown": {"weight": 1}},
            {"action_rate_l2": -0.01},
            {"action_rate_l2": {"enabled": False}},
            {"action_rate_l2": {"weight": True}},
            {"action_rate_l2": {"weight": "-0.1"}},
            {"action_rate_l2": {"weight": float("nan")}},
            {"action_rate_l2": {"weight": float("inf")}},
            {"action_rate_l2": {"weight": 0.1}},
            {"track_lin_vel_xy_exp": {"weight": -1}},
            {"feet_air_time": {"weight": -0.1}},
            {"joint_posture": {"weight": 0.1}},
            {"feet_air_time": {"params": []}},
            {"feet_air_time": {"params": {"threshold": -0.1}}},
            {"feet_air_time": {"params": {"threshold": True}}},
            {"feet_air_time": {"params": {"sensor_cfg": "other"}}},
            {"track_lin_vel_xy_exp": {"params": {"std": 0.3}}},
            {"track_ang_vel_z_exp": {"params": {"std": 0.3}}},
            {"joint_posture": {"params": {"stand_still_scale": 0.9}}},
            {"joint_posture": {"params": {"velocity_threshold": -0.1}}},
            {"joint_posture": {"params": {"velocity_threshold": float("inf")}}},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TaskConfig(rewards=value)

    def test_zero_disables_terms_and_parameter_lower_bounds_are_allowed(self):
        overrides = {
            "action_rate_l2": {"weight": 0},
            "track_lin_vel_xy_exp": {"weight": 0},
            "feet_air_time": {"weight": 0, "params": {"threshold": 0}},
            "joint_posture": {
                "weight": 0,
                "params": {"stand_still_scale": 1, "velocity_threshold": 0},
            },
        }
        self.assertEqual(validate_rewards(overrides), overrides)

    def test_receipt_uses_actual_manager_terms_and_serializes_fixed_selector(self):
        cfg = stock_config()
        with native_config_modules():
            configure_rewards(
                cfg, TaskConfig(rewards={"action_rate_l2": {"weight": 0}})
            )
        actual = deepcopy(
            {name: term for name, term in vars(cfg.rewards).items() if term is not None}
        )
        actual["track_lin_vel_xy_exp"].params["std"] = 0.37
        env = NS(
            step_dt=0.02,
            reward_manager=NS(
                active_terms=list(actual), get_term_cfg=actual.__getitem__
            ),
        )
        receipt = reward_recipe(env)
        self.assertEqual(receipt["control_dt_s"], 0.02)
        self.assertEqual(
            receipt["terms"]["track_lin_vel_xy_exp"]["params"]["std"], 0.37
        )
        self.assertEqual(cfg.rewards.track_lin_vel_xy_exp.params["std"], 0.5)
        self.assertEqual(receipt["terms"]["action_rate_l2"]["weight"], 0)
        self.assertEqual(receipt["terms"]["joint_posture"]["weight"], 0)
        self.assertEqual(
            receipt["terms"]["feet_air_time"]["params"]["sensor_cfg"],
            {"name": "contact_forces", "body_names": ".*_foot"},
        )
        self.assertEqual(json.loads(json.dumps(receipt)), receipt)
        self.assertTrue(all(term["function"] for term in receipt["terms"].values()))
        self.assertTrue(receipt["terms"]["joint_posture"]["formula"])
        # Native configuration doubles have unrelated callable identities.
        self.assertIsNone(receipt["terms"]["track_lin_vel_xy_exp"]["formula"])
        receipt["terms"]["track_lin_vel_xy_exp"]["params"]["std"] = 99
        self.assertEqual(actual["track_lin_vel_xy_exp"].params["std"], 0.37)


class CustomRewardMetadataTests(unittest.TestCase):
    def test_replacing_known_function_drops_its_original_formula_annotation(self):
        def replacement(env):
            return torch.zeros(env.num_envs)

        term = NativeConfig(func=joint_posture, weight=-0.1, params={})
        manager = NS(active_terms=["joint_posture"], get_term_cfg=lambda name: term)
        env = NS(reward_manager=manager, step_dt=0.02)
        original = reward_recipe(env)["terms"]["joint_posture"]
        self.assertIsNotNone(original["formula"])

        term.func = replacement
        changed = reward_recipe(env)["terms"]["joint_posture"]
        self.assertEqual(
            changed["function"], f"{replacement.__module__}:{replacement.__qualname__}"
        )
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


class JointPostureTests(unittest.TestCase):
    def test_unsquared_all_joint_error_and_exact_stop_speed_gating(self):
        commands = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1e-12, 0.0, 0.0],
                [0.0, 0.0, 0.1],
                [0.0, 0.2, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        nominal = torch.full((7, 12), 0.5)
        joint = nominal.clone()
        joint[:, (0, 1, 11)] += torch.tensor([3.0, 4.0, 12.0])
        velocity = torch.zeros((7, 3))
        velocity[1, 0] = 0.3
        velocity[2, 0] = torch.nextafter(torch.tensor(0.3), torch.tensor(float("inf")))
        velocity[6] = torch.tensor([0.2, 0.2, 99.0])
        data = NS(
            joint_pos=joint,
            default_joint_pos=nominal,
            root_lin_vel_b=velocity,
        )
        env = NS(
            scene={"robot": NS(data=data)},
            command_manager=NS(get_command=lambda name: commands),
        )
        before = joint.clone()
        torch.testing.assert_close(
            joint_posture(env), torch.tensor([65.0, 65.0, 13.0, 13.0, 13.0, 13.0, 65.0])
        )
        torch.testing.assert_close(
            joint_posture(env, stand_still_scale=1.0), torch.full((7,), 13.0)
        )
        torch.testing.assert_close(joint, before)
        data.joint_pos = nominal
        torch.testing.assert_close(joint_posture(env), torch.zeros(7))

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


if __name__ == "__main__":
    unittest.main()
