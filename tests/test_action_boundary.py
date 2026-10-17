"""Bounded native delivery, reward feedback and outgoing terminal observations."""

import unittest

import torch

from parkour_lab.control.action import (
    BOUNDED_RAW_ACTION_MEANING,
    JointActionTransform,
)
from parkour_lab.control.controller import JointTargets
from parkour_lab.control.motor_contract import make_motor_contract
from parkour_lab.runtime.motor import (
    NativeJointTargetBridge,
    _runtime_motor_binding,
)
from tests.support.public_task import PublicTaskFixture


def fixture_with_limits():
    fixture = PublicTaskFixture()
    default = fixture.data.default_joint_pos
    width = torch.linspace(0.4, 2.0, 12)
    mechanical = torch.stack((default - width, default + 1.2 * width), dim=-1)
    center = (mechanical[..., 0] + mechanical[..., 1]) / 2
    half = 0.5 * (mechanical[..., 1] - mechanical[..., 0]) * 0.9
    fixture.data.soft_joint_pos_limits = torch.stack((center - half, center + half), -1)
    fixture.data.joint_limits = mechanical
    return fixture


class ActionBoundaryTests(unittest.TestCase):
    def test_extreme_requests_reach_bounded_targets_feedback_and_action_rate(self):
        fixture = fixture_with_limits()
        original_step = fixture.native_step
        fixture.env.reward_manager.active_terms = ["action_rate_l2", "unused"]

        def step(action):
            previous = fixture.env.action_manager.action.clone()
            result = original_step(action)
            rate = -0.01 * (action - previous).square().sum(-1)
            fixture.env.reward_manager._step_reward[:, 0] = rate
            fixture.env.reward_manager._step_reward[:, 1] = 0
            return result[0], rate * fixture.env.step_dt, *result[2:]

        fixture.native_step = step
        with fixture.task() as task:
            task.reset()
            limits = fixture.data.soft_joint_pos_limits
            delivered_previous = torch.zeros(4, 12)
            for index, sign in enumerate((1, -1)):
                raw = torch.full((4, 12), sign * 4.46e8)
                preserved = raw.clone()
                if index:
                    fixture.next_terminated = torch.tensor([True, False, False, False])
                observation, reward, terminated, truncated, info = task.step(raw)
                done = terminated | truncated
                delivered = fixture.deliveries[-1]
                target = fixture.data.joint_pos_target
                self.assertTrue(torch.equal(raw, preserved))
                self.assertTrue((target >= limits[..., 0]).all())
                self.assertTrue((target <= limits[..., 1]).all())
                self.assertLess(float(delivered.abs().max()), 20)
                expected_reward = (
                    -0.01 * (delivered - delivered_previous).square().sum(-1) * 0.02
                )
                torch.testing.assert_close(reward, expected_reward, rtol=0, atol=0)
                self.assertLess(float(reward.abs().max()), 2)
                self.assertTrue(
                    torch.equal(observation["actor"][~done, 33:45], delivered[~done])
                )
                self.assertFalse(observation["actor"][done, 33:45].any())
                if done.any():
                    self.assertTrue(
                        torch.equal(
                            info["final_observation"]["actor"][:, 33:45],
                            delivered[done],
                        )
                    )
                delivered_previous = delivered.clone()

    def test_named_target_bridge_matches_public_delivery_and_checks_native_limits(self):
        fixture = fixture_with_limits()
        with fixture.task() as task:
            task.reset()
            binding, digest = _runtime_motor_binding(fixture.env)
            profile = "native_motor_sha256:" + digest
            limits = fixture.data.soft_joint_pos_limits[0].tolist()
            contract = make_motor_contract(binding, profile, target_limits_rad=limits)
            manifest = {
                "joint_names": binding["joint_names"],
                "period_s": 0.02,
                "actuator_profile": profile,
                "configuration": {
                    "default_position_rad": binding["default_position_rad"],
                    "action_clip": {
                        "version": "joint_limits_v1",
                        "target_limits_rad": limits,
                    },
                },
            }
            manifest["raw_action_meaning"] = BOUNDED_RAW_ACTION_MEANING
            bridge = NativeJointTargetBridge(
                fixture.env, contract, manifest, preserve_native_raw=True
            )
            for sign in (1, -1):
                raw = sign * torch.linspace(-4.46e8, 4.46e8, 12).repeat(4, 1)
                delivered, target = JointActionTransform(bridge.default, limits)(raw)
                encoded = bridge.encode(JointTargets(bridge.joint_names, target, raw))
                task.step(raw)
                self.assertTrue(torch.equal(fixture.data.joint_pos_target, target))
                torch.testing.assert_close(encoded, delivered, rtol=0, atol=0)
                bridge.verify_delivery(
                    torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)
                )
            boundary = bridge.progress()["action_boundary"]
            self.assertEqual(boundary["total_joint_commands"], 96)
            self.assertEqual(boundary["saturated_joint_commands"], 96)
            self.assertEqual(boundary["saturated_row_commands"], 8)
            self.assertEqual(
                boundary["max_requested_action_abs"], float(raw.abs().max())
            )
            fixture.data.soft_joint_pos_limits[:, 0, 1] += 0.01
            with self.assertRaisesRegex(ValueError, "native soft joint limits"):
                NativeJointTargetBridge(fixture.env, contract, manifest)

    def test_transform_requires_finite_ordered_joint_limits_containing_default(self):
        default = torch.linspace(-0.3, 0.3, 12)
        with self.assertRaises(TypeError):
            JointActionTransform(default)
        for limits in (
            None,
            [],
            [[0.0, 0.0]] * 12,
            [[0.0, 1.0]] * 12,
            [[-1.0, float("inf")]] * 12,
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                JointActionTransform(default, limits)
