"""Native motor timing and delivery contracts with CPU doubles, not physics."""

import unittest

import torch

from parkour_lab.control.action import BOUNDED_RAW_ACTION_MEANING
from parkour_lab.control.controller import JointTargets
from parkour_lab.control.motor_contract import make_motor_contract
from parkour_lab.runtime.motor import (
    NativeJointTargetBridge,
    _runtime_motor_binding,
)
from tests.support.native import NativeEnvironmentFixture


def bridge_for(env):
    binding, digest = _runtime_motor_binding(env)
    limits = env.scene["robot"].data.soft_joint_pos_limits[0].tolist()
    manifest = {
        "joint_names": binding["joint_names"],
        "configuration": {
            "default_position_rad": binding["default_position_rad"],
            "action_clip": {
                "version": "joint_limits_v1",
                "target_limits_rad": limits,
            },
        },
        "period_s": 0.02,
        "actuator_profile": "native_motor_sha256:" + digest,
        "raw_action_meaning": BOUNDED_RAW_ACTION_MEANING,
    }
    contract = make_motor_contract(
        binding, manifest["actuator_profile"], target_limits_rad=limits
    )
    return NativeJointTargetBridge(env, contract, manifest, preserve_native_raw=True)


class NativeTimingTests(unittest.TestCase):
    def test_both_rates_retain_control_tick_and_exact_action_delivery(self):
        for dt, decimation in ((0.005, 4), (0.0025, 8)):
            with self.subTest(dt=dt):
                fixture = NativeEnvironmentFixture(count=3)
                fixture.env.physics_dt = dt
                fixture.env.cfg.decimation = decimation
                bridge = bridge_for(fixture.env)
                native, _ = fixture.env.reset()
                self.assertEqual(native["proprio"].shape, (3, 49))
                self.assertEqual(bridge.binding["step_dt_s"], 0.02)
                self.assertEqual(bridge.binding["physics_dt_s"], dt)
                self.assertEqual(bridge.binding["decimation"], decimation)
                action = torch.full((3, 12), 0.125)
                target = bridge.default + 0.25 * action
                delivered = bridge.encode(
                    JointTargets(bridge.joint_names, target, action)
                )
                _, _, terminated, truncated, _ = fixture.env.step(delivered)
                bridge.verify_delivery(terminated, truncated)
                self.assertTrue(torch.equal(fixture.env.action_manager.action, action))
                self.assertEqual(bridge.progress()["verified_delivery_steps"], 1)
                self.assertEqual(bridge.progress()["native_verified_rows"], 3)

    def test_wrong_control_period_or_substep_pair_rejects(self):
        for step, dt, decimation in (
            (0.02, 0.005, 8),
            (0.02, 0.0025, 4),
            (0.04, 0.005, 8),
            (0.02, 0.001, 20),
        ):
            fixture = NativeEnvironmentFixture(count=3)
            fixture.env.step_dt, fixture.env.physics_dt = step, dt
            fixture.env.cfg.decimation = decimation
            with self.subTest(step=step, dt=dt), self.assertRaises(ValueError):
                bridge_for(fixture.env)
