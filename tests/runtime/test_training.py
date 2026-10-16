"""Native learner timing and delivery contract with CPU doubles, not physics."""

import unittest

import torch

from parkour_lab.runtime.training import TrainingHost
from tests.support.native import TrainingEnvironmentFixture


class TrainingTimingTests(unittest.TestCase):
    def test_both_rates_retain_control_tick_and_exact_action_delivery(self):
        for dt, decimation in ((0.005, 4), (0.0025, 8)):
            with self.subTest(dt=dt):
                fixture = TrainingEnvironmentFixture(count=3)
                fixture.env.physics_dt = dt
                fixture.env.cfg.decimation = decimation
                host = TrainingHost(fixture.env, fixture.app)
                native, _ = host.reset()
                self.assertEqual(native["proprio"].shape, (3, 49))
                self.assertEqual(host.manifest["period_s"], 0.02)
                self.assertEqual(host.motor_contract["binding"]["physics_dt_s"], dt)
                self.assertEqual(
                    host.motor_contract["binding"]["decimation"], decimation
                )
                action = torch.full((3, 12), 0.125)
                host.step(action, "timing")
                self.assertTrue(torch.equal(fixture.env.action_manager.action, action))
                self.assertEqual(host.steps, 1)
                self.assertEqual(host.bridge.progress()["native_verified_rows"], 3)

    def test_wrong_control_period_or_substep_pair_rejects(self):
        for step, dt, decimation in (
            (0.02, 0.005, 8),
            (0.02, 0.0025, 4),
            (0.04, 0.005, 8),
            (0.02, 0.001, 20),
        ):
            fixture = TrainingEnvironmentFixture(count=3)
            fixture.env.step_dt, fixture.env.physics_dt = step, dt
            fixture.env.cfg.decimation = decimation
            with (
                self.subTest(step=step, dt=dt, decimation=decimation),
                self.assertRaises(ValueError),
            ):
                TrainingHost(fixture.env, fixture.app)
