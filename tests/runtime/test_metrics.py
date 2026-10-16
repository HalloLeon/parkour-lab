"""Signed native reward accounting and compact pre-reset physical diagnostics."""

import unittest

import torch

from parkour_lab.runtime.metrics import TransitionMetrics, diagnostic_state
from tests.support.native import make_native_env


class TransitionMetricsTests(unittest.TestCase):
    def setUp(self):
        self.env = make_native_env(5)
        self.metrics = TransitionMetrics(self.env)
        self.rates = torch.tensor([[2.0, -0.5]]).repeat(5, 1)
        self.reward = self.rates.sum(-1) * self.env.step_dt
        self.state = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.1]]).repeat(5, 1)
        self.zero = torch.zeros(5, dtype=torch.bool)

    def add(self, command, phase="ppo", terminated=None, truncated=None):
        self.metrics.add(
            phase,
            command,
            self.reward,
            self.rates,
            self.state,
            self.zero if terminated is None else terminated,
            self.zero if truncated is None else truncated,
        )

    def test_signed_terms_apply_dt_once_and_learning_phases_remain_separate(self):
        commands = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5],
                [0.2, 0.0, 0.0],
                [0.2, 0.0, 0.5],
                [0.0, 0.0, 0.0],
            ]
        )
        self.add(commands)
        self.add(
            torch.zeros_like(commands),
            "history",
            terminated=torch.tensor([False, False, True, False, False]),
            truncated=torch.tensor([False, True, True, False, False]),
        )
        result = self.metrics.drain()
        self.assertEqual(set(result), {"ppo", "history"})
        ppo, history = result["ppo"]["regimes"], result["history"]["regimes"]
        self.assertEqual(
            {k: v["samples"] for k, v in ppo.items()},
            dict(initial_stop=2, pivot=1, planar=1, arc=1),
        )
        self.assertEqual(
            {k: v["samples"] for k, v in history.items()},
            dict(initial_stop=2, later_stop=3),
        )
        stop = history["later_stop"]
        self.assertEqual(
            (stop["terminated"], stop["truncated"], stop["simultaneous"]), (1, 2, 1)
        )
        self.assertAlmostEqual(stop["simulated_seconds"], 0.06)
        self.assertEqual(stop["reward_terms"]["tracking"]["weighted_rate_mean"], 2)
        self.assertAlmostEqual(
            stop["reward_terms"]["tracking"]["contribution_sum"], 0.12
        )
        self.assertAlmostEqual(
            stop["reward_terms"]["effort"]["contribution_sum"], -0.03
        )
        self.assertAlmostEqual(stop["reward_contribution_sum"], 0.09)
        self.assertAlmostEqual(ppo["pivot"]["physical_mean"]["yaw_error_rad_s"], 0.5)
        self.assertAlmostEqual(ppo["planar"]["physical_mean"]["planar_error_m_s"], 0.2)
        self.assertEqual(self.metrics.drain(), {})
        self.add(torch.zeros_like(commands))
        following = self.metrics.drain()["ppo"]["regimes"]
        self.assertEqual(following["initial_stop"]["samples"], 4)
        self.assertEqual(following["later_stop"]["samples"], 1)
        self.metrics.reset()
        self.add(torch.zeros_like(commands))
        self.assertEqual(
            self.metrics.drain()["ppo"]["regimes"]["initial_stop"]["samples"], 5
        )

    def test_bad_reward_or_nonfinite_state_cannot_produce_valid_summary(self):
        for field in ("reward", "state"):
            with self.subTest(field=field):
                self.setUp()
                getattr(self, field)[0] = 100 if field == "reward" else float("nan")
                self.add(torch.zeros(5, 3))
                with self.assertRaisesRegex(ValueError, "diagnostics|reconcile"):
                    self.metrics.drain()

    def test_float32_signed_cancellation_uses_absolute_contribution_scale(self):
        self.rates[:] = torch.tensor([10000.0, -9999.5])
        # The SDK sums separately timestep-scaled terms, rather than scaling a sum.
        self.reward = self.rates[:, 0] * 0.02 + self.rates[:, 1] * 0.02
        self.add(torch.zeros(5, 3))
        result = self.metrics.drain()["ppo"]
        self.assertGreater(result["reward_reconciliation_abs_max"], 1e-6)
        self.reward += 0.1
        self.add(torch.zeros(5, 3))
        with self.assertRaisesRegex(ValueError, "reconcile"):
            self.metrics.drain()

    def test_compact_state_uses_surface_height_posture_and_body_com_velocity(self):
        data = self.env.scene["robot"].data
        data.root_pos_w[:, 2] = 1.4
        self.env.scene["base_height_scanner"].data.ray_hits_w[:, 0, 2] = 1.0
        data.root_lin_vel_b[:] = torch.tensor([0.2, 0.3, 0.4])
        data.root_ang_vel_b[:, 2] = 0.5
        data.joint_pos.copy_(data.default_joint_pos)
        data.joint_pos[:, 0] += 0.1
        actual = diagnostic_state(self.env)
        self.assertEqual(actual.shape, (5, 7))
        expected = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.4, 0.1, 0.0]).repeat(5, 1)
        torch.testing.assert_close(actual, expected)
        data.root_lin_vel_b.zero_()
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
