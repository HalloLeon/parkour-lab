"""ROA observation history, measured dynamics and native delivery on CPU doubles."""

import unittest
from unittest.mock import patch

import torch
from parkour_lab.runtime.training import TrainingHost
from parkour_lab.methods.roa import environment

from tests.support.native import TrainingEnvironmentFixture


class ROAEnvironmentTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        torch.set_num_threads(1)
        torch.manual_seed(137)
        self.fixture = TrainingEnvironmentFixture()

    def host(self):
        return environment.ROAEnvironment(
            TrainingHost(self.fixture.env, self.fixture.app)
        )

    def test_measured_dynamics_are_ordered_nonconstant_and_not_aliased(self):
        measured = environment.read_dynamics(self.fixture.env)
        self.assertEqual(measured.shape, (8, 7))
        torch.testing.assert_close(measured[:, 0], self.fixture.mass[:, 0] / 10 - 1)
        torch.testing.assert_close(measured[:, 1:4], self.fixture.com[:, 0, :3])
        torch.testing.assert_close(measured[:, 4:], self.fixture.material.mean(dim=1))
        self.assertGreater(float(measured[:, 0].std()), 0)
        original = measured.clone()
        self.fixture.mass.add_(1)
        torch.testing.assert_close(measured, original, rtol=0, atol=0)

    def test_bad_mass_nonfinite_or_wrong_width_dynamics_reject(self):
        for defect in ("mass", "nominal", "com", "material", "width"):
            fixture = TrainingEnvironmentFixture()
            if defect == "mass":
                fixture.mass[0] = 0
            elif defect == "nominal":
                fixture.data.default_mass[0] = -1
            elif defect == "com":
                fixture.com[0, 0, 0] = float("nan")
            elif defect == "material":
                fixture.material[0, 0, 1] = float("inf")
            else:
                fixture.material = torch.ones(8, 2, 4)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                environment.read_dynamics(fixture.env)

    def test_post_reset_previous_action_and_timeout_bootstrap_masks_are_exact(self):
        self.fixture.env.scene["contact_forces"].data.net_forces_w.fill_(2)
        host = self.host()
        observations, reset = host.reset()
        self.assertTrue(reset.all())
        self.assertTrue((observations["policy"][:, 33:45] == 0).all())
        terminated = torch.tensor(
            [True, False, True, False, False, False, False, False]
        )
        timeout = torch.tensor([False, True, True, False, False, False, False, False])
        self.fixture.next_terminated, self.fixture.next_timed_out = terminated, timeout
        raw = torch.linspace(-0.25, 0.25, 12).repeat(8, 1)
        before = observations["history"].reshape(8, 25, 49).clone()
        observations, _, done, extras = host.step(raw, "privileged_ppo")
        self.assertTrue(torch.equal(done, terminated | timeout))
        self.assertTrue(torch.equal(extras["time_outs"], timeout & ~terminated))
        self.assertEqual(extras["native_extra"], "retained")
        self.assertFalse(extras["time_outs"][2])
        self.assertTrue(torch.equal(extras["terminated"], terminated))
        self.assertTrue(torch.equal(extras["truncated"], timeout))
        self.assertEqual(extras["final_env_ids"].tolist(), [0, 1, 2])
        final = extras["final_observation"]
        self.assertEqual(set(final.keys()), {"critic_state", "terrain"})
        torch.testing.assert_close(final["critic_state"][:, 36:48], raw[:3])
        self.assertTrue((final["critic_state"][:, -4:] == 1).all())
        self.assertTrue((observations["policy"][done, -4:] == 0).all())
        self.assertTrue((observations["policy"][~done, -4:] == 1).all())
        self.assertTrue(torch.equal(observations["policy"][~done, 33:45], raw[~done]))
        self.assertTrue((observations["policy"][done, 33:45] == 0).all())
        history = observations["history"].reshape(8, 25, 49)
        self.assertTrue(torch.equal(history[:, -1], observations["policy"]))
        self.assertTrue(torch.equal(history[~done, :-1], before[~done, 1:]))
        self.assertTrue(
            torch.equal(
                history[done], observations["policy"][done, None].repeat(1, 25, 1)
            )
        )
        self.assertEqual(host.host.steps, 1)
        self.assertEqual(host.host.resets, 3)
        self.assertEqual(host.host.partial_reset_steps, 1)
        self.assertEqual(
            host.host.phase_counts["privileged_ppo"],
            {"control_steps": 1, "terminated_rows": 2, "timeout_rows": 2},
        )
        self.assertEqual(host.host.bridge.progress()["excluded_terminal_rows"], 3)
        self.assertEqual(host.host.bridge.progress()["native_verified_rows"], 5)
        retained = final.clone()
        host.step(raw * 2, "privileged_ppo")
        for key in retained.keys():
            torch.testing.assert_close(final[key], retained[key], rtol=0, atol=0)

    def test_both_actors_use_shared_noisy_frame_history_and_critic_uses_clean_state(
        self,
    ):
        host = self.host()
        host.reset()
        native = self.fixture.observations()
        delivered = host.observations(native, torch.zeros(8, dtype=torch.bool))
        torch.testing.assert_close(
            delivered["policy"], native["proprio"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            delivered["critic_state"], native["policy"], rtol=0, atol=0
        )
        self.assertFalse(
            torch.equal(delivered["policy"][:, :3], delivered["critic_state"][:, 3:6])
        )
        self.assertEqual(delivered["policy"].shape, (8, 49))
        self.assertEqual(delivered["history"].shape, (8, 1225))
        self.assertTrue(
            torch.equal(
                delivered["history"].reshape(8, 25, 49)[:, -1], delivered["policy"]
            )
        )
        self.assertEqual(delivered["terrain"].shape, (8, 264))
        snapshot = delivered.clone()
        for value in native.values():
            value.zero_()
        for name in snapshot.keys():
            self.assertTrue(torch.equal(delivered[name], snapshot[name]))

    def test_misaligned_pre_action_evidence_rejects_without_another_native_step(self):
        for defect in (
            "previous",
            "command",
            "com",
            "shape",
            "contact",
            "reset_contact",
        ):
            host = self.host()
            host.reset()
            native = self.fixture.observations()
            if defect == "previous":
                native["proprio"][0, 33] = 1
            elif defect == "command":
                native["proprio"][0, 6] += 0.1
            elif defect == "com":
                native["policy"][0, 0] += 0.1
            elif defect == "shape":
                native["proprio"] = native["proprio"][:, :-1]
            else:
                value = 0.5 if defect == "contact" else 1
                native["proprio"][0, -1] = native["policy"][0, -1] = value
            with (
                self.subTest(defect=defect),
                self.assertRaisesRegex(ValueError, "alignment"),
            ):
                host.host.observations(native, torch.ones(8, dtype=torch.bool))
            self.assertEqual(self.fixture.env.index, 0)

    def test_persistent_dynamics_change_on_reset_and_app_closure_fail_closed(self):
        host = self.host()
        self.fixture.on_reset = lambda: self.fixture.mass.add_(1)
        with self.assertRaisesRegex(ValueError, "Persistent dynamics"):
            host.reset()
        self.fixture.on_reset = lambda: None
        host = self.host()
        host.reset()
        self.fixture.running = False
        with self.assertRaisesRegex(RuntimeError, "application stopped"):
            host.step(torch.zeros(8, 12), "privileged_ppo")
        self.assertEqual(self.fixture.env.index, 0)

    def test_motor_delivery_mismatch_is_not_a_learning_sample(self):
        host = self.host()
        host.reset()
        self.fixture.on_step = lambda: self.fixture.term.processed_actions.add_(0.01)
        with self.assertRaisesRegex(ValueError, "positions differ"):
            host.step(torch.zeros(8, 12), "privileged_ppo")
        self.assertEqual(host.host.steps, 0)
        self.assertEqual(self.fixture.env.index, 1)
        self.assertEqual(host.host.bridge.progress()["native_step_returns"], 1)

    def test_command_override_preserves_sensor_draw_and_pushes_history_once(self):
        host = self.host()
        with patch.object(host.history, "push", wraps=host.history.push) as push:
            observations, _ = host.reset(seed=1042, command=(0.4, 0.1, 0.0))
            self.assertEqual(push.call_count, 1)
            self.assertEqual(self.fixture.reset_seeds[-1], 1042)
            expected = torch.tensor((0.4, 0.1, 0.0)).expand(8, 3)
            self.assertTrue(torch.equal(observations["policy"][:, 6:9], expected))
            self.assertTrue(
                torch.equal(observations["critic_state"][:, 9:12], expected)
            )
            self.assertTrue(torch.isinf(self.fixture.command_term.time_left).all())
            self.assertFalse(self.fixture.command_term.is_standing_env.any())
            self.assertFalse(self.fixture.command_term.is_heading_env.any())
            history_before = observations["history"].reshape(8, 25, 49).clone()
            observations, _, done, _ = host.step(
                torch.full((8, 12), 0.125),
                "command_screen",
                next_command=(0.0, 0.0, -0.5),
            )
            self.assertEqual(push.call_count, 2)
        native = self.fixture.records[-1]["next_proprio"]
        keep = torch.tensor([i for i in range(49) if i not in (6, 7, 8)])
        self.assertTrue(torch.equal(observations["policy"][:, keep], native[:, keep]))
        self.assertTrue(torch.equal(native[:, 6:9], expected))
        self.assertTrue(
            torch.equal(
                observations["policy"][:, 6:9],
                torch.tensor((0.0, 0.0, -0.5)).expand(8, 3),
            )
        )
        history = observations["history"].reshape(8, 25, 49)
        self.assertTrue(torch.equal(history[:, -1], observations["policy"]))
        self.assertTrue(torch.equal(history[~done, :-1], history_before[~done, 1:]))


if __name__ == "__main__":
    unittest.main()
