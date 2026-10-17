"""Caller-owned policies receive only causal frames and retain bounded delivery."""

import hashlib
import unittest

import torch

from parkour_lab.control.controller import ControllerSession, Sample
from parkour_lab.control.policy import PolicyController


class LoadedPolicy:
    def __init__(self):
        self.weights = torch.arange(12, dtype=torch.float32)
        self.normalization = torch.ones(49)
        self.memory = torch.ones(2, 12)
        self.frames = []

    def state_sha256(self):
        state = torch.cat((self.weights, self.normalization)).numpy().tobytes()
        return hashlib.sha256(state).hexdigest()

    def reset(self, mask):
        self.memory[mask] = 0

    def act(self, observation):
        self.frames.append(observation.clone())
        self.memory += 1
        return self.weights.expand(len(observation), -1) * 100


class PolicyControllerTests(unittest.TestCase):
    def controller(self, policy):
        return PolicyController(
            policy,
            joint_names=tuple(f"joint_{index}" for index in range(12)),
            default_position_rad=torch.zeros(12),
            target_limits_rad=[[-0.5, 0.5]] * 12,
            actuator_profile="fixture",
        )

    def step(self, controller, session, *, time=0.0, reset=None):
        sensors = {}
        values = []
        for index, (name, spec) in enumerate(controller.spec.sensors.items()):
            value = torch.full((2, *spec.shape), float(index + 1))
            values.append(value)
            sensors[name] = Sample(
                value, time, torch.ones(2, dtype=torch.bool), spec.units, spec.frame
            )
        command = torch.tensor([[0.35, 0.0, -0.3]]).expand(2, -1)
        result = session.step(
            time_s=time,
            command=command,
            command_time_s=time,
            sensors=sensors,
            reset_mask=torch.ones(2, dtype=torch.bool) if reset is None else reset,
        )
        return result, torch.cat((*values[:2], command, *values[2:]), dim=-1)

    def session(self, controller):
        return ControllerSession(
            controller,
            joint_names=controller.spec.joint_names,
            actuator_profile=controller.spec.actuator_profile,
        )

    def test_loaded_policy_gets_exact_frame_and_bounded_targets_without_state_updates(
        self,
    ):
        policy = LoadedPolicy()
        controller = self.controller(policy)
        before = controller.state_sha256()
        result, expected = self.step(controller, self.session(controller))
        torch.testing.assert_close(policy.frames[0], expected)
        self.assertEqual(policy.frames[0].shape, (2, 49))
        self.assertFalse(policy.frames[0].requires_grad)
        torch.testing.assert_close(result.position_rad, result.raw_action * 0.25)
        self.assertTrue((result.position_rad.abs() <= 0.5).all())
        self.assertEqual(before, controller.state_sha256())
        policy.normalization[0] += 1
        self.assertNotEqual(before, controller.state_sha256())

    def test_episode_memory_resets_selected_rows_and_command_changes_do_not_reset(self):
        policy = LoadedPolicy()
        controller = self.controller(policy)
        session = self.session(controller)
        self.step(controller, session)
        self.step(
            controller, session, time=0.02, reset=torch.zeros(2, dtype=torch.bool)
        )
        torch.testing.assert_close(policy.memory, torch.full((2, 12), 2.0))
        self.step(controller, session, time=0.04, reset=torch.tensor([True, False]))
        torch.testing.assert_close(policy.memory[0], torch.ones(12))
        torch.testing.assert_close(policy.memory[1], torch.full((12,), 3.0))

    def test_invalid_output_latches_existing_session_without_fallback(self):
        for invalid in (torch.zeros(2, 11), torch.full((2, 12), torch.nan)):
            with self.subTest(shape=invalid.shape):
                policy = LoadedPolicy()
                policy.act = lambda observation: invalid
                controller = self.controller(policy)
                session = self.session(controller)
                with self.assertRaises(ValueError):
                    self.step(controller, session)
                self.assertTrue(session.faulted)

    def test_invalid_state_identity_and_physical_binding_reject(self):
        policy = LoadedPolicy()
        policy.state_sha256 = lambda: "opaque-checkpoint-name"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.controller(policy)
        policy = LoadedPolicy()
        with self.assertRaisesRegex(ValueError, "contain"):
            PolicyController(
                policy,
                joint_names=tuple(f"joint_{index}" for index in range(12)),
                default_position_rad=torch.ones(12),
                target_limits_rad=[[-0.5, 0.5]] * 12,
                actuator_profile="fixture",
            )


if __name__ == "__main__":
    unittest.main()
