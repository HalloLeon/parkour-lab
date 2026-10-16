"""Fixed causal ROA export/inference parity; no native or deployment evidence."""

import copy
import unittest
from unittest.mock import patch

import torch
from parkour_lab.control.controller import ControllerInput, ControllerSession
from parkour_lab.control.motor_contract import (
    nominal_binding_sha256,
    make_motor_contract,
)
from parkour_lab.methods.roa import model as roa
from parkour_lab.methods.roa import runtime
from parkour_lab.environments.randomization import TaskRandomization
from parkour_lab.methods.roa.environment import ROAEnvironment
from parkour_lab.runtime.native import sensor_noise_report
from parkour_lab.runtime.training import TrainingHost
from torch import nn

from tests.support.native import TrainingEnvironmentFixture, motor_binding
from tests.support.roa import packet, stock_motor, random_frames


class ROARuntimeTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(141)
        self.actor = roa.ROAActor(stock_motor())
        with torch.no_grad():
            self.actor.motor.latent_projection.weight.normal_(std=0.07)
        self.names = tuple(f"j{i}" for i in range(12))
        self.default = torch.linspace(-0.5, 0.5, 12)
        binding = motor_binding(self.names, self.default.tolist(), count=3)
        self.profile = "native_motor_sha256:" + nominal_binding_sha256(binding)
        self.contract = make_motor_contract(binding, self.profile)
        self.controller = self.make_controller()

    def make_controller(self, motor=None, estimator=None, **overrides):
        motor = self.actor.motor if motor is None else motor
        estimator = self.actor.estimator if estimator is None else estimator
        args = {
            "joint_names": self.names,
            "default_position_rad": self.default,
            "artifact_sha256": runtime.roa_tensor_sha256(motor, estimator),
            "actuator_profile": self.profile,
        }
        args.update(overrides)
        return runtime.ROAHistoryController(motor, estimator, **args)

    def session(self, controller=None):
        controller = self.controller if controller is None else controller
        return ControllerSession(
            controller, joint_names=self.names, actuator_profile=self.profile
        )

    def test_terminal_critic_reads_do_not_consume_or_receive_causal_noise(self):
        fixture = TrainingEnvironmentFixture(count=3)
        # The corrected flat native group is clean; corruption belongs to its host.
        observation = fixture.observations

        def clean_observations():
            result = observation()
            result["proprio"] = result["policy"][:, 3:].clone()
            return result

        fixture.observations = clean_observations
        task = dict(num_envs=3, seed=71, dynamics="nominal")
        fixture.env.cfg.parkour_task = task
        fixture.env.parkour_randomization = TaskRandomization(task)
        expected_stream = TaskRandomization(task)
        host = TrainingHost(fixture.env, fixture.app)
        environment = ROAEnvironment(host)
        initial, _ = environment.reset()
        self.assertEqual(sensor_noise_report(fixture.env)["generated_frames"], 1)
        self.assertFalse(
            torch.equal(initial["policy"][:, :6], initial["critic_state"][:, 3:9])
        )
        expected_stream.uniform(None, 30, "observation-noise", -1.0, 1.0)
        fixture.next_terminated = torch.tensor([True, False, False])
        fixture.next_timed_out = torch.tensor([False, True, False])
        following, _, done, info = environment.step(torch.zeros(3, 12), "ppo")
        self.assertEqual(sensor_noise_report(fixture.env)["generated_frames"], 2)
        self.assertEqual(
            set(info["final_observation"].keys()), {"critic_state", "terrain"}
        )
        final_clean = info["final_observation"]["critic_state"]
        self.assertTrue(
            torch.equal(final_clean[:, 3:6], fixture.data.root_ang_vel_b[:2])
        )
        noise = expected_stream.uniform(None, 30, "observation-noise", -1.0, 1.0)
        noise *= [0.2] * 3 + [0.05] * 3 + [0.01] * 12 + [1.5] * 12
        noise = torch.tensor(noise, dtype=torch.float32)
        expected = fixture.records[-1]["next_proprio"].clone()
        expected[:, :6] += noise[:, :6]
        expected[:, 9:33] += noise[:, 6:]
        self.assertTrue(torch.equal(following["policy"], expected))
        self.assertTrue(
            torch.equal(following["history"][done], expected[done].repeat(1, 25))
        )
        self.assertTrue((following["policy"][done, 33:] == 0).all())

    def test_exact_history_action_and_estimate_parity_with_partial_resets(self):
        controller = self.make_controller()
        session = self.session(controller)
        history = roa.CausalHistory()
        previous = torch.zeros(3, 12)
        with patch.object(
            controller._history, "push", wraps=controller._history.push
        ) as push:
            for step in range(70):
                reset = torch.tensor([step == 0, step in (0, 9, 41), step in (0, 25)])
                frame = random_frames(3)
                frame[:, 33:45] = previous
                frame[reset, 33:] = 0
                frames = history.push(frame, reset)
                with torch.inference_mode():
                    expected = self.actor.history_action(frame, frames)
                    estimate = self.actor.estimate(frames)
                targets = session.step(**packet(controller, frame, step, reset))
                self.assertTrue(torch.equal(targets.raw_action, expected))
                self.assertTrue(
                    torch.equal(targets.position_rad, self.default + 0.25 * expected)
                )
                self.assertTrue(torch.equal(controller.last_estimate, estimate))
                self.assertTrue(torch.equal(controller.history_frames, frames))
                self.assertEqual(targets.joint_names, self.names)
                previous = targets.raw_action.clone()
            self.assertEqual(push.call_count, 70)

    def test_explicit_kernel_parity_for_arbitrary_history_without_private_push(self):
        for count in (1, 3, 32):
            with self.subTest(count=count), torch.inference_mode():
                history = random_frames(count, 25)
                frame = history[:, -1]
                action, estimate = self.controller.infer(frame, history)
                self.assertTrue(
                    torch.equal(action, self.actor.history_action(frame, history))
                )
                self.assertTrue(torch.equal(estimate, self.actor.estimate(history)))
                self.assertIsNone(self.controller.history_frames)
                self.assertIsNone(self.controller.last_estimate)

    def test_current_command_and_previous_delivered_action_are_packed_exactly(self):
        frame = random_frames(2)
        frame[:, 33:] = 0
        session = self.session()
        args = packet(self.controller, frame, 0, torch.ones(2, dtype=torch.bool))
        args["command"] = torch.tensor([[0.4, 0.2, 0.5], [-0.3, -0.2, -0.5]])
        first = session.step(**args)
        expected = frame.clone()
        expected[:, 6:9] = args["command"]
        self.assertTrue(
            torch.equal(
                self.controller.history_frames, expected[:, None].repeat(1, 25, 1)
            )
        )
        frame[:, 33:45] = first.raw_action
        session.step(
            **packet(self.controller, frame, 1, torch.zeros(2, dtype=torch.bool))
        )
        self.assertTrue(
            torch.equal(self.controller.history_frames[:, -1, 33:45], first.raw_action)
        )

    def test_controller_has_only_frozen_causal_modules_and_no_weight_alias(self):
        self.assertEqual(self.controller.spec.period_s, 0.02)
        self.assertEqual(self.controller.spec.configuration["history_length"], 25)
        self.assertEqual(
            self.controller.spec.configuration["foot_contacts"]["timing"],
            "latest completed physics sample; zero until first post-reset step",
        )
        modules = {
            name
            for name, value in vars(self.controller).items()
            if isinstance(value, nn.Module)
        }
        self.assertEqual(modules, {"motor", "estimator"})
        self.assertFalse(
            any(sensor.privileged for sensor in self.controller.spec.sensors.values())
        )
        for original, private in (
            (self.actor.motor, self.controller.motor),
            (self.actor.estimator, self.controller.estimator),
        ):
            self.assertFalse(private.training)
            for before, after in zip(original.parameters(), private.parameters()):
                self.assertFalse(after.requires_grad)
                self.assertNotEqual(before.data_ptr(), after.data_ptr())
                self.assertTrue(torch.equal(before, after))
        with torch.no_grad():
            for parameter in self.actor.encoder.parameters():
                parameter.fill_(float("nan"))
        history = random_frames(2, 25)
        self.assertTrue(
            torch.isfinite(self.controller.infer(history[:, -1], history)[0]).all()
        )

    def test_history_diagnostics_are_clones_and_new_controller_has_no_live_state(self):
        frame = random_frames(2)
        frame[:, 33:] = 0
        self.session().step(
            **packet(self.controller, frame, 0, torch.ones(2, dtype=torch.bool))
        )
        history = self.controller.history_frames
        code = self.controller.last_estimate
        history.zero_()
        code.zero_()
        self.assertFalse(torch.equal(history, self.controller.history_frames))
        self.assertFalse(torch.equal(code, self.controller.last_estimate))
        fresh = self.make_controller()
        self.assertIsNone(fresh.history_frames)
        self.assertIsNone(fresh.last_estimate)

    def test_reset_contract_and_zero_previous_action_enforced(self):
        for reset in (
            torch.tensor([False, True]),
            torch.zeros(2),
            torch.ones(2, 1, dtype=torch.bool),
            None,
        ):
            with self.subTest(reset=reset), self.assertRaises(ValueError):
                self.controller.reset(reset)
        frame = torch.zeros(2, 49)
        args = packet(self.controller, frame, 0, torch.ones(2, dtype=torch.bool))
        inputs = ControllerInput(
            **{key: value for key, value in args.items() if key != "reset_mask"}
        )
        with self.assertRaisesRegex(ValueError, "reset"):
            self.controller.act(inputs)
        self.controller.reset(args["reset_mask"])
        with self.assertRaises(ValueError):
            self.controller.reset(args["reset_mask"])
        frame[:, 33:45] = 1
        with self.assertRaisesRegex(ValueError, "zero"):
            self.controller.act(inputs)
        frame[:, 33:45] = 0
        self.controller.act(inputs)
        with self.assertRaisesRegex(ValueError, "reset"):
            self.controller.act(inputs)
        with self.assertRaises(ValueError):
            self.controller.reset(torch.ones(3, dtype=torch.bool))

    def test_finite_fixed_shape_dtype_and_newest_frame_guard(self):
        history = random_frames(3, 25)
        frame = history[:, -1].clone()
        for changed_frame, changed_history in (
            (frame.double(), history.double()),
            (frame[:, :44], history),
            (frame, history[:, :24]),
            (frame[:0], history[:0]),
            (frame + 1, history),
            (frame, history.clone().fill_(float("nan"))),
        ):
            with (
                self.subTest(frame_shape=changed_frame.shape),
                self.assertRaises(ValueError),
            ):
                self.controller.infer(changed_frame, changed_history)

    def test_nonbinary_or_stale_reset_contacts_are_rejected(self):
        history = random_frames(2, 25)
        history[:, 0, 45] = 0.5
        with self.assertRaisesRegex(ValueError, "binary"):
            self.controller.infer(history[:, -1], history)
        frame = torch.zeros(2, 49)
        frame[:, 45] = 1
        with self.assertRaisesRegex(ValueError, "contacts must be zero"):
            self.session().step(
                **packet(self.controller, frame, 0, torch.ones(2, dtype=torch.bool))
            )

    def test_input_sensor_dtype_failure_faults_generic_session(self):
        frame = torch.zeros(2, 49, dtype=torch.float64)
        session = self.session()
        with self.assertRaises(ValueError):
            session.step(
                **packet(self.controller, frame, 0, torch.ones(2, dtype=torch.bool))
            )
        self.assertTrue(session.faulted)
        self.assertIsNone(self.controller.history_frames)

    def test_constructor_rejects_nonfixed_architecture_or_identity(self):
        for change in ("activation", "extra_encoder", "estimator_width", "dtype"):
            motor, estimator = (
                copy.deepcopy(self.actor.motor),
                copy.deepcopy(self.actor.estimator),
            )
            if change == "activation":
                motor.tail[0] = nn.ReLU()
            elif change == "extra_encoder":
                motor.encoder = copy.deepcopy(self.actor.encoder)
            elif change == "estimator_width":
                estimator[0] = nn.Linear(1226, 128)
            else:
                motor.double()
                estimator.double()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.make_controller(motor, estimator)
        for overrides in (
            {"artifact_sha256": "f" * 64},
            {"joint_names": ("same",) * 12},
            {"actuator_profile": "unknown"},
            {"default_position_rad": torch.zeros(11)},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.make_controller(**overrides)


if __name__ == "__main__":
    unittest.main()
