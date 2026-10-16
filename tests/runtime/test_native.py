"""Native host contracts for non-GRU mock controllers, entirely on CPU."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

import parkour_lab.runtime.native as host_module
from parkour_lab.control.controller import ControllerSpec, JointTargets, SensorSpec
from parkour_lab.control.motor_contract import (
    make_motor_contract,
    nominal_binding_sha256,
)
from parkour_lab.control.proprioception import FOOT_NAMES
from parkour_lab.environments.randomization import TaskRandomization
from parkour_lab.runtime.training import TrainingHost
from tests.support.native import make_native_env, rebatch


class MockController:
    """Named inputs and absolute targets, with no neural-network assumptions."""

    def __init__(self, reference, sensors, default, *, private_raw=False):
        self.spec = ControllerSpec(
            name="cpu_mock_target_controller",
            artifact_sha256="c" * 64,
            preprocessing_version="mock_named_inputs_v1",
            joint_names=reference.joint_names,
            period_s=reference.period_s,
            actuator_profile=reference.actuator_profile,
            sensors=sensors,
            raw_action_meaning="three private diagnostic values, never native actions",
            configuration={"default_position_rad": default[0].tolist()},
        )
        self.default = default.clone()
        self.private_raw = private_raw
        self.calls, self.resets = [], []

    def reset(self, mask):
        self.resets.append(mask.clone())

    def act(self, inputs):
        self.calls.append(inputs)
        raw = inputs.command + 123.0 if self.private_raw else None
        return JointTargets(self.spec.joint_names, self.default.clone(), raw)


class NativeControllerHostTests(unittest.TestCase):
    def test_connected_motion_owns_aligned_named_forces_feet_and_rays(self):
        from types import SimpleNamespace as NS

        self.env.cfg.parkour_task = {"terrain": "connected"}
        robot = self.env.scene["robot"]
        robot.body_names = ["base", "FL_foot", "RR_foot", "RL_foot", "FR_foot"]
        robot.data.body_pos_w.copy_(torch.arange(15).reshape(1, 5, 3))
        sensor = self.env.scene["contact_forces"]
        sensor.data.net_forces_w.copy_(torch.arange(15).reshape(1, 5, 3) * 2)
        self.env.scene["height_scanner"] = NS(
            data=NS(ray_hits_w=torch.ones(80, 132, 3))
        )
        sample = host_module.motion_state(self.env)
        torch.testing.assert_close(
            sample["foot_position_w"], robot.data.body_pos_w[:, [4, 1, 2, 3]]
        )
        torch.testing.assert_close(
            sample["foot_net_forces_w"], sensor.data.net_forces_w[:, [2, 4, 3, 1]]
        )
        robot.data.body_pos_w.zero_()
        sensor.data.net_forces_w.zero_()
        self.env.scene["height_scanner"].data.ray_hits_w.zero_()
        self.assertTrue(sample["foot_position_w"].any())
        self.assertTrue(sample["foot_net_forces_w"].any())
        self.assertTrue((sample["terrain_hits_w"] == 1).all())
        self.assertEqual(sample["base_surface_hits_w"].shape, (80, 1, 3))

    def setUp(self):
        self.env = make_native_env(80)
        from parkour_lab.runtime.motor import _runtime_motor_binding

        binding = rebatch(_runtime_motor_binding(self.env)[0], 4)
        profile = "native_motor_sha256:" + nominal_binding_sha256(binding)
        self.reference = ControllerSpec(
            name="native_fixture",
            artifact_sha256="a" * 64,
            preprocessing_version="test",
            joint_names=tuple(binding["joint_names"]),
            period_s=0.02,
            actuator_profile=profile,
            sensors={},
            raw_action_meaning="none",
        )
        self.contract = make_motor_contract(binding, profile)
        self.robot = self.env.scene["robot"].data

    def controller(self, sensors=None, *, private_raw=False):
        if sensors is None:
            sensors = {"joint_position": SensorSpec((12,), "rad", "joint", 0.02)}
        return MockController(
            self.reference,
            sensors,
            self.robot.default_joint_pos,
            private_raw=private_raw,
        )

    def host(self, controller):
        return host_module.NativeControllerSession(self.env, controller, self.contract)

    def act(self, host, *, step=0, reset=None):
        command = torch.tensor([[0.4, -0.2, 0.5]]).repeat(self.env.num_envs, 1)
        if reset is None:
            reset = torch.full((self.env.num_envs,), step == 0, dtype=torch.bool)
        return host.act(command, time_s=step * 0.02, reset_mask=reset)

    def enable_noise(self, env):
        env.cfg.parkour_task = dict(num_envs=env.num_envs, seed=71, dynamics="nominal")
        env.parkour_randomization = TaskRandomization(
            env.cfg.parkour_task, evaluation=True
        )

    def all_sensor_controller(self):
        return self.controller(
            {
                name: SensorSpec(*meaning, max_age_s=0.02)
                for name, meaning in host_module.NATIVE_SENSORS.items()
            }
        )

    def test_seeded_noise_reaches_named_sensors_without_changing_simulator_truth(self):
        self.enable_noise(self.env)
        controller = self.all_sensor_controller()
        host = self.host(controller)
        source = {
            "base_ang_vel": self.robot.root_ang_vel_b.clone(),
            "projected_gravity": self.robot.projected_gravity_b.clone(),
            "joint_position_relative_default": (
                self.robot.joint_pos - self.robot.default_joint_pos
            ).clone(),
            "joint_position": self.robot.joint_pos.clone(),
            "joint_velocity": self.robot.joint_vel.clone(),
        }
        raw_truth = host_module.motion_state(self.env)
        self.act(host)
        samples, noise = controller.calls[-1].sensors, host.last_sensor_noise
        amplitudes = noise.new_tensor([0.2] * 3 + [0.05] * 3 + [0.01] * 12 + [1.5] * 12)
        self.assertTrue((noise.abs() <= amplitudes).all())
        for name, delta in zip(
            (
                "base_ang_vel",
                "projected_gravity",
                "joint_position_relative_default",
                "joint_velocity",
            ),
            noise.split((3, 3, 12, 12), dim=-1),
            strict=True,
        ):
            self.assertTrue(torch.equal(samples[name].value, source[name] + delta))
            self.assertFalse(torch.equal(samples[name].value, source[name]))
        self.assertTrue(
            torch.equal(
                samples["joint_position"].value,
                source["joint_position"] + noise[:, 6:18],
            )
        )
        torch.testing.assert_close(
            samples["joint_position"].value - self.robot.default_joint_pos,
            samples["joint_position_relative_default"].value,
            rtol=0,
            atol=6e-8,
        )
        for key, value in raw_truth.items():
            self.assertTrue(torch.equal(value, host_module.motion_state(self.env)[key]))
        self.assertTrue(torch.equal(self.robot.joint_pos, source["joint_position"]))
        self.assertTrue(torch.equal(self.robot.joint_vel, source["joint_velocity"]))
        self.assertTrue(
            torch.equal(
                samples["foot_contacts"].value, host_module.foot_contacts(self.env)
            )
        )
        self.assertTrue(
            torch.equal(
                samples["stock_previous_raw_action"].value,
                self.env.action_manager.action,
            )
        )
        self.assertTrue(
            torch.equal(
                controller.calls[-1].command,
                self.env.command_manager.get_command("base_velocity"),
            )
        )
        self.assertEqual(
            host_module.sensor_noise_report(self.env)["generated_frames"], 1
        )

    def test_replay_reproduces_noise_and_privileged_reads_do_not_advance_it(self):
        self.enable_noise(self.env)
        host = self.host(self.all_sensor_controller())
        expected = TaskRandomization(self.env.cfg.parkour_task, evaluation=True)
        for step in range(3):
            host_module.motion_state(self.env)
            host_module.foot_contacts(self.env)
            self.act(host, step=step)
            draw = expected.uniform(None, 30, "observation-noise", -1.0, 1.0)
            draw *= [0.2] * 3 + [0.05] * 3 + [0.01] * 12 + [1.5] * 12
            self.assertTrue(
                torch.equal(
                    host.last_sensor_noise, torch.tensor(draw, dtype=torch.float32)
                )
            )
        original_report = host_module.sensor_noise_report(self.env)
        replay = make_native_env(self.env.num_envs)
        self.enable_noise(replay)
        for _ in range(3):
            host_module.causal_sensor_noise(replay)
        self.assertEqual(host_module.sensor_noise_report(replay), original_report)

    def test_training_and_inference_apply_identical_seeded_noise_to_causal_values(self):
        self.enable_noise(self.env)
        training_env = make_native_env(self.env.num_envs)
        self.enable_noise(training_env)
        training = TrainingHost(training_env, None)
        command = torch.tensor([[0.4, -0.2, 0.5]]).repeat(self.env.num_envs, 1)
        training_env.command_manager.get_term("base_velocity").vel_command_b.copy_(
            command
        )
        robot = training_env.scene["robot"].data
        frame = torch.cat(
            (
                robot.root_ang_vel_b,
                robot.projected_gravity_b,
                command,
                robot.joint_pos - robot.default_joint_pos,
                robot.joint_vel,
                training_env.action_manager.action,
                host_module.foot_contacts(training_env),
            ),
            dim=1,
        )
        native = dict(
            proprio=frame,
            policy=torch.cat((robot.root_lin_vel_b, frame), dim=1),
            terrain=torch.ones(self.env.num_envs, 264),
        )
        original = {key: value.clone() for key, value in native.items()}
        delivered = training.observations(
            native, torch.ones(self.env.num_envs, dtype=torch.bool)
        )
        controller = self.all_sensor_controller()
        inference = self.host(controller)
        self.act(inference)
        sensors = controller.calls[-1].sensors
        inference_frame = torch.cat(
            (
                sensors["base_ang_vel"].value,
                sensors["projected_gravity"].value,
                controller.calls[-1].command,
                sensors["joint_position_relative_default"].value,
                sensors["joint_velocity"].value,
                sensors["stock_previous_raw_action"].value,
                sensors["foot_contacts"].value,
            ),
            dim=1,
        )
        self.assertTrue(torch.equal(delivered["proprio"], inference_frame))
        for key, value in original.items():
            self.assertTrue(torch.equal(native[key], value))
        for key in ("policy", "terrain"):
            self.assertTrue(torch.equal(delivered[key], original[key]))
        for selection in (slice(6, 9), slice(33, 49)):
            self.assertTrue(
                torch.equal(delivered["proprio"][:, selection], frame[:, selection])
            )

    def test_shuffled_sensor_declarations_receive_correct_named_current_values(self):
        names = (
            "joint_velocity",
            "projected_gravity",
            "stock_previous_raw_action",
            "joint_position",
            "base_ang_vel",
            "joint_position_relative_default",
        )
        controller = self.controller(
            {
                name: SensorSpec(*host_module.NATIVE_SENSORS[name], max_age_s=0.02)
                for name in names
            }
        )
        self.robot.root_ang_vel_b[:] = torch.tensor([1.0, 2.0, 3.0])
        self.robot.projected_gravity_b[:] = torch.tensor([0.0, 0.6, -0.8])
        self.robot.joint_pos[:] = self.robot.default_joint_pos + 0.125
        self.robot.joint_vel[:] = torch.arange(12)
        host = self.host(controller)
        with (
            patch.object(
                self.env.observation_manager,
                "compute",
                side_effect=AssertionError("no privileged or all-group observations"),
            ),
            patch.object(
                self.env.observation_manager,
                "compute_group",
                side_effect=AssertionError(
                    "generic host must not demand the archived GRU frame"
                ),
            ),
        ):
            result = self.act(host)
        self.assertIsNone(result.raw_action)
        self.assertFalse(hasattr(controller, "hidden_state"))
        self.assertFalse(hasattr(host, "frame"))
        samples = controller.calls[0].sensors
        self.assertEqual(tuple(samples), names)
        expected = {
            "base_ang_vel": self.robot.root_ang_vel_b,
            "projected_gravity": self.robot.projected_gravity_b,
            "joint_position_relative_default": self.robot.joint_pos
            - self.robot.default_joint_pos,
            "joint_position": self.robot.joint_pos,
            "joint_velocity": self.robot.joint_vel,
            "stock_previous_raw_action": self.env.action_manager.action,
        }
        for name, sample in samples.items():
            torch.testing.assert_close(sample.value, expected[name], rtol=0, atol=0)
            self.assertEqual(
                (sample.units, sample.frame), host_module.NATIVE_SENSORS[name][1:]
            )
            self.assertEqual(sample.acquired_at_s, 0.0)
            self.assertTrue(sample.valid.all())
            self.assertFalse(sample.privileged)
        received = samples["joint_velocity"].value.clone()
        self.robot.joint_vel.zero_()
        torch.testing.assert_close(
            samples["joint_velocity"].value, received, rtol=0, atol=0
        )
        with self.assertRaises(TypeError):
            samples["extra"] = None
        torch.testing.assert_close(
            self.env.command_manager.get_term("base_velocity").vel_command_b,
            controller.calls[0].command,
            rtol=0,
            atol=0,
        )
        self.assertEqual(self.env.index, 0)

    def test_contact_flags_use_named_latest_net_vectors_and_strict_threshold(self):
        sensor = self.env.scene["contact_forces"]
        self.env.episode_length_buf.fill_(1)
        sensor.data.net_forces_w_history = torch.full((80, 3, 5, 3), 100.0)
        for name in FOOT_NAMES:
            sensor.data.net_forces_w.zero_()
            sensor.data.net_forces_w[:, sensor.body_names.index(name)] = torch.tensor(
                [1.2, 1.2, 0.0]
            )
            expected = torch.zeros(80, 4)
            expected[:, FOOT_NAMES.index(name)] = 1
            self.assertTrue(torch.equal(host_module.foot_contacts(self.env), expected))
        ids = [sensor.body_names.index(name) for name in FOOT_NAMES]
        sensor.data.net_forces_w[:, ids] = torch.tensor(
            [[1.5, 0, 0], [0, 0, -1.5001], [0, 1.4999, 0], [0, 0, 2]]
        )
        self.assertEqual(host_module.foot_contacts(self.env)[0].tolist(), [0, 1, 0, 1])

    def test_contact_flags_mask_stale_reset_rows_and_reach_controller_without_forces(
        self,
    ):
        sensor = self.env.scene["contact_forces"]
        sensor.data.net_forces_w.fill_(2)
        reset = torch.zeros(80, dtype=torch.bool)
        reset[:3] = True
        controller = self.controller(
            {"foot_contacts": SensorSpec((4,), "binary", "FR_FL_RR_RL", 0.02)}
        )
        host = self.host(controller)
        self.act(host)
        self.env.episode_length_buf.fill_(1)
        self.env.episode_length_buf[reset] = 0
        self.act(host, step=1, reset=reset)
        samples = controller.calls[1].sensors
        self.assertEqual(set(samples), {"foot_contacts"})
        self.assertTrue((samples["foot_contacts"].value[reset] == 0).all())
        self.assertTrue((samples["foot_contacts"].value[~reset] == 1).all())
        self.env.episode_length_buf.fill_(1)
        self.assertTrue((host_module.foot_contacts(self.env) == 1).all())

    def test_contact_sensor_rejects_ambiguous_missing_or_nonfinite_forces(self):
        sensor = self.env.scene["contact_forces"]
        names = list(sensor.body_names)
        for defect in ("missing", "duplicate", "nan", "shape"):
            sensor.body_names = list(names)
            sensor.data.net_forces_w = torch.zeros(80, 5, 3)
            if defect == "missing":
                sensor.body_names[1] = "not_a_foot"
            elif defect == "duplicate":
                sensor.body_names[0] = "FR_foot"
            elif defect == "nan":
                sensor.data.net_forces_w[:, 2, 0] = float("nan")
            else:
                sensor.data.net_forces_w = torch.zeros(80, 5, 6)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                host_module.foot_contacts(self.env)

    def test_subset_and_unknown_optional_sensor_do_not_expose_other_native_state(self):
        controller = self.controller(
            {
                "future_depth": SensorSpec((2, 2), "m", "camera", 0.1, required=False),
                "base_ang_vel": SensorSpec((3,), "rad/s", "body", 0.02),
            }
        )
        host = self.host(controller)
        self.act(host)
        self.assertEqual(
            set(controller.calls[0].sensors), {"future_depth", "base_ang_vel"}
        )
        self.assertIsNone(controller.calls[0].sensors["future_depth"])
        self.assertNotIn("joint_position", controller.calls[0].sensors)
        self.assertNotIn("base_lin_vel", controller.calls[0].sensors)
        self.assertNotIn("terrain", controller.calls[0].sensors)
        self.assertFalse(hasattr(controller.calls[0], "env"))

    def test_unsupported_required_privileged_and_changed_semantics_reject_before_inference(
        self,
    ):
        sensors = (
            {"base_lin_vel": SensorSpec((3,), "m/s", "body", 0.02)},
            {"terrain": SensorSpec((8,), "m", "world", 0.02, privileged=True)},
            {
                "optional_privileged": SensorSpec(
                    (3,), "m", "world", 0.02, required=False, privileged=True
                )
            },
            {"base_ang_vel": SensorSpec((3,), "rad/s", "body", 0.02, privileged=True)},
            {"base_ang_vel": SensorSpec((3,), "deg/s", "body", 0.02)},
            {"base_ang_vel": SensorSpec((3,), "rad/s", "world", 0.02)},
            {"base_ang_vel": SensorSpec((4,), "rad/s", "body", 0.02)},
        )
        for specification in sensors:
            controller = self.controller(specification)
            with (
                self.subTest(specification=specification),
                self.assertRaises(ValueError),
            ):
                self.host(controller)
            self.assertEqual(controller.calls, [])
            self.assertEqual(controller.resets, [])
            self.assertEqual(self.env.index, 0)

    def test_target_only_and_private_diagnostic_outputs_use_absolute_joint_targets(
        self,
    ):
        for private_raw in (False, True):
            controller = self.controller(private_raw=private_raw)
            host = self.host(controller)
            result = self.act(host)
            encoded = host.motor.encode(result)
            self.assertTrue(
                torch.equal(encoded, torch.zeros_like(self.robot.joint_pos))
            )
            if private_raw:
                self.assertEqual(result.raw_action.shape, (self.env.num_envs, 3))
                self.assertTrue((result.raw_action > 100).all())
            else:
                self.assertIsNone(result.raw_action)
            self.env.action_manager.action.copy_(encoded)
            self.env.action_manager.get_term("joint_pos").processed_actions.copy_(
                result.position_rad
            )
            self.robot.joint_pos_target.copy_(result.position_rad)
            host.motor.verify_delivery(
                torch.zeros(self.env.num_envs, dtype=torch.bool),
                torch.zeros(self.env.num_envs, dtype=torch.bool),
            )
            progress = host.motor.progress()
            self.assertEqual(progress["target_only_steps"], 1)
            self.assertEqual(progress["native_hint_steps"], 0)
            self.assertEqual(progress["native_verified_rows"], self.env.num_envs)
            self.assertFalse(progress["qualification_passed"])
            self.assertFalse(progress["exit_allowed"])
            self.assertEqual(self.env.index, 0)

    def test_source_stop_does_not_reset_controller_but_explicit_rows_do(self):
        controller = self.controller()
        host = self.host(controller)
        self.act(host)
        command = torch.zeros(self.env.num_envs, 3)
        host.act(
            command,
            time_s=0.02,
            reset_mask=torch.zeros(self.env.num_envs, dtype=torch.bool),
        )
        mask = torch.zeros(self.env.num_envs, dtype=torch.bool)
        mask[3] = True
        self.act(host, step=2, reset=mask)
        self.assertTrue(controller.resets[0].all())
        self.assertFalse(controller.resets[1].any())
        self.assertTrue(torch.equal(controller.resets[2], mask))
        self.assertEqual(len(controller.calls), 3)
        self.assertEqual(self.env.index, 0)

    def test_bad_cold_reset_and_nonzero_previous_reset_action_latch_before_inference(
        self,
    ):
        for defect in ("cold", "dtype", "previous_action"):
            controller = self.controller()
            host = self.host(controller)
            reset = torch.ones(self.env.num_envs, dtype=torch.bool)
            if defect == "cold":
                reset[0] = False
            elif defect == "dtype":
                reset = reset.to(torch.int64)
            else:
                self.env.action_manager.action[0, 0] = 1.0
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.act(host, reset=reset)
            self.assertTrue(host.failed)
            self.assertEqual(controller.calls, [])
            self.assertEqual(controller.resets, [])
            with self.assertRaisesRegex(RuntimeError, "faulted"):
                self.act(host)
            self.env.action_manager.action.zero_()

    def test_stale_provider_sample_and_nonfinite_required_sensor_latch_without_inference(
        self,
    ):
        for defect in ("stale", "nonfinite"):
            controller = self.controller(
                {"base_ang_vel": SensorSpec((3,), "rad/s", "body", 0.02)}
            )
            host = self.host(controller)
            original_sample = host_module.Sample

            def provider(value, acquired_at_s, *args, **kwargs):
                return original_sample(
                    value,
                    acquired_at_s - 0.04 if defect == "stale" else acquired_at_s,
                    *args,
                    **kwargs,
                )

            if defect == "nonfinite":
                self.robot.root_ang_vel_b[0, 0] = float("nan")
            with (
                self.subTest(defect=defect),
                patch.object(host_module, "Sample", side_effect=provider),
                self.assertRaises(ValueError),
            ):
                self.act(host)
            self.assertTrue(host.failed)
            self.assertEqual(controller.calls, [])
            self.assertEqual(controller.resets, [])
            self.assertEqual(self.env.index, 0)

    def test_skipped_decision_tick_latches_without_second_controller_step(self):
        controller = self.controller()
        host = self.host(controller)
        self.act(host)
        with self.assertRaisesRegex(ValueError, "cadence"):
            self.act(host, step=2)
        self.assertTrue(host.failed)
        self.assertEqual(len(controller.calls), 1)
        self.assertEqual(len(controller.resets), 1)

    def test_runtime_motor_and_controller_declaration_cannot_change_silently(self):
        controller = self.controller()
        host = self.host(controller)
        controller.spec = replace(controller.spec, name="changed")
        with self.assertRaisesRegex(ValueError, "specification changed"):
            self.act(host)
        self.assertEqual(controller.calls, [])
        self.assertTrue(host.failed)
        controller = self.controller()
        self.env.scene["robot"].actuators["legs"].stiffness.add_(1)
        with self.assertRaisesRegex(ValueError, "motor contract"):
            self.host(controller)
        self.assertEqual(controller.calls, [])

    def test_cpu_sensor_validation_rejects_privilege_before_native_scene_is_available(
        self,
    ):
        for name, spec in (
            ("base_ang_vel", SensorSpec((3,), "rad/s", "body", 0.02, privileged=True)),
            ("hidden_teacher", SensorSpec((4,), "unitless", "latent", 0.02)),
            ("joint_velocity", SensorSpec((12,), "m/s", "joint", 0.02)),
        ):
            controller = self.controller({name: spec})
            with self.subTest(name=name), self.assertRaises(ValueError):
                host_module.validate_native_controller_spec(controller.spec)
            self.assertEqual(controller.calls, [])
        allowed = self.controller(
            {"optional_depth": SensorSpec((2, 2), "m", "camera", 0.1, required=False)}
        )
        host_module.validate_native_controller_spec(allowed.spec)


if __name__ == "__main__":
    unittest.main()
