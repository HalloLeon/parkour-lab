"""CPU-only native target bridge tests; no simulator or policy qualification."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import parkour_lab.runtime.motor as motor_bridge
from parkour_lab.control import motor_contract
from parkour_lab.control.controller import JointTargets
from tests.support.native import make_native_env


class NativeJointTargetBridgeTests(unittest.TestCase):
    def fixture(self, *, preserve=False, count=3):
        env = make_native_env(count)
        binding, digest = motor_bridge._runtime_motor_binding(env)
        manifest = {
            "joint_names": binding["joint_names"],
            "configuration": {"default_position_rad": binding["default_position_rad"]},
            "period_s": 0.02,
            "actuator_profile": "native_motor_sha256:" + digest,
            "raw_action_meaning": motor_bridge.NATIVE_RAW_ACTION_MEANING,
        }
        contract = motor_contract.make_motor_contract(
            binding, manifest["actuator_profile"]
        )
        bridge = motor_bridge.NativeJointTargetBridge(
            env, contract, manifest, preserve_native_raw=preserve
        )
        return env, bridge, contract, manifest

    @staticmethod
    def output(bridge, raw=None):
        action = (
            torch.arange(12, dtype=bridge.default.dtype).repeat(bridge.num_envs, 1) / 4
        )
        target = action * 0.25 + bridge.default
        return JointTargets(bridge.joint_names, target, raw)

    @staticmethod
    def delivered(env, bridge, action):
        env.action_manager.action = action.clone()
        bridge.term.processed_actions = action * 0.25 + bridge.default
        env.scene["robot"].data.joint_pos_target = bridge.term.processed_actions.clone()

    @staticmethod
    def masks(bridge):
        return torch.zeros(bridge.num_envs, dtype=torch.bool), torch.zeros(
            bridge.num_envs, dtype=torch.bool
        )

    def assert_faulted(self, bridge):
        self.assertTrue(bridge.faulted)
        with self.assertRaisesRegex(RuntimeError, "faulted"):
            bridge.encode(self.output(bridge))
        with self.assertRaisesRegex(RuntimeError, "faulted"):
            bridge.verify_delivery(*self.masks(bridge))

    def test_exact_target_only_conversion_and_native_fixture_delivery(self):
        env, bridge, _, _ = self.fixture()
        output = self.output(bridge)
        action = bridge.encode(output)
        self.assertTrue(
            torch.equal(action * 0.25 + bridge.default, output.position_rad)
        )
        self.assertFalse(action.requires_grad)
        _, _, terminated, timed_out, _ = env.step(action)
        bridge.verify_delivery(terminated, timed_out)
        progress = bridge.progress()
        self.assertEqual(progress["encoded_steps"], 1)
        self.assertEqual(progress["target_only_steps"], 1)
        self.assertEqual(progress["native_hint_steps"], 0)
        self.assertEqual(progress["native_verified_rows"], 3)
        self.assertEqual(progress["excluded_terminal_rows"], 0)
        self.assertFalse(progress["pending_delivery"])
        self.assertFalse(progress["qualification_passed"])
        self.assertFalse(progress["exit_allowed"])
        json.dumps(progress, allow_nan=False)

    def test_default_ignores_any_private_diagnostic_vector(self):
        for private in (
            None,
            object(),
            torch.full((3, 17), float("nan")),
            torch.ones(1, dtype=torch.int64),
        ):
            with self.subTest(private=private):
                _, bridge, _, _ = self.fixture()
                output = self.output(bridge, private)
                encoded = bridge.encode(output)
                self.assertTrue(
                    torch.equal(encoded * 0.25 + bridge.default, output.position_rad)
                )
                self.assertEqual(bridge.progress()["target_only_steps"], 1)

    def test_trusted_hint_retains_native_bits_and_checks_absolute_targets(self):
        env, bridge, _, _ = self.fixture(preserve=True)
        torch.manual_seed(19)
        raw = torch.randn(3, 12)
        output = JointTargets(bridge.joint_names, raw * 0.25 + bridge.default, raw)
        action = bridge.encode(output)
        self.assertTrue(torch.equal(action, raw))
        self.assertNotEqual(action.data_ptr(), raw.data_ptr())
        self.delivered(env, bridge, action)
        bridge.verify_delivery(*self.masks(bridge))
        self.assertEqual(bridge.progress()["native_hint_steps"], 1)
        self.assertEqual(bridge.progress()["target_only_steps"], 0)

    def test_missing_trusted_hint_still_uses_authoritative_targets(self):
        _, bridge, _, _ = self.fixture(preserve=True)
        output = self.output(bridge)
        action = bridge.encode(output)
        self.assertTrue(
            torch.equal(action * 0.25 + bridge.default, output.position_rad)
        )
        self.assertEqual(bridge.progress()["target_only_steps"], 1)

    def test_raw_hint_requires_explicit_bool_and_exact_manifest_meaning(self):
        env, _, contract, manifest = self.fixture()
        for flag in (None, 1, "true"):
            with (
                self.subTest(flag=flag),
                self.assertRaisesRegex(ValueError, "explicit bool"),
            ):
                motor_bridge.NativeJointTargetBridge(
                    env, contract, manifest, preserve_native_raw=flag
                )
        for meaning in (
            "private latent",
            None,
            motor_bridge.NATIVE_RAW_ACTION_MEANING + " ",
        ):
            changed = {**manifest, "raw_action_meaning": meaning}
            with (
                self.subTest(meaning=meaning),
                self.assertRaisesRegex(ValueError, "exact stock"),
            ):
                motor_bridge.NativeJointTargetBridge(
                    env, contract, changed, preserve_native_raw=True
                )
            motor_bridge.NativeJointTargetBridge(env, contract, changed)

    def test_invalid_trusted_hint_faults_without_delivery(self):
        for hint in (
            torch.ones(3, 11),
            torch.ones(3, 12, dtype=torch.float64),
            torch.full((3, 12), float("nan")),
            torch.ones(3, 12),
        ):
            env, bridge, _, _ = self.fixture(preserve=True)
            output = JointTargets(bridge.joint_names, bridge.default.clone(), hint)
            with self.subTest(hint=hint), self.assertRaises(ValueError):
                bridge.encode(output)
            self.assertEqual(env.index, 0)
            self.assertEqual(bridge.progress()["encoded_steps"], 0)
            self.assert_faulted(bridge)

    def test_unrepresentable_target_rejects_instead_of_rounding_or_clamping(self):
        env, bridge, _, _ = self.fixture()
        target = bridge.default.clone()
        target[0, 0] = 1e-9
        self.assertFalse(
            torch.equal(
                ((target - bridge.default) / 0.25) * 0.25 + bridge.default, target
            )
        )
        with self.assertRaisesRegex(ValueError, "exactly reproduce"):
            bridge.encode(JointTargets(bridge.joint_names, target))
        self.assertEqual(env.index, 0)
        self.assert_faulted(bridge)

    def test_input_and_return_mutation_do_not_change_pending_comparison(self):
        env, bridge, _, _ = self.fixture(preserve=True)
        raw = torch.full((3, 12), 0.5, requires_grad=True)
        target = raw * 0.25 + bridge.default
        output = JointTargets(bridge.joint_names, target, raw)
        encoded = bridge.encode(output)
        with torch.no_grad():
            raw.fill_(1)
            target.fill_(99)
        original = encoded.clone()
        with torch.inference_mode():
            encoded.add_(0.1)
        self.delivered(env, bridge, original)
        bridge.verify_delivery(*self.masks(bridge))
        self.assertEqual(bridge.progress()["verified_delivery_steps"], 1)

    def test_mutated_delivered_action_is_detected_even_when_target_buffer_matches(self):
        env, bridge, _, _ = self.fixture()
        encoded = bridge.encode(self.output(bridge))
        self.delivered(env, bridge, encoded)
        env.action_manager.action[0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "delivered action"):
            bridge.verify_delivery(*self.masks(bridge))
        self.assert_faulted(bridge)

    def test_mutated_processed_target_is_detected(self):
        env, bridge, _, _ = self.fixture()
        encoded = bridge.encode(self.output(bridge))
        self.delivered(env, bridge, encoded)
        bridge.term.processed_actions[1, 4] += 0.1
        with self.assertRaisesRegex(ValueError, "processed joint positions"):
            bridge.verify_delivery(*self.masks(bridge))
        self.assert_faulted(bridge)

    def test_mutated_articulation_target_is_detected_and_return_count_survives(self):
        env, bridge, _, _ = self.fixture()
        encoded = bridge.encode(self.output(bridge))
        self.delivered(env, bridge, encoded)
        env.scene["robot"].data.joint_pos_target[0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "articulation joint targets"):
            bridge.verify_delivery(*self.masks(bridge))
        self.assertEqual(bridge.progress()["native_step_returns"], 1)
        self.assertEqual(bridge.progress()["verified_delivery_steps"], 0)
        self.assertTrue(bridge.pending_delivery)
        self.assert_faulted(bridge)

    def test_terminal_rows_excluded_and_native_fixture_reset_does_not_compare_replacement_state(
        self,
    ):
        env, bridge, _, _ = self.fixture()
        env.episode_length_buf[1] = env.max_episode_length - 1
        action = bridge.encode(self.output(bridge))
        _, _, terminated, timed_out, _ = env.step(action)
        self.assertTrue(timed_out[1])
        env.action_manager.action[1].fill_(float("nan"))
        bridge.term.processed_actions[1].fill_(float("nan"))
        env.scene["robot"].data.joint_pos_target[1].fill_(float("nan"))
        bridge.verify_delivery(terminated, timed_out)
        self.assertEqual(bridge.progress()["native_verified_rows"], 2)
        self.assertEqual(bridge.progress()["excluded_terminal_rows"], 1)

    def test_all_terminal_skips_native_action_and_target_access_entirely(self):
        env, bridge, _, _ = self.fixture()
        bridge.encode(self.output(bridge))

        class NoRead:
            @property
            def action(self):
                raise AssertionError("terminal action buffer inspected")

            @property
            def processed_actions(self):
                raise AssertionError("terminal target buffer inspected")

            @property
            def joint_pos_target(self):
                raise AssertionError("terminal articulation target buffer inspected")

        env.action_manager = bridge.term = bridge._robot_data = NoRead()
        bridge.verify_delivery(
            torch.tensor([True, False, True]), torch.tensor([False, True, True])
        )
        self.assertEqual(bridge.progress()["native_verified_rows"], 0)
        self.assertEqual(bridge.progress()["excluded_terminal_rows"], 3)
        self.assertEqual(bridge.progress()["verified_delivery_steps"], 1)
        self.assertEqual(bridge.progress()["native_step_returns"], 1)

    def test_pending_encoding_and_delivery_are_one_to_one(self):
        _, bridge, _, _ = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "requires one encoded"):
            bridge.verify_delivery(*self.masks(bridge))
        self.assert_faulted(bridge)
        _, bridge, _, _ = self.fixture()
        output = self.output(bridge)
        bridge.encode(output)
        with self.assertRaisesRegex(RuntimeError, "not been delivery-verified"):
            bridge.encode(output)
        self.assertEqual(bridge.progress()["encoded_steps"], 1)
        self.assert_faulted(bridge)

    def test_wrong_names_shape_dtype_nonfinite_and_non_targets_fault(self):
        for defect in ("names", "shape", "dtype", "nonfinite", "not_targets"):
            _, bridge, _, _ = self.fixture()
            output = self.output(bridge)
            if defect == "names":
                output = JointTargets(
                    tuple(reversed(output.joint_names)), output.position_rad
                )
            elif defect == "shape":
                output = JointTargets(output.joint_names, torch.ones(3, 11))
            elif defect == "dtype":
                output = JointTargets(output.joint_names, output.position_rad.double())
            elif defect == "nonfinite":
                output.position_rad[0, 0] = float("nan")
            else:
                output = SimpleNamespace(
                    joint_names=output.joint_names, position_rad=output.position_rad
                )
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                bridge.encode(output)
            self.assert_faulted(bridge)

    def test_invalid_done_masks_and_nonterminal_buffers_fault(self):
        for defect in (
            "mask_shape",
            "mask_dtype",
            "action_shape",
            "action_dtype",
            "nonfinite",
            "processed_shape",
        ):
            env, bridge, _, _ = self.fixture()
            action = bridge.encode(self.output(bridge))
            self.delivered(env, bridge, action)
            terminated, timed_out = self.masks(bridge)
            if defect == "mask_shape":
                terminated = torch.zeros(1, dtype=torch.bool)
            elif defect == "mask_dtype":
                timed_out = timed_out.float()
            elif defect == "action_shape":
                env.action_manager.action = torch.zeros(3, 11)
            elif defect == "action_dtype":
                env.action_manager.action = action.double()
            elif defect == "nonfinite":
                env.action_manager.action[0, 0] = float("nan")
            else:
                bridge.term.processed_actions = torch.zeros(3, 11)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                bridge.verify_delivery(terminated, timed_out)
            self.assertEqual(bridge.progress()["verified_delivery_steps"], 0)
            self.assert_faulted(bridge)

    def test_descriptor_and_motor_receipts_only_read_during_startup(self):
        env, bridge, _, _ = self.fixture()

        class NativeTerm:
            processed_actions = None

            @property
            def IO_descriptor(self):
                raise AssertionError("per-step descriptor read")

        term = NativeTerm()
        bridge.term = term
        with patch.object(
            motor_bridge,
            "_runtime_motor_binding",
            side_effect=AssertionError("per-step motor receipt read"),
        ):
            for _ in range(3):
                action = bridge.encode(self.output(bridge))
                self.delivered(env, bridge, action)
                bridge.verify_delivery(*self.masks(bridge))
        self.assertEqual(bridge.progress()["encoded_steps"], 3)
        self.assertEqual(bridge.progress()["native_verified_rows"], 9)

    def test_changed_native_motor_or_affine_contract_rejected_at_startup(self):
        for defect in ("gain", "scale", "clip", "joint_order", "default", "physics"):
            env, _, contract, manifest = self.fixture()
            if defect == "gain":
                env.scene["robot"].actuators["legs"].stiffness += 1
            elif defect == "scale":
                env.action_manager.get_term("joint_pos").cfg.scale = 0.5
            elif defect == "clip":
                env.action_manager.get_term("joint_pos").cfg.clip = {".*": [-1, 1]}
            elif defect == "joint_order":
                env.scene["robot"].joint_names = list(
                    reversed(env.scene["robot"].joint_names)
                )
            elif defect == "default":
                env.scene["robot"].data.default_joint_pos += 0.1
            else:
                env.physics_dt = 0.01
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                motor_bridge.NativeJointTargetBridge(env, contract, manifest)

    def test_binding_copy_matches_recorded_schema_hash_and_preserves_error_context(
        self,
    ):
        env, bridge, _, _ = self.fixture()
        binding, digest = motor_bridge._runtime_motor_binding(env)
        self.assertEqual(binding, bridge.binding)
        self.assertEqual(digest, motor_contract.nominal_binding_sha256(binding))
        env.scene["robot"].data.default_joint_vel[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "runtime default joint velocity"):
            motor_bridge._runtime_motor_binding(env)


if __name__ == "__main__":
    unittest.main()
