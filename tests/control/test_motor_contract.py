"""CPU tests for nominal motor identity and bounded runtime variation."""

import copy
import unittest

from parkour_lab.control import motor_contract as motor
from tests.support.native import motor_binding, rebatch


class MotorContractTests(unittest.TestCase):
    def setUp(self):
        self.binding = motor_binding([f"j{i}" for i in range(12)], [0.0] * 12)
        self.limits = [[-1.0, 1.0] for _ in range(12)]
        self.manifest = {
            "actuator_profile": "native_motor_sha256:"
            + motor.nominal_binding_sha256(self.binding),
            "joint_names": self.binding["joint_names"],
            "configuration": {
                "default_position_rad": [0.0] * 12,
                "action_clip": {
                    "version": "joint_limits_v1",
                    "target_limits_rad": self.limits,
                },
            },
            "period_s": 0.02,
        }
        self.contract = motor.make_motor_contract(
            self.binding,
            self.manifest["actuator_profile"],
            target_limits_rad=self.limits,
        )

    def test_nominal_identity_is_preserved_across_runtime_batches(self):
        for count in (80, 1280):
            report = motor.verify_runtime_motor(
                self.contract, self.manifest, rebatch(self.binding, count)
            )
            self.assertEqual(report["source_num_envs"], 4)
            self.assertEqual(report["runtime_num_envs"], count)
            self.assertNotEqual(
                report["source_motor_binding_sha256"],
                report["runtime_motor_binding_sha256"],
            )
        self.assertEqual(
            self.contract,
            motor.make_motor_contract(
                self.binding,
                self.manifest["actuator_profile"],
                target_limits_rad=self.limits,
            ),
        )

    def test_candidate_timing_has_distinct_identity_and_cross_rate_rejects(self):
        candidate = copy.deepcopy(self.binding)
        candidate.update(physics_dt_s=0.0025, decimation=8)
        profile = "native_motor_sha256:" + motor.nominal_binding_sha256(candidate)
        manifest = dict(self.manifest, actuator_profile=profile)
        contract = motor.make_motor_contract(
            candidate, profile, target_limits_rad=self.limits
        )
        self.assertNotEqual(profile, self.manifest["actuator_profile"])
        report = motor.verify_runtime_motor(contract, manifest, rebatch(candidate, 80))
        self.assertEqual(report["runtime_num_envs"], 80)
        for frozen, description, actual in (
            (self.contract, self.manifest, candidate),
            (contract, manifest, self.binding),
        ):
            with self.assertRaisesRegex(ValueError, "differs"):
                motor.verify_runtime_motor(frozen, description, actual)

    def test_only_exact_legal_timing_pairs_are_accepted(self):
        for dt, decimation in (
            (0.005, 8),
            (0.0025, 4),
            (0.01, 2),
            (0.005, 4.0),
            (0.0025, 8.0),
            (0.005, True),
            ("0.0025", 8),
        ):
            binding = copy.deepcopy(self.binding)
            binding.update(physics_dt_s=dt, decimation=decimation)
            with (
                self.subTest(dt=dt, decimation=decimation),
                self.assertRaises(ValueError),
            ):
                motor.nominal_binding_sha256(binding)

    def test_nonuniform_malformed_and_type_changed_rows_reject(self):
        for mutation in (
            lambda b: b["actuators"]["legs"]["resolved_parameters"]["stiffness"][
                1
            ].__setitem__(0, 1.1),
            lambda b: b["actuators"]["legs"]["resolved_parameters"]["stiffness"][
                1
            ].__setitem__(0, 1),
            lambda b: b["actuators"]["legs"]["resolved_parameters"]["stiffness"][
                1
            ].__setitem__(0, True),
            lambda b: b["actuators"]["legs"]["resolved_parameters"]["stiffness"][
                1
            ].__setitem__(0, float("nan")),
            lambda b: b["actuators"]["legs"]["resolved_parameters"]["stiffness"].pop(),
            lambda b: b["actuators"]["legs"]["resolved_parameters"].pop("friction"),
            lambda b: b["actuators"]["legs"]["joint_names"].__setitem__(0, "j1"),
            lambda b: b.update(unreviewed=True),
        ):
            bad = copy.deepcopy(self.binding)
            mutation(bad)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                motor.verify_runtime_motor(self.contract, self.manifest, bad)

    def test_every_other_physical_field_is_preserved(self):
        mutations = [
            lambda b: b["joint_names"].reverse(),
            lambda b: b["default_position_rad"].__setitem__(0, 0.1),
            lambda b: b["action"].update(scale=0.5),
            lambda b: b["action"].update(clip=1.0),
            lambda b: b["actuators"]["legs"]["configuration"].update(extra_setting=1),
            lambda b: b.update(step_dt_s=0.04),
            lambda b: b.update(physics_dt_s=0.01),
            lambda b: b.update(decimation=2),
        ]
        for name in motor.PARAMETERS:
            mutations.append(
                lambda b, n=name: b["actuators"]["legs"]["resolved_parameters"].update(
                    {n: [[2.0] * 12] * 4}
                )
            )
        for mutate in mutations:
            bad = copy.deepcopy(self.binding)
            mutate(bad)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                motor.verify_runtime_motor(self.contract, self.manifest, bad)

    def test_nominal_receipt_must_match_archived_identity(self):
        for mutate in (
            lambda c: c.update(source_num_envs=True),
            lambda c: c.update(version="unknown"),
            lambda c: c.update(extra="unreviewed"),
            lambda c: c["binding"]["action"].update(scale=0.5),
            lambda c: c["binding"]["actuators"]["legs"]["resolved_parameters"][
                "damping"
            ][0].__setitem__(0, 2.0),
        ):
            bad = copy.deepcopy(self.contract)
            mutate(bad)
            with self.assertRaises(ValueError):
                motor.validate_motor_contract(bad, self.manifest)
        with self.assertRaises(ValueError):
            motor.make_motor_contract(
                self.binding,
                "native_motor_sha256:" + "0" * 64,
                target_limits_rad=self.limits,
            )

    def test_joint_limits_are_required_validated_and_owned(self):
        profile = self.manifest["actuator_profile"]
        with self.assertRaises(TypeError):
            motor.make_motor_contract(self.binding, profile)
        for limits in (
            None,
            [],
            [[0.0, 0.0]] * 12,
            [[0.1, 1.0]] * 12,
            [[-1.0, float("inf")]] * 12,
        ):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                motor.make_motor_contract(
                    self.binding, profile, target_limits_rad=limits
                )
        limits = copy.deepcopy(self.limits)
        contract = motor.make_motor_contract(
            self.binding, profile, target_limits_rad=limits
        )
        limits[0][0] = -2.0
        self.assertEqual(contract["target_limits_rad"], self.limits)
        for mutate in (
            lambda c: c.pop("target_limits_rad"),
            lambda c: c.update(version="unsupported"),
            lambda c: c.update(target_limits_rad=None),
        ):
            bad = copy.deepcopy(contract)
            mutate(bad)
            with self.assertRaises(ValueError):
                motor.validate_motor_contract(bad, self.manifest)
        for clip in (
            None,
            {"version": "joint_limits_v1", "target_limits_rad": [[-2.0, 2.0]] * 12},
        ):
            manifest = copy.deepcopy(self.manifest)
            manifest["configuration"]["action_clip"] = clip
            with self.assertRaisesRegex(ValueError, "boundary differs"):
                motor.validate_motor_contract(contract, manifest)

    def test_declared_variation_is_verified_but_not_actor_identity(self):
        binding = copy.deepcopy(self.binding)
        actuator = binding["actuators"]["legs"]
        actuator["randomization"] = dict(
            ranges=copy.deepcopy(motor.RANDOMIZATION_RANGES),
            nominal_gains={key: [1.0] * 12 for key in ("stiffness", "damping")},
            scales={
                key: [[motor._float32(0.95 + row * 0.03)] * 12 for row in range(4)]
                for key in motor.RANDOMIZATION_RANGES
            },
        )
        for key in ("stiffness", "damping"):
            actuator["resolved_parameters"][key] = copy.deepcopy(
                actuator["randomization"]["scales"][key]
            )
        profile = "native_motor_sha256:" + motor.nominal_binding_sha256(binding)
        manifest = dict(self.manifest, actuator_profile=profile)
        contract = motor.make_motor_contract(
            binding, profile, target_limits_rad=self.limits
        )
        motor.verify_runtime_motor(contract, manifest, binding)
        nominal = copy.deepcopy(binding)
        for key in motor.RANDOMIZATION_RANGES:
            nominal["actuators"]["legs"]["randomization"]["scales"][key] = [
                [1.0] * 12
            ] * 4
            if key != "motor_strength":
                nominal["actuators"]["legs"]["resolved_parameters"][key] = [
                    [1.0] * 12
                ] * 4
        self.assertEqual(
            motor.nominal_binding_sha256(nominal), motor.nominal_binding_sha256(binding)
        )
        motor.verify_runtime_motor(contract, manifest, nominal)
        for mutate in (
            lambda a: a["randomization"]["scales"]["motor_strength"][0].__setitem__(
                0, 1.2
            ),
            lambda a: a["resolved_parameters"]["stiffness"][0].__setitem__(0, 1.0),
            lambda a: a["randomization"]["ranges"].update(stiffness=[0.8, 1.2]),
            lambda a: a["resolved_parameters"].update(effort_limit=[[2.0] * 12] * 4),
        ):
            bad = copy.deepcopy(binding)
            mutate(bad["actuators"]["legs"])
            with self.assertRaises(ValueError):
                motor.verify_runtime_motor(contract, manifest, bad)


if __name__ == "__main__":
    unittest.main()
