"""Saved-attempt integrity boundary; kinematic arithmetic is tested separately."""

import copy
import json
import hashlib
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from parkour_lab.control.command_tape import TapeBuilder, write_tape
from parkour_lab.control.proprioception import FOOT_NAMES
from parkour_lab.evaluation.runner import analyze_terrain_attempt
from parkour_lab.provenance import file_sha256


class TerrainAnalysisTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name)
        self.scorer = self.enterContext(
            patch(
                "parkour_lab.evaluation.terrain.score_terrain_attempt",
                return_value={
                    "kinematic_passed": True,
                    "complete": True,
                    "failures": [],
                },
            )
        )

    def fixture(self, steps=1500, *, terminal=False, truncated=False, requested=1500):
        command = np.zeros((steps, 1, 3), dtype=np.float32)
        command[100 : min(1350, steps), 0, 0] = 0.35
        ended = terminal or truncated
        self.motion = {
            "command": command,
            "terminated": np.zeros((steps, 1), dtype=bool),
            "truncated": np.zeros((steps, 1), dtype=bool),
        }
        self.motion["terminated"][-1] = terminal
        self.motion["truncated"][-1] = truncated
        np.savez(self.path / "motion.npz", **self.motion)
        self.tracking = {
            "reset_mask": np.zeros((steps, 1), dtype=bool),
            "root_com_velocity": np.zeros((steps, 1, 3)),
            "causal_sensor_noise": np.zeros((steps, 1, 30)),
            "initial_raw_sensors": np.zeros((1, 30)),
            "initial_noisy_sensors": np.zeros((1, 30)),
            "foot_contacts": np.zeros((steps, 1, 4)),
            "foot_net_forces_w": np.zeros((steps, 1, 4, 3)),
        }
        self.tracking["reset_mask"][0] = True
        np.savez(self.path / "tracking.npz", **self.tracking)
        np.savez(self.path / "world.npz", vertices=np.zeros((3, 3)))
        (self.path / "manifest.json").write_text("{}")
        builder = TapeBuilder({})
        for i, row in enumerate(command[:, 0]):
            builder.append(i, row.astype(float).tolist())
        write_tape(self.path / "commands.json", builder.finish(completed=not ended))
        self.report = {
            "config": {"task": {"terrain": "connected", "num_envs": 1}},
            "connected_world": {
                "first_attempt_only": True,
                "foot_names": list(FOOT_NAMES),
                "source": {},
            },
            "status": "FIRST_ATTEMPT_ENDED_NOT_QUALIFIED"
            if ended
            else "EVALUATION_COMPLETE_NOT_QUALIFIED",
            "policy_unchanged": True,
            "cleanup": {"environment": "complete", "application": "complete"},
            "control_steps": steps,
            "requested_control_steps": requested,
            "environment_transitions": steps,
            "terminated_rows": int(terminal),
            "timeout_rows": int(truncated and not terminal),
            "actor_sha256": "a" * 64,
            "package_sources": {"module.py": "b" * 64},
            "manifest_sha256": file_sha256(self.path / "manifest.json"),
            "evidence_sha256": {
                name: file_sha256(self.path / name)
                for name in (
                    "commands.json",
                    "motion.npz",
                    "tracking.npz",
                    "world.npz",
                )
            },
            "motor_delivery": {
                "encoded_steps": steps,
                "verified_delivery_steps": steps,
                "native_step_returns": steps,
                "faulted": False,
                "pending_delivery": False,
                "native_verified_rows": steps - int(ended),
                "excluded_terminal_rows": int(ended),
            },
        }
        self.save_report()

    def save_report(self):
        (self.path / "report.json").write_text(json.dumps(self.report))

    def save_motion(self):
        np.savez(self.path / "motion.npz", **self.motion)
        self.report["evidence_sha256"]["motion.npz"] = file_sha256(
            self.path / "motion.npz"
        )
        self.save_report()

    def bound_fixture(self, index=50):
        from parkour_lab.config import TaskConfig
        from parkour_lab.environments.randomization import (
            TaskRandomization,
            terrain_development_attempt,
        )

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name)
        self.fixture(8, terminal=True)
        self.scorer.return_value = dict(
            kinematic_passed=False, complete=False, failures=["native termination"]
        )
        task = TaskConfig(
            terrain="connected",
            num_envs=1,
            episode_length_s=31,
            terrain_group="ramp/down/10deg",
            terrain_attempt_index=index,
        )
        assignment = terrain_development_attempt(task.terrain_group, index)
        heading = task.world["world_yaw"] - np.pi / 2
        sampler = TaskRandomization(task, evaluation=True, nominal_heading=heading)
        sampler.dynamics()
        start = sampler.sample_start(None)
        self.manifest = sampler.manifest()
        noise = sampler.uniform(None, 8 * 30, "observation-noise", -1, 1).reshape(
            8, 1, 30
        )
        noise = (noise * ([0.2] * 3 + [0.05] * 3 + [0.01] * 12 + [1.5] * 12)).astype(
            np.float32
        )
        self.tracking["causal_sensor_noise"] = noise
        self.tracking["initial_raw_sensors"] = np.zeros((1, 30), dtype=np.float32)
        self.tracking["initial_raw_sensors"][0, 6:18] = start["joint_offset"][0]
        self.tracking["initial_noisy_sensors"] = (
            self.tracking["initial_raw_sensors"] + noise[0]
        )
        position = np.array([[3.0, 7.0, 1.1]])
        position[0, :2] += start["start_xy"][0]
        quaternion = np.array(
            [
                [
                    np.cos(start["start_yaw"][0] / 2),
                    0,
                    0,
                    np.sin(start["start_yaw"][0] / 2),
                ]
            ]
        )
        self.motion.update(
            initial_position_w=position,
            initial_quaternion_w=quaternion,
            initial_linear_velocity_b=np.zeros((1, 3)),
            initial_angular_velocity_b=np.zeros((1, 3)),
        )
        self.report["config"]["task"] = asdict(task)
        self.report["terrain_assignment"] = assignment
        self.report["task_realization"] = {
            "randomization": copy.deepcopy(self.manifest)
        }
        draw = self.manifest["attempts"][0]["dynamics"]
        names = [f"joint{i}" for i in range(12)]
        masses = np.array([[6.0, 1.0]], dtype=np.float32)
        masses[0, 0] += np.float32(draw["added_mass"])
        materials = np.array(
            [[[draw["static_friction"], draw["dynamic_friction"], 0.0]]],
            dtype=np.float32,
        )
        physics = dict(
            masses=masses.tolist(),
            materials=materials.tolist(),
            inertias=np.eye(3).flatten().reshape(1, 1, 9).repeat(2, axis=1).tolist(),
            coms=np.zeros((1, 2, 7)).tolist(),
            joint_limits=np.tile([[-3.0, 3.0]], (1, 12, 1)).tolist(),
        )
        scales = {
            parameter: np.asarray(draw[key], dtype=np.float32)[None, :]
            for parameter, key in (
                ("stiffness", "kp_scale"),
                ("damping", "kd_scale"),
                ("motor_strength", "motor_strength"),
            )
        }
        motor = dict(
            joint_names=names,
            stiffness=(25 * scales["stiffness"]).tolist(),
            damping=(0.5 * scales["damping"]).tolist(),
            motor_strength=scales["motor_strength"].tolist(),
            effort_limit=[[23.5] * 12],
            velocity_limit=[[30.0] * 12],
        )
        self.report["task_realization"].update(
            body_names=["base", "leg"],
            joint_names=names,
            default_masses=[[6.0, 1.0]],
            requested_physics=copy.deepcopy(physics),
            physical_readback=physics,
            robot_material_combine_modes={},
            motors={"legs": motor},
        )
        self.report["motor_verification"] = dict(
            runtime_num_envs=1,
            realized_randomization={
                "legs": dict(
                    nominal_gains=dict(stiffness=[25.0] * 12, damping=[0.5] * 12),
                    scales={key: value.tolist() for key, value in scales.items()},
                )
            },
        )
        self.report["initial_state"] = dict(
            root_position=position.tolist(),
            root_orientation=quaternion.tolist(),
            root_velocity=[[0.0] * 6],
            joint_velocity=[[0.0] * 12],
        )
        self.report["sensor_noise"] = dict(
            generated_frames=8, draw_sha256=hashlib.sha256(noise.tobytes()).hexdigest()
        )
        vertices, faces = np.zeros((3, 3)), np.array([[0, 1, 2]])
        np.savez(self.path / "world.npz", vertices=vertices, faces=faces)
        transform = np.eye(4)
        yaw = task.world["world_yaw"]
        transform[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
        source = dict(
            coarse_seed=task.world["coarse_seed"],
            fine_seed=task.world["fine_seed"],
            size_m=[24.0, 24.0],
            resolution_limit_m=0.02,
            vertices_float32_sha256=hashlib.sha256(
                vertices.astype("<f4").tobytes()
            ).hexdigest(),
            faces_int64_sha256=hashlib.sha256(
                faces.astype("<i8").tobytes()
            ).hexdigest(),
            trial=dict(
                family="ramp",
                orientation="down",
                heading_world_rad=heading,
                local_to_world_column_transform=transform.tolist(),
            ),
        )
        native = dict(
            requested_world=task.world,
            start_heading_rad=heading,
            start_position_m=[3.0, 7.0, 0.7],
            vertices_float32_sha256=source["vertices_float32_sha256"],
            faces_int64_sha256=source["faces_int64_sha256"],
            local_to_world_column_transform=transform.tolist(),
            material=dict(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        )
        self.report["connected_world"].update(source=source, native=native)
        self.save_bound()

    def save_bound(self):
        (self.path / "manifest.json").write_text(json.dumps(self.manifest))
        np.savez(self.path / "motion.npz", **self.motion)
        np.savez(self.path / "tracking.npz", **self.tracking)
        self.report["manifest_sha256"] = file_sha256(self.path / "manifest.json")
        for name in self.report["evidence_sha256"]:
            self.report["evidence_sha256"][name] = file_sha256(self.path / name)
        self.save_report()

    def test_bound_development_identity_verifies_and_keeps_first_failure(self):
        for index in (49, 50):
            with self.subTest(index=index):
                self.bound_fixture(index)
                before = {p.name: p.read_bytes() for p in self.path.iterdir()}
                result = analyze_terrain_attempt(self.path)
                self.assertTrue(result["terrain_development"]["binding_verified"])
                self.assertEqual(result["terrain_development"]["attempt_index"], index)
                self.assertEqual(
                    result["terrain_development"]["stratum"],
                    "nominal" if index < 50 else "randomized",
                )
                self.assertFalse(result["diagnostic_passed"])
                self.assertFalse(result["qualified"])
                self.assertEqual(
                    before, {p.name: p.read_bytes() for p in self.path.iterdir()}
                )

    def test_bound_identity_rejects_changed_draws_or_receipts_even_with_valid_file_hashes(
        self,
    ):
        mutations = (
            lambda: self.manifest["attempts"][0].update(attempt_index=0),
            lambda: self.manifest["attempts"][0]["streams"]["start"].update(
                sha256="0" * 64
            ),
            lambda: self.manifest["attempts"][0]["start"].update(start_yaw=0),
            lambda: self.report["task_realization"]["randomization"]["attempts"][0][
                "dynamics"
            ].update(added_mass=0),
            lambda: self.report["terrain_assignment"].update(attempt_index=0),
            lambda: self.report["connected_world"]["native"]["requested_world"].update(
                coarse_seed=1
            ),
            lambda: self.report["connected_world"]["source"].update(fine_seed=1),
            lambda: self.report["connected_world"]["native"].update(
                faces_int64_sha256="0" * 64
            ),
            lambda: self.report["connected_world"]["native"]["material"].update(
                static_friction=0.8
            ),
            lambda: self.report["connected_world"]["native"].update(
                start_heading_rad=0
            ),
            lambda: self.tracking["causal_sensor_noise"].__setitem__((7, 0, 0), 0),
            lambda: self.motion["initial_position_w"].__setitem__((0, 0), 0),
            lambda: self.motion["initial_quaternion_w"].__setitem__((0, 0), 0),
            lambda: self.tracking["initial_raw_sensors"].__setitem__((0, 6), 0),
            lambda: self.report["sensor_noise"].update(generated_frames=7),
            lambda: self.report["task_realization"].update(motors={}),
            lambda: self.report["task_realization"]["physical_readback"]["masses"][
                0
            ].__setitem__(0, 999),
            lambda: self.report["task_realization"]["requested_physics"]["masses"][
                0
            ].__setitem__(0, 999),
            lambda: self.report["task_realization"]["physical_readback"]["materials"][
                0
            ][0].__setitem__(0, 99),
            lambda: self.report["task_realization"]["motors"]["legs"]["stiffness"][
                0
            ].__setitem__(0, 99),
            lambda: self.report["motor_verification"]["realized_randomization"]["legs"][
                "scales"
            ]["motor_strength"][0].__setitem__(0, 1),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                self.bound_fixture()
                mutate()
                self.save_bound()
                with self.assertRaises(ValueError):
                    analyze_terrain_attempt(self.path)

    def test_bound_manifest_accepts_portable_float_roundoff_but_rejects_changed_start(
        self,
    ):
        self.bound_fixture()
        for manifest in (
            self.manifest,
            self.report["task_realization"]["randomization"],
        ):
            start = manifest["attempts"][0]["start"]
            for key in ("start_xy", "start_xy_local", "joint_offset"):
                start[key] = [
                    float(np.nextafter(value, np.inf)) for value in start[key]
                ]
        self.save_bound()
        result = analyze_terrain_attempt(self.path)
        self.assertTrue(result["terrain_development"]["binding_verified"])
        self.assertFalse(result["diagnostic_passed"])
        self.manifest["attempts"][0]["start"]["joint_offset"][0] += 1e-6
        self.save_bound()
        with self.assertRaisesRegex(ValueError, "first start or dynamics changed"):
            analyze_terrain_attempt(self.path)

    def test_complete_verified_attempt_remains_nonqualifying_and_read_only(self):
        self.fixture()
        before = {p.name: p.read_bytes() for p in self.path.iterdir()}
        result = analyze_terrain_attempt(self.path)
        self.assertTrue(result["diagnostic_passed"])
        self.assertTrue(result["evidence_verified"])
        self.assertFalse(result["qualified"])
        self.assertFalse(result["qualification_eligible"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.path.iterdir()})
        self.scorer.assert_called_once()

    def test_short_first_ending_preserves_failure_and_cannot_become_30_second_trial(
        self,
    ):
        self.fixture(556, terminal=True, requested=750)
        self.scorer.return_value = {
            "kinematic_passed": False,
            "complete": False,
            "failures": ["native termination"],
        }
        result = analyze_terrain_attempt(self.path / "report.json")
        self.assertFalse(result["diagnostic_passed"])
        self.assertFalse(result["prescribed_duration"])
        self.assertEqual(result["requested_control_steps"], 750)
        self.assertIn("native termination", result["failures"])
        self.assertTrue(result["evidence_verified"])

    def test_missing_changed_archive_and_identity_receipts_reject(self):
        self.fixture()
        original = copy.deepcopy(self.report)
        for key, value in (
            ("policy_unchanged", False),
            ("cleanup", {}),
            ("actor_sha256", ""),
            ("package_sources", {}),
            ("status", "ERROR"),
            ("requested_control_steps", 1499),
            ("control_steps", 1499),
            ("manifest_sha256", "0" * 64),
        ):
            with self.subTest(key=key):
                self.report = copy.deepcopy(original)
                self.report[key] = value
                self.save_report()
                with self.assertRaises(ValueError):
                    analyze_terrain_attempt(self.path)
        self.report = original
        for name in original["evidence_sha256"]:
            with self.subTest(archive=name):
                good = self.report["evidence_sha256"][name]
                self.report["evidence_sha256"][name] = "0" * 64
                self.save_report()
                with self.assertRaisesRegex(ValueError, "changed evidence"):
                    analyze_terrain_attempt(self.path)
                self.report["evidence_sha256"][name] = good
        self.save_report()
        (self.path / "world.npz").unlink()
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            analyze_terrain_attempt(self.path)

    def test_motor_fault_or_incomplete_delivery_rejects(self):
        self.fixture()
        original = copy.deepcopy(self.report["motor_delivery"])
        for key, value in (
            ("faulted", True),
            ("pending_delivery", True),
            ("encoded_steps", 1499),
            ("verified_delivery_steps", 1499),
            ("native_step_returns", 1499),
            ("native_verified_rows", 1499),
            ("excluded_terminal_rows", 1),
        ):
            with self.subTest(key=key):
                self.report["motor_delivery"] = {**original, key: value}
                self.save_report()
                with self.assertRaisesRegex(ValueError, "motor delivery"):
                    analyze_terrain_attempt(self.path)

    def test_consistent_hashes_do_not_allow_reset_assistance_or_tape_mismatch(self):
        self.fixture()
        self.tracking["reset_mask"][600] = True
        np.savez(self.path / "tracking.npz", **self.tracking)
        self.report["evidence_sha256"]["tracking.npz"] = file_sha256(
            self.path / "tracking.npz"
        )
        self.save_report()
        with self.assertRaisesRegex(ValueError, "replacement attempt"):
            analyze_terrain_attempt(self.path)
        self.motion["command"][100, 0, 0] = 0.5
        self.save_motion()
        with self.assertRaisesRegex(ValueError, "tape disagree"):
            analyze_terrain_attempt(self.path)
        self.motion["command"][100, 0, 0] = np.nan
        self.save_motion()
        with self.assertRaisesRegex(ValueError, "nonfinite motion"):
            analyze_terrain_attempt(self.path)

    def test_first_native_ending_is_not_replaced_by_later_samples(self):
        self.fixture()
        self.motion["terminated"][500] = True
        self.save_motion()
        with self.assertRaisesRegex(ValueError, "continues after its first ending"):
            analyze_terrain_attempt(self.path)

    def test_hash_consistent_missing_sensor_arrays_reject(self):
        self.fixture()
        np.savez(self.path / "tracking.npz", reset_mask=self.tracking["reset_mask"])
        self.report["evidence_sha256"]["tracking.npz"] = file_sha256(
            self.path / "tracking.npz"
        )
        self.save_report()
        with self.assertRaisesRegex(ValueError, "incomplete tracking/input"):
            analyze_terrain_attempt(self.path)
