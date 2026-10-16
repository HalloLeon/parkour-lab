"""Analytic trajectories test scoring arithmetic, not simulated locomotion."""

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from parkour_lab.control.command_tape import TapeBuilder
from parkour_lab.environments.randomization import TaskRandomization
from parkour_lab.evaluation.flat import (
    PROFILES,
    STEPS,
    aggregate_development_bank,
    analyze_development_bank,
    profile_commands,
    score_command_phase,
    score_development_profile,
    score_flat_profile,
    summarize_development_profile,
)


def ideal_trace(profile, count=2):
    command = np.asarray(profile_commands(profile))
    heading = np.cumsum(command[:, 2]) * 0.02
    velocity_w = np.stack(
        (
            command[:, 0] * np.cos(heading) - command[:, 1] * np.sin(heading),
            command[:, 0] * np.sin(heading) + command[:, 1] * np.cos(heading),
            np.zeros(STEPS),
        ),
        axis=-1,
    )
    position = np.cumsum(velocity_w, axis=0) * 0.02
    quat = np.stack(
        (np.cos(heading / 2), np.zeros(STEPS), np.zeros(STEPS), np.sin(heading / 2)),
        axis=-1,
    )
    zeros = np.zeros((STEPS, 3))
    lin = np.column_stack((command[:, :2], np.zeros(STEPS)))
    ang = zeros.copy()
    ang[:, 2] = command[:, 2]
    trace = {
        key: np.repeat(value[:, None], count, axis=1)
        for key, value in {
            "command": command,
            "position_w": position,
            "quaternion_w": quat,
            "linear_velocity_b": lin,
            "angular_velocity_b": ang,
            "angular_velocity_w": ang,
            "base_contact_force_w": zeros,
        }.items()
    }
    trace.update(
        body_position_w=np.repeat(trace["position_w"][:, :, None], 5, axis=2),
        initial_position_w=np.zeros((count, 3)),
        initial_quaternion_w=np.tile([1.0, 0.0, 0.0, 0.0], (count, 1)),
        env_origins=np.zeros((count, 3)),
        terminated=np.zeros((STEPS, count), dtype=bool),
        truncated=np.zeros((STEPS, count), dtype=bool),
    )
    return trace


class CommandPhaseTests(unittest.TestCase):
    def phase(self, command=(0, 0, 0), length=150, **changes):
        values = dict(
            velocity=np.tile(command[:2], (length, 1)).astype(float),
            body_yaw=np.full(length, command[2], dtype=float),
            world_yaw=np.full(length, command[2], dtype=float),
            position=np.zeros((length + 1, 2)),
            heading=np.zeros(length + 1),
        )
        values.update(changes)
        return score_command_phase(command, **values)

    def test_deadline_block_later_blocks_and_last_partial_are_required(self):
        for offset, acquisition in ((30, True), (50, False), (110, False)):
            velocity = np.zeros((111, 2))
            velocity[offset, 0] = 3
            result = self.phase(length=111, velocity=velocity)
            self.assertFalse(result["kinematic_passed"])
            self.assertEqual(result["acquisition_failed"], acquisition)
            self.assertEqual(result["blocks"][0]["start_s"], 0.6)
            self.assertEqual(result["blocks"][0]["duration_s"], 0.4)
            self.assertEqual(result["blocks"][-1]["duration_s"], 0.02)
        self.assertTrue(self.phase()["kinematic_passed"])

    def test_stationary_speed_and_planar_euclidean_error_are_not_component_rmse(self):
        velocity = np.tile([0.079, 0.079], (150, 1))
        result = self.phase(command=(0.2, 0.2, 0), velocity=velocity + [0.2, 0.2])
        self.assertFalse(result["kinematic_passed"])
        self.assertTrue(
            self.phase(velocity=np.tile([0.079, 0], (150, 1)))["kinematic_passed"]
        )
        self.assertFalse(
            self.phase(velocity=np.tile([0.081, 0], (150, 1)))["kinematic_passed"]
        )

    def test_world_up_yaw_and_wrong_sign_cannot_be_hidden_by_body_rate(self):
        result = self.phase(command=(0, 0, 0.5), world_yaw=np.zeros(150))
        self.assertFalse(result["kinematic_passed"])
        wrong = np.full(150, 0.5)
        wrong[50:56] = -0.5
        result = self.phase(command=(0, 0, 0.5), world_yaw=wrong)
        self.assertEqual(result["wrong_sign_fraction"], 0.06)
        self.assertIn("missing or excessive wrong-sign yaw", result["failures"])

    def test_drift_measures_excursion_not_just_net_end_displacement(self):
        position = np.zeros((201, 2))
        position[90:110, 0] = 0.101
        result = self.phase(length=200, position=position)
        self.assertAlmostEqual(result["maximum_settled_drift_2s_m"], 0.101)
        self.assertFalse(result["kinematic_passed"])
        self.assertLess(result["onset_excursion_m"], 0.15)

    def test_stopping_distance_and_heading_use_the_pre_command_anchor(self):
        position = np.zeros((151, 2))
        position[1:, 1] = 0.151
        self.assertFalse(self.phase(position=position)["kinematic_passed"])
        heading = np.full(151, 0.151)
        heading[0] = 0
        self.assertFalse(self.phase(heading=heading)["kinematic_passed"])

    def test_short_nonfinite_or_missing_anchor_rejects_evidence(self):
        for changes in (
            {"length": 49},
            {"position": np.zeros((150, 2))},
            {"body_yaw": np.full(150, np.nan)},
        ):
            with self.subTest(changes=changes.keys()), self.assertRaises(ValueError):
                self.phase(**changes)


class FlatProfileTests(unittest.TestCase):
    def test_every_profile_has_correct_phases_and_ideal_first_attempt_passes(self):
        self.assertEqual(len(PROFILES), 10)
        for profile in PROFILES:
            with self.subTest(profile=profile):
                commands = profile_commands(profile)
                self.assertEqual(len(commands), 550)
                self.assertEqual(commands[:100], [(0, 0, 0)] * 100)
                self.assertEqual(commands[-150:], [(0, 0, 0)] * 150)
                result = score_flat_profile(ideal_trace(profile), profile)
                self.assertEqual(result["passed"], 2)
                self.assertEqual(
                    len(result["trials"][0]["phases"]), 1 if profile == "stand" else 3
                )
                self.assertFalse(result["qualified"])
                self.assertFalse(result["qualification_eligible"])

    def test_stand_is_one_phase_without_repeated_drift_allowances(self):
        trace = ideal_trace("stand")
        trace["position_w"][:, :, 0] = np.linspace(0, 0.16, STEPS)[:, None]
        result = score_flat_profile(trace, "stand")
        self.assertEqual(result["passed"], 0)
        self.assertEqual(len(result["trials"][0]["phases"]), 1)

    def test_survival_and_episode_mean_error_do_not_rescue_lateral_refusal(self):
        trace = ideal_trace("left-02")
        trace["linear_velocity_b"][:] = 0
        trace["position_w"][:] = 0
        result = score_flat_profile(trace, "left-02")
        self.assertEqual(result["passed"], 0)
        self.assertTrue(result["trials"][0]["phases"][1]["acquisition_failed"])

    def test_first_termination_latches_failure_even_if_later_episode_is_perfect(self):
        for key in ("terminated", "truncated"):
            for tick in (3, STEPS - 1):
                trace = ideal_trace("forward-05")
                trace[key][tick, 0] = True
                result = score_flat_profile(trace, "forward-05")
                self.assertEqual(result["passed"], 1)
                self.assertEqual(result["trials"][0]["first_attempt_steps"], tick + 1)
        # Bad later episodes cannot erase or redefine the original failure either.
        trace = ideal_trace("stand")
        trace["terminated"][3, 0] = True
        trace["quaternion_w"][4:, 0] = np.nan
        result = score_flat_profile(trace, "stand")
        self.assertNotIn(
            "nonfinite first-attempt state", result["trials"][0]["failures"]
        )

    def test_posture_checks_initial_pose_and_every_post_physics_sample(self):
        for key in ("initial_quaternion_w", "quaternion_w"):
            trace = ideal_trace("stand")
            quat = [np.cos(np.deg2rad(15.1) / 2), np.sin(np.deg2rad(15.1) / 2), 0, 0]
            if key == "quaternion_w":
                trace[key][1, 0] = quat
            else:
                trace[key][0] = quat
            result = score_flat_profile(trace, "stand")
            self.assertEqual(result["passed"], 1)
            self.assertIn(
                "flat attitude exceeds 15 degrees", result["trials"][0]["failures"]
            )

    def test_heading_wrap_is_not_a_full_turn(self):
        trace = ideal_trace("stand")
        angle = np.linspace(np.pi - 0.04, np.pi + 0.04, STEPS)
        trace["initial_quaternion_w"][:] = [
            np.cos(angle[0] / 2),
            0,
            0,
            np.sin(angle[0] / 2),
        ]
        trace["quaternion_w"][:, :, 0] = np.cos(angle[:, None] / 2)
        trace["quaternion_w"][:, :, 3] = np.sin(angle[:, None] / 2)
        self.assertEqual(score_flat_profile(trace, "stand")["passed"], 2)

    def test_base_contact_requires_five_consecutive_samples(self):
        for ticks, failed in (
            ([1, 3, 5, 7, 9], False),
            (range(1, 5), False),
            (range(1, 6), True),
        ):
            trace = ideal_trace("stand")
            trace["base_contact_force_w"][list(ticks), 0, :2] = 15
            result = score_flat_profile(trace, "stand")
            self.assertEqual(
                "sustained base contact" in result["trials"][0]["failures"], failed
            )

    def test_domain_and_stall_use_world_displacements_not_global_x_progress(self):
        trace = ideal_trace("left-02")
        self.assertEqual(score_flat_profile(trace, "left-02")["passed"], 2)
        trace["position_w"][:] = 0
        result = score_flat_profile(trace, "left-02")
        self.assertIn("stall", result["trials"][0]["phases"][1]["failures"])
        trace = ideal_trace("stand")
        trace["body_position_w"][2, 0, 4, 1] = 6.01
        result = score_flat_profile(trace, "stand")
        self.assertEqual(result["passed"], 1)
        self.assertIn("body link leaves flat domain", result["trials"][0]["failures"])

    def test_malformed_or_changed_commands_reject_entire_evidence(self):
        for key in ("position_w", "body_position_w", "command", "terminated"):
            trace = ideal_trace("stand")
            trace[key] = trace[key][1:]
            with self.subTest(key=key), self.assertRaises(ValueError):
                score_flat_profile(trace, "stand")
        trace = ideal_trace("stand")
        trace["command"][0, 0, 0] = 0.2
        with self.assertRaisesRegex(ValueError, "Executed commands"):
            score_flat_profile(trace, "stand")
        trace = ideal_trace("stand")
        trace["terminated"] = trace["terminated"].astype(float)
        with self.assertRaisesRegex(ValueError, "boolean"):
            score_flat_profile(trace, "stand")

    def test_nonfinite_and_bad_quaternion_fail_only_affected_original_row(self):
        for key, value in (("linear_velocity_b", np.nan), ("quaternion_w", 0.0)):
            trace = ideal_trace("stand")
            trace[key][1, 0] = value
            self.assertEqual(score_flat_profile(trace, "stand")["passed"], 1)


class DevelopmentBankTests(unittest.TestCase):
    def aggregate(self, results):
        return aggregate_development_bank(
            results, {profile: self.manifest(profile) for profile in PROFILES}
        )

    def manifest(self, profile="stand"):
        draw = TaskRandomization(
            dict(num_envs=100, seed=42, bank_profile=profile), evaluation=True
        )
        draw.dynamics()
        draw.sample_start(None)
        return draw.manifest()

    def outcomes(self, nominal=50, randomized=50):
        return [
            dict(attempt_index=i, passed=i < nominal if i < 50 else i - 50 < randomized)
            for i in range(100)
        ]

    def trace(self, profile="stand"):
        manifest = self.manifest(profile)
        trace = ideal_trace(profile, count=100)
        for index, attempt in enumerate(manifest["attempts"]):
            yaw = attempt["start"]["start_yaw"]
            xy = np.asarray(attempt["start"]["start_xy"])
            cosine, sine = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[cosine, -sine], [sine, cosine]])
            for key in ("position_w", "body_position_w"):
                trace[key][:, index, ..., :2] = (
                    trace[key][:, index, ..., :2] @ rotation.T + xy
                )
            trace["initial_position_w"][index, :2] = xy
            angle = (
                2
                * np.arctan2(
                    trace["quaternion_w"][:, index, 3],
                    trace["quaternion_w"][:, index, 0],
                )
                + yaw
            )
            trace["quaternion_w"][:, index, 0] = np.cos(angle / 2)
            trace["quaternion_w"][:, index, 3] = np.sin(angle / 2)
            trace["initial_quaternion_w"][index] = [
                np.cos(yaw / 2),
                0,
                0,
                np.sin(yaw / 2),
            ]
        return trace, manifest

    def test_group_floors_are_independent_of_overall_and_confidence_interval(self):
        for nominal, randomized, passed in (
            (45, 45, True),
            (44, 50, False),
            (50, 44, False),
        ):
            result = summarize_development_profile(
                "stand", self.outcomes(nominal, randomized), self.manifest()
            )
            self.assertEqual(result["group_passed"], passed)
            self.assertEqual(result["overall"]["passed"], nominal + randomized)
            self.assertEqual(result["strata"]["nominal"]["total"], 50)
            self.assertFalse(result["qualified"])
        interval = result["overall"]["wilson_95"]
        self.assertLess(interval[0], result["overall"]["fraction"])
        self.assertGreater(interval[1], result["overall"]["fraction"])

    def test_missing_attempts_keep_denominators_and_do_not_erase_established_failure(
        self,
    ):
        trials = self.outcomes()[:90]
        trials[0].update(
            passed=False, failures=["native termination before interruption"]
        )
        result = summarize_development_profile("stand", trials, self.manifest())
        self.assertFalse(result["complete"])
        self.assertFalse(result["group_passed"])
        self.assertEqual(result["overall"]["total"], 100)
        self.assertEqual(result["overall"]["passed"], 89)
        self.assertEqual(result["strata"]["randomized"]["total"], 50)
        self.assertEqual(result["missing_attempt_ids"], list(range(90, 100)))
        self.assertEqual(result["trials"][0]["failures"], trials[0]["failures"])
        with self.assertRaisesRegex(ValueError, "unique assigned"):
            summarize_development_profile(
                "stand", trials + [dict(attempt_index=0, passed=True)], self.manifest()
            )

    def test_native_trace_requires_all_rows_and_preserves_first_episode_scoring(self):
        trace, manifest = self.trace()
        trace["terminated"][5, 0] = True
        result = score_development_profile(trace, "stand", manifest)
        self.assertEqual(result["overall"]["passed"], 99)
        self.assertFalse(result["trials"][0]["passed"])
        self.assertEqual(result["trials"][0]["first_attempt_steps"], 6)
        with self.assertRaisesRegex(ValueError, "100 assigned"):
            score_development_profile(ideal_trace("stand"), "stand", self.manifest())

    def test_native_initial_pose_must_match_recorded_draw_not_just_bounds(self):
        trace, manifest = self.trace("forward-05")
        self.assertEqual(
            score_development_profile(trace, "forward-05", manifest)["overall"][
                "passed"
            ],
            100,
        )
        trace["initial_position_w"][0, 0] += 0.001
        with self.assertRaisesRegex(ValueError, "root poses"):
            score_development_profile(trace, "forward-05", manifest)
        trace, manifest = self.trace()
        trace["initial_quaternion_w"][0] = [1, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "root poses"):
            score_development_profile(trace, "stand", manifest)

    def test_manifest_must_name_exact_group_ids_and_balanced_realized_settings(self):
        for change in (
            lambda m: m.update(namespace="go2-roa-exit-v2-r6/qualification"),
            lambda m: m.update(group_id="flat/left-02"),
            lambda m: m["attempts"][1].update(attempt_index=0),
            lambda m: m["attempts"][0].update(stratum="randomized"),
            lambda m: m["attempts"][0].update(start=None),
            lambda m: m["attempts"][0].update(start={}),
            lambda m: m["attempts"][0]["dynamics"]["kp_scale"].__setitem__(0, 9.0),
            lambda m: m["attempts"][50]["dynamics"].update(static_friction=0.7),
            lambda m: m["attempts"][0]["streams"]["start"].update(sha256="0" * 64),
            lambda m: m["attempts"][0]["start"].update(nominal_heading=0.0),
            lambda m: m["attempts"][0].update(attempt_index=False),
            lambda m: m["attempts"][0]["dynamics"].update(added_mass=0),
            lambda m: m["attempts"][0]["start"]["joint_offset"].__setitem__(
                0, float("nan")
            ),
        ):
            manifest = self.manifest()
            change(manifest)
            with self.assertRaisesRegex(ValueError, "manifest"):
                summarize_development_profile("stand", self.outcomes(), manifest)

    def test_complete_bank_never_averages_away_a_failed_or_missing_profile(self):
        results = [
            dict(
                summarize_development_profile(
                    profile, self.outcomes(), self.manifest(profile)
                ),
                controller_state_sha256="a" * 64,
            )
            for profile in PROFILES
        ]
        result = self.aggregate(results)
        self.assertTrue(result["bank_passed"])
        self.assertFalse(result["qualified"])
        self.assertFalse(self.aggregate(results[:-1])["bank_passed"])
        results[0] = dict(
            summarize_development_profile(
                "stand", self.outcomes(44, 50), self.manifest()
            ),
            controller_state_sha256="a" * 64,
        )
        self.assertFalse(self.aggregate(results)["bank_passed"])
        with self.assertRaisesRegex(ValueError, "unique known"):
            self.aggregate(results + [results[0]])
        with self.assertRaisesRegex(ValueError, "Ordinary diagnostics"):
            self.aggregate([score_flat_profile(ideal_trace("stand"), "stand")])

    def test_aggregation_rejects_counter_tampering_and_mixed_controllers(self):
        stand = dict(
            summarize_development_profile("stand", self.outcomes(), self.manifest()),
            controller_state_sha256="a" * 64,
        )
        forward = dict(
            summarize_development_profile(
                "forward-05", self.outcomes(), self.manifest("forward-05")
            ),
            controller_state_sha256="b" * 64,
        )
        with self.assertRaisesRegex(ValueError, "same frozen controller"):
            self.aggregate([stand, forward])
        stand["overall"]["passed"] = 99
        with self.assertRaisesRegex(ValueError, "counts or evidence metadata"):
            self.aggregate([stand])

    def test_manifest_roundoff_keeps_saved_digest_and_exact_identities(self):
        manifest = self.manifest()
        value = manifest["attempts"][0]["start"]["joint_offset"][0]
        manifest["attempts"][0]["start"]["joint_offset"][0] = float(
            np.nextafter(value, np.inf)
        )
        result = dict(
            summarize_development_profile("stand", self.outcomes(), manifest),
            controller_state_sha256="a" * 64,
        )
        digest = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(result["manifest_content_sha256"], digest)
        aggregate_development_bank([result], {"stand": manifest})
        with self.assertRaisesRegex(ValueError, "counts or evidence metadata"):
            self.aggregate([result])
        manifest["attempts"][0]["start"]["joint_offset"][0] = value + 1e-12
        with self.assertRaisesRegex(ValueError, "manifest"):
            summarize_development_profile("stand", self.outcomes(), manifest)

    def write_run(self, root, profile="stand"):
        folder = Path(root) / profile / "evaluate_fixture"
        folder.mkdir(parents=True)
        trace, manifest = self.trace(profile)
        np.savez_compressed(folder / "motion.npz", **trace)
        sampler = TaskRandomization(
            dict(num_envs=100, seed=0, bank_profile=profile), evaluation=True
        )
        amplitudes = np.array([0.2] * 3 + [0.05] * 3 + [0.01] * 12 + [1.5] * 12)
        noise = np.stack(
            [
                (
                    sampler.uniform(None, 30, "observation-noise", -1, 1) * amplitudes
                ).astype(np.float32)
                for _ in range(STEPS)
            ]
        )
        np.savez_compressed(
            folder / "tracking.npz",
            causal_sensor_noise=noise,
            initial_raw_sensors=np.zeros((100, 30), dtype=np.float32),
            initial_noisy_sensors=noise[0],
            root_com_velocity=trace["linear_velocity_b"],
        )
        builder = TapeBuilder({"controller_sha256": "a" * 64})
        for step, command in enumerate(profile_commands(profile)):
            builder.append(step, command)
        (folder / "commands.json").write_text(
            json.dumps(builder.finish(completed=True))
        )
        (folder / "manifest.json").write_text(json.dumps(manifest))
        report = dict(
            status="EVALUATION_COMPLETE_NOT_QUALIFIED",
            policy_unchanged=True,
            config={
                "task": {"terrain": "flat", "num_envs": 100, "bank_profile": profile}
            },
            control_steps=STEPS,
            environment_transitions=STEPS * 100,
            cleanup={"environment": "complete", "application": "complete"},
            actor_sha256="b" * 64,
            package_sources={"evaluation/flat.py": "c" * 64},
            motor_delivery=dict(
                encoded_steps=STEPS,
                verified_delivery_steps=STEPS,
                native_step_returns=STEPS,
                native_verified_rows=STEPS * 100,
                excluded_terminal_rows=0,
                faulted=False,
                pending_delivery=False,
            ),
            evidence_sha256={
                name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                for name in ("commands.json", "motion.npz", "tracking.npz")
            },
            manifest_sha256=hashlib.sha256(
                (folder / "manifest.json").read_bytes()
            ).hexdigest(),
            sensor_noise=dict(
                generated_frames=STEPS,
                draw_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
            ),
            flat_development=dict(
                score_development_profile(trace, profile, manifest),
                controller_state_sha256="a" * 64,
            ),
        )
        (folder / "report.json").write_text(json.dumps(report))
        return folder, report

    def test_saved_bank_analysis_rescores_and_keeps_missing_groups_incomplete(self):
        with tempfile.TemporaryDirectory() as root:
            folder, _ = self.write_run(root)
            before = {path: path.read_bytes() for path in folder.iterdir()}
            result = analyze_development_bank(root)
            self.assertFalse(result["complete"])
            self.assertFalse(result["bank_passed"])
            self.assertEqual(result["groups"]["flat/stand"]["overall"]["passed"], 100)
            self.assertEqual(len(result["missing_groups"]), 9)
            self.assertEqual(
                before, {path: path.read_bytes() for path in folder.iterdir()}
            )
            shutil.copytree(folder, Path(root) / "repeated" / "evaluate_fixture")
            with self.assertRaisesRegex(ValueError, "repeated group"):
                analyze_development_bank(root)

    def test_saved_bank_accepts_numerical_roundoff_but_not_changed_decisions(self):
        with tempfile.TemporaryDirectory() as root:
            folder, report = self.write_run(root)
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["attempts"][0]["start"]["joint_offset"][0] += 1e-16
            manifest_path.write_text(json.dumps(manifest))
            report["manifest_sha256"] = hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
            stored = report["flat_development"]
            stored["manifest_content_sha256"] = hashlib.sha256(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            block = stored["trials"][0]["phases"][0]["blocks"][0]
            block["yaw_error_rad_s"] += float(2 * np.finfo(np.float32).eps)
            (folder / "report.json").write_text(json.dumps(report))
            self.assertEqual(
                analyze_development_bank(root)["groups"]["flat/stand"]["overall"][
                    "passed"
                ],
                100,
            )
            for target, key, value in (
                (block, "yaw_error_rad_s", 1e-4),
                (stored["trials"][0], "passed", False),
                (stored["trials"][0], "passed", 1),
                (stored["thresholds"], "planar_error_m_s", 0.10000000001),
                (stored, "manifest_content_sha256", "0" * 64),
            ):
                before = target[key]
                target[key] = value
                (folder / "report.json").write_text(json.dumps(report))
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(ValueError, "rescored evidence"),
                ):
                    analyze_development_bank(root)
                target[key] = before

    def test_saved_bank_rejects_failed_cleanup_changed_hash_and_rewritten_pass_count(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            folder, original = self.write_run(root)
            for mutate, reason in (
                (
                    lambda report: report["cleanup"].update(application="pending"),
                    "successful cleanup",
                ),
                (
                    lambda report: report["evidence_sha256"].update(
                        **{"motion.npz": "0" * 64}
                    ),
                    "changed evidence",
                ),
                (
                    lambda report: report["flat_development"]["overall"].update(
                        passed=99
                    ),
                    "rescored evidence",
                ),
            ):
                report = json.loads(json.dumps(original))
                mutate(report)
                (folder / "report.json").write_text(json.dumps(report))
                with (
                    self.subTest(reason=reason),
                    self.assertRaisesRegex(ValueError, reason),
                ):
                    analyze_development_bank(root)

    def test_saved_bank_rejects_noise_changes_even_if_report_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as root:
            folder, report = self.write_run(root)
            with np.load(folder / "tracking.npz") as archive:
                tracking = dict(archive)
            tracking["causal_sensor_noise"][100, 5, 3] *= -1
            np.savez_compressed(folder / "tracking.npz", **tracking)
            report["evidence_sha256"]["tracking.npz"] = hashlib.sha256(
                (folder / "tracking.npz").read_bytes()
            ).hexdigest()
            report["sensor_noise"]["draw_sha256"] = hashlib.sha256(
                tracking["causal_sensor_noise"].tobytes()
            ).hexdigest()
            (folder / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "assigned stream"):
                analyze_development_bank(root)

    def test_saved_bank_requires_same_actor_and_executed_sources(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_run(root)
            folder, original = self.write_run(root, "forward-05")
            for change in (
                {"actor_sha256": "d" * 64},
                {"package_sources": {"different.py": "e" * 64}},
            ):
                report = dict(original, **change)
                (folder / "report.json").write_text(json.dumps(report))
                with self.assertRaisesRegex(
                    ValueError, "actor or executed sources changed"
                ):
                    analyze_development_bank(root)


if __name__ == "__main__":
    unittest.main()
