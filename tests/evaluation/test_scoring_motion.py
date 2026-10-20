"""Analytic good/bad motions, first endings, support exposure and missing phases."""

import json
import unittest

import numpy as np

from parkour_lab.evaluation.cases import EvaluationCase
from parkour_lab.evaluation.scoring import score_case
from tests.support.evaluation import analytic_trace


def flat(profile):
    return EvaluationCase("validation", "Flat", None, "plane", profile)


class MotionScoringTests(unittest.TestCase):
    def test_correct_full_pivots_retain_six_radians_across_heading_wraps(self):
        for profile, sign in (("pivot-left", 1), ("pivot-right", -1)):
            case = flat(profile)
            score = score_case(case, analytic_trace(case, heading_rad=2.9))
            self.assertAlmostEqual(score["motion"]["body_yaw_integral_rad"], sign * 6)
            self.assertAlmostEqual(
                score["motion"]["world_heading_change_rad"], sign * 6
            )
            self.assertEqual(score["motion"]["wrong_direction_fraction"], 0)
            self.assertEqual(score["motion"]["maximum_onset_xy_excursion_m"], 0)
            self.assertEqual(score["motion"]["flat_reference"]["path_rmse_m"], 0)
            self.assertEqual(score["core_return"], 1000)
            self.assertTrue(score["survived_horizon"])
            json.dumps(score, allow_nan=False)

    def test_wrong_sign_and_translation_during_pivot_remain_distinct(self):
        case = flat("pivot-left")
        trace = analytic_trace(case)
        trace["root_com_linear_velocity_b"][100:700, 0] = 0.2
        trace["root_position_w"][100:700, 0] += np.arange(1, 601) * 0.004
        trace["root_angular_velocity_b"][100:700, 2] = -0.5
        score = score_case(case, trace)
        self.assertAlmostEqual(score["motion"]["unwanted_planar_speed_m_s"], 0.2)
        self.assertAlmostEqual(score["motion"]["maximum_onset_xy_excursion_m"], 2.4)
        self.assertEqual(score["motion"]["wrong_direction_fraction"], 1)
        self.assertEqual(score["motion"]["yaw_rmse_rad_s"], 1)
        trace["root_angular_velocity_b"][100:700, 2] = 0
        zero = score_case(case, trace)
        self.assertEqual(zero["motion"]["wrong_direction_fraction"], 0)
        self.assertEqual(zero["motion"]["yaw_rmse_rad_s"], 0.5)

    def test_tilted_body_yaw_is_primary_world_heading_is_separate(self):
        case = EvaluationCase("validation", "Slopes", 8, "pyramid", "pivot-left")
        trace = analytic_trace(case)
        angle = np.cumsum(trace["command"][:, 2]) * 0.02
        cosine, sine = np.cos(np.pi / 6), np.sin(np.pi / 6)
        trace["initial_root_quaternion_w"] = np.array([cosine, 0, sine, 0])
        trace["root_quaternion_w"] = np.column_stack(
            (
                cosine * np.cos(angle / 2),
                sine * np.sin(angle / 2),
                sine * np.cos(angle / 2),
                cosine * np.sin(angle / 2),
            )
        )
        score = score_case(case, trace)
        self.assertEqual(score["motion"]["yaw_rmse_rad_s"], 0)
        self.assertAlmostEqual(score["motion"]["body_yaw_integral_rad"], 6)
        self.assertGreater(abs(score["motion"]["world_heading_change_rad"] - 6), 0.1)
        self.assertIsNone(score["motion"]["flat_reference"])

    def test_correct_arc_matches_anchored_reference_stationary_spinner_does_not(self):
        for profile in ("arc-left", "arc-right"):
            case = flat(profile)
            trace = analytic_trace(case, heading_rad=-2.7)
            good = score_case(case, trace)
            self.assertLess(good["motion"]["flat_reference"]["path_rmse_m"], 1e-12)
            self.assertAlmostEqual(
                good["motion"]["flat_reference"]["requested_radius_m"], 0.35 / 0.3
            )
            self.assertAlmostEqual(abs(good["motion"]["world_heading_change_rad"]), 3.6)
            trace["root_com_linear_velocity_b"][:, :2] = 0
            trace["root_position_w"][:, :2] = trace["initial_root_position_w"][:2]
            bad = score_case(case, trace)
            self.assertAlmostEqual(bad["motion"]["planar_rmse_m_s"], 0.35)
            self.assertEqual(bad["motion"]["yaw_rmse_rad_s"], 0)
            self.assertGreater(bad["motion"]["flat_reference"]["path_rmse_m"], 1)

    def test_stop_hold_residual_motion_and_drift_use_fixed_stop_onset(self):
        case = flat("forward-035")
        trace = analytic_trace(case)
        trace["root_com_linear_velocity_b"][700:, 0] = 0.1
        trace["root_angular_velocity_b"][700:, 2] = 0.2
        trace["root_position_w"][700:, 0] += np.arange(1, 301) * 0.002
        score = score_case(case, trace)
        self.assertAlmostEqual(score["stopping"]["residual_planar_speed_m_s"], 0.1)
        self.assertAlmostEqual(score["stopping"]["residual_absolute_yaw_rad_s"], 0.2)
        self.assertAlmostEqual(
            score["stopping"]["maximum_drift_from_stop_onset_m"], 0.6
        )
        self.assertEqual(
            {k: v["samples"] for k, v in score["phases"].items()},
            dict(stand=100, onset=50, settled_motion=550, braking=50, stop_hold=250),
        )

    def test_first_terminal_reward_once_and_replacement_episode_cannot_improve(self):
        case = flat("pivot-left")
        trace = analytic_trace(case)
        trace["terminated"][49] = trace["truncated"][49] = True
        trace["ending_base_contact"] = np.zeros(1000, dtype=bool)
        trace["ending_base_contact"][49] = True
        trace["reward"][49] = -50
        trace["reward"][50:] = 1e9
        trace["decision_time_s"][50:] = 0
        trace["command"][50:] = 123
        score = score_case(case, trace)
        self.assertEqual(score["observed_steps"], 50)
        self.assertEqual(score["core_return"], -1)
        self.assertEqual(score["first_ending"]["time_s"], 1)
        self.assertTrue(score["physical_failure"])
        self.assertTrue(score["first_ending"]["truncated"])
        self.assertIn("base_contact", score["first_ending"]["causes"])
        self.assertIsNone(score["phases"]["settled_motion"]["yaw_rmse_rad_s"])
        self.assertIsNone(score["stopping"]["residual_planar_speed_m_s"])

    def test_boundary_and_full_timeout_are_not_physical_failures(self):
        case = flat("stand")
        for index, cause in ((100, "terrain_boundary"), (999, "time_out")):
            trace = analytic_trace(case)
            trace["truncated"][index] = True
            trace[f"ending_{cause}"] = np.zeros(1000, dtype=bool)
            trace[f"ending_{cause}"][index] = True
            score = score_case(case, trace)
            self.assertFalse(score["physical_failure"])
            self.assertEqual(score["first_ending"]["causes"], [cause])
            self.assertEqual(score["survived_horizon"], index == 999)

    def test_short_unfinished_or_zero_trace_has_unknown_full_return(self):
        case = flat("stand")
        for steps in (0, 100):
            score = score_case(case, analytic_trace(case, steps=steps))
            self.assertIsNone(score["core_return"])
            self.assertEqual(score["recorded_return"], steps)
            self.assertIsNone(score["phases"]["settled_motion"]["planar_rmse_m_s"])
            json.dumps(score, allow_nan=False)

    def test_command_timing_and_normalized_quaternion_are_enforced(self):
        case = flat("stand")
        for name, index, value in (
            ("decision_time_s", 1, 0.03),
            ("command", (0, 0), 0.1),
            ("root_quaternion_w", (0, 0), 2),
        ):
            trace = analytic_trace(case)
            trace[name][index] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                score_case(case, trace)


class ExposureScoringTests(unittest.TestCase):
    def test_v2_plane_entry_and_fine_exposure_do_not_reinterpret_legacy_zero(self):
        from dataclasses import replace

        zero = EvaluationCase("validation", "Slopes", 0, "pyramid", "stand")
        for case in (zero, replace(zero, condition="composition")):
            self.assertEqual(
                score_case(case, analytic_trace(case))["exposure"]["label"],
                "flat_entry",
            )
        old = replace(zero, version="v1")
        self.assertEqual(
            score_case(old, analytic_trace(old))["exposure"]["label"],
            "platform_only_supported_contacts",
        )
        for version, level, expected in (
            ("v1", 3, "platform_only_supported_contacts"),
            ("v2", 4, "nonflat_contact_demonstrated"),
        ):
            case = replace(zero, version=version, physical_level=level)
            trace = analytic_trace(case)
            trace["support_height_w"][100:, 0] = 0.001
            self.assertEqual(score_case(case, trace)["exposure"]["label"], expected)

    def test_platform_pivot_has_no_nonflat_proof_and_true_support_change_does(self):
        case = EvaluationCase("validation", "Stairs", 8, "pyramid", "pivot-left")
        trace = analytic_trace(case)
        platform = score_case(case, trace)
        self.assertEqual(
            platform["exposure"]["label"], "platform_only_supported_contacts"
        )
        trace["support_height_w"][100:, 0] = 0.08
        trace["foot_position_w"][100:, 0, 2] = 0.1
        exposed = score_case(case, trace)
        self.assertEqual(exposed["exposure"]["label"], "nonflat_contact_demonstrated")
        self.assertEqual(
            exposed["exposure"]["demonstrated_nonflat_control_samples"], 900
        )
        self.assertIsNone(exposed["motion"]["flat_reference"])

    def test_missing_support_or_no_contact_is_unavailable_never_exposure(self):
        case = EvaluationCase("test", "Uneven", 4, "noise", "stand")
        trace = analytic_trace(case)
        trace["support_height_w"][:] = np.nan
        trace["support_valid"][:] = False
        score = score_case(case, trace)
        self.assertEqual(score["exposure"]["surface_sample_valid_fraction"], 0)
        self.assertEqual(
            score["exposure"]["label"], "unavailable_no_supported_contacts"
        )
        self.assertIsNone(
            score["exposure"]["demonstrated_nonflat_fraction_of_contact_samples"]
        )
        trace["support_valid"][0, 0] = True
        with self.assertRaises(ValueError):
            score_case(case, trace)


if __name__ == "__main__":
    unittest.main()
