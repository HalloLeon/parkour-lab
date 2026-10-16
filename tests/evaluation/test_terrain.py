"""Analytic first-attempt geometry/timing cases, not simulated locomotion."""

from copy import deepcopy
import math
import unittest

import numpy as np

from parkour_lab.evaluation.terrain import score_terrain_attempt


def ideal(*, axis=0, reverse=False, yaw=0.0, family="ramp"):
    direction = -1 if reverse and family != "hill" else 1
    start = np.array([5.0, 5.0, 0.5])
    start[axis] = 18 if direction < 0 else 4
    approach = start[:2].copy()
    approach[axis] += direction * 2
    exit_point = approach.copy()
    exit_point[axis] += direction * 2
    footprint = np.array(
        [np.minimum(approach, exit_point), np.maximum(approach, exit_point)]
    )
    footprint[:, 1 - axis] += [-1, 1]
    finish = start[:2].copy()
    finish[axis] += direction * 8.75
    landing = footprint.copy()
    landing[:, axis] = sorted((exit_point[axis], finish[axis] + direction))
    rotation = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    metadata = dict(
        size_m=[24.0, 24.0],
        trial=dict(
            family=family,
            orientation=("long-first" if reverse else "short-first")
            if family == "hill"
            else ("down" if reverse else "up"),
            local_to_world_column_transform=matrix.tolist(),
            target_footprint_local_m=footprint.tolist(),
            approach_edge_local_m=approach.tolist(),
            exit_edge_local_m=exit_point.tolist(),
            far_side_landing_local_m=landing.tolist(),
            final_stop_region_local_m=[(finish - 1).tolist(), (finish + 1).tolist()],
        ),
    )
    command = np.zeros((1500, 1, 3))
    command[100:1350, 0, 0] = 0.35
    position = np.broadcast_to(start, (1500, 1, 3)).copy()
    position[:, 0, axis] += np.cumsum(command[:, 0, 0]) * 0.02 * direction
    feet = np.repeat(position[:, :, None], 4, axis=2)
    offsets = np.zeros((4, 3))
    offsets[:, axis] = [0.22, 0.22, -0.22, -0.22]
    offsets[:, 1 - axis] = [-0.15, 0.15, -0.15, 0.15]
    offsets[:, 2] = -0.47
    feet += offsets
    initial_feet = start + offsets
    heading = yaw + math.atan2(direction if axis else 0, direction if not axis else 0)
    q = np.array([math.cos(heading / 2), 0, 0, math.sin(heading / 2)])
    trace = dict(
        position_w=position @ rotation.T,
        quaternion_w=np.broadcast_to(q, (1500, 1, 4)).copy(),
        linear_velocity_b=command.copy(),
        angular_velocity_b=np.zeros((1500, 1, 3)),
        base_contact_force_w=np.zeros((1500, 1, 3)),
        command=command,
        foot_position_w=feet @ rotation.T,
        body_position_w=np.concatenate((position[:, :, None], feet), axis=2)
        @ rotation.T,
        initial_position_w=(start @ rotation.T)[None],
        initial_quaternion_w=q[None],
        initial_foot_position_w=(initial_feet @ rotation.T)[None],
        initial_body_position_w=(np.vstack((start, initial_feet)) @ rotation.T)[None],
        terminated=np.zeros((1500, 1), dtype=bool),
        truncated=np.zeros((1500, 1), dtype=bool),
    )
    return trace, metadata


def move_base(trace, rows, xy):
    """Keep the root body centre consistent while changing the geometric base path."""
    trace["position_w"][rows, 0, :2] = xy
    trace["body_position_w"][rows, 0, 0, :2] = xy


def cut(trace, length):
    return {
        key: value.copy() if key.startswith("initial_") else value[:length].copy()
        for key, value in trace.items()
    }


class TerrainTests(unittest.TestCase):
    def test_rotated_up_down_and_reflected_hill_crossings_pass(self):
        for axis, reverse, yaw, family in (
            (0, False, 0, "stairs"),
            (0, True, 0.7, "stairs"),
            (1, False, -0.8, "ramp"),
            (1, True, 0.37, "ramp"),
            (0, False, 1.1, "hill"),
            (0, True, -1.2, "hill"),
        ):
            with self.subTest(axis=axis, reverse=reverse, family=family):
                trace, meta = ideal(axis=axis, reverse=reverse, yaw=yaw, family=family)
                result = score_terrain_attempt(trace, meta)
                self.assertTrue(result["kinematic_passed"], result["failures"])
                self.assertTrue(result["complete"])
                self.assertFalse(result["qualified"])
                self.assertFalse(result["qualification_eligible"])
                self.assertEqual(result["finish"]["samples"], 101)
                self.assertEqual(
                    [p["tracking"]["samples"] for p in result["phases"]],
                    [50, 1200, 100],
                )
                cross = result["crossing"]
                self.assertEqual(
                    cross["entry_time_s"], (cross["entry_step"] + 1) * 0.02
                )
                self.assertEqual(
                    cross["tracking"]["samples"],
                    cross["clear_step"] - cross["entry_step"] + 1,
                )

    def test_bypass_side_entry_and_far_entry_cannot_pass(self):
        for kind in ("bypass", "side", "far", "initial_inside"):
            trace, meta = ideal()
            if kind == "bypass":
                move_base(
                    trace,
                    slice(0, 800),
                    np.column_stack(
                        (trace["position_w"][:800, 0, 0], np.full(800, 7.0))
                    ),
                )
            elif kind == "side":
                move_base(trace, slice(0, 200), [7.0, 7.0])
                trace["initial_position_w"][0, :2] = [7.0, 7.0]
                move_base(trace, slice(200, 300), [7.0, 5.0])
            elif kind == "far":
                trace["initial_position_w"][0, :2] = [9.0, 5.0]
                move_base(trace, slice(0, 200), [9.0, 5.0])
                move_base(trace, slice(200, 300), [7.0, 5.0])
            else:
                trace["initial_position_w"][0, :2] = [7.0, 5.0]
            with self.subTest(kind=kind):
                result = score_terrain_attempt(trace, meta)
                self.assertFalse(result["kinematic_passed"])
                self.assertIsNone(result["crossing"]["entry_step"])

    def test_segment_enters_side_despite_preceding_point_on_approach_halfplane(self):
        trace, meta = ideal()
        move_base(trace, slice(0, 200), [5.0, 9.0])
        move_base(trace, 200, [7.0, 5.0])
        result = score_terrain_attempt(trace, meta)
        self.assertIn(
            "first footprint entry is not from the approach side", result["failures"]
        )

    def test_side_exit_and_skipped_whole_footprint_fail(self):
        trace, meta = ideal()
        entry = score_terrain_attempt(trace, meta)["crossing"]["entry_step"]
        move_base(trace, entry + 2, [6.1, 7.0])
        result = score_terrain_attempt(trace, meta)
        self.assertIn(
            "base leaves the footprint through a side edge", result["failures"]
        )
        trace, meta = ideal()
        move_base(trace, slice(0, 200), [5.0, 5.0])
        move_base(trace, slice(200, None), [9.0, 5.0])
        result = score_terrain_attempt(trace, meta)
        self.assertIn(
            "footprint intersection lacks an observed inside sample", result["failures"]
        )

    def test_retreat_reentry_retains_first_anchor_and_side_bypass_still_fails(self):
        trace, meta = ideal()
        first = score_terrain_attempt(trace, meta)["crossing"]["entry_step"]
        move_base(trace, slice(first + 1, first + 10), [5.8, 5.0])
        result = score_terrain_attempt(trace, meta)
        self.assertTrue(result["kinematic_passed"], result["failures"])
        self.assertEqual(result["crossing"]["entry_step"], first)
        trace["linear_velocity_b"][first : first + 9, 0, 0] = 20
        self.assertFalse(
            score_terrain_attempt(trace, meta)["crossing"]["tracking"]["passed"]
        )
        trace, meta = ideal()
        move_base(trace, first + 1, [5.8, 5.0])
        move_base(trace, first + 2, [5.8, 7.0])
        move_base(trace, slice(first + 3, 900), [9.0, 7.0])
        move_base(trace, slice(900, None), [9.0, 5.0])
        result = score_terrain_attempt(trace, meta)
        self.assertIsNone(result["crossing"]["clear_step"])

    def test_first_invalid_entry_is_not_replaced_by_later_valid_entry(self):
        trace, meta = ideal()
        move_base(trace, slice(0, 150), [7.0, 7.0])
        trace["initial_position_w"][0, :2] = [7.0, 7.0]
        move_base(trace, 150, [7.0, 5.0])
        move_base(trace, 151, [5.0, 5.0])
        result = score_terrain_attempt(trace, meta)
        self.assertIsNone(result["crossing"]["entry_step"])
        self.assertFalse(result["kinematic_passed"])

    def test_retreat_then_side_or_far_reentry_cannot_replace_traversal(self):
        for reentry in ([7.0, 5.0], [9.0, 5.0]):
            trace, meta = ideal()
            first = score_terrain_attempt(trace, meta)["crossing"]["entry_step"]
            move_base(trace, first + 1, [5.8, 5.0])
            move_base(trace, first + 2, [5.8, 7.0])
            move_base(trace, first + 3, [9.0, 7.0])
            move_base(trace, first + 4, reentry)
            move_base(trace, first + 5, [7.9, 5.0])
            move_base(trace, slice(first + 6, None), [9.0, 5.0])
            with self.subTest(reentry=reentry):
                result = score_terrain_attempt(trace, meta)
                self.assertEqual(result["crossing"]["entry_step"], first)
                self.assertIsNone(result["crossing"]["clear_step"])
                self.assertIn(
                    "footprint re-entry is not from the approach side",
                    result["failures"],
                )

    def test_segment_corner_exit_is_also_a_side_exit(self):
        trace, meta = ideal()
        first = score_terrain_attempt(trace, meta)["crossing"]["entry_step"]
        move_base(trace, first + 1, [7.0, 5.0])
        move_base(trace, first + 2, [9.0, 7.0])  # exactly through the far-side corner
        self.assertIn(
            "base leaves the footprint through a side edge",
            score_terrain_attempt(trace, meta)["failures"],
        )

    def test_all_feet_must_clear_not_just_root_and_aerial_phases_are_allowed(self):
        trace, meta = ideal()
        trace["foot_position_w"][..., 2] += 2
        self.assertTrue(score_terrain_attempt(trace, meta)["kinematic_passed"])
        trace["foot_position_w"][:, 0, 3, 0] = 8.0  # exit plane is not beyond it
        result = score_terrain_attempt(trace, meta)
        self.assertIsNone(result["crossing"]["clear_step"])

    def test_crossing_tracking_is_inclusive_and_separate_from_phase_average(self):
        trace, meta = ideal()
        event = score_terrain_attempt(trace, meta)["crossing"]
        for step in (event["entry_step"], event["clear_step"]):
            changed = deepcopy(trace)
            changed["linear_velocity_b"][step, 0, 0] += 100
            result = score_terrain_attempt(changed, meta)
            self.assertFalse(result["crossing"]["tracking"]["passed"])
            self.assertTrue(result["phases"][1]["tracking"]["passed"])

    def test_phase_skip_and_euclidean_and_body_yaw_limits(self):
        trace, meta = ideal()
        trace["linear_velocity_b"][:50, 0, 0] = 100
        self.assertTrue(
            score_terrain_attempt(trace, meta)["phases"][0]["tracking"]["passed"]
        )
        trace["linear_velocity_b"][50:100, 0, :2] = [0.15, 0.15]
        result = score_terrain_attempt(trace, meta)
        self.assertFalse(result["phases"][0]["tracking"]["passed"])
        trace, meta = ideal()
        trace["angular_velocity_b"][150:1350, 0, 2] = 0.201
        self.assertFalse(
            score_terrain_attempt(trace, meta)["phases"][1]["tracking"]["passed"]
        )

    def test_crossing_deadline_27s_includes_row1349_not1350(self):
        for step, passed in ((1349, True), (1350, False)):
            trace, meta = ideal()
            trace["foot_position_w"][:step, 0, 3, 0] = 8.0
            result = score_terrain_attempt(trace, meta)
            self.assertEqual(result["crossing"]["clear_step"], step)
            self.assertEqual(
                "crossing completes after 27 s" not in result["failures"], passed
            )

    def test_finish_inclusive_time_edges_region_and_tilt(self):
        for step, should_pass in ((1398, True), (1399, False), (1499, False)):
            trace, meta = ideal()
            trace["foot_position_w"][step, 0, 0, 1] = 8.0
            self.assertEqual(
                score_terrain_attempt(trace, meta)["finish"]["passed"], should_pass
            )
        trace, meta = ideal()
        angle = math.radians(21)
        trace["quaternion_w"][1399, 0] = [
            math.cos(angle / 2),
            math.sin(angle / 2),
            0,
            0,
        ]
        self.assertFalse(score_terrain_attempt(trace, meta)["finish"]["passed"])

    def test_finish_joint_fraction_not_separate_speed_and_yaw_fractions(self):
        trace, meta = ideal()
        trace["linear_velocity_b"][1399:1409, 0, 0] = 0.081
        self.assertTrue(score_terrain_attempt(trace, meta)["finish"]["passed"])
        trace["angular_velocity_b"][1409:1419, 0, 2] = 0.101
        result = score_terrain_attempt(trace, meta)
        self.assertAlmostEqual(result["finish"]["joint_velocity_fraction"], 81 / 101)
        self.assertFalse(result["finish"]["passed"])

    def test_inclusive_velocity_force_and_tilt_thresholds(self):
        trace, meta = ideal()
        trace["linear_velocity_b"][50:100, 0, 0] = 0.20
        trace["angular_velocity_b"][50:100, 0, 2] = 0.20
        trace["base_contact_force_w"][200:205, 0, 2] = 20
        angle = math.radians(60)
        trace["quaternion_w"][200:205, 0] = [
            math.cos(angle / 2),
            math.sin(angle / 2),
            0,
            0,
        ]
        angle = math.radians(20)
        trace["quaternion_w"][1399:, 0] = [
            math.cos(angle / 2),
            math.sin(angle / 2),
            0,
            0,
        ]
        trace["linear_velocity_b"][1399:, 0, 0] = 0.08
        trace["angular_velocity_b"][1399:, 0, 2] = 0.10
        result = score_terrain_attempt(trace, meta)
        self.assertTrue(result["kinematic_passed"], result["failures"])

    def test_finish_speed_rotates_full_body_com_velocity(self):
        trace, meta = ideal()
        angle = math.radians(19)
        trace["quaternion_w"][1399:, 0] = [
            math.cos(angle / 2),
            0,
            math.sin(angle / 2),
            0,
        ]
        trace["linear_velocity_b"][1399:, 0, 2] = 0.3
        result = score_terrain_attempt(trace, meta)
        self.assertEqual(result["finish"]["joint_velocity_fraction"], 0)
        self.assertFalse(result["finish"]["passed"])

    def test_finish_uses_27s_anchor_and_28_to30_net_drift(self):
        trace, meta = ideal()
        move_base(
            trace, slice(1350, None), trace["position_w"][1349, 0, :2] + [0.151, 0]
        )
        self.assertGreater(
            score_terrain_attempt(trace, meta)["finish"]["onset_excursion_m"], 0.15
        )
        trace, meta = ideal()
        base = trace["position_w"][1349, 0, :2].copy()
        move_base(trace, 1399, base + [0.101, 0])
        result = score_terrain_attempt(trace, meta)
        self.assertLess(result["finish"]["onset_excursion_m"], 0.15)
        self.assertGreater(result["finish"]["net_drift_m"], 0.10)

    def test_sustained_falls_need_five_and_native_ending_cuts_retry(self):
        for count in (4, 5):
            trace, meta = ideal()
            trace["base_contact_force_w"][200 : 200 + count, 0, 2] = 21
            result = score_terrain_attempt(trace, meta)
            self.assertEqual("sustained base contact" in result["failures"], count == 5)
            trace, meta = ideal()
            angle = math.radians(61)
            trace["quaternion_w"][200 : 200 + count, 0] = [
                math.cos(angle / 2),
                math.sin(angle / 2),
                0,
                0,
            ]
            self.assertEqual(
                "sustained tilt fall" in score_terrain_attempt(trace, meta)["failures"],
                count == 5,
            )
        trace, meta = ideal()
        trace["terminated"][199, 0] = True
        result = score_terrain_attempt(trace, meta)
        self.assertEqual(result["first_attempt_steps"], 200)
        self.assertIsNone(result["crossing"]["entry_step"])
        self.assertFalse(result["kinematic_passed"])

    def test_stall_uses_net_xy_full3s_after_first1s_and_domain_uses_all_bodies(self):
        trace, meta = ideal()
        move_base(trace, slice(149, 300), trace["position_w"][149, 0, :2])
        result = score_terrain_attempt(trace, meta)
        self.assertIn("translating phase stall", result["failures"])
        self.assertEqual(result["failure_events"]["stall_window_start_s"], 3.0)
        trace, meta = ideal()
        trace["body_position_w"][50, 0, 4, 0] = -0.001
        self.assertIn(
            "body-link centre leaves finite map",
            score_terrain_attempt(trace, meta)["failures"],
        )
        trace, meta = ideal()
        trace["position_w"][10, 0, 0] = -0.1
        result = score_terrain_attempt(trace, meta)
        self.assertIn("body-link centre leaves finite map", result["failures"])
        self.assertEqual(result["failure_events"]["out_of_domain_step"], 10)

    def test_partial_prefix_keeps_events_but_cannot_pass(self):
        trace, meta = ideal()
        result = score_terrain_attempt(cut(trace, 750), meta)
        self.assertTrue(result["crossing"]["complete"])
        self.assertFalse(result["complete"])
        self.assertFalse(result["kinematic_passed"])
        trace["command"][600:750] = 0
        self.assertIn(
            "commands differ from fixed 2/25/3 s terrain tape",
            score_terrain_attempt(cut(trace, 750), meta)["failures"],
        )

    def test_malformed_missing_nonfinite_and_invalid_metadata_never_pass(self):
        for key in ("foot_position_w", "initial_position_w", "body_position_w"):
            trace, meta = ideal()
            del trace[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                score_terrain_attempt(trace, meta)
        trace, meta = ideal()
        trace["terminated"] = trace["terminated"].astype(float)
        with self.assertRaises(ValueError):
            score_terrain_attempt(trace, meta)
        trace, meta = ideal()
        trace["position_w"][100] = np.nan
        self.assertFalse(score_terrain_attempt(trace, meta)["kinematic_passed"])
        trace, meta = ideal()
        meta["trial"]["local_to_world_column_transform"][0][0] = 2
        with self.assertRaises(ValueError):
            score_terrain_attempt(trace, meta)


if __name__ == "__main__":
    unittest.main()
