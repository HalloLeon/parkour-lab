"""Validate a three-run scene-placement probe without certifying robot behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from . import startup_probe_report as probe
except ImportError:
    import startup_probe_report as probe


def _outcomes(left, right):
    return {
        key: probe._delta_stats(
            [s["post"]["state"][key] for s in left["samples"][:10]],
            [s["post"]["state"][key] for s in right["samples"][:10]],
        )
        for key in probe.OUTCOME_FIELDS
    }


def _centering(report, level):
    meta = report["metadata"].get("scene_intervention", {})
    for key, expected in (
        ("kind", "startup_centered_terrain"),
        ("schema_version", 1),
        ("configured", True),
        ("applied", True),
        ("selected_row", level),
        ("selected_column", 0),
        ("centering_abs_tolerance_m", 1e-9),
    ):
        probe._match(meta.get(key), expected, f"centering.{key}")
    origin = probe._flatten(meta.get("source_origin_w_m"))
    shift = probe._flatten(meta.get("translation_w_m"))
    probe._require(len(origin) == len(shift) == 3, "centering: expected xyz vectors")
    probe._match(shift, [-v for v in origin], "centering translation", 1e-9)
    probe._match(meta.get("centered_origin_w_m"), [0.0] * 3, "centered origin", 1e-9)
    probe._match(
        report["physical_metadata"]["environment_origin_w_m"],
        [0.0] * 3,
        "actual centered reset origin",
        2e-6,
    )
    error = meta.get("local_geometry_max_abs_error_m")
    probe._require(
        type(error) in (int, float) and 0 <= error <= 1e-9,
        "centering: local mesh geometry was not preserved",
    )
    for key in ("source_mesh_sha256", "centered_mesh_sha256"):
        value = meta.get(key)
        probe._require(
            isinstance(value, str)
            and len(value) == 64
            and all(c in "0123456789abcdef" for c in value),
            f"centering: invalid {key}",
        )
    for key in ("mesh_vertex_count", "mesh_face_count"):
        probe._require(
            type(meta.get(key)) is int and meta[key] > 0, f"centering: invalid {key}"
        )
    probe._require(
        len(probe._flatten(meta.get("construction_robot_position_w_m"))) == 3,
        "centering: construction pose missing",
    )
    orientation = meta.get("construction_robot_orientation_wxyz")
    probe._require(
        probe._unavailable(orientation) or len(probe._flatten(orientation)) == 4,
        "centering: construction orientation missing",
    )
    probe._require(
        isinstance(meta.get("original_generator_class"), str)
        and meta["original_generator_class"],
        "centering: generator class missing",
    )
    probe._require(
        isinstance(meta.get("limitations"), list) and meta["limitations"],
        "centering: interpretation limits missing",
    )
    return meta


TRANSLATED_POSITION_HARD_CAP_M = 1e-5


def _float32_ulp(value):
    """Spacing of the float32 bin containing a finite captured world coordinate."""
    value = abs(value)
    probe._require(
        math.isfinite(value) and value <= (2.0 - 2.0**-23) * 2.0**127,
        "position outside finite float32 coordinate range",
    )
    exponent = math.frexp(value)[1] - 24 if value else -149
    return math.ldexp(1.0, max(-149, exponent))


def _translated_initial_positions(a, b, left_initial, right_initial, label):
    """Allow only a bounded, reported XYZ normalization precision budget.

    One float32 ULP per stored world coordinate and origin is a conservative
    diagnostic budget for world-space pose construction and origin subtraction.
    It is not a claim that all simulator state errors obey this bound.
    """
    left_origin, right_origin = a["environment_origin_w_m"], b["environment_origin_w_m"]
    translated = [x != y for x, y in zip(left_origin, right_origin)]
    origin_ulps = [
        [_float32_ulp(x) for x in origin] for origin in (left_origin, right_origin)
    ]
    result = {
        "status": "NUMERICALLY_CLOSE",
        "interpretation": "Numerical closeness of initial positions, not bitwise equality, equal hidden simulator state, or a causal verdict.",
        "base_abs_tolerance_m": probe.INITIAL_TOLERANCE,
        "translated_position_hard_cap_m": TRANSLATED_POSITION_HARD_CAP_M,
        "translated_axes_xyz": translated,
        "origin_float32_ulp_m": {"left": origin_ulps[0], "right": origin_ulps[1]},
        "rule": "Unchanged-origin axes retain 2e-6 m. Changed-origin axes use min(1e-5 m, 2e-6 m + ulp32(left world coordinate) + ulp32(right world coordinate) + ulp32(left origin) + ulp32(right origin)).",
        "non_position_initial_state_abs_tolerance": probe.INITIAL_TOLERANCE,
        "fields": {},
    }
    for field in ("root_transform_w_xyzw", "link_transform_w_xyzw"):
        left_raw = a["initial_authoritative_state"][field]
        right_raw = b["initial_authoritative_state"][field]
        if not isinstance(left_raw, list) or not isinstance(right_raw, list):
            probe._match(left_raw, right_raw, f"{label}.initial_state.{field}")
            result["fields"][field] = {
                "status": "UNAVAILABLE",
                "left": left_raw,
                "right": right_raw,
            }
            continue
        root = field.startswith("root_")
        raw_rows = ([left_raw], [right_raw]) if root else (left_raw, right_raw)
        local_rows = (
            ([left_initial[field]], [right_initial[field]])
            if root
            else (left_initial[field], right_initial[field])
        )
        probe._require(
            len(raw_rows[0]) == len(raw_rows[1]), f"{label}.{field}: pose count differs"
        )
        differences, tolerances, coordinate_ulps = [], [], []
        for index, (raw_left, raw_right, local_left, local_right) in enumerate(
            zip(*raw_rows, *local_rows, strict=True)
        ):
            probe._match(
                local_left[3:],
                local_right[3:],
                f"{label}.initial_state.{field}[{index}].quaternion",
                probe.INITIAL_TOLERANCE,
            )
            error_row, tolerance_row, ulp_row = [], [], []
            for axis in range(3):
                ulps = [_float32_ulp(raw_left[axis]), _float32_ulp(raw_right[axis])]
                budget = sum(ulps) + origin_ulps[0][axis] + origin_ulps[1][axis]
                tolerance = (
                    min(
                        TRANSLATED_POSITION_HARD_CAP_M, probe.INITIAL_TOLERANCE + budget
                    )
                    if translated[axis]
                    else probe.INITIAL_TOLERANCE
                )
                error = local_left[axis] - local_right[axis]
                probe._require(
                    abs(error) <= tolerance,
                    f"{label}.initial_state.{field}[{index}][{axis}]: position difference {error:g} m exceeds bounded tolerance {tolerance:g} m",
                )
                error_row.append(error)
                tolerance_row.append(tolerance)
                ulp_row.append(ulps)
            differences.append(error_row)
            tolerances.append(tolerance_row)
            coordinate_ulps.append(ulp_row)
        result["fields"][field] = {
            "row_names": ["root"] if root else a["body_names"],
            "left_minus_right_xyz_m": differences,
            "component_tolerances_xyz_m": tolerances,
            "world_coordinate_float32_ulp_m_left_right": coordinate_ulps,
            "per_axis_max_abs_difference_m": [
                max(abs(row[j]) for row in differences) for j in range(3)
            ],
            "per_axis_max_tolerance_m": [
                max(row[j] for row in tolerances) for j in range(3)
            ],
            "max_abs_difference_m": max(abs(x) for row in differences for x in row),
        }
    return result


def _same_original_physics(left, right, label, *, translation_rounding=False):
    """Keep the established input contract independent of new solver readings."""
    a, b = left["physical_metadata"], right["physical_metadata"]
    for key in ("joint_names", "body_names", "foot_names", "physics_dt_s"):
        probe._match(a[key], b[key], f"{label}.{key}")
    for key in probe.PHYSICAL_PROPERTY_WIDTHS:
        probe._match(
            a["properties"][key],
            b["properties"][key],
            f"{label}.properties.{key}",
            probe.TARGET_TOLERANCE,
        )

    def initial(value):
        return probe._physical_state(
            {
                key: item
                for key, item in value["initial_authoritative_state"].items()
                if key not in probe.SOLVER_STATE_FIELDS
            },
            value["environment_origin_w_m"],
        )

    left_initial, right_initial = initial(a), initial(b)
    if not translation_rounding:
        probe._match(
            left_initial,
            right_initial,
            f"{label}.initial_state",
            probe.INITIAL_TOLERANCE,
        )
        return None
    positions = {"root_transform_w_xyzw", "link_transform_w_xyzw"}
    probe._match(
        {k: v for k, v in left_initial.items() if k not in positions},
        {k: v for k, v in right_initial.items() if k not in positions},
        f"{label}.initial_state",
        probe.INITIAL_TOLERANCE,
    )
    return _translated_initial_positions(a, b, left_initial, right_initial, label)


def _solver_comparison(left, right):
    return {
        "first_pre_physics": probe.solver_input_differences(
            {
                k: left["physics_substeps"][0]["pre"].get(k)
                for k in probe.SOLVER_STATE_FIELDS
            },
            {
                k: right["physics_substeps"][0]["pre"].get(k)
                for k in probe.SOLVER_STATE_FIELDS
            },
        ),
        "properties": probe.solver_input_differences(
            {
                k: left["physical_metadata"]["properties"].get(k)
                for k in probe.SOLVER_PROPERTY_FIELDS
            },
            {
                k: right["physical_metadata"]["properties"].get(k)
                for k in probe.SOLVER_PROPERTY_FIELDS
            },
        ),
    }


def compare_centered_reports(
    control, flat, obstacle, old_flat, old_obstacle, reference, *, reference_sha256
):
    old = probe.compare_probe_reports(
        old_flat, old_obstacle, reference, reference_sha256=reference_sha256
    )
    for label, value in (
        ("old flat", old_flat),
        ("old obstacle", old_obstacle),
        ("new control", control),
    ):
        probe._require(
            "scene_intervention" not in value["metadata"],
            f"{label}: must be an uncentered replay",
        )
    probe._require(
        control["metadata"]["terrain_family"] == "high_step"
        and control["metadata"]["difficulty_level"] == 6,
        "fresh control must use high_step L6",
    )
    probe._validate_probe(
        control, reference, reference_sha256, "new uncentered control"
    )
    centered = probe.compare_probe_reports(
        flat, obstacle, reference, reference_sha256=reference_sha256
    )
    _same_original_physics(control, old_obstacle, "fresh control vs previous replay")
    a, b = _centering(flat, 0), _centering(obstacle, 6)
    initial_pose_comparison = _same_original_physics(
        obstacle, control, "centered L6 vs fresh control", translation_rounding=True
    )
    for key in (
        "source_mesh_sha256",
        "mesh_vertex_count",
        "mesh_face_count",
        "original_generator_class",
        "construction_robot_position_w_m",
        "construction_robot_orientation_wxyz",
    ):
        probe._match(a.get(key), b.get(key), f"centering source.{key}")
    for value, original, label in ((a, old_flat, "L0"), (b, old_obstacle, "L6")):
        probe._match(
            value["source_origin_w_m"],
            original["physical_metadata"]["environment_origin_w_m"],
            f"{label} source origin",
            2e-6,
        )
    # Require the new capture boundary to exist; unavailable APIs stay visible.
    for label, value in (("control", control), ("flat", flat), ("obstacle", obstacle)):
        state = value["physics_substeps"][0]["pre"]
        for key in probe.SOLVER_STATE_FIELDS:
            probe._require(key in state, f"{label}: missing newly instrumented {key}")
        # Later optional probes may add fields (e.g. a legacy friction getter).
        # They must not invalidate captures made with the original boundary.
        for key in ("joint_friction_static_dynamic_viscous",):
            probe._require(
                key in value["physical_metadata"]["properties"],
                f"{label}: missing newly instrumented {key}",
            )
    comparisons = {
        "fresh_uncentered_L6_minus_original_policy": _outcomes(control, reference),
        "fresh_uncentered_L6_minus_previous_replay": _outcomes(control, old_obstacle),
        "centered_L0_minus_previous_L0": _outcomes(flat, old_flat),
        "centered_L6_minus_fresh_uncentered_L6": _outcomes(obstacle, control),
    }
    exact = all(
        field["max_abs_difference"] == 0.0
        for name, fields in comparisons.items()
        if name.startswith("fresh_")
        for field in fields.values()
    )
    return {
        "kind": "startup_centered_scene_comparison",
        "schema_version": 1,
        "evidence_status": "VALID_DIAGNOSTIC"
        if exact
        else "CONTROL_REPRODUCTION_MISMATCH",
        "reference_sha256": reference_sha256,
        "checkpoint_sha256": reference["metadata"]["checkpoint_sha256"],
        "fresh_control_recorded_outcomes_exact": exact,
        "new_control_steps_total": 30,
        "new_simulated_duration_s": 30 * reference["metadata"]["step_dt_s"],
        "scene_interventions": {"L0": a, "L6": b},
        "initial_pose_centered_L6_vs_fresh_uncentered_L6": initial_pose_comparison,
        "control_step_comparisons": comparisons,
        "physics_substep_centered_L6_minus_fresh_uncentered_L6": probe._substep_outcomes(
            obstacle, control
        ),
        "solver_centered_L6_minus_fresh_uncentered_L6": _solver_comparison(
            obstacle, control
        ),
        "previous_pair": old,
        "centered_pair": centered,
        "limitations": [
            "A valid diagnostic is not a policy, traversal, pivot or operator acceptance PASS.",
            "Interpret centering only if the fresh uncentered control reproduces; a mismatch reopens runtime/instrumentation effects.",
            "This jointly changes placement, initialization geometry exposure and cooking coordinates, not only world-coordinate arithmetic.",
            "Convergence does not promote recentering to a robot repair; divergence does not alone prove a PhysX defect.",
            "Backend actuation readback is the submitted command, not net constraint torque; inspect solver-input deltas and unavailable readings.",
            "Previous traces lack the new force-delivery and generalized-dynamics fields; missing data are not evidence of equal physics.",
            "Only centered-versus-uncentered initial root/link XYZ uses a reported float32-coordinate rounding budget capped at 10 micrometers; same-origin axes, quaternions, joints and other initial fields retain the original strict tolerance. Numerical closeness is not equality or a causal explanation.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_probe", type=Path)
    parser.add_argument("--original-probe", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args(argv)

    def read(path):
        return json.loads(path.read_bytes(), object_pairs_hook=probe._unique_object)

    try:
        raw = args.reference.read_bytes()
        result = compare_centered_reports(
            *[
                read(args.new_probe / name / "startup_diagnostics.json")
                for name in ("uncentered_L6", "centered_L0", "centered_L6")
            ],
            *[
                read(
                    args.original_probe / f"level_{level}" / "startup_diagnostics.json"
                )
                for level in (0, 6)
            ],
            json.loads(raw, object_pairs_hook=probe._unique_object),
            reference_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result = {
            "kind": "startup_centered_scene_comparison",
            "evidence_status": "INVALID_DIAGNOSTIC",
            "error": str(error),
        }
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["evidence_status"] == "VALID_DIAGNOSTIC" else 2


if __name__ == "__main__":
    raise SystemExit(main())
