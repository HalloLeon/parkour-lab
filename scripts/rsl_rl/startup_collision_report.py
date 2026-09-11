"""Read a four-case terrain-collision bisection; never certify locomotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from . import startup_probe_report as probe
    from .startup_centered_report import _same_original_physics
    from .startup_collision_probe import validate_collision_metadata
except ImportError:
    import startup_probe_report as probe
    from startup_centered_report import _same_original_physics
    from startup_collision_probe import validate_collision_metadata


CASES = ("native_L0", "native_L6", "ground_off_L0", "ground_off_L6")
CAPS = {
    "joint_position_rad": 1e-4,
    "joint_velocity_rad_s": 0.01,
    "root_transform_w_xyzw": 1e-5,
    "root_com_velocity_w_m_s_rad_s": 0.01,
}


def _read(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=probe._unique_object)


def _receipt(raw):
    return {
        "kind": "startup_collision_completion",
        "schema_version": 1,
        "status": "CAPTURE_AND_ENVIRONMENT_CLEANUP_COMPLETE",
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_completion(directory):
    """Written only on the normal path after env.close, before App.close."""
    directory = Path(directory)
    raw = (directory / "startup_diagnostics.json").read_bytes()
    with (directory / "completion.json").open("x", encoding="utf-8") as stream:
        json.dump(_receipt(raw), stream, allow_nan=False)


def read_completed(directory):
    directory = Path(directory)
    raw = (directory / "startup_diagnostics.json").read_bytes()
    probe._match(
        _read(directory / "completion.json"), _receipt(raw), "completion receipt"
    )
    return json.loads(raw, object_pairs_hook=probe._unique_object)


def _validate(trace, source, digest, level, mode):
    probe._validate_probe(
        trace,
        source,
        digest,
        f"{mode} L{level}",
        allow_collision_probe=True,
        prefix_steps=1,
    )
    meta = trace["metadata"]
    for key, value in (
        ("terrain_family", "high_step"),
        ("difficulty_level", level),
        ("command_profile", "translation_only"),
        ("geometry_variant", 0),
    ):
        probe._match(meta.get(key), value, key)
    probe._require(
        not any(
            key in meta
            for key in ("scene_intervention", "solver_probe", "legacy_friction_probe")
        ),
        "Collision bisection cannot mix interventions",
    )
    evidence = meta.get("ground_collision_probe")
    validate_collision_metadata(evidence)
    probe._match(evidence["mode"], mode, "collision selection")
    # An equal UNAVAILABLE marker is not evidence of equal robot properties.
    for name in probe.PHYSICAL_PROPERTY_WIDTHS:
        probe._flatten(trace["physical_metadata"]["properties"][name])
    first = trace["physics_substeps"][0]["pre"]
    for key in ("joint_submitted_effort_nm", "joint_physx_actuation_force_nm"):
        probe._numeric_shape(first.get(key), (12,), key)
    last = None
    for step in trace["physics_substeps"]:
        clocks = [step.get(key) for key in ("clock_before", "clock_after")]
        for clock in clocks:
            probe._require(
                isinstance(clock, dict)
                and type(clock.get("step_index")) is int
                and clock["step_index"] >= 0
                and type(clock.get("time_s")) in (int, float)
                and math.isfinite(clock["time_s"]),
                "Missing actual simulation clock",
            )
        before, after = clocks
        probe._require(
            after["step_index"] == before["step_index"] + 1
            and abs(after["time_s"] - before["time_s"] - 0.005) <= 1e-8,
            "Expected exactly one measured 5ms physics step (10ns clock tolerance)",
        )
        if last is not None:
            probe._match(before, last, "No hidden steps between measured substeps")
        last = after
    return evidence


def _same_start(left, right, label):
    _same_original_physics(left, right, label)
    probe._match(
        left["physical_metadata"]["environment_origin_w_m"],
        right["physical_metadata"]["environment_origin_w_m"],
        f"{label} origin",
    )
    for name in probe.SOLVER_PROPERTY_FIELDS:
        a, b = (t["physical_metadata"]["properties"].get(name) for t in (left, right))
        # Older references may lack newly instrumented friction fields. Fresh
        # ground-off/native comparisons below require both and exact equality.
        if a is not None and b is not None:
            probe._match(a, b, f"{label} {name}", 1e-6)
    for key in (
        "joint_position_rad",
        "joint_velocity_rad_s",
        "joint_computed_torque_nm",
        "joint_applied_torque_nm",
        "joint_submitted_effort_nm",
        "joint_physx_actuation_force_nm",
    ):
        a, b = (t["physics_substeps"][0]["pre"].get(key) for t in (left, right))
        if a is not None and b is not None:
            probe._match(a, b, f"{label} first {key}", 1e-6)
    # Require actual pre-physics root/link states, not only pre-inference buffers.
    for key in (
        "root_transform_w_xyzw",
        "root_com_velocity_w_m_s_rad_s",
        "link_transform_w_xyzw",
        "link_com_velocity_w_m_s_rad_s",
    ):
        probe._match(
            left["physics_substeps"][0]["pre"][key],
            right["physics_substeps"][0]["pre"][key],
            f"{label} first {key}",
            probe.INITIAL_TOLERANCE,
        )


def _difference(left, right):
    states = [
        probe._physical_state(
            t["physics_substeps"][0]["post"],
            t["physical_metadata"]["environment_origin_w_m"],
        )
        for t in (left, right)
    ]
    return {
        key: probe._delta_stats([states[0][key]], [states[1][key]])[
            "max_abs_difference"
        ]
        for key in CAPS
    }


def preflight(native, old, source, digest):
    probe.compare_probe_reports(*old, source, reference_sha256=digest)
    for level, new, prior in zip((0, 6), native, old, strict=True):
        _validate(new, source, digest, level, "native")
        _same_start(new, prior, f"native L{level} reproduction")
        # Instrumentation must reproduce the actual recorded four substeps and
        # first control transition, not just preserve configuration metadata.
        for index in range(4):
            for key, value in prior["physics_substeps"][index]["post"].items():
                if key not in probe.SOLVER_STATE_FIELDS:
                    probe._match(
                        new["physics_substeps"][index]["post"].get(key),
                        value,
                        f"native L{level} substep {index} reproduction",
                    )
        probe._match(
            new["samples"][0]["post"],
            prior["samples"][0]["post"],
            f"native L{level} first transition",
        )
    for key in (
        "joint_computed_torque_nm",
        "joint_applied_torque_nm",
        "joint_submitted_effort_nm",
        "joint_physx_actuation_force_nm",
    ):
        probe._match(
            native[0]["physics_substeps"][0]["pre"][key],
            native[1]["physics_substeps"][0]["pre"][key],
            f"native pair identical first {key}",
            1e-6,
        )
    split = _difference(*native)
    probe._require(
        split["joint_position_rad"] > CAPS["joint_position_rad"]
        and split["joint_velocity_rad_s"] > CAPS["joint_velocity_rad_s"],
        "Original joint-state discrepancy not reproduced",
    )
    return {
        "kind": "startup_collision_preflight",
        "schema_version": 1,
        "evidence_status": "READY_FOR_GROUND_OFF",
        "native_first_step_difference": split,
        "reference_sha256": digest,
        "native_reproduction_exact": True,
    }


def compare(cases, old, source, digest):
    native, disabled = cases[:2], cases[2:]
    control = preflight(native, old, source, digest)
    interventions = {}
    motion = {}
    for level, trace, baseline in zip((0, 6), disabled, native, strict=True):
        evidence = _validate(trace, source, digest, level, "ground_off")
        _same_start(trace, baseline, f"ground off L{level}")
        probe._match(
            evidence["before"],
            baseline["metadata"]["ground_collision_probe"]["before"],
            f"L{level} identical ground geometry/offsets before edit",
        )
        probe._match(
            evidence["enabled_robot_collider_count"],
            baseline["metadata"]["ground_collision_probe"][
                "enabled_robot_collider_count"
            ],
            "robot collision count",
        )
        for key in probe.SOLVER_PROPERTY_FIELDS:
            a, b = (
                t["physical_metadata"]["properties"].get(key) for t in (trace, baseline)
            )
            probe._flatten(a)
            probe._flatten(b)
            probe._match(a, b, f"L{level} unchanged friction {key}")
        for substep in trace["physics_substeps"]:
            for when in ("pre", "post"):
                for sensor, data in substep[when]["contacts"].items():
                    probe._require(
                        max(map(abs, probe._flatten(data["net_force_w_n"]))) <= 1e-6,
                        f"Unexpected ground-off contact: {sensor}",
                    )
        first = trace["physics_substeps"][0]
        displacement = max(
            abs(a - b)
            for a, b in zip(
                first["post"]["joint_position_rad"],
                first["pre"]["joint_position_rad"],
                strict=True,
            )
        )
        speed = max(map(abs, first["post"]["joint_velocity_rad_s"]))
        effort = max(map(abs, first["pre"]["joint_physx_actuation_force_nm"]))
        probe._require(
            displacement >= 1e-4 and speed >= 0.1 and effort > 0,
            "Ground-off result has inadequate response; do not mistake a frozen robot for agreement",
        )
        motion[f"L{level}"] = {
            "displacement_rad": displacement,
            "speed_rad_s": speed,
            "effort_nm": effort,
        }
        interventions[f"L{level}"] = _difference(trace, baseline)
    difference = _difference(*disabled)
    closed = all(
        difference[key] <= cap
        and difference[key] <= 0.1 * control["native_first_step_difference"][key]
        for key, cap in CAPS.items()
    )
    return {
        "kind": "startup_collision_comparison",
        "schema_version": 1,
        "evidence_status": "COLLISION_SCENE_DEPENDENCE_OBSERVED"
        if closed
        else "GROUND_REMOVAL_INSUFFICIENT",
        "reference_sha256": digest,
        "preflight": control,
        "ground_off_first_step_difference": difference,
        "absolute_caps": CAPS,
        "required_reduction_fraction": 0.9,
        "first_step_change_from_native": interventions,
        "ground_off_nonzero_response": motion,
        "physics_substep_ground_off_L0_minus_L6": probe._substep_outcomes(*disabled),
        "limitations": [
            "No ground support: neither outcome is a locomotion repair or operator acceptance.",
            "Convergence implicates ground collision participation/initialization/topology jointly, not an isolated PhysX bug.",
            "Persistence does not exonerate all contact or reset behavior; it rejects sufficiency of this intervention.",
            "Only the first 5ms contrast has matched initial effort; subsequent PD efforts may diverge.",
            "No policy weights, gains, solver, reward or production defaults are changed.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--original-probe", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = args.reference.read_bytes()
        source = json.loads(raw, object_pairs_hook=probe._unique_object)
        digest = hashlib.sha256(raw).hexdigest()
        old = [
            _read(args.original_probe / f"level_{level}/startup_diagnostics.json")
            for level in (0, 6)
        ]
        cases = [
            read_completed(args.directory / case)
            for case in (CASES[:2] if args.preflight else CASES)
        ]
        result = (
            preflight(cases, old, source, digest)
            if args.preflight
            else compare(cases, old, source, digest)
        )
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result = {
            "kind": "startup_collision_comparison",
            "evidence_status": "INVALID_DIAGNOSTIC",
            "error": str(error),
        }
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result["evidence_status"] == "INVALID_DIAGNOSTIC" else 0


if __name__ == "__main__":
    raise SystemExit(main())
