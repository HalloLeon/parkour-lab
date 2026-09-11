"""Gate a solver-only replay experiment, not simulator truth or robot acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from . import startup_probe_report as probe
    from . import startup_friction_report as friction
    from . import startup_solver_probe as solver
    from .startup_centered_report import _outcomes, _same_original_physics
except ImportError:
    import startup_probe_report as probe
    import startup_friction_report as friction
    import startup_solver_probe as solver
    from startup_centered_report import _outcomes, _same_original_physics


def _zero_friction(trace):
    properties = trace["physical_metadata"]["properties"]
    for name, zeros in (
        ("joint_legacy_friction_coefficient", [0.0] * 12),
        ("joint_friction_static_dynamic_viscous", [[0.0] * 3 for _ in range(12)]),
    ):
        probe._match(properties.get(name), zeros, f"independent {name} zero readback")


def _solver_metadata(trace, source, requested):
    meta = trace["metadata"]
    probe._require(
        "scene_intervention" not in meta and "legacy_friction_probe" not in meta,
        "Solver replay must not include centering or friction intervention",
    )
    value = probe._field(meta, "solver_probe", "trace.metadata")
    solver.validate_solver_metadata(value, require_readback=True)
    probe._match(value["requested_solver"], requested, "requested solver")
    solver.validate_solver_environment_physics(
        source["metadata"]["environment_physics"], meta["environment_physics"], value
    )
    _zero_friction(trace)
    return value


def _same_start(trace, original, label):
    _same_original_physics(trace, original, label)
    probe._match(
        trace["physical_metadata"]["environment_origin_w_m"],
        original["physical_metadata"]["environment_origin_w_m"],
        f"{label}.unchanged world placement",
    )
    first, old_first = (r["physics_substeps"][0]["pre"] for r in (trace, original))
    probe._match(
        {k: v for k, v in first.items() if k not in probe.SOLVER_STATE_FIELDS},
        {k: v for k, v in old_first.items() if k not in probe.SOLVER_STATE_FIELDS},
        f"{label}.first_pre original physical state",
        probe.INITIAL_TOLERANCE,
    )
    for field in (
        "joint_position_rad",
        "joint_velocity_rad_s",
        "joint_computed_torque_nm",
        "joint_applied_torque_nm",
    ):
        probe._match(
            trace["physics_substeps"][0]["pre"][field],
            original["physics_substeps"][0]["pre"][field],
            f"{label}.first_pre.{field}",
        )


def _same_first_torques(trace, baseline):
    for key in (
        "joint_computed_torque_nm",
        "joint_applied_torque_nm",
        "joint_submitted_effort_nm",
        "joint_physx_actuation_force_nm",
    ):
        left, right = (
            value["physics_substeps"][0]["pre"].get(key) for value in (trace, baseline)
        )
        probe._numeric_shape(left, (12,), f"first torque {key}")
        probe._numeric_shape(right, (12,), f"baseline first torque {key}")
        probe._match(left, right, f"unchanged first torque {key}")


def solver_preflight(
    native, baseline, old_flat, old_obstacle, source, *, reference_sha256
):
    for name, trace in (
        ("source", source),
        ("old L0", old_flat),
        ("old L6", old_obstacle),
        ("friction baseline", baseline),
    ):
        probe._require(
            "solver_probe" not in trace["metadata"]
            and "scene_intervention" not in trace["metadata"],
            f"{name}: requires original uncentered TGS evidence",
        )
    old_pair = probe.compare_probe_reports(
        old_flat, old_obstacle, source, reference_sha256=reference_sha256
    )
    probe._validate_probe(baseline, source, reference_sha256, "friction baseline")
    probe._match(baseline["metadata"]["difficulty_level"], 6, "baseline L6")
    observed = friction._metadata(baseline, "observe")
    probe._match(observed["status"], "OBSERVED", "baseline friction availability")
    probe._match(
        observed["declared_actuator_friction"], [0.0] * 12, "declared zero friction"
    )
    _zero_friction(baseline)
    probe._validate_probe(
        native, source, reference_sha256, "fresh native", allow_solver_probe=True
    )
    probe._match(native["metadata"]["difficulty_level"], 6, "native L6")
    metadata = _solver_metadata(native, source, "TGS")
    for trace, original, label in (
        (baseline, old_obstacle, "friction baseline vs old replay"),
        (native, baseline, "fresh native vs friction baseline"),
    ):
        _same_start(trace, original, label)
        probe._match(
            [
                {
                    k: v
                    for k, v in row["post"].items()
                    if k not in probe.SOLVER_STATE_FIELDS
                }
                for row in trace["physics_substeps"]
            ],
            [
                {
                    k: v
                    for k, v in row["post"].items()
                    if k not in probe.SOLVER_STATE_FIELDS
                }
                for row in original["physics_substeps"]
            ],
            f"{label}.exact original post-substep reproduction",
        )
    _same_first_torques(native, baseline)
    for name, value in (
        ("source", source),
        ("previous L6", old_obstacle),
        ("friction baseline", baseline),
    ):
        probe._match(
            [s["post"] for s in native["samples"]],
            [s["post"] for s in value["samples"][: probe.PREFIX_STEPS]],
            f"fresh native exact post-state reproduction of {name}",
        )
    return {
        "kind": "startup_solver_preflight",
        "schema_version": 1,
        "evidence_status": "READY_FOR_PGS",
        "reference_sha256": reference_sha256,
        "checkpoint_sha256": source["metadata"]["checkpoint_sha256"],
        "native_recorded_post_states_exact": True,
        "solver": metadata,
        "previous_pair": old_pair,
        "limitations": [
            "Reproduction and readbacks authorize a bounded solver experiment, not a physics-bug verdict or robot PASS."
        ],
    }


def _first_difference(left, right, field):
    rows = [
        probe._physical_state(
            r["physics_substeps"][0]["post"],
            r["physical_metadata"]["environment_origin_w_m"],
        )[field]
        for r in (left, right)
    ]
    return probe._delta_stats([rows[0]], [rows[1]])


def _gate(flat, obstacle, old_flat, old_obstacle):
    joints = {}
    checks = []
    for field, cap in (("joint_position_rad", 0.001), ("joint_velocity_rad_s", 0.1)):
        previous = _first_difference(old_flat, old_obstacle, field)[
            "max_abs_difference"
        ]
        current = _first_difference(flat, obstacle, field)["max_abs_difference"]
        passed = previous > 0 and current <= cap and current <= 0.1 * previous
        checks.append(passed)
        joints[field] = {
            "previous_max_abs_difference": previous,
            "pgs_max_abs_difference": current,
            "required_reduction_fraction": 0.9,
            "observed_reduction_fraction": 1 - current / previous if previous else None,
            "absolute_cap": cap,
            "passed": passed,
        }
    motion = {}
    for name, trace in (("L0", flat), ("L6", obstacle)):
        step = trace["physics_substeps"][0]
        delta = max(
            abs(a - b)
            for a, b in zip(
                step["post"]["joint_position_rad"], step["pre"]["joint_position_rad"]
            )
        )
        speed = max(map(abs, step["post"]["joint_velocity_rad_s"]))
        torque = max(map(abs, step["pre"]["joint_physx_actuation_force_nm"]))
        passed = delta >= 1e-4 and speed >= 0.1 and torque > 0
        checks.append(passed)
        motion[name] = {
            "max_abs_joint_displacement_rad": delta,
            "max_abs_joint_velocity_rad_s": speed,
            "max_abs_first_actuation_nm": torque,
            "minimum_displacement_rad": 1e-4,
            "minimum_speed_rad_s": 0.1,
            "passed": passed,
        }
    left, right = [
        probe._physical_state(
            r["physics_substeps"][0]["post"],
            r["physical_metadata"]["environment_origin_w_m"],
        )
        for r in (flat, obstacle)
    ]
    for state in (left, right):
        probe._numeric_shape(state["root_transform_w_xyzw"], (7,), "first root pose")
        probe._numeric_shape(
            state["root_com_velocity_w_m_s_rad_s"], (6,), "first root velocity"
        )
    pose = [s["root_transform_w_xyzw"] for s in (left, right)]
    velocity = [s["root_com_velocity_w_m_s_rad_s"] for s in (left, right)]
    position_error = max(abs(a - b) for a, b in zip(pose[0][:3], pose[1][:3]))
    velocity_error = max(abs(a - b) for a, b in zip(velocity[0][:3], velocity[1][:3]))
    quaternions = [p[3:] for p in pose]
    norms = [math.sqrt(sum(v * v for v in q)) for q in quaternions]
    probe._require(
        all(abs(n - 1) <= 1e-3 for n in norms), "Nonphysical first root quaternion"
    )
    dot = sum(a * b for a, b in zip(*quaternions)) / (norms[0] * norms[1])
    root_passed = position_error <= 1e-4 and velocity_error <= 0.01
    checks.append(root_passed)
    return {
        "passed": all(checks),
        "time_after_s": flat["physics_substeps"][0]["time_after_s"],
        "joint_pair": joints,
        "non_frozen_response": motion,
        "root_pair": {
            "max_abs_position_difference_m": position_error,
            "position_cap_m": 1e-4,
            "max_abs_linear_velocity_difference_m_s": velocity_error,
            "linear_velocity_cap_m_s": 0.01,
            "orientation_geodesic_difference_rad": 2 * math.acos(min(1.0, abs(dot))),
            "max_abs_angular_velocity_difference_rad_s": max(
                abs(a - b) for a, b in zip(velocity[0][3:], velocity[1][3:])
            ),
            "passed": root_passed,
        },
        "interpretation": "Engineering triage thresholds only, not physical-truth, stability or robot-reliability bounds. No velocity finite-difference consistency claim.",
    }


def compare_solver_reports(
    native,
    flat,
    obstacle,
    baseline,
    old_flat,
    old_obstacle,
    source,
    *,
    reference_sha256,
):
    control = solver_preflight(
        native,
        baseline,
        old_flat,
        old_obstacle,
        source,
        reference_sha256=reference_sha256,
    )
    pair = probe.compare_probe_reports(
        flat,
        obstacle,
        source,
        reference_sha256=reference_sha256,
        allow_solver_probe=True,
    )
    for trace, original, label in (
        (flat, old_flat, "PGS L0"),
        (obstacle, old_obstacle, "PGS L6"),
    ):
        _solver_metadata(trace, source, "PGS")
        _same_start(trace, original, label)
        _same_first_torques(trace, baseline)
    gate = _gate(flat, obstacle, old_flat, old_obstacle)
    return {
        "kind": "startup_solver_comparison",
        "schema_version": 1,
        "evidence_status": "READY_FOR_POLICY_CHECK"
        if gate["passed"]
        else "INSUFFICIENT_SOLVER_EFFECT",
        "reference_sha256": reference_sha256,
        "checkpoint_sha256": source["metadata"]["checkpoint_sha256"],
        "control": control,
        "first_physics_step_gate": gate,
        "pgs_pair": pair,
        "pgs_L6_minus_native_L6": _outcomes(obstacle, native),
        "first_action_pgs_L6_minus_native_L6": probe._substep_outcomes(
            obstacle, native
        ),
        "first_step_contacts": {
            name: r["physics_substeps"][0]["post"]["contacts"]
            for name, r in (("PGS_L0", flat), ("PGS_L6", obstacle))
        },
        "limitations": [
            "PGS is an explicit diagnostic intervention, not a default solver repair.",
            "First-step agreement only permits the next normal-policy check; ten-step open-loop trajectories need not match and do not establish useful feedback control.",
            "Native reproduction, source hash, action delivery, placement and captured friction are independently gated; hidden solver state is not proven equal.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--original-probe", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    hashes = {}

    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw, object_pairs_hook=probe._unique_object)

    try:
        source, baseline = read(args.reference), read(args.baseline)
        native = read(args.probe_dir / "native_L6/startup_diagnostics.json")
        old_flat, old_obstacle = [
            read(args.original_probe / f"level_{level}/startup_diagnostics.json")
            for level in (0, 6)
        ]
        kwargs = {"reference_sha256": hashes[str(args.reference)]}
        if args.preflight:
            result = solver_preflight(
                native, baseline, old_flat, old_obstacle, source, **kwargs
            )
        else:
            flat, obstacle = [
                read(args.probe_dir / f"pgs_L{level}/startup_diagnostics.json")
                for level in (0, 6)
            ]
            result = compare_solver_reports(
                native,
                flat,
                obstacle,
                baseline,
                old_flat,
                old_obstacle,
                source,
                **kwargs,
            )
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result = {
            "kind": "startup_solver_comparison",
            "evidence_status": "INVALID_DIAGNOSTIC",
            "error": str(error),
        }
    result["input_sha256"] = hashes
    print(json.dumps(result, indent=2, allow_nan=False))
    return (
        0
        if result["evidence_status"] in ("READY_FOR_PGS", "READY_FOR_POLICY_CHECK")
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
