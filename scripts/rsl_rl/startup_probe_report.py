"""Compare a matched-action startup probe; never certify control or causation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from .failure_trace_report import validate_failure_trace
except ImportError:
    from failure_trace_report import validate_failure_trace


PREFIX_STEPS = 10
INITIAL_TOLERANCE = 2e-6
TARGET_TOLERANCE = 1e-6
INITIAL_FIELDS = (
    "root_position_env_m",
    "root_orientation_wxyz",
    "linear_velocity_body_m_s",
    "linear_velocity_w_m_s",
    "angular_velocity_body_rad_s",
    "angular_velocity_w_rad_s",
    "joint_position_rad",
    "joint_velocity_rad_s",
    "foot_position_env_m",
    "foot_linear_velocity_w_m_s",
    "joint_position_target_rad",
    "joint_default_position_rad",
    "joint_soft_position_limits_rad",
    "safe_joint_target_limits_rad",
)
TARGET_FIELDS = (
    "environment_action",
    "delayed_raw_action",
    "affine_joint_target_rad",
    "configured_clip_joint_target_rad",
    "processed_joint_target_rad",
    "joint_position_target_rad",
    "reconstructed_safe_joint_target_rad",
)
OUTCOME_FIELDS = (
    "joint_position_rad",
    "joint_velocity_rad_s",
    "root_position_env_m",
    "linear_velocity_body_m_s",
    "linear_velocity_w_m_s",
    "angular_velocity_body_rad_s",
    "angular_velocity_w_rad_s",
    "foot_position_env_m",
    "foot_force_w_n",
    "joint_computed_torque_nm",
    "joint_applied_torque_nm",
)
PHYSICS_FIELDS = ("dt", "gravity", "physx", "physics_material", "use_fabric")
PHYSICAL_PROPERTY_WIDTHS = {
    "body_mass_kg": "body",
    "body_inertia_kg_m2": "body9",
    "body_com_pose": "body7",
    "shape_material_static_dynamic_restitution": "shape3",
    "joint_armature": 12,
    "joint_physx_stiffness": 12,
    "joint_physx_damping": 12,
    "joint_max_velocity_rad_s": 12,
    "joint_max_force_nm": 12,
    "joint_position_limits_rad": 24,
    "joint_operative_stiffness": 12,
    "joint_operative_damping": 12,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _flatten(value):
    if isinstance(value, list):
        return [number for item in value for number in _flatten(item)]
    _require(
        type(value) in (int, float) and math.isfinite(value),
        "expected finite numeric array",
    )
    return [value]


def _match(left, right, label, tolerance=0.0):
    """Compare complete structured evidence, rejecting missing keys/shapes."""
    if isinstance(left, dict) and isinstance(right, dict):
        _require(left.keys() == right.keys(), f"{label}: fields differ")
        for key in left:
            _match(left[key], right[key], f"{label}.{key}", tolerance)
    elif isinstance(left, list) and isinstance(right, list):
        _require(len(left) == len(right), f"{label}: array lengths differ")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _match(a, b, f"{label}[{index}]", tolerance)
    elif type(left) in (int, float) and type(right) in (int, float):
        _require(
            math.isfinite(left)
            and math.isfinite(right)
            and abs(left - right) <= tolerance,
            f"{label}: numerical evidence differs beyond {tolerance:g}",
        )
    else:
        _require(type(left) is type(right) and left == right, f"{label}: values differ")


def _field(mapping, key, label):
    _require(isinstance(mapping, dict) and key in mapping, f"{label}.{key}: missing")
    return mapping[key]


def _unavailable(value):
    if isinstance(value, dict) and value.get("status") == "UNAVAILABLE":
        _require(
            isinstance(value.get("reason"), str) and value["reason"],
            "UNAVAILABLE field needs a reason",
        )
        return True
    return False


def _validate_authoritative_state(state, bodies, label):
    _require(isinstance(state, dict), f"{label}: expected state object")
    for key in (
        "joint_position_rad",
        "joint_velocity_rad_s",
        "joint_position_target_rad",
        "processed_joint_target_rad",
        "joint_computed_torque_nm",
        "joint_applied_torque_nm",
    ):
        _require(
            len(_flatten(_field(state, key, label))) == 12,
            f"{label}.{key}: expected 12 values",
        )
    for key, count in (
        ("root_transform_w_xyzw", 7),
        ("root_com_velocity_w_m_s_rad_s", 6),
        ("link_transform_w_xyzw", 7 * bodies),
        ("link_com_velocity_w_m_s_rad_s", 6 * bodies),
    ):
        value = _field(state, key, label)
        if not _unavailable(value):
            _require(
                len(_flatten(value)) == count,
                f"{label}.{key}: incompatible physical body map",
            )
    contacts = _field(state, "contacts", label)
    for sensor in ("feet_contact", "undesired_contact", "chassis_contact"):
        data = _field(contacts, sensor, label)
        if not _unavailable(data):
            names = _field(data, "body_names", label)
            _require(
                isinstance(names, list)
                and names
                and len(set(names)) == len(names)
                and all(isinstance(name, str) and name for name in names),
                f"{label}.{sensor}: invalid body map",
            )
            _require(
                len(_flatten(_field(data, "net_force_w_n", label))) == 3 * len(names),
                f"{label}.{sensor}: force/body shape mismatch",
            )


def _delta_stats(left, right):
    left_rows, right_rows = list(map(_flatten, left)), list(map(_flatten, right))
    _require(
        len(left_rows) == len(right_rows) and len(left_rows) > 0,
        "outcome count differs",
    )
    deltas = []
    for a, b in zip(left_rows, right_rows, strict=True):
        _require(len(a) == len(b), "outcome shapes differ")
        deltas.append([x - y for x, y in zip(a, b, strict=True)])
    return {
        "first_post_difference": deltas[0],
        "first_post_max_abs_difference": max(map(abs, deltas[0])),
        "max_abs_difference": max(abs(x) for row in deltas for x in row),
        "per_component_max_abs_difference": [
            max(abs(row[j]) for row in deltas) for j in range(len(deltas[0]))
        ],
        "sample_count": len(deltas),
    }


def _physical_state(state, origin):
    """Normalize authoritative world transforms; retain quaternion xyzw order."""
    result = dict(state)
    for key in ("root_transform_w_xyzw", "link_transform_w_xyzw"):
        value = result.get(key)
        if isinstance(value, list):
            rows = [value] if key.startswith("root_") else value
            normalized = [
                [row[j] - origin[j] if j < 3 else row[j] for j in range(len(row))]
                for row in rows
            ]
            result[key] = normalized[0] if key.startswith("root_") else normalized
    return result


def _validate_probe(report, source, source_sha256, label):
    validate_failure_trace(report)
    meta, source_meta = report["metadata"], source["metadata"]
    _require(
        meta.get("action_source") == "recorded_action_replay",
        f"{label}: not a recorded-action replay",
    )
    replay = _field(meta, "action_replay", label)
    expected = {
        "kind": "startup_action_replay_probe",
        "action_source": "recorded_policy_action",
        "source_sha256": source_sha256,
        "source_checkpoint_sha256": source_meta["checkpoint_sha256"],
        "source_teacher_interface_sha256": source_meta["teacher_interface_sha256"],
        "prefix_steps": PREFIX_STEPS,
        "runtime_validated": True,
        "initial_state_validated": True,
        "initial_state_abs_tolerance": INITIAL_TOLERANCE,
        "target_abs_tolerance": TARGET_TOLERANCE,
    }
    for key, value in expected.items():
        _match(
            _field(replay, key, label + ".action_replay"),
            value,
            f"{label}.action_replay.{key}",
        )
    _require(
        isinstance(replay.get("source_path"), str) and replay["source_path"],
        f"{label}: source path missing",
    )
    _require(
        meta["max_steps"] == PREFIX_STEPS
        and len(report["samples"]) == PREFIX_STEPS
        and report["stop_reason"] == "step_limit",
        f"{label}: requires complete ten-step capture",
    )
    for key in (
        "checkpoint_sha256",
        "teacher_interface_sha256",
        "seed",
        "reset_profile",
        "geometry_variant",
        "step_dt_s",
        "policy_mode",
        "num_envs",
        "kit_args",
    ):
        _match(
            _field(meta, key, label + ".metadata"),
            _field(source_meta, key, "reference.metadata"),
            f"{label}.metadata.{key}",
        )
    for key in ("joint_names", "foot_names"):
        _match(
            meta["capture_metadata"][key],
            source_meta["capture_metadata"][key],
            f"{label}.{key}",
        )
    if "contact_body_names" in source_meta["capture_metadata"]:
        _match(
            _field(meta["capture_metadata"], "contact_body_names", label),
            source_meta["capture_metadata"]["contact_body_names"],
            f"{label}.contact_body_names",
        )
    for key in PHYSICS_FIELDS:
        _match(
            _field(_field(meta, "environment_physics", label), key, label),
            _field(
                _field(source_meta, "environment_physics", "reference"),
                key,
                "reference",
            ),
            f"{label}.physics.{key}",
            TARGET_TOLERANCE,
        )
    for report_meta, name in ((meta, label), (source_meta, "reference")):
        action = _field(
            _field(report_meta, "runtime_teacher_interface", name), "action", name
        )
        _match(
            action["joint_names"],
            report_meta["capture_metadata"]["joint_names"],
            f"{name}.action_joint_order",
        )
    _match(
        meta["runtime_teacher_interface"]["action"],
        source_meta["runtime_teacher_interface"]["action"],
        f"{label}.action_contract",
        TARGET_TOLERANCE,
    )
    for key in INITIAL_FIELDS:
        _match(
            _field(report["samples"][0]["pre"]["state"], key, label),
            _field(source["samples"][0]["pre"]["state"], key, "reference"),
            f"{label}.initial.{key}",
            INITIAL_TOLERANCE,
        )
    for index, (sample, original) in enumerate(
        zip(report["samples"], source["samples"][:PREFIX_STEPS], strict=True)
    ):
        _require(
            not sample["post"]["done"]
            and not any(sample["post"]["termination"].values()),
            f"{label}: terminated at step {index}",
        )
        _match(
            _field(sample, "replayed_action", label),
            original["policy_action"],
            f"{label}.replayed_action[{index}]",
            TARGET_TOLERANCE,
        )
        # Replay records the pre-wrapper policy vector. The unchanged wrapper
        # may legitimately clip it before ActionManager; compare each executed
        # pipeline stage to the source stage, not raw policy to manager action.
        for key in TARGET_FIELDS:
            _match(
                sample["post"]["state"][key],
                original["post"]["state"][key],
                f"{label}.{key}[{index}]",
                TARGET_TOLERANCE,
            )
        _match(
            sample["post"]["state"]["processed_joint_target_rad"],
            sample["post"]["state"]["joint_position_target_rad"],
            f"{label}.actual_target[{index}]",
            TARGET_TOLERANCE,
        )
    physical = _field(report, "physical_metadata", label)
    for key in ("joint_names", "foot_names"):
        _match(
            _field(physical, key, label),
            meta["capture_metadata"][key],
            f"{label}.physical.{key}",
        )
    bodies = _field(physical, "body_names", label)
    _require(
        isinstance(bodies, list)
        and bodies
        and len(set(bodies)) == len(bodies)
        and all(isinstance(name, str) and name for name in bodies),
        f"{label}: invalid physical body map",
    )
    for key in ("properties", "initial_authoritative_state", "frames"):
        _require(
            isinstance(physical.get(key), dict) and physical[key],
            f"{label}.physical.{key}: missing",
        )
    _require(
        isinstance(physical.get("unavailable"), dict),
        f"{label}.physical.unavailable: missing",
    )
    origin = _field(physical, "environment_origin_w_m", label)
    _require(len(_flatten(origin)) == 3, f"{label}: invalid environment origin")
    for key, count in PHYSICAL_PROPERTY_WIDTHS.items():
        value = _field(physical["properties"], key, label + ".physical.properties")
        if _unavailable(value):
            continue
        numbers = _flatten(value)
        if count == "shape3":
            _require(
                isinstance(value, list)
                and value
                and all(isinstance(row, list) and len(row) == 3 for row in value),
                f"{label}.{key}: expected per-shape material triples",
            )
        else:
            expected_count = {
                "body": len(bodies),
                "body9": len(bodies) * 9,
                "body7": len(bodies) * 7,
            }.get(count, count)
            _require(
                len(numbers) == expected_count,
                f"{label}.{key}: property/map shape mismatch",
            )
    _validate_authoritative_state(
        physical["initial_authoritative_state"],
        len(bodies),
        label + ".initial_authoritative",
    )
    initial = physical["initial_authoritative_state"]
    recorded_initial = report["samples"][0]["pre"]["state"]
    for key in ("joint_position_rad", "joint_velocity_rad_s"):
        _match(
            initial[key],
            recorded_initial[key],
            f"{label}.authoritative_vs_recorded.{key}",
            INITIAL_TOLERANCE,
        )
    if isinstance(initial["root_transform_w_xyzw"], list):
        root = initial["root_transform_w_xyzw"]
        _match(
            [root[j] - origin[j] for j in range(3)],
            recorded_initial["root_position_env_m"],
            f"{label}.authoritative_vs_recorded.root_position",
            INITIAL_TOLERANCE,
        )
        _match(
            [root[6], *root[3:6]],
            recorded_initial["root_orientation_wxyz"],
            f"{label}.authoritative_vs_recorded.root_orientation",
            INITIAL_TOLERANCE,
        )
    dt = _field(physical, "physics_dt_s", label)
    _require(
        type(dt) in (int, float) and dt > 0 and math.isfinite(dt),
        f"{label}: invalid physics timestep",
    )
    _match(dt, meta["environment_physics"]["dt"], f"{label}.physics_timestep", 1e-10)
    decimation = round(meta["step_dt_s"] / dt)
    _require(
        decimation == 4 and abs(meta["step_dt_s"] - decimation * dt) <= 1e-10,
        f"{label}: expected four physics substeps per control step",
    )
    substeps = _field(report, "physics_substeps", label)
    _require(
        isinstance(substeps, list) and len(substeps) == decimation,
        f"{label}: incomplete first-action physics substeps",
    )
    for index, substep in enumerate(substeps):
        step, subindex = divmod(index, decimation)
        _require(
            type(substep.get("control_step")) is int
            and substep["control_step"] == step
            and type(substep.get("physics_substep")) is int
            and substep["physics_substep"] == subindex,
            f"{label}: invalid physics substep indices",
        )
        for key, expected_time in (
            ("physics_dt_s", dt),
            ("time_before_s", index * dt),
            ("time_after_s", (index + 1) * dt),
        ):
            _match(
                _field(substep, key, label),
                expected_time,
                f"{label}.substep.{key}",
                1e-8,
            )
        for when in ("pre", "post"):
            state = _field(substep, when, label)
            _validate_authoritative_state(state, len(bodies), f"{label}.substep.{when}")
            for key in ("joint_position_target_rad", "processed_joint_target_rad"):
                _match(
                    state[key],
                    report["samples"][step]["post"]["state"][key],
                    f"{label}.substep.{when}.{key}",
                    TARGET_TOLERANCE,
                )
    return physical


def _substep_outcomes(left, right):
    """Summarize every shared available numeric leaf; expose unavailable fields."""
    results, unavailable = {}, {}
    left_states = [
        _physical_state(
            row["post"], left["physical_metadata"]["environment_origin_w_m"]
        )
        for row in left["physics_substeps"]
    ]
    right_states = [
        _physical_state(
            row["post"], right["physical_metadata"]["environment_origin_w_m"]
        )
        for row in right["physics_substeps"]
    ]

    def visit(a, b, prefix):
        if isinstance(a[0], dict) and isinstance(b[0], dict):
            if (
                a[0].get("status") == "UNAVAILABLE"
                or b[0].get("status") == "UNAVAILABLE"
            ):
                unavailable[prefix] = {"left": a[0], "right": b[0]}
                return
            _require(
                all(x.keys() == a[0].keys() for x in a + b),
                f"substep {prefix}: fields differ",
            )
            for key in a[0]:
                visit(
                    [x[key] for x in a],
                    [x[key] for x in b],
                    f"{prefix}.{key}".strip("."),
                )
        else:
            try:
                results[prefix] = _delta_stats(a, b)
            except ValueError:
                for x, y in zip(a, b, strict=True):
                    _match(x, y, f"substep {prefix}: nonnumeric metadata")

    visit(left_states, right_states, "")
    return {"post_physics": results, "unavailable": unavailable}


def compare_probe_reports(flat, obstacle, reference, *, reference_sha256):
    """Validate matching inputs, then report physical differences without a pass gate."""
    validate_failure_trace(reference)
    _require(
        len(reference["samples"]) >= PREFIX_STEPS
        and all(
            not s["post"]["done"] and not any(s["post"]["termination"].values())
            for s in reference["samples"][:PREFIX_STEPS]
        ),
        "reference: requires ten nonterminal source steps",
    )
    _require(
        reference["metadata"].get("action_source", "policy")
        in ("policy", "policy_action")
        and not any(
            key in reference["metadata"]
            for key in (
                "action_replay",
                "action_probe",
                "action_replay_probe",
                "startup_action_probe",
                "replay",
            )
        )
        and not any("replayed_action" in sample for sample in reference["samples"]),
        "reference must be the original policy trace",
    )
    for value, label, level in (
        (flat, "flat", 0),
        (obstacle, "obstacle", 6),
        (reference, "reference", 6),
    ):
        _require(
            value.get("metadata", {}).get("terrain_family") == "high_step"
            and value["metadata"].get("difficulty_level") == level,
            f"{label}: expected high_step level {level}",
        )
    left_physics = _validate_probe(flat, reference, reference_sha256, "flat")
    right_physics = _validate_probe(obstacle, reference, reference_sha256, "obstacle")
    for key in (
        "joint_names",
        "body_names",
        "foot_names",
        "properties",
        "unavailable",
        "frames",
        "physics_dt_s",
    ):
        _match(
            left_physics[key],
            right_physics[key],
            f"matched physical metadata.{key}",
            TARGET_TOLERANCE,
        )
    _match(
        _physical_state(
            left_physics["initial_authoritative_state"],
            left_physics["environment_origin_w_m"],
        ),
        _physical_state(
            right_physics["initial_authoritative_state"],
            right_physics["environment_origin_w_m"],
        ),
        "matched authoritative initial state",
        INITIAL_TOLERANCE,
    )
    comparisons = {}
    for name, left, right in (
        ("flat_minus_obstacle", flat, obstacle),
        ("obstacle_minus_reference", obstacle, reference),
    ):
        comparisons[name] = {
            key: _delta_stats(
                [s["post"]["state"][key] for s in left["samples"][:PREFIX_STEPS]],
                [s["post"]["state"][key] for s in right["samples"][:PREFIX_STEPS]],
            )
            for key in OUTCOME_FIELDS
        }
    return {
        "kind": "startup_action_probe_comparison",
        "schema_version": 1,
        "evidence_status": "VALID_PROBE",
        "reference_sha256": reference_sha256,
        "checkpoint_sha256": reference["metadata"]["checkpoint_sha256"],
        "steps": PREFIX_STEPS,
        "duration_s": PREFIX_STEPS * reference["metadata"]["step_dt_s"],
        "joint_names": left_physics["joint_names"],
        "foot_names": left_physics["foot_names"],
        "comparison_sign": "named left trace minus named right trace; flattened component order follows source fields",
        "tolerances": {
            "initial_state_abs": INITIAL_TOLERANCE,
            "target_abs_rad": TARGET_TOLERANCE,
        },
        "control_step_comparisons": comparisons,
        "physics_substep_flat_minus_obstacle": _substep_outcomes(flat, obstacle),
        "physical_properties_unavailable": left_physics["unavailable"],
        "limitations": [
            "VALID_PROBE/exit 0 validates this evidence contract, not robot safety or successful behavior.",
            "Physical outcome differences are observations, not an automatic physics-bug verdict; no outcome tolerance is a safety bound.",
            "Obstacle replay versus reference exposes same-course reproducibility before attributing flat/obstacle divergence to geometry.",
            "Reference lacks authoritative substeps/static-property capture; those are compared between new probes only.",
            "Authoritative 5ms substeps cover only the first control action; the ten-step comparison uses control-step snapshots.",
            "Current computed policy actions and commands may differ; only the recorded replay actions are applied.",
            "This ten-step open-loop probe is not a locomotion or operator acceptance test.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flat_trace", type=Path)
    parser.add_argument("obstacle_trace", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        source_bytes = args.reference.read_bytes()
        result = compare_probe_reports(
            json.loads(args.flat_trace.read_text(), object_pairs_hook=_unique_object),
            json.loads(
                args.obstacle_trace.read_text(), object_pairs_hook=_unique_object
            ),
            json.loads(source_bytes, object_pairs_hook=_unique_object),
            reference_sha256=hashlib.sha256(source_bytes).hexdigest(),
        )
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        if args.json:
            print(
                json.dumps(
                    {
                        "kind": "startup_action_probe_comparison",
                        "evidence_status": "INVALID_PROBE",
                        "error": str(error),
                    }
                )
            )
        else:
            print(f"INVALID_PROBE: {error}")
        return 2
    if args.json:
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(
            "VALID_PROBE: matched action evidence only; not a robot PASS or physics-bug verdict."
        )
        for comparison, fields in result["control_step_comparisons"].items():
            print(comparison + ":")
            for key in (
                "joint_position_rad",
                "joint_velocity_rad_s",
                "root_position_env_m",
                "linear_velocity_w_m_s",
            ):
                value = fields[key]
                print(
                    f"  {key}: first post max |delta|={value['first_post_max_abs_difference']:.6g}; ten-step max={value['max_abs_difference']:.6g}"
                )
        print(
            "Use --json for per-joint differences, forces, substeps and unavailable-property details."
        )
    return 0


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


if __name__ == "__main__":
    raise SystemExit(main())
