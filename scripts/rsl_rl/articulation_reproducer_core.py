"""Pure JSON contract for a one-step, contact-free Go2 placement diagnostic.

No simulator, policy, pickle, YAML constructor, or source-specified callable is
loaded here. A valid contrast is not a locomotion or simulator-accuracy test.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

try:
    from .failure_trace_report import validate_failure_trace
    from .startup_action_probe import _finite, _unique_object
    from .startup_probe_report import _match, _numeric_shape, _physical_state
except ImportError:
    from failure_trace_report import validate_failure_trace
    from startup_action_probe import _finite, _unique_object
    from startup_probe_report import _match, _numeric_shape, _physical_state


CASES = {"origin": (0.0, 0.0), "translated": (24.0, 24.0), "teleported": (0.0, 24.0)}
DT = 0.005
INITIAL_ATOL = 2e-6
POSITION_ATOL = 1e-5
EFFORT_ATOL = 1e-6
THRESHOLDS = {
    "joint_position_rad": 1e-4,
    "joint_velocity_rad_s": 0.01,
    "root_position_local_m": 1e-5,
    "root_com_velocity_w_m_s_rad_s": 0.01,
}
STATE_SHAPES = {
    "joint_position_rad": (12,),
    "joint_velocity_rad_s": (12,),
    "root_transform_w_xyzw": (7,),
    "root_com_velocity_w_m_s_rad_s": (6,),
    "link_transform_w_xyzw": (19, 7),
    "link_com_velocity_w_m_s_rad_s": (19, 6),
}
PROPERTY_SHAPES = {
    "body_mass_kg": (19,),
    "body_inertia_kg_m2": (19, 9),
    "body_com_pose": (19, 7),
    "joint_armature": (12,),
    "joint_physx_stiffness": (12,),
    "joint_physx_damping": (12,),
    "joint_max_velocity_rad_s": (12,),
    "joint_max_force_nm": (12,),
    "joint_position_limits_rad": (12, 2),
    "joint_operative_stiffness": (12,),
    "joint_operative_damping": (12,),
    "joint_friction_static_dynamic_viscous": (12, 3),
}
GO2_JOINTS = [
    f"{leg}_{joint}_joint"
    for joint in ("hip", "thigh", "calf")
    for leg in ("FL", "FR", "RL", "RR")
]
GO2_BODIES = [
    "base",
    "FL_hip",
    "FR_hip",
    "Head_upper",
    "RL_hip",
    "RR_hip",
    "FL_thigh",
    "FR_thigh",
    "Head_lower",
    "RL_thigh",
    "RR_thigh",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=_unique_object)
    _require(isinstance(value, dict) and _finite(value), "Expected finite JSON object")
    return value


def _digest(value, label):
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        f"{label}: expected SHA256",
    )


def _state(value, label, *, contacts=False):
    _require(isinstance(value, dict), f"{label}: missing state")
    for key, shape in STATE_SHAPES.items():
        _numeric_shape(value.get(key), shape, f"{label}.{key}")
    for row in [value["root_transform_w_xyzw"], *value["link_transform_w_xyzw"]]:
        _require(
            abs(sum(x * x for x in row[3:]) - 1.0) <= 1e-4,
            f"{label}: non-unit quaternion",
        )
    if contacts:
        _numeric_shape(
            value.get("joint_physx_actuation_force_nm"), (12,), f"{label}.effort"
        )
        _numeric_shape(value.get("contacts_w_n"), (19, 3), f"{label}.contacts")
        _require(
            all(abs(x) <= 1e-6 for row in value["contacts_w_n"] for x in row),
            f"{label}: unexpected contact force",
        )


def _properties(value, label):
    _require(isinstance(value, dict), f"{label}: missing properties")
    for key, shape in PROPERTY_SHAPES.items():
        _numeric_shape(value.get(key), shape, f"{label}.{key}")
    materials = value.get("shape_material_static_dynamic_restitution")
    _require(
        isinstance(materials, list) and materials, f"{label}: missing material readings"
    )
    _numeric_shape(materials, (len(materials), 3), f"{label}.materials")
    _require(all(x > 0 for x in value["body_mass_kg"]), f"{label}: nonpositive mass")
    for key in ("joint_armature", "joint_physx_stiffness", "joint_physx_damping"):
        _match(value[key], [0.0] * 12, f"{label}.{key}")
    _match(
        value["joint_friction_static_dynamic_viscous"],
        [[0.0] * 3 for _ in range(12)],
        f"{label}.friction",
    )
    _match(materials, [[1.0, 1.0, 0.0] for _ in materials], f"{label}.nominal_material")


def physics_contract(value):
    """Exclude only presentation/log fields; never import serialized functions."""
    _require(isinstance(value, dict), "Missing physics settings")
    result = copy.deepcopy(value)
    for key in (
        "render",
        "render_interval",
        "logging_level",
        "save_logs_to_file",
        "log_dir",
    ):
        result.pop(key, None)
    if isinstance(result.get("physics_material"), dict):
        result["physics_material"].pop("func", None)
    return result


def validate_reproducer_source(source):
    _require(
        isinstance(source, dict) and _finite(source), "Source plan must be finite JSON"
    )
    _require(
        source.get("kind") == "articulation_reproducer_source"
        and source.get("schema_version") == 1,
        "Invalid source plan kind/version",
    )
    _digest(source.get("source_sha256"), "source digest")
    _require(
        isinstance(source.get("source_path"), str) and source["source_path"],
        "Missing source path",
    )
    for key, expected in (
        ("joint_names", GO2_JOINTS),
        ("raw_joint_names", GO2_JOINTS),
        ("body_names", GO2_BODIES),
        ("source_origin_w_m", [24.0, 0.0, 0.0]),
    ):
        _match(source.get(key), expected, f"source.{key}")
    robot = source.get("robot", {})
    _require(
        robot.get("model") == "go2"
        and isinstance(robot.get("asset_path"), str)
        and robot["asset_path"].endswith("/IsaacLab/Robots/Unitree/Go2/go2.usd")
        and "/Isaac/5.1/" in robot["asset_path"],
        "Source is not the expected Go2 asset",
    )
    versions = source.get("source_versions", {})
    _require(
        str(versions.get("isaaclab", "")).startswith("2.3.2")
        and str(versions.get("isaacsim", "")).startswith("5.1."),
        "Source version anchors missing or unsupported",
    )
    physics = source["environment_physics"]
    _require(
        physics.get("device") == "cuda:0"
        and physics.get("dt") == DT
        and physics.get("use_fabric") is True
        and physics.get("physx", {}).get("solver_type") == 1,
        "Source requires GPU TGS, Fabric and 5ms dt",
    )
    _match(physics.get("gravity"), [0.0, 0.0, -9.81], "source.gravity")
    _require(
        "--/physics/collisionApproximateCylinders=true"
        in source.get("kit_args", "").split(),
        "Missing cylinder approximation setting",
    )
    _properties(source.get("properties"), "source.properties")
    _state(source.get("initial_state"), "source.initial_state")
    _state(source.get("original_first_post"), "source.original_first_post")
    effort = source.get("effort_nm")
    _numeric_shape(effort, (12,), "source.effort_nm")
    _require(
        max(map(abs, effort)) > 1e-6 and max(map(abs, effort)) <= 23.5,
        "Source effort must be nonzero and bounded by 23.5Nm",
    )


def load_reproducer_source(path):
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    report = json.loads(raw, object_pairs_hook=_unique_object)
    _require(isinstance(report, dict) and _finite(report), "Source must be finite JSON")
    validate_failure_trace(report)
    meta = report["metadata"]
    replay = meta.get("action_replay", {})
    _require(
        meta.get("action_source") == "recorded_action_replay"
        and replay.get("kind") == "startup_action_replay_probe"
        and replay.get("runtime_validated") is True
        and replay.get("initial_state_validated") is True
        and replay.get("prefix_steps") == 10,
        "Source is not a validated recorded-action startup replay",
    )
    for key in (
        "solver_probe",
        "scene_probe",
        "scene_intervention",
        "centering",
        "legacy_friction_probe",
    ):
        _require(key not in meta, f"Source contains unexpected intervention {key}")
    _require(
        report.get("stop_reason") == "step_limit"
        and len(report["samples"]) == 10
        and not any(s["post"]["done"] for s in report["samples"]),
        "Source prefix incomplete or terminal",
    )
    substeps = report.get("physics_substeps", [])
    _require(len(substeps) == 4, "Source needs four recorded first-action substeps")
    for i, step in enumerate(substeps):
        _require(
            step.get("control_step") == 0
            and step.get("physics_substep") == i
            and step.get("physics_dt_s") == DT,
            "Malformed source substep chronology",
        )
        _match(step.get("time_before_s"), i * DT, "source.substep.before", 1e-10)
        _match(step.get("time_after_s"), (i + 1) * DT, "source.substep.after", 1e-10)
    physical = report["physical_metadata"]
    initial = {
        k: copy.deepcopy(physical["initial_authoritative_state"][k])
        for k in STATE_SHAPES
    }
    for key in STATE_SHAPES:
        _match(
            initial[key],
            substeps[0]["pre"][key],
            f"source.initial_vs_first_pre.{key}",
            INITIAL_ATOL,
        )
    interface = meta["runtime_teacher_interface"]
    dynamics = report["samples"][0]["pre"]["observations"]["dynamics"]
    expected_dynamics = [
        0.0,
        *physical["properties"]["body_com_pose"][0][:3],
        1.0,
        1.0,
        0.0,
        *([0.0] * 24),
    ]
    _match(dynamics, expected_dynamics, "source.nominal_dynamics", INITIAL_ATOL)
    source = {
        "kind": "articulation_reproducer_source",
        "schema_version": 1,
        "source_path": str(path),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_origin_w_m": physical["environment_origin_w_m"],
        "joint_names": physical["joint_names"],
        "body_names": physical["body_names"],
        "raw_joint_names": substeps[0]["pre"]["generalized_dynamics"]["raw_dof_names"],
        "initial_state": initial,
        "effort_nm": substeps[0]["pre"]["joint_physx_actuation_force_nm"],
        "original_first_post": {k: substeps[0]["post"][k] for k in STATE_SHAPES},
        "properties": physical["properties"],
        "environment_physics": meta["environment_physics"],
        "kit_args": meta["kit_args"],
        "robot": interface["robot"],
        "source_versions": interface["training_provenance"]["runtime_versions"],
        "limitations": [
            "Replay validation flags are inherited evidence, not revalidation against its remote original policy file.",
            "Nominal recorded dynamics are checked; no unrecorded internal simulator state is inferred.",
        ],
    }
    validate_reproducer_source(source)
    return copy.deepcopy(source)


def _initial_match(left, right, label):
    for key in STATE_SHAPES:
        if "transform" in key:
            a = [left[key]] if key.startswith("root") else left[key]
            b = [right[key]] if key.startswith("root") else right[key]
            for i, (x, y) in enumerate(zip(a, b, strict=True)):
                _match(x[:3], y[:3], f"{label}.{key}[{i}].xyz", POSITION_ATOL)
                _match(x[3:], y[3:], f"{label}.{key}[{i}].quat", INITIAL_ATOL)
        else:
            _match(left[key], right[key], f"{label}.{key}", INITIAL_ATOL)


def validate_case(report, source):
    validate_reproducer_source(source)
    _require(isinstance(report, dict) and _finite(report), "Case must be finite JSON")
    _require(
        report.get("kind") == "articulation_reproducer_case"
        and report.get("schema_version") == 1,
        "Invalid case kind/version",
    )
    case = report.get("case")
    _require(case in CASES, "Unknown case")
    _match(report.get("source_sha256"), source["source_sha256"], "case.source_sha256")
    for key, x in zip(
        ("construction_origin_w_m", "reset_origin_w_m"), CASES[case], strict=True
    ):
        _match(report.get(key), [x, 0.0, 0.0], key)
    _match(
        report.get("construction_robot_position_w_m"),
        [CASES[case][0], 0.0, 0.4],
        "construction robot position",
        POSITION_ATOL,
    )
    _match(
        report.get("configured_construction_robot_position_w_m"),
        [CASES[case][0], 0.0, 0.4],
        "configured construction position",
    )
    _require(
        isinstance(report.get("construction_pose_readback_basis"), str)
        and report["construction_pose_readback_basis"],
        "Missing measured pre-reset construction-pose provenance",
    )
    _state(report.get("after_sim_reset_state"), "after_sim_reset_state", contacts=True)
    for key in ("joint_names", "raw_joint_names", "body_names"):
        _match(report.get(key), source[key], key)
    _match(
        report.get("configured_robot_usd_path"),
        source["robot"]["asset_path"],
        "Go2 asset path",
    )
    for key in ("isaaclab", "isaacsim"):
        _match(
            report.get("runtime_versions", {}).get(key),
            source["source_versions"][key],
            f"runtime_versions.{key}",
        )
    _match(
        physics_contract(report.get("environment_physics")),
        physics_contract(source["environment_physics"]),
        "runtime physics",
    )
    props = report.get("physical_properties")
    _properties(props, "case.properties")
    for key in PROPERTY_SHAPES | {"shape_material_static_dynamic_restitution": ()}:
        _match(
            props[key],
            source["properties"][key],
            f"case.properties.{key}",
            INITIAL_ATOL,
        )
    _numeric_shape(
        props.get("joint_legacy_friction_coefficient"), (12,), "case.legacy_friction"
    )
    _match(
        props["joint_legacy_friction_coefficient"], [0.0] * 12, "case.legacy_friction"
    )
    readback = report.get("scene_readback", {})
    for key, expected in {
        "simulation_manager_solver_type": "TGS",
        "usd_scene_solver_type": "TGS",
        "articulation_position_iterations": 4,
        "articulation_velocity_iterations": 0,
        "self_collisions_enabled": False,
        "is_fixed_base": False,
        "enabled_external_collision_prim_paths": [],
        "contact_body_names": source["body_names"],
        "gravity_w_m_s2": [0.0, 0.0, -9.81],
    }.items():
        _match(
            readback.get(key),
            expected,
            f"scene_readback.{key}",
            INITIAL_ATOL if key == "gravity_w_m_s2" else 0.0,
        )
    for key in (
        "physics_scene_path",
        "articulation_prim_path",
        "contact_evidence_basis",
    ):
        _require(
            isinstance(readback.get(key), str) and readback[key],
            f"Missing scene_readback.{key}",
        )
    paths = readback.get("enabled_robot_collision_prim_paths")
    _require(
        isinstance(paths, list)
        and paths
        and len(set(paths)) == len(paths)
        and all(isinstance(p, str) and p.startswith("/World/Robot/") for p in paths),
        "Invalid robot collision scope",
    )
    chronology = report.get("chronology", {})
    events = (
        "before_sim_reset",
        "after_sim_reset",
        "after_state_writes",
        "after_sim_forward",
        "pre_step",
        "post_step",
    )
    _require(set(chronology) == set(events), "Incomplete chronology")
    for event in events:
        row = chronology[event]
        _require(
            isinstance(row, dict)
            and type(row.get("sim_time_s")) in (int, float)
            and row["sim_time_s"] >= 0,
            "Invalid simulation clock",
        )
        for key in ("physics_step_index", "measured_step_count"):
            _require(
                type(row.get(key)) is int and row[key] >= 0, "Invalid step counter"
            )
        _require(
            row["measured_step_count"] == int(event == "post_step"),
            "Unexpected measured step count",
        )
    anchor = chronology["after_state_writes"]
    for event in ("after_sim_forward", "pre_step"):
        _match(chronology[event], anchor, f"Hidden physics advancement at {event}")
    for key in ("physics_step_index", "sim_time_s"):
        _match(
            anchor[key],
            chronology["after_sim_reset"][key],
            f"Hidden physics advancement before writes: {key}",
        )
    _match(
        chronology["post_step"]["sim_time_s"] - anchor["sim_time_s"],
        DT,
        "Measured dt",
        1e-10,
    )
    _require(
        chronology["post_step"]["physics_step_index"]
        == anchor["physics_step_index"] + 1,
        "Measured physics step index must advance once",
    )
    steps = report.get("steps")
    _require(
        type(report.get("measured_physics_steps")) is int
        and report["measured_physics_steps"] == 1
        and isinstance(steps, list)
        and len(steps) == 1,
        "Exactly one measured physics step is required",
    )
    step = steps[0]
    _require(
        type(step.get("physics_substep")) is int
        and step["physics_substep"] == 0
        and step.get("physics_dt_s") == DT,
        "Invalid measured step",
    )
    for when in ("pre", "post"):
        _state(step.get(when), f"case.{when}", contacts=True)
        _match(
            step[when]["joint_physx_actuation_force_nm"],
            source["effort_nm"],
            f"case.{when}.backend_effort",
            EFFORT_ATOL,
        )
    _initial_match(
        _physical_state(step["pre"], report["reset_origin_w_m"]),
        _physical_state(source["initial_state"], source["source_origin_w_m"]),
        "case.initial",
    )


def _differences(left, right):
    values = {}
    for key in STATE_SHAPES:
        a, b = left[key], right[key]
        if isinstance(a[0], list):
            a, b = [x for row in a for x in row], [x for row in b for x in row]
        values[key] = {
            "max_abs_difference": max(abs(x - y) for x, y in zip(a, b, strict=True)),
            "signed_difference": [x - y for x, y in zip(a, b, strict=True)],
        }
    values["root_position_local_m"] = {
        "max_abs_difference": max(
            abs(left["root_transform_w_xyzw"][i] - right["root_transform_w_xyzw"][i])
            for i in range(3)
        )
    }
    return values


def compare_cases(reports, source):
    """Report conditional contrasts; invalid evidence is returned, never passed."""
    try:
        _require(
            isinstance(reports, dict) and set(reports) == set(CASES),
            "Exactly origin, translated and teleported are required",
        )
        for name, report in reports.items():
            _require(report.get("case") == name, "Case key/name mismatch")
            validate_case(report, source)
        first = reports["origin"]
        for name in ("translated", "teleported"):
            for key in (
                "environment_physics",
                "runtime_versions",
                "physical_properties",
                "scene_readback",
            ):
                _match(reports[name][key], first[key], f"matched cases.{name}.{key}")
            _initial_match(
                _physical_state(
                    reports[name]["steps"][0]["pre"], reports[name]["reset_origin_w_m"]
                ),
                _physical_state(first["steps"][0]["pre"], first["reset_origin_w_m"]),
                f"matched cases.{name}.pre",
            )
        contrasts = {}
        for left, right in (
            ("origin", "translated"),
            ("translated", "teleported"),
            ("origin", "teleported"),
        ):
            delta = _differences(
                _physical_state(
                    reports[left]["steps"][0]["post"], reports[left]["reset_origin_w_m"]
                ),
                _physical_state(
                    reports[right]["steps"][0]["post"],
                    reports[right]["reset_origin_w_m"],
                ),
            )
            exceeded = [
                key
                for key, threshold in THRESHOLDS.items()
                if delta[key]["max_abs_difference"] > threshold
            ]
            contrasts[f"{left}_vs_{right}"] = {
                "post_step_differences": delta,
                "thresholds_exceeded": exceeded,
                "sensitivity_observed": bool(exceeded),
            }
        original = _physical_state(
            source["original_first_post"], source["source_origin_w_m"]
        )
        result = {
            "kind": "articulation_reproducer_comparison",
            "schema_version": 1,
            "status": "SENSITIVITY_OBSERVED"
            if any(x["sensitivity_observed"] for x in contrasts.values())
            else "NO_SENSITIVITY_OBSERVED",
            "source_sha256": source["source_sha256"],
            "measured_step_duration_s": DT,
            "engineering_thresholds": THRESHOLDS.copy(),
            "initial_abs_tolerance": INITIAL_ATOL,
            "initial_position_abs_tolerance_m": POSITION_ATOL,
            "contrasts": contrasts,
            "original_first_step_descriptive_only": {
                name: _differences(
                    _physical_state(
                        report["steps"][0]["post"], report["reset_origin_w_m"]
                    ),
                    original,
                )
                for name, report in reports.items()
            },
            "response_descriptive_only": {
                name: {
                    "max_abs_joint_position_change_rad": max(
                        abs(a - b)
                        for a, b in zip(
                            report["steps"][0]["post"]["joint_position_rad"],
                            report["steps"][0]["pre"]["joint_position_rad"],
                            strict=True,
                        )
                    ),
                    "max_abs_post_joint_velocity_rad_s": max(
                        map(abs, report["steps"][0]["post"]["joint_velocity_rad_s"])
                    ),
                    "max_abs_backend_effort_nm": max(map(abs, source["effort_nm"])),
                }
                for name, report in reports.items()
            },
            "limitations": [
                "One contact-free constant-effort step is not normal policy behavior, a robot acceptance test, or proof of physical accuracy.",
                "Origin versus translated changes placement; translated versus teleported changes construction/reset history conditional on matched recorded inputs.",
                "Sensitivity does not identify an engine defect; no sensitivity does not exonerate the original scene or longer dynamics.",
                "Original terrain/control path is removed; original first-step differences are descriptive and are not a rejection gate.",
                "Initial translation comparisons allow 10 micrometers for float32 world-coordinate subtraction; other initial states allow 2e-6 in their stated units.",
                "Source replay validation is inherited from recorded flags; only the supplied source bytes and this standalone contract are independently checked.",
            ],
        }
        result["evidence_status"] = result["status"]
        _require(_finite(result), "Comparison produced nonfinite data")
        return result
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        return {
            "kind": "articulation_reproducer_comparison",
            "schema_version": 1,
            "status": "INVALID_DIAGNOSTIC",
            "evidence_status": "INVALID_DIAGNOSTIC",
            "error": str(error),
        }
