"""Compare a matched native/centered feedback screen without promoting a repair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from . import startup_probe_report as probe
    from . import startup_solver_probe as solver
    from .startup_centered_scene import validate_centered_evaluation_evidence
    from .evaluation_screen import screen_failures
except ImportError:
    import startup_probe_report as probe
    import startup_solver_probe as solver
    from startup_centered_scene import validate_centered_evaluation_evidence
    from evaluation_screen import screen_failures


STATE_SHAPES = {
    "root_position_env_m": 3,
    "root_orientation_wxyz": 4,
    "linear_velocity_body_m_s": 3,
    "angular_velocity_body_rad_s": 3,
    "joint_position_rad": 12,
    "joint_velocity_rad_s": 12,
}
CONTRACT_FIELDS = (
    "interface_version",
    "action",
    "actor",
    "adaptation",
    "command_contract",
    "information_contract",
    "robot",
    "state_estimation",
    "terrain_scan",
    "timing",
)
OUTCOME_TERMS = {
    "chassis_contact": "chassis_contact",
    "fell_below_course": "fell_below_course",
    "off_route": "off_route",
    "success": "success",
    "active_timeout": "time_out",
    "wall_only_timeout": "wall_time_out",
}


def _hash(value, label):
    probe._require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        f"{label}: invalid SHA256",
    )


def _numbers(value, label):
    if isinstance(value, list):
        probe._require(bool(value), f"{label}: empty observation")
        probe._require(
            len({_shape(row) for row in value}) == 1, f"{label}: ragged observation"
        )
        return [n for row in value for n in _numbers(row, label)]
    probe._numeric_shape(value, (), label)
    return [value]


def _shape(value):
    return (
        (len(value), *(_shape(value[0]) if value else ()))
        if isinstance(value, list)
        else ()
    )


def _episodes(report):
    episodes = report["scene_probe"].get("episodes")
    probe._require(
        isinstance(episodes, list) and len(episodes) == 3,
        "Need three completed episode records",
    )
    interface = report["teacher_interface"]
    dimensions = {
        g["name"]: g["dimension"]
        for g in interface["actor"].get("observation_groups", [])
    }
    history = interface["adaptation"].get("deployable_history", {})
    if history:
        dimensions[history["name"]] = history["dimension"]
    target = interface["state_estimation"].get("target_observation", {})
    if target:
        dimensions[target["name"]] = target["dimension"]
    for index, episode in enumerate(episodes):
        probe._require(
            type(episode.get("episode_index")) is int,
            "Episode index must be an integer",
        )
        probe._match(episode.get("episode_index"), index, "contiguous episode index")
        state = episode["initial_state"]
        for key, width in STATE_SHAPES.items():
            probe._numeric_shape(state.get(key), (width,), key)
        probe._match(
            state.get("environment_origin_w_m"),
            report["scene_probe"]["environment_origin_w_m"],
            "episode runtime origin",
        )
        names = state.get("joint_names")
        probe._require(
            isinstance(names, list)
            and len(names) == 12
            and len(set(names)) == 12
            and set(names) == set(interface["action"]["joint_names"]),
            "Episode raw joint map invalid",
        )
        probe._require(
            abs(sum(v * v for v in state["root_orientation_wxyz"]) - 1) <= 0.001,
            "Nonphysical root quaternion",
        )
        observations = episode["initial_observations"]
        probe._require(
            isinstance(observations, dict)
            and observations
            and all(isinstance(v, list) for v in observations.values()),
            "Missing initial policy observations",
        )
        flattened = {
            key: _numbers(value, f"observation {key}")
            for key, value in observations.items()
        }
        for key, size in dimensions.items():
            probe._require(
                type(size) is int and size > 0 and len(flattened.get(key, [])) == size,
                f"Observation {key} dimension mismatch",
            )
        probe._numeric_shape(
            episode.get("first_policy_action"), (12,), "first policy action"
        )
        steps, duration = episode.get("duration_steps"), episode.get("duration_s")
        probe._require(
            type(steps) is int
            and steps > 0
            and type(duration) in (int, float)
            and math.isclose(duration, steps * 0.02, rel_tol=0, abs_tol=1e-7),
            "Episode duration does not match complete 20ms steps",
        )
        reasons = episode.get("termination_reasons")
        probe._require(
            isinstance(reasons, list)
            and reasons
            and len(set(reasons)) == len(reasons)
            and set(reasons) <= set(OUTCOME_TERMS.values()),
            "Invalid raw termination reasons",
        )
        remaining, expected = True, {}
        for outcome, term in OUTCOME_TERMS.items():
            expected[outcome] = remaining and term in reasons
            remaining = remaining and not expected[outcome]
        expected["timeout"] = (
            expected["active_timeout"] or expected["wall_only_timeout"]
        )
        probe._match(
            episode.get("outcomes"),
            expected,
            "Episode outcome precedence/raw termination mismatch",
        )
        probe._numeric_shape(
            episode.get("max_course_progress_m"), (), "episode progress"
        )
        probe._require(
            type(episode.get("max_waypoints_reached")) is int
            and episode["max_waypoints_reached"] >= 0,
            "Invalid episode waypoint count",
        )
    for outcome in (*OUTCOME_TERMS, "timeout"):
        rate = sum(e["outcomes"][outcome] for e in episodes) / 3
        probe._match(
            report["summary"].get(outcome + "_rate"),
            rate,
            "Summary outcome rate disagrees with completed episodes",
            1e-9,
        )
    for key, field in (
        ("mean_episode_length_steps", "duration_steps"),
        ("mean_episode_length_seconds", "duration_s"),
        ("mean_max_course_progress_m", "max_course_progress_m"),
        ("mean_max_waypoints_reached", "max_waypoints_reached"),
    ):
        probe._match(
            report["summary"].get(key),
            sum(e[field] for e in episodes) / 3,
            f"Summary {key} disagrees with episodes",
            1e-7,
        )
    return episodes


def _case(report, placement):
    json.dumps(report, allow_nan=False)
    expected = {
        "task": "Parkour-Lab-v0",
        "policy_mode": "history_mean",
        "reset_profile": "jitter",
        "terrain_family": "high_step",
        "difficulty_level": 6,
        "geometry_variant_index": 0,
        "desired_speed_m_s": 0.55,
        "desired_yaw_rate_rad_s": 0.0,
        "command_profile": "translation_only",
        "seed": 42,
        "num_envs": 1,
        "requested_episodes": 3,
        "completed_episodes": 3,
    }
    for key, value in expected.items():
        if type(value) is int:
            probe._require(
                type(report.get(key)) is int, f"{placement}.{key} must be an integer"
            )
        probe._match(report.get(key), value, f"{placement}.{key}")
    probe._require(
        report.get("action_source", "policy") in ("policy", "policy_action")
        and not any(
            key in report
            for key in (
                "action_replay",
                "action_probe",
                "replay",
                "legacy_friction_probe",
                "scene_intervention",
                "samples",
            )
        ),
        "Feedback case must be normal policy, not replay/intervention trace",
    )
    for key in ("checkpoint_sha256", "teacher_interface_sha256"):
        _hash(report.get(key), key)
    probe._require(
        isinstance(report.get("kit_args"), str)
        and [
            arg
            for arg in report["kit_args"].split()
            if arg.startswith("--/physics/collisionApproximateCylinders=")
        ]
        == ["--/physics/collisionApproximateCylinders=true"],
        "Missing explicit collisionApproximateCylinders=true launch evidence",
    )
    for key in (
        "teacher_interface",
        "evaluation_reward_config",
        "reset_parameters",
        "difficulty_metadata",
        "summary",
    ):
        probe._require(
            isinstance(report.get(key), dict) and report[key], f"Missing {key}"
        )
    for key in CONTRACT_FIELDS:
        probe._require(
            key in report["teacher_interface"], f"Missing interface contract {key}"
        )
    speed_error = report["summary"].get("mean_moving_speed_absolute_error_m_s")
    probe._numeric_shape(speed_error, (), "Observed moving-speed error")
    probe._require(speed_error >= 0, "Moving-speed absolute error must be nonnegative")
    probe._require(
        isinstance(report["evaluation_reward_config"].get("terms"), dict),
        "Missing runtime reward terms",
    )
    if "training_config" in report:
        probe._require(
            isinstance(report["training_config"], dict),
            "Invalid training config provenance",
        )
        for key, value in report["training_config"].items():
            probe._require(
                isinstance(value, dict)
                and isinstance(value.get("path"), str)
                and value["path"],
                "Invalid training config path",
            )
            _hash(value.get("sha256"), key)
    noise = report.get("action_noise", {})
    probe._match(
        noise.get("source"), "deterministic", "Normal mean-policy action source"
    )
    probe._match(
        noise.get("effective_std"),
        {"min": 0.0, "mean": 0.0, "max": 0.0},
        "Mean policy must not sample actions",
    )
    meta = report.get("scene_probe", {})
    probe._require(
        "incomplete_episode" in meta and meta["incomplete_episode"] is None,
        "Incomplete or missing episode completion evidence",
    )
    for key, value in (
        ("kind", "scene_feedback"),
        ("schema_version", 1),
        ("placement", placement),
    ):
        probe._match(meta.get(key), value, f"scene_probe.{key}")
    probe._numeric_shape(
        meta.get("environment_origin_w_m"), (3,), "scene runtime origin"
    )
    selection = report["solver_probe"]
    solver.validate_solver_metadata(selection)
    probe._match(
        selection["requested_solver"], "TGS", "Feedback solver must remain TGS"
    )
    solver.validate_solver_environment_physics(
        report["environment_physics"], report["environment_physics"], selection
    )
    if placement == "native":
        probe._require(
            any(meta["environment_origin_w_m"]), "Native origin must be nonzero"
        )
        probe._match(meta.get("centering"), None, "Native cannot be centered")
    else:
        probe._match(
            meta["environment_origin_w_m"], [0.0] * 3, "Centered runtime origin", 2e-6
        )
        validate_centered_evaluation_evidence(
            meta.get("centering"), require_runtime=True
        )
        probe._match(
            meta["centering"]["runtime_environment_origin_w_m"],
            meta["environment_origin_w_m"],
            "Centering readback binding",
        )
    return _episodes(report)


def _initial_difference(left, right, *, gate):
    result = {}
    probe._match(
        left["initial_state"]["joint_names"],
        right["initial_state"]["joint_names"],
        "Initial raw joint order",
    )
    for key in STATE_SHAPES:
        a, b = left["initial_state"][key], right["initial_state"][key]
        tolerance = 1e-5 if key == "root_position_env_m" else 1e-6
        if gate:
            probe._match(a, b, "First episode initial " + key, tolerance)
        result[key] = {
            "max_abs_difference": max(abs(x - y) for x, y in zip(a, b)),
            "first_episode_tolerance": tolerance,
        }
    a, b = left["initial_observations"], right["initial_observations"]
    probe._require(a.keys() == b.keys(), "Policy observation groups differ")
    result["initial_observations"] = {}
    for key in a:
        if gate:
            probe._match(a[key], b[key], "First episode observation " + key, 1e-4)
        x, y = _numbers(a[key], key), _numbers(b[key], key)
        probe._require(len(x) == len(y), "Observation dimensions differ")
        result["initial_observations"][key] = {
            "max_abs_difference": max(abs(u - v) for u, v in zip(x, y)),
            "first_episode_tolerance": 1e-4,
        }
    a, b = left["first_policy_action"], right["first_policy_action"]
    if gate:
        probe._match(a, b, "First episode policy action", 1e-4)
    result["first_policy_action"] = {
        "max_abs_difference": max(abs(x - y) for x, y in zip(a, b)),
        "first_episode_tolerance": 1e-4,
    }
    return result


def compare_scene_feedback(native, centered):
    episodes = [_case(native, "native"), _case(centered, "centered")]
    for key in (
        "checkpoint_sha256",
        "teacher_interface_sha256",
        "reset_parameters",
        "action_noise",
        "difficulty_metadata",
        "evaluation_reward_config",
        "environment_physics",
        "kit_args",
    ):
        probe._match(native[key], centered[key], "Matched case " + key)
    for key in CONTRACT_FIELDS:
        probe._match(
            native["teacher_interface"][key],
            centered["teacher_interface"][key],
            "Matched interface " + key,
        )
    probe._match(
        native.get("training_config"),
        centered.get("training_config"),
        "Training config provenance",
    )
    probe._match(
        native["teacher_interface"].get("training_provenance"),
        centered["teacher_interface"].get("training_provenance"),
        "Interface training provenance",
    )
    probe._match(
        centered["scene_probe"]["centering"]["source_origin_w_m"],
        native["scene_probe"]["environment_origin_w_m"],
        "Centering source/native origin",
    )
    comparisons = [
        _initial_difference(a, b, gate=i == 0)
        for i, (a, b) in enumerate(zip(*episodes))
    ]
    cases = {}
    for name, value, records in zip(
        ("native", "centered"), (native, centered), episodes
    ):
        cases[name] = {
            "failures": screen_failures(value),
            "scalar_summary": {
                k: v
                for k, v in value["summary"].items()
                if v is None or type(v) in (int, float, bool)
            },
            "episodes": [
                {
                    k: v
                    for k, v in e.items()
                    if k
                    not in (
                        "initial_state",
                        "initial_observations",
                        "first_policy_action",
                    )
                }
                for e in records
            ],
        }
    return {
        "kind": "scene_feedback_comparison",
        "schema_version": 1,
        "evidence_status": "CENTERED_SCREEN_FAILED"
        if cases["centered"]["failures"]
        else "CENTERED_SCREEN_PASSED",
        "checkpoint_sha256": native["checkpoint_sha256"],
        "teacher_interface_sha256": native["teacher_interface_sha256"],
        "cases": cases,
        "initial_episode_comparisons": comparisons,
        "limitations": [
            "Three completed episodes per placement are a behavioral screen, not a reliability certificate or production-fix approval.",
            "Only the first episode initial state/input/action is tolerance-gated. Later reset differences are descriptive: diverging trajectories can consume random numbers differently.",
            "Centering changes placement, initialization exposure and cooking coordinates; improved behavior alone does not isolate an engine defect.",
            "No cases are averaged together. Raw simultaneous termination reasons are retained alongside precedence-classified outcomes.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_dir", type=Path)
    parser.add_argument(
        "--native-only",
        action="store_true",
        help="Validate native evidence before running centered; a valid native behavioral failure returns zero.",
    )
    args = parser.parse_args(argv)
    inputs = {}
    try:
        reports = []
        for name in (
            ("native_L6",) if args.native_only else ("native_L6", "centered_L6")
        ):
            paths = list((args.pair_dir / name).rglob("metrics.json"))
            probe._require(
                len(paths) == 1,
                f"{name}: expected exactly one metrics.json, found {len(paths)}",
            )
            raw = paths[0].read_bytes()
            inputs[name] = {
                "path": str(paths[0]),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            reports.append(json.loads(raw, object_pairs_hook=probe._unique_object))
        if args.native_only:
            episodes = _case(reports[0], "native")
            result = {
                "kind": "scene_feedback_native_validation",
                "schema_version": 1,
                "evidence_status": "NATIVE_EVIDENCE_VALID",
                "native_screen_failures": screen_failures(reports[0]),
                "completed_episode_records": len(episodes),
                "limitations": [
                    "Valid native evidence may fail its behavioral screen; this permits only the centered comparison, not robot acceptance."
                ],
            }
        else:
            result = compare_scene_feedback(*reports)
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result = {
            "kind": "scene_feedback_comparison",
            "evidence_status": "INVALID_DIAGNOSTIC",
            "error": str(error),
        }
    result["inputs"] = inputs
    print(json.dumps(result, indent=2, allow_nan=False))
    return {
        "CENTERED_SCREEN_PASSED": 0,
        "NATIVE_EVIDENCE_VALID": 0,
        "CENTERED_SCREEN_FAILED": 1,
        "INVALID_DIAGNOSTIC": 2,
    }[result["evidence_status"]]


if __name__ == "__main__":
    raise SystemExit(main())
