"""Read-only, CPU development audit joining qualification components.

This is NOT the frozen qualification runner. Archived tapes are independently
reconstructed, never inferred from achieved motion. Original reports stay intact;
historical short holds/timeouts are not repaired. Every declared trial is listed,
including missing evidence. No checkpoint is loaded and no simulator is launched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np

from .operator_benchmark_core import Phase
from .operator_qualification import score_trial_kinematics
from .operator_qualification_physical import score_trial_physical
from .operator_support_capture import (
    validate_support_capture,
    validate_support_geometry_binding,
)


VERSION = "operator_qualification_development_audit_v1"
_MODES = {
    "go2_operator_proprio_command_coverage_v1": "command_coverage",
    "go2_operator_proprio_command_source_v1": "command_source",
    "go2_operator_proprio_out_and_back_v1": "out_and_back",
}
_FILES = ("params/env.yaml", "params/agent.yaml", "trace.npz")
_PROFILES = ("plane", "rough_flat", "hills", "step_hills", "tilted_ramps")
_UNSCORED = (
    "terrain_support_acceptance",
    "supported_ascent_descent",
    "ordered_flat_rough_flat",
    "supported_stop_restart",
    "substep_joint_and_nonfoot_contact_extrema",
    "real_streamed_input",
    "fresh_frozen_matrix",
    "new_parent_observed_process_exit",
)
_DYNAMIC = (
    "command",
    "position",
    "pre_position",
    "quaternion",
    "pre_quaternion",
    "linear_velocity_b",
    "angular_velocity_b",
    "angular_velocity_w",
    "root_link_lin_vel_b",
    "terminated",
    "time_out",
    "valid_first_attempt",
    "procedural_workspace",
    "joint_position_post",
    "joint_velocity_post",
    "computed_torque_substeps",
    "applied_torque_substeps",
    "contact_force_norm_n",
    "observation",
    "action",
    "joint_target",
    "default_joint_position",
)


def _canonical(protocol):
    # Lazy import: --help and pure component use do not initialize PyTorch.
    from .operator_student_bridge import recurrent_evaluation_protocol

    if not isinstance(protocol, dict):
        raise ValueError("Evaluation protocol must be a JSON object")
    mode = _MODES.get(protocol.get("version"))
    if mode is None or protocol.get("difficulty_range") is None:
        raise ValueError(
            "Require a canonical native coverage, source or out-and-back tape"
        )
    if "root_point_telemetry" not in protocol:
        raise ValueError("Require declared root-link and COM telemetry")
    expected = recurrent_evaluation_protocol(
        seed=protocol.get("seed"),
        difficulty_range=protocol["difficulty_range"],
        root_point_diagnostics=True,
        support_capture="support_capture" in protocol,
        **{mode: True},
    )
    if any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "Archived tape differs from its canonical independent declaration"
        )
    for key in ("reward_telemetry", "command_limits", "source_lease", "foot_telemetry"):
        if key in protocol and key not in expected:
            raise ValueError("Undeclared evaluation variant: " + key)
    return expected


def _assignments(trace):
    columns, profiles = trace.get("terrain_column_id"), trace.get("terrain_profile_id")
    for name, array in (("columns", columns), ("profiles", profiles)):
        if (
            not isinstance(array, np.ndarray)
            or array.shape != (80,)
            or array.dtype.kind not in "iu"
        ):
            raise ValueError("Invalid canonical terrain assignment: " + name)
    if (
        np.any(columns >= 20)
        or np.any(columns < 0)
        or not np.array_equal(
            np.bincount(columns.astype(int), minlength=20), np.full(20, 4)
        )
        or not np.array_equal(profiles, columns // 4)
    ):
        raise ValueError("Require all four declared trials in every terrain column")
    return columns, profiles


def declared_trial_phases(protocol, trace, env_id):
    """Build expected commands from protocol/strata, not measured command edges."""
    protocol = _canonical(protocol)
    columns, profiles = _assignments(trace)
    if type(env_id) is not int or not 0 <= env_id < 80:
        raise ValueError("Invalid declared environment id")
    if protocol["version"] == "go2_operator_proprio_command_source_v1":
        from .operator_command_source import assignment

        _, arms = assignment(columns)
        zero = protocol["source_lease"]["expected_zero_steps"][int(arms[env_id])]
        return [
            Phase("cold_stand", 2.0, (0.0, 0.0, 0.0)),
            Phase("forward", (zero - 100) * 0.02, (0.4, 0.0, 0.0)),
            Phase("source_loss", (protocol["steps"] - zero) * 0.02, (0.0, 0.0, 0.0)),
        ]
    sign = 1
    if protocol["version"] == "go2_operator_proprio_out_and_back_v1":
        from .operator_student_bridge import out_and_back_assignment

        _, signs = out_and_back_assignment(columns)
        sign = int(signs[env_id])
    phases = []
    for phase in protocol["phases"]:
        command = list(
            phase["flat_command" if profiles[env_id] < 2 else "rough_command"]
        )
        command[2] *= sign
        phases.append(Phase(phase["name"], phase["duration_s"], tuple(command)))
    return phases


def _blocked_trials(reason):
    return [
        {
            "env_id": i,
            "status": "MISSING_OR_INVALID_EVIDENCE",
            "reason": reason,
            "qualification_passed": False,
            "exit_allowed": False,
        }
        for i in range(80)
    ]


def _base():
    return {
        "version": VERSION,
        "status": "AUDITED_NOT_QUALIFIED",
        "qualification_passed": False,
        "exit_allowed": False,
        "learning_updates": 0,
        "scope": "Supplemental retrospective development audit; no original result is changed, no fresh qualification or policy promotion",
        "trial_count": 80,
        "unscored_requirements": list(_UNSCORED),
        "issues": [],
    }


def score_evaluation_components(
    trace, protocol, *, velocity_reference, measurement=None
):
    """Assemble all 80 first attempts. Never convert component success to acceptance."""
    if velocity_reference not in ("root_link", "com"):
        raise ValueError("Explicit root_link or com velocity reference required")
    protocol = _canonical(protocol)
    result = _base()
    result["velocity_reference"] = velocity_reference
    result["protocol_version"] = protocol["version"]
    try:
        _, profiles = _assignments(trace)
        command = trace.get("command")
        if (
            not isinstance(command, np.ndarray)
            or command.ndim != 3
            or command.shape[1:] != (80, 3)
        ):
            raise ValueError("Missing or malformed full trial axis")
        length = len(command)
        index = trace.get("sample_index")
        if index is None:
            if "support_capture" in protocol:
                raise ValueError(
                    "Declared support capture is missing native sample_index"
                )
            index = np.arange(length, dtype=np.int64)
            result["sample_index_evidence"] = (
                "LEGACY_DERIVED_NPZ_ROW_ORDER_NOT_NATIVE_INDEX_EVIDENCE"
            )
        else:
            result["sample_index_evidence"] = "RECORDED_NATIVE_INDEX"
        for name in _DYNAMIC:
            value = trace.get(name)
            if (
                value is None
                or not isinstance(value, np.ndarray)
                or value.ndim < 2
                or value.shape[:2] != (length, 80)
            ):
                raise ValueError("Missing or malformed batch field: " + name)
        if np.asarray(trace.get("joint_pos_limits")).shape != (80, 12, 2):
            raise ValueError("Missing or malformed measured hard joint limits")
    except (ValueError, TypeError) as exc:
        result.update(status="INVALID_INPUT", trials=_blocked_trials(str(exc)))
        result["issues"].append(str(exc))
        return result

    # Replay the old capture/admission contract, not old floating metric equality:
    # cross-platform floating calculations need not be bitwise identical.
    from .operator_student_bridge import summarize_recurrent_evaluation

    try:
        summarize_recurrent_evaluation(trace, protocol)
        result["legacy_capture_replay"] = "VALIDATED_NOT_QUALIFICATION"
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        result["legacy_capture_replay"] = "INVALID_OR_INCOMPLETE"
        result["issues"].append("Legacy capture replay: " + str(exc))

    result["support_capture"] = {"status": "NOT_SCORED", "capture": "NOT_DECLARED"}
    if "support_capture" in protocol:
        try:
            summary = validate_support_capture(trace, protocol)
            validate_support_geometry_binding(
                (measurement or {}).get("foot_geometry_binding"), trace
            )
            result["support_capture"] = summary
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            result["support_capture"] = {
                "status": "INVALID_EVIDENCE",
                "error": str(exc),
            }
            result["issues"].append("Support capture: " + str(exc))
    elif "foot_telemetry" in protocol:
        result["support_capture"][
            "capture"
        ] = "LEGACY_FOOT_DIAGNOSTICS_NOT_CURRENT_SUPPORT_MANIFEST"

    trials = []
    for env_id in range(80):
        single = {name: trace[name][:, env_id] for name in _DYNAMIC}
        single.update(
            sample_index=index,
            joint_pos_limits=trace["joint_pos_limits"][env_id],
            contact_body_names=trace.get("contact_body_names"),
        )
        trial = {
            "env_id": env_id,
            "terrain": _PROFILES[int(profiles[env_id])],
            "qualification_passed": False,
            "exit_allowed": False,
        }
        phases = declared_trial_phases(protocol, trace, env_id)
        trial["expected_phases"] = [
            {"name": p.name, "duration_s": p.duration_s, "command": list(p.command)}
            for p in phases
        ]
        for key, operation in (
            (
                "kinematics",
                lambda: score_trial_kinematics(
                    single,
                    phases,
                    velocity_reference=velocity_reference,
                    plane=bool(profiles[env_id] == 0),
                ),
            ),
            (
                "physical",
                lambda: score_trial_physical(
                    single, expected_steps=protocol["steps"], contract=None
                ),
            ),
        ):
            try:
                trial[key] = operation()
            except (ValueError, TypeError, KeyError, IndexError) as exc:
                trial[key] = {
                    "status": "INVALID_EVIDENCE",
                    "error": str(exc),
                    "qualification_passed": False,
                }
                result["issues"].append(f"Trial {env_id} {key}: {exc}")
        trials.append(trial)
    result["trials"] = trials
    result["protocol_insufficiencies"] = {
        "stationary_phases_shorter_than_three_seconds": sorted(
            {
                phase["name"]
                for trial in trials
                for phase in trial["expected_phases"]
                if np.linalg.norm(phase["command"][:2]) == 0
                and phase["duration_s"] < 3.0
            }
        ),
        "trials_with_final_sample_timeout": (
            int(np.sum(trace["time_out"][-1] & trace["valid_first_attempt"][-1]))
            if length
            else 0
        ),
        "interpretation": "Historical short holds and native final timeouts are retained evidence limitations under these supplemental strict rules, not proof of newly degraded policy behavior",
    }
    result["component_counts"] = {
        key: dict(Counter(t[key]["status"] for t in trials))
        for key in ("kinematics", "physical")
    }
    result["status"] = "INVALID_INPUT" if result["issues"] else "AUDITED_NOT_QUALIFIED"
    return result


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path, hashes):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)

    data = path.read_bytes()
    value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: " + path.name)
    hashes[path.name] = hashlib.sha256(data).hexdigest()
    return value


def audit_run(run_dir, *, velocity_reference):
    """Verify archive receipts, then audit without writing into the source run."""
    run = Path(run_dir).resolve()
    result = _base()
    result.update(run_directory=str(run), velocity_reference=velocity_reference)
    hashes = {}
    try:
        if velocity_reference not in ("root_link", "com"):
            raise ValueError("Explicit root_link or com velocity reference required")
        protocol = _json(run / "evaluation_protocol.json", hashes)
        tape = _canonical(protocol)
        measurement = _json(run / "measurement_report.json", hashes)
        sources = protocol.get("evaluation_sources")
        if (
            not isinstance(sources, dict)
            or set(sources)
            != {
                "checkpoint",
                "training_protocol.json",
                "params/env.yaml",
                "params/agent.yaml",
            }
            or any(
                not isinstance(v, str) or re.fullmatch(r"[a-f0-9]{64}", v) is None
                for v in sources.values()
            )
        ):
            raise ValueError("Missing or invalid archived checkpoint/config identities")
        if (
            type(protocol.get("checkpoint_learning_updates")) is not int
            or protocol["checkpoint_learning_updates"] < 0
            or protocol.get("policy_version") != "go2_operator_proprio_gru_v1"
            or type(protocol.get("learning_updates")) is not int
            or protocol["learning_updates"] != 0
        ):
            raise ValueError("Missing or invalid archived policy/update identity")
        if "actor_bundle" in protocol:
            bundle = protocol["actor_bundle"]
            if (
                not isinstance(bundle, dict)
                or set(bundle) != {"path", "sha256"}
                or not isinstance(bundle["path"], str)
                or not bundle["path"]
                or not isinstance(bundle["sha256"], str)
                or re.fullmatch(r"[a-f0-9]{64}", bundle["sha256"]) is None
            ):
                raise ValueError("Invalid archived actor-bundle identity")
        declared_trials = measurement.get("trials")
        if (
            not isinstance(declared_trials, list)
            or len(declared_trials) != 80
            or any(
                not isinstance(t, dict) or type(t.get("env_id")) is not int
                for t in declared_trials
            )
            or [t["env_id"] for t in declared_trials] != list(range(80))
        ):
            raise ValueError("Require all 80 distinct ordered trial receipts")
        if (
            measurement.get("protocol") != tape
            or measurement.get("protocol_sha256") != hashes["evaluation_protocol.json"]
            or measurement.get("status") != "DEVELOPMENT_EVALUATED_NOT_ACCEPTED"
            or measurement.get("summary_version") != "go2_operator_proprio_summary_v3"
            or measurement.get("exit_allowed") is not False
            or type(measurement.get("learning_updates")) is not int
            or measurement["learning_updates"] != 0
            or measurement.get("control_steps") != tape["steps"]
            or measurement.get("environment_transitions") != tape["steps"] * 80
            or measurement.get("checkpoint_sha256")
            != protocol.get("evaluation_sources", {}).get("checkpoint")
            or any(
                measurement.get(k) != protocol.get(k)
                for k in (
                    "evaluation_sources",
                    "checkpoint_learning_updates",
                    "policy_version",
                    "actor_bundle",
                )
            )
        ):
            raise ValueError(
                "Archived measurement/protocol identity or trial receipt differs"
            )
        recorded = measurement.get("sha256", {})
        if set(recorded) != set(_FILES):
            raise ValueError("Require the exact native trace/config artifact receipt")
        for name in _FILES:
            hashes[name] = _digest(run / name)
            if hashes[name] != recorded[name]:
                raise ValueError("Artifact SHA256 differs: " + name)
        # Own the stream even when NumPy fails while opening a corrupt ZIP.
        with (run / "trace.npz").open("rb") as stream:
            with np.load(stream, allow_pickle=False) as archive:
                if len(archive.files) != len(set(archive.files)):
                    raise ValueError("Duplicate NPZ field names")
                trace = dict(archive)
        columns, profiles = _assignments(trace)
        if any(
            t.get("terrain") != _PROFILES[int(profiles[i])]
            or type(t.get("terrain_column_id")) is not int
            or t["terrain_column_id"] != int(columns[i])
            for i, t in enumerate(declared_trials)
        ):
            raise ValueError(
                "Trial receipt terrain assignment differs from raw evidence"
            )
        result.update(
            score_evaluation_components(
                trace,
                tape,
                velocity_reference=velocity_reference,
                measurement=measurement,
            )
        )
        result["recorded_candidate"] = {
            k: protocol.get(k)
            for k in (
                "evaluation_sources",
                "checkpoint_learning_updates",
                "actor_bundle",
            )
        }
        result["candidate_validation_scope"] = (
            "Archived identity only; checkpoint/bundle are not reloaded or independently reverified by this offline audit"
        )
        result["process_exit"] = {
            "status": "MISSING_RECORDED_RECEIPT",
            "new_parent_observation": False,
        }
        if (run / "report.json").is_file() and (run / "worker_status.json").is_file():
            parent = _json(run / "report.json", hashes)
            worker = _json(run / "worker_status.json", hashes)
            if (
                parent.get("worker") != worker
                or {k: v for k, v in parent.items() if k != "worker"} != measurement
            ):
                raise ValueError("Parent/worker receipt differs from measurement")
            if (
                type(worker.get("returncode")) is not int
                or type(worker.get("timed_out")) is not bool
            ):
                raise ValueError("Invalid recorded worker exit receipt")
            result["process_exit"] = {
                "status": "RECORDED_ONLY",
                "receipt": worker,
                "new_parent_observation": False,
            }
            if worker["returncode"] != 0 or worker["timed_out"]:
                result["issues"].append("Recorded worker did not exit successfully")
        # Detect concurrent source modification rather than bind mixed snapshots.
        if any(_digest(run / name) != digest for name, digest in hashes.items()):
            raise ValueError("Source artifact changed during the offline audit")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        BadZipFile,
        EOFError,
    ) as exc:
        result["issues"].append(str(exc))
        result.setdefault("trials", _blocked_trials(str(exc)))
    result["source_artifact_sha256"] = hashes
    result["integrity_scope"] = (
        "Local consistency/change detection, not authentication or newly observed native execution"
    )
    if result["issues"]:
        result["status"] = "INVALID_INPUT"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--velocity-reference", required=True, choices=("root_link", "com")
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New JSON file; existing files are never overwritten",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output already exists; choose a new file")
    reports = [
        audit_run(path, velocity_reference=args.velocity_reference)
        for path in args.run_dirs
    ]
    result = {
        "version": VERSION,
        "qualification_passed": False,
        "exit_allowed": False,
        "runs": reports,
    }
    try:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Cannot write audit: {exc}\n")
    for report in reports:
        print(f"{report['status']}: {report['run_directory']}")
        print(json.dumps(report.get("component_counts", {}), sort_keys=True))
        for issue in report["issues"]:
            print("  " + issue)
    print(
        f"Audit: {args.output.resolve()}\nQualification: NOT ACCEPTED; exit gate remains closed."
    )
    return 2 if any(r["status"] == "INVALID_INPUT" for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
