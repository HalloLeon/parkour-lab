"""Read-only CPU audit of completed progressive-teacher development checks.

Run from the repository root with ``python -m scripts.analysis.operator_terrain_audit RUN``.
Default/--json only replays existing artifacts. Opt-in --capture SOURCE_CHECKPOINT
reuses the supervised teacher evaluator for frozen 500/3000 native diagnostics;
it never trains or overwrites the supplied run. Its consumer identity is separate
from the historical training producer identity.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from scripts.rsl_rl.operator_benchmark import validate_motor_trace
from scripts.rsl_rl.operator_benchmark_core import (
    DT,
    command_schedule,
    file_sha256,
    read_yaml_data,
    score_course_trace,
    score_trace,
)


FAMILIES = ("gap", "high_step", "hurdle", "tilted_ramps")
INITIAL = {
    "initial_position",
    "initial_quaternion",
    "env_origins",
    "terrain_levels",
    "terrain_columns",
}
FAILURES = ("base_contact", "course_chassis", "course_fall", "course_off_route")
TERM_FUNCTIONS = {
    "course_action_rate_l2": "isaaclab.envs.mdp.rewards:action_rate_l2",
    "course_ang_vel_xy_l2": "isaaclab.envs.mdp.rewards:ang_vel_xy_l2",
    "course_flat_orientation_l2": "isaaclab.envs.mdp.rewards:flat_orientation_l2",
    "course_base_clearance_below": "parkour_lab.tasks.manager_based.parkour_lab.mdp.reward_terms.safety:base_clearance_below_l2",
}

# Reviewed instrumentation-only migration from the completed v2 producer set.
# New/removed files and every model, command, physics or scorer change fail shut.
DIAGNOSTIC_ADAPTER_BASE = {
    "scripts/rsl_rl/operator_train.py": "c9b44d81742e9bb3470195ac00deb231f8a4d3042dce79e3e37605d96aa37656",
    "scripts/rsl_rl/operator_benchmark.py": "443e8043a30e5a30feb71a53d6a1baa0646352d9ab0523883c20e43fab2a8fcb",
    "scripts/rsl_rl/operator_control_trace.py": "ee201b00d3fd22e7d4dee20a216857ca814e0a32218ceb94bc961ee0aadf8006",
    "scripts/rsl_rl/operator_rewards.py": "6aeb923a97901bf0fc69759d930911fe2b552e4e7a0f6814a5cf9e600c5bcc5c",
}

# CPU replay executes these two consumers, not the retired simulator/trainer.
# Pin BOTH the recorded source and reviewed consumer. This permits unrelated
# task replacement without allowing arbitrary scorer changes or new GPU capture.
OFFLINE_REPLAY_SOURCES = {
    "scripts/rsl_rl/operator_benchmark.py": (
        "443e8043a30e5a30feb71a53d6a1baa0646352d9ab0523883c20e43fab2a8fcb",
        "2914c12937020a2b8d0133f924d657a29b2208c30f0abd5e26839e835832f90e",
    ),
    "scripts/rsl_rl/operator_benchmark_core.py": (
        "faee1501fc323fc7775d395ac37164afadb25c99d258217439b448048ad11ae9",
        "f1531a4e490b006439637e211ec26194322b28ba38f82097fca13e9b31070e95",
    ),
}


def validate_offline_replay_sources(original, current):
    """Validate the narrow, versioned CPU consumer boundary; never authorize capture."""
    evidence = {}
    for name, (recorded, reviewed) in OFFLINE_REPLAY_SOURCES.items():
        if original.get(name) != recorded or current.get(name) != reviewed:
            raise ValueError(f"Unreviewed offline replay source: {name}")
        evidence[name] = {"recorded_sha256": recorded, "consumer_sha256": reviewed}
    return evidence


def diagnostic_producer_delta(original, current):
    if original.keys() != current.keys():
        raise ValueError("Producer roster drift; use the recorded source revision")
    delta = {}
    for name, digest in original.items():
        if digest != current[name]:
            if DIAGNOSTIC_ADAPTER_BASE.get(name) != digest:
                raise ValueError(f"Unreviewed producer drift: {name}")
            delta[name] = {"training_sha256": digest, "consumer_sha256": current[name]}
    return delta


def first_attempt_end(terminated, time_out):
    """Include the first terminal sample, never an automatically reset episode."""
    if (
        terminated.ndim != 1
        or terminated.shape != time_out.shape
        or not terminated.size
        or terminated.dtype != np.bool_
        or time_out.dtype != np.bool_
    ):
        raise ValueError("Expected nonempty one-dimensional boolean reset masks")
    stops = np.flatnonzero(terminated | time_out)
    return int(stops[0]) + 1 if stops.size else len(terminated)


def selected_weights(saved):
    """Accept only the implemented formulas, not arbitrary functions from YAML."""
    if float(saved["sim"]["dt"]) * int(saved["decimation"]) != DT:
        raise ValueError("Expected the source-bound 50 Hz motor")
    result = {}
    for name, function in TERM_FUNCTIONS.items():
        term = saved["rewards"][name]
        binding = term["params"]
        inner = binding["term"]
        if (
            term["func"]
            not in (
                "operator_command:RoleReward",
                "scripts.rsl_rl.operator_command:RoleReward",
            )
            or set(binding) != {"operator", "term"}
            or binding["operator"] != "false"
            or inner["func"] != function
            or float(term["weight"]) != float(inner["weight"])
        ):
            raise ValueError(f"Unsupported course reward contract: {name}")
        params = inner["params"]
        allowed = set() if name == "course_action_rate_l2" else {"asset_cfg"}
        if not isinstance(params, dict) or not set(params).issubset(allowed):
            raise ValueError(f"Unsupported reward parameters: {name}")
        if "asset_cfg" in params and (
            not isinstance(params["asset_cfg"], dict)
            or params["asset_cfg"].get("name") != "robot"
        ):
            raise ValueError(f"Unsupported reward asset: {name}")
        result[name] = float(term["weight"])
    failure = saved["rewards"]["physical_failure"]
    if (
        failure["func"]
        not in (
            "operator_rewards:teacher_physical_failure",
            "scripts.rsl_rl.operator_rewards:teacher_physical_failure",
        )
        or failure["params"] != {}
    ):
        raise ValueError("Expected the once-per-failure impulse")
    result["observed_physical_failure"] = float(failure["weight"])
    if any(not np.isfinite(v) or v > 0 for v in result.values()):
        raise ValueError("Expected finite non-positive cost weights")
    return result


def course_trial_costs(trace, env_id, course, weights):
    """Selected counterfactual costs on evaluation states, NOT training returns.

    Actions/previous actions are paired in the delivered pre-action frame;
    velocities, quaternion, ray and failure bits are post-physics/pre-reset.
    The four continuous terms are integrated with dt. Observed failure reasons
    are unioned once without dt. Training-only persistent tilt is unavailable.
    """
    end = first_attempt_end(
        trace["terminated"][:, env_id], trace["time_out"][:, env_id]
    )
    q = trace["quaternion"][:end, env_id].astype(np.float64)
    if not np.isfinite(q).all() or not np.allclose(
        np.linalg.norm(q, axis=1), 1, atol=1e-5, rtol=0
    ):
        raise ValueError("Expected finite unit wxyz quaternions")
    w, x, y, z = q.T
    gravity_xy_squared = (2 * (x * z - w * y)) ** 2 + (2 * (y * z + w * x)) ** 2
    ray = trace["base_height_ray"][:end, env_id]
    ray_valid = np.isfinite(ray).all(axis=1)
    clearance = trace["position"][:end, env_id, 2] - np.where(ray_valid, ray[:, 2], 0)
    minimum = float(course["min_clearance"])
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError("Expected positive course clearance")
    failure = np.logical_or.reduce([trace[name][:end, env_id] for name in FAILURES])
    signals = {
        "course_action_rate_l2": np.square(
            trace["action"][:end, env_id] - trace["observation"][:end, env_id, 36:48]
        ).sum(axis=1),
        "course_ang_vel_xy_l2": np.square(
            trace["angular_velocity_b"][:end, env_id, :2]
        ).sum(axis=1),
        "course_flat_orientation_l2": gravity_xy_squared,
        "course_base_clearance_below": np.where(
            ray_valid, np.clip((minimum - clearance) / minimum, 0, 1) ** 2, 0
        ),
        "observed_physical_failure": failure.astype(float),
    }
    contributions = {
        name: values
        * weights[name]
        * (1 if name == "observed_physical_failure" else DT)
        for name, values in signals.items()
    }
    if any(not np.isfinite(values).all() for values in contributions.values()):
        raise ValueError("Nonfinite replayed cost")
    tail = slice(max(0, end - round(1 / DT)), end)
    return {
        "env_id": int(env_id),
        "duration_s": end * DT,
        "ended_in_reset": bool(
            trace["terminated"][end - 1, env_id] | trace["time_out"][end - 1, env_id]
        ),
        "waypoint_index": int(trace["waypoint_index"][end - 1, env_id]),
        "tail_duration_s": (end - tail.start) * DT,
        "tail_local_position_m": (
            trace["position"][tail, env_id] - trace["env_origins"][env_id]
        )
        .mean(axis=0)
        .tolist(),
        "tail_command_vx_m_s": float(trace["command"][tail, env_id, 0].mean()),
        "tail_actual_vx_m_s": float(trace["linear_velocity_b"][tail, env_id, 0].mean()),
        "selected_cost_sums": {
            name: float(value.sum()) for name, value in contributions.items()
        },
        "valid_base_ray_samples": int(ray_valid.sum()),
    }


def require_matching_report(actual, expected, path="report"):
    """Preserve decisions exactly; permit only sub-micro numeric aggregation drift."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"Schema mismatch: {path}")
        for key in expected:
            require_matching_report(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"List mismatch: {path}")
        for i, (a, b) in enumerate(zip(actual, expected)):
            require_matching_report(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        if not np.isfinite(expected) or not np.isclose(
            actual, expected, atol=1e-6, rtol=0
        ):
            raise ValueError(f"Metric mismatch: {path}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"Decision/value mismatch: {path}")


def audit_run(run):
    run = run.resolve(strict=True)

    def read(name):
        return json.loads((run / name).read_text())

    root, protocol = read("report.json"), read("terrain_protocol.json")
    if (
        root["status"] != "COMPLETED"
        or root["checks_not_run"]
        or protocol["check_updates"] != [500, 1500, 3000]
        or protocol["version"]
        not in (
            "go2_operator_terrain_progressive_v1",
            "go2_operator_terrain_progressive_task_critic_v2",
        )
        or root["promoted"] is not False
        or root["exit_gate_passed"] is not False
    ):
        raise ValueError(
            "Expected a complete, unpromoted 500/1500/3000 development run"
        )
    repo = Path(__file__).resolve().parents[2]
    producers = protocol["source_identity"]["producers"]
    consumers = validate_offline_replay_sources(
        producers,
        {name: file_sha256(repo / name) for name in OFFLINE_REPLAY_SOURCES},
    )
    weights = selected_weights(read_yaml_data(run / "params/env.yaml"))
    if weights["observed_physical_failure"] != protocol["physical_failure_impulse"]:
        raise ValueError("Saved failure weight differs from declared protocol")
    checks = []
    for step in protocol["check_updates"]:
        folder = run / "teacher_check" / f"model_{step}"
        report = json.loads((folder / "report.json").read_text())
        if (
            root["teacher_checks"][str(step)] != report
            or report["status"] != "MEASURED"
            or report["worker"] != {"returncode": 0, "timed_out": False}
            or report["source_identity"] != protocol["source_identity"]
            or report["checkpoint_sha256"] != file_sha256(run / f"model_{step}.pt")
            or report["learning_updates"] != step
            or list(folder.glob("*_cleanup_error.json"))
        ):
            raise ValueError(f"Incomplete or inconsistent check: {step}")
        for name in ("trace.npz", "measurement_report.json"):
            if file_sha256(folder / name) != report["sha256"][name]:
                raise ValueError(f"Changed check artifact: {step}/{name}")
        measurement = json.loads((folder / "measurement_report.json").read_text())
        # The normalized pre-action scan cannot reconstruct the unclipped,
        # post-action world-height mask used by stable_orientation_l2.
        with np.load(folder / "trace.npz", allow_pickle=False) as archive:
            trace = {k: archive[k] for k in archive.files if k != "terrain_observation"}
        if (
            trace["action"].shape != (1000, 160, 12)
            or not np.array_equal(trace["terrain_levels"], np.tile([0, 1, 3, 6], 40))
            or not np.array_equal(
                trace["terrain_columns"], np.repeat([0, 10, 20, 30], 40)
            )
        ):
            raise ValueError("Unexpected evaluation layout")
        motor = validate_motor_trace(trace)

        def sliced(ids):
            return {k: v[ids] if k in INITIAL else v[:, ids] for k, v in trace.items()}

        operator = score_trace(sliced(np.arange(0, 160, 4)), command_schedule(4)[0])
        require_matching_report(operator, measurement["operator"])
        courses, partial = {}, {}
        for family_index, family in enumerate(FAMILIES):
            for level in (1, 3, 6):
                key = f"{family}_L{level}"
                ids = np.flatnonzero(
                    (trace["terrain_columns"] == family_index * 10)
                    & (trace["terrain_levels"] == level)
                )
                metadata = measurement["course_metadata"][key]
                result = score_course_trace(sliced(ids), metadata)
                require_matching_report(result, measurement["courses"][key], key)
                if result["status"] != report["course_statuses"][key]:
                    raise ValueError(f"Course status mismatch: {key}")
                courses[key] = {k: result[k] for k in ("passed", "total", "status")}
                if level == 1:
                    partial[key] = [
                        dict(
                            course_trial_costs(trace, i, metadata["course"], weights),
                            passed=trial["passed"],
                        )
                        for i, trial in zip(ids, result["trials"])
                    ]
        if operator["status"] != report["operator_status"]:
            raise ValueError("Operator status mismatch")
        checks.append(
            {
                "update": step,
                "checkpoint_sha256": report["checkpoint_sha256"],
                "artifact_sha256": report["sha256"],
                "motor": motor,
                "operator_passed": sum(t["passed"] for t in operator["trials"]),
                "operator_total": len(operator["trials"]),
                "courses": courses,
                "first_attempt_L1": partial,
            }
        )
    return {
        "schema_version": "operator_terrain_offline_audit_v2",
        "status": "AUDITED",
        "run": str(run),
        "audit_sha256": file_sha256(Path(__file__)),
        "report_sha256": file_sha256(run / "report.json"),
        "training_env_sha256": file_sha256(run / "params/env.yaml"),
        "replay_sources_verified": consumers,
        "historical_simulator_producers_revalidated": False,
        "weights": weights,
        "checks": checks,
        "promoted": False,
        "exit_gate_passed": False,
        "scope": "CPU physical-score replay and selected counterfactual costs on evaluation trajectories; NOT logged training returns, a new rollout, action replay or a causal explanation.",
        "missing_for_full_objective": [
            "actual applied torque and post-step joint state (especially terminal frames)",
            "unclipped post-step scan for stable-orientation support mask",
            "complete native waypoint/route/stationary reward and sparse-event decomposition",
            "training-only persistent-tilt/workspace events and alternative-action outcomes",
        ],
        "interpretation": "Selected costs must not be summed or compared as complete policy returns. The observed-failure impulse excludes unrecorded training-only tilt; survivor tails and successful gap trials must be reported separately.",
    }


def format_summary(result):
    lines = [
        f"{result['status']}: {len(result['replay_sources_verified'])} pinned CPU replay sources; physical scores and motor targets replayed.",
        "First attempts only. Costs below are selected counterfactual terms, NOT complete training returns.",
        "Historical simulator/trainer reproduction is not asserted; GPU capture retains its stricter full-source check.",
    ]
    for check in result["checks"]:
        lines.append(
            f"Update {check['update']}: operator {check['operator_passed']}/{check['operator_total']}"
        )
        for name, trials in check["first_attempt_L1"].items():
            survivors = [t for t in trials if not t["ended_in_reset"]]
            result_row = check["courses"][name]
            tail = (
                f"; survivor tail vx/cmd {np.mean([t['tail_actual_vx_m_s'] for t in survivors]):.5f}/"
                f"{np.mean([t['tail_command_vx_m_s'] for t in survivors]):.3f} m/s"
                if survivors
                else ""
            )
            failure = np.mean(
                [t["selected_cost_sums"]["observed_physical_failure"] for t in trials]
            )
            clearance = np.mean(
                [t["selected_cost_sums"]["course_base_clearance_below"] for t in trials]
            )
            lines.append(
                f"  {name}: {result_row['passed']}/{result_row['total']} {result_row['status']}"
                f"; no-reset tails {len(survivors)}{tail}; mean failure/clearance {failure:.3f}/{clearance:.3f}"
            )
        higher = [
            f"{k}={v['passed']}/{v['total']}"
            for k, v in check["courses"].items()
            if not k.endswith("_L1")
        ]
        lines.append("  " + ", ".join(higher))
    lines.extend(
        [
            "Missing: actual torque, terminal joint state, post-step scan mask and full native reward/event decomposition.",
            "No training or simulator launched; no acceptance/promotion. Use --json for per-trial values and hashes.",
        ]
    )
    return "\n".join(lines)


def summarize_native_capture(trace, interface, baseline, measurement):
    """Terminal-inclusive native costs; no reset bridging or causal attribution."""
    selected = np.flatnonzero(np.isin(trace["terrain_levels"], [0, 1]))
    names = interface["reward_names"]
    contributions = trace["reward_contribution"]
    if (
        contributions.shape != (*trace["terminated"].shape, len(names))
        or not np.isfinite(contributions).all()
    ):
        raise ValueError("Invalid native reward contributions")
    error = np.abs(contributions.sum(-1) - trace["reward_total"])
    if not np.allclose(
        contributions.sum(-1), trace["reward_total"], atol=2e-6, rtol=1e-5
    ):
        raise ValueError("Native reward checksum mismatch")
    trials = []
    for i in selected:
        end = first_attempt_end(trace["terminated"][:, i], trace["time_out"][:, i])
        joint_error = (
            trace["joint_position_target"][:end, i]
            - trace["joint_position_post"][:end, i]
        )
        computed = trace["computed_torque_substeps"][:end, i]
        applied = trace["applied_torque_substeps"][:end, i]
        if (
            computed.shape != (end, 4, 12)
            or applied.shape != computed.shape
            or not np.isfinite(computed).all()
            or not np.isfinite(applied).all()
        ):
            raise ValueError("Invalid four-substep actuator capture")
        sums = contributions[:end, i].sum(0, dtype=np.float64)
        clipping = np.abs(computed - applied)
        # Do not conflate operator-role returns with course-role returns.
        # Only name windows whose actual commands satisfy the claimed condition.
        forward = end >= 300 and np.allclose(
            trace["command"][100:300, i], [0.55, 0, 0], atol=1e-6, rtol=0
        )
        trials.append(
            {
                "env_id": int(i),
                "role": "operator" if i % 4 == 0 else "course",
                "family": FAMILIES[int(trace["terrain_columns"][i]) // 10],
                "level": int(trace["terrain_levels"][i]),
                "duration_s": end * DT,
                "failure_terms": [k for k in FAILURES if trace[k][:end, i].any()],
                "reward_sums": dict(zip(names, sums.tolist())),
                "post_hold_reward_rate_bins": {
                    label: {
                        "start_s": start * DT,
                        "end_s": stop * DT,
                        "weighted_rates": dict(
                            zip(
                                names,
                                (contributions[start:stop, i].mean(0) / DT).tolist(),
                            )
                        ),
                        "clipped_substep_joint_fraction": float(
                            np.mean(clipping[start:stop] > 1e-5)
                        ),
                    }
                    for label, start, stop in (
                        ("post_hold", 100, end),
                        ("last_second", max(100, end - 50), end),
                    )
                    if stop > start
                },
                "impulse_events": {
                    name: [
                        {
                            "post_step_time_s": (int(t) + 1) * DT,
                            "contribution": float(contributions[t, i, j]),
                        }
                        for t in np.flatnonzero(contributions[:end, i, j])
                    ]
                    for j, name in enumerate(names)
                    if name
                    in (
                        "physical_failure",
                        "course_completed_course",
                        "course_intermediate_milestone",
                    )
                },
                "joint_target_error_rms_rad": float(np.sqrt(np.mean(joint_error**2))),
                "max_abs_applied_torque_nm": float(np.max(np.abs(applied))),
                "max_torque_clipping_nm": float(clipping.max()),
                "clipped_substep_joint_fraction": float(np.mean(clipping > 1e-5)),
                "invalid_post_scan_fraction": float(
                    np.mean(~np.isfinite(trace["height_scan_post"][:end, i]))
                ),
                "operator_forward_2_to_6_s_vx": (
                    float(trace["linear_velocity_b"][100:300, i, 0].mean())
                    if i % 4 == 0 and forward
                    else None
                ),
            }
        )
    reproduction = {
        "matched": True,
        "bitwise_equal": True,
        "atol": 1e-5,
        "rtol": 1e-5,
        "differences": {},
        "scope": "Tolerance-based first-attempt trace comparison PLUS exact operator/L1 decision/failure parity and 1e-6 absolute metric tolerance; not necessarily bitwise reproduction",
    }
    # Compare every original field, including commands, actor frames, scans,
    # contacts and physical event bits; only the selected first attempts count.
    with np.load(baseline / "trace.npz", allow_pickle=False) as original:
        old_end = np.array(
            [
                first_attempt_end(
                    original["terminated"][:, i], original["time_out"][:, i]
                )
                for i in selected
            ]
        )
        new_end = np.array(
            [
                first_attempt_end(trace["terminated"][:, i], trace["time_out"][:, i])
                for i in selected
            ]
        )
        mask = (
            np.arange(len(trace["terminated"]))[:, None]
            < np.minimum(old_end, new_end)[None, :]
        )
        if not np.array_equal(old_end, new_end):
            reproduction["differences"]["first_attempt_end"] = {
                "original": old_end.tolist(),
                "captured": new_end.tolist(),
            }
        for name in original.files:
            old, new = original[name], trace[name]
            if old.shape != new.shape:
                raise ValueError(f"Reproduction layout mismatch: {name}")
            old, new = (
                (old[selected], new[selected])
                if name in INITIAL
                else (old[:, selected][mask], new[:, selected][mask])
            )
            reproduction["bitwise_equal"] &= np.array_equal(old, new, equal_nan=True)
            if old.dtype.kind in "biu":
                match = old == new
            else:
                match = np.isclose(old, new, atol=1e-5, rtol=1e-5, equal_nan=True)
            if not match.all():
                finite = np.isfinite(old) & np.isfinite(new)
                reproduction["differences"][name] = {
                    "different_values": int((~match).sum()),
                    "max_finite_abs_error": (
                        float(np.max(np.abs(old[finite] - new[finite])))
                        if finite.any()
                        else None
                    ),
                }
    archived = json.loads((baseline / "measurement_report.json").read_text())
    for key, actual, expected in [
        ("operator_scores", measurement["operator"], archived["operator"]),
        *(
            (name, value, archived["courses"][name])
            for name, value in measurement["courses"].items()
        ),
    ]:
        try:
            require_matching_report(actual, expected, key)
        except ValueError as mismatch:
            reproduction["differences"][key] = str(mismatch)
    reproduction["matched"] = not reproduction["differences"]
    return {
        "interface": interface,
        "analyzed_env_ids": selected.tolist(),
        "reward_checksum_max_error": float(error.max()),
        "reproduction": reproduction,
        "trials": trials,
        "scope": "Frozen-policy evaluation objective excluding training-only tilt and PPO losses. L0 rows are operator controls, NOT course-L0 controls. L3/L6 remain background fixtures, unscored. Divergence invalidates attribution to archived trajectories; these costs do not prove a cause or justify changed physical gates.",
    }


def capture_run(args):
    """Reuse one supervised evaluation path, fixed endpoints, no checkpoint search."""
    import torch
    from scripts.rsl_rl import operator_train as training

    run = args.run.resolve(strict=True)
    source = args.capture.resolve(strict=True)
    original = json.loads((run / "terrain_protocol.json").read_text())
    root = json.loads((run / "report.json").read_text())
    if (
        original["version"] != training.TERRAIN_TASK_CRITIC_VERSION
        or root["status"] != "COMPLETED"
        or root["checks_not_run"]
    ):
        raise ValueError("Native capture requires the completed v2 teacher run")
    mesh_report = (
        args.mesh_flat_report or Path(original["mesh_flat"]["report"])
    ).resolve(strict=True)
    worker_args = SimpleNamespace(
        checkpoint=source,
        mesh_flat_report=mesh_report,
        device=args.device,
        seed=42,
        num_envs=3200,
        iterations=3000,
        terrain_critic_context=True,
        terrain_diagnostics=run,
        terrain_evaluate_checkpoint=None,
    )
    consumer = training.terrain_execution_identity(worker_args)
    historical = original["source_identity"]
    if (
        consumer["runtime"]["packages"]["rsl-rl-lib"]
        != historical["packages"]["rsl-rl-lib"]
    ):
        raise ValueError("Capture requires the recorded RSL-RL version")
    for key in ("checkpoint", "agent", "environment", "mesh_flat_report"):
        if historical[key] != consumer["runtime"][key]:
            raise ValueError(f"Changed diagnostic source input: {key}")
    delta = diagnostic_producer_delta(
        historical["producers"], consumer["runtime"]["producers"]
    )
    agent = read_yaml_data(source.parent / "params/agent.yaml")
    saved = read_yaml_data(source.parent / "params/env.yaml")
    basis = {
        str(p.relative_to(run)): file_sha256(p)
        for p in (
            run / "report.json",
            run / "terrain_protocol.json",
            run / "params/env.yaml",
            *(
                run / "teacher_check" / f"model_{s}" / name
                for s in (500, 3000)
                for name in (
                    "params/env.yaml",
                    "trace.npz",
                    "measurement_report.json",
                    "report.json",
                )
            ),
        )
    }
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        stock = training.load_reference_checkpoint(source, agent)["model_state_dict"]
        observations = {
            "policy": torch.zeros(4, 48),
            "terrain": torch.ones(4, 264),
            "critic_context": torch.zeros(4, 4),
        }
        for step in (500, 3000):
            report = json.loads(
                (run / "teacher_check" / f"model_{step}" / "report.json").read_text()
            )
            if (
                report != root["teacher_checks"][str(step)]
                or report["source_identity"] != historical
                or report["status"] != "MEASURED"
                or report["worker"] != {"returncode": 0, "timed_out": False}
            ):
                raise ValueError(f"Incomplete historical check: {step}")
            for name in ("trace.npz", "measurement_report.json"):
                if (
                    report["sha256"][name]
                    != basis[f"teacher_check/model_{step}/{name}"]
                ):
                    raise ValueError(f"Changed historical artifact: {step}/{name}")
            checkpoint = run / f"model_{step}.pt"
            if file_sha256(checkpoint) != report["checkpoint_sha256"]:
                raise ValueError(f"Changed checkpoint: {step}")
            _, metadata = training.load_terrain_teacher(
                checkpoint,
                observations,
                stock,
                historical,
                expected_version=original["version"],
            )
            if metadata["learning_updates"] != step:
                raise ValueError("Wrong checkpoint update")
    protocol = {
        "version": original["version"],
        "source_identity": historical,
        "mesh_flat": original["mesh_flat"],
        "diagnostic_consumer_identity": consumer,
        "diagnostics": {
            "version": "native_reward_actuator_v1",
            "source_run": str(run),
            "basis_sha256": basis,
            "producer_delta": delta,
            "check_updates": [500, 3000],
            "scope": "Frozen evaluation only; native reward capture minus unavailable training-only tilt impulse; stock physics/commands/scorers; no promotion",
        },
    }
    if args.validate_only:
        print(
            "Validated frozen 500/3000 checkpoints, original source and diagnostic producer delta. GPU configuration/reproduction remain UNRUN; no output or training."
        )
        return 0
    if args.worker_output is not None:
        selected = run / f"model_{args.capture_update}.pt"
        protocol["evaluation_checkpoint_sha256"] = file_sha256(selected)
        if (
            json.loads((args.worker_output / "terrain_protocol.json").read_text())
            != protocol
        ):
            raise ValueError(
                "Diagnostic inputs/consumers changed after parent preflight"
            )
        worker_args.terrain_evaluate_checkpoint = selected
        training.run_terrain_teacher(
            worker_args, args.worker_output, saved, agent, protocol
        )
        return 0

    # Complete the existing score replay before allocating a GPU process.
    audit_run(run)
    return supervise_native_capture(args, worker_args, protocol)


def supervise_native_capture(args, worker_args, protocol):
    """Persist partial checks on ordinary failures; never turn cleanup into success."""
    from scripts.rsl_rl import operator_train as training
    from scripts.rsl_rl.operator_benchmark import supervise, write_json
    from scripts.rsl_rl.run_provenance import write_run_provenance

    run, source = worker_args.terrain_diagnostics, worker_args.checkpoint
    mesh_report = worker_args.mesh_flat_report
    consumer = protocol["diagnostic_consumer_identity"]
    basis = protocol["diagnostics"]["basis_sha256"]
    output = None
    result = {
        "status": "RUNNING",
        "checks": {},
        "promoted": False,
        "exit_gate_passed": False,
    }
    try:
        parent = args.output_parent or run
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="native_capture_", dir=parent)).resolve()
        write_run_provenance(output, Path(__file__))
        write_json(output / "terrain_protocol.json", protocol)
        write_json(output / "report.json", result)
        for step in (500, 3000):
            folder = output / f"model_{step}"
            folder.mkdir()
            write_json(
                folder / "terrain_protocol.json",
                {
                    **protocol,
                    "evaluation_checkpoint_sha256": file_sha256(
                        run / f"model_{step}.pt"
                    ),
                },
            )
            command = [
                sys.executable,
                "-u",
                "-m",
                "scripts.analysis.operator_terrain_audit",
                str(run),
                "--capture",
                str(source),
                "--mesh-flat-report",
                str(mesh_report),
                "--device",
                args.device,
                "--worker-output",
                str(folder),
                "--capture-update",
                str(step),
            ]
            check = supervise(
                command,
                folder,
                timeout_s=args.timeout,
                report_filename="training_status.json",
                valid_statuses=("MEASURED", "REPRODUCTION_DIVERGED", "ERROR"),
            )
            if check["status"] in ("MEASURED", "REPRODUCTION_DIVERGED") and (
                check.get("diagnostic_consumer_identity") != consumer
                or check.get("source_identity") != protocol["source_identity"]
                or check.get("learning_updates") != step
                or training.terrain_execution_identity(worker_args) != consumer
                or check.get("checkpoint_sha256")
                != file_sha256(run / f"model_{step}.pt")
                or check.get("sha256")
                != {
                    name: file_sha256(folder / name)
                    for name in ("trace.npz", "measurement_report.json")
                }
                or any(
                    file_sha256(run / name) != digest for name, digest in basis.items()
                )
            ):
                check = {
                    "status": "ERROR",
                    "error": "Diagnostic identity/artifacts changed",
                    "measurement_result": check,
                }
            result["checks"][str(step)] = check
            write_json(folder / "report.json", check)
            result["status"] = (
                ("CAPTURED" if len(result["checks"]) == 2 else "RUNNING")
                if check["status"] == "MEASURED"
                else check["status"]
            )
            write_json(output / "report.json", result)
            print(f"Frozen {step}: {check['status']}", flush=True)
            if check["status"] != "MEASURED":
                break
        print(
            f"{result['status']}: {output / 'report.json'}\nNo training or promotion. Review native_control in each measurement_report.json before choosing a learning intervention."
        )
        return 0 if result["status"] == "CAPTURED" else 2
    except Exception as error:
        result.update(status="ERROR", error=str(error), error_type=type(error).__name__)
        if output is not None:
            try:
                write_json(output / "report.json", result)
            except Exception as publication_error:
                print(
                    f"Could not publish capture error: {publication_error}",
                    file=sys.stderr,
                )
        print(f"Capture ERROR: {error}", file=sys.stderr)
        return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="Full machine-readable audit on stdout"
    )
    parser.add_argument(
        "--capture",
        type=Path,
        metavar="SOURCE_CHECKPOINT",
        help="Opt-in frozen 500/3000 native reward/actuator capture; original stock source checkpoint required; no training",
    )
    parser.add_argument(
        "--mesh-flat-report",
        type=Path,
        help="Relocated copy of the recorded prerequisite, if needed",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument(
        "--timeout", type=float, default=1800, help="Seconds per capture worker"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Capture CPU preflight; no simulator or output",
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--capture-update", type=int, choices=(500, 3000), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    try:
        if args.capture is not None:
            if (
                args.json
                or not math.isfinite(args.timeout)
                or args.timeout <= 0
                or ((args.worker_output is None) != (args.capture_update is None))
                or (args.validate_only and args.worker_output is not None)
            ):
                raise ValueError("Invalid capture options")
            return capture_run(args)
        if (
            args.validate_only
            or args.worker_output is not None
            or args.capture_update is not None
            or args.mesh_flat_report is not None
            or args.output_parent is not None
        ):
            raise ValueError("Capture options require --capture SOURCE_CHECKPOINT")
        result = audit_run(args.run)
        rendered = (
            json.dumps(result, indent=2, allow_nan=False)
            if args.json
            else format_summary(result)
        )
    except Exception as error:
        # Malformed archives may raise BadZipFile/EOFError as well as NumPy
        # errors. Do not catch user cancellation (KeyboardInterrupt/SystemExit).
        parser.exit(2 if args.capture is not None else 1, f"Audit failed: {error}\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
