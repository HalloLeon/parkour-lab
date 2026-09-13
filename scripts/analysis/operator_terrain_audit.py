"""Read-only CPU audit of completed progressive-teacher development checks.

Run from the repository root with ``python -m scripts.analysis.operator_terrain_audit RUN``.
Use --json for the full report on stdout. No run artifact, simulator, checkpoint
or RNG is modified.
This lives outside the hashed training producers intentionally: it is an offline
consumer, not a new training/evaluation interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    expected_producers = {
        str(p.relative_to(repo))
        for p in (
            *repo.joinpath("scripts/rsl_rl").glob("*.py"),
            *repo.joinpath("source/parkour_lab/parkour_lab").rglob("*.py"),
        )
    }
    if set(producers) != expected_producers:
        raise ValueError("Producer roster drift; use the recorded source revision")
    for name, digest in producers.items():
        path = (repo / name).resolve(strict=True)
        if not path.is_relative_to(repo) or file_sha256(path) != digest:
            raise ValueError(
                f"Producer drift: {name}; audit using the recorded source revision"
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
        "schema_version": "operator_terrain_offline_audit_v1",
        "status": "AUDITED",
        "run": str(run),
        "audit_sha256": file_sha256(Path(__file__)),
        "report_sha256": file_sha256(run / "report.json"),
        "training_env_sha256": file_sha256(run / "params/env.yaml"),
        "producer_hashes_verified": len(producers),
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
        f"{result['status']}: {result['producer_hashes_verified']} producer hashes; physical scores and motor targets replayed.",
        "First attempts only. Costs below are selected counterfactual terms, NOT complete training returns.",
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="Full machine-readable audit on stdout"
    )
    args = parser.parse_args(argv)
    try:
        result = audit_run(args.run)
        rendered = (
            json.dumps(result, indent=2, allow_nan=False)
            if args.json
            else format_summary(result)
        )
    except Exception as error:
        # Malformed archives may raise BadZipFile/EOFError as well as NumPy
        # errors. Do not catch user cancellation (KeyboardInterrupt/SystemExit).
        parser.exit(1, f"Audit failed: {error}\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
