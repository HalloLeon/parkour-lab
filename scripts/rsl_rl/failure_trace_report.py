"""Describe one exact startup trace without simulation, numpy, or causal verdicts.

The source is never modified. Commands and velocity-estimator targets are read
before each action; robot/contact/action outcomes use retained pre-reset post
states. Report thresholds are conventions, not simulator termination criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from .startup_report import validate_startup_report
except ImportError:
    from startup_report import validate_startup_report


NOTICE = (
    "Descriptive observed telemetry only; no causal diagnosis or reliability/pass verdict.",
    "A command window is complete only when a following pre-action sample shows a command-mode "
    "transition (or a changed pivot yaw request). Termination/step-limit endings are incomplete.",
    "Phase classification uses pre-action intent speed/yaw and effective target speed. "
    "Changing waypoint direction alone does not split a phase.",
    "Foot velocity is world-space rigid-body-origin velocity, not contact-point slip. "
    "Contact masks use the explicit reporting force-norm threshold, not the termination threshold.",
    "Computed/applied torque differences and target-clipping flags describe different mechanisms; "
    "neither establishes why motion failed. Estimator error alone does not establish causation.",
    "Pivot acquisition is the first 1 s of observed phase action starts; sustained starts at 1 s. "
    "Late means the last complete onset-aligned 0.4 s block, not a detected stride. "
    "No complete late block is reported if its boundaries do not align with control steps.",
)
VECTOR_3 = (
    "root_position_w_m",
    "root_position_env_m",
    "linear_velocity_body_m_s",
    "linear_velocity_w_m_s",
    "angular_velocity_body_rad_s",
    "angular_velocity_w_rad_s",
)
JOINT_VECTORS = (
    "joint_position_rad",
    "joint_velocity_rad_s",
    "joint_position_target_rad",
    "joint_computed_torque_nm",
    "joint_applied_torque_nm",
    "environment_action",
    "delayed_raw_action",
    "processed_joint_target_rad",
    "affine_joint_target_rad",
    "configured_clip_joint_target_rad",
    "reconstructed_safe_joint_target_rad",
)
FOOT_MATRICES = (
    "foot_position_w_m",
    "foot_position_env_m",
    "foot_force_w_n",
    "foot_linear_velocity_w_m_s",
)


def _number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _vector(value, size, label, *, boolean=False):
    valid = (lambda item: type(item) is bool) if boolean else _number
    if (
        not isinstance(value, list)
        or len(value) != size
        or not all(valid(item) for item in value)
    ):
        raise ValueError(
            f"{label}: expected {size} {'boolean' if boolean else 'finite numeric'} values"
        )


def _matrix(value, rows, columns, label):
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{label}: expected {rows} x {columns} matrix")
    for index, row in enumerate(value):
        _vector(row, columns, f"{label}[{index}]")


def _names(value, size, label):
    if (
        not isinstance(value, list)
        or len(value) != size
        or not all(isinstance(name, str) and name for name in value)
        or len(set(value)) != size
    ):
        raise ValueError(f"{label}: expected {size} unique nonempty names")


def validate_failure_trace(report):
    """Extend the existing envelope validator with the current analyzed shapes."""
    validate_startup_report(report, allow_empty_interruption=True)
    capture = report["metadata"].get("capture_metadata")
    if not isinstance(capture, dict):
        raise ValueError("metadata.capture_metadata: missing")
    _names(capture.get("joint_names"), 12, "capture_metadata.joint_names")
    _names(capture.get("foot_names"), 4, "capture_metadata.foot_names")
    for index, sample in enumerate(report["samples"]):
        _vector(sample["policy_action"], 12, f"samples[{index}].policy_action")
        _vector(
            sample["pre"]["observations"]["velocity_target"],
            3,
            f"samples[{index}].pre.observations.velocity_target",
        )
        for when in ("pre", "post"):
            state = sample[when]["state"]
            label = f"samples[{index}].{when}.state"
            for name in VECTOR_3:
                _vector(state.get(name), 3, f"{label}.{name}")
            _vector(
                state.get("root_orientation_wxyz"), 4, f"{label}.root_orientation_wxyz"
            )
            norm = math.hypot(*state["root_orientation_wxyz"])
            if not math.isfinite(norm) or abs(norm - 1.0) > 1e-3:
                raise ValueError(
                    f"{label}.root_orientation_wxyz: expected unit quaternion (norm tolerance 0.001)"
                )
            _vector(state.get("intent_command"), 4, f"{label}.intent_command")
            if (
                not _number(state.get("target_speed_m_s"))
                or state["target_speed_m_s"] < 0
                or state["intent_command"][2] < 0
            ):
                raise ValueError(
                    f"{label}: target and preferred speeds must be finite and nonnegative"
                )
            for name in JOINT_VECTORS:
                _vector(state.get(name), 12, f"{label}.{name}")
            for name in ("configured_target_clipped", "safe_target_clipped"):
                _vector(state.get(name), 12, f"{label}.{name}", boolean=True)
            for name in (
                "joint_soft_position_limits_rad",
                "safe_joint_target_limits_rad",
            ):
                _matrix(state.get(name), 12, 2, f"{label}.{name}")
                if any(low > high for low, high in state[name]):
                    raise ValueError(f"{label}.{name}: lower limit exceeds upper limit")
            for name in FOOT_MATRICES:
                _matrix(state.get(name), 4, 3, f"{label}.{name}")


def _mean(values):
    return sum(values) / len(values) if values else None


def _delta(left, right):
    return [a - b for a, b in zip(left, right)]


def _abs_stats(rows):
    flat = [abs(value) for row in rows for value in row]
    return {
        "mean": _mean(flat),
        "max": max(flat) if flat else None,
        "per_joint_mean": [_mean([abs(row[j]) for row in rows]) for j in range(12)],
    }


def _euler(quaternion):
    norm = math.hypot(*quaternion)
    w, x, y, z = (value / norm for value in quaternion)
    return (
        math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
        math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
    )


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _command_key(sample):
    state = sample["pre"]["state"]
    _, _, preferred_speed, yaw = state["intent_command"]
    if state["target_speed_m_s"] > 0:
        return ("translation_with_yaw" if yaw != 0 else "translation", None)
    if preferred_speed > 0:
        return ("terminal_hold", None)
    if yaw != 0:
        return ("pivot", yaw)
    return ("stop", None)


def _clip_stats(states, key):
    flags = [state[key] for state in states]
    return {
        "joint_sample_fraction": sum(map(sum, flags)) / (12 * len(flags)),
        "any_joint_sample_fraction": sum(any(row) for row in flags) / len(flags),
        "per_joint_fraction": [_mean([row[j] for row in flags]) for j in range(12)],
    }


def _summarize(samples, *, phase_anchor, foot_names, contact_threshold):
    if not samples:
        return None
    pre = [sample["pre"]["state"] for sample in samples]
    post = [sample["post"]["state"] for sample in samples]
    dt = samples[0]["time_after_s"] - samples[0]["time_before_s"]
    errors = [
        _delta(
            sample["pre"]["estimated_velocity_body_m_s"],
            state["linear_velocity_body_m_s"],
        )
        for sample, state in zip(samples, pre)
    ]
    actions = [sample["policy_action"] for sample in samples]
    # Use the delayed affine request and the executed processed target, not
    # target-minus-joint tracking error. This includes both target clamps.
    target_overflow = [
        _delta(state["affine_joint_target_rad"], state["processed_joint_target_rad"])
        for state in post
    ]
    masks = [
        [math.hypot(*force) > contact_threshold for force in state["foot_force_w_n"]]
        for state in post
    ]
    mask_counts = {}
    for mask in masks:
        key = "".join("1" if flag else "0" for flag in mask)
        mask_counts[key] = mask_counts.get(key, 0) + 1
    angles = [_euler(state["root_orientation_wxyz"]) for state in post]
    previous_yaw = _euler(pre[0]["root_orientation_wxyz"])[2]
    heading_change = 0.0
    for _, _, yaw in angles:
        heading_change += _wrap(yaw - previous_yaw)
        previous_yaw = yaw
    commanded_yaw = [state["intent_command"][3] for state in pre]
    achieved_yaw = [state["angular_velocity_w_rad_s"][2] for state in post]
    nonzero_yaw = [
        (command, actual)
        for command, actual in zip(commanded_yaw, achieved_yaw)
        if command != 0
    ]
    feet = {}
    for j, name in enumerate(foot_names):
        speeds = [math.hypot(*state["foot_linear_velocity_w_m_s"][j]) for state in post]
        contact_speeds = [speed for speed, mask in zip(speeds, masks) if mask[j]]
        feet[name] = {
            "contact_fraction": _mean([mask[j] for mask in masks]),
            "body_origin_world_speed_m_s_mean": _mean(speeds),
            "body_origin_world_speed_m_s_max": max(speeds),
            "contact_conditioned_body_origin_world_speed_m_s_mean": _mean(
                contact_speeds
            ),
        }
    result = {
        "sample_count": len(samples),
        "start_s": samples[0]["time_before_s"],
        "end_s": samples[-1]["time_after_s"],
        "duration_s": len(samples) * dt,
        "velocity_estimation": {
            "timing": "pre estimate minus pre true body velocity",
            "component_mae_m_s": [
                _mean([abs(row[j]) for row in errors]) for j in range(3)
            ],
            "mean_component_mae_m_s": _mean(
                [abs(value) for row in errors for value in row]
            ),
            "error_norm_m_s_mean": _mean([math.hypot(*row) for row in errors]),
            "error_norm_m_s_max": max(math.hypot(*row) for row in errors),
        },
        "action_and_motion": {
            "requested_to_processed_target_overflow": {
                "abs_rad": _abs_stats(target_overflow),
                "sum_square_rad2_mean": _mean(
                    [sum(value * value for value in row) for row in target_overflow]
                ),
                # Same resolved action-joint order as the top-level joint_names.
                "per_joint_abs_rad_mean": [
                    _mean([abs(row[j]) for row in target_overflow]) for j in range(12)
                ],
                "per_joint_abs_rad_max": [
                    max(abs(row[j]) for row in target_overflow) for j in range(12)
                ],
            },
            "policy_action_abs": _abs_stats(actions),
            "policy_action_adjacent_delta_abs": _abs_stats(
                [_delta(b, a) for a, b in zip(actions, actions[1:])]
            ),
            "policy_action_adjacent_delta_count": max(0, len(samples) - 1),
            "processed_target_pre_post_delta_abs_rad": _abs_stats(
                [
                    _delta(
                        b["processed_joint_target_rad"], a["processed_joint_target_rad"]
                    )
                    for a, b in zip(pre, post)
                ]
            ),
            "joint_pre_post_motion_abs_rad": _abs_stats(
                [
                    _delta(b["joint_position_rad"], a["joint_position_rad"])
                    for a, b in zip(pre, post)
                ]
            ),
            "joint_velocity_abs_rad_s": _abs_stats(
                [state["joint_velocity_rad_s"] for state in post]
            ),
            "policy_to_environment_action_delta_abs": _abs_stats(
                [
                    _delta(action, state["environment_action"])
                    for action, state in zip(actions, post)
                ]
            ),
            "environment_to_delayed_action_delta_abs": _abs_stats(
                [
                    _delta(state["environment_action"], state["delayed_raw_action"])
                    for state in post
                ]
            ),
            "processed_to_reconstructed_safe_target_delta_abs_rad": _abs_stats(
                [
                    _delta(
                        state["processed_joint_target_rad"],
                        state["reconstructed_safe_joint_target_rad"],
                    )
                    for state in post
                ]
            ),
            "actual_to_processed_target_delta_abs_rad": _abs_stats(
                [
                    _delta(
                        state["joint_position_target_rad"],
                        state["processed_joint_target_rad"],
                    )
                    for state in post
                ]
            ),
            "actual_target_to_joint_position_error_abs_rad": _abs_stats(
                [
                    _delta(
                        state["joint_position_target_rad"], state["joint_position_rad"]
                    )
                    for state in post
                ]
            ),
            "configured_target_clipping": _clip_stats(
                post, "configured_target_clipped"
            ),
            "safe_target_clipping": _clip_stats(post, "safe_target_clipped"),
        },
        "torque": {
            "computed_abs_nm": _abs_stats(
                [state["joint_computed_torque_nm"] for state in post]
            ),
            "applied_abs_nm": _abs_stats(
                [state["joint_applied_torque_nm"] for state in post]
            ),
            "computed_minus_applied_abs_nm": _abs_stats(
                [
                    _delta(
                        state["joint_computed_torque_nm"],
                        state["joint_applied_torque_nm"],
                    )
                    for state in post
                ]
            ),
        },
        "feet": feet,
        "contact_mask_counts": mask_counts,
        "all_feet_contact_fraction": sum(all(mask) for mask in masks) / len(masks),
        "no_feet_contact_fraction": sum(not any(mask) for mask in masks) / len(masks),
        "attitude": {
            "max_abs_roll_rad": max(abs(row[0]) for row in angles),
            "max_abs_pitch_rad": max(abs(row[1]) for row in angles),
        },
        "yaw": {
            "mean_world_z_rate_rad_s": _mean(achieved_yaw),
            "mean_command_aligned_rate_rad_s": _mean(
                [math.copysign(1, command) * actual for command, actual in nonzero_yaw]
            ),
            "mean_abs_rate_error_rad_s": _mean(
                [abs(a - b) for a, b in zip(commanded_yaw, achieved_yaw)]
            ),
            "wrong_sign_fraction": _mean(
                [command * actual < 0 for command, actual in nonzero_yaw]
            ),
            "commanded_angle_rad": sum(commanded_yaw) * dt,
            "world_z_rate_integral_rad": sum(achieved_yaw) * dt,
            "unwrapped_euler_heading_change_rad": heading_change,
        },
        "max_xy_excursion_from_phase_onset_m": max(
            math.hypot(
                state["root_position_w_m"][0] - phase_anchor[0],
                state["root_position_w_m"][1] - phase_anchor[1],
            )
            for state in post
        ),
        "planar_speed_m_s_mean": _mean(
            [math.hypot(*state["linear_velocity_w_m_s"][:2]) for state in post]
        ),
        "target_speed_m_s_range": [
            min(state["target_speed_m_s"] for state in pre),
            max(state["target_speed_m_s"] for state in pre),
        ],
    }
    return result


def _threshold(value, name, *, optional=False):
    if optional and value is None:
        return
    if not _number(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


def analyze_failure_trace(
    report,
    *,
    contact_threshold=1.0,
    estimate_error_threshold=None,
    torque_difference_threshold=None,
):
    validate_failure_trace(report)
    if report["metadata"].get("action_source", "policy") != "policy":
        raise ValueError(
            "Action-replay probes are not policy rollouts; use startup_probe_report.py."
        )
    _threshold(contact_threshold, "contact_threshold")
    _threshold(estimate_error_threshold, "estimate_error_threshold", optional=True)
    _threshold(
        torque_difference_threshold, "torque_difference_threshold", optional=True
    )
    samples, meta = report["samples"], report["metadata"]
    foot_names = meta["capture_metadata"]["foot_names"]
    joint_names = meta["capture_metadata"]["joint_names"]
    output = {
        "kind": "failure_trace_summary",
        "schema_version": 1,
        "source_metadata": meta,
        "stop_reason": report["stop_reason"],
        "capture_error": report.get("error"),
        "sample_count": len(samples),
        "captured_duration_s": len(samples) * meta["step_dt_s"],
        "terminal_retained": bool(samples and samples[-1]["post"]["done"]),
        "termination": [
            key for key, flag in samples[-1]["post"]["termination"].items() if flag
        ]
        if samples
        else [],
        "reporting_thresholds": {
            "contact_force_norm_strictly_greater_n": contact_threshold,
            "pre_velocity_error_norm_strictly_greater_m_s": estimate_error_threshold,
            "post_torque_difference_abs_strictly_greater_nm": torque_difference_threshold,
        },
        "joint_names": joint_names,
        "foot_names": foot_names,
        "first_events": {},
        "phases": [],
        "limitations": list(NOTICE),
    }
    events = output["first_events"]
    for name in (
        "configured_target_clipping",
        "safe_target_clipping",
        "estimate_error_threshold",
        "torque_difference_threshold",
        "fewer_than_four_contacts",
    ):
        events[name] = None
    for sample in samples:
        pre, post = sample["pre"]["state"], sample["post"]["state"]
        for name, field in (
            ("configured_target_clipping", "configured_target_clipped"),
            ("safe_target_clipping", "safe_target_clipped"),
        ):
            if events[name] is None and any(post[field]):
                events[name] = {
                    "time_s": sample["time_after_s"],
                    "timing": "post_physics_pre_reset",
                    "joints": [
                        joint_names[j] for j, flag in enumerate(post[field]) if flag
                    ],
                }
        error = math.hypot(
            *_delta(
                sample["pre"]["estimated_velocity_body_m_s"],
                pre["linear_velocity_body_m_s"],
            )
        )
        if (
            estimate_error_threshold is not None
            and error > estimate_error_threshold
            and events["estimate_error_threshold"] is None
        ):
            events["estimate_error_threshold"] = {
                "time_s": sample["time_before_s"],
                "timing": "pre_action",
                "error_norm_m_s": error,
            }
        torque_error = max(
            abs(value)
            for value in _delta(
                post["joint_computed_torque_nm"], post["joint_applied_torque_nm"]
            )
        )
        if (
            torque_difference_threshold is not None
            and torque_error > torque_difference_threshold
            and events["torque_difference_threshold"] is None
        ):
            events["torque_difference_threshold"] = {
                "time_s": sample["time_after_s"],
                "timing": "post_physics_pre_reset",
                "maximum_abs_difference_nm": torque_error,
            }
        mask = [
            math.hypot(*force) > contact_threshold for force in post["foot_force_w_n"]
        ]
        if not all(mask) and events["fewer_than_four_contacts"] is None:
            events["fewer_than_four_contacts"] = {
                "time_s": sample["time_after_s"],
                "timing": "post_physics_pre_reset",
                "mask_in_foot_order": mask,
            }
    groups = []
    for sample in samples:
        key = _command_key(sample)
        if not groups or groups[-1][0] != key:
            groups.append((key, []))
        groups[-1][1].append(sample)
    for index, (key, group) in enumerate(groups):
        anchor = group[0]["pre"]["state"]["root_position_w_m"]
        kwargs = {
            "phase_anchor": anchor,
            "foot_names": foot_names,
            "contact_threshold": contact_threshold,
        }
        complete = index < len(groups) - 1
        phase = {
            "phase": key[0],
            "pivot_yaw_request_rad_s": key[1],
            "command_onset_observed": index > 0,
            "complete_command_window": complete,
            "end_reason": "observed_command_transition"
            if complete
            else report["stop_reason"],
            "summary": _summarize(group, **kwargs),
        }
        if key[0] == "pivot":
            onset = group[0]["time_before_s"]
            acquisition = [
                sample
                for sample in group
                if sample["time_before_s"] < onset + 1.0 - 1e-9
            ]
            sustained = group[len(acquisition) :]
            duration = group[-1]["time_after_s"] - onset
            full_blocks = math.floor((duration + 1e-9) / 0.4)
            start, end = onset + (full_blocks - 1) * 0.4, onset + full_blocks * 0.4
            late = [
                sample
                for sample in group
                if sample["time_before_s"] >= start - 1e-9
                and sample["time_after_s"] <= end + 1e-9
            ]
            if (
                not late
                or full_blocks < 1
                or not math.isclose(late[0]["time_before_s"], start, abs_tol=1e-8)
                or not math.isclose(late[-1]["time_after_s"], end, abs_tol=1e-8)
            ):
                late = []
            phase["pivot_segments"] = {
                "acquisition": _summarize(acquisition, **kwargs),
                "sustained": _summarize(sustained, **kwargs),
                "late_full_0_4s_block": _summarize(late, **kwargs),
                "late_discarded_tail_s": max(0.0, duration - full_blocks * 0.4)
                if late
                else None,
                "late_block_is_sustained": bool(
                    late and late[0]["time_before_s"] >= onset + 1.0 - 1e-9
                ),
                "observed_window_complete": complete,
            }
        output["phases"].append(phase)
    pivots = [phase for phase in output["phases"] if phase["phase"] == "pivot"]
    output["pivot_coverage"] = {
        "observed_windows": len(pivots),
        "complete_windows": sum(phase["complete_command_window"] for phase in pivots),
        "status": "NOT_OBSERVED"
        if not pivots
        else (
            "OBSERVED_COMPLETE"
            if all(phase["complete_command_window"] for phase in pivots)
            else "INCOMPLETE"
        ),
        "absence_reason": report["stop_reason"] if not pivots else None,
    }
    # Finite inputs with absurd magnitudes can still overflow derived sums.
    json.dumps(output, allow_nan=False)
    return output


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def print_summary(result):
    print(f"Trace: {result['source_path']}  sha256={result['source_sha256']}")
    print(
        f"Observed {result['sample_count']} steps / {result['captured_duration_s']:.3f} s; stop={result['stop_reason']}; termination={result['termination']}"
    )
    print(
        f"Reporting thresholds: {json.dumps(result['reporting_thresholds'], sort_keys=True)}"
    )
    print(f"Pivot coverage: {json.dumps(result['pivot_coverage'], sort_keys=True)}")
    for phase in result["phases"]:
        segments = [
            (phase["phase"], phase["summary"]),
            *phase.get("pivot_segments", {}).items(),
        ]
        for name, summary in segments:
            if not isinstance(summary, dict):
                continue
            action = summary["action_and_motion"]
            velocity = summary["velocity_estimation"]
            print(
                f"{name} {summary['start_s']:.3f}–{summary['end_s']:.3f} s: "
                f"yaw={summary['yaw']['mean_world_z_rate_rad_s']:.3f}, "
                f"yawMAE={summary['yaw']['mean_abs_rate_error_rad_s']:.3f} rad/s; "
                f"excursion={summary['max_xy_excursion_from_phase_onset_m']:.3f} m; "
                f"velocity-estimate componentMAE={velocity['mean_component_mae_m_s']:.3f} m/s; "
                f"configured/safe clip fractions={action['configured_target_clipping']['joint_sample_fraction']:.3f}/{action['safe_target_clipping']['joint_sample_fraction']:.3f}; "
                f"all-feet-contact={summary['all_feet_contact_fraction']:.3f}; "
                f"window-complete={phase['complete_command_window']}"
            )
        motion = phase["summary"]["action_and_motion"]
        print(
            f"  Mean |action step delta|={motion['policy_action_adjacent_delta_abs']['mean']}; "
            f"|target delta|={motion['processed_target_pre_post_delta_abs_rad']['mean']:.6f} rad; "
            f"|joint motion|={motion['joint_pre_post_motion_abs_rad']['mean']:.6f} rad; "
            f"|computed-applied torque|={phase['summary']['torque']['computed_minus_applied_abs_nm']['mean']:.6f} Nm"
        )
    print(
        f"First observed events (null = not observed or threshold disabled): {json.dumps(result['first_events'], sort_keys=True)}"
    )
    for notice in result["limitations"]:
        print(f"LIMIT: {notice}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "trace",
        type=Path,
        help="One exact startup_diagnostics.json file; no directory/glob selection",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON rather than concise text; writes no files",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=1.0,
        help="Reporting foot-force norm threshold in N (default 1), strictly greater",
    )
    parser.add_argument(
        "--estimate-error-threshold",
        type=float,
        help="Optional pre-action estimated-vs-true velocity error-norm event threshold, m/s",
    )
    parser.add_argument(
        "--torque-difference-threshold",
        type=float,
        help="Optional post-action per-joint computed-applied torque difference event threshold, Nm",
    )
    args = parser.parse_args(argv)
    try:
        path = args.trace.expanduser().resolve()
        raw = path.read_bytes()
        report = json.loads(raw, object_pairs_hook=_pairs)
        result = analyze_failure_trace(
            report,
            contact_threshold=args.contact_threshold,
            estimate_error_threshold=args.estimate_error_threshold,
            torque_difference_threshold=args.torque_difference_threshold,
        )
        result.update(
            source_path=str(path), source_sha256=hashlib.sha256(raw).hexdigest()
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        else:
            print_summary(result)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
    ) as error:
        parser.error(str(error))
    return 2 if result["stop_reason"] in {"error", "application_stopped"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
