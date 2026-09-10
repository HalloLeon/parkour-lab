# SPDX-License-Identifier: BSD-3-Clause
"""Small, fail-closed behavioral screen; usable without Isaac Sim or PyTorch.

Passing three episodes is a regression screen, not a reliability certificate.
Use physical outcomes, never reward totals, to compare reward ablations.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class OperatorThresholds:
    """Engineering regression targets, not certified robot stability limits.

    Attitude applies only to complete, supported-flat scripted control trials.
    Stop onset excursion is measured from the command's pre-action position;
    it is deliberately separate from the existing settled two-second drift.
    """

    max_flat_attitude_rad: float = math.radians(15.0)
    max_stop_onset_excursion_m: float = 0.15
    max_late_pivot_yaw_error_rad_s: float = 0.1

    def __post_init__(self):
        for name, value in vars(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_flat_attitude_rad >= math.pi / 2:
            raise ValueError("max_flat_attitude_rad must be less than pi/2")


def add_operator_threshold_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-flat-attitude-deg", type=float, default=15.0)
    parser.add_argument("--max-stop-onset-excursion-m", type=float, default=0.15)
    parser.add_argument("--max-late-pivot-yaw-error-rad-s", type=float, default=0.1)


def operator_thresholds_from_args(args) -> OperatorThresholds:
    return OperatorThresholds(
        max_flat_attitude_rad=math.radians(args.max_flat_attitude_deg),
        max_stop_onset_excursion_m=args.max_stop_onset_excursion_m,
        max_late_pivot_yaw_error_rad_s=args.max_late_pivot_yaw_error_rad_s,
    )


def load_command_windows(path: Path, report: dict) -> list[dict] | None:
    """Load colocated, complete command evidence; old/missing traces fail closed."""
    if report.get("command_profile") not in ("stop_restart", "pivot_restart"):
        return None
    trace = json.loads(
        path.with_name("command_windows.json").read_text(encoding="utf-8")
    )
    windows, metadata = trace["windows"], trace["metadata"]
    if not isinstance(windows, list) or not all(
        isinstance(window, dict) for window in windows
    ):
        raise ValueError("command windows must be a list of objects")
    for key, expected in (
        ("completed_episode_count", report["requested_episodes"]),
        ("window_count", len(windows)),
        ("complete_window_count", len(windows)),
        ("partial_window_count", 0),
    ):
        if type(metadata.get(key)) is not int or metadata[key] != expected:
            raise ValueError(f"command trace {key} must be {expected}")
    step_dt = metadata.get("step_dt_s")
    if (
        type(step_dt) not in (int, float)
        or not math.isfinite(step_dt)
        or not 0 < step_dt <= 0.1
    ):
        raise ValueError("command trace step_dt_s must be finite and in (0, .1]")
    count = 0
    for window in windows:
        samples, duration = window.get("sample_count"), window.get("duration_s")
        if type(samples) is not int or samples < 1:
            raise ValueError("command window sample_count must be a positive integer")
        if (
            type(duration) not in (int, float)
            or not math.isfinite(duration)
            or not math.isclose(duration, samples * step_dt, rel_tol=0, abs_tol=1e-7)
        ):
            raise ValueError(
                "command window duration does not match sample_count * step_dt_s"
            )
        count += samples
    if (
        type(metadata.get("sample_count")) is not int
        or metadata["sample_count"] != count
    ):
        raise ValueError("command trace sample_count does not match its windows")
    return windows


def screen_failures(
    report: dict,
    windows: list[dict] | None = None,
    thresholds: OperatorThresholds | None = None,
) -> list[str]:
    """Return failed engineering targets, including missing/incomplete evidence."""
    failures: list[str] = []
    thresholds = thresholds or OperatorThresholds()

    def bounded(data: dict, key: str, low: float, high: float, label: str = "") -> bool:
        if not isinstance(data, dict):
            failures.append(f"{label}{key}: expected a metric object")
            return False
        value = data.get(key)
        ok = (
            type(value) in (int, float)
            and math.isfinite(value)
            and low <= value <= high
        )
        if not ok:
            failures.append(f"{label}{key}={value!r}; expected [{low:g}, {high:g}]")
        return ok

    requested = report.get("requested_episodes")
    if type(requested) is not int or requested < 3:
        return ["requested_episodes must be an integer >= 3"]
    bounded(report, "completed_episodes", requested, requested)
    bounded(report, "num_envs", 1, 1)
    for key, expected in (("policy_mode", "history_mean"), ("reset_profile", "jitter")):
        if report.get(key) != expected:
            failures.append(f"{key} must be {expected!r} for the deployment screen")
    summary = report.get("summary") or {}
    bounded(summary, "success_rate", 1, 1)
    for key in ("chassis_contact_rate", "fell_below_course_rate", "off_route_rate"):
        bounded(summary, key, 0, 0)
    bounded(summary, "mean_moving_speed_absolute_error_m_s", 0, 0.2)

    profile = report.get("command_profile")
    if profile == "translation_only":
        return failures
    if profile not in ("stop_restart", "pivot_restart"):
        failures.append(
            "screen requires translation_only, stop_restart or pivot_restart"
        )
        return failures

    # All default family level-zero tiles are flat. Do not extend an upright
    # attitude criterion to steps or banked ramps, or silently accept unknown
    # scene metadata. Commands on unsupported terrain need a different gate.
    if report.get("task") != "Parkour-Lab-v0":
        failures.append("supported-flat control screen requires task='Parkour-Lab-v0'")
    if (
        type(report.get("difficulty_level")) is not int
        or report["difficulty_level"] != 0
    ):
        failures.append("supported-flat control screen requires difficulty_level=0")
    if report.get("terrain_family") not in {
        "gap",
        "high_step",
        "hurdle",
        "tilted_ramps",
    }:
        failures.append("supported-flat control screen requires a known terrain_family")
    variant = report.get("geometry_variant_index")
    if type(variant) is not int or not 0 <= variant < 10:
        failures.append(
            "supported-flat control screen requires a known geometry_variant_index (0..9)"
        )
    if not isinstance(windows, list) or not all(
        isinstance(window, dict) for window in windows
    ):
        failures.append(
            "complete command_windows.json required for control qualification (--telemetry)"
        )
        windows = []
    phase_name = "stop" if profile == "stop_restart" else "pivot"
    yaw_target = report.get("desired_yaw_rate_rad_s")
    if profile == "stop_restart":
        bounded(report, "desired_yaw_rate_rad_s", 0, 0)
    elif (
        type(yaw_target) not in (int, float)
        or not math.isfinite(yaw_target)
        or not 0 < abs(yaw_target) <= 0.8
    ):
        failures.append(
            "pivot screen requires a finite nonzero desired_yaw_rate_rad_s within +/-0.8"
        )
    expected_phases = ["translating", phase_name, "translating"]
    # Check all three phases, including restarted motion; an upright stop alone
    # must not hide a strongly pitched translation or an omitted late window.
    for episode in range(requested):
        episode_windows = [
            window for window in windows if window.get("episode") == episode
        ]
        if [window.get("phase") for window in episode_windows] != expected_phases:
            failures.append(
                f"episode {episode}: need complete translate-{phase_name}-restart evidence"
            )
    if len(windows) != requested * 3:
        failures.append("control trace must contain exactly three windows per episode")
    for index, window in enumerate(windows):
        label = f"control window {index}: "
        if type(window.get("window")) is not int or window["window"] != index:
            failures.append(label + "window index must be contiguous")
        if (
            type(window.get("episode")) is not int
            or not 0 <= window["episode"] < requested
        ):
            failures.append(label + "episode index is invalid")
        expected_end = "episode_success" if index % 3 == 2 else "command_change"
        if (
            window.get("complete") is not True
            or window.get("end_reason") != expected_end
        ):
            failures.append(
                label + "incomplete or unexpectedly terminated control phase"
            )
        for key in ("max_abs_roll_rad", "max_abs_pitch_rad"):
            bounded(window, key, 0, thresholds.max_flat_attitude_rad, label)

    if profile == "stop_restart":
        bounded(summary, "stop_window_count", requested, math.inf)
        bounded(summary, "stop_settled_within_1s_fraction", 1, 1)
        bounded(summary, "stop_drift_2s_sample_count", requested, math.inf)
        bounded(summary, "maximum_stop_drift_2s_m", 0, 0.1)
        bounded(summary, "restart_episode_count", requested, requested)
        bounded(summary, "restart_success_fraction", 1, 1)
        for window in windows:
            if window.get("phase") == "stop":
                label = f"stop window {window.get('window', '?')}, command-onset excursion: "
                bounded(window, "duration_s", 3.4, math.inf, label)
                bounded(
                    window,
                    "max_planar_excursion_m",
                    0,
                    thresholds.max_stop_onset_excursion_m,
                    label,
                )
        return failures

    bounded(summary, "pivot_episode_count", requested, requested)
    bounded(summary, "pivot_restart_episode_count", requested, requested)
    bounded(summary, "pivot_restart_success_fraction", 1, 1)
    bounded(summary, "maximum_pivot_xy_excursion_m", 0, 0.15)
    pivots = [window for window in (windows or []) if window.get("phase") == "pivot"]
    # A missing/unobserved phase is not a zero-error turn. Require a full pivot
    # followed by a command change in every episode, not failure-truncated data.
    completed_pivot_episodes = set()
    for window in pivots:
        label = f"pivot window {window.get('window', '?')}: "
        if (
            window.get("complete") is not True
            or window.get("end_reason") != "command_change"
        ):
            failures.append(label + "pivot did not finish before restart")
            continue
        bounded(window, "duration_s", 1.9, math.inf, label)
        # Window telemetry anchors before the first pivot action, including
        # its braking displacement. Older summary metrics started one step late.
        bounded(window, "max_planar_excursion_m", 0, 0.15, label)
        tracking = window.get("post_acquisition_world_yaw_rate_tracking") or {}
        bounded(tracking, "excluded_initial_s", 0.5, 0.5, label)
        bounded(tracking, "sample_count", 1, math.inf, label)
        bounded(tracking, "mean_abs_error_rad_s", 0, 0.1, label)
        bounded(tracking, "wrong_sign_sample_fraction", 0, 0.05, label)
        # Diagnostic acquisition ends after one second (not the legacy .5 s
        # exclusion above). This mean catches sustained under-turning but cannot
        # alone protect against a collapse confined to the end of a long pivot.
        diagnostics = window.get("phase_diagnostics")
        sustained = (
            diagnostics.get("pivot_sustained")
            if isinstance(diagnostics, dict)
            else None
        )
        sustained = sustained if isinstance(sustained, dict) else {}
        bounded(sustained, "sample_count", 1, math.inf, label + "sustained: ")
        bounded(sustained, "duration_s", 0.8, math.inf, label + "sustained: ")
        duration, samples = window.get("duration_s"), window.get("sample_count")
        if type(sustained.get("sample_count")) is not int:
            failures.append(label + "sustained sample_count must be an integer")
        if (
            type(duration) in (int, float)
            and math.isfinite(duration)
            and duration > 0
            and type(samples) is int
            and samples > 0
        ):
            step_dt = duration / samples
            expected_samples = max(0, samples - max(1, round(1.0 / step_dt)))
            bounded(
                sustained,
                "sample_count",
                expected_samples,
                expected_samples,
                label + "sustained: ",
            )
            expected_duration = expected_samples * step_dt
            bounded(
                sustained,
                "duration_s",
                expected_duration - 1e-7,
                expected_duration + 1e-7,
                label + "sustained: ",
            )
        signals = sustained.get("signal_means") or {}
        bounded(
            signals,
            "abs_yaw_error_rad_s",
            0,
            thresholds.max_late_pivot_yaw_error_rad_s,
            label + "sustained: ",
        )
        if type(yaw_target) in (int, float) and math.isfinite(yaw_target):
            bounded(
                signals,
                "command_yaw_rate_rad_s",
                yaw_target - 1e-7,
                yaw_target + 1e-7,
                label + "sustained: ",
            )
        # Existing telemetry has nonoverlapping .4 s blocks. Inspect its last
        # full block, not its whole-window average. A <.4 s tail is explicitly
        # outside this metric; the instantaneous post-.5 s gate remains active.
        averages = window.get("time_averaged_world_yaw_rate_tracking")
        averages = averages if isinstance(averages, dict) else {}
        blocks = averages.get("windows")
        bounded(averages, "window_s", 0.4, 0.4, label + "late block: ")
        if (
            not isinstance(blocks, list)
            or not blocks
            or not all(isinstance(block, dict) for block in blocks)
        ):
            failures.append(label + "missing late .4 s yaw block")
        else:
            bounded(averages, "full_window_count", len(blocks), len(blocks), label)
            duration = window.get("duration_s")
            if type(duration) in (int, float) and math.isfinite(duration):
                expected_count = math.floor((duration + 1e-9) / 0.4)
                if len(blocks) != expected_count:
                    failures.append(
                        label + "yaw blocks do not cover all complete .4 s intervals"
                    )
                tail = max(0.0, duration - len(blocks) * 0.4)
                bounded(
                    averages,
                    "discarded_tail_s",
                    max(0, tail - 1e-7),
                    tail + 1e-7,
                    label + "late block: ",
                )
            for index, block in enumerate(blocks):
                for key, expected in (
                    ("start_s", index * 0.4),
                    ("end_s", (index + 1) * 0.4),
                ):
                    bounded(
                        block,
                        key,
                        expected - 1e-7,
                        expected + 1e-7,
                        label + "yaw block: ",
                    )
                actual = block.get("mean_world_angular_velocity_z_rad_s")
                commanded = block.get("mean_command_yaw_rate_rad_s")
                bounded(
                    block,
                    "mean_world_angular_velocity_z_rad_s",
                    -math.inf,
                    math.inf,
                    label + "yaw block: ",
                )
                bounded(block, "abs_error_rad_s", 0, math.inf, label + "yaw block: ")
                if type(yaw_target) in (int, float) and math.isfinite(yaw_target):
                    bounded(
                        block,
                        "mean_command_yaw_rate_rad_s",
                        yaw_target - 1e-7,
                        yaw_target + 1e-7,
                        label + "yaw block: ",
                    )
                if all(
                    type(value) in (int, float) and math.isfinite(value)
                    for value in (actual, commanded)
                ):
                    expected_error = abs(actual - commanded)
                    bounded(
                        block,
                        "abs_error_rad_s",
                        max(0, expected_error - 1e-7),
                        expected_error + 1e-7,
                        label + "yaw block: ",
                    )
            last = blocks[-1]
            bounded(last, "start_s", 1.0, math.inf, label + "late block: ")
            bounded(
                last,
                "abs_error_rad_s",
                0,
                thresholds.max_late_pivot_yaw_error_rad_s,
                label + "late block: ",
            )
        if type(window.get("episode")) is int:
            completed_pivot_episodes.add(window["episode"])
    if completed_pivot_episodes != set(range(requested)):
        failures.append(
            "need a complete pivot/restart window in every evaluated episode (--telemetry)"
        )
    return failures


def check_metrics_file(
    path: Path, thresholds: OperatorThresholds | None = None
) -> bool:
    """Print a verdict for exactly one report, never silently select an old run."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        windows = load_command_windows(path, report)
        failures = screen_failures(report, windows, thresholds)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        failures = [f"invalid or missing evaluation artifacts: {exc}"]
    print(f"[SCREEN] {'FAIL' if failures else 'PASS'}: {path}", flush=True)
    for failure in failures:
        print(f"  - {failure}", flush=True)
    if not failures:
        print(
            "  Short regression screen only; repeat on held-out seeds before operator use."
        )
    return not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics", type=Path, nargs="+", help="Exact metrics.json file(s)."
    )
    add_operator_threshold_arguments(parser)
    args = parser.parse_args()
    try:
        thresholds = operator_thresholds_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"Engineering screen thresholds (not stability certification): {thresholds}")
    passed = [check_metrics_file(path, thresholds) for path in args.metrics]
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
