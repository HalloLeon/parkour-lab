# SPDX-License-Identifier: BSD-3-Clause
"""Small, fail-closed behavioral screen; usable without Isaac Sim or PyTorch.

Passing three episodes is a regression screen, not a reliability certificate.
Use physical outcomes, never reward totals, to compare reward ablations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def screen_failures(report: dict, windows: list[dict] | None = None) -> list[str]:
    """Return failed engineering targets, including missing/incomplete evidence."""
    failures: list[str] = []

    def bounded(data: dict, key: str, low: float, high: float, label: str = "") -> bool:
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
    if profile == "stop_restart":
        bounded(summary, "stop_window_count", requested, math.inf)
        bounded(summary, "stop_settled_within_1s_fraction", 1, 1)
        bounded(summary, "stop_drift_2s_sample_count", requested, math.inf)
        bounded(summary, "maximum_stop_drift_2s_m", 0, 0.1)
        bounded(summary, "restart_episode_count", requested, requested)
        bounded(summary, "restart_success_fraction", 1, 1)
        return failures
    if profile != "pivot_restart":
        failures.append(
            "screen requires translation_only, stop_restart or pivot_restart"
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
        completed_pivot_episodes.add(window.get("episode"))
    if completed_pivot_episodes != set(range(requested)):
        failures.append(
            "need a complete pivot/restart window in every evaluated episode (--telemetry)"
        )
    return failures


def check_metrics_file(path: Path) -> bool:
    """Print a verdict for exactly one report, never silently select an old run."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        windows = None
        if report.get("command_profile") == "pivot_restart":
            windows = json.loads(
                path.with_name("command_windows.json").read_text(encoding="utf-8")
            )["windows"]
        failures = screen_failures(report, windows)
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
    args = parser.parse_args()
    passed = [check_metrics_file(path) for path in args.metrics]
    return 0 if all(passed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
