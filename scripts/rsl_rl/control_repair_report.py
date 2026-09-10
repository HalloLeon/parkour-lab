"""Validate one fresh four-case control screen without Isaac Sim.

This is a short checkpoint selection gate, not operator qualification. In
particular, previously passed terrain cases do not certify a new checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evaluation_screen import (
    OperatorThresholds,
    add_operator_threshold_arguments,
    load_command_windows,
    operator_thresholds_from_args,
    screen_failures,
)


CASES = {
    ("high_step", 6, "translation_only", 0.0),
    ("tilted_ramps", 0, "stop_restart", 0.0),
    ("tilted_ramps", 0, "pivot_restart", -0.5),
    ("tilted_ramps", 0, "pivot_restart", 0.5),
}


def inspect_bundle(
    root: Path,
    checkpoint_sha256: str,
    seed: int,
    thresholds: OperatorThresholds | None = None,
) -> list[str]:
    """Reject partial, duplicate, mixed-checkpoint or physically failing bundles."""
    failures = []
    seen = set()
    paths = sorted(root.rglob("metrics.json"))
    if len(paths) != len(CASES):
        failures.append(f"expected exactly four reports, found {len(paths)}")
    for path in paths:
        try:
            case_failures = []
            report = json.loads(path.read_text(encoding="utf-8"))
            case = tuple(
                report.get(key)
                for key in (
                    "terrain_family",
                    "difficulty_level",
                    "command_profile",
                    "desired_yaw_rate_rad_s",
                )
            )
            if case not in CASES or case in seen:
                case_failures.append("unexpected or duplicate case")
            seen.add(case)
            for key, expected in (
                ("checkpoint_sha256", checkpoint_sha256),
                ("seed", seed),
                ("task", "Parkour-Lab-v0"),
                ("geometry_variant_index", 0),
                ("desired_speed_m_s", 0.55),
                ("requested_episodes", 3),
            ):
                value = report.get(key)
                if isinstance(value, bool) or value != expected:
                    case_failures.append(f"{key} must be {expected!r}")
            windows = load_command_windows(path, report)
            case_failures.extend(screen_failures(report, windows, thresholds))
            print(f"[CONTROL SCREEN] {'FAIL' if case_failures else 'PASS'} {case}")
            failures.extend(f"{case}: {failure}" for failure in case_failures)
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            failures.append(f"invalid report {path}: {error}")
    if seen != CASES:
        failures.append(f"missing cases: {CASES - seen}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    add_operator_threshold_arguments(parser)
    args = parser.parse_args()
    try:
        thresholds = operator_thresholds_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    print(f"Engineering screen thresholds (not stability certification): {thresholds}")
    try:
        with args.checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        failures = inspect_bundle(args.root, digest, args.seed, thresholds)
    except OSError as error:
        failures = [str(error)]
    for failure in failures:
        print(f"  - {failure}")
    print(
        "FAIL: retain these artifacts; do not extend training blindly."
        if failures
        else "PASS: short control screen only. Check remaining terrains, held-out seeds and operator ingress before promotion."
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
