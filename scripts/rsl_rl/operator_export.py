"""CPU actor extraction; no simulator launch, training, or hardware commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motor-report",
        type=Path,
        help="Report containing the exact source motor binding (default: checkpoint sibling report.json)",
    )
    parser.add_argument(
        "checkpoint", type=Path, help="Native recurrent training checkpoint"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New actor-only .pt file; never overwrite",
    )
    args = parser.parse_args(argv)
    from parkour_lab.learning.recurrent_operator import export_recurrent_actor

    try:
        report = export_recurrent_actor(
            args.checkpoint, args.output, motor_report=args.motor_report
        )
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Actor export failed: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
