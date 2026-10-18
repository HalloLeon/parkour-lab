"""Offline case inspection and scoring; policy execution belongs to the caller."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Score an existing evaluation capture")
    score.add_argument("capture", type=Path)
    score.add_argument("--output", type=Path)
    report = commands.add_parser(
        "report",
        help="Check planned campaign endpoints and independent-seed uncertainty",
    )
    report.add_argument("campaign", type=Path)
    report.add_argument("--output", type=Path)
    cases = commands.add_parser("cases", help="Inspect the fixed evaluation case bank")
    cases.add_argument("--partition", default="validation")
    cases.add_argument("--attempts", type=int, default=1)
    cases.add_argument("--suite", choices=("core", "robustness"), default="core")
    cases.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "score":
        from parkour_lab.evaluation import score_capture

        result = score_capture(args.capture)
    elif args.command == "report":
        from parkour_lab.evaluation import report_campaign

        result = report_campaign(args.campaign)
    else:
        from parkour_lab.evaluation.cases import case_bank

        bank = case_bank
        if args.suite == "robustness":
            from parkour_lab.evaluation.robustness import robustness_bank

            bank = robustness_bank
        result = [
            case.as_dict()
            for case in bank(partition=args.partition, attempts=args.attempts)
        ]
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        from parkour_lab.provenance import write_json

        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
