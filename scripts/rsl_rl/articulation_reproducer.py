"""Run three isolated Go2 free-space cases, without training or loading a policy.

The source JSON supplies recorded state and the first actual backend effort.
Each fresh process records exactly one 5-ms step. This is a dynamics diagnostic,
not a locomotion benchmark or a production configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from .articulation_reproducer_core import (
        compare_cases,
        load_reproducer_source,
        read_json,
        validate_case,
    )
except ImportError:
    from articulation_reproducer_core import (
        compare_cases,
        load_reproducer_source,
        read_json,
        validate_case,
    )


CASES = ("origin", "translated", "teleported")
KIT_ARGS = (
    "--/physics/collisionApproximateCylinders=true --/crashreporter/preserveDump=true"
)
PROCESS_TIMEOUT_S = 180


def implementation_hashes():
    directory = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in (
            "articulation_reproducer.py",
            "articulation_reproducer_core.py",
            "articulation_reproducer_runtime.py",
        )
    }


def _write(path, value):
    # Exclusive creation protects earlier evidence, including failed attempts.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_worker(source, case, directory):
    """Launch Kit only in the child, with no task-specific argv left for Kit."""
    directory.mkdir(exist_ok=False)
    from isaaclab.app import AppLauncher

    saved_argv = sys.argv
    sys.argv = [sys.argv[0], "--info"]
    try:
        launcher = AppLauncher(
            headless=True,
            livestream=0,
            device="cuda:0",
            enable_cameras=False,
            kit_args=KIT_ARGS,
        )
    finally:
        sys.argv = saved_argv
    try:
        # All physics/asset imports happen after AppLauncher, as required by Kit.
        try:
            from .articulation_reproducer_runtime import run_case
        except ImportError:
            from articulation_reproducer_runtime import run_case

        report = run_case(source, case, directory)
        report["implementation_sha256"] = implementation_hashes()
        report["launch"] = {
            "headless": True,
            "livestream": 0,
            "device": "cuda:0",
            "kit_args": KIT_ARGS,
        }
        _write(directory / "trace.json", report)
    finally:
        launcher.app.close()


def run_triplet(source_path, output_parent, *, run_process=None):
    """Validate before GPU startup; stop on invalid evidence, not on divergence."""
    source_path = Path(source_path).resolve(strict=True)
    source = load_reproducer_source(source_path)
    output_parent = Path(output_parent).resolve(strict=True)
    if not output_parent.is_dir():
        raise ValueError("Output parent must be an existing directory")
    hashes = implementation_hashes()
    directory = Path(
        tempfile.mkdtemp(prefix="articulation_reproducer_", dir=output_parent)
    )
    print(f"Free-space Go2 diagnostic: {directory}", flush=True)
    run_process = subprocess.run if run_process is None else run_process
    reports = {}
    result = {}
    try:
        for case in CASES:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(source_path),
                "--_case",
                case,
                "--_output",
                str(directory / case),
            ]
            print(f"Running {case}: one measured 5-ms physics step", flush=True)
            with (directory / f"{case}.console.log").open("x") as console:
                completed = run_process(
                    command,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    timeout=PROCESS_TIMEOUT_S,
                    check=False,
                )
            if completed.returncode != 0:
                raise ValueError(
                    f"{case}: simulator process exited {completed.returncode}; "
                    f"see {case}.console.log. Remaining cases not launched."
                )
            report = read_json(directory / case / "trace.json")
            if report.get("implementation_sha256") != hashes:
                raise ValueError("Reproducer implementation changed between processes")
            validate_case(report, source)
            if report.get("case") != case:
                raise ValueError("Child returned the wrong case")
            reports[case] = report
        result = compare_cases(reports, source)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.TimeoutExpired,
    ) as error:
        result = {
            "kind": "articulation_reproducer_comparison",
            "evidence_status": "INVALID_DIAGNOSTIC",
            "error": str(error),
        }
    result["implementation_sha256"] = hashes
    result["source_path"] = str(source_path)
    result["completed_cases"] = list(reports)
    _write(directory / "comparison.json", result)
    print(f"{result['evidence_status']}: {directory / 'comparison.json'}", flush=True)
    return 2 if result["evidence_status"] == "INVALID_DIAGNOSTIC" else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_trace", type=Path, help="Existing uncentered L6 action-replay JSON"
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        help="Existing run directory for a fresh output folder",
    )
    parser.add_argument("--_case", choices=CASES, help=argparse.SUPPRESS)
    parser.add_argument("--_output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (args._case is None) != (args._output is None):
        parser.error("Internal worker case and output must be supplied together")
    if args._case is not None and args.output_parent is not None:
        parser.error("Worker output cannot be combined with --output-parent")
    try:
        if args._case is not None:
            source = load_reproducer_source(args.source_trace)
            run_worker(source, args._case, args._output)
            return 0
        return run_triplet(
            args.source_trace,
            args.output_parent or args.source_trace.resolve().parent,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Invalid reproducer setup: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
