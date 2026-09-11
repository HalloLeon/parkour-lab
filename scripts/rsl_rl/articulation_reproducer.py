"""Run three isolated Go2 free-space cases, without training or loading a policy.

The source JSON supplies recorded state and the first actual backend effort.
Each fresh process records exactly one 5-ms step. This is a dynamics diagnostic,
not a locomotion benchmark or a production configuration.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import ExitStack
from pathlib import Path

try:
    from .articulation_reproducer_core import (
        compare_cases,
        load_reproducer_source,
        parse_json,
        read_json,
        validate_case,
    )
except ImportError:
    from articulation_reproducer_core import (
        compare_cases,
        load_reproducer_source,
        parse_json,
        read_json,
        validate_case,
    )


CASES = ("origin", "translated", "teleported")
KIT_ARGS = (
    "--/physics/collisionApproximateCylinders=true --/crashreporter/preserveDump=true"
)
PROCESS_TIMEOUT_S = 180
STACK_INTERVAL_S = 60


class _WorkerJournal:
    """Keep execution evidence independent of simulator cleanup and stdout buffering."""

    def __init__(self, directory):
        self.directory = directory
        self.started = time.monotonic()
        self.stage = "worker_start"
        # Retain references: a later distinct exception may reuse a freed id.
        self.recorded_errors = {}

    def __enter__(self):
        self.files = ExitStack()
        try:
            self.progress_file = self.files.enter_context(
                (self.directory / "progress.jsonl").open("x", encoding="utf-8")
            )
            self.error_file = self.files.enter_context(
                (self.directory / "errors.jsonl").open("x", encoding="utf-8")
            )
            stacks = self.files.enter_context(
                (self.directory / "python_stacks.log").open("x", encoding="utf-8")
            )
            # This watchdog also runs while the Python thread is in a native
            # call. It diagnoses hangs; it neither steps nor kills the simulator.
            faulthandler.dump_traceback_later(
                STACK_INTERVAL_S, repeat=True, file=stacks, exit=False
            )
            self.files.callback(faulthandler.cancel_dump_traceback_later)
            self.progress("worker_start")
            return self
        except BaseException:
            self.files.close()
            raise

    def __exit__(self, *exc):
        self.files.close()

    def _record(self, stream, **details):
        row = {
            "stage": self.stage,
            "elapsed_s": time.monotonic() - self.started,
            "pid": os.getpid(),
            **details,
        }
        stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()

    def progress(self, stage):
        self.stage = stage
        self._record(self.progress_file)
        print(f"[reproducer] {stage}", flush=True)

    def record_error(self, error):
        if id(error) in self.recorded_errors:
            return
        self.recorded_errors[id(error)] = error
        details = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        # Emit before any potentially blocking simulator cleanup.
        print(details, file=sys.stderr, flush=True)
        try:
            self._record(
                self.error_file,
                error_type=type(error).__name__,
                error=str(error),
                traceback=details,
            )
        except OSError as journal_error:
            print(
                f"Could not persist worker error: {journal_error}",
                file=sys.stderr,
                flush=True,
            )


def _failure_context(directory):
    """Retain usable progress even if a killed process left a partial last line."""
    result = {"trace_present": (directory / "trace.json").is_file()}
    for filename, key in (
        ("progress.jsonl", "last_progress"),
        ("errors.jsonl", "worker_errors"),
    ):
        path = directory / filename
        rows = []
        try:
            if path.stat().st_size > 262144:
                result[f"{key}_read_error"] = (
                    "Worker journal exceeds 256 KiB; inspect the original file"
                )
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rows.append(parse_json(line))
                except ValueError:
                    continue
        except (OSError, UnicodeError) as error:
            result[f"{key}_read_error"] = str(error)
            continue
        result[key] = (rows[-1] if rows else None) if key == "last_progress" else rows
    result["python_stacks_path"] = str(directory / "python_stacks.log")
    return result


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
    with _WorkerJournal(directory) as journal:
        launcher = None
        primary_error = None
        captured = False

        def record_error(error):
            try:
                journal.record_error(error)
            except BaseException as journal_error:
                # Broken logging must neither replace a physics failure nor
                # prevent the independent application-close attempt below.
                error.add_note(f"Could not persist worker error: {journal_error!r}")

        def capture(report):
            nonlocal captured
            report["implementation_sha256"] = implementation_hashes()
            report["launch"] = {
                "headless": True,
                "livestream": 0,
                "device": "cuda:0",
                "kit_args": KIT_ARGS,
            }
            journal.progress("trace_write_begin")
            _write(directory / "trace.json", report)
            captured = True
            journal.progress("trace_saved")
            # Preserve the raw trace even if its contract fails. Neither its
            # presence nor clean simulator cleanup implies valid evidence.
            validate_case(report, source)
            journal.progress("trace_validated")

        try:
            journal.progress("app_import_begin")
            from isaaclab.app import AppLauncher

            saved_argv = sys.argv
            sys.argv = [sys.argv[0], "--info"]
            try:
                journal.progress("app_launch_begin")
                launcher = AppLauncher(
                    headless=True,
                    livestream=0,
                    device="cuda:0",
                    enable_cameras=False,
                    kit_args=KIT_ARGS,
                )
            finally:
                sys.argv = saved_argv
            journal.progress("app_launch_complete")
            # All physics/asset imports happen after AppLauncher.
            try:
                from .articulation_reproducer_runtime import run_case
            except ImportError:
                from articulation_reproducer_runtime import run_case

            run_case(
                source,
                case,
                directory,
                progress=journal.progress,
                capture=capture,
                record_error=record_error,
            )
            if not captured:
                raise RuntimeError(
                    "Runtime returned without capturing a measured trace"
                )
            journal.progress("runtime_complete")
        except BaseException as error:
            primary_error = error
            record_error(error)
            raise
        finally:
            if launcher is not None:
                cleanup_errors = []

                def attempt_close_operation(operation):
                    try:
                        operation()
                        return True
                    except BaseException as error:
                        cleanup_errors.append(error)
                        record_error(error)
                        return False

                attempt_close_operation(lambda: journal.progress("app_close_begin"))
                if attempt_close_operation(launcher.app.close):
                    attempt_close_operation(
                        lambda: journal.progress("app_close_complete")
                    )
                if cleanup_errors:
                    error = primary_error or cleanup_errors[0]
                    for cleanup_error in cleanup_errors:
                        if cleanup_error is not error:
                            error.add_note(
                                f"Additional application cleanup failure: {cleanup_error!r}"
                            )
                    if primary_error is None:
                        raise error
        journal.progress("worker_complete")


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
                "-u",
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
            "failed_case": case,
            "process_timed_out": isinstance(error, subprocess.TimeoutExpired),
            "worker_context": _failure_context(directory / case),
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
