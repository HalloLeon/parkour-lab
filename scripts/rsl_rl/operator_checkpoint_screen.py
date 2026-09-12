"""Screen a small, predeclared set of saved operator checkpoints; never train.

Reuses one checkpoint-bound archived endpoint screen. New and archived traces
are replayed with unchanged physical gates. A development pass is only eligible
for held-out confirmation, not promotion to RMA, obstacles or hardware.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

try:
    from .operator_benchmark import write_json
    from .operator_benchmark_core import (
        STEPS,
        Thresholds,
        command_schedule,
        file_sha256,
        load_reference_checkpoint,
        read_yaml_data,
        score_trace,
    )
    from .operator_train import run_final_check
except ImportError:
    from operator_benchmark import write_json
    from operator_benchmark_core import (
        STEPS,
        Thresholds,
        command_schedule,
        file_sha256,
        load_reference_checkpoint,
        read_yaml_data,
        score_trace,
    )
    from operator_train import run_final_check


def replay_report(report_path: Path, checkpoint: Path) -> dict:
    """Recompute evidence, rejecting changed gates, identities or trial outcomes."""
    saved = json.loads(report_path.read_text())
    if (
        saved.get("status") not in ("PASS", "FAIL")
        or saved.get("seed") != 43
        or saved.get("simulation_steps") != STEPS
        or saved.get("thresholds") != asdict(Thresholds())
    ):
        raise ValueError("Expected the unchanged complete seed-43 development screen")
    params = checkpoint.parent / "params"
    expected_hashes = {
        "checkpoint": file_sha256(checkpoint),
        "env.yaml": file_sha256(params / "env.yaml"),
        "agent.yaml": file_sha256(params / "agent.yaml"),
    }
    if any(
        saved.get("provenance", {}).get("sha256", {}).get(key) != value
        for key, value in expected_hashes.items()
    ):
        raise ValueError(
            "Archived screen checkpoint/environment/agent identity mismatch"
        )
    data = load_reference_checkpoint(checkpoint, read_yaml_data(params / "agent.yaml"))
    if saved.get("checkpoint_iteration") != data["iter"]:
        raise ValueError("Archived checkpoint iteration mismatch")
    labels, _ = command_schedule(10)
    trace_path = report_path.with_name("trace.npz")
    with np.load(trace_path, allow_pickle=False) as archive:
        result = score_trace(dict(archive), labels)
    if (
        saved.get("status") != result["status"]
        or saved.get("profiles") != result["profiles"]
    ):
        raise ValueError("Archived pass/fail counts disagree with raw physical trace")
    fields = ("env_id", "profile", "passed", "failures", "observed_duration_s")
    before = [{key: row[key] for key in fields} for row in saved["trials"]]
    after = [{key: row[key] for key in fields} for row in result["trials"]]
    if before != after:
        raise ValueError("Archived trial outcomes disagree with raw physical trace")
    return {
        "checkpoint": str(checkpoint),
        "iteration": data["iter"],
        "report": str(report_path),
        "sha256": {
            **expected_hashes,
            "report": file_sha256(report_path),
            "trace": file_sha256(trace_path),
        },
        "status": result["status"],
        "passed": sum(row["passed"] for row in result["trials"]),
        "total": len(result["trials"]),
        "profiles": result["profiles"],
        "phase_summary": result["phase_summary"],
        "packages": saved.get("provenance", {}).get("packages", {}),
        "student_interface_audit": saved.get("student_interface_audit"),
    }


def decision(candidates: list[dict]) -> dict:
    eligible = [row["checkpoint"] for row in candidates if row["status"] == "PASS"]
    return {
        "schema_version": 1,
        "status": "DEVELOPMENT_PASS" if eligible else "NO_DEVELOPMENT_PASS",
        "eligible_for_heldout_confirmation": eligible,
        "promoted": False,
        "scope": (
            "Only these predeclared checkpoints were tested; this is not a convergence "
            "test, exhaustive search, RMA acceptance or hardware certification. "
            "Phase-local passes cannot override any whole-trajectory gate. "
            "Selection uses seed 43; confirmation seeds 44 and 45 remain held out."
        ),
        "candidates": candidates,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--iterations",
        nargs="+",
        type=int,
        required=True,
        help="One to three preselected saved checkpoint iterations",
    )
    parser.add_argument(
        "--reference-report",
        type=Path,
        required=True,
        help="Archived endpoint report.json from this same run, with trace.npz",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-parent", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= len(args.iterations) <= 3 or len(set(args.iterations)) != len(
        args.iterations
    ):
        parser.error(
            "Predeclare one to three distinct checkpoints; do not start a broad sweep"
        )
    if any(i < 0 for i in args.iterations):
        parser.error("Checkpoint iterations must be nonnegative")
    output = None
    try:
        run = args.run.resolve(strict=True)
        reference_path = args.reference_report.resolve(strict=True)
        reference_iteration = json.loads(reference_path.read_text())[
            "checkpoint_iteration"
        ]
        if type(reference_iteration) is not int or reference_iteration < 0:
            raise ValueError("Invalid reference iteration")
        if reference_iteration in args.iterations:
            raise ValueError("Do not rerun the already measured reference endpoint")
        # Read-only preflight verifies every requested checkpoint before launching
        # any simulator or creating a new output directory.
        checkpoints = [run / f"model_{iteration}.pt" for iteration in args.iterations]
        agent = read_yaml_data(run / "params/agent.yaml")
        for iteration, checkpoint in zip(args.iterations, checkpoints, strict=True):
            if load_reference_checkpoint(checkpoint, agent)["iter"] != iteration:
                raise ValueError(f"Filename/iteration mismatch: {checkpoint}")
        reference = replay_report(
            reference_path, run / f"model_{reference_iteration}.pt"
        )
        parent = args.output_parent or run
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(
            tempfile.mkdtemp(prefix="operator_selection_", dir=parent)
        ).resolve()
        write_json(
            output / "protocol.json",
            {
                "iterations": args.iterations,
                "seed": 43,
                "repetitions": 10,
                "thresholds": asdict(Thresholds()),
                "reference": reference,
                "implementation_sha256": {
                    name: file_sha256(Path(__file__).with_name(name))
                    for name in (
                        "operator_checkpoint_screen.py",
                        "operator_benchmark_core.py",
                        "operator_benchmark.py",
                        "operator_train.py",
                        "operator_student_bridge.py",
                    )
                },
            },
        )
        print(f"Saved-checkpoint screen (no training): {output}", flush=True)
        candidates = [reference]
        for checkpoint in checkpoints:
            check, code = run_final_check(
                checkpoint,
                output / checkpoint.stem,
                args.device,
                audit_student_interface=True,
            )
            if code not in (0, 1):
                raise RuntimeError(f"Benchmark error: {check}")
            # Physical FAIL is evidence, not a process error: continue the fixed
            # list. ERROR or inconsistent artifacts stop immediately.
            measured = replay_report(Path(check["report"]), checkpoint)
            audit = measured["student_interface_audit"]
            if not isinstance(audit, dict) or any(
                audit.get(k) != v
                for k, v in {
                    "status": "ORACLE_PARITY_PASS",
                    "control_steps": STEPS,
                    "action_comparisons": STEPS * 100,
                    "exact_action_equality": True,
                    "student_status": "UNTRAINED_NOT_RUN",
                }.items()
            ):
                raise ValueError(
                    "Missing or incomplete runtime oracle/history parity audit"
                )
            if measured["packages"] != reference["packages"]:
                raise ValueError(
                    "Runtime package versions differ from the archived reference; not a matched screen"
                )
            candidates.append(measured)
            write_json(output / "progress.json", {"candidates": candidates})
        result = decision(candidates)
        write_json(output / "report.json", result)
        for row in candidates:
            print(
                f"  model_{row['iteration']}: {row['passed']}/{row['total']} complete trajectories"
            )
        print(f"{result['status']}: {output / 'report.json'}", flush=True)
        return 0 if result["eligible_for_heldout_confirmation"] else 1
    except Exception as error:
        # This is the process/evidence boundary, not the behavioral scorer.
        # Restricted torch/NPZ readers can raise UnpicklingError, BadZipFile,
        # EOFError, etc.; malformed nested metadata can raise AttributeError.
        # None may escape as exit 1 (reserved for a valid physical FAIL).
        result = {
            "status": "ERROR",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if output is not None:
            try:
                write_json(output / "report.json", result)
            except OSError as reporting_error:
                print(
                    f"ERROR writing failure report: {reporting_error}",
                    file=sys.stderr,
                    flush=True,
                )
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
