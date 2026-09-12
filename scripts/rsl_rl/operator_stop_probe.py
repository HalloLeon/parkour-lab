"""Two bounded, matched-prefix reference stop-recovery experiments; no training.

The frozen learner controls the approach. The original reference controls only
named post-motion stops, either at onset or after one second, then the learner
controls restart. These are MIXED-CONTROLLER diagnostics, never policy acceptance.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

try:
    from . import operator_benchmark as benchmark
    from .operator_benchmark_core import (
        DT,
        STEPS,
        command_schedule,
        file_sha256,
        load_reference_actor,
        profiles,
        read_yaml_data,
        score_trace,
    )
    from .operator_control_trace import (
        load_diagnostic_reference,
        summarize_control_trace,
        validate_control_artifacts,
    )
except ImportError:
    import operator_benchmark as benchmark
    from operator_benchmark_core import (
        DT,
        STEPS,
        command_schedule,
        file_sha256,
        load_reference_actor,
        profiles,
        read_yaml_data,
        score_trace,
    )
    from operator_control_trace import (
        load_diagnostic_reference,
        summarize_control_trace,
        validate_control_artifacts,
    )


VERSION = "operator_stop_recovery_probe_v1"
ARMS = {"onset": 0, "after_one_second": round(1 / DT)}
SCOPE = (
    "Mixed-controller diagnostic only: learner approach, original reference during "
    "named post-motion stops, learner restart. No training, checkpoint promotion, "
    "policy acceptance, action blending or simulation-state restoration."
)
FILES = (
    "report.json",
    "measurement_report.json",
    "provenance.json",
    "trace.npz",
    "control_trace.npz",
    "control_report.json",
    "control_interface.json",
    "resolved_env.yaml",
    "worker_status.json",
)
STABLE_SOURCES = {
    "scoring": "operator_benchmark_core.py",
    "reward_profiles": "operator_profiles.py",
    "operator_rewards": "operator_rewards.py",
    "operator_student_bridge": "operator_student_bridge.py",
    "operator_control_trace": "operator_control_trace.py",
}


def read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_arrays(path):
    with np.load(path, allow_pickle=False) as values:
        return dict(values)


def reports_agree(actual, expected):
    """Allow cross-NumPy descriptive rounding only; discrete outcomes stay exact.

    Archived Linux and local CPU NumPy scalar promotion differs. This tolerance
    never applies to raw prefix arrays, thresholds or pass/failure decisions.
    """
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(
                (actual[k] == v if k == "thresholds" else reports_agree(actual[k], v))
                for k, v in expected.items()
            )
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(reports_agree(a, b) for a, b in zip(actual, expected))
        )
    if isinstance(expected, (float, np.floating)):
        return isinstance(actual, (float, np.floating)) and bool(
            np.isfinite(actual)
            and np.isfinite(expected)
            and np.isclose(actual, expected, rtol=0, atol=1e-6)
        )
    return type(actual) is type(expected) and actual == expected


def stop_mask(repetitions, arm):
    """No initial standing substitutions, measured-state gating or trial targeting."""
    delay = ARMS[arm]
    labels, commands = command_schedule(repetitions)
    mask = np.zeros(commands.shape[:2], dtype=bool)
    for env_id, label in enumerate(labels):
        start = 0
        for phase in profiles()[label]:
            end = start + round(phase.duration_s / DT)
            if phase.name == "stop":
                if start == 0 or not commands[start - 1, env_id].any():
                    raise ValueError("A probe stop must follow a nonzero command")
                if commands[start:end, env_id].any() or start + delay >= end:
                    raise ValueError("Invalid post-motion stop interval")
                mask[start + delay : end, env_id] = True
            start = end
    return labels, mask


def load_baseline(checkpoint, reference, directory):
    """Replay and identity-bind the complete development capture before Kit starts."""
    checkpoint, reference, directory = (
        path.resolve(strict=True) for path in (checkpoint, reference, directory)
    )
    report = read_json(directory / "report.json")
    provenance = read_json(directory / "provenance.json")
    interface = read_json(directory / "control_interface.json")
    worker = read_json(directory / "worker_status.json")
    if (
        report.get("status") not in ("PASS", "FAIL")
        or report.get("seed") != 43
        or report.get("simulation_steps") != STEPS
        or report.get("provenance") != provenance
        or worker != {"returncode": 0, "timed_out": False}
        or report.get("worker") != worker
        or any(directory.glob("*_cleanup_error.json"))
    ):
        raise ValueError("Require a complete seed-43 development diagnostic capture")
    measured = read_json(directory / "measurement_report.json")
    if any(report.get(k) != v for k, v in measured.items()):
        raise ValueError("Worker and parent measurement disagree")
    packages = provenance.get("packages", {})
    if any(
        not isinstance(packages.get(name), str) or packages[name] == "unknown"
        for name in ("isaaclab", "isaacsim", "rsl-rl-lib", "torch")
    ):
        raise ValueError("Baseline runtime package identities are incomplete")
    agent = read_yaml_data(checkpoint.parent / "params/agent.yaml")
    saved = read_yaml_data(checkpoint.parent / "params/env.yaml")
    with torch.random.fork_rng(devices=[]):
        _, iteration = load_reference_actor(checkpoint, agent)
        _, reference_identity = load_diagnostic_reference(checkpoint, reference)
    expected = {
        "checkpoint": file_sha256(checkpoint),
        "agent.yaml": file_sha256(checkpoint.parent / "params/agent.yaml"),
        "env.yaml": file_sha256(checkpoint.parent / "params/env.yaml"),
    }
    if any(provenance["sha256"].get(k) != v for k, v in expected.items()):
        raise ValueError("Learner checkpoint/configuration differs from baseline")
    original_reference = provenance["diagnostic_reference"]
    if (
        reference_identity["sha256"] != original_reference["sha256"]
        or reference_identity["iteration"] != original_reference["iteration"]
        or iteration != report.get("checkpoint_iteration")
    ):
        raise ValueError("Reference/iteration differs from baseline")
    for name, filename in STABLE_SOURCES.items():
        if (
            file_sha256(Path(__file__).with_name(filename))
            != provenance["sha256"][name]
        ):
            raise ValueError(f"Capture/scoring implementation changed: {name}")
    validate_control_artifacts(
        directory, report, original_reference, expected["checkpoint"]
    )
    trace, control = (
        read_arrays(directory / "trace.npz"),
        read_arrays(directory / "control_trace.npz"),
    )
    labels, _ = command_schedule(10)
    replay = score_trace(trace, labels)
    if not reports_agree({k: report.get(k) for k in replay}, replay):
        raise ValueError("Physical trace does not reproduce baseline report")
    if trace["terminated"].any() or trace["time_out"].any():
        raise ValueError("Recovery baseline must contain uninterrupted trajectories")
    metadata = {
        k: v for k, v in interface.items() if k not in ("reference", "learner_sha256")
    }
    summary = summarize_control_trace(control, trace, labels, metadata)
    archived = read_json(directory / "control_report.json")
    if any(archived.get(k) != v for k, v in summary.items()):
        raise ValueError("Control trace does not reproduce baseline report")
    return {
        "directory": directory,
        "checkpoint": checkpoint,
        "reference": reference,
        "agent": agent,
        "saved": saved,
        "trace": trace,
        "control": control,
        "interface": interface,
        "report": replay,
        "reference_identity": reference_identity,
        "identity": {
            "files": {name: file_sha256(directory / name) for name in FILES},
            "learner": expected,
            "reference": reference_identity["sha256"],
            "packages": provenance["packages"],
        },
    }


def compare_probe(baseline, trace, control, selection, arm):
    """Recompute exact source selection, matched prefixes and descriptive outcomes."""
    repetitions = len(baseline["report"]["trials"]) // len(profiles())
    labels, planned = stop_mask(repetitions, arm)
    scored = score_trace(trace, labels)
    summarize_control_trace(control, trace, labels, baseline["interface"])
    if set(selection) != {"reference_selected", "learner_action"}:
        raise ValueError("Incomplete selection trace")
    selected, learner = selection["reference_selected"], selection["learner_action"]
    if (
        selected.shape != planned.shape
        or selected.dtype != np.bool_
        or learner.shape != control["action"].shape
        or not np.isfinite(learner).all()
    ):
        raise ValueError("Invalid selection trace layout")
    done = trace["terminated"] | trace["time_out"]
    alive = np.ones_like(done)
    alive[1:] = ~np.maximum.accumulate(done[:-1], axis=0)
    if not np.array_equal(selected, planned & alive):
        raise ValueError(
            "Reference selection differs from the predeclared stop schedule"
        )
    executed = np.where(selected[..., None], control["reference_action"], learner)
    if not np.array_equal(executed, control["action"]):
        raise ValueError("Delivered action differs from the selected controller")
    switches = []
    for env_id in range(len(labels)):
        indices = np.flatnonzero(planned[:, env_id])
        first = int(indices[0]) if len(indices) else STEPS
        for key, previous in baseline["trace"].items():
            actual = trace[key]
            left, right = (
                (previous[env_id], actual[env_id])
                if key.startswith("initial_")
                else (previous[:first, env_id], actual[:first, env_id])
            )
            if not np.array_equal(left, right):
                raise ValueError(f"Unmatched physical prefix: env {env_id}, {key}")
        for key, previous in baseline["control"].items():
            # Include the switch's PRE-action state, not its intervened post-state.
            end = (
                min(first + 1, STEPS)
                if key.endswith("_pre") or key == "reference_action"
                else first
            )
            if not np.array_equal(previous[:end, env_id], control[key][:end, env_id]):
                raise ValueError(f"Unmatched control prefix: env {env_id}, {key}")
        if not np.array_equal(
            baseline["control"]["action"][: min(first + 1, STEPS), env_id],
            learner[: min(first + 1, STEPS), env_id],
        ):
            raise ValueError(f"Unmatched learner action at handoff: env {env_id}")
        switches.append(
            {
                "env_id": env_id,
                "profile": labels[env_id],
                "first_switch_s": first * DT if len(indices) else None,
                "reference_selected_steps": int(selected[:, env_id].sum()),
                "baseline_passed": baseline["report"]["trials"][env_id]["passed"],
                "mixed_controller_passed": scored["trials"][env_id]["passed"],
            }
        )
    return {
        "schema_version": VERSION,
        "arm": arm,
        "delay_s": ARMS[arm] * DT,
        "prefix_validation": "EXACT_ALL_TRIALS",
        "policy_acceptance": False,
        "scope": SCOPE,
        "trials": switches,
        "behavioral_result": scored,
    }


class StopRecoveryProbe:
    scope = SCOPE

    def __init__(self, baseline, arm):
        self.baseline, self.arm = baseline, arm
        _, self.planned = stop_mask(
            len(baseline["report"]["trials"]) // len(profiles()), arm
        )
        self.alive = np.ones(self.planned.shape[1], dtype=bool)
        self.reference = None
        self.selections, self.learner_actions = [], []

    def select(self, step, observation, learner_action):
        if step != len(self.selections) or step >= STEPS:
            raise ValueError("Out-of-order recovery probe action")
        if (
            self.reference is None
            or self.reference.training
            or any(p.requires_grad for p in self.reference.parameters())
        ):
            raise ValueError("Require the frozen reference")
        selected = self.planned[step] & self.alive
        self.selections.append(selected.copy())
        self.learner_actions.append(
            learner_action.detach().to("cpu", copy=True).numpy()
        )
        if not selected.any():
            return learner_action
        with torch.inference_mode():
            action = self.reference(observation)
        if action.shape != learner_action.shape or not torch.isfinite(action).all():
            raise ValueError("Invalid reference recovery action")
        return torch.where(
            torch.as_tensor(selected, device=action.device)[:, None],
            action,
            learner_action,
        )

    def observe_done(self, done):
        self.alive &= ~done.detach().cpu().numpy()

    def publish(self, output, result, trace, control):
        selection = {
            "reference_selected": np.stack(self.selections),
            "learner_action": np.stack(self.learner_actions),
        }
        # Persist full evidence even if the prefix comparison invalidates the probe.
        np.savez_compressed(output / "selection_trace.npz", **selection)
        comparison = compare_probe(self.baseline, trace, control, selection, self.arm)
        return {
            "status": "PROBE_COMPLETE",
            "policy_acceptance": False,
            "scope": SCOPE,
            "comparison": comparison,
            "control_diagnostics": result["control_diagnostics"],
            "runtime": result["runtime"],
            "baseline_identity": self.baseline["identity"],
            "selection_sha256": file_sha256(output / "selection_trace.npz"),
        }


def validate_probe_output(output, report, baseline, arm):
    if (
        report.get("status") != "PROBE_COMPLETE"
        or report.get("policy_acceptance") is not False
        or report.get("baseline_identity") != baseline["identity"]
        or report.get("selection_sha256") != file_sha256(output / "selection_trace.npz")
    ):
        raise ValueError("Invalid recovery probe identity/report")
    validate_control_artifacts(
        output,
        report,
        baseline["reference_identity"],
        baseline["identity"]["learner"]["checkpoint"],
    )
    interface = read_json(output / "control_interface.json")
    for key in ("joint_names", "foot_names", "contact_body_names", "step_dt_s"):
        if interface.get(key) != baseline["interface"].get(key):
            raise ValueError(f"Probe interface changed: {key}")
    # Recorder callables have a different import prefix in the dedicated CLI;
    # every other resolved field, including startup events, must be identical.
    previous = read_yaml_data(baseline["directory"] / "resolved_env.yaml")
    actual = read_yaml_data(output / "resolved_env.yaml")
    previous.pop("recorders", None)
    actual.pop("recorders", None)
    if previous != actual:
        raise ValueError(
            "Resolved physical/runtime configuration differs from baseline"
        )
    replay = compare_probe(
        baseline,
        read_arrays(output / "trace.npz"),
        read_arrays(output / "control_trace.npz"),
        read_arrays(output / "selection_trace.npz"),
        arm,
    )
    if not reports_agree(report.get("comparison"), replay):
        raise ValueError("Probe report disagrees with raw evidence")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--baseline-capture", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=tuple(ARMS), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        if bool(args.arm) != bool(args.worker_output) or (
            args.worker_output and args.validate_only
        ):
            raise ValueError("Arm selection is reserved for a supervised probe worker")
        baseline = load_baseline(args.checkpoint, args.reference, args.baseline_capture)
        if not args.validate_only:
            for package, expected in baseline["identity"]["packages"].items():
                if (
                    expected != "unknown"
                    and importlib.metadata.version(package) != expected
                ):
                    raise ValueError(
                        f"Runtime package differs from baseline: {package}"
                    )
    except Exception as error:
        parser.error(f"Preflight failed: {type(error).__name__}: {error}")
    if args.validate_only:
        print(
            "Validated complete baseline and checkpoint identities. Two mixed-controller probes remain UNRUN."
        )
        return 0
    args.checkpoint = baseline["checkpoint"]
    args.diagnostic_reference = baseline["reference"]
    args.seed, args.repetitions, args.audit_student_interface = 43, 10, False
    if args.worker_output:
        try:
            benchmark.run_benchmark(
                args,
                args.worker_output,
                baseline["agent"],
                baseline["saved"],
                stop_probe=StopRecoveryProbe(baseline, args.arm),
            )
        except Exception as error:
            benchmark.write_json(
                args.worker_output / "measurement_report.json",
                {"status": "ERROR", "error": str(error)},
            )
        return 0  # Supervisor owns status, including hard Kit exit and cleanup.
    output = None
    try:
        parent = args.output_parent or args.checkpoint.parent
        parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="operator_stop_probe_", dir=parent))
        return run_probes(args, baseline, output)
    except Exception as error:
        # Launch/status/report publication can fail too. Exit 1 must never turn
        # an incomplete diagnostic into an apparent measured behavioral failure.
        failure = {
            "status": "ERROR",
            "policy_acceptance": False,
            "error": f"Probe coordination failed: {type(error).__name__}: {error}",
        }
        if output is not None:
            try:
                report_path = output / "report.json"
                if report_path.exists():
                    failure["partial_result"] = read_json(report_path)
                benchmark.write_json(report_path, failure)
            except Exception as publication_error:
                failure["publication_error"] = str(publication_error)
        print(failure["error"], file=sys.stderr)
        return 2


def run_probes(args, baseline, output):
    """Exactly two supervised arms; partial progress cannot be marked complete."""
    protocol = {
        "schema_version": VERSION,
        "scope": SCOPE,
        "policy_acceptance": False,
        "arms": {name: delay * DT for name, delay in ARMS.items()},
        "seed": 43,
        "repetitions": 10,
        "baseline_identity": baseline["identity"],
        "source_sha256": {
            name: file_sha256(Path(__file__).with_name(name))
            for name in (
                "operator_stop_probe.py",
                "operator_benchmark.py",
                *STABLE_SOURCES.values(),
            )
        },
    }
    benchmark.write_json(output / "protocol.json", protocol)
    result = {
        "status": "RUNNING",
        "policy_acceptance": False,
        "protocol": protocol,
        "arms": {},
    }
    benchmark.write_json(output / "report.json", result)
    for arm in ARMS:
        directory = output / arm
        directory.mkdir()
        print(f"Mixed-controller stop diagnostic ({arm}): {directory}", flush=True)
        report = benchmark.supervise(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                str(args.checkpoint),
                "--reference",
                str(baseline["reference"]),
                "--baseline-capture",
                str(baseline["directory"]),
                "--device",
                args.device,
                "--worker-output",
                str(directory.resolve()),
                "--arm",
                arm,
            ],
            directory,
            valid_statuses=("PROBE_COMPLETE", "ERROR"),
        )
        if report.get("status") != "ERROR":
            try:
                validate_probe_output(directory, report, baseline, arm)
            except Exception as error:
                report = {
                    "status": "ERROR",
                    "error": str(error),
                    "measurement_result": report,
                }
        benchmark.write_json(directory / "report.json", report)
        result["arms"][arm] = report
        if report["status"] == "ERROR":
            result["status"] = "ERROR"
        elif len(result["arms"]) == len(ARMS):
            result["status"] = "PROBE_COMPLETE"
        benchmark.write_json(output / "report.json", result)
        if result["status"] == "ERROR":
            break  # Invalid comparisons cannot support a recovery interpretation.
    print(f"{result['status']} (NOT policy acceptance): {output / 'report.json'}")
    return 2 if result["status"] == "ERROR" else 0


if __name__ == "__main__":
    raise SystemExit(main())
