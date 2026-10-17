"""Bounded caller-loaded policy evaluation and independent offline scoring check."""

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

from parkour_lab.provenance import (
    dependency_identity,
    file_sha256,
    package_source_identity,
    write_json,
    write_run_provenance,
)
from parkour_lab.runtime.session import exit_native_process, finish_session


class LoadedPolicy:
    """Tiny untrained caller-format fixture with frozen normalization and memory."""

    def __init__(self, state, device):
        self.state = {name: value.to(device).clone() for name, value in state.items()}
        self.memory = None
        self.frame_calls = self.reset_rows = 0

    def state_sha256(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.state.items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def reset(self, mask):
        import torch

        if self.memory is None or len(self.memory) != len(mask):
            self.memory = torch.zeros(
                len(mask), device=mask.device, dtype=torch.float32
            )
        self.memory[mask] = 0
        self.reset_rows += int(mask.sum())

    def act(self, observation):
        import torch

        assert observation.shape == (len(observation), 49)
        assert observation.dtype == torch.float32
        assert torch.isfinite(observation).all() and not observation.requires_grad
        normalized = (observation - self.state["offset"]) / self.state["scale"]
        self.memory += 1
        self.frame_calls += 1
        return normalized @ self.state["weight"].T + self.state["bias"]


def _load_fixture(output, device):
    import torch

    state = {
        "weight": torch.arange(12 * 49, dtype=torch.float32).reshape(12, 49).sin()
        * 0.0001,
        "bias": torch.zeros(12, dtype=torch.float32),
        "offset": torch.zeros(49, dtype=torch.float32),
        "scale": torch.ones(49, dtype=torch.float32),
    }
    path = output / "caller_policy.pt"
    torch.save(state, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert set(loaded) == set(state)
    for name, value in state.items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)
    return LoadedPolicy(loaded, device), file_sha256(path)


def _cases(full_bank):
    from parkour_lab.evaluation import EvaluationCase, case_bank

    if full_bank:
        return case_bank("validation")
    selected = (
        ("Flat", None, "plane", "stand"),
        ("Uneven", 0, "noise", "forward-035"),
        ("Slopes", 3, "pyramid", "pivot-left"),
        ("Stairs", 7, "inverted_pyramid", "pivot-right"),
        ("Hills", 3, "wave", "arc-left"),
        ("Flat", None, "plane", "arc-right"),
    )
    return tuple(EvaluationCase("validation", *fields) for fields in selected)


def _dependencies(output):
    import isaaclab

    result = {
        name: dependency_identity(name, distribution)
        for name, distribution in (
            ("isaaclab", "isaaclab"),
            ("isaaclab_assets", "isaaclab_assets"),
            ("isaaclab_tasks", "isaaclab_tasks"),
            ("isaacsim", "isaacsim"),
            ("torch", "torch"),
            ("gymnasium", "gymnasium"),
            ("numpy", "numpy"),
        )
    }
    if "commit" in result["isaaclab"]:
        write_run_provenance(output / "sdk", isaaclab.__file__)
    return result


def _offline(capture, output):
    # Exercise the actual command in a fresh process, blocking native/learner imports.
    program = """
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ('isaaclab', 'isaaclab_assets', 'isaaclab_tasks', 'isaacsim',
                   'omni', 'pxr', 'warp', 'rsl_rl', 'tensordict', 'parkour_lab.methods')
        if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
            raise AssertionError('Offline scoring imported ' + fullname)
sys.meta_path.insert(0, Guard())
sys.argv = ['parkour_lab', 'score', sys.argv[1]]
runpy.run_module('parkour_lab', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(capture)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    (output / "offline_stdout.log").write_text(result.stdout)
    (output / "offline_stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Offline scoring failed with code {result.returncode}")
    return json.loads(result.stdout)


def _check_capture(capture, cases, result, policy):
    import numpy as np

    assert result["capture_status"] == "COMPLETE"
    assert result["capture_valid"]
    assert result["aggregate"]["complete"]
    assert len(result["cases"]) == len(cases)
    total, witness = 0, None
    for case, scored in zip(cases, result["cases"], strict=True):
        assert scored["capture_status"] == "CAPTURED"
        receipt = json.loads((capture / case.case_id / "capture.json").read_text())
        assert receipt["cleanup"]["environment"] == "complete"
        realized = receipt["realization"]
        assert realized["training_information_available"] is False
        definition = realized["task"]["terrain"]
        assert definition["fixed_level"] == case.physical_level
        count = receipt["observed_steps"]
        assert receipt["sensor_noise"]["generated_frames"] == count
        assert receipt["interactions"]["session"] == count
        with np.load(capture / case.case_id / "trace.npz", allow_pickle=False) as trace:
            times = np.arange(count) * 0.02
            np.testing.assert_allclose(
                trace["decision_time_s"], times, rtol=0, atol=1e-9
            )
            np.testing.assert_allclose(
                trace["command"],
                [case.command_at(time) for time in times],
                rtol=0,
                atol=1e-7,
            )
            if witness is None and case.profile != "stand" and count == 1000:
                indices = (99, 100, 149, 150, 699, 700, 749, 750, 999)
                witness = {
                    "case_id": case.case_id,
                    "observed_steps": count,
                    "trace_rows": list(indices),
                    **{
                        name: trace[name][list(indices)].tolist()
                        for name in (
                            "decision_time_s",
                            "command",
                            "root_position_w",
                            "root_com_linear_velocity_b",
                            "root_angular_velocity_b",
                        )
                    },
                }
        total += count
    assert policy.frame_calls == total
    assert (
        witness is not None
    ), "No non-standing native case covered all phase boundaries"
    return witness


def _run(args, app, output, report):
    import torch

    from parkour_lab.evaluation import evaluate, score_capture

    torch.set_num_threads(2)
    policy, checkpoint_sha = _load_fixture(output, args.device)
    before = policy.state_sha256()
    cases = _cases(args.full_bank)
    report["prescribed_cases"] = [case.as_dict() for case in cases]
    capture = output / "capture"
    result = evaluate(
        policy,
        cases,
        capture,
        device=args.device,
        training_seed="untrained-engineering-fixture",
        checkpoint_id="caller-policy-fixture",
        checkpoint_sha256=checkpoint_sha,
        app=app,
    )
    assert policy.state_sha256() == before
    witness = _check_capture(capture, cases, result, policy)
    write_json(output / "phase_boundary_witness.json", witness)
    scored = score_capture(capture)
    assert result == scored
    assert scored["aggregate"]["full_partition_bank"] is args.full_bank
    assert scored["valid_for_comparison"] is args.full_bank
    offline = _offline(capture, output)
    assert offline == scored, "Fresh offline CLI scoring differs from the pure scorer"
    write_json(output / "offline_score.json", offline)
    report.update(
        status="EVALUATION_CHECK_PASSED",
        checkpoint_sha256=checkpoint_sha,
        policy_state_sha256=before,
        policy_unchanged=True,
        policy_frame_calls=policy.frame_calls,
        policy_reset_rows=policy.reset_rows,
        offline_cli_parity=True,
        full_native_phase_coverage=True,
        phase_witness_case_id=witness["case_id"],
        evaluation=result,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--full-bank", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="RUNNING",
        full_bank=args.full_bank,
        fixture_trained=False,
        behavior_validated=False,
        calibration_validated=False,
        native_environments_owned_by="evaluation runner",
        package_sources=package_source_identity(),
        interpreter=dict(executable=sys.executable, version=sys.version),
    )

    def publish():
        write_json(output / "report.json", report)

    app = None
    code = 2
    try:
        write_run_provenance(output, __file__)
        publish()
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device=args.device, fast_shutdown=False).app
        report["dependencies"] = _dependencies(output)
        _run(args, app, output, report)
        code = 0
    except BaseException as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        code = finish_session(None, app, report, publish, code)
        if app is not None:
            exit_native_process(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
