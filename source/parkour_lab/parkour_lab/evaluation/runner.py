"""Fixed-case playback of an already-loaded policy; no learner or model loader."""

from pathlib import Path

import numpy as np
import torch

from parkour_lab.control.command_tape import TapeBuilder, write_tape
from parkour_lab.provenance import (
    dependency_identity,
    package_source_identity,
    write_json,
    write_run_provenance,
)
from parkour_lab.runtime.native import motion_state, sensor_noise_report
from parkour_lab.runtime.session import finish_session

from .capture import (
    IDENTITY_FIELDS,
    VERSION,
    publish_capture,
    reward_identity,
    save_trace,
    state_arrays,
)

__all__ = ["evaluate"]


def _native_dependencies():
    """Identify the running stack after the caller has started AppLauncher."""
    return {
        name: dependency_identity(name, name)
        for name in (
            "isaaclab",
            "isaaclab_assets",
            "isaaclab_tasks",
            "isaacsim",
            "torch",
            "gymnasium",
            "numpy",
        )
    }


def _identity(policy):
    identity = policy.state_sha256()
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError("Require a lowercase SHA-256 policy state identity")
    return identity


def evaluate(
    policy,
    cases,
    output,
    *,
    device="cuda:0",
    training_seed="unspecified",
    checkpoint_id="caller-loaded",
    checkpoint_sha256=None,
    app=None,
    repository_path=None,
):
    """Evaluate frozen caller-owned inference after the caller starts AppLauncher.

    ``policy`` is a FramePolicy or a named Controller. Its state hash includes
    parameters, buffers and normalization, excluding causal episode memory.
    Cases execute sequentially with one replica and independent semantic seeds;
    the caller retains ownership of the application and checkpoint format.
    """
    from .cases import EvaluationCase

    cases = tuple(cases)
    if (
        not cases
        or any(not isinstance(case, EvaluationCase) for case in cases)
        or len({case.case_id for case in cases}) != len(cases)
        or len({case.partition for case in cases}) != 1
    ):
        raise ValueError("Require distinct prescribed cases from one partition")
    if any(case.version != "v2" for case in cases):
        raise ValueError(
            "Legacy case versions are offline-only; collect the current recipe"
        )
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        raise ValueError("Require the caller's checkpoint identity")
    if checkpoint_sha256 is not None and (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(c not in "0123456789abcdef" for c in checkpoint_sha256)
    ):
        raise ValueError("Invalid caller checkpoint SHA-256")
    if not isinstance(training_seed, (str, int)) or isinstance(training_seed, bool):
        raise ValueError("Require a caller training-seed label")
    before = _identity(policy)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": VERSION,
        "status": "RUNNING",
        "cases": [case.as_dict() for case in cases],
        "case_version": "v2",
        "training_seed": training_seed,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "policy_state_sha256": before,
        "package_sources": package_source_identity(),
        "native_dependencies": None,
        "execution": "sequential single-replica first attempts",
    }
    if any(case.condition != "id" for case in cases):
        from .robustness import VERSION as robustness_version

        manifest["robustness_version"] = robustness_version
    try:
        manifest["native_dependencies"] = _native_dependencies()
        if repository_path is not None:
            write_run_provenance(folder, repository_path)
        publish_capture(folder, manifest)
        for case in cases:
            if app is not None and not app.is_running():
                manifest["status"] = "INTERRUPTED"
                break
            interrupted = _run_case(
                policy, case, folder / case.case_id, device, manifest, app
            )
            if _identity(policy) != before:
                raise RuntimeError("Policy parameters or preprocessing changed")
            if interrupted:
                manifest["status"] = "INTERRUPTED"
                break
        else:
            manifest["status"] = "COMPLETE"
    except BaseException as error:
        manifest.update(status="ERROR", error=str(error))
        raise
    finally:
        manifest.setdefault("core_reward_recipe_sha256", None)
        try:
            manifest["final_policy_state_sha256"] = _identity(policy)
        except Exception as error:
            manifest.update(status="ERROR", state_identity_error=str(error))
            manifest["final_policy_state_sha256"] = None
        if manifest["final_policy_state_sha256"] != before:
            manifest["status"] = "ERROR"
        report = publish_capture(folder, manifest)
    return report


def _run_case(policy, case, folder, device, manifest, app):
    from .environment import (
        bind_policy,
        check_fixed_terrain,
        make_case_task,
        realization,
        reset_native,
    )

    folder.mkdir()
    task = playback = None
    receipt = {
        "version": VERSION,
        "case": case.as_dict(),
        "status": "RUNNING",
        **{key: manifest[key] for key in IDENTITY_FIELDS},
    }
    try:
        task = make_case_task(case, device)
        reset = reset_native(task)
        receipt["realization"] = realization(task)
        recipe = receipt["realization"]["task"]["reward_recipe"]
        digest = reward_identity(recipe)
        if manifest.setdefault("core_reward_recipe_sha256", digest) != digest:
            raise RuntimeError("Fixed evaluation core reward recipe changed")
        receipt["core_reward_recipe_sha256"] = digest
        # Persist the common recipe before native stepping/cleanup: a process
        # ending there must still leave independently scoreable physical traces.
        write_json(folder.parent / "manifest.json", manifest)
        host = bind_policy(task._env, policy)
        receipt["controller"] = host.controller.spec.manifest()
        receipt["motor_verification"] = host.motor_verification
        playback = _Playback(task._env, host, case, reset)
        playback.run(app)
        check_fixed_terrain(task, receipt["realization"])
        if _identity(policy) != manifest["policy_state_sha256"]:
            raise RuntimeError("Policy parameters or preprocessing changed")
        receipt["status"] = "INTERRUPTED" if playback.interrupted else "CAPTURED"
    except BaseException as error:
        receipt.update(status="ERROR", error=str(error))
        raise
    finally:
        try:
            if playback is not None:
                playback.save(folder, receipt)
            if task is not None:
                receipt["interactions"] = task.interactions
        except BaseException as error:
            receipt.update(status="ERROR", error=str(error))
            raise
        finally:
            failed = receipt["status"] == "ERROR"
            code = finish_session(
                task,
                None,
                receipt,
                lambda: write_json(folder / "capture.json", receipt),
                2 if failed else 0,
            )
            if code and not failed:
                raise RuntimeError("Evaluation capture cleanup/publication failed")
    return playback.interrupted


class _Playback:
    """The single native stepping loop, retaining only the original attempt."""

    def __init__(self, env, host, case, reset):
        self.env, self.host, self.case, self.reset = env, host, case, reset
        self.rows = []
        self.initial = state_arrays(motion_state(env))
        self.origin = env.scene.env_origins[0].detach().cpu().numpy().copy()
        self.tape = TapeBuilder({"case": case.as_dict(), "capture_version": VERSION})
        self.interrupted = False

    @torch.inference_mode()
    def run(self, app):
        for step in range(1000):
            if app is not None and not app.is_running():
                self.interrupted = True
                break
            if self.step(step):
                break

    def step(self, step):
        time_s = step * self.env.step_dt
        command = self.case.command_at(time_s)
        applied = torch.tensor([command], dtype=torch.float32, device=self.env.device)
        targets = self.host.act(applied, time_s=time_s, reset_mask=self.reset)
        action = self.host.motor.encode(targets)
        _, reward, terminated, truncated, extras = self.env.step(action)
        # Physical collection has happened even if later delivery verification
        # raises. Retain its outgoing sample and reward before that check.
        state = extras["motion_state"]
        if state is None:
            raise RuntimeError("Native playback requires owned outgoing motion")
        self.rows.append(
            {
                **state_arrays(state),
                "decision_time_s": np.asarray(time_s),
                "command": applied[0].cpu().numpy().copy(),
                "reward": reward[0].detach().cpu().numpy().copy(),
                "terminated": terminated[0].cpu().numpy().copy(),
                "truncated": truncated[0].cpu().numpy().copy(),
            }
        )
        self.tape.append(step, command)
        self.host.motor.verify_delivery(terminated, truncated)
        self.reset = terminated | truncated
        return bool(self.reset.any())

    def save(self, folder, receipt):
        receipt["trace_sha256"] = save_trace(
            folder, self.rows, self.initial, self.origin
        )
        receipt["sensor_noise"] = sensor_noise_report(self.env)
        receipt["motor_delivery"] = self.host.motor.progress()
        receipt["observed_steps"] = len(self.rows)
        write_tape(
            folder / "commands.json",
            self.tape.finish(
                completed=len(self.rows) == 1000 and receipt["status"] == "CAPTURED",
                error=receipt.get("error"),
            ),
        )
