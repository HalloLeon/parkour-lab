"""Frozen causal execution and command recording; not an acceptance evaluator.

Tracking uses true body-frame root-COM velocity only as a simulator metric.
Tracking metrics alone do not establish terrain traversal or qualification.
"""

from pathlib import Path
import json

import numpy as np
import torch

from parkour_lab.provenance import file_sha256
from parkour_lab.runtime.metrics import TransitionMetrics
from parkour_lab.control.command_tape import TapeBuilder, validate_tape, write_tape
from parkour_lab.runtime.native import (
    NativeControllerSession,
    foot_contacts,
    foot_contact_forces,
    motion_state,
    sensor_noise_report,
)


def command_sequence(*, tape=None, command=(0.3, 0.0, 0.0), steps=900):
    if tape is not None:
        validate_tape(tape)
        return [
            segment["command"]
            for segment in tape["segments"]
            for _ in range(segment["start_step"], segment["end_step_exclusive"])
        ]
    if type(steps) is not int or not 1 <= steps <= 30000:
        raise ValueError("steps must be in [1, 30000]")
    # Finish with a one-second zero-command window. This is not proof of stopping.
    builder = TapeBuilder({})
    for step in range(steps):
        builder.append(step, command if step < steps - 50 else (0.0, 0.0, 0.0))
    generated = builder.finish(completed=True)
    return command_sequence(tape=generated)


@torch.no_grad()
def evaluate(env, app, loaded, commands, output, report, *, profile=None):
    if not commands:
        raise ValueError("Evaluation requires a nonempty command sequence")
    if profile is not None:
        from .flat import profile_commands

        if commands != profile_commands(profile) or env.step_dt != 0.02:
            raise ValueError("Flat diagnostic requires its complete 50 Hz profile")
    host = NativeControllerSession(
        env, loaded.controller, loaded.motor_contract, preserve_native_raw=True
    )
    connected = getattr(env.cfg, "parkour_task", {}).get("terrain") == "connected"
    if connected:
        terrain = env.scene.terrain
        np.savez_compressed(
            Path(output) / "world.npz",
            **{
                name: value
                for name, value in terrain.world.items()
                if name != "metadata"
            },
        )
        from parkour_lab.control.proprioception import FOOT_NAMES

        report["connected_world"] = {
            "source": terrain.metadata,
            "native": terrain.native_receipt,
            "world_sha256": file_sha256(Path(output) / "world.npz"),
            "foot_names": list(FOOT_NAMES),
            "trace_phase": "motion.npz: post-control-step, outgoing state before auto-reset; tracking.npz: pre-action",
            "contact_scope": "Latest physics-sample net normal forces (N), not substep history; initial raw forces may be stale after reset (actor flags are zero); foot link positions are not collision surfaces",
            "first_attempt_only": True,
        }
    controller = loaded.controller
    before = controller.state_sha256()
    recorder = TapeBuilder({"seed": env.cfg.seed, "controller_sha256": before})
    randomization = getattr(env, "parkour_randomization", None)
    if randomization is not None:
        randomization.begin_attempts()
        for name in ("_causal_noise_digest", "_causal_noise_decisions"):
            if hasattr(env, name):
                delattr(env, name)
    _, _ = env.reset(seed=env.cfg.seed)
    manifest = None
    if randomization is not None:
        from parkour_lab.environments.dynamics import dynamics_report, start_report
        from parkour_lab.provenance import write_json

        manifest = randomization.manifest()
        write_json(Path(output) / "manifest.json", manifest)
        report["manifest_sha256"] = file_sha256(Path(output) / "manifest.json")
        report["task_realization"] = dynamics_report(env)
        report["initial_state"] = start_report(env)
        report["motor_verification"] = host.motor_verification
    reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    squared_error = torch.zeros(3, device=env.device)
    terminated_count = timeout_count = 0
    trace = []
    contact_trace, force_trace, reset_trace = [], [], []
    noise_trace = []
    motion_trace = []
    previous_capture = env.capture_motion
    previous_diagnostics = env.capture_diagnostics
    metrics = TransitionMetrics(env)
    initial = motion_state(env)
    origins = env.scene.env_origins.detach().cpu().numpy().copy()
    env.capture_motion = True
    env.capture_diagnostics = True
    error = None
    first_attempt_ended = False
    try:
        for step, command in enumerate(commands):
            if not app.is_running():
                raise RuntimeError(
                    "Simulation application stopped before the tape ended"
                )
            applied = torch.tensor(
                command, dtype=torch.float32, device=env.device
            ).expand(env.num_envs, 3)
            robot = env.scene["robot"].data
            actual = torch.cat(
                (robot.root_lin_vel_b[:, :2], robot.root_ang_vel_b[:, 2:3]), dim=-1
            )
            if not torch.isfinite(actual).all():
                raise RuntimeError("Nonfinite tracking measurement")
            targets = host.act(applied, time_s=step * env.step_dt, reset_mask=reset)
            if host.last_sensor_noise is not None:
                noise_trace.append(host.last_sensor_noise.cpu().numpy())
            contact_sample = None
            if "foot_contacts" in controller.spec.sensors:
                # Same pre-action physics sample; raw forces are diagnostics only.
                contact_sample = (
                    foot_contacts(env).cpu().numpy(),
                    foot_contact_forces(env).cpu().numpy(),
                    reset.cpu().numpy(),
                )
            action = host.motor.encode(targets)
            _, reward, terminated, timed_out, extras = env.step(action)
            host.motor.verify_delivery(terminated, timed_out)
            metrics.add(
                "evaluation",
                applied,
                reward,
                env.reward_manager._step_reward,
                extras["diagnostic_state"],
                terminated,
                timed_out,
            )
            motion = extras["motion_state"]
            if motion is None:
                raise RuntimeError(
                    "Native evaluation requires pre-reset motion evidence"
                )
            motion_trace.append(
                {
                    **{key: value.cpu().numpy() for key, value in motion.items()},
                    "command": applied.cpu().numpy(),
                    "terminated": terminated.cpu().numpy(),
                    "truncated": timed_out.cpu().numpy(),
                }
            )
            recorder.append(step, command)
            squared_error += (actual - applied).square().sum(dim=0)
            terminated_count += int(terminated.sum())
            timeout_count += int((timed_out & ~terminated).sum())
            reset = terminated | timed_out
            trace.append(actual.cpu().numpy())
            if contact_sample is not None:
                contact_trace.append(contact_sample[0])
                force_trace.append(contact_sample[1])
                reset_trace.append(contact_sample[2])
            if connected and reset.any():
                # The SDK already reset this row; never execute a replacement attempt.
                first_attempt_ended = True
                break
        completed_steps = len(trace)
        report.update(
            status=(
                "FIRST_ATTEMPT_ENDED_NOT_QUALIFIED"
                if first_attempt_ended
                else "EVALUATION_COMPLETE_NOT_QUALIFIED"
            ),
            requested_control_steps=len(commands),
            control_steps=completed_steps,
            environment_transitions=completed_steps * env.num_envs,
            tracking_rmse=(squared_error / (completed_steps * env.num_envs))
            .sqrt()
            .cpu()
            .tolist(),
            terminated_rows=terminated_count,
            timeout_rows=timeout_count,
            motor_delivery=host.motor.progress(),
            task_metrics=metrics.drain(),
        )
        after = controller.state_sha256()
        if before != after:
            raise RuntimeError("Frozen evaluation changed controller weights")
        report["policy_unchanged"] = True
        if profile is not None:
            from .flat import score_flat_profile

            report["flat_diagnostic"] = score_flat_profile(
                _motion_arrays(motion_trace, initial, origins), profile
            )
            if getattr(env.cfg, "parkour_task", {}).get("bank_profile") is not None:
                from .flat import score_development_profile

                report["flat_development"] = score_development_profile(
                    _motion_arrays(motion_trace, initial, origins), profile, manifest
                )
                report["flat_development"]["controller_state_sha256"] = before
        if randomization is not None:
            report["task_realization"] = dynamics_report(env)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        env.capture_motion = previous_capture
        env.capture_diagnostics = previous_diagnostics
        report["sensor_noise"] = sensor_noise_report(env)
        write_tape(
            Path(output) / "commands.json",
            recorder.finish(
                completed=error is None and not first_attempt_ended, error=error
            ),
        )
        if trace:
            sensors = {}
            if contact_trace:
                sensors = dict(
                    foot_contacts=np.stack(contact_trace),
                    foot_net_forces_w=np.stack(force_trace),
                    reset_mask=np.stack(reset_trace),
                )
            np.savez_compressed(
                Path(output) / "tracking.npz",
                root_com_velocity=np.stack(trace),
                **(
                    {
                        "causal_sensor_noise": np.stack(noise_trace),
                        **host.first_sensor_sample,
                    }
                    if noise_trace
                    else {}
                ),
                **sensors,
            )
        if motion_trace:
            np.savez_compressed(
                Path(output) / "motion.npz",
                **_motion_arrays(motion_trace, initial, origins),
            )
        report["evidence_sha256"] = {
            name: file_sha256(Path(output) / name)
            for name in ("commands.json", "tracking.npz", "motion.npz", "world.npz")
            if (Path(output) / name).is_file()
        }


def _motion_arrays(rows, initial, origins):
    return {
        **{key: np.stack([row[key] for row in rows]) for key in rows[0]},
        **{f"initial_{key}": value.cpu().numpy() for key, value in initial.items()},
        "env_origins": origins,
    }


def analyze_terrain_attempt(run):
    """Verify saved evidence and score one attempt; never execute or replace it.

    This is single-attempt kinematics, not a frozen terrain bank, independently
    validated layout/randomization, cooked-contact certification or qualification.
    """
    from .terrain import score_terrain_attempt
    from parkour_lab.control.proprioception import FOOT_NAMES

    path = Path(run)
    path = path / "report.json" if path.is_dir() else path
    report = json.loads(path.read_text())
    task = report.get("config", {}).get("task", {})
    world = report.get("connected_world", {})
    steps = report.get("control_steps")

    def require(condition, message):
        if not condition:
            raise ValueError(f"{path}: {message}")

    require(
        task.get("terrain") == "connected"
        and task.get("num_envs") == 1
        and world.get("first_attempt_only") is True
        and world.get("foot_names") == list(FOOT_NAMES),
        "require a single connected-world attempt with named feet",
    )
    require(
        report.get("status")
        in {"EVALUATION_COMPLETE_NOT_QUALIFIED", "FIRST_ATTEMPT_ENDED_NOT_QUALIFIED"}
        and report.get("policy_unchanged") is True
        and report.get("cleanup")
        == {"environment": "complete", "application": "complete"}
        and type(steps) is int
        and 0 < steps <= 30000
        and report.get("environment_transitions") == steps,
        "require recorded steps, frozen policy and complete native cleanup",
    )
    actor = report.get("actor_sha256", "")
    require(
        isinstance(actor, str)
        and len(actor) == 64
        and all(c in "0123456789abcdef" for c in actor)
        and isinstance(report.get("package_sources"), dict)
        and bool(report["package_sources"]),
        "missing actor or executed-source identity",
    )
    for name, digest in {
        "manifest.json": report.get("manifest_sha256"),
        **{
            name: report.get("evidence_sha256", {}).get(name)
            for name in ("commands.json", "motion.npz", "tracking.npz", "world.npz")
        },
    }.items():
        evidence = path.parent / name
        require(
            evidence.is_file() and file_sha256(evidence) == digest,
            f"missing or changed evidence: {name}",
        )

    tape = validate_tape(
        json.loads((path.parent / "commands.json").read_text()), require_complete=False
    )
    require(
        tape["steps"] == steps and tape["error"] is None, "inconsistent consumed tape"
    )
    commands = np.asarray(
        [
            segment["command"]
            for segment in tape["segments"]
            for _ in range(segment["start_step"], segment["end_step_exclusive"])
        ]
    )
    with np.load(path.parent / "motion.npz", allow_pickle=False) as archive:
        motion = dict(archive)
    require(
        all(np.isfinite(value).all() for value in motion.values()),
        "nonfinite motion evidence",
    )
    result = score_terrain_attempt(motion, world["source"])
    require(
        motion["command"].shape == (steps, 1, 3)
        and np.allclose(motion["command"][:, 0], commands, rtol=0, atol=1e-6),
        "motion and consumed tape disagree",
    )
    ended = motion["terminated"] | motion["truncated"]
    endings = np.flatnonzero(ended[:, 0])
    require(
        not len(endings) or endings.tolist() == [steps - 1],
        "trace continues after its first ending",
    )
    first_ended = report["status"] == "FIRST_ATTEMPT_ENDED_NOT_QUALIFIED"
    requested = report.get("requested_control_steps")
    require(
        type(requested) is int
        and steps <= requested <= 30000
        and first_ended == bool(len(endings))
        and tape["complete"] == (not first_ended)
        and (first_ended or steps == requested)
        and report.get("terminated_rows") == int(motion["terminated"].sum())
        and report.get("timeout_rows")
        == int((motion["truncated"] & ~motion["terminated"]).sum()),
        "inconsistent first-ending, completion or requested-step receipts",
    )
    motor = report.get("motor_delivery", {})
    require(
        all(
            motor.get(key) == steps
            for key in (
                "encoded_steps",
                "verified_delivery_steps",
                "native_step_returns",
            )
        )
        and motor.get("faulted") is False
        and motor.get("pending_delivery") is False
        and motor.get("excluded_terminal_rows") == len(endings)
        and motor.get("native_verified_rows") == steps - len(endings),
        "incomplete or invalid native motor delivery",
    )
    with np.load(path.parent / "tracking.npz", allow_pickle=False) as archive:
        shapes = {
            "root_com_velocity": (steps, 1, 3),
            "causal_sensor_noise": (steps, 1, 30),
            "initial_raw_sensors": (1, 30),
            "initial_noisy_sensors": (1, 30),
            "foot_contacts": (steps, 1, 4),
            "foot_net_forces_w": (steps, 1, 4, 3),
            "reset_mask": (steps, 1),
        }
        require(
            all(
                name in archive and archive[name].shape == shape
                for name, shape in shapes.items()
            ),
            "incomplete tracking/input evidence",
        )
        reset = archive["reset_mask"]
        require(
            reset.shape == (steps, 1)
            and reset.dtype == np.bool_
            and reset[0, 0]
            and not reset[1:].any(),
            "missing initial reset or replacement attempt",
        )
        require(
            all(
                value.shape[0] == steps
                for name, value in archive.items()
                if not name.startswith("initial_")
            ),
            "incomplete tracking trace",
        )
        require(
            all(np.isfinite(value).all() for value in archive.values()),
            "nonfinite tracking evidence",
        )

    # The short diagnostic can share a valid prefix, but cannot become a 30 s trial.
    prescribed = requested == 1500
    failures = list(result["failures"])
    if not prescribed:
        failures.append("requested tape is not the prescribed 30 s terrain trial")
    return {
        **result,
        "diagnostic_passed": result["kinematic_passed"] and prescribed,
        "failures": failures,
        "requested_control_steps": requested,
        "prescribed_duration": prescribed,
        "evidence_verified": True,
        "report": str(path.resolve()),
        "actor_sha256": actor,
        "executed_package_sources": report["package_sources"],
        "qualified": False,
        "qualification_eligible": False,
        "scope": "Single-attempt recorded kinematics and archive integrity only; not layout/dynamics-bank validation, cooked-contact certification or qualification.",
    }
