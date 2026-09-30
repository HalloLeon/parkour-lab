"""Frozen causal execution and command recording; not an acceptance evaluator.

Tracking uses true body-frame root-COM velocity only as a simulator metric.
Tracking metrics alone do not establish terrain traversal or qualification.
"""

from pathlib import Path

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
        report.update(
            status="EVALUATION_COMPLETE_NOT_QUALIFIED",
            control_steps=len(commands),
            environment_transitions=len(commands) * env.num_envs,
            tracking_rmse=(squared_error / (len(commands) * env.num_envs))
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
            recorder.finish(completed=error is None, error=error),
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
            for name in ("commands.json", "tracking.npz", "motion.npz")
            if (Path(output) / name).is_file()
        }


def _motion_arrays(rows, initial, origins):
    return {
        **{key: np.stack([row[key] for row in rows]) for key in rows[0]},
        "initial_position_w": initial["position_w"].cpu().numpy(),
        "initial_quaternion_w": initial["quaternion_w"].cpu().numpy(),
        "env_origins": origins,
    }
