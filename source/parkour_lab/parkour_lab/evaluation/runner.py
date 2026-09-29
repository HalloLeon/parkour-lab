"""Frozen causal execution and command recording; not an acceptance evaluator.

Tracking uses true body-frame root-COM velocity only as a simulator metric.
Tracking metrics alone do not establish terrain traversal or qualification.
"""

from pathlib import Path

import numpy as np
import torch

from parkour_lab.control.command_tape import TapeBuilder, validate_tape, write_tape
from parkour_lab.methods.roa.runtime import roa_tensor_sha256
from parkour_lab.runtime.native import NativeControllerSession


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
def evaluate(env, app, loaded, commands, output, report):
    if not commands:
        raise ValueError("Evaluation requires a nonempty command sequence")
    host = NativeControllerSession(
        env, loaded.controller, loaded.motor_contract, preserve_native_raw=True
    )
    controller = loaded.controller
    before = roa_tensor_sha256(controller.motor, controller.estimator)
    recorder = TapeBuilder({"seed": env.cfg.seed, "controller_sha256": before})
    _, _ = env.reset(seed=env.cfg.seed)
    reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    squared_error = torch.zeros(3, device=env.device)
    terminated_count = timeout_count = 0
    trace = []
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
            action = host.motor.encode(targets)
            _, _, terminated, timed_out, _ = env.step(action)
            host.motor.verify_delivery(terminated, timed_out)
            recorder.append(step, command)
            squared_error += (actual - applied).square().sum(dim=0)
            terminated_count += int(terminated.sum())
            timeout_count += int((timed_out & ~terminated).sum())
            reset = terminated | timed_out
            trace.append(actual.cpu().numpy())
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
        )
        after = roa_tensor_sha256(controller.motor, controller.estimator)
        if before != after:
            raise RuntimeError("Frozen evaluation changed controller weights")
        report["policy_unchanged"] = True
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        write_tape(
            Path(output) / "commands.json",
            recorder.finish(completed=error is None, error=error),
        )
        if trace:
            np.savez_compressed(
                Path(output) / "tracking.npz", root_com_velocity=np.stack(trace)
            )
