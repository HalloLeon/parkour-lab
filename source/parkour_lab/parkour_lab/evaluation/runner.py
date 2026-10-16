"""Frozen causal execution and command recording; not an acceptance evaluator.

Tracking uses true body-frame root-COM velocity only as a simulator metric.
Tracking metrics alone do not establish terrain traversal or qualification.
"""

from pathlib import Path

import numpy as np
import torch

from parkour_lab.control.command_tape import TapeBuilder, validate_tape, write_tape

# Preserve the original import entry point while archive analysis has its own owner.
from parkour_lab.evaluation.terrain_analysis import (
    analyze_terrain_attempt as analyze_terrain_attempt,
)
from parkour_lab.provenance import file_sha256
from parkour_lab.runtime.metrics import TransitionMetrics
from parkour_lab.runtime.native import (
    NativeControllerSession,
    foot_contact_forces,
    foot_contacts,
    motion_state,
    sensor_noise_report,
)

__all__ = ["command_sequence", "evaluate", "analyze_terrain_attempt"]


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
    assignment = _validate_playback(env, commands, profile, report)
    host = NativeControllerSession(
        env, loaded.controller, loaded.motor_contract, preserve_native_raw=True
    )
    connected = getattr(env.cfg, "parkour_task", {}).get("terrain") == "connected"
    if connected:
        _save_connected_world(env, output, report)
    controller = loaded.controller
    before = controller.state_sha256()
    recorder = TapeBuilder({"seed": env.cfg.seed, "controller_sha256": before})
    randomization, manifest = _reset_attempt(env, assignment, host, output, report)
    reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    previous_capture = env.capture_motion
    previous_diagnostics = env.capture_diagnostics
    playback = _Playback(env, host, controller, recorder)
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
            reset = playback.step(step, command, reset)
            if connected and reset.any():
                # Never execute a replacement attempt after native autoreset.
                first_attempt_ended = True
                break
        playback.publish(report, len(commands), first_attempt_ended)
        if before != controller.state_sha256():
            raise RuntimeError("Frozen evaluation changed controller weights")
        report["policy_unchanged"] = True
        _score_profiles(env, playback, profile, manifest, before, randomization, report)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        env.capture_motion = previous_capture
        env.capture_diagnostics = previous_diagnostics
        playback.save(output, report, error, first_attempt_ended)


def _validate_playback(env, commands, profile, report):
    if not commands:
        raise ValueError("Evaluation requires a nonempty command sequence")
    task = getattr(env.cfg, "parkour_task", {})
    assignment = None
    if (
        task.get("terrain_group") is not None
        or task.get("terrain_attempt_index") is not None
    ):
        from parkour_lab.environments.randomization import terrain_development_attempt

        assignment = terrain_development_attempt(
            task.get("terrain_group"), task.get("terrain_attempt_index")
        )
        prescribed = [
            command
            for duration, command in assignment["command_phases"]
            for _ in range(round(duration * assignment["control_hz"]))
        ]
        if (
            task.get("terrain") != "connected"
            or env.num_envs != 1
            or task.get("world") != assignment["world"]
            or task.get("dynamics") != assignment["stratum"]
            or commands != prescribed
            or env.step_dt != 0.02
            or profile is not None
        ):
            raise ValueError(
                "Terrain development playback differs from its assigned world, stratum or tape"
            )
        report["terrain_assignment"] = assignment
    if profile is not None:
        from .flat import profile_commands

        if commands != profile_commands(profile) or env.step_dt != 0.02:
            raise ValueError("Flat diagnostic requires its complete 50 Hz profile")
    return assignment


def _save_connected_world(env, output, report):
    terrain = env.scene.terrain
    np.savez_compressed(
        Path(output) / "world.npz",
        **{name: value for name, value in terrain.world.items() if name != "metadata"},
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


def _reset_attempt(env, assignment, host, output, report):
    randomization = getattr(env, "parkour_randomization", None)
    if assignment is not None and randomization is None:
        raise ValueError(
            "Terrain development playback requires its native randomization"
        )
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
        if assignment is not None and (
            manifest["namespace"] != assignment["namespace"]
            or manifest["group_id"] != assignment["group_id"]
            or len(manifest["attempts"]) != 1
            or any(
                manifest["attempts"][0][key] != assignment[key]
                for key in ("attempt_index", "stratum", "streams")
            )
        ):
            raise ValueError(
                "Native randomization differs from the selected terrain assignment"
            )
        write_json(Path(output) / "manifest.json", manifest)
        report["manifest_sha256"] = file_sha256(Path(output) / "manifest.json")
        report["task_realization"] = dynamics_report(env)
        report["initial_state"] = start_report(env)
        report["motor_verification"] = host.motor_verification
    return randomization, manifest


def _score_profiles(env, playback, profile, manifest, before, randomization, report):
    if profile is not None:
        from .flat import score_flat_profile

        report["flat_diagnostic"] = score_flat_profile(
            _motion_arrays(playback.motion, playback.initial, playback.origins), profile
        )
        if getattr(env.cfg, "parkour_task", {}).get("bank_profile") is not None:
            from .flat import score_development_profile

            report["flat_development"] = score_development_profile(
                _motion_arrays(playback.motion, playback.initial, playback.origins),
                profile,
                manifest,
            )
            report["flat_development"]["controller_state_sha256"] = before
    if randomization is not None:
        from parkour_lab.environments.dynamics import dynamics_report

        report["task_realization"] = dynamics_report(env)


class _Playback:
    """State and evidence for one frozen playback; native lifecycle stays with the caller."""

    def __init__(self, env, host, controller, recorder):
        self.env = env
        self.host = host
        self.controller = controller
        self.recorder = recorder
        self.squared_error = torch.zeros(3, device=env.device)
        self.terminated_count = self.timeout_count = 0
        self.velocity = []
        self.contacts, self.forces, self.resets = [], [], []
        self.noise, self.motion = [], []
        self.estimates, self.linear_velocity = [], []
        self.metrics = TransitionMetrics(env)
        self.initial = motion_state(env)
        self.origins = env.scene.env_origins.detach().cpu().numpy().copy()

    def step(self, step, command, reset):
        applied = torch.tensor(
            command, dtype=torch.float32, device=self.env.device
        ).expand(self.env.num_envs, 3)
        robot = self.env.scene["robot"].data
        linear_velocity = robot.root_lin_vel_b.clone()
        actual = torch.cat(
            (linear_velocity[:, :2], robot.root_ang_vel_b[:, 2:3]), dim=-1
        )
        if not torch.isfinite(actual).all():
            raise RuntimeError("Nonfinite tracking measurement")
        targets = self.host.act(
            applied, time_s=step * self.env.step_dt, reset_mask=reset
        )
        estimate = getattr(self.controller, "last_estimate", None)
        if estimate is not None:
            estimate = estimate[:, :3].detach().cpu().numpy().copy()
        if self.host.last_sensor_noise is not None:
            self.noise.append(self.host.last_sensor_noise.cpu().numpy())
        contact_sample = None
        if "foot_contacts" in self.controller.spec.sensors:
            # Same pre-action physics sample; raw forces are diagnostics only.
            contact_sample = (
                foot_contacts(self.env).cpu().numpy(),
                foot_contact_forces(self.env).cpu().numpy(),
                reset.cpu().numpy(),
            )
        action = self.host.motor.encode(targets)
        _, reward, terminated, timed_out, extras = self.env.step(action)
        self.host.motor.verify_delivery(terminated, timed_out)
        self.metrics.add(
            "evaluation",
            applied,
            reward,
            self.env.reward_manager._step_reward,
            extras["diagnostic_state"],
            terminated,
            timed_out,
        )
        motion = extras["motion_state"]
        if motion is None:
            raise RuntimeError("Native evaluation requires pre-reset motion evidence")
        self.motion.append(
            {
                **{key: value.cpu().numpy() for key, value in motion.items()},
                "command": applied.cpu().numpy(),
                "terminated": terminated.cpu().numpy(),
                "truncated": timed_out.cpu().numpy(),
            }
        )
        self.recorder.append(step, command)
        self.squared_error += (actual - applied).square().sum(dim=0)
        self.terminated_count += int(terminated.sum())
        self.timeout_count += int((timed_out & ~terminated).sum())
        reset = terminated | timed_out
        self.velocity.append(actual.cpu().numpy())
        self.estimates.append(estimate)
        self.linear_velocity.append(linear_velocity.cpu().numpy())
        if contact_sample is not None:
            self.contacts.append(contact_sample[0])
            self.forces.append(contact_sample[1])
            self.resets.append(contact_sample[2])
        return reset

    def publish(self, report, requested_steps, first_attempt_ended):
        completed_steps = len(self.velocity)
        report.update(
            status=(
                "FIRST_ATTEMPT_ENDED_NOT_QUALIFIED"
                if first_attempt_ended
                else "EVALUATION_COMPLETE_NOT_QUALIFIED"
            ),
            requested_control_steps=requested_steps,
            control_steps=completed_steps,
            environment_transitions=completed_steps * self.env.num_envs,
            tracking_rmse=(self.squared_error / (completed_steps * self.env.num_envs))
            .sqrt()
            .cpu()
            .tolist(),
            terminated_rows=self.terminated_count,
            timeout_rows=self.timeout_count,
            motor_delivery=self.host.motor.progress(),
            task_metrics=self.metrics.drain(),
        )

    def save(self, output, report, error, first_attempt_ended):
        report["sensor_noise"] = sensor_noise_report(self.env)
        write_tape(
            Path(output) / "commands.json",
            self.recorder.finish(
                completed=error is None and not first_attempt_ended, error=error
            ),
        )
        if self.velocity:
            sensors = {}
            if self.contacts:
                sensors = dict(
                    foot_contacts=np.stack(self.contacts),
                    foot_net_forces_w=np.stack(self.forces),
                    reset_mask=np.stack(self.resets),
                )
            if any(value is not None for value in self.estimates):
                sensors.update(
                    estimated_linear_velocity_b=np.stack(
                        [
                            (
                                value
                                if value is not None
                                else np.full_like(self.linear_velocity[0], np.nan)
                            )
                            for value in self.estimates
                        ]
                    ),
                    root_com_linear_velocity_b=np.stack(self.linear_velocity),
                )
            np.savez_compressed(
                Path(output) / "tracking.npz",
                root_com_velocity=np.stack(self.velocity),
                **(
                    {
                        "causal_sensor_noise": np.stack(self.noise),
                        **self.host.first_sensor_sample,
                    }
                    if self.noise
                    else {}
                ),
                **sensors,
            )
        if self.motion:
            np.savez_compressed(
                Path(output) / "motion.npz",
                **_motion_arrays(self.motion, self.initial, self.origins),
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
