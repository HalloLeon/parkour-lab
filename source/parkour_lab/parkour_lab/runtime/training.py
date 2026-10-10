"""Shared native control ticks for learners; preprocessing belongs to methods."""

import torch

from parkour_lab.control.controller import JointTargets
from parkour_lab.control.action import BOUNDED_MODE, BOUNDED_RAW_ACTION_MEANING
from parkour_lab.control.motor_contract import make_motor_contract
from parkour_lab.control.proprioception import ACTION_SLICE, CONTACT_SLICE, FRAME_DIM
from parkour_lab.runtime.motor import (
    NATIVE_RAW_ACTION_MEANING,
    NativeJointTargetBridge,
    _runtime_motor_binding,
)
from parkour_lab.runtime.metrics import TransitionMetrics


class TrainingHost:
    """Deliver named observations and verified raw joint actions without an RL API.

    Native groups: proprio = causal 49D, policy = clean privileged 52D,
    terrain = privileged 264D. The historical native group name ``policy`` does
    NOT grant it causal access. Flat causal noise is added at consumption here.
    Final rows stay clean for critic-only bootstrap; they are not actor decisions.
    """

    def __init__(self, env, app, *, action_mode="unbounded"):
        self.env, self.app = env, app
        binding, digest = _runtime_motor_binding(env)
        limits = None
        if action_mode == BOUNDED_MODE:
            native_limits = env.scene["robot"].data.soft_joint_pos_limits
            if not torch.equal(
                native_limits, native_limits[0].expand_as(native_limits)
            ):
                raise ValueError("Require shared native joint target limits")
            limits = native_limits[0].cpu().tolist()
        elif action_mode != "unbounded":
            raise ValueError("Unknown action mode")
        self.manifest = {
            "joint_names": binding["joint_names"],
            "period_s": 0.02,
            "configuration": {"default_position_rad": binding["default_position_rad"]},
            "actuator_profile": "native_motor_sha256:" + digest,
            "raw_action_meaning": NATIVE_RAW_ACTION_MEANING,
        }
        if limits is not None:
            self.manifest["configuration"]["action_clip"] = {
                "version": BOUNDED_MODE,
                "target_limits_rad": limits,
            }
            self.manifest["raw_action_meaning"] = BOUNDED_RAW_ACTION_MEANING
        self.motor_contract = make_motor_contract(
            binding, self.manifest["actuator_profile"], target_limits_rad=limits
        )
        self.bridge = NativeJointTargetBridge(
            env, self.motor_contract, self.manifest, preserve_native_raw=True
        )
        self.previous = torch.zeros((env.num_envs, 12), device=env.device)
        self.first_causal_sample = None
        self.resets = self.partial_reset_steps = self.steps = 0
        self.phase_counts = {}
        self.metrics = TransitionMetrics(env)
        env.capture_diagnostics = True
        if env.step_dt != 0.02 or (env.physics_dt, env.cfg.decimation) not in (
            (0.005, 4),
            (0.0025, 8),
        ):
            raise ValueError("Require 50Hz control and 200 or 400Hz physics")

    def observations(self, native, reset, command=None):
        frame, clean = native["proprio"], native["policy"]
        if command is not None:
            # Replace only commands; do not redraw noisy sensors.
            term = self.env.command_manager.get_term("base_velocity")
            desired = frame.new_tensor(command)
            if (
                desired.shape != (3,)
                or not torch.isfinite(desired).all()
                or not torch.equal(frame[:, 6:9], term.vel_command_b)
                or not torch.equal(clean[:, 9:12], term.vel_command_b)
            ):
                raise ValueError(
                    "Invalid command or native command observation binding"
                )
            desired = desired.expand(self.env.num_envs, 3)
            term.time_left.fill_(float("inf"))
            term.is_standing_env.fill_(False)
            term.is_heading_env.fill_(False)
            term.vel_command_b.copy_(desired)
            frame, clean = frame.clone(), clean.clone()
            frame[:, 6:9], clean[:, 9:12] = desired, desired
        expected = self.previous.clone()
        expected[reset] = 0
        if (
            frame.shape != (self.env.num_envs, FRAME_DIM)
            or clean.shape != (self.env.num_envs, FRAME_DIM + 3)
            or not torch.equal(frame[:, ACTION_SLICE], expected)
            or not torch.equal(frame[:, CONTACT_SLICE], clean[:, -4:])
            or not (
                (frame[:, CONTACT_SLICE] == 0) | (frame[:, CONTACT_SLICE] == 1)
            ).all()
            or (frame[reset, CONTACT_SLICE] != 0).any()
            or not torch.equal(
                frame[:, 6:9], self.env.command_manager.get_command("base_velocity")
            )
            or not torch.equal(
                clean[:, :3], self.env.scene["robot"].data.root_lin_vel_b
            )
        ):
            raise ValueError(
                "Pre-action sensor, contact, command, COM-velocity or previous-action alignment failed"
            )
        from parkour_lab.runtime.native import causal_sensor_noise

        frame = frame.clone()
        noise = causal_sensor_noise(self.env)
        if noise is not None:
            raw = frame.clone() if self.first_causal_sample is None else None
            frame[:, :6] += noise[:, :6]
            frame[:, 9:33] += noise[:, 6:]
            if raw is not None:
                self.first_causal_sample = {
                    "raw": raw.cpu().numpy(),
                    "delivered": frame.cpu().numpy().copy(),
                    "noise": noise.cpu().numpy(),
                }
        return {**native, "proprio": frame, "policy": clean.clone()}

    def reset(self, *, seed=None, command=None):
        native, _ = self.env.reset(**({"seed": seed} if seed is not None else {}))
        self.previous.zero_()
        self.metrics.reset()
        first = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        return self.observations(native, first, command), first

    def step(self, raw, phase, *, next_command=None):
        if not self.app.is_running():
            raise RuntimeError("Simulation application stopped during training")
        raw = raw.detach().clone()
        command = self.env.command_manager.get_command("base_velocity").clone()
        _, target = self.bridge.action_transform(raw)
        delivered = self.bridge.encode(
            JointTargets(self.bridge.joint_names, target, raw)
        )
        with torch.no_grad():
            native, reward, terminated, truncated, extras = self.env.step(delivered)
        self.bridge.verify_delivery(terminated, truncated)
        self.metrics.add(
            phase,
            command,
            reward,
            self.env.reward_manager._step_reward,
            extras["diagnostic_state"],
            terminated,
            truncated,
        )
        done = terminated | truncated
        ids, final = extras["final_env_ids"], extras["final_observation"]
        if not torch.equal(ids, done.nonzero(as_tuple=False).flatten()):
            raise ValueError("Final observation rows do not match episode endings")
        if len(ids) and (
            final is None
            or not torch.equal(final["proprio"][:, ACTION_SLICE], delivered[ids])
        ):
            raise ValueError("Final observations must precede the action-buffer reset")
        self.previous = delivered.clone()
        self.steps += 1
        self.resets += int(done.sum())
        self.partial_reset_steps += int(done.any() and not done.all())
        counts = self.phase_counts.setdefault(
            phase, {"control_steps": 0, "terminated_rows": 0, "timeout_rows": 0}
        )
        counts["control_steps"] += 1
        counts["terminated_rows"] += int(terminated.sum())
        counts["timeout_rows"] += int(truncated.sum())
        return (
            self.observations(native, done, next_command),
            reward,
            done,
            {
                **extras,
                "terminated": terminated,
                "truncated": truncated,
                "time_outs": truncated & ~terminated,
            },
        )
