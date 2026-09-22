"""Native simulator input binding for a frozen, motor-bound actor artifact.

No RSL-RL, training policy, command sampler, GUI or hardware driver. The caller
owns source leases and resolves them before act(), then delivers raw_action via
the native environment. time_s is the exact physics-frame clock, not a remote
packet timestamp or wall clock. Scene/motor configuration must remain frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from parkour_lab.learning.controller import ControllerSession, Sample
from parkour_lab.learning.motor_contract import verify_runtime_motor
from parkour_lab.learning.recurrent_runtime import (
    BOUND_ACTOR_BUNDLE_VERSION,
    FRAME_DIM,
    FRAME_TERMS,
    _check_tensor,
    load_actor_bundle,
)


def _runtime_motor_binding(env):
    """Bind the resolved native stock motor, not merely an action tensor width."""
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    cfg, descriptor = term.cfg, term.IO_descriptor
    joints = tuple(robot.joint_names)
    default = robot.data.default_joint_pos
    _check_tensor(default, (env.num_envs, 12), "runtime default joint position")
    _check_tensor(
        robot.data.default_joint_vel, default.shape, "runtime default joint velocity"
    )
    if (
        tuple(env.action_manager.active_terms) != ("joint_pos",)
        or env.action_manager.total_action_dim != 12
        or len(joints) != 12
        or len(set(joints)) != 12
        or tuple(descriptor.joint_names) != joints
        or cfg.asset_name != "robot"
        or cfg.joint_names != [".*"]
        or cfg.preserve_order
        or not cfg.use_default_offset
        or cfg.scale != 0.25
        or descriptor.scale != 0.25
        or cfg.clip is not None
        or descriptor.clip is not None
        or cfg.class_type.__name__ != "JointPositionAction"
        or cfg.class_type.__module__ != "isaaclab.envs.mdp.actions.joint_actions"
        or not torch.equal(default, default[0].expand_as(default))
        or torch.any(robot.data.default_joint_vel != 0)
        or not torch.equal(
            torch.as_tensor(
                descriptor.offset, device=default.device, dtype=default.dtype
            ),
            default[0],
        )
    ):
        raise ValueError(
            "Native stock joint order, default pose/velocity or action transform differs"
        )
    actuators = {}
    covered = []
    for name, actuator in robot.actuators.items():
        covered.extend(actuator.joint_names)
        parameters = {}
        for parameter in (
            "stiffness",
            "damping",
            "effort_limit",
            "velocity_limit",
            "effort_limit_sim",
            "velocity_limit_sim",
            "armature",
            "friction",
        ):
            value = getattr(actuator, parameter)
            _check_tensor(value, (env.num_envs, len(actuator.joint_names)), parameter)
            parameters[parameter] = value.detach().cpu().tolist()
        actuators[name] = {
            "joint_names": list(actuator.joint_names),
            "configuration": actuator.cfg.to_dict(),
            "resolved_parameters": parameters,
        }
    if sorted(covered) != sorted(joints):
        raise ValueError("Native actuators must cover each motor joint exactly once")
    binding = {
        "joint_names": list(joints),
        "default_position_rad": default[0].cpu().tolist(),
        "action": cfg.to_dict(),
        "actuators": actuators,
        "step_dt_s": env.step_dt,
        "physics_dt_s": env.physics_dt,
        "decimation": env.cfg.decimation,
    }
    digest = hashlib.sha256(
        json.dumps(binding, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return binding, digest


def _check_bundle_source(
    controller, metadata, checkpoint_sha256, source_manifest, learning_updates
):
    if (
        metadata["format"] != BOUND_ACTOR_BUNDLE_VERSION
        or metadata["source_checkpoint_sha256"] != checkpoint_sha256
        or metadata["source_learning_updates"] != learning_updates
        or json.dumps(controller.spec.manifest(), sort_keys=True, allow_nan=False)
        != json.dumps(source_manifest, sort_keys=True, allow_nan=False)
    ):
        raise ValueError(
            "Native inference requires a motor-bound bundle of this exact source"
        )


def actor_bundle_source(path, checkpoint_sha256, metadata):
    """CPU preflight; bind the exact artifact bytes before starting a worker."""
    path = Path(path).resolve(strict=True)
    controller, bundle_metadata, digest = load_actor_bundle(path)
    _check_bundle_source(
        controller,
        bundle_metadata,
        checkpoint_sha256,
        metadata["controller_manifest"],
        metadata["learning_updates"],
    )
    return {"path": str(path), "sha256": digest}


class NativeActorSession:
    """Verify actual motors before inference; preserve archived actor identity."""

    def __init__(
        self,
        env,
        artifact,
        *,
        checkpoint_sha256,
        source_manifest,
        learning_updates,
        artifact_sha256=None,
    ):
        self.env = env
        self.failed = False
        self.controller, self.metadata, self.artifact_sha256 = load_actor_bundle(
            artifact, env.device, expected_sha256=artifact_sha256
        )
        _check_bundle_source(
            self.controller,
            self.metadata,
            checkpoint_sha256,
            source_manifest,
            learning_updates,
        )
        binding, _ = _runtime_motor_binding(env)
        self.motor_verification = verify_runtime_motor(
            self.metadata["motor_contract"], self.controller.spec.manifest(), binding
        )
        command = env.command_manager.get_term("base_velocity")
        if (
            env.cfg.observations.proprio.enable_corruption
            or tuple(env.observation_manager.active_terms["proprio"])
            != tuple(name for name, _ in FRAME_TERMS)
            or command.cfg.heading_command
            or command.cfg.rel_heading_envs
            or command.cfg.rel_standing_envs
        ):
            raise ValueError(
                "Native actor requires clean proprioception and direct body twist"
            )
        # The profile is not rebound to the current batch hash. Exact normalized
        # equality above establishes compatibility with this original profile.
        self.session = ControllerSession(
            self.controller,
            joint_names=tuple(binding["joint_names"]),
            actuator_profile=self.controller.spec.actuator_profile,
        )
        self.frame = None

    @torch.inference_mode()
    def act(self, applied_command, *, time_s, reset_mask):
        """Pack fresh simulator sensors; source events never create a reset mask."""
        if self.failed:
            raise RuntimeError("Native actor session is faulted")
        try:
            env = self.env
            robot = env.scene["robot"].data
            _check_tensor(applied_command, (env.num_envs, 3), "applied body twist")
            if (
                applied_command.device != robot.joint_pos.device
                or applied_command.dtype != robot.joint_pos.dtype
            ):
                raise ValueError("Body twist must match native sensor device/dtype")
            command = env.command_manager.get_term("base_velocity")
            command.time_left.fill_(float("inf"))
            command.is_standing_env.fill_(False)
            command.is_heading_env.fill_(False)
            command.vel_command_b.copy_(applied_command)
            frame = env.observation_manager.compute()["proprio"]
            values = (
                robot.root_ang_vel_b,
                robot.projected_gravity_b,
                robot.joint_pos - robot.default_joint_pos,
                robot.joint_vel,
                env.action_manager.action,
            )
            independent = torch.cat((*values[:2], applied_command, *values[2:]), dim=-1)
            _check_tensor(frame, (env.num_envs, FRAME_DIM), "native proprioception")
            if not torch.equal(frame, independent):
                raise ValueError(
                    "Native proprioception differs from the causal sensor contract"
                )
            if (
                not isinstance(reset_mask, torch.Tensor)
                or reset_mask.shape != (env.num_envs,)
                or reset_mask.dtype != torch.bool
                or reset_mask.device != frame.device
            ):
                raise ValueError("Native reset mask must match the sensor batch")
            if torch.any(values[-1][reset_mask] != 0):
                raise ValueError("Native previous action must be zero after reset")
            samples = {
                name: Sample(
                    value,
                    time_s,
                    torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
                    spec.units,
                    spec.frame,
                )
                for (name, spec), value in zip(
                    self.controller.spec.sensors.items(), values, strict=True
                )
            }
            result = self.session.step(
                time_s=time_s,
                command=applied_command,
                command_time_s=time_s,
                sensors=samples,
                reset_mask=reset_mask,
            )
            self.frame = frame.detach().clone()
            return result
        except Exception:
            self.failed = True
            raise
