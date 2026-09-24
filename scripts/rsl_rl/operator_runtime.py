"""Native simulator input binding for a frozen, motor-bound actor artifact.

No RSL-RL, training policy, command sampler, GUI or hardware driver. The caller
owns source leases and resolves them before act(), then uses the verified motor
bridge to deliver absolute joint targets. time_s is the physics-frame clock, not a remote
packet timestamp or wall clock. Scene/motor configuration must remain frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from parkour_lab.learning.controller import ControllerSession, Sample, finite_tensor
from parkour_lab.learning.recurrent_runtime import (
    BOUND_ACTOR_BUNDLE_VERSION,
    FRAME_DIM,
    FRAME_TERMS,
    _check_tensor,
    load_actor_bundle,
)


if __package__:
    from .operator_motor_bridge import NativeJointTargetBridge, _runtime_motor_binding
else:
    from operator_motor_bridge import NativeJointTargetBridge, _runtime_motor_binding


# Trusted native sensor semantics, not metadata supplied by a backbone. No
# simulator velocity, terrain, privileged latent or adapter-order assumptions.
NATIVE_SENSORS = {
    "base_ang_vel": ((3,), "rad/s", "body"),
    "projected_gravity": ((3,), "unitless", "body"),
    "joint_position_relative_default": ((12,), "rad", "joint"),
    "joint_position": ((12,), "rad", "joint"),
    "joint_velocity": ((12,), "rad/s", "joint"),
    "stock_previous_raw_action": ((12,), "unitless", "joint"),
}


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


class NativeControllerSession:
    """Native sensing/motor binding for an already-loaded causal Controller.

    Artifact loaders/adapters own weights, normalization, history and provenance;
    this host owns named sensing, command delivery and the native motor contract.
    No dynamic imports, policy hot-swaps or backend-specific latent assumptions.
    """

    def __init__(
        self,
        env,
        controller,
        motor_contract,
        *,
        preserve_native_raw=False,
    ):
        self.env = env
        self.failed = False
        self.controller = controller
        self.motor = NativeJointTargetBridge(
            env,
            motor_contract,
            controller.spec.manifest(),
            preserve_native_raw=preserve_native_raw,
        )
        self.motor_verification = self.motor.motor_verification
        for name, spec in controller.spec.sensors.items():
            trusted = NATIVE_SENSORS.get(name)
            if spec.privileged or (trusted is None and spec.required):
                raise ValueError(f"Unsupported native deployment sensor: {name}")
            if trusted is not None and (spec.shape, spec.units, spec.frame) != trusted:
                raise ValueError(f"Native sensor semantics differ: {name}")
        command = env.command_manager.get_term("base_velocity")
        if (
            command.cfg.heading_command
            or command.cfg.rel_heading_envs
            or command.cfg.rel_standing_envs
        ):
            raise ValueError("Native controller requires direct body twist")
        # The profile is not rebound to the current batch hash. Exact normalized
        # equality above establishes compatibility with this original profile.
        self.session = ControllerSession(
            self.controller,
            joint_names=tuple(self.motor.binding["joint_names"]),
            actuator_profile=self.controller.spec.actuator_profile,
        )

    def _check_sensing(self, applied_command, values):
        """A source-specific subclass may additionally audit native observations."""

    @torch.inference_mode()
    def act(self, applied_command, *, time_s, reset_mask):
        """Pack fresh simulator sensors; source events never create a reset mask."""
        if self.failed or self.motor.faulted:
            raise RuntimeError("Native actor session is faulted")
        try:
            env = self.env
            robot = env.scene["robot"].data
            finite_tensor(applied_command, (env.num_envs, 3), robot.joint_pos)
            command = env.command_manager.get_term("base_velocity")
            command.time_left.fill_(float("inf"))
            command.is_standing_env.fill_(False)
            command.is_heading_env.fill_(False)
            command.vel_command_b.copy_(applied_command)
            values = {
                "base_ang_vel": robot.root_ang_vel_b,
                "projected_gravity": robot.projected_gravity_b,
                "joint_position_relative_default": robot.joint_pos
                - robot.default_joint_pos,
                "joint_position": robot.joint_pos,
                "joint_velocity": robot.joint_vel,
                "stock_previous_raw_action": env.action_manager.action,
            }
            self._check_sensing(applied_command, values)
            if (
                not isinstance(reset_mask, torch.Tensor)
                or reset_mask.shape != (env.num_envs,)
                or reset_mask.dtype != torch.bool
                or reset_mask.device != applied_command.device
            ):
                raise ValueError("Native reset mask must match the sensor batch")
            if torch.any(values["stock_previous_raw_action"][reset_mask] != 0):
                raise ValueError("Native previous action must be zero after reset")
            samples = {
                name: (
                    Sample(
                        values[name],
                        time_s,
                        torch.ones(env.num_envs, dtype=torch.bool, device=env.device),
                        NATIVE_SENSORS[name][1],
                        NATIVE_SENSORS[name][2],
                    )
                    if name in values
                    else None
                )
                for name in self.controller.spec.sensors
            }
            return self.session.step(
                time_s=time_s,
                command=applied_command,
                command_time_s=time_s,
                sensors=samples,
                reset_mask=reset_mask,
            )
        except Exception:
            self.failed = True
            raise


class NativeActorSession(NativeControllerSession):
    """Compatibility loader for the exact archived GRU; no shared GRU host API."""

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
        controller, self.metadata, self.artifact_sha256 = load_actor_bundle(
            artifact, env.device, expected_sha256=artifact_sha256
        )
        _check_bundle_source(
            controller,
            self.metadata,
            checkpoint_sha256,
            source_manifest,
            learning_updates,
        )
        if env.cfg.observations.proprio.enable_corruption or tuple(
            env.observation_manager.active_terms["proprio"]
        ) != tuple(name for name, _ in FRAME_TERMS):
            raise ValueError("Native actor requires clean proprioception")
        super().__init__(
            env, controller, self.metadata["motor_contract"], preserve_native_raw=True
        )
        self.frame = None

    def _check_sensing(self, applied_command, values):
        # The archived actor's 45-D packing is an additional parity audit, not
        # an input format imposed on other backbones by the generic host.
        frame = self.env.observation_manager.compute_group(
            "proprio", update_history=False
        )
        independent = torch.cat(
            (
                values["base_ang_vel"],
                values["projected_gravity"],
                applied_command,
                values["joint_position_relative_default"],
                values["joint_velocity"],
                values["stock_previous_raw_action"],
            ),
            dim=-1,
        )
        _check_tensor(frame, (self.env.num_envs, FRAME_DIM), "native proprioception")
        if not torch.equal(frame, independent):
            raise ValueError(
                "Native proprioception differs from the causal sensor contract"
            )
        self.frame = frame.detach().clone()
