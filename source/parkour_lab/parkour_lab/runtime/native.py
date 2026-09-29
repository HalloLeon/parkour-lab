"""Native simulator input binding for a frozen, motor-bound actor artifact.

No RSL-RL, training policy, training command sampler, GUI or hardware driver. The caller
owns source leases and resolves them before act(), then uses the verified motor
bridge to deliver absolute joint targets. time_s is the physics-frame clock, not a remote
packet timestamp or wall clock. Scene/motor configuration must remain frozen.
"""

from __future__ import annotations


import torch

from parkour_lab.control.controller import ControllerSession, Sample, finite_tensor
from parkour_lab.control.proprioception import NATIVE_SENSORS


from parkour_lab.runtime.motor import (
    NativeJointTargetBridge,
)


def configure_external_command(command, *, command_class=None):
    """Zero on native reset; the host owns every applied body-twist command.

    Call after AppLauncher when using the default native class. No terrain
    sampler, heading assistance or standing override may compete with the host.
    The host still freezes the timer and clears mode flags on each delivery.
    """
    if command_class is None:
        from isaaclab.envs.mdp.commands import UniformVelocityCommand

        command_class = UniformVelocityCommand
    command.class_type = command_class
    command.heading_command = False
    command.rel_heading_envs = command.rel_standing_envs = 0.0
    command.ranges.heading = None
    command.ranges.lin_vel_x = command.ranges.lin_vel_y = command.ranges.ang_vel_z = (
        0.0,
        0.0,
    )
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.debug_vis = False


def validate_native_controller_spec(spec):
    """Check trusted deployment sensor semantics without starting a simulator."""
    for name, sensor in spec.sensors.items():
        trusted = NATIVE_SENSORS.get(name)
        if sensor.privileged or (trusted is None and sensor.required):
            raise ValueError(f"Unsupported native deployment sensor: {name}")
        if (
            trusted is not None
            and (sensor.shape, sensor.units, sensor.frame) != trusted
        ):
            raise ValueError(f"Native sensor semantics differ: {name}")


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
        validate_native_controller_spec(controller.spec)
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
