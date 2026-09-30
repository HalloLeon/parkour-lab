"""Native simulator input binding for a frozen, motor-bound actor artifact.

No RSL-RL, training policy, training command sampler, GUI or hardware driver. The caller
owns source leases and resolves them before act(), then uses the verified motor
bridge to deliver absolute joint targets. time_s is the physics-frame clock, not a remote
packet timestamp or wall clock. Scene/motor configuration must remain frozen.
"""

from __future__ import annotations

import hashlib
import torch

from parkour_lab.control.controller import ControllerSession, Sample, finite_tensor
from parkour_lab.control.proprioception import (
    CONTACT_THRESHOLD_N,
    FOOT_NAMES,
    NATIVE_SENSORS,
)


from parkour_lab.runtime.motor import (
    NativeJointTargetBridge,
)


def causal_sensor_noise(env):
    """Draw only when a causal frame is consumed, never on privileged/terminal reads."""
    if not getattr(env.cfg, "parkour_task", None):
        return None
    randomization = env.parkour_randomization
    noise = randomization.uniform(None, 30, "observation-noise", low=-1.0, high=1.0)
    noise *= [0.2] * 3 + [0.05] * 3 + [0.01] * 12 + [1.5] * 12
    noise = noise.astype("float32")
    if not hasattr(env, "_causal_noise_digest"):
        env._causal_noise_digest = hashlib.sha256()
        env._causal_noise_decisions = 0
    env._causal_noise_digest.update(noise.tobytes())
    env._causal_noise_decisions += 1
    return torch.as_tensor(noise, dtype=torch.float32, device=env.device)


def sensor_noise_report(env):
    if not hasattr(env, "_causal_noise_digest"):
        return None
    return {
        "version": "causal_uniform_noise_v1",
        "generated_frames": env._causal_noise_decisions,
        "draw_sha256": env._causal_noise_digest.hexdigest(),
        "order": ["angular_velocity", "gravity", "joint_position", "joint_velocity"],
        "widths": [3, 3, 12, 12],
        "amplitudes": [0.2, 0.05, 0.01, 1.5],
        "commands_actions_contacts_noiseless": True,
        "scope": "prepared causal frames (training includes one next frame); privileged and terminal-only reads do not draw noise",
    }


def foot_contact_forces(env):
    """Latest native net normal forces in named FR/FL/RR/RL order, world axes."""
    sensor = env.scene["contact_forces"]
    if any(sensor.body_names.count(name) != 1 for name in FOOT_NAMES):
        raise ValueError("Contact sensor must contain each named foot exactly once")
    ids = [sensor.body_names.index(name) for name in FOOT_NAMES]
    forces = sensor.data.net_forces_w[:, ids]
    finite_tensor(forces, (env.num_envs, 4, 3))
    return forces


def foot_contacts(env):
    """Binary net-force flags; no force history, wrench torque or friction added.

    The sensor updates at every physics step. Reset buffers may still expose the
    previous PhysX sample, so rows without a post-reset step report four zeros.
    """
    forces = foot_contact_forces(env)
    flags = torch.linalg.vector_norm(forces, dim=-1) > CONTACT_THRESHOLD_N
    return (flags & (env.episode_length_buf[:, None] > 0)).to(forces.dtype)


def motion_state(env):
    """Owned simulator truth for evaluation only; never part of actor sensing."""
    robot, sensor = env.scene["robot"].data, env.scene["contact_forces"]
    if sensor.body_names.count("base") != 1:
        raise ValueError("Motion capture requires exactly one named base contact link")
    state = {
        "position_w": robot.root_pos_w,
        "quaternion_w": robot.root_quat_w,
        "linear_velocity_b": robot.root_lin_vel_b,
        "angular_velocity_b": robot.root_ang_vel_b,
        "angular_velocity_w": robot.root_ang_vel_w,
        "base_contact_force_w": sensor.data.net_forces_w[
            :, sensor.body_names.index("base")
        ],
        "body_position_w": robot.body_pos_w,
    }
    return {name: value.detach().clone() for name, value in state.items()}


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
        self.last_sensor_noise = None
        self.first_sensor_sample = None
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
            if "foot_contacts" in self.controller.spec.sensors:
                values["foot_contacts"] = foot_contacts(env)
            self.last_sensor_noise = causal_sensor_noise(env)
            if self.last_sensor_noise is not None:
                names = (
                    "base_ang_vel",
                    "projected_gravity",
                    "joint_position_relative_default",
                    "joint_velocity",
                )
                raw = torch.cat([values[name] for name in names], dim=-1)
                for name, perturbation in zip(
                    names,
                    self.last_sensor_noise.split((3, 3, 12, 12), dim=-1),
                    strict=True,
                ):
                    values[name] = values[name] + perturbation
                values["joint_position"] = (
                    values["joint_position"] + self.last_sensor_noise[:, 6:18]
                )
                if self.first_sensor_sample is None:
                    self.first_sensor_sample = {
                        "initial_raw_sensors": raw.cpu().numpy().copy(),
                        "initial_noisy_sensors": torch.cat(
                            [values[name] for name in names], dim=-1
                        )
                        .cpu()
                        .numpy()
                        .copy(),
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
