"""Seeded flat-task dynamics and starts; import only after the simulator starts."""

from __future__ import annotations

from dataclasses import asdict

import torch
from isaaclab.actuators import DCMotor
from isaaclab.managers import EventTermCfg

from .randomization import TaskRandomization


class RandomizedDCMotor(DCMotor):
    """Scale generated effort before the unchanged native torque-speed limiter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nominal_stiffness = self.stiffness.clone()
        self.nominal_damping = self.damping.clone()
        self.stiffness_scale = torch.ones_like(self.stiffness)
        self.damping_scale = torch.ones_like(self.damping)
        self.motor_strength = torch.ones_like(self.stiffness)

    def _clip_effort(self, effort):
        self.computed_effort = effort * self.motor_strength
        return super()._clip_effort(self.computed_effort)


def get_randomization(env):
    if not hasattr(env, "parkour_randomization"):
        env.parkour_randomization = TaskRandomization(
            env.cfg.parkour_task, evaluation=env.cfg.parkour_evaluation
        )
    return env.parkour_randomization


def _tensor(value, like):
    return torch.as_tensor(value, device=like.device, dtype=like.dtype)


def _material_combine_modes(env):
    """Ensure robot material priority cannot override the terrain's multiply rule."""
    from isaaclab.sim import get_all_matching_child_prims
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    modes = {}
    for links in env.scene["robot"].root_physx_view.link_paths:
        for path in links:
            for prim in get_all_matching_child_prims(
                path,
                lambda prim: prim.HasAPI(UsdPhysics.CollisionAPI),
                stage=env.scene.stage,
            ):
                material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(
                    "physics"
                )
                if material:
                    mode = (
                        PhysxSchema.PhysxMaterialAPI(material.GetPrim())
                        .GetFrictionCombineModeAttr()
                        .Get()
                        or "average"
                    )
                    modes[str(material.GetPath())] = mode
                    if mode not in ("average", "min", "multiply"):
                        raise ValueError(
                            "Robot material overrides the required multiply friction rule"
                        )
    return modes


def initialize_dynamics(env, env_ids=None):
    """Fix one draw per robot; retain stock mass-scaled inertia and native limits."""
    if env_ids is not None:
        raise ValueError("Dynamics initialization applies to the complete environment")
    robot = env.scene["robot"]
    view = robot.root_physx_view
    draw = get_randomization(env).dynamics(len(robot.joint_names))
    masses = robot.data.default_mass.clone().cpu()
    base = robot.body_names.index("base")
    masses[:, base] += _tensor(draw["added_mass"], masses)
    if torch.any(masses <= 0):
        raise ValueError("Randomized masses must remain positive")
    inertias = robot.data.default_inertia.clone().cpu()
    inertias[:, base] *= (masses[:, base] / robot.data.default_mass[:, base].cpu())[
        :, None
    ]
    ids = torch.arange(env.num_envs, device="cpu")
    materials = view.get_material_properties().clone()
    materials[:, :, 0] = _tensor(draw["static_friction"], materials)[:, None]
    materials[:, :, 1] = _tensor(draw["dynamic_friction"], materials)[:, None]
    materials[:, :, 2] = 0.0
    env._parkour_fixed_physics = {
        "masses": masses,
        "materials": materials,
        "coms": view.get_coms().clone(),
        "inertias": inertias,
        "joint_limits": view.get_dof_limits().clone(),
    }
    view.set_masses(masses, ids)
    view.set_inertias(inertias, ids)
    view.set_material_properties(materials, ids)
    env._parkour_material_modes = _material_combine_modes(env)
    for actuator in robot.actuators.values():
        if not isinstance(actuator, RandomizedDCMotor):
            raise ValueError("Flat dynamics require the bounded native DC motor")
        joints = [robot.joint_names.index(name) for name in actuator.joint_names]
        for parameter, key in (("stiffness", "kp_scale"), ("damping", "kd_scale")):
            scale = getattr(actuator, parameter + "_scale")
            scale.copy_(_tensor(draw[key][:, joints], scale))
            getattr(actuator, parameter).copy_(
                getattr(actuator, "nominal_" + parameter) * scale
            )
        actuator.motor_strength.copy_(
            _tensor(draw["motor_strength"][:, joints], actuator.motor_strength)
        )
    dynamics_report(env)


def reset_state(env, env_ids):
    """Jitter nominal positions; start with level attitude and zero velocities."""
    robot = env.scene["robot"]
    ids = env_ids.to(device=env.device, dtype=torch.long)
    draw = get_randomization(env).sample_start(
        ids.cpu().tolist(), len(robot.joint_names)
    )
    root = robot.data.default_root_state[ids].clone()
    root[:, :3] += env.scene.env_origins[ids]
    root[:, :2] += _tensor(draw["start_xy"], root)
    yaw = _tensor(draw["start_yaw"], root)
    root[:, 3:7] = 0.0
    root[:, 3] = torch.cos(yaw / 2)
    root[:, 6] = torch.sin(yaw / 2)
    root[:, 7:] = 0.0
    position = robot.data.default_joint_pos[ids] + _tensor(draw["joint_offset"], root)
    limits = robot.data.soft_joint_pos_limits[ids]
    if torch.any(position < limits[:, :, 0]) or torch.any(position > limits[:, :, 1]):
        raise ValueError("Declared reset jitter exceeds unchanged joint limits")
    robot.write_root_pose_to_sim(root[:, :7], env_ids=ids)
    robot.write_root_velocity_to_sim(root[:, 7:], env_ids=ids)
    robot.write_joint_state_to_sim(position, torch.zeros_like(position), env_ids=ids)


def start_report(env):
    """Verify the first reset's realized state before any physics decision."""
    data = env.scene["robot"].data
    starts = [row["start"] for row in get_randomization(env).manifest()["attempts"]]
    if any(row is None for row in starts):
        raise ValueError("A first start must be sampled for every environment")
    position = data.default_root_state[:, :3] + env.scene.env_origins
    position[:, :2] += _tensor([row["start_xy"] for row in starts], position)
    yaw = _tensor([row["start_yaw"] for row in starts], position)
    orientation = torch.zeros(
        env.num_envs, 4, device=position.device, dtype=position.dtype
    )
    orientation[:, 0], orientation[:, 3] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    joints = data.default_joint_pos + _tensor(
        [row["joint_offset"] for row in starts], position
    )
    actual = dict(
        root_position=data.root_pos_w,
        root_orientation=data.root_quat_w,
        root_velocity=data.root_state_w[:, 7:],
        joint_position=data.joint_pos,
        joint_velocity=data.joint_vel,
    )
    expected = dict(
        root_position=position,
        joint_position=joints,
        root_velocity=torch.zeros_like(actual["root_velocity"]),
        joint_velocity=torch.zeros_like(actual["joint_velocity"]),
    )
    if any(
        not torch.allclose(actual[key], value, rtol=0, atol=2e-6)
        for key, value in expected.items()
    ):
        raise ValueError("Native first start differs from the declared reset draw")
    q = actual["root_orientation"]
    error = torch.minimum(
        (q - orientation).abs().amax(-1), (q + orientation).abs().amax(-1)
    )
    if not torch.isfinite(error).all() or torch.any(error > 2e-6):
        raise ValueError(
            "Native first-start attitude differs from the declared level yaw"
        )
    return {key: value.detach().cpu().tolist() for key, value in actual.items()}


def dynamics_report(env):
    """Check physical readback, not only sampled or configured values."""
    robot = env.scene["robot"]
    view = robot.root_physx_view
    fixed = env._parkour_fixed_physics
    actual = {
        "masses": view.get_masses(),
        "materials": view.get_material_properties(),
        "coms": view.get_coms(),
        "inertias": view.get_inertias(),
        "joint_limits": view.get_dof_limits(),
    }
    for key, value in actual.items():
        expected = fixed[key].to(value)
        # PhysX can round mass/inertia while converting internal representations.
        matches = (
            torch.allclose(value, expected, rtol=1e-6, atol=1e-7)
            if key in ("masses", "inertias")
            else torch.equal(value, expected)
        )
        if not matches:
            raise ValueError(
                "Native dynamics readback differs from the declared fixed draw"
            )
    if hasattr(env, "_parkour_actual_physics"):
        if any(
            not torch.equal(value, env._parkour_actual_physics[key])
            for key, value in actual.items()
        ):
            raise ValueError("Native dynamics changed after initialization")
    else:
        env._parkour_actual_physics = {
            key: value.clone() for key, value in actual.items()
        }
    draw = get_randomization(env).dynamics(len(robot.joint_names))
    motors = {}
    for name, actuator in robot.actuators.items():
        joints = [robot.joint_names.index(joint) for joint in actuator.joint_names]
        for parameter, key in (("stiffness", "kp_scale"), ("damping", "kd_scale")):
            value = getattr(actuator, parameter)
            scale = _tensor(draw[key][:, joints], value)
            if not torch.equal(
                getattr(actuator, parameter + "_scale"), scale
            ) or not torch.equal(
                value, getattr(actuator, "nominal_" + parameter) * scale
            ):
                raise ValueError("Native PD gains differ from the declared fixed draw")
        if not torch.equal(
            actuator.motor_strength,
            _tensor(draw["motor_strength"][:, joints], actuator.motor_strength),
        ):
            raise ValueError(
                "Native motor strength differs from the declared fixed draw"
            )
        motors[name] = {
            key: getattr(actuator, key).detach().cpu().tolist()
            for key in (
                "stiffness",
                "damping",
                "motor_strength",
                "effort_limit",
                "velocity_limit",
            )
        }
    return {
        "randomization": get_randomization(env).manifest(),
        "physical_readback": {
            key: value.detach().cpu().tolist() for key, value in actual.items()
        },
        "requested_physics": {
            key: value.detach().cpu().tolist() for key, value in fixed.items()
        },
        "robot_material_combine_modes": env._parkour_material_modes,
        "motors": motors,
    }


def configure_dynamics(cfg, task, *, evaluation=False):
    """Replace stock random/reset events, retaining nominal native motor limits."""
    cfg.parkour_task = asdict(task)
    cfg.parkour_evaluation = evaluation
    for actuator in cfg.scene.robot.actuators.values():
        if actuator.class_type is not DCMotor:
            raise ValueError("Expected the stock Go2 DC motor configuration")
        actuator.class_type = RandomizedDCMotor
    for name in (
        "physics_material",
        "add_base_mass",
        "base_com",
        "base_external_force_torque",
        "reset_robot_joints",
        "push_robot",
    ):
        setattr(cfg.events, name, None)
    cfg.events.dynamics = EventTermCfg(func=initialize_dynamics, mode="startup")
    cfg.events.reset_base = EventTermCfg(func=reset_state, mode="reset")
    material = cfg.scene.terrain.physics_material
    material.static_friction = material.dynamic_friction = 1.0
    material.restitution = 0.0
    material.friction_combine_mode = "multiply"
    material.restitution_combine_mode = "multiply"
    cfg.sim.physics_material = material
