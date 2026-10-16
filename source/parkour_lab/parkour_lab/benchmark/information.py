"""Fixed clean physical context for training, independent of actor sensing."""

from __future__ import annotations

import torch

from parkour_lab.control.proprioception import FOOT_NAMES

from .schema import FIELD_SPECS, information_schema, validate_joint_names

__all__ = ["capture_information", "information_schema", "FIELD_SPECS"]


def _copy(value, shape, name, device):
    if (
        not isinstance(value, torch.Tensor)
        or not value.is_floating_point()
        or tuple(value.shape) != shape
    ):
        raise ValueError(f"Invalid information shape: {name}; expected {shape}")
    result = value.detach().to(device=device, dtype=torch.float32).clone()
    if not torch.isfinite(result).all():
        raise ValueError(f"Nonfinite information: {name}")
    return result


def _dynamics(env, robot, joint_names):
    count, links, device = env.num_envs, len(robot.body_names), env.device
    cache = getattr(env, "_parkour_actual_physics", None)
    if (
        not isinstance(cache, dict)
        or not {"masses", "coms", "materials"} <= cache.keys()
    ):
        raise ValueError("Information requires verified native dynamics readback")
    masses = _copy(cache["masses"], (count, links), "masses", device)
    coms = _copy(cache["coms"], (count, links, 7), "coms", device)
    material = cache["materials"]
    if (
        not isinstance(material, torch.Tensor)
        or material.ndim != 3
        or (
            material.shape[0] != count
            or material.shape[1] < 1
            or material.shape[2] != 3
        )
    ):
        raise ValueError("Invalid information shape: materials")
    material = _copy(material, tuple(material.shape), "materials", device)
    if not torch.equal(
        material[:, :, :2], material[:, :1, :2].expand(-1, material.shape[1], -1)
    ):
        raise ValueError("Information requires uniform robot material friction")
    if robot.body_names.count("base") != 1:
        raise ValueError("Information requires exactly one named base link")
    base = robot.body_names.index("base")
    result = {
        "base_mass": masses[:, base : base + 1].clone(),
        "base_com_offset_b": coms[:, base, :3].clone(),
        "material_friction": material[:, 0, :2].clone(),
    }
    covered = []
    for key in ("joint_stiffness", "joint_damping", "motor_strength"):
        result[key] = torch.empty(count, 12, device=device, dtype=torch.float32)
    for actuator in robot.actuators.values():
        names = tuple(actuator.joint_names)
        if not names or any(name not in joint_names for name in names):
            raise ValueError("Invalid information actuator joint names")
        covered.extend(names)
        ids = [joint_names.index(name) for name in names]
        for target, source in (
            ("joint_stiffness", "stiffness"),
            ("joint_damping", "damping"),
            ("motor_strength", "motor_strength"),
        ):
            value = _copy(
                getattr(actuator, source, None), (count, len(names)), source, device
            )
            result[target][:, ids] = value
    if sorted(covered) != sorted(joint_names):
        raise ValueError("Information actuators must cover every joint exactly once")
    if (result["base_mass"] <= 0).any() or any(
        (result[name] < 0).any()
        for name in (
            "material_friction",
            "joint_stiffness",
            "joint_damping",
            "motor_strength",
        )
    ):
        raise ValueError("Invalid physical dynamics information")
    return result


@torch.no_grad()
def capture_information(env):
    """Copy current physical context; callers control training-only exposure.

    Call at the control boundary, and before reset for outgoing final rows.
    This function neither samples noise nor advances sensor/episode history.
    """
    robot = env.scene["robot"]
    data, count, device = robot.data, env.num_envs, env.device
    joints = validate_joint_names(robot.joint_names)
    position = _copy(data.joint_pos, (count, 12), "joint_pos", device)
    default = _copy(data.default_joint_pos, (count, 12), "default_joint_pos", device)
    result = {
        "root_com_linear_velocity_b": data.root_com_lin_vel_b,
        "root_angular_velocity_b": data.root_ang_vel_b,
        "projected_gravity_b": data.projected_gravity_b,
        "joint_position_relative_default": position - default,
        "joint_velocity": data.joint_vel,
    }
    result = {
        name: _copy(value, (count, *FIELD_SPECS[name][0]), name, device)
        for name, value in result.items()
    }
    quaternion = _copy(data.root_link_quat_w, (count, 4), "root_link_quat_w", device)
    if not torch.allclose(
        quaternion.norm(dim=-1),
        torch.ones_like(quaternion[:, 0]),
        atol=1e-4,
        rtol=0,
    ):
        raise ValueError("Invalid information root quaternion")
    sensor = env.scene["contact_forces"]
    if any(sensor.body_names.count(name) != 1 for name in FOOT_NAMES):
        raise ValueError("Information requires each named foot exactly once")
    forces = sensor.data.net_forces_w
    if not isinstance(forces, torch.Tensor) or tuple(forces.shape) != (
        count,
        len(sensor.body_names),
        3,
    ):
        raise ValueError("Invalid information contact force shape")
    elapsed = env.episode_length_buf
    if (
        not isinstance(elapsed, torch.Tensor)
        or tuple(elapsed.shape) != (count,)
        or elapsed.dtype != torch.long
        or (elapsed < 0).any()
    ):
        raise ValueError("Invalid information episode counters")
    valid = (elapsed > 0).to(device=device)
    forces = (
        forces[:, [sensor.body_names.index(name) for name in FOOT_NAMES]]
        .detach()
        .to(device)
    )
    forces = torch.where(valid[:, None, None], forces, 0.0)
    forces = _copy(forces, (count, 4, 3), "foot_net_forces_w", device)
    xyz = quaternion[:, None, 1:].expand(-1, 4, -1)
    cross = torch.cross(xyz, forces, dim=-1)
    result["foot_net_forces_b"] = (
        forces
        - 2 * quaternion[:, None, :1] * cross
        + 2 * torch.cross(xyz, cross, dim=-1)
    )
    result["foot_force_valid"] = valid[:, None].clone()
    result.update(_dynamics(env, robot, joints))
    root = _copy(data.root_link_pos_w, (count, 3), "root_link_pos_w", device)
    hits = env.scene["height_scanner"].data.ray_hits_w
    if not isinstance(hits, torch.Tensor) or tuple(hits.shape) != (count, 132, 3):
        raise ValueError("Invalid information terrain scan shape")
    hits = hits.detach().to(device=device, dtype=torch.float32)
    valid = torch.isfinite(hits).all(dim=-1)
    distance = root[:, 2:3] - 0.3 - hits[..., 2]
    result["terrain_height"] = torch.where(valid, distance, 0.5).clamp(-0.5, 0.5)
    result["terrain_valid"] = valid.clone()
    for name, value in result.items():
        if value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Nonfinite information: {name}")
    return {name: result[name].detach().clone() for name in FIELD_SPECS}
