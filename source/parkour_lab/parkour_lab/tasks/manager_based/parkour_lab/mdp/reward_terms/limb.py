# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Limb-level joint, contact, and motion rewards."""

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from .. import config
from .._shared import contact, robot, runtime
from ..curriculums.config import DEFAULT_PARKOUR_CURRICULUM, ParkourCurriculumCfg
from ..terrain import edges


def feet_edge(
    env: ManagerBasedRLEnv,
    curriculum_cfg: ParkourCurriculumCfg = DEFAULT_PARKOUR_CURRICULUM,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """Count contacted feet near a traversable support boundary.

    Contact gating prevents a swinging foot that merely passes over an edge
    from being penalized. Use the returned count with a negative reward weight.

    Returns:
        Floating tensor with shape ``(num_envs,)``.
    """

    return edges.foot_edge_contact_mask(
        env,
        curriculum_cfg=curriculum_cfg,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    ).sum(dim=-1, dtype=torch.float32)


def feet_stumble(
    env: ManagerBasedRLEnv,
    stumble_cfg: config.FeetStumbleCfg = config.DEFAULT_FEET_STUMBLE,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """
    Penalize feet hitting near-vertical surfaces.

    A stumble is detected when lateral contact force is large compared with
    vertical contact force.

    Returns:
        [num_envs]
    """

    contact_forces = contact._selected_contact_forces_w_history(
        env, sensor_cfg=sensor_cfg
    )

    lateral_force = torch.linalg.norm(contact_forces[..., :2], dim=-1)
    vertical_force = torch.abs(contact_forces[..., 2])

    valid_vertical_contact = vertical_force > stumble_cfg.min_vertical_force

    stumble = torch.logical_and(
        valid_vertical_contact,
        lateral_force > stumble_cfg.lateral_to_vertical_force_ratio * vertical_force,
    )

    return torch.any(stumble, dim=(1, 2)).float()


def joint_deviation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize selected joints deviating from their default pose.

    Returns:
        [num_envs]
    """

    joint_error = robot._selected_joint_pos_error(env, asset_cfg)

    return torch.sum(joint_error.square(), dim=-1)


def rapid_feet_motion_l2(
    env: ManagerBasedRLEnv,
    motion_cfg: config.FeetMotionCfg = config.DEFAULT_FOOT_MOTION_PENALTY,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """
    Penalize excessive foot speed in a contact-aware way.

    Stance feet are expected to move slowly.
    Swing feet are allowed to move faster.

    The penalty is:

        penalty = max(foot_speed - allowed_speed, 0)^2

    where allowed_speed is:
        - max_stance_speed for feet in contact
        - max_swing_speed for feet not in contact

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    foot_speed = robot._selected_body_speed_w(env, asset_cfg)

    force_norm = contact._force_norm_mask(env, sensor_cfg=sensor_cfg)

    in_contact = torch.any(force_norm > motion_cfg.contact_threshold, dim=1)

    runtime._validate_matching_shape(
        in_contact, foot_speed, lhs_name="foot contact mask", rhs_name="foot speed"
    )

    stance_speed_limit = torch.full_like(foot_speed, motion_cfg.max_stance_speed)

    swing_speed_limit = torch.full_like(foot_speed, motion_cfg.max_swing_speed)

    speed_limit = torch.where(in_contact, stance_speed_limit, swing_speed_limit)

    excess_speed = torch.clamp(foot_speed - speed_limit, min=0.0)

    penalty_per_foot = torch.clamp(
        excess_speed.square(), max=motion_cfg.max_penalty_per_foot
    )

    return penalty_per_foot.mean(dim=-1)


def terrain_relative_foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    sensor_names: tuple[str, ...],
    contact_threshold: float = 1.0,
    target_clearance: float = 0.05,
    velocity_scale: float = 2.0,
) -> torch.Tensor:
    """Reward moving feet for reaching terrain-relative swing clearance.

    One downward ray per foot supplies the underlying surface height. The
    clearance score saturates at the target, so the extra lift required by an
    obstacle is never penalized. Scores are averaged over non-contact feet so
    their magnitude is independent of the number of feet in swing. Stationary
    feet, missing terrain hits, and unsupported flight phases earn nothing.

    Returns:
        Floating tensor with shape ``(num_envs,)`` in ``[0, 1]``.
    """

    if not sensor_names:
        raise ValueError("sensor_names must contain at least one foot ray sensor.")
    if not math.isfinite(contact_threshold) or contact_threshold < 0.0:
        raise ValueError("contact_threshold must be finite and non-negative.")
    if not math.isfinite(target_clearance) or target_clearance <= 0.0:
        raise ValueError("target_clearance must be finite and positive.")
    if not math.isfinite(velocity_scale) or velocity_scale <= 0.0:
        raise ValueError("velocity_scale must be finite and positive.")

    surface_heights = []
    valid_hits = []
    for sensor_name in sensor_names:
        sensor = env.scene[sensor_name]
        if not isinstance(sensor, RayCaster):
            raise TypeError(
                f"Expected '{sensor_name}' to be a RayCaster, got {type(sensor).__name__}."
            )
        data = sensor.data
        if data.ray_hits_w.ndim != 3 or data.ray_hits_w.shape[1:] != (1, 3):
            raise RuntimeError(
                f"'{sensor_name}' must contain exactly one downward ray."
            )
        surface_heights.append(data.ray_hits_w[:, 0, 2])
        valid_hits.append(torch.isfinite(data.ray_hits_w[:, 0, :]).all(dim=-1))

    foot_height = robot._selected_body_pos_w(env, asset_cfg)[..., 2]
    surface_height = torch.stack(surface_heights, dim=-1)
    valid_hit = torch.stack(valid_hits, dim=-1)
    foot_velocity = robot._selected_body_lin_vel_w(env, asset_cfg)
    horizontal_speed = torch.linalg.norm(foot_velocity[..., :2], dim=-1)
    runtime._validate_matching_shape(
        foot_height,
        surface_height,
        lhs_name="foot body height",
        rhs_name="foot surface height",
    )
    runtime._validate_matching_shape(
        foot_height,
        horizontal_speed,
        lhs_name="foot body height",
        rhs_name="foot horizontal speed",
    )

    valid = valid_hit & torch.isfinite(foot_height) & torch.isfinite(horizontal_speed)
    clearance = torch.where(
        valid, foot_height - surface_height, torch.zeros_like(foot_height)
    )
    clearance_score = torch.clamp(clearance / float(target_clearance), min=0.0, max=1.0)
    finite_speed = torch.where(
        valid, horizontal_speed, torch.zeros_like(horizontal_speed)
    )
    moving_gate = torch.tanh(float(velocity_scale) * finite_speed)
    current_force_norm = contact._force_norm_mask(env, sensor_cfg=contact_sensor_cfg)[
        :, -1
    ]
    in_contact = current_force_norm > contact_threshold
    runtime._validate_matching_shape(
        foot_height,
        in_contact,
        lhs_name="foot body height",
        rhs_name="foot contact mask",
    )
    has_support = torch.any(in_contact, dim=-1)
    swing_mask = ~in_contact
    swing_score = moving_gate * clearance_score * swing_mask.to(dtype=foot_height.dtype)
    swing_count = swing_mask.sum(dim=-1).clamp_min(1)
    return (swing_score.sum(dim=-1) / swing_count) * has_support.to(
        dtype=foot_height.dtype
    )
