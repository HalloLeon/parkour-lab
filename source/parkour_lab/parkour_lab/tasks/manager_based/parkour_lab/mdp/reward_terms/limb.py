# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Limb-level joint, contact, and motion rewards."""

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .. import config
from .._shared import contact, robot, runtime
from ..curriculums.config import DEFAULT_PARKOUR_CURRICULUM, ParkourCurriculumCfg
from ..navigation import route
from ..terrain import edges


def flat_foot_clearance_exp(
    env: ManagerBasedRLEnv,
    target_height: float = 0.08,
    std: float = 0.04,
    tanh_mult: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
) -> torch.Tensor:
    """Reward moving feet near a target height on the flat bootstrap.

    Horizontal foot speed gates the height kernel, so stationary feet earn
    nothing. Obstacle rows receive zero to leave takeoff and landing mechanics
    unconstrained.

    Returns:
        Floating tensor with shape ``(num_envs,)`` in ``[0, 1]``.
    """

    if not math.isfinite(target_height) or target_height < 0.0:
        raise ValueError("target_height must be finite and non-negative.")
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("std must be finite and positive.")
    if not math.isfinite(tanh_mult) or tanh_mult <= 0.0:
        raise ValueError("tanh_mult must be finite and positive.")

    foot_position = robot._selected_body_pos_env(env, asset_cfg)
    foot_velocity = robot._selected_body_lin_vel_w(env, asset_cfg)
    horizontal_speed = torch.linalg.norm(foot_velocity[..., :2], dim=-1)
    moving_gate = torch.tanh(float(tanh_mult) * horizontal_speed)
    height_kernel = torch.exp(-((foot_position[..., 2] - float(target_height)) / float(std)).square())
    reward = (moving_gate * height_kernel).mean(dim=-1)
    flat_mask = route.active_difficulty_indices(env) == 0
    return reward * flat_mask.to(dtype=reward.dtype)


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

    contact_forces = contact._selected_contact_forces_w_history(env, sensor_cfg=sensor_cfg)

    lateral_force = torch.linalg.norm(contact_forces[..., :2], dim=-1)
    vertical_force = torch.abs(contact_forces[..., 2])

    valid_vertical_contact = vertical_force > stumble_cfg.min_vertical_force

    stumble = torch.logical_and(
        valid_vertical_contact,
        lateral_force > stumble_cfg.lateral_to_vertical_force_ratio * vertical_force,
    )

    return torch.any(stumble, dim=(1, 2)).float()


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
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

    runtime._validate_matching_shape(in_contact, foot_speed, lhs_name="foot contact mask", rhs_name="foot speed")

    stance_speed_limit = torch.full_like(foot_speed, motion_cfg.max_stance_speed)

    swing_speed_limit = torch.full_like(foot_speed, motion_cfg.max_swing_speed)

    speed_limit = torch.where(in_contact, stance_speed_limit, swing_speed_limit)

    excess_speed = torch.clamp(foot_speed - speed_limit, min=0.0)

    penalty_per_foot = torch.clamp(excess_speed.square(), max=motion_cfg.max_penalty_per_foot)

    return penalty_per_foot.mean(dim=-1)
