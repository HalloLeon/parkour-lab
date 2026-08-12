# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Limb-level joint, contact, and motion rewards."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .. import config
from .._shared import contact, robot, runtime
from ..commands import get_target_speed
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


def touchdown_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "feet_contact", body_names=".*_foot"
    ),
    threshold: float = 0.25,
) -> torch.Tensor:
    """Reward sufficiently long swing periods once, when each foot lands.

    This adapts Unitree RL Lab's Go2 air-time reward to the scalar route-speed
    command used here. Durations are summed over the fixed foot set rather than
    normalized by the number of feet currently in swing. Commands below
    0.1 m/s disable the term.

    Returns:
        Floating tensor with shape ``(num_envs,)``.
    """

    contact._require_body_ids(sensor_cfg, role="air-time reward")
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = ((last_air_time - threshold) * first_contact).sum(dim=-1)
    return reward * (get_target_speed(env) > 0.1)
