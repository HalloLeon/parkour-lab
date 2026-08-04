# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Limb-level joint, contact, and motion regularizers."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .. import config
from .._shared import contact, robot, runtime
from ..commands import get_target_speed
from ..curriculums.config import DEFAULT_PARKOUR_CURRICULUM, ParkourCurriculumCfg
from ..navigation import route
from ..terrain import edges


def feet_air_time(
    env: ManagerBasedRLEnv,
    threshold: float = 0.5,
    flat_weight: float = 0.25,
    obstacle_weight: float = 0.01,
    curriculum_cfg: ParkourCurriculumCfg = DEFAULT_PARKOUR_CURRICULUM,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """Reward foot swing duration with curriculum-level-aware strength.

    The raw event matches Isaac Lab's quadruped locomotion term: air time above
    ``threshold`` is credited on first contact, and the term is disabled for a
    near-zero speed command.  Row zero uses the flat-gait acquisition weight;
    obstacle rows retain only the much smaller rough-terrain weight.

    The configured reward-manager weight should be ``1.0`` because
    ``flat_weight`` and ``obstacle_weight`` are applied here per environment.

    Returns:
        Tensor with shape ``(num_envs,)``.
    """

    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    if flat_weight < 0.0 or obstacle_weight < 0.0:
        raise ValueError("feet-air-time weights must be non-negative.")

    contact._require_body_ids(sensor_cfg, role="feet air time")
    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum(
        (last_air_time - threshold) * first_contact.to(dtype=last_air_time.dtype),
        dim=-1,
    )
    reward *= (get_target_speed(env) > 0.1).to(
        device=reward.device,
        dtype=reward.dtype,
    )
    return reward * _difficulty_level_weight(
        env,
        reference=reward,
        flat_value=flat_weight,
        obstacle_value=obstacle_weight,
        curriculum_cfg=curriculum_cfg,
    )


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


def obstacle_only_feet_edge(
    env: ManagerBasedRLEnv,
    curriculum_cfg: ParkourCurriculumCfg = DEFAULT_PARKOUR_CURRICULUM,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """Return the edge-contact penalty only on obstacle-bearing rows."""

    penalty = feet_edge(
        env,
        curriculum_cfg=curriculum_cfg,
        asset_cfg=asset_cfg,
        sensor_cfg=sensor_cfg,
    )
    return penalty * _difficulty_level_weight(
        env,
        reference=penalty,
        flat_value=0.0,
        obstacle_value=1.0,
        curriculum_cfg=curriculum_cfg,
    )


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


def obstacle_only_feet_stumble(
    env: ManagerBasedRLEnv,
    stumble_cfg: config.FeetStumbleCfg = config.DEFAULT_FEET_STUMBLE,
    curriculum_cfg: ParkourCurriculumCfg = DEFAULT_PARKOUR_CURRICULUM,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """Return the stumble penalty only on obstacle-bearing rows."""

    penalty = feet_stumble(
        env,
        stumble_cfg=stumble_cfg,
        sensor_cfg=sensor_cfg,
    )
    return penalty * _difficulty_level_weight(
        env,
        reference=penalty,
        flat_value=0.0,
        obstacle_value=1.0,
        curriculum_cfg=curriculum_cfg,
    )


def obstacle_only_undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    curriculum_cfg: ParkourCurriculumCfg = DEFAULT_PARKOUR_CURRICULUM,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("leg_contact"),
) -> torch.Tensor:
    """Count undesired contacts only after leaving the flat bootstrap row."""

    force_norm = contact._force_norm_mask(env, sensor_cfg=sensor_cfg)
    penalty = torch.any(force_norm > threshold, dim=1).sum(
        dim=-1,
        dtype=torch.float32,
    )
    return penalty * _difficulty_level_weight(
        env,
        reference=penalty,
        flat_value=0.0,
        obstacle_value=1.0,
        curriculum_cfg=curriculum_cfg,
    )


def level_scaled_feet_slide(
    env: ManagerBasedRLEnv,
    flat_scale: float = 0.5,
    obstacle_scale: float = 1.0,
    contact_threshold: float = 1.0,
    curriculum_cfg: ParkourCurriculumCfg = DEFAULT_PARKOUR_CURRICULUM,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """Penalize foot sliding at half strength on the flat bootstrap by default."""

    if flat_scale < 0.0 or obstacle_scale < 0.0:
        raise ValueError("feet-slide scales must be non-negative.")

    force_norm = contact._force_norm_mask(env, sensor_cfg=sensor_cfg)
    in_contact = torch.any(force_norm > contact_threshold, dim=1)
    foot_velocity_xy = robot._selected_body_lin_vel_w(env, asset_cfg)[:, :, :2]
    foot_speed_xy = torch.linalg.norm(foot_velocity_xy, dim=-1)
    runtime._validate_matching_shape(
        in_contact,
        foot_speed_xy,
        lhs_name="foot contact mask",
        rhs_name="foot planar speed",
    )
    penalty = torch.sum(
        foot_speed_xy * in_contact.to(dtype=foot_speed_xy.dtype),
        dim=-1,
    )
    return penalty * _difficulty_level_weight(
        env,
        reference=penalty,
        flat_value=flat_scale,
        obstacle_value=obstacle_scale,
        curriculum_cfg=curriculum_cfg,
    )


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Penalize selected joints deviating from their default pose.

    Returns:
        [num_envs]
    """

    joint_error = robot._selected_joint_pos_error(env, asset_cfg)

    return torch.sum(joint_error.square(), dim=-1)


def no_feet_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """
    Penalize having no feet in contact with the ground.

    This discourages hopping/skipping in flat walking.

    Returns:
        [num_envs]
    """

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # [num_envs, history_length, num_bodies, 3]
    net_forces = contact_sensor.data.net_forces_w_history

    if sensor_cfg.body_ids is not None:
        net_forces = net_forces[:, :, sensor_cfg.body_ids, :]

    # [num_envs, history_length, num_bodies]
    force_norm = torch.linalg.norm(net_forces, dim=-1)

    # Has each foot contacted recently?
    # [num_envs, num_bodies]
    feet_in_contact = torch.any(force_norm > threshold, dim=1)

    # [num_envs]
    num_feet_in_contact = torch.sum(feet_in_contact.float(), dim=-1)

    no_contact = num_feet_in_contact < 1.0

    return no_contact.float()


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


def _difficulty_level_weight(
    env: ManagerBasedRLEnv,
    *,
    reference: torch.Tensor,
    flat_value: float,
    obstacle_value: float,
    curriculum_cfg: ParkourCurriculumCfg,
) -> torch.Tensor:
    """Return one flat/obstacle scalar for each retained episode course."""

    course_indices = route.active_course_indices(env)
    difficulty_indices = torch.remainder(
        course_indices,
        curriculum_cfg.num_difficulties,
    )
    flat = torch.full_like(reference, flat_value)
    obstacle = torch.full_like(reference, obstacle_value)
    return torch.where(difficulty_indices == 0, flat, obstacle)
