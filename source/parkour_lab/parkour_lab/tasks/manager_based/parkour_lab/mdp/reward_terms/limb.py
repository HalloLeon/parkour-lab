# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Limb-level joint, contact, and motion rewards."""

import math

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .._shared import contact, robot
from ..commands import get_target_speed
from ..curriculums.config import DEFAULT_PARKOUR_CURRICULUM, ParkourCurriculumCfg
from ..navigation import route
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
    lateral_to_vertical_force_ratio: float = 1.0,
    min_vertical_force: float = 0.5,
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

    valid_vertical_contact = vertical_force > min_vertical_force

    stumble = torch.logical_and(
        valid_vertical_contact,
        lateral_force > lateral_to_vertical_force_ratio * vertical_force,
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


def touchdown_air_time_bonus(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
    threshold: float = 0.25,
    max_air_time: float = 0.5,
) -> torch.Tensor:
    """Reward the bounded air-time surplus of each completed swing.

    Only the interval above ``threshold`` earns credit, and air time beyond
    ``max_air_time`` provides no further benefit. Durations are summed over
    the fixed foot set rather than normalized by the number of feet currently
    in swing. Commands below 0.1 m/s disable the term.

    Returns:
        Floating tensor with shape ``(num_envs,)``.
    """

    _validate_air_time_threshold(threshold)
    if not math.isfinite(max_air_time) or max_air_time <= threshold:
        raise ValueError("max_air_time must be finite and greater than threshold.")

    completed_touchdown, last_air_time = _completed_touchdown_air_time(env, sensor_cfg)
    credited_air_time = torch.clamp(
        last_air_time - threshold,
        min=0.0,
        max=max_air_time - threshold,
    )
    reward = (credited_air_time * completed_touchdown).sum(dim=-1)
    return reward * _moving_command_mask(env)


def touchdown_short_air_time_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
    threshold: float = 0.25,
) -> torch.Tensor:
    """Penalize the air-time shortfall of each completed swing.

    The returned value is a non-negative penalty magnitude intended for use
    with a negative reward weight. Keeping this term separate from
    :func:`touchdown_air_time_bonus` allows short-swing discouragement to be
    tuned without strengthening the incentive for longer swings.

    Returns:
        Floating tensor with shape ``(num_envs,)``.
    """

    _validate_air_time_threshold(threshold)
    completed_touchdown, last_air_time = _completed_touchdown_air_time(env, sensor_cfg)
    shortfall = torch.relu(threshold - last_air_time)
    penalty = (shortfall * completed_touchdown).sum(dim=-1)
    return penalty * _moving_command_mask(env)


def air_contact_time_variance(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
    max_time: float = 0.5,
    flat_only: bool = True,
) -> torch.Tensor:
    """Penalize persistent differences in completed foot timing.

    Completed air and contact intervals are clipped before their variances are
    summed across feet. This discourages one foot from consistently dominating
    swing or stance duration without prescribing a gait phase. By default the
    term applies only on the shared flat-bootstrap curriculum row, leaving
    obstacle traversal free to use deliberately asymmetric contacts.

    Returns:
        Floating tensor with shape ``(num_envs,)``.
    """

    if not math.isfinite(max_time) or max_time <= 0.0:
        raise ValueError("max_time must be positive and finite.")

    contact._require_body_ids(sensor_cfg, role="air/contact-time variance")
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    penalty = torch.var(
        torch.clamp(last_air_time, min=0.0, max=max_time),
        dim=-1,
        correction=0,
    ) + torch.var(
        torch.clamp(last_contact_time, min=0.0, max=max_time),
        dim=-1,
        correction=0,
    )

    active = _moving_command_mask(env)
    if flat_only:
        active = torch.logical_and(active, route.active_difficulty_indices(env) == 0)
    return penalty * active


def _completed_touchdown_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected first-contact flags and their completed air times."""

    contact._require_body_ids(sensor_cfg, role="touchdown air-time shaping")
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # Initial stance can be reported as first contact before any swing has
    # completed. Exclude that reset sample so it cannot incur a shortfall.
    completed_touchdown = torch.logical_and(first_contact, last_air_time > 0.0)
    return completed_touchdown, last_air_time


def _moving_command_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return environments whose scalar route-speed command requests motion."""

    return get_target_speed(env) > 0.1


def _validate_air_time_threshold(threshold: float) -> None:
    """Validate the shared minimum completed-swing duration."""

    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative.")
