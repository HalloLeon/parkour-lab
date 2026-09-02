# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Limb-level joint, contact, and motion rewards."""

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
    min_force: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """
    Penalize feet hitting near-vertical surfaces.

    A stumble is detected when total contact is strong and lateral force is
    large compared with vertical force. Gating on total force preserves
    near-horizontal impacts whose vertical component is intentionally small.

    Returns:
        [num_envs]
    """

    contact_forces = contact._selected_contact_forces_w_history(
        env, sensor_cfg=sensor_cfg
    )

    lateral_force = torch.linalg.norm(contact_forces[..., :2], dim=-1)
    vertical_force = torch.abs(contact_forces[..., 2])

    strong_contact = torch.linalg.norm(contact_forces, dim=-1) > min_force

    stumble = torch.logical_and(
        strong_contact,
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


def stable_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize roll and pitch only where a level body is unambiguously useful.

    The shared flat curriculum and zero-speed phases provide the ordinary gait
    and settling practice needed by every course.  Banked-ramp traversal stays
    under the milder global orientation prior, so this term cannot suppress a
    necessary obstacle attitude.
    """

    projected_gravity_xy = robot._root_projected_gravity_xy(env, asset_cfg)
    penalty = torch.sum(projected_gravity_xy.square(), dim=-1)
    return penalty * _stable_gait_mask(env).to(dtype=penalty.dtype)


def excessive_foot_air_time_l2(
    env: ManagerBasedRLEnv,
    max_air_time_s: float = 0.35,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """Penalize only the tail of an unnecessarily long foot swing.

    Air time up to ``max_air_time_s`` is free.  The normalized excess is
    squared, capped per foot, and summed over the fixed foot set.  Applying the
    term only on flat terrain or during an intentional stop leaves gap flight,
    climbing, and obstacle clearance unconstrained while making a persistently
    carried leg costly.
    """

    contact._require_body_ids(sensor_cfg, role="excessive foot air-time penalty")
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    current_air_time = sensor.data.current_air_time
    if current_air_time is None:
        raise RuntimeError(
            f"'{sensor_cfg.name}' must enable track_air_time for gait regularization."
        )

    selected_air_time = current_air_time[:, sensor_cfg.body_ids]
    normalized_excess = torch.clamp(
        (selected_air_time - max_air_time_s) / max_air_time_s,
        min=0.0,
        max=1.0,
    )
    penalty = torch.sum(normalized_excess.square(), dim=-1)
    return penalty * _stable_gait_mask(env).to(dtype=penalty.dtype)


def _stable_gait_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Select obstacle-free locomotion and intentional zero-speed phases."""

    return torch.logical_or(
        route.active_difficulty_indices(env) == 0,
        get_target_speed(env) <= 0.0,
    )
