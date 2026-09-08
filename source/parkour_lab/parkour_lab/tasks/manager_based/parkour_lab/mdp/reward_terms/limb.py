# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Limb-level joint, contact, and motion rewards."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

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
    terrain_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    max_support_height_variation_m: float = 0.02,
) -> torch.Tensor:
    """Penalize roll and pitch only where a level body is unambiguously useful.

    Include flat approaches and platforms on obstacle courses, not just the
    level-zero curriculum. A height change anywhere in the nearby scan relaxes
    this additional prior during locomotion, leaving the milder global prior
    for steps, gaps and banked ramps. Intentional stops retain the settling prior.
    """

    projected_gravity_xy = robot._root_projected_gravity_xy(env, asset_cfg)
    penalty = torch.sum(projected_gravity_xy.square(), dim=-1)
    return penalty * _stable_gait_mask(
        env, terrain_sensor_cfg, max_support_height_variation_m
    ).to(dtype=penalty.dtype)


def _stable_gait_mask(
    env: ManagerBasedRLEnv,
    terrain_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    max_support_height_variation_m: float = 0.02,
) -> torch.Tensor:
    """Select verified flat support and the existing flat/stop practice phases."""

    if not 0.0 <= max_support_height_variation_m < float("inf"):
        raise ValueError(
            "max_support_height_variation_m must be finite and non-negative."
        )

    # World hit heights do not change when the robot pitches or heaves. Reuse
    # the policy's dense scan; a missing ray cannot certify flat support over a
    # gap. Keep all reductions on the simulation device.
    heights = env.scene[terrain_sensor_cfg.name].data.ray_hits_w[..., 2]
    finite = torch.isfinite(heights)
    safe_heights = torch.where(finite, heights, 0.0)
    height_span = safe_heights.amax(dim=-1) - safe_heights.amin(dim=-1)
    flat_support = finite.all(dim=-1) & (height_span <= max_support_height_variation_m)

    return flat_support | torch.logical_or(
        route.active_difficulty_indices(env) == 0,
        get_target_speed(env) <= 0.0,
    )
