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


def upright_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared chord distance from upright: zero upright, four upside down.

    ``2 * (1 + gravity_z)`` has the same small-angle curvature as the usual
    gravity-XY square, without treating an inverted base as level.
    """
    gravity_z = env.scene[asset_cfg.name].data.projected_gravity_b[:, 2]
    return 2.0 * (1.0 + gravity_z.clamp(-1.0, 1.0))


def _level_support_mask(
    env: ManagerBasedRLEnv,
    terrain_sensor_cfg: SceneEntityCfg,
    feet_sensor_cfg: SceneEntityCfg,
    support_radius_m: float,
    max_support_height_variation_m: float,
    support_force_threshold_n: float,
    min_support_feet: int,
) -> torch.Tensor:
    """Recognize loaded, level terrain under the base, not future obstacles.

    Reuse the existing yaw-aligned downward scan. Select rays by their local
    *origins*, including the scanner's forward offset, so a missed selected
    ray cannot silently disappear from the flatness test. No level/command
    shortcut can certify a slope or flight as steady support.
    """
    scanner = env.scene[terrain_sensor_cfg.name]
    heights = scanner.data.ray_hits_w[..., 2]
    selected = scanner.ray_starts[..., :2].square().sum(dim=-1) <= support_radius_m**2
    valid = torch.isfinite(heights)
    high = torch.where(selected & valid, heights, -torch.inf).amax(dim=-1)
    low = torch.where(selected & valid, heights, torch.inf).amin(dim=-1)
    level = (
        (selected.sum(dim=-1) >= 3)
        & (~selected | valid).all(dim=-1)
        & (high - low <= max_support_height_variation_m)
    )
    # Current world-Z load, not a recent impact or a contact-history latch.
    forces = contact._selected_contact_forces_w(env, feet_sensor_cfg)
    loaded = (forces[..., 2] > support_force_threshold_n).sum(dim=-1)
    return level & (loaded >= min_support_feet)


def supported_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "feet_contact", body_names=".*_foot"
    ),
    support_radius_m: float = 0.35,
    max_support_height_variation_m: float = 0.02,
    support_force_threshold_n: float = 5.0,
    min_support_feet: int = 2,
) -> torch.Tensor:
    """Additional upright prior only on loaded, locally level support."""
    supported = _level_support_mask(
        env,
        terrain_sensor_cfg,
        feet_sensor_cfg,
        support_radius_m,
        max_support_height_variation_m,
        support_force_threshold_n,
        min_support_feet,
    )
    return torch.where(supported, upright_orientation_l2(env, asset_cfg), 0.0)


def supported_vertical_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg(
        "feet_contact", body_names=".*_foot"
    ),
    support_radius_m: float = 0.35,
    max_support_height_variation_m: float = 0.02,
    support_force_threshold_n: float = 5.0,
    min_support_feet: int = 2,
) -> torch.Tensor:
    """Regularize world-Z heave on level support; relax for flight/climbing.

    Unlike body-Z velocity, this does not count forward travel projected onto
    a pitched body axis as vertical motion. It is a separate opt-in replacement
    for the unconditional body-frame term, not an additional cost.
    """
    supported = _level_support_mask(
        env,
        terrain_sensor_cfg,
        feet_sensor_cfg,
        support_radius_m,
        max_support_height_variation_m,
        support_force_threshold_n,
        min_support_feet,
    )
    return torch.where(supported, robot._root_lin_vel_z(env, asset_cfg).square(), 0.0)
