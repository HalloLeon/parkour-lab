# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Active-waypoint task rewards."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .._shared import robot
from ..commands import get_target_speed, get_target_yaw_rate
from ..navigation import geometry, route

_MAX_NORMALIZED_OVERSPEED = 4.0

# Dense waypoint tracking.


def _mask_waypoint_change(
    env: ManagerBasedRLEnv,
    values: torch.Tensor,
) -> torch.Tensor:
    """Discard the sample whose action targeted the preceding waypoint."""

    return torch.where(
        route.active_waypoint_changed_this_step(env),
        torch.zeros_like(values),
        values,
    )


def _obstacle_speed_cap(
    env: ManagerBasedRLEnv,
    *,
    target_speed: torch.Tensor,
    obstacle_mask: torch.Tensor,
    cap_multiplier: float,
    approach_allowance_distance_m: float,
    waypoint_marker_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the phase-local obstacle speed ceiling."""

    waypoint_distance = geometry._active_waypoint_distance_xy(
        env,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )
    reach_radius = route.active_waypoint_root_reach_radii(env).to(
        device=target_speed.device,
        dtype=target_speed.dtype,
    )
    approach_allowance = torch.clamp(
        (reach_radius + approach_allowance_distance_m - waypoint_distance)
        / approach_allowance_distance_m,
        min=0.0,
        max=1.0,
    )
    terminal_landing = route.active_waypoint_is_terminal_landing(env)
    traversal_allowance = torch.where(
        route.active_waypoint_is_rewarded_milestone(env) | terminal_landing,
        torch.ones_like(approach_allowance),
        approach_allowance,
    )
    traversal_allowance = torch.where(
        route.active_waypoint_is_final(env) & ~terminal_landing,
        torch.zeros_like(traversal_allowance),
        traversal_allowance,
    )
    traversal_allowance *= obstacle_mask.to(dtype=traversal_allowance.dtype)
    return target_speed * (1.0 + (cap_multiplier - 1.0) * traversal_allowance)


def waypoint_heading_alignment_exp(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Score heading alignment with signed waypoint-directed motion.

    The exponential heading kernel is multiplied by the signed fraction of
    commanded forward speed. Standing still therefore earns zero, while
    retreating with the same heading cancels forward-only alignment credit.
    Speed regulation remains the responsibility of
    :func:`waypoint_velocity_tracking_exp`.

    The action preceding a waypoint transition targeted the old waypoint, so
    that single retarget sample is masked rather than credited against the new
    heading.

    Returns:
        Tensor with shape ``(num_envs,)``.
    """

    heading_error = geometry._heading_error_to_active_waypoint_xy(
        env,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )
    velocity_along_waypoint = geometry._velocity_along_active_waypoint_xy(
        env,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )
    target_speed = get_target_speed(env).to(
        device=velocity_along_waypoint.device,
        dtype=velocity_along_waypoint.dtype,
    )
    moving = target_speed > 0.0
    normalization_speed = torch.where(
        moving, target_speed, torch.ones_like(target_speed)
    )
    signed_progress = torch.clamp(
        velocity_along_waypoint / normalization_speed,
        min=-1.0,
        max=1.0,
    )
    reward = torch.where(
        moving,
        signed_progress * torch.exp(-torch.abs(heading_error)),
        torch.zeros_like(signed_progress),
    )
    return _mask_waypoint_change(env, reward)


def waypoint_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_speed_cap_multiplier: float = 1.5,
    approach_allowance_distance_m: float = 0.6,
) -> torch.Tensor:
    """Reward signed waypoint progress with bounded speed adaptation.

    Forward reward saturates at the base command on every terrain, while
    reverse motion receives a symmetric penalty. On obstacle rows, a
    phase-local ceiling permits bounded speed adaptation near an approach
    waypoint and during traversal::

        reward = clamp(v_parallel / command, -1, 1)
                 * exp(-||v_perpendicular||^2 / std^2)
                 - relu((v_parallel - phase_ceiling) / command)^2

    The phase ceiling equals the command on flat terrain. This makes standing
    worth zero, penalizes retreat, and preserves the commanded-speed optimum
    without rewarding a zero-net fore-aft oscillation.

    Normalized overspeed is capped only as a numerical guard, leaving the term
    bounded in ``[-16, 1]`` without making ordinary sprinting saturate at the
    same value. The one sample on which the route switches to a waypoint is
    masked because the preceding action targeted the old waypoint.

    Returns:
        Tensor with shape ``(num_envs,)``.
    """

    waypoint_direction_xy = geometry._active_waypoint_direction_xy(
        env,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )
    root_velocity_xy = robot._root_lin_vel_xy(env, asset_cfg=asset_cfg)
    velocity_along_waypoint = torch.sum(
        root_velocity_xy * waypoint_direction_xy,
        dim=-1,
    )
    lateral_velocity_xy = (
        root_velocity_xy - velocity_along_waypoint.unsqueeze(-1) * waypoint_direction_xy
    )

    target_speed = get_target_speed(env).to(
        device=root_velocity_xy.device,
        dtype=root_velocity_xy.dtype,
    )
    moving = target_speed > 0.0
    normalization_speed = torch.where(
        moving, target_speed, torch.ones_like(target_speed)
    )
    lateral_alignment = torch.exp(
        -torch.sum(lateral_velocity_xy.square(), dim=-1) / float(std) ** 2
    )
    obstacle_mask = route.active_difficulty_indices(env) > 0
    speed_cap = _obstacle_speed_cap(
        env,
        target_speed=target_speed,
        obstacle_mask=obstacle_mask,
        cap_multiplier=obstacle_speed_cap_multiplier,
        approach_allowance_distance_m=approach_allowance_distance_m,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )
    forward_fraction = torch.clamp(
        velocity_along_waypoint / normalization_speed,
        min=-1.0,
        max=1.0,
    )
    normalized_overspeed = torch.clamp(
        torch.relu(velocity_along_waypoint - speed_cap) / normalization_speed,
        max=_MAX_NORMALIZED_OVERSPEED,
    )
    tracking = torch.where(
        moving,
        forward_fraction * lateral_alignment - normalized_overspeed.square(),
        torch.zeros_like(forward_fraction),
    )
    return _mask_waypoint_change(
        env,
        tracking,
    )


# Stop and route-envelope shaping.


def route_cross_track_excess_l2(
    env: ManagerBasedRLEnv,
    soft_half_width_m: float,
    hard_half_width_m: float,
) -> torch.Tensor:
    """Penalize only the moving distance outside the soft route envelope."""

    error = route.route_cross_track_error_m(env)
    bounded_error = torch.where(
        torch.isfinite(error),
        error.clamp_max(hard_half_width_m),
        torch.full_like(error, soft_half_width_m),
    )
    excess = torch.relu(bounded_error - soft_half_width_m).square()
    active = (get_target_speed(env) > 0.0) & torch.isfinite(error)
    return torch.where(active, excess, torch.zeros_like(excess))


def off_route_failure(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a timestep-independent impulse for an off-route termination."""

    # Isaac Lab computes terminations before rewards, then integrates by step_dt.
    return env.termination_manager.get_term("off_route").float() / float(env.step_dt)


def stationary_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    planar_speed_std: float = 0.15,
    yaw_rate_std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track stationary or pivot velocity whenever translation is zero."""

    planar_speed_sq = torch.sum(robot._root_lin_vel_xy(env, asset_cfg).square(), dim=-1)
    yaw_rate_error_sq = (
        robot._root_ang_vel_z(env, asset_cfg) - get_target_yaw_rate(env)
    ).square()
    score = torch.exp(-planar_speed_sq / planar_speed_std**2) * torch.exp(
        -yaw_rate_error_sq / yaw_rate_std**2
    )
    nontranslating = get_target_speed(env).eq(0)
    return torch.where(nontranslating, score, torch.zeros_like(score))


# Sparse course events.


def completed_course_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a timestep-independent bonus for safe course completion.

    Returns:
        [num_envs]
    """

    # The success termination updates route state before rewards are evaluated.
    # Isaac Lab then multiplies reward terms by ``env.step_dt``, so divide this
    # explicit one-step event here to retain the exact configured bonus.
    completed = route.course_completed_this_step(env)
    return completed.float() / float(env.step_dt)


def intermediate_milestone_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a safe, one-shot fraction of the course's milestone bonus.

    The termination term advances route cursors before rewards are computed.
    Only waypoints explicitly marked as physical milestones produce this event;
    approach and alignment markers remain unrewarded. Dividing by the number
    of rewarded milestones prevents routes with extra control markers from
    receiving a larger shaping budget.

    A physical milestone can advance only after a recently contacted foot is
    on its explicitly named support polygon. The route transition therefore
    establishes safe landing before this reward consumes the cursor-change
    event; no duplicate reward-specific landing buffer is needed.

    Returns:
        [num_envs]
    """

    # Isaac Lab multiplies reward terms by ``env.step_dt``. Divide this one-step
    # event here so the configured weight remains the exact per-course bonus.
    return route.reached_milestone_reward_fractions(env) / float(env.step_dt)
