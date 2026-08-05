# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Active-waypoint task rewards."""

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .. import config
from .._shared import robot, runtime
from ..commands import get_min_clearance, get_target_speed
from ..navigation import geometry, route
from ..terrain import queries

# Dense waypoint-progress shaping.


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


def waypoint_heading_alignment_exp(
    env: ManagerBasedRLEnv,
    overspeed_std: float = 0.3,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward alignment while advancing at a suitable course speed.

    The exponential heading kernel is multiplied by the fraction of commanded
    forward speed, so alignment alone cannot earn return while standing still.
    On the flat bootstrap, a one-sided exponential gate also suppresses
    overspeed. Obstacle rows retain heading guidance during takeoff.

    The action preceding a waypoint transition targeted the old waypoint, so
    that single retarget sample is masked rather than credited against the new
    heading.

    Returns:
        Tensor with shape ``(num_envs,)``.
    """

    if overspeed_std <= 0.0:
        raise ValueError("overspeed_std must be positive.")

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
    advancing_gate = torch.clamp(
        velocity_along_waypoint / target_speed.clamp_min(torch.finfo(target_speed.dtype).eps),
        min=0.0,
        max=1.0,
    )
    flat_speed_gate = torch.exp(
        -torch.relu(velocity_along_waypoint - target_speed).square() / float(overspeed_std) ** 2
    )
    speed_gate = torch.where(
        route.active_difficulty_indices(env) == 0,
        flat_speed_gate,
        torch.ones_like(flat_speed_gate),
    )
    reward = advancing_gate * speed_gate * torch.exp(-torch.abs(heading_error))
    return _mask_waypoint_change(env, reward)


def waypoint_velocity_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.5,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    flat_speed_std: float = 0.25,
) -> torch.Tensor:
    """Track flat target speed while preserving obstacle-speed acquisition.

    Flat rows use symmetric exponential tracking so the commanded speed is the
    unique optimum. Obstacle rows retain the capped acquisition signal that
    permits faster takeoff motion::

        where(flat, exp(-(v_parallel - target_speed)^2 / flat_speed_std^2),
                    clamp(v_parallel / target_speed, 0, 1))
        * exp(-||v_perpendicular||^2 / std^2)

    The term is bounded in ``[0, 1]`` and masks the one sample on which the route
    switches to a waypoint that did not produce the current action.

    Returns:
        Tensor with shape ``(num_envs,)``.
    """

    if std <= 0.0 or flat_speed_std <= 0.0:
        raise ValueError("std and flat_speed_std must be positive.")

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
    lateral_velocity_xy = root_velocity_xy - velocity_along_waypoint.unsqueeze(-1) * waypoint_direction_xy

    target_speed = get_target_speed(env).to(
        device=root_velocity_xy.device,
        dtype=root_velocity_xy.dtype,
    )
    forward_fraction = torch.clamp(
        velocity_along_waypoint / target_speed.clamp_min(torch.finfo(target_speed.dtype).eps),
        min=0.0,
        max=1.0,
    )
    flat_tracking = torch.exp(-(velocity_along_waypoint - target_speed).square() / float(flat_speed_std) ** 2)
    forward_reward = torch.where(
        route.active_difficulty_indices(env) == 0,
        flat_tracking,
        forward_fraction,
    )
    lateral_alignment = torch.exp(-torch.sum(lateral_velocity_xy.square(), dim=-1) / float(std) ** 2)
    return _mask_waypoint_change(
        env,
        forward_reward * lateral_alignment,
    )


def flat_waypoint_overspeed_l2(
    env: ManagerBasedRLEnv,
    max_excess_ratio: float = 2.0,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize normalized overspeed only on the flat bootstrap.

    The quadratic remains corrective throughout the expected speed range and
    is capped before squaring to bound collision outliers. Configure this term
    with a negative weight.
    """

    if max_excess_ratio <= 0.0:
        raise ValueError("max_excess_ratio must be positive.")

    velocity_along_waypoint = geometry._velocity_along_active_waypoint_xy(
        env,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )
    target_speed = get_target_speed(env).to(
        device=velocity_along_waypoint.device,
        dtype=velocity_along_waypoint.dtype,
    )
    excess_ratio = torch.relu(velocity_along_waypoint - target_speed) / target_speed.clamp_min(
        torch.finfo(target_speed.dtype).eps
    )
    penalty = torch.clamp(excess_ratio, max=max_excess_ratio).square()
    penalty *= (route.active_difficulty_indices(env) == 0).to(dtype=penalty.dtype)
    return _mask_waypoint_change(env, penalty)


def waypoint_heading_misalignment_l2(
    env: ManagerBasedRLEnv,
    heading_cfg: config.WaypointHeadingCfg = config.DEFAULT_WAYPOINT_HEADING,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize heading misalignment only while the robot is advancing toward
    the active XY waypoint.

    This avoids rewarding the robot for merely staring at the waypoint while
    standing still.

    The penalty is active only when velocity along the waypoint direction is
    positive enough.

    Use with a negative reward weight.

    Returns:
        [num_envs]
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

    advancing_gate = runtime._linear_ramp(
        value=velocity_along_waypoint,
        lower=heading_cfg.min_forward_speed,
        upper=heading_cfg.full_forward_speed,
    )

    normalized_heading_error = torch.clamp(heading_error / heading_cfg.max_heading_error, min=0.0, max=1.0)

    penalty = advancing_gate * normalized_heading_error.square()
    # A waypoint switch changes the desired heading after the robot has already
    # acted toward the previous target. Skip that transition sample so route
    # retargeting is not penalized as policy-induced heading misalignment.
    return _mask_waypoint_change(env, penalty)


def waypoint_progress_xy_stable(
    env: ManagerBasedRLEnv,
    progress_cfg: config.StableWaypointProgressCfg = (config.DEFAULT_STABLE_WAYPOINT_PROGRESS),
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Dense reward for stable reduction of XY distance to the active waypoint.

    progress = previous_distance - current_distance

    Positive progress is counted only when the robot root is stable.
    Negative progress is preserved, so moving away from the waypoint is still
    penalized even if the robot is unstable.

    Stability includes:
      - limited roll/pitch angular velocity,
      - limited roll/pitch tilt,
      - sufficient base/root clearance above the support surface underneath it.

    The support surface may be flat ground, an obstacle top, or later another
    terrain/platform surface.

    Returns:
        [num_envs]
    """

    current_distance = geometry._active_waypoint_distance_xy(
        env,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )

    root_xy_buffer_name = runtime._private_buffer_name(
        "parkour_prev_root_xy_for_waypoint_progress",
        waypoint_marker_cfg.name,
        asset_cfg.name,
    )

    just_reset = runtime._episode_start_mask(
        env, reference=current_distance, grace_steps=progress_cfg.reset_grace_steps
    )

    # Switching from a reached waypoint to the next one makes the measured
    # distance jump discontinuously. Suppress that single transition sample so
    # route retargeting is not mistaken for motion away from the waypoint.
    distance_reference_changed = torch.logical_or(
        just_reset,
        route.active_waypoint_changed_this_step(env),
    )

    root_delta_xy = robot._root_xy_delta_from_previous(
        env, buffer_name=root_xy_buffer_name, reset_mask=just_reset, asset_cfg=asset_cfg
    )
    current_root_xy = robot._root_pos_env(env, asset_cfg)[:, :2]
    previous_root_xy = current_root_xy - root_delta_xy
    active_waypoint_xy = route.active_waypoint_positions(
        env,
        waypoint_marker_cfg,
    )[:, :2]
    previous_distance = torch.linalg.norm(
        previous_root_xy - active_waypoint_xy,
        dim=-1,
    )
    progress = previous_distance - current_distance
    progress = torch.where(
        distance_reference_changed,
        torch.zeros_like(progress),
        progress,
    )

    stable = _root_stability_mask(env, stability_cfg=progress_cfg.stability, asset_cfg=asset_cfg)

    progress = runtime._gate_positive_values(values=progress, gate=stable)

    positive_progress = torch.clamp(progress, min=0.0)
    negative_progress = torch.clamp(-progress, min=0.0)

    positive_reward = torch.clamp(
        positive_progress / progress_cfg.progress_scale,
        max=progress_cfg.max_positive_reward,
    )

    negative_penalty = torch.clamp(
        negative_progress / progress_cfg.progress_scale,
        max=progress_cfg.max_negative_penalty,
    )

    lateral_drift = geometry._lateral_drift_to_active_waypoint_xy(
        env,
        root_delta_xy=root_delta_xy,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )

    lateral_penalty = torch.clamp(
        lateral_drift / progress_cfg.progress_scale,
        max=progress_cfg.max_lateral_penalty,
    )

    # Only penalize lateral drift while stable and making positive progress.
    # This avoids over-penalizing reset artifacts, falls, and recovery behavior.
    lateral_penalty = torch.where(
        stable & (positive_progress > 0.0),
        lateral_penalty,
        torch.zeros_like(lateral_penalty),
    )

    return positive_reward - negative_penalty - progress_cfg.lateral_drift_weight * lateral_penalty


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


# Dense waypoint-velocity shaping.


def velocity_along_waypoint_xy_capped(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward normalized world-frame XY velocity toward the active waypoint.

    The reward follows:

        clamp(dot(root_velocity_xy_w, waypoint_direction_xy_w) / target_speed,
              -1, 1)

    Projecting world-frame velocity onto the waypoint direction avoids
    rewarding a robot that turns around and moves in its body-forward direction
    away from the waypoint. The symmetric clamp prevents overspeed from earning
    extra reward and bounds collision-induced reverse spikes. Dividing by the
    command gives every curriculum level the same ``[-1, 1]`` scale. The
    retarget step is zeroed because its new direction did not produce the
    action being evaluated.

    This reward does not check whether the robot is upright or has enough
    clearance. Use velocity_along_waypoint_xy_clearance_capped for the gated
    version.

    Returns:
        [num_envs]
    """

    velocity_along_waypoint = geometry._velocity_along_active_waypoint_xy(
        env,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )

    target_speed = get_target_speed(env).to(
        device=velocity_along_waypoint.device,
        dtype=velocity_along_waypoint.dtype,
    )
    normalization_speed = target_speed.clamp_min(torch.finfo(target_speed.dtype).eps)

    normalized_velocity = torch.clamp(
        velocity_along_waypoint / normalization_speed,
        min=-1.0,
        max=1.0,
    )
    return _mask_waypoint_change(env, normalized_velocity)


def velocity_along_waypoint_xy_clearance_capped(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Clearance-gated version of velocity_along_waypoint_xy_capped.

    The velocity reward is only paid when the robot base/root has enough
    clearance above the surface underneath it.

    The surface underneath it may be:
        - flat ground
        - obstacle top
        - later, another terrain/support surface

    This prevents rewarding forward velocity while the robot is collapsed,
    scraping, or too close to the support surface.

    Returns:
        [num_envs]
    """

    reward = velocity_along_waypoint_xy_capped(
        env,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )

    clearance = queries._base_clearance(env, asset_cfg=asset_cfg)

    has_enough_clearance = clearance > get_min_clearance(env).to(device=clearance.device, dtype=clearance.dtype)

    return reward * has_enough_clearance.to(dtype=reward.dtype)


# Private helpers.


def _root_stability_mask(
    env: ManagerBasedRLEnv,
    stability_cfg: config.RootStabilityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Gate positive progress on attitude, angular speed, and clearance."""

    asset: Articulation = env.scene[asset_cfg.name]
    roll_pitch_speed = torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=-1)
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
    clearance = queries._base_clearance(env, asset_cfg)
    min_clearance = get_min_clearance(env).to(
        device=clearance.device,
        dtype=clearance.dtype,
    )
    return (
        (roll_pitch_speed < stability_cfg.max_roll_pitch_ang_speed)
        & (tilt < stability_cfg.max_projected_gravity_xy_norm)
        & (clearance > min_clearance)
    )
