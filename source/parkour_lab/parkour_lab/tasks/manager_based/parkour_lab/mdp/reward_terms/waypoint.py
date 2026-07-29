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

    normalized_heading_error = torch.clamp(
        heading_error / heading_cfg.max_heading_error, min=0.0, max=1.0
    )

    return advancing_gate * normalized_heading_error.square()


def waypoint_progress_xy_stable(
    env: ManagerBasedRLEnv,
    progress_cfg: config.StableWaypointProgressCfg = (
        config.DEFAULT_STABLE_WAYPOINT_PROGRESS
    ),
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

    distance_buffer_name = runtime._private_buffer_name(
        "parkour_prev_active_waypoint_distance_xy",
        waypoint_marker_cfg.name,
        asset_cfg.name,
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

    progress = runtime._difference_from_previous_env_buffer(
        env,
        buffer_name=distance_buffer_name,
        current_value=current_distance,
        reset_mask=distance_reference_changed,
    )

    root_delta_xy = robot._root_xy_delta_from_previous(
        env, buffer_name=root_xy_buffer_name, reset_mask=just_reset, asset_cfg=asset_cfg
    )

    stable = _root_stability_mask(
        env, stability_cfg=progress_cfg.stability, asset_cfg=asset_cfg
    )

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

    return (
        positive_reward
        - negative_penalty
        - progress_cfg.lateral_drift_weight * lateral_penalty
    )


# Sparse course events.


def completed_course_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a timestep-independent bonus for safe course completion.

    Returns:
        [num_envs]
    """

    # ManagerBasedRLEnv computes and stores terminations before rewards. Isaac
    # Lab then multiplies reward terms by ``env.step_dt``, so divide this
    # one-step event here to retain the exact configured completion bonus.
    completed = env.termination_manager.get_term("success")
    return completed.float() / float(env.step_dt)


def intermediate_milestone_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a safe, one-shot fraction of the course's milestone bonus.

    The termination term advances route cursors before rewards are computed.
    Only waypoints explicitly marked as physical milestones produce this event;
    approach and alignment markers remain unrewarded. Dividing by the number
    of rewarded milestones prevents routes with extra control markers from
    receiving a larger shaping budget.

    Intermediate route advancement uses proximity or a corridor-constrained
    route-plane crossing. The reward additionally requires safe base clearance
    so a collapsed robot cannot earn shaping merely by triggering the route
    transition.

    Returns:
        [num_envs]
    """

    required_state = (
        "_parkour_active_waypoint_index",
        "_parkour_course_index",
        "_parkour_milestone_count_by_course",
        "_parkour_waypoint_milestone_table",
    )
    if not all(hasattr(env, name) for name in required_state):
        raise RuntimeError(
            "Active routes must be initialized before milestone rewards."
        )

    changed = route.active_waypoint_changed_this_step(env)
    course_indices = env._parkour_course_index
    reached_indices = (env._parkour_active_waypoint_index - 1).clamp_min(0)
    reached = (
        changed
        & env._parkour_waypoint_milestone_table[
            course_indices,
            reached_indices,
        ]
    )
    milestone_counts = env._parkour_milestone_count_by_course[course_indices].clamp_min(
        1
    )

    clearance = queries._base_clearance(env, asset_cfg)
    min_clearance = get_min_clearance(env).to(
        device=clearance.device,
        dtype=clearance.dtype,
    )
    safely_reached = reached & (clearance > min_clearance)
    # Isaac Lab multiplies reward terms by ``env.step_dt``. Divide this one-step
    # event here so the configured weight remains the exact per-course bonus.
    reward_rate = safely_reached.float() / float(env.step_dt)
    return reward_rate / milestone_counts.to(dtype=reward_rate.dtype)


# Dense waypoint-velocity shaping.


def velocity_along_waypoint_xy_capped(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward normalized world-frame XY velocity toward the active waypoint.

    The reward follows:

        min(dot(root_velocity_xy_w, waypoint_direction_xy_w), target_speed)
        ----------------------------------------------------------------
                              target_speed

    Projecting world-frame velocity onto the waypoint direction avoids
    rewarding a robot that turns around and moves in its body-forward direction
    away from the waypoint. Capping at the command prevents additional reward for
    overspeed without penalizing short speed bursts needed for parkour. Dividing
    by the command gives every curriculum level the same maximum reward of 1.0.
    Moving away from the waypoint produces a negative reward.

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

    return torch.minimum(velocity_along_waypoint, target_speed) / normalization_speed


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

    has_enough_clearance = clearance > get_min_clearance(env).to(
        device=clearance.device, dtype=clearance.dtype
    )

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
