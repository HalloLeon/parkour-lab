# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-directed task rewards."""

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .. import config
from .._shared import robot, runtime, terrain
from ..commands import get_min_clearance, get_target_speed
from ..navigation import geometry, route


def goal_heading_misalignment_l2(
    env: ManagerBasedRLEnv,
    heading_cfg: config.GoalHeadingCfg = config.DEFAULT_GOAL_HEADING,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize heading misalignment only while the robot is advancing toward
    the XY goal.

    This avoids rewarding the robot for merely staring at the goal while
    standing still.

    The penalty is active only when velocity along the goal direction is
    positive enough.

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    heading_error = geometry._heading_error_to_goal_xy(
        env, goal_cfg=goal_cfg, asset_cfg=asset_cfg
    )

    velocity_along_goal = geometry._velocity_along_goal_xy(
        env, goal_cfg=goal_cfg, asset_cfg=asset_cfg
    )

    advancing_gate = runtime._linear_ramp(
        value=velocity_along_goal,
        lower=heading_cfg.min_forward_speed,
        upper=heading_cfg.full_forward_speed,
    )

    normalized_heading_error = torch.clamp(
        heading_error / heading_cfg.max_heading_error, min=0.0, max=1.0
    )

    return advancing_gate * normalized_heading_error.square()


def goal_progress_xy_stable(
    env: ManagerBasedRLEnv,
    progress_cfg: config.StableGoalProgressCfg = config.DEFAULT_STABLE_GOAL_PROGRESS,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Dense reward for stable reduction of XY distance to the goal.

    progress = previous_distance - current_distance

    Positive progress is counted only when the robot root is stable.
    Negative progress is preserved, so moving away from the goal is still
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

    current_distance = geometry._goal_distance_xy(
        env, goal_cfg=goal_cfg, asset_cfg=asset_cfg
    )

    distance_buffer_name = runtime._private_buffer_name(
        "parkour_prev_goal_distance_xy", goal_cfg.name, asset_cfg.name
    )

    root_xy_buffer_name = runtime._private_buffer_name(
        "parkour_prev_root_xy_for_goal_progress", goal_cfg.name, asset_cfg.name
    )

    just_reset = runtime._episode_start_mask(
        env, reference=current_distance, grace_steps=progress_cfg.reset_grace_steps
    )

    # Switching from a reached waypoint to the next one makes the measured
    # distance jump discontinuously. Suppress that single transition sample so
    # route retargeting is not mistaken for motion away from the goal.
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

    lateral_drift = geometry._lateral_drift_to_goal_xy(
        env, root_delta_xy=root_delta_xy, goal_cfg=goal_cfg, asset_cfg=asset_cfg
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


def completed_course_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a sparse bonus only for safely reaching the final waypoint.

    Returns:
        [num_envs]
    """

    # ManagerBasedRLEnv computes terminations before rewards. The success term
    # advances intermediate waypoints and records this one-step completion event.
    return route.course_completed_this_step(env).float()


def velocity_along_goal_xy_capped(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward normalized world-frame XY velocity toward the goal.

    The reward follows:

        min(dot(root_velocity_xy_w, goal_direction_xy_w), target_speed)
        ----------------------------------------------------------------
                              target_speed

    Projecting world-frame velocity onto the world-frame goal direction avoids
    rewarding a robot that turns around and moves in its body-forward direction
    away from the goal. Capping at the command prevents additional reward for
    overspeed without penalizing short speed bursts needed for parkour. Dividing
    by the command gives every curriculum level the same maximum reward of 1.0.
    Moving away from the goal produces a negative reward.

    This reward does not check whether the robot is upright or has enough
    clearance. Use velocity_along_goal_xy_clearance_capped for the gated version.

    Returns:
        [num_envs]
    """

    velocity_along_goal = geometry._velocity_along_goal_xy(
        env, goal_cfg=goal_cfg, asset_cfg=asset_cfg
    )

    target_speed = get_target_speed(env).to(
        device=velocity_along_goal.device, dtype=velocity_along_goal.dtype
    )
    normalization_speed = target_speed.clamp_min(torch.finfo(target_speed.dtype).eps)

    return torch.minimum(velocity_along_goal, target_speed) / normalization_speed


def velocity_along_goal_xy_clearance_capped(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Clearance-gated version of velocity_along_goal_xy_capped.

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

    reward = velocity_along_goal_xy_capped(env, goal_cfg=goal_cfg, asset_cfg=asset_cfg)

    clearance = terrain._base_clearance(env, asset_cfg=asset_cfg)

    has_enough_clearance = clearance > get_min_clearance(env).to(
        device=clearance.device, dtype=clearance.dtype
    )

    return reward * has_enough_clearance.to(dtype=reward.dtype)


def _root_stability_mask(
    env: ManagerBasedRLEnv,
    stability_cfg: config.RootStabilityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Gate positive progress on attitude, angular speed, and clearance."""

    asset: Articulation = env.scene[asset_cfg.name]
    roll_pitch_speed = torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=-1)
    tilt = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1)
    clearance = terrain._base_clearance(env, asset_cfg)
    min_clearance = get_min_clearance(env).to(
        device=clearance.device,
        dtype=clearance.dtype,
    )
    return (
        (roll_pitch_speed < stability_cfg.max_roll_pitch_ang_speed)
        & (tilt < stability_cfg.max_projected_gravity_xy_norm)
        & (clearance > min_clearance)
    )
