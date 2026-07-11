# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Goal-directed task rewards."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .. import config
from .._shared import navigation, runtime, stability, state, terrain
from ..commands import get_min_clearance, get_target_speed


def velocity_along_goal_xy_exp(
    env: ManagerBasedRLEnv,
    tracking_cfg: config.GoalVelocityCfg = config.DEFAULT_GOAL_VELOCITY,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward tracking a desired XY velocity along the direction to the goal.

    Far from the goal:
        desired velocity is the current per-environment target-speed command.

    Near the goal:
        desired velocity decreases toward zero to reduce overshooting.

    The exponential tracking kernel reduces the reward for both underspeed
    and overspeed. It shapes the policy toward the commanded speed; it does
    not clamp the robot's physical velocity.

    This reward does not check whether the robot is upright or has enough
    clearance. Use velocity_along_goal_xy_clearance_exp for the gated version.

    Returns:
        [num_envs]
    """

    velocity_along_goal = navigation._velocity_along_goal_xy(env, goal_cfg=goal_cfg, asset_cfg=asset_cfg)

    goal_dist_xy = navigation._goal_distance_xy(env, goal_cfg, asset_cfg)

    slowdown_scale = torch.clamp(goal_dist_xy / tracking_cfg.slow_down_distance, min=0.0, max=1.0)

    target_speed = get_target_speed(env).to(device=goal_dist_xy.device, dtype=goal_dist_xy.dtype)

    desired_velocity = target_speed * slowdown_scale

    # Symmetric command tracking is intentional: overspeed must lose reward,
    # especially while the desired velocity is reduced near the goal.
    velocity_error = velocity_along_goal - desired_velocity

    return torch.exp(-velocity_error.square() / tracking_cfg.speed_tracking_scale**2)


def velocity_along_goal_xy_clearance_exp(
    env: ManagerBasedRLEnv,
    tracking_cfg: config.GoalVelocityCfg = config.DEFAULT_GOAL_VELOCITY,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Clearance-gated version of velocity_along_goal_xy_exp.

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

    reward = velocity_along_goal_xy_exp(env, tracking_cfg=tracking_cfg, goal_cfg=goal_cfg, asset_cfg=asset_cfg)

    clearance = terrain._base_clearance(env, asset_cfg=asset_cfg)

    has_enough_clearance = clearance > get_min_clearance(env).to(device=clearance.device, dtype=clearance.dtype)

    return reward * has_enough_clearance.to(dtype=reward.dtype)


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

    current_distance = navigation._goal_distance_xy(env, goal_cfg=goal_cfg, asset_cfg=asset_cfg)

    distance_buffer_name = runtime._private_buffer_name("parkour_prev_goal_distance_xy", goal_cfg.name, asset_cfg.name)

    root_xy_buffer_name = runtime._private_buffer_name(
        "parkour_prev_root_xy_for_goal_progress", goal_cfg.name, asset_cfg.name
    )

    just_reset = runtime._episode_start_mask(
        env, reference=current_distance, grace_steps=progress_cfg.reset_grace_steps
    )

    progress = runtime._difference_from_previous_env_buffer(
        env, buffer_name=distance_buffer_name, current_value=current_distance, reset_mask=just_reset
    )

    root_delta_xy = state._root_xy_delta_from_previous(
        env, buffer_name=root_xy_buffer_name, reset_mask=just_reset, asset_cfg=asset_cfg
    )

    stable = stability._root_stability_mask(env, stability_cfg=progress_cfg.stability, asset_cfg=asset_cfg)

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

    lateral_drift = navigation._lateral_drift_to_goal_xy(
        env, root_delta_xy=root_delta_xy, goal_cfg=goal_cfg, asset_cfg=asset_cfg
    )

    lateral_penalty = torch.clamp(lateral_drift / progress_cfg.progress_scale, max=progress_cfg.max_lateral_penalty)

    # Only penalize lateral drift while stable and making positive progress.
    # This avoids over-penalizing reset artifacts, falls, and recovery behavior.
    lateral_penalty = torch.where(
        stable & (positive_progress > 0.0), lateral_penalty, torch.zeros_like(lateral_penalty)
    )

    return positive_reward - negative_penalty - progress_cfg.lateral_drift_weight * lateral_penalty


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

    heading_error = navigation._heading_error_to_goal_xy(env, goal_cfg=goal_cfg, asset_cfg=asset_cfg)

    velocity_along_goal = navigation._velocity_along_goal_xy(env, goal_cfg=goal_cfg, asset_cfg=asset_cfg)

    advancing_gate = runtime._linear_ramp(
        value=velocity_along_goal, lower=heading_cfg.min_forward_speed, upper=heading_cfg.full_forward_speed
    )

    normalized_heading_error = torch.clamp(heading_error / heading_cfg.max_heading_error, min=0.0, max=1.0)

    return advancing_gate * normalized_heading_error.square()


def reached_goal_xy_reward(
    env: ManagerBasedRLEnv,
    threshold: float = 0.25,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Sparse success reward based on XY goal distance.

    Returns:
        [num_envs]
    """

    dist_to_goal = navigation._goal_distance_xy(env, goal_cfg, asset_cfg)
    clearance = terrain._base_clearance(env, asset_cfg)

    reached = dist_to_goal < threshold
    clear_enough = clearance > get_min_clearance(env).to(device=clearance.device, dtype=clearance.dtype)

    return torch.logical_and(reached, clear_enough).float()
