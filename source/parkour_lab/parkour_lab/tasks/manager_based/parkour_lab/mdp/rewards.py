import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from . import utils


def illegal_contact_l2(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg=SceneEntityCfg("base_contact", body_names="trunk")
) -> torch.Tensor:
    """
    Penalty signal for illegal trunk/base contact.

    Returns:
        Tensor of shape [num_envs].
    """

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # [num_envs, history_length, num_bodies, 3]
    net_forces = contact_sensor.data.net_forces_w_history

    if sensor_cfg.body_ids is not None:
        net_forces = net_forces[:, :, sensor_cfg.body_ids, :]

    # [num_envs, history_length, selected_bodies]
    force_norm = torch.linalg.norm(net_forces, dim=-1)

    # [num_envs]
    has_illegal_contact = torch.any(force_norm > threshold, dim=(1, 2))

    return has_illegal_contact.float()


def velocity_along_goal_xy_exp(
    env: ManagerBasedRLEnv,
    target_speed: float = 0.5,
    speed_tracking_scale: float = 0.25,
    slow_down_distance: float = 0.5,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward moving toward the XY goal at a desired speed.

    The desired speed decreases near the goal to reduce lunging,
    jumping, and overshooting.

    Returns:
        [num_envs]
    """

    goal_vec_xy = utils._goal_vector_xy(env, goal_cfg, asset_cfg)
    goal_dis_xy = torch.linalg.norm(goal_vec_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    goal_dir_xy = goal_vec_xy / goal_dis_xy

    asset: Articulation = env.scene[asset_cfg.name]
    root_vel_xy = asset.data.root_lin_vel_w[:, :2]

    velocity_along_goal = torch.sum(root_vel_xy * goal_dir_xy, dim=-1)

    # Far from the goal: desired speed ~= target_speed.
    # Near the goal: desired speed gradually decreases.
    slowdown_scale = torch.clamp(
        goal_dis_xy.squeeze(-1) / slow_down_distance,
        min=0.0,
        max=1.0)
    desired_vel_along_goal = target_speed * slowdown_scale

    vel_err_along_goal = velocity_along_goal - desired_vel_along_goal

    return torch.exp(
        -vel_err_along_goal.square()
        / (speed_tracking_scale * speed_tracking_scale)
    )


def lateral_velocity_to_goal_xy_l2_sq(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Reward velocity along the XY direction from robot to goal.

    Returns:
        [num_envs]
    """

    to_goal_xy = utils._goal_vector_xy(env, goal_cfg, asset_cfg)

    # [num_envs, 1]
    to_goal_norm = torch.linalg.norm(to_goal_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)

    # [num_envs, 2]
    goal_dir_xy = to_goal_xy / to_goal_norm

    asset: Articulation = env.scene[asset_cfg.name]

    # [num_envs, 2]
    vel_xy = asset.data.root_lin_vel_w[:, :2]

    return torch.sum(vel_xy * goal_dir_xy, dim=-1).clamp(min=-1.0, max=1.0)


def goal_closeness_xy_l2(
    env: ManagerBasedRLEnv,
    max_distance: float,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Dense reward for being horizontally close to the goal.

    Returns:
        [num_envs], roughly in [0, 1].
    """

    dist_to_goal = utils._goal_distance_xy(env, goal_cfg, asset_cfg)

    closeness = 1.0 - torch.clamp(dist_to_goal / max_distance, min=0.0, max=1.0)

    return closeness


def reached_goal_xy_l2(
    env: ManagerBasedRLEnv,
    threshold: float,
    min_base_height: float = 0.22,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Sparse success reward based on XY goal distance,
    while requiring the robot base to remain above a minimum height.

    Returns:
        [num_envs]
    """

    dist_to_goal = utils._goal_distance_xy(env, goal_cfg, asset_cfg)
    base_height = utils._robot_base_height(env, asset_cfg)

    reached = dist_to_goal < threshold
    base_high_enough = base_height > min_base_height

    return torch.logical_and(reached, base_high_enough).float()


def base_height_error_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    max_error: float = 0.5,
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalty term for deviating from a desired base/root height.

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    base_height = utils._robot_base_height(env, asset_cfg)

    error = base_height - target_height
    error_l2 = error.square()

    return torch.clamp(error_l2, max=max_error * max_error)
