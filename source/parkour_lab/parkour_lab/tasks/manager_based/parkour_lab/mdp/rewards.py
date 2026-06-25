import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from . import utils


def illegal_contact_l2(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
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


def velocity_along_goal_xy_clearance_exp(
    env: ManagerBasedRLEnv,
    tracking_cfg: constants.GoalVelocityTrackingCfg = constants.DEFAULT_GOAL_VELOCITY_TRACKING,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
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

    reward = velocity_along_goal_xy_exp(
        env,
        tracking_cfg=tracking_cfg,
        goal_cfg=goal_cfg,
        asset_cfg=asset_cfg
    )

    clearance = utils._base_clearance(
        env,
        asset_cfg=asset_cfg
    )

    has_enough_clearance = clearance > tracking_cfg.min_clearance

    return reward * has_enough_clearance.to(dtype=reward.dtype)


def velocity_along_goal_xy_exp(
    env: ManagerBasedRLEnv,
    tracking_cfg: constants.GoalVelocityTrackingCfg = constants.DEFAULT_GOAL_VELOCITY_TRACKING,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward tracking a desired XY velocity along the direction to the goal.

    Far from the goal:
        desired velocity is close to tracking_cfg.target_speed.

    Near the goal:
        desired velocity decreases toward zero to reduce overshooting.

    This reward does not check whether the robot is upright or has enough
    clearance. Use velocity_along_goal_xy_clearance_exp for the gated version.

    Returns:
        [num_envs]
    """

    velocity_along_goal = utils._velocity_along_goal_xy(
        env,
        goal_cfg=goal_cfg,
        asset_cfg=asset_cfg
    )

    goal_dist_xy = utils._goal_distance_xy(env, goal_cfg, asset_cfg)

    slowdown_scale = torch.clamp(
        goal_dist_xy / tracking_cfg.slow_down_distance,
        min=0.0,
        max=1.0
    )

    desired_velocity = tracking_cfg.target_speed * slowdown_scale

    velocity_error = velocity_along_goal - desired_velocity

    return torch.exp(
        -velocity_error.square() / tracking_cfg.speed_tracking_scale**2
    )


def lateral_velocity_to_goal_xy_l2_sq(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize velocity perpendicular to the XY goal direction.

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    goal_vec_xy = utils._goal_vector_xy(env, goal_cfg, asset_cfg)
    goal_dis_xy = torch.linalg.norm(goal_vec_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)
    goal_dir_xy = goal_vec_xy / goal_dis_xy

    asset: Articulation = env.scene[asset_cfg.name]
    root_vel_xy = asset.data.root_lin_vel_w[:, :2]

    vel_along_goal = torch.sum(root_vel_xy * goal_dir_xy, dim=-1, keepdim=True)

    vel_parallel_to_goal = vel_along_goal * goal_dir_xy
    vel_lateral_to_goal = root_vel_xy - vel_parallel_to_goal

    return torch.sum(vel_lateral_to_goal.square(), dim=-1)


def goal_progress_xy_l2(
    env: ManagerBasedRLEnv,
    max_progress: float = 0.25,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Dense reward for reducing XY distance to the goal.

    progress = previous_distance - current_distance

    Returns:
        [num_envs]
    """

    current_dist = utils._goal_distance_xy(env, goal_cfg, asset_cfg)

    buffer_name = "_parkour_prev_goal_distance"

    if (
        not hasattr(env, buffer_name)
        or getattr(env, buffer_name).shape != current_dist.shape
    ):
        setattr(env, buffer_name, current_dist.detach().clone())

    previous_dist = getattr(env, buffer_name)

    progress = previous_dist - current_dist

    # Avoid artificial progress spikes right after reset.
    if hasattr(env, "episode_length_buf"):
        just_reset = env.episode_length_buf <= 1
        progress = torch.where(
            just_reset,
            torch.zeros_like(progress),
            progress
        )

    # Store current distance for the next step.
    setattr(env, buffer_name, current_dist.detach().clone())

    return torch.clamp(progress, min=-max_progress, max=max_progress)


def reached_goal_xy_l2(
    env: ManagerBasedRLEnv,
    threshold: float,
    min_base_height: float = 0.25,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
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


def obstacle_progress_l2(
    env: ManagerBasedRLEnv,
    obstacle_x: float,
    obstacle_length: float,
    max_progress: float = 0.25,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Reward actual increase in obstacle-region progress.

    Standing still gives 0.
    Moving backward gives negative reward.
    Moving forward gives positive reward.

    Returns:
        [num_envs]
    """

    x = utils._robot_x_env(env, asset_cfg)

    obstacle_front_x = obstacle_x - obstacle_length / 2.0
    obstacle_back_x = obstacle_x + obstacle_length / 2.0

    current_progress = (x - obstacle_front_x) / (obstacle_back_x - obstacle_front_x)
    current_progress = torch.clamp(current_progress, min=0.0, max=1.0)

    buffer_name = "_parkour_prev_obstacle_progress"

    if (
        not hasattr(env, buffer_name)
        or getattr(env, buffer_name).shape != current_progress.shape
    ):
        setattr(env, buffer_name, current_progress.detach().clone())

    previous_progress = getattr(env, buffer_name)

    progress_delta = current_progress - previous_progress

    # Avoid artificial progress spikes right after reset.
    if hasattr(env, "episode_length_buf"):
        just_reset = env.episode_length_buf <= 1
        progress_delta = torch.where(
            just_reset,
            torch.zeros_like(progress_delta),
            progress_delta,
        )

    setattr(env, buffer_name, current_progress.detach().clone())

    return torch.clamp(progress_delta, min=-max_progress, max=max_progress)


def base_height_band_l2(
    env: ManagerBasedRLEnv,
    min_height: float = 0.25,
    max_height: float = 0.50,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalize base/root height outside a healthy walking band.

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    base_height = utils._robot_base_height(env, asset_cfg)

    below = torch.clamp(min_height - base_height, min=0.0)
    above = torch.clamp(base_height - max_height, min=0.0)

    return below.square() + above.square()


def no_feet_contact_l2(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg=SceneEntityCfg("feet_contact", body_names=".*_foot")
) -> torch.Tensor:
    """
    Penalty for having no feet in contact with the ground.

    This discourages hopping/skipping in flat walking.

    Returns:
        [num_envs]
    """

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # [num_envs, history_length, num_bodies, 3]
    net_forces = contact_sensor.data.net_forces_w_history

    if sensor_cfg.body_ids is not None:
        net_forces = net_forces[:, :, sensor_cfg.body_ids, :]

    # [num_envs, history_length, num_bodies]
    force_norm = torch.linalg.norm(net_forces, dim=-1)

    # Has each foot contacted recently?
    # [num_envs, num_bodies]
    feet_in_contact = torch.any(force_norm > threshold, dim=1)

    # [num_envs]
    num_feet_in_contact = torch.sum(feet_in_contact.float(), dim=-1)

    no_contact = num_feet_in_contact < 1.0

    return no_contact.float()
