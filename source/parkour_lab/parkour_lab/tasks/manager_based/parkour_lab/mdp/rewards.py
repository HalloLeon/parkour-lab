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


def goal_progress_xy_stable(
    env: ManagerBasedRLEnv,
    progress_cfg: constants.StableGoalProgressCfg = constants.DEFAULT_STABLE_GOAL_PROGRESS,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
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

    current_distance = utils._goal_distance_xy(
        env,
        goal_cfg=goal_cfg,
        asset_cfg=asset_cfg
    )

    buffer_name = utils._private_buffer_name(
        "parkour_prev_goal_distance_xy",
        goal_cfg.name,
        asset_cfg.name
    )

    just_reset = utils._episode_start_mask(
        env,
        reference=current_distance,
        grace_steps=progress_cfg.reset_grace_steps
    )

    progress = utils._difference_from_previous_env_buffer(
        env,
        buffer_name=buffer_name,
        current_value=current_distance,
        reset_mask=just_reset
    )

    stable = utils._root_stability_mask(
        env,
        stability_cfg=progress_cfg.stability,
        asset_cfg=asset_cfg
    )

    progress = utils._gate_positive_values(
        values=progress,
        gate=stable
    )

    return torch.clamp(
        progress,
        min=-progress_cfg.max_progress,
        max=progress_cfg.max_progress
    )


def base_clearance_below_l2(
    env: ManagerBasedRLEnv,
    min_clearance: float,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalty signal for the robot base/root being too close to the surface
    directly underneath it.

    The surface may be:
      - the ground
      - the top of an obstacle
      - later, another support surface

    This is an L2 penalty:

        penalty = max(min_clearance - clearance, 0)^2

    where:

        clearance = base_height - support_surface_height_under_base

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    clearance = utils._base_clearance(env, asset_cfg)

    clearance_error = torch.clamp(min_clearance - clearance, min=0.0)

    return clearance_error.square()


def rapid_feet_motion_l2(
    env: ManagerBasedRLEnv,
    motion_cfg: constants.FeetMotionPenaltyCfg = constants.DEFAULT_FOOT_MOTION_PENALTY,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot")
) -> torch.Tensor:
    """
    Penalize excessive foot speed in a contact-aware way.

    Stance feet are expected to move slowly.
    Swing feet are allowed to move faster.

    The penalty is:

        penalty = max(foot_speed - allowed_speed, 0)^2

    where allowed_speed is:
        - max_stance_speed for feet in contact
        - max_swing_speed for feet not in contact

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    motion_cfg.validate()

    foot_speed = utils._selected_body_speed_w(env, asset_cfg)

    in_contact = utils._contact_mask(
        env,
        sensor_cfg=sensor_cfg,
        threshold=motion_cfg.contact_threshold
    )

    utils._validate_matching_shape(
        in_contact,
        foot_speed,
        lhs_name="foot contact mask",
        rhs_name="foot speed"
    )

    stance_speed_limit = torch.full_like(
        foot_speed,
        motion_cfg.max_stance_speed
    )

    swing_speed_limit = torch.full_like(
        foot_speed,
        motion_cfg.max_swing_speed
    )

    speed_limit = torch.where(
        in_contact,
        stance_speed_limit,
        swing_speed_limit
    )

    excess_speed = torch.clamp(
        foot_speed - speed_limit,
        min=0.0
    )

    penalty_per_foot = torch.clamp(
        excess_speed.square(),
        max=motion_cfg.max_penalty_per_foot
    )

    return penalty_per_foot.mean(dim=-1)


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
