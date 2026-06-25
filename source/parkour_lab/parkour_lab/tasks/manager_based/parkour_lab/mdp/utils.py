from isaaclab.assets import Articulation
from isaaclab.assets import AssetBase
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply
import torch

from . import constants


def _get_scene_entity_or_none(
    env: ManagerBasedRLEnv,
    name: str
) -> AssetBase | None:
    """
    Return a scene entity if it exists, otherwise None.

    This keeps optional scene objects, such as an obstacle, from making
    reward functions crash in simpler environments.
    """

    try:
        return env.scene[name]
    except KeyError:
        return None


def _require_body_ids(
    entity_cfg: SceneEntityCfg,
    *,
    role: str
) -> None:
    """
    Ensure that a SceneEntityCfg has resolved body_ids.

    Raises:
        ValueError: If body_ids are missing.
    """

    if entity_cfg.body_ids is None:
        raise ValueError(
            f"SceneEntityCfg for '{entity_cfg.name}' must resolve body_ids "
            f"when used for {role}. Pass body_names, for example "
            "body_names='.*_foot'."
        )


def _validate_matching_shape(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    lhs_name: str,
    rhs_name: str
) -> None:
    """
    Validate that two tensors have identical shape.

    Raises:
        RuntimeError: If shapes differ.
    """

    if lhs.shape != rhs.shape:
        raise RuntimeError(
            f"{lhs_name} shape does not match {rhs_name} shape. "
            f"Got {lhs_name} shape {tuple(lhs.shape)} and "
            f"{rhs_name} shape {tuple(rhs.shape)}."
        )


def _gate_positive_values(
    values: torch.Tensor,
    gate: torch.Tensor
) -> torch.Tensor:
    """
    Keep negative values always, but allow positive values only when gate is true.

    This is useful for rewards where:
      - positive progress should require valid behavior,
      - negative progress should still be penalized.

    Returns:
        Tensor with the same shape as values.
    """

    keep_value = torch.logical_or(values <= 0.0, gate)

    return torch.where(
        keep_value,
        values,
        torch.zeros_like(values)
    )


def _private_buffer_name(
    prefix: str,
    *parts: str
) -> str:
    """
    Create a private environment-buffer name.

    Returns:
        str
    """

    safe_parts = [
        str(part).replace("/", "_").replace(" ", "_")
        for part in parts
    ]

    return "_" + "_".join((prefix, *safe_parts))


def _get_or_init_env_buffer(
    env: ManagerBasedRLEnv,
    name: str,
    value: torch.Tensor
) -> torch.Tensor:
    """
    Get an environment-level tensor buffer, creating or resizing it if needed.

    Returns:
        Tensor with the same shape, device, and dtype as value.
    """

    needs_init = (
        not hasattr(env, name)
        or getattr(env, name).shape != value.shape
        or getattr(env, name).device != value.device
        or getattr(env, name).dtype != value.dtype
    )

    if needs_init:
        setattr(env, name, value.detach().clone())

    return getattr(env, name)


def _set_env_buffer(
    env: ManagerBasedRLEnv,
    name: str,
    value: torch.Tensor
) -> None:
    """
    Store a detached clone as an environment-level tensor buffer.
    """

    setattr(env, name, value.detach().clone())


def _difference_from_previous_env_buffer(
    env: ManagerBasedRLEnv,
    *,
    buffer_name: str,
    current_value: torch.Tensor,
    reset_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute previous_value - current_value using an environment-level buffer.

    The buffer is always updated, even when reset_mask is true.

    Returns:
        [num_envs]
    """

    previous_value = _get_or_init_env_buffer(
        env=env,
        name=buffer_name,
        value=current_value
    )

    difference = previous_value - current_value

    if reset_mask is not None:
        difference = torch.where(
            reset_mask,
            torch.zeros_like(difference),
            difference
        )

    _set_env_buffer(
        env=env,
        name=buffer_name,
        value=current_value
    )

    return difference


def _box_surface_height_under_xy(
    env: ManagerBasedRLEnv,
    xy: torch.Tensor,
    box_cfg: constants.BoxSurfaceCfg
) -> torch.Tensor:
    """
    Height of a box top surface under a given XY position.

    If the XY position is outside the box footprint, returns -inf.

    Args:
        env: The RL environment.
        xy: Query positions in environment-local XY coordinates, shape [num_envs, 2].
        box_cfg: Box surface configuration.

    Returns:
        [num_envs]
    """

    box = _get_scene_entity_or_none(env, box_cfg.name)

    if box is None:
        return torch.full(
            (xy.shape[0],),
            -torch.inf,
            device=xy.device,
            dtype=xy.dtype,
        )

    box_pos_env = box.data.root_pos_w - env.scene.env_origins

    half_size_x = 0.5 * box_cfg.size[0] + box_cfg.xy_margin
    half_size_y = 0.5 * box_cfg.size[1] + box_cfg.xy_margin
    half_size_z = 0.5 * box_cfg.size[2]

    dx = torch.abs(xy[:, 0] - box_pos_env[:, 0])
    dy = torch.abs(xy[:, 1] - box_pos_env[:, 1])

    above_footprint = torch.logical_and(
        dx <= half_size_x,
        dy <= half_size_y,
    )

    box_top_height = box_pos_env[:, 2] + half_size_z

    return torch.where(
        above_footprint,
        box_top_height,
        torch.full_like(box_top_height, -torch.inf),
    )


def _support_surface_height_under_base(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Highest support surface directly underneath the robot base/root.

    Currently considers:
      - flat ground
      - optional box obstacle

    Returns:
        [num_envs]
    """

    root_pos = _root_pos_env(env, asset_cfg)
    base_xy = root_pos[:, :2]

    ground_height = torch.full(
        (root_pos.shape[0],),
        constants.GROUND_HEIGHT,
        device=root_pos.device,
        dtype=root_pos.dtype,
    )

    obstacle_height = _box_surface_height_under_xy(
        env=env,
        xy=base_xy,
        box_cfg=constants.OBSTACLE_SURFACE,
    )

    return torch.maximum(ground_height, obstacle_height)


def _selected_body_lin_vel_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Linear velocity of selected articulation bodies in world frame.

    Returns:
        [num_envs, num_bodies, 3]
    """

    _require_body_ids(asset_cfg, role="body velocity selection")

    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]


def _selected_body_speed_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Speed magnitude of selected articulation bodies in world frame.

    Returns:
        [num_envs, num_bodies]
    """

    body_lin_vel_w = _selected_body_lin_vel_w(env, asset_cfg)

    return torch.linalg.norm(body_lin_vel_w, dim=-1)


def _selected_contact_forces_w_history(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Contact forces for selected contact-sensor bodies.

    Returns:
        [num_envs, history_length, num_bodies, 3]
    """

    _require_body_ids(sensor_cfg, role="contact force selection")

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    return contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]


def _selected_joint_pos_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Position error of selected joints relative to their default joint positions.

    Returns:
        [num_envs, num_joints]
    """

    if asset_cfg.joint_ids is None:
        raise ValueError(
            f"SceneEntityCfg for '{asset_cfg.name}' must resolve joint_ids. "
            "Pass joint_names, for example joint_names='.*_hip_joint'."
        )

    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    return joint_pos - default_joint_pos


def _contact_mask(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float
) -> torch.Tensor:
    """
    Boolean contact mask for selected contact-sensor bodies.

    Returns:
        [num_envs, num_bodies]
    """

    _require_body_ids(sensor_cfg, role="contact detection")

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # [num_envs, history_length, num_sensor_bodies, 3]
    net_forces_w = contact_sensor.data.net_forces_w_history

    # [num_envs, history_length, selected_bodies, 3]
    net_forces_w = net_forces_w[:, :, sensor_cfg.body_ids, :]

    # [num_envs, history_length, selected_bodies]
    force_norm = torch.linalg.norm(net_forces_w, dim=-1)

    # [num_envs, selected_bodies]
    return torch.any(force_norm > threshold, dim=1)


def _episode_start_mask(
    env: ManagerBasedRLEnv,
    reference: torch.Tensor,
    grace_steps: int
) -> torch.Tensor:
    """
    Boolean mask for environments that have just reset.

    Returns:
        [num_envs]
    """

    if grace_steps <= 0 or not hasattr(env, "episode_length_buf"):
        return torch.zeros_like(reference, dtype=torch.bool)

    episode_length = env.episode_length_buf.to(device=reference.device)

    return episode_length <= grace_steps


def _root_stability_mask(
    env: ManagerBasedRLEnv,
    stability_cfg: constants.RootStabilityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Check whether the robot root is stable and sufficiently clear of the
    support surface underneath it.

    Stability requires:
      - limited roll/pitch angular velocity,
      - limited roll/pitch tilt,
      - enough base/root clearance above the current support surface.

    The support surface may be:
      - flat ground,
      - obstacle top,
      - later another terrain/platform surface.

    Returns:
        [num_envs]
    """

    asset: Articulation = env.scene[asset_cfg.name]

    # Roll/pitch angular speed.
    # [num_envs]
    roll_pitch_ang_speed = torch.linalg.norm(
        asset.data.root_ang_vel_b[:, :2],
        dim=-1
    )

    # Projected gravity x/y norm is small when the base is upright.
    # [num_envs]
    projected_gravity_xy_norm = torch.linalg.norm(
        asset.data.projected_gravity_b[:, :2],
        dim=-1
    )

    # Clearance above whatever support surface is underneath the base.
    # This is not raw world height.
    # [num_envs]
    base_clearance = _base_clearance(
        env,
        asset_cfg=asset_cfg
    )

    ang_vel_stable = (
        roll_pitch_ang_speed < stability_cfg.max_roll_pitch_ang_speed
    )

    orientation_stable = (
        projected_gravity_xy_norm < stability_cfg.max_projected_gravity_xy_norm
    )

    clearance_stable = (
        base_clearance > stability_cfg.min_clearance
    )

    return torch.logical_and(
        torch.logical_and(ang_vel_stable, orientation_stable),
        clearance_stable,
    )


def _root_lin_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot root linear velocity in the world XY plane.

    Returns:
        [num_envs, 2]
    """

    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.root_lin_vel_w[:, :2]


def _root_pos_env(
        env: ManagerBasedRLEnv,
        asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot root position in each environment's local frame.

    Returns:
        [num_envs, 3]
    """

    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def _robot_xy_env(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot root XY position in each environment's local frame.

    Returns:
        [num_envs, 2]
    """

    return _robot_root_pos_env(env, asset_cfg)[:, :2]


def _robot_x_env(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot root x-position in each environment's local frame.

    Returns:
        [num_envs]
    """

    return _robot_root_pos_env(env, asset_cfg)[:, 0]


def _robot_y_env(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot root y-position in each environment's local frame.

    Returns:
        [num_envs]
    """

    return _robot_root_pos_env(env, asset_cfg)[:, 1]


def _obstacle_pos_env(
        env: ManagerBasedRLEnv,
        asset_cfg=SceneEntityCfg("obstacle")
) -> torch.Tensor:
    """
    Obstacle position in each environment's local frame.

    Returns:
        [num_envs, 3]
    """

    obstacle: AssetBase = env.scene[asset_cfg.name]
    return obstacle.data.root_pos_w - env.scene.env_origins


def _goal_pos_env(
        env: ManagerBasedRLEnv,
        goal_cfg=SceneEntityCfg("goal")
) -> torch.Tensor:
    """
    Goal position in each environment's local frame.

    Returns:
        [num_envs, 3]
    """

    goal: AssetBase = env.scene[goal_cfg.name]
    return goal.data.root_pos_w - env.scene.env_origins


def _goal_vector_xyz(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    XYZ vector from robot root to goal.

    Returns:
        [num_envs, 3]
    """

    robot_root_pos = _root_pos_env(env, asset_cfg)
    goal_pos = _goal_pos_env(env, goal_cfg)

    return goal_pos - robot_root_pos


def _goal_vector_xy(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    XY vector from robot root to goal.

    Returns:
        [num_envs, 2]
    """

    to_goal_xyz = _goal_vector_xyz(env, goal_cfg, asset_cfg)
    return to_goal_xyz[:, :2]


def _goal_distance_xyz(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    XYZ distance from robot root to goal.

    Returns:
        [num_envs]
    """

    to_goal = _goal_vector_xyz(env, goal_cfg, asset_cfg)
    return torch.linalg.norm(to_goal, dim=-1)


def _goal_distance_xy(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    XY distance from robot root to goal.

    Returns:
        [num_envs]
    """

    to_goal_xy = _goal_vector_xy(env, goal_cfg, asset_cfg)
    return torch.linalg.norm(to_goal_xy, dim=-1)


def _robot_base_height(
        env: ManagerBasedRLEnv,
        asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot root/base height in each local environment frame.

    Returns:
        [num_envs]
    """

    robot_root_pos = _robot_root_pos_env(env, asset_cfg)
    return robot_root_pos[:, 2]
