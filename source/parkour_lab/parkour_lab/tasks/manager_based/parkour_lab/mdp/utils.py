from isaaclab.assets import Articulation
from isaaclab.assets import AssetBase
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply
from isaaclab.utils.math import quat_apply_inverse

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


def _xy_vector_w_to_xy_vector_b(
    env: ManagerBasedRLEnv,
    vector_xy_w: torch.Tensor,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Convert a world-frame XY vector into body-frame XY.

    Args:
        env: The RL environment.
        vector_xy_w: World-frame XY vector, shape [num_envs, 2].
        asset_cfg: Robot asset config.

    Returns:
        [num_envs, 2]
    """

    asset: Articulation = env.scene[asset_cfg.name]

    vector_w = torch.zeros(
        (vector_xy_w.shape[0], 3),
        device=vector_xy_w.device,
        dtype=vector_xy_w.dtype
    )
    vector_w[:, :2] = vector_xy_w

    vector_b = quat_apply_inverse(asset.data.root_quat_w, vector_w)

    return vector_b[:, :2]


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
            dtype=xy.dtype
        )

    box_pos_env = box.data.root_pos_w - env.scene.env_origins

    half_size_x = 0.5 * box_cfg.size[0] + box_cfg.xy_margin
    half_size_y = 0.5 * box_cfg.size[1] + box_cfg.xy_margin
    half_size_z = 0.5 * box_cfg.size[2]

    dx = torch.abs(xy[:, 0] - box_pos_env[:, 0])
    dy = torch.abs(xy[:, 1] - box_pos_env[:, 1])

    above_footprint = torch.logical_and(
        dx <= half_size_x,
        dy <= half_size_y
    )

    box_top_height = box_pos_env[:, 2] + half_size_z

    return torch.where(
        above_footprint,
        box_top_height,
        torch.full_like(box_top_height, -torch.inf)
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
        dtype=root_pos.dtype
    )

    obstacle_height = _box_surface_height_under_xy(
        env=env,
        xy=base_xy,
        box_cfg=constants.OBSTACLE_SURFACE
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
        clearance_stable
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


def _root_height(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot root/base height in each environment's local frame.

    Returns:
        [num_envs]
    """

    return _root_pos_env(env, asset_cfg)[:, 2]


def _root_forward_xy_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot root forward direction in world XY.

    Returns:
        [num_envs, 2]
    """

    asset: Articulation = env.scene[asset_cfg.name]

    # In the robot's own body frame, we define "forward" as +X.
    #
    # So before considering the robot's current orientation, the local forward
    # vector is:
    #
    #     forward_b = [1, 0, 0]
    #
    # The suffix "_b" means body frame.
    forward_b = torch.zeros(
        (asset.data.root_quat_w.shape[0], 3),
        device=asset.data.root_quat_w.device,
        dtype=asset.data.root_quat_w.dtype
    )

    forward_b[:, 0] = 1.0

    # Rotate the body-frame forward vector into the world frame.
    #
    # If q is the robot root orientation, this applies:
    #
    #     forward_w = q * forward_b * q^-1
    #
    # Examples:
    #   yaw =   0 deg -> forward_w ≈ [ 1,  0, 0]
    #   yaw =  90 deg -> forward_w ≈ [ 0,  1, 0]
    #   yaw = 180 deg -> forward_w ≈ [-1,  0, 0]
    #
    # The suffix "_w" means world frame.
    forward_w = quat_apply(asset.data.root_quat_w, forward_b)

    # Keep only the horizontal part of the world-frame forward vector.
    #
    # We discard z because goal navigation uses the ground-plane heading:
    #
    #     forward_w  = [x, y, z]
    #     forward_xy = [x, y]
    #
    # This tells us where the robot is facing in the world XY plane.
    forward_xy = forward_w[:, :2]

    # Normalize the XY vector to unit length.
    #
    #     forward_xy_unit = forward_xy / sqrt(x^2 + y^2)
    #
    # This makes the result a direction only, independent of vector magnitude.
    #
    # clamp_min(1.0e-6) avoids division by zero if the horizontal projection is
    # extremely small, for example if the robot is nearly vertical.
    return forward_xy / torch.linalg.norm(
        forward_xy,
        dim=-1,
        keepdim=True
    ).clamp_min(1.0e-6)


def _base_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Vertical clearance between robot base/root and the surface underneath it.

    Returns:
        [num_envs]
    """

    base_height = _root_height(env, asset_cfg)
    surface_height = _support_surface_height_under_base(env, asset_cfg)

    return base_height - surface_height


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


def _goal_direction_xy(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Unit XY direction from robot root to goal.

    Returns:
        [num_envs, 2]
    """

    goal_vec_xy = _goal_vector_xy(env, goal_cfg, asset_cfg)

    goal_dist_xy = torch.linalg.norm(
        goal_vec_xy,
        dim=-1,
        keepdim=True
    ).clamp_min(1.0e-6)

    return goal_vec_xy / goal_dist_xy


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


def _heading_error_to_goal_xy(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Heading error between robot forward direction and XY goal direction.

    Returns:
        [num_envs]
    """

    # Unit vector in world XY showing where the robot's body is facing.
    #
    # Example:
    #   [1, 0] means facing along world +x
    #   [0, 1] means facing along world +y
    forward_xy = _root_forward_xy_w(env, asset_cfg)

    # Unit vector in world XY pointing from the robot root to the goal.
    #
    # Example:
    #   [1, 0] means the goal is in front along world +x
    #   [0, 1] means the goal is to the world +y side
    goal_dir_xy = _goal_direction_xy(env, goal_cfg, asset_cfg)

    # Dot product between two unit vectors.
    #
    # For unit vectors:
    #
    #     dot = cos(theta)
    #
    # where theta is the angle between them.
    #
    # Examples:
    #   dot =  1 -> same direction       -> theta = 0
    #   dot =  0 -> perpendicular        -> theta = pi / 2
    #   dot = -1 -> opposite directions  -> theta = pi
    dot = torch.sum(forward_xy * goal_dir_xy, dim=-1).clamp(
        min=-1.0,
        max=1.0
    )

    # Convert cosine similarity into an angle.
    #
    # For two unit vectors in the XY plane, we can write:
    #
    #     forward_xy = [cos(a), sin(a)]
    #     goal_dir_xy = [cos(b), sin(b)]
    #
    # Their dot product is:
    #
    #     dot = forward_xy · goal_dir_xy
    #         = cos(a) cos(b) + sin(a) sin(b)
    #
    # Using the trigonometric identity:
    #
    #     cos(a - b) = cos(a) cos(b) + sin(a) sin(b)
    #
    # we get:
    #
    #     dot = cos(a - b)
    #
    # Therefore, the angle between the robot heading and the goal direction is:
    #
    #     heading_error = acos(dot)
    #
    # This gives the unsigned heading error in radians:
    #     0      -> facing the goal
    #     pi / 2 -> facing sideways
    #     pi     -> facing away from the goal
    #
    # It is unsigned because cos(+theta) == cos(-theta), so this tells us how large
    # the error is, but not whether the goal is to the left or right.
    return torch.acos(dot)


def _velocity_along_goal_xy(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot root velocity projected onto the XY goal direction.

    Positive:
        moving toward the goal

    Negative:
        moving away from the goal

    Returns:
        [num_envs]
    """

    goal_dir_xy = _goal_direction_xy(env, goal_cfg, asset_cfg)
    root_vel_xy = _root_lin_vel_xy(env, asset_cfg)

    return torch.sum(root_vel_xy * goal_dir_xy, dim=-1)
