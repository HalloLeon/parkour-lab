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

    robot_root_pos = _robot_root_pos_env(env, asset_cfg)
    goal_pos = _goal_pos_env(env, goal_cfg)

    return goal_pos - robot_root_pos


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
