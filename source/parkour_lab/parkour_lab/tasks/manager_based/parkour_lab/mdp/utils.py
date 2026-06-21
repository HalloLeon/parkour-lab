from isaaclab.assets import Articulation
from isaaclab.assets import AssetBase
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import torch


def _robot_root_pos_env(env: ManagerBasedRLEnv, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Robot root position in each environment's local frame.

    Returns:
        [num_envs, 3]
    """

    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def _goal_pos_env(env: ManagerBasedRLEnv, goal_cfg=SceneEntityCfg("goal")) -> torch.Tensor:
    """
    Goal position in each environment's local frame.

    Returns:
        [num_envs, 3]
    """

    goal: AssetBase = env.scene[goal_cfg.name]
    return goal.data.root_pos_w - env.scene.env_origins


def _goal_vector_xyz(env: ManagerBasedRLEnv, goal_cfg=SceneEntityCfg("goal"), asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """
    XYZ vector from robot root to goal.

    Returns:
        [num_envs, 3]
    """

    robot_root_pos = _robot_root_pos_env(env, asset_cfg)
    goal_pos = _goal_pos_env(env, goal_cfg)

    return goal_pos - robot_root_pos


def _goal_distance_xyz(env: ManagerBasedRLEnv, goal_cfg=SceneEntityCfg("goal"), asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
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
    asset_cfg=SceneEntityCfg("robot"),
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
    asset_cfg=SceneEntityCfg("robot"),
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
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Robot root/base height in each local environment frame.

    Returns:
        [num_envs]
    """

    robot_root_pos = _robot_root_pos_env(env, asset_cfg)
    return robot_root_pos[:, 2]
