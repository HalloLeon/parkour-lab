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


def _goal_distance(env: ManagerBasedRLEnv, goal_cfg=SceneEntityCfg("goal"), asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """
    XYZ distance from robot root to goal.

    Returns:
        [num_envs]
    """

    robot_root_pos = _robot_root_pos_env(env, asset_cfg)
    goal: AssetBase = env.scene[goal_cfg.name]
    goal_pos = goal.data.root_pos_w - env.scene.env_origins

    return torch.linalg.norm(robot_root_pos - goal_pos, dim=-1)
