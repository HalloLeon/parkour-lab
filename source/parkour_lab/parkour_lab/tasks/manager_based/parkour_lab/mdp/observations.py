from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import torch

from . import utils


def goal_distance_xy(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    XY distance from robot root to goal.

    Returns:
        [num_envs, 1]
    """

    return utils._goal_distance_xy(env, goal_cfg, asset_cfg).unsqueeze(-1)


def base_height_w(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Robot base/root height.

    Returns:
        [num_envs, 1]
    """

    return utils._robot_base_height(env, asset_cfg).unsqueeze(-1)


def goal_distance_xyz_w(env: ManagerBasedRLEnv, goal_cfg=SceneEntityCfg("goal"), asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """
    XYZ distance from robot root to goal.

    Returns:
        [num_envs, 1]
    """

    return utils._goal_distance_xyz(env, goal_cfg, asset_cfg).unsqueeze(-1)
