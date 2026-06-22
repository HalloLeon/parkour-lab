from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import torch

from . import utils


def base_height_w(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot base/root height.

    Returns:
        [num_envs, 1]
    """

    return utils._robot_base_height(env, asset_cfg).unsqueeze(-1)


def goal_distance_xy_w(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    XY distance from robot root to goal.

    Returns:
        [num_envs, 1]
    """

    return utils._goal_distance_xy(env, goal_cfg, asset_cfg).unsqueeze(-1)


def goal_distance_xyz_w(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    XYZ distance from robot root to goal.

    Returns:
        [num_envs, 1]
    """

    return utils._goal_distance_xyz(env, goal_cfg, asset_cfg).unsqueeze(-1)


def goal_direction_xy_w(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Normalized XY direction from robot root to goal.

    Returns:
        [num_envs, 2]
    """

    to_goal_xy = utils._goal_vector_xy(env, goal_cfg, asset_cfg)

    norm = torch.linalg.norm(to_goal_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)

    return to_goal_xy / norm
