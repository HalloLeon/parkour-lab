from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import torch

from . import utils


def reached_goal_xy(
    env: ManagerBasedRLEnv,
    threshold: float,
    min_base_height: float = 0.22,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Success termination based on XY distance to goal,
    while requiring the robot not to be collapsed.

    Returns:
        [num_envs]
    """

    dist_to_goal = utils._goal_distance_xy(env, goal_cfg, asset_cfg)
    base_height = utils._robot_base_height(env, asset_cfg)

    reached = dist_to_goal < threshold
    base_high_enough = base_height > min_base_height

    return torch.logical_and(reached, base_high_enough)
