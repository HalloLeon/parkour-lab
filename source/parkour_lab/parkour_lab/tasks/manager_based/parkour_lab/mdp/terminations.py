from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import torch

from ._shared import navigation


def reached_goal_xy(
    env: ManagerBasedRLEnv,
    threshold: float,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Success termination based on XY distance to goal,
    while requiring the robot not to be collapsed.

    Returns:
        [num_envs]
    """

    dist_to_goal = navigation._goal_distance_xy(env, goal_cfg, asset_cfg)
    reached = dist_to_goal < threshold

    return reached
