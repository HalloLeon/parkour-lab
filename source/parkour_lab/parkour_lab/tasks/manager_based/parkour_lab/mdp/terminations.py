from isaaclab.assets import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch

import utils


def reached_goal(env: ManagerBasedRLEnv, goal_x: float, goal_y: float, goal_z: float, threshold: float = 0.5, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    dist_to_goal = utils._goal_distance(env, goal_x, goal_y, goal_z, asset_cfg)
    return (dist_to_goal < threshold)
