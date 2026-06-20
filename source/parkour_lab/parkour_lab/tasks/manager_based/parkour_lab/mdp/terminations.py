from isaaclab.assets import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch

import utils


def reached_goal(env: ManagerBasedRLEnv, threshold: float, goal_cfg=SceneEntityCfg("goal"), asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    dist_to_goal = utils._goal_distance(env, goal_cfg, asset_cfg)
    return (dist_to_goal < threshold)
