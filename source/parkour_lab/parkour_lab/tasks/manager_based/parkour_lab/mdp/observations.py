from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import torch

import utils


def goal_distance_w(env: ManagerBasedRLEnv, goal_x: float, goal_y: float, goal_z: float, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    return utils._goal_distance(env, goal_x, goal_y, goal_z, asset_cfg)
