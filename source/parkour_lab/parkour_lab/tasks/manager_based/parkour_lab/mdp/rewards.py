import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from . import utils


def illegal_contact_l2(env: ManagerBasedRLEnv, sensor_cfg=SceneEntityCfg("base_contact", body_names="trunk")) -> torch.Tensor:
    contact_sensor = env.scene[sensor_cfg.name]
    illegal_contact = contact_sensor.data.contact > 0
    return illegal_contact.float()


def reached_goal_l2(env: ManagerBasedRLEnv, threshold: float, goal_cfg=SceneEntityCfg("goal"), asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    dist_to_goal = utils._goal_distance(env, goal_cfg, asset_cfg)
    return (dist_to_goal < threshold).float()


def velocity_towards_goal_l2(env: ManagerBasedRLEnv, goal_cfg=SceneEntityCfg("goal"), asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Reward velocity along the XYZ direction from robot to goal.

    Returns:
        [num_envs]
    """

    to_goal = utils._goal_vector_xyz(env, goal_cfg, asset_cfg)

    # [num_envs, 1]
    to_goal_norm = torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1.0e-6)

    # [num_envs, 3]
    goal_dir = to_goal / to_goal_norm

    asset: Articulation = env.scene[asset_cfg.name]

    # [num_envs, 3]
    vel_asset = asset.data.root_lin_vel_w

    return torch.sum(vel_asset * goal_dir, dim=-1).clamp(min=-1.0, max=1.0)
