import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

import utils


def illegal_contact_l2(env: ManagerBasedRLEnv, sensor_cfg=SceneEntityCfg("base_contact", body_names="trunk")) -> torch.Tensor:
    contact_sensor = env.scene[sensor_cfg.name]
    illegal_contact = contact_sensor.data.contact > 0
    return illegal_contact.float()


def reached_goal_l2(env: ManagerBasedRLEnv, goal_x: float, goal_y: float, goal_z: float, threshold: float = 0.5, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    dist_to_goal = utils._goal_distance(env, goal_x, goal_y, goal_z, asset_cfg)
    return (dist_to_goal < threshold).float()


def velocity_towards_goal_l2(env: ManagerBasedRLEnv, goal_x: float, goal_y: float, goal_z: float, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    robot_root_pos = utils._robot_root_pos_env(env, asset_cfg)
    goal_pos = torch.tensor([goal_x, goal_y, goal_z], device=robot_root_pos.device, dtype=robot_root_pos.dtype).unsqueeze(0)

    to_goal = goal_pos - robot_root_pos
    to_goal_norm = torch.linalg.norm(to_goal, dim=-1, keepdim=True).clamp_min(1.0e-6)
    goal_dir = to_goal / to_goal_norm

    asset: Articulation = env.scene[asset_cfg.name]
    vel_asset = asset.data.root_lin_vel_w

    return torch.sum(vel_asset * goal_dir, dim=-1).clamp(min=-1.0, max=1.0)
