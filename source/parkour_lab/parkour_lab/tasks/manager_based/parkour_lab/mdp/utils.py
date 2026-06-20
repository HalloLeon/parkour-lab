from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import torch


def _robot_root_pos_env(env: ManagerBasedRLEnv, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_base_w - env.scene.env_origins


def _goal_distance(env: ManagerBasedRLEnv, goal_x: float, goal_y: float, goal_z: float, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    robot_root_pos = _robot_root_pos_env(env, asset_cfg)
    goal_pos = torch.tensor([goal_x, goal_y, goal_z], device=robot_root_pos.device, dtype=robot_root_pos.dtype).unsqueeze(0)

    return torch.linalg.norm(robot_root_pos - goal_pos, dim=-1, keepdim=True)
