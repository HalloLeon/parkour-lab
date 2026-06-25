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

    return utils._root_height(env, asset_cfg).unsqueeze(-1)


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


def desired_speed_obs(
    env: ManagerBasedRLEnv,
    target_speed: float = 0.6
) -> torch.Tensor:
    """
    Constant desired forward speed observation.

    Returns:
        [num_envs, 1]
    """

    return torch.full(
        (env.num_envs, 1),
        target_speed,
        device=env.device
    )


def base_clearance_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Base/root clearance above the support surface underneath the robot.

    Returns:
        [num_envs, 1]
    """

    return utils._base_clearance(env, asset_cfg).unsqueeze(-1)


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
