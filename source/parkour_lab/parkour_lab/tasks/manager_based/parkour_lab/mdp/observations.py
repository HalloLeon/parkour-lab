from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse
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


def foot_contact_state(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot")
) -> torch.Tensor:
    """
    Foot contact state, centered.

    Official-style convention:
        no contact -> -0.5
        contact    ->  0.5

    Returns:
        [num_envs, num_feet]
    """

    contact = utils._contact_mask(
        env,
        sensor_cfg=sensor_cfg,
        threshold=threshold,
    )

    return contact.float() - 0.5


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


def goal_direction_body_xy(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Direction from robot to goal, expressed in the robot body frame.

    Returns:
        [num_envs, 2]

    Interpretation:
        x component: goal is in front/behind robot
        y component: goal is left/right of robot
    """

    asset: Articulation = env.scene[asset_cfg.name]

    goal_vec_xy = utils._goal_vector_xy(env, goal_cfg, asset_cfg)

    goal_vec_w = torch.zeros(
        (goal_vec_xy.shape[0], 3),
        device=goal_vec_xy.device,
        dtype=goal_vec_xy.dtype
    )
    goal_vec_w[:, :2] = goal_vec_xy

    # Rotate the world-frame goal vector into the robot body frame.
    #
    # If q is the robot root orientation, this applies the inverse rotation:
    #
    #     goal_vec_b = q^-1 * goal_vec_w * q
    #
    # This answers the question:
    #
    #     "Where is the goal relative to the robot's own forward/left axes?"
    #
    # Examples:
    #   robot yaw =   0 deg, goal world +x -> goal_vec_b ≈ [ 1,  0, 0]
    #   robot yaw =  90 deg, goal world +x -> goal_vec_b ≈ [ 0, -1, 0]
    #   robot yaw = 180 deg, goal world +x -> goal_vec_b ≈ [-1,  0, 0]
    #
    # The suffix "_b" means body frame.
    goal_vec_b = quat_rotate_inverse(asset.data.root_quat_w, goal_vec_w)
    goal_dir_b_xy = goal_vec_b[:, :2]

    return goal_dir_b_xy / torch.linalg.norm(
        goal_dir_b_xy,
        dim=-1,
        keepdim=True
    ).clamp_min(1.0e-6)


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
