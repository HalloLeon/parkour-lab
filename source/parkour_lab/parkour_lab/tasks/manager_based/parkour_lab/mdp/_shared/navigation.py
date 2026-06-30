from isaaclab.assets import AssetBase
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import torch

from .state import _root_forward_xy_w
from .state import _root_lin_vel_xy
from .state import _root_pos_env


def _goal_pos_env(
        env: ManagerBasedRLEnv,
        goal_cfg=SceneEntityCfg("goal")
) -> torch.Tensor:
    """
    Goal position in each environment's local frame.

    Returns:
        [num_envs, 3]
    """

    goal: AssetBase = env.scene[goal_cfg.name]
    return goal.data.root_pos_w - env.scene.env_origins


def _goal_vector_xy(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    XY vector from robot root to goal.

    Returns:
        [num_envs, 2]
    """

    to_goal_xyz = _goal_vector_xyz(env, goal_cfg, asset_cfg)
    return to_goal_xyz[:, :2]


def _goal_vector_xyz(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    XYZ vector from robot root to goal.

    Returns:
        [num_envs, 3]
    """

    robot_root_pos = _root_pos_env(env, asset_cfg)
    goal_pos = _goal_pos_env(env, goal_cfg)

    return goal_pos - robot_root_pos


def _goal_direction_xy(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Unit XY direction from robot root to goal.

    Returns:
        [num_envs, 2]
    """

    goal_vec_xy = _goal_vector_xy(env, goal_cfg, asset_cfg)

    goal_dist_xy = torch.linalg.norm(
        goal_vec_xy,
        dim=-1,
        keepdim=True
    ).clamp_min(1.0e-6)

    return goal_vec_xy / goal_dist_xy


def _goal_distance_xy(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    XY distance from robot root to goal.

    Returns:
        [num_envs]
    """

    to_goal_xy = _goal_vector_xy(env, goal_cfg, asset_cfg)
    return torch.linalg.norm(to_goal_xy, dim=-1)


def _goal_distance_xyz(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    XYZ distance from robot root to goal.

    Returns:
        [num_envs]
    """

    to_goal = _goal_vector_xyz(env, goal_cfg, asset_cfg)
    return torch.linalg.norm(to_goal, dim=-1)


def _heading_error_to_goal_xy(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Heading error between robot forward direction and XY goal direction.

    Returns:
        [num_envs]
    """

    # Unit vector in world XY showing where the robot's body is facing.
    #
    # Example:
    #   [1, 0] means facing along world +x
    #   [0, 1] means facing along world +y
    forward_xy = _root_forward_xy_w(env, asset_cfg)

    # Unit vector in world XY pointing from the robot root to the goal.
    #
    # Example:
    #   [1, 0] means the goal is in front along world +x
    #   [0, 1] means the goal is to the world +y side
    goal_dir_xy = _goal_direction_xy(env, goal_cfg, asset_cfg)

    # Dot product between two unit vectors.
    #
    # For unit vectors:
    #
    #     dot = cos(theta)
    #
    # where theta is the angle between them.
    #
    # Examples:
    #   dot =  1 -> same direction       -> theta = 0
    #   dot =  0 -> perpendicular        -> theta = pi / 2
    #   dot = -1 -> opposite directions  -> theta = pi
    dot = torch.sum(forward_xy * goal_dir_xy, dim=-1).clamp(
        min=-1.0,
        max=1.0
    )

    # Convert cosine similarity into an angle.
    #
    # For two unit vectors in the XY plane, we can write:
    #
    #     forward_xy = [cos(a), sin(a)]
    #     goal_dir_xy = [cos(b), sin(b)]
    #
    # Their dot product is:
    #
    #     dot = forward_xy · goal_dir_xy
    #         = cos(a) cos(b) + sin(a) sin(b)
    #
    # Using the trigonometric identity:
    #
    #     cos(a - b) = cos(a) cos(b) + sin(a) sin(b)
    #
    # we get:
    #
    #     dot = cos(a - b)
    #
    # Therefore, the angle between the robot heading and the goal direction is:
    #
    #     heading_error = acos(dot)
    #
    # This gives the unsigned heading error in radians:
    #     0      -> facing the goal
    #     pi / 2 -> facing sideways
    #     pi     -> facing away from the goal
    #
    # It is unsigned because cos(+theta) == cos(-theta), so this tells us how large
    # the error is, but not whether the goal is to the left or right.
    return torch.acos(dot)


def _velocity_along_goal_xy(
    env: ManagerBasedRLEnv,
    goal_cfg=SceneEntityCfg("goal"),
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Robot root velocity projected onto the XY goal direction.

    Positive:
        moving toward the goal

    Negative:
        moving away from the goal

    Returns:
        [num_envs]
    """

    goal_dir_xy = _goal_direction_xy(env, goal_cfg, asset_cfg)
    root_vel_xy = _root_lin_vel_xy(env, asset_cfg)

    return torch.sum(root_vel_xy * goal_dir_xy, dim=-1)
