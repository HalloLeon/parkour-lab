import torch
from isaaclab.assets import AssetBase
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .state import _root_forward_xy_w, _root_lin_vel_xy, _root_pos_env


def _goal_direction_xy(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Unit XY direction from robot root to goal in world-aligned axes.

    Positions are expressed relative to each environment origin before
    subtraction, but environment origins are translations only. The resulting
    direction therefore retains the world-frame XY orientation.

    Returns:
        [num_envs, 2]
    """

    goal_vec_xy = _goal_vector_xy(env, goal_cfg, asset_cfg)

    goal_dist_xy = torch.linalg.norm(goal_vec_xy, dim=-1, keepdim=True).clamp_min(
        1.0e-6
    )

    return goal_vec_xy / goal_dist_xy


def _goal_direction_yaw_xy(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Unit XY goal direction in the robot's yaw-aligned body frame.

    The first component points forward and the second points left. Roll and
    pitch do not affect this heading target, so it is equivalent to
    ``[cos(heading_error), sin(heading_error)]`` without an angle wrap at
    ``-pi`` or ``+pi``.

    Returns:
        [num_envs, 2]
    """

    forward_direction_w = _root_forward_xy_w(env, asset_cfg)
    forward_norm = torch.linalg.norm(forward_direction_w, dim=-1, keepdim=True)
    # A vertical body has no projected forward direction. World +X provides a
    # deterministic yaw convention for this otherwise undefined edge case.
    fallback_forward_w = torch.zeros_like(forward_direction_w)
    fallback_forward_w[:, 0] = 1.0
    forward_direction_w = torch.where(
        forward_norm > 1.0e-6,
        forward_direction_w / forward_norm,
        fallback_forward_w,
    )
    goal_vector_w = _goal_vector_xy(env, goal_cfg, asset_cfg)
    goal_distance = torch.linalg.norm(goal_vector_w, dim=-1, keepdim=True)

    # Heading is undefined exactly at the waypoint. Use body-forward as a
    # deterministic, unit-length target for that degenerate state.
    goal_direction_w = torch.where(
        goal_distance > 1.0e-6,
        goal_vector_w / goal_distance,
        forward_direction_w,
    )

    # Rotate each world-frame forward vector ``[x, y]`` by 90 degrees
    # counterclockwise to obtain the corresponding unit left vector
    # ``[-y, x]``. Together they form the robot's yaw-aligned XY basis.
    left_direction_w = torch.stack(
        (-forward_direction_w[:, 1], forward_direction_w[:, 0]), dim=-1
    )

    # Project the world-frame goal direction onto that basis. The first dot
    # product is the forward component (cosine of the heading error), and the
    # second is the signed left component (sine of the heading error).
    return torch.stack(
        (
            torch.sum(goal_direction_w * forward_direction_w, dim=-1),
            torch.sum(goal_direction_w * left_direction_w, dim=-1),
        ),
        dim=-1,
    )


def _goal_distance_xy(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    XY distance from robot root to goal.

    Returns:
        [num_envs]
    """

    to_goal_xy = _goal_vector_xy(env, goal_cfg, asset_cfg)
    return torch.linalg.norm(to_goal_xy, dim=-1)


def _goal_pos_env(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
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
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    XYZ vector from robot root to goal.

    Returns:
        [num_envs, 3]
    """

    robot_root_pos = _root_pos_env(env, asset_cfg)
    goal_pos = _goal_pos_env(env, goal_cfg)

    return goal_pos - robot_root_pos


def _heading_error_to_goal_xy(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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
    dot = torch.sum(forward_xy * goal_dir_xy, dim=-1).clamp(min=-1.0, max=1.0)

    # Convert cosine similarity into an angle.
    #
    # Any 2D vector can be described in polar coordinates by its length r and
    # its angle from the positive X axis:
    #
    #     [x, y] = [r cos(angle), r sin(angle)]
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


def _lateral_drift_to_goal_xy(
    env: ManagerBasedRLEnv,
    *,
    root_delta_xy: torch.Tensor,
    goal_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Lateral part of root displacement relative to the current goal direction.

    Pure motion toward the goal has zero lateral drift.

    Returns:
        [num_envs]
    """

    # Let d be the robot's XY displacement and g the unit vector pointing
    # toward the goal. The dot product d · g is the signed scalar amount of d
    # along g: positive means toward the goal and negative means away from it.
    goal_dir_xy = _goal_direction_xy(env, goal_cfg=goal_cfg, asset_cfg=asset_cfg)

    # keepdim=True keeps the result shaped [num_envs, 1], allowing it to
    # broadcast across the X and Y components of goal_dir_xy [num_envs, 2].
    forward_delta = torch.sum(root_delta_xy * goal_dir_xy, dim=-1, keepdim=True)

    # Because g has unit length, (d · g)g is the vector projection of d onto
    # the goal direction. Subtracting that parallel component from d leaves
    # only the perpendicular component:
    #
    #     d_lateral = d - (d · g)g
    #
    # If the robot moves exactly toward or away from the goal, d is parallel
    # to g and this residual is zero.
    lateral_delta_xy = root_delta_xy - forward_delta * goal_dir_xy

    # The Euclidean norm converts the lateral XY vector into one non-negative
    # drift magnitude per environment.
    return torch.linalg.norm(lateral_delta_xy, dim=-1)


def _velocity_along_goal_xy(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    World-frame root velocity projected onto the world-aligned XY goal direction.

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
