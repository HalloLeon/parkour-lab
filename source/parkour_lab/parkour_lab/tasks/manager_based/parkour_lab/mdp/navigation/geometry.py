import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .._shared.robot import _root_forward_xy_w, _root_lin_vel_xy, _root_pos_env
from .route import active_waypoint_positions


def _active_waypoint_direction_xy(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the world-aligned unit XY direction to the active waypoint."""

    waypoint_vector_xy = _active_waypoint_vector_xy(
        env,
        waypoint_marker_cfg,
        asset_cfg,
    )
    waypoint_distance_xy = torch.linalg.norm(
        waypoint_vector_xy,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-6)
    return waypoint_vector_xy / waypoint_distance_xy


def _active_waypoint_direction_yaw_xy(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return active-waypoint direction in the robot's yaw-aligned XY frame."""

    forward_direction_w = _root_forward_xy_w(env, asset_cfg)
    forward_norm = torch.linalg.norm(forward_direction_w, dim=-1, keepdim=True)
    fallback_forward_w = torch.zeros_like(forward_direction_w)
    fallback_forward_w[:, 0] = 1.0
    forward_direction_w = torch.where(
        forward_norm > 1.0e-6,
        forward_direction_w / forward_norm,
        fallback_forward_w,
    )
    waypoint_vector_w = _active_waypoint_vector_xy(
        env,
        waypoint_marker_cfg,
        asset_cfg,
    )
    waypoint_distance = torch.linalg.norm(
        waypoint_vector_w,
        dim=-1,
        keepdim=True,
    )
    waypoint_direction_w = torch.where(
        waypoint_distance > 1.0e-6,
        waypoint_vector_w / waypoint_distance,
        forward_direction_w,
    )
    left_direction_w = torch.stack(
        (-forward_direction_w[:, 1], forward_direction_w[:, 0]),
        dim=-1,
    )
    return torch.stack(
        (
            torch.sum(waypoint_direction_w * forward_direction_w, dim=-1),
            torch.sum(waypoint_direction_w * left_direction_w, dim=-1),
        ),
        dim=-1,
    )


def _active_waypoint_distance_xy(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return XY distance from the robot root to the active waypoint."""

    waypoint_vector_xy = _active_waypoint_vector_xy(
        env,
        waypoint_marker_cfg,
        asset_cfg,
    )
    return torch.linalg.norm(waypoint_vector_xy, dim=-1)


def _active_waypoint_position_env(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
) -> torch.Tensor:
    """Return the active-waypoint position in each environment's local frame."""

    return active_waypoint_positions(env, waypoint_marker_cfg)


def _active_waypoint_vector_xy(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the XY vector from the robot root to the active waypoint."""

    return _active_waypoint_vector_xyz(
        env,
        waypoint_marker_cfg,
        asset_cfg,
    )[:, :2]


def _active_waypoint_vector_xyz(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the XYZ vector from the robot root to the active waypoint."""

    robot_root_position = _root_pos_env(env, asset_cfg)
    waypoint_position = _active_waypoint_position_env(
        env,
        waypoint_marker_cfg,
    )
    return waypoint_position - robot_root_position


def _heading_error_to_active_waypoint_xy(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return unsigned heading error to the active waypoint in radians."""

    forward_xy = _root_forward_xy_w(env, asset_cfg)
    waypoint_direction_xy = _active_waypoint_direction_xy(
        env,
        waypoint_marker_cfg,
        asset_cfg,
    )
    cosine = torch.sum(
        forward_xy * waypoint_direction_xy,
        dim=-1,
    ).clamp(min=-1.0, max=1.0)
    return torch.acos(cosine)


def _lateral_drift_to_active_waypoint_xy(
    env: ManagerBasedRLEnv,
    *,
    root_delta_xy: torch.Tensor,
    waypoint_marker_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return root displacement perpendicular to active-waypoint direction."""

    waypoint_direction_xy = _active_waypoint_direction_xy(
        env,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )
    forward_delta = torch.sum(
        root_delta_xy * waypoint_direction_xy,
        dim=-1,
        keepdim=True,
    )
    lateral_delta_xy = root_delta_xy - forward_delta * waypoint_direction_xy
    return torch.linalg.norm(lateral_delta_xy, dim=-1)


def _velocity_along_active_waypoint_xy(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Project world-frame root velocity onto active-waypoint direction."""

    waypoint_direction_xy = _active_waypoint_direction_xy(
        env,
        waypoint_marker_cfg,
        asset_cfg,
    )
    root_velocity_xy = _root_lin_vel_xy(env, asset_cfg)
    return torch.sum(root_velocity_xy * waypoint_direction_xy, dim=-1)
