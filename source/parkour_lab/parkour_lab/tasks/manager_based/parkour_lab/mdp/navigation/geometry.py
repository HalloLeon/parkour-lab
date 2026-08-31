import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .._shared.robot import _root_forward_xy_w, _root_lin_vel_xy, _root_pos_env
from . import route
from .route import active_waypoint_positions, final_waypoint_positions

# Active-waypoint geometry.


def _active_waypoint_direction_xy(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the world-aligned unit XY direction to the active waypoint."""

    waypoint_vector_xy = _active_waypoint_guidance_vector_xy(
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

    direction, _ = _world_vector_to_yaw_direction_xy(
        env,
        _active_waypoint_guidance_vector_xy(env, waypoint_marker_cfg, asset_cfg),
        asset_cfg,
    )
    return direction


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


def _active_waypoint_guidance_vector_xy(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return point guidance, with non-reversing terminal arrival guidance.

    Ordinary targets use their direct point bearing. Within the last root-reach
    radius of a terminal segment, its longitudinal component is held forward
    while lateral error remains corrective. Passing the marker therefore cannot
    flip the oracle direction by 180 degrees during the settling dwell.
    """

    point_vector = _active_waypoint_vector_xy(env, waypoint_marker_cfg, asset_cfg)
    terminal = route.active_waypoint_is_terminal_landing(env)
    inbound = route.active_waypoint_inbound_direction_xy(env).to(
        device=point_vector.device,
        dtype=point_vector.dtype,
    )
    longitudinal = torch.sum(point_vector * inbound, dim=-1)
    reach_radius = route.active_waypoint_root_reach_radii(env).to(
        device=point_vector.device,
        dtype=point_vector.dtype,
    )
    terminal_vector = (
        point_vector + torch.relu(reach_radius - longitudinal).unsqueeze(-1) * inbound
    )
    inside_reach_circle = torch.linalg.norm(point_vector, dim=-1) <= reach_radius
    return torch.where(
        (terminal & inside_reach_circle).unsqueeze(-1),
        terminal_vector,
        point_vector,
    )


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


# Reference-direction geometry.


def _final_waypoint_direction_yaw_xy_components(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return final-goal yaw direction and a non-zero-distance validity mask."""

    root_xy = _root_pos_env(env, asset_cfg)[:, :2]
    final_vector_w = final_waypoint_positions(env)[:, :2] - root_xy
    return _world_vector_to_yaw_direction_xy(env, final_vector_w, asset_cfg)


def _world_vector_to_yaw_direction_xy(
    env: ManagerBasedRLEnv,
    vector_w: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a world XY vector and express it as yaw-frame forward/left."""

    forward_direction_w = _root_forward_xy_w(env, asset_cfg)
    forward_norm = torch.linalg.norm(forward_direction_w, dim=-1, keepdim=True)
    fallback_forward_w = torch.zeros_like(forward_direction_w)
    fallback_forward_w[:, 0] = 1.0
    safe_forward_norm = torch.where(
        forward_norm > 1.0e-6, forward_norm, torch.ones_like(forward_norm)
    )
    forward_direction_w = torch.where(
        forward_norm > 1.0e-6,
        forward_direction_w / safe_forward_norm,
        fallback_forward_w,
    )
    vector_distance = torch.linalg.norm(
        vector_w,
        dim=-1,
        keepdim=True,
    )
    valid = vector_distance[:, 0] > 1.0e-6
    safe_vector_distance = torch.where(
        vector_distance > 1.0e-6,
        vector_distance,
        torch.ones_like(vector_distance),
    )
    direction_w = torch.where(
        valid[:, None],
        vector_w / safe_vector_distance,
        forward_direction_w,
    )
    left_direction_w = torch.stack(
        (-forward_direction_w[:, 1], forward_direction_w[:, 0]),
        dim=-1,
    )
    return (
        torch.stack(
            (
                torch.sum(direction_w * forward_direction_w, dim=-1),
                torch.sum(direction_w * left_direction_w, dim=-1),
            ),
            dim=-1,
        ),
        valid,
    )
