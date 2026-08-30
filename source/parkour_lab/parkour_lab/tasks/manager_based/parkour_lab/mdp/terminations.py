import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ._shared import contact, robot
from .commands import active_motion_time_s
from .navigation import route

# Time budget.


def active_motion_time_out(
    env: ManagerBasedRLEnv, max_active_motion_time_s: float = 25.0
) -> torch.Tensor:
    """Terminate after a fixed amount of commanded translation or pivot time.

    Fully stationary command windows pause this task budget. The environment's
    ordinary wall-clock timeout remains a separate hard safety cap.
    """

    return active_motion_time_s(env) >= max_active_motion_time_s


# Safety failures.


def chassis_contact_done(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("chassis_contact"),
) -> torch.Tensor:
    """Return which environments exceed the fatal chassis-contact threshold."""

    return torch.any(
        contact._force_norm_mask(env, sensor_cfg=sensor_cfg) > threshold,
        dim=(1, 2),
    )


def fell_below_course(
    env: ManagerBasedRLEnv,
    minimum_height: float = -0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate robots that fall far below their environment-local course."""

    root_height = robot._root_height_env(env, asset_cfg)
    return (~torch.isfinite(root_height)) | (root_height < minimum_height)


def off_route(
    env: ManagerBasedRLEnv,
    hard_half_width_m: float,
) -> torch.Tensor:
    """Terminate when the root leaves the finite approved route envelope."""

    error = route.route_cross_track_error_m(env)
    return (~torch.isfinite(error)) | (error > hard_half_width_m)


# Successful route completion.


def completed_course_done(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    feet_asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot",
        body_names=".*_foot",
    ),
    feet_contact_cfg: SceneEntityCfg = SceneEntityCfg(
        "feet_contact",
        body_names=".*_foot",
    ),
    chassis_contact_cfg: SceneEntityCfg = SceneEntityCfg("chassis_contact"),
    contact_threshold: float = 1.0,
    max_completion_tilt_sine: float = 0.5,
    max_completion_vertical_speed_m_s: float = 0.5,
    support_margin: float = 0.05,
    support_plane_tolerance: float = 0.12,
    terminal_support_load_threshold_n: float = 10.0,
    terminal_min_support_feet: int = 2,
    terminal_stability_dwell_s: float = 0.2,
    terminal_max_planar_speed_m_s: float = 0.2,
    terminal_max_vertical_speed_m_s: float = 0.2,
    terminal_max_yaw_rate_rad_s: float = 0.35,
    terminal_max_roll_pitch_rate_rad_s: float = 0.35,
    terminal_max_tilt_sine: float = 0.25,
    progress_route_half_width_m: float = 0.2,
    hard_route_half_width_m: float | None = None,
) -> torch.Tensor:
    """Advance routes and terminate only on a supported, stable, crash-free finish.

    Returns:
        [num_envs]
    """

    return route.advance_active_waypoints(
        env,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
        feet_asset_cfg=feet_asset_cfg,
        feet_contact_cfg=feet_contact_cfg,
        chassis_contact_cfg=chassis_contact_cfg,
        contact_threshold=contact_threshold,
        max_completion_tilt_sine=max_completion_tilt_sine,
        max_completion_vertical_speed_m_s=max_completion_vertical_speed_m_s,
        support_margin=support_margin,
        support_plane_tolerance=support_plane_tolerance,
        terminal_support_load_threshold_n=terminal_support_load_threshold_n,
        terminal_min_support_feet=terminal_min_support_feet,
        terminal_stability_dwell_s=terminal_stability_dwell_s,
        terminal_max_planar_speed_m_s=terminal_max_planar_speed_m_s,
        terminal_max_vertical_speed_m_s=terminal_max_vertical_speed_m_s,
        terminal_max_yaw_rate_rad_s=terminal_max_yaw_rate_rad_s,
        terminal_max_roll_pitch_rate_rad_s=terminal_max_roll_pitch_rate_rad_s,
        terminal_max_tilt_sine=terminal_max_tilt_sine,
        progress_route_half_width_m=progress_route_half_width_m,
        hard_route_half_width_m=hard_route_half_width_m,
    )
