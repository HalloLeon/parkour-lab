import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ._shared import contact
from .navigation.route import advance_active_waypoints


def base_contact_done(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_contact", body_names="trunk"),
) -> torch.Tensor:
    """Return which environments exceed the trunk-contact force threshold."""

    force_norm = contact._force_norm_mask(env, sensor_cfg=sensor_cfg)

    # [num_envs]
    base_contact = torch.any(force_norm > threshold, dim=(1, 2))

    return base_contact


def completed_course_done(
    env: ManagerBasedRLEnv,
    reach_threshold: float,
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
    trunk_contact_cfg: SceneEntityCfg = SceneEntityCfg(
        "base_contact",
        body_names="trunk",
    ),
    contact_threshold: float = 1.0,
    max_completion_tilt: float = 0.5,
    max_completion_vertical_speed: float = 0.5,
    support_margin: float = 0.05,
    support_plane_tolerance: float = 0.12,
) -> torch.Tensor:
    """Advance routes and terminate only on a supported, stable, crash-free finish.

    Returns:
        [num_envs]
    """

    return advance_active_waypoints(
        env,
        reach_threshold=reach_threshold,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
        feet_asset_cfg=feet_asset_cfg,
        feet_contact_cfg=feet_contact_cfg,
        trunk_contact_cfg=trunk_contact_cfg,
        contact_threshold=contact_threshold,
        max_completion_tilt=max_completion_tilt,
        max_completion_vertical_speed=max_completion_vertical_speed,
        support_margin=support_margin,
        support_plane_tolerance=support_plane_tolerance,
    )
