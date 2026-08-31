import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from . import config
from ._shared import contact
from .commands import (
    get_target_speed,
    get_target_yaw_rate,
)
from .domain_randomization import PrivilegedDynamicsRecorder
from .navigation import geometry, route
from .terrain import queries

# Robot state and course commands.


def base_clearance_obs(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Base/root clearance above the support surface underneath the robot.

    Returns:
        [num_envs, 1]
    """

    return queries._base_clearance_components(env, asset_cfg)[0].unsqueeze(-1)


def desired_speed_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Per-environment route-conditioned translational target speed.

    Returns:
        [num_envs, 1]
    """

    return get_target_speed(env).unsqueeze(-1)


def desired_yaw_rate_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the signed in-place yaw-rate command in radians per second."""

    return get_target_yaw_rate(env).unsqueeze(-1)


def foot_contact_state(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_contact", body_names=".*_foot"),
) -> torch.Tensor:
    """
    Foot contact state, centered.

    Official-style convention:
        no contact -> -0.5
        contact    ->  0.5

    Returns:
        [num_envs, num_feet]
    """

    force_norm = contact._force_norm_mask(env, sensor_cfg=sensor_cfg)

    in_contact = torch.any(force_norm > threshold, dim=1)

    return in_contact.float() - 0.5


# Route state.


def active_waypoint_direction_yaw_xy(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Wrap-safe direction to the active waypoint in the robot's yaw-aligned frame.

    The unit vector is ``[forward, left]`` and supplies the privileged teacher's
    oracle travel-direction input.

    Returns:
        [num_envs, 2]
    """

    return geometry._active_waypoint_direction_yaw_xy(
        env, waypoint_marker_cfg, asset_cfg
    )


def active_waypoint_distance_xy(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    XY distance from robot root to the active waypoint.

    Returns:
        [num_envs, 1]
    """

    return geometry._active_waypoint_distance_xy(
        env,
        waypoint_marker_cfg,
        asset_cfg,
    ).unsqueeze(-1)


def route_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return normalized route-cursor and safe-progress phase for the critic.

    The first component locates the active waypoint in the ordered route. The
    second reports the greatest corridor-valid route progress divided by the
    selected course length. Both components are fixed to ``[0, 1]`` and expose
    task phase without leaking obstacle-family or difficulty labels.

    Returns:
        [num_envs, 2]
    """

    return route.route_phase(env)


# Privileged teacher observations.


def recorded_privileged_dynamics(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the persistent randomized dynamics cached at startup."""

    recorder = env.event_manager.get_term_cfg("record_privileged_dynamics").func
    if not isinstance(recorder, PrivilegedDynamicsRecorder):
        raise TypeError(
            "record_privileged_dynamics must use PrivilegedDynamicsRecorder."
        )
    return recorder.values


def terrain_height_scan(
    env: ManagerBasedRLEnv,
    obs_cfg: config.HeightScanObservationCfg = config.DEFAULT_HEIGHT_SCAN_OBSERVATION,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return one fixed-size privileged terrain scan for the Phase 1 teacher.

    The configured ray caster is required. Failing when it is absent prevents
    training a supposedly terrain-aware teacher on an accidental all-zero
    terrain input. The first ``num_rays`` entries are normalized heights and
    the remaining entries are their floating validity mask. Concatenating them
    here reads and preprocesses the ray caster only once per observation.

    Returns:
        Heights followed by validity with shape ``[num_envs, 2 * num_rays]``.
    """

    sensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if not isinstance(sensor, RayCaster):
        raise TypeError(
            f"Expected '{sensor_cfg.name}' to be a RayCaster, got {type(sensor).__name__}."
        )

    heights, validity = queries._terrain_height_components(
        asset.data.root_pos_w[:, 2],
        sensor.data.ray_hits_w,
        num_rays=obs_cfg.num_rays,
        vertical_offset=obs_cfg.vertical_offset,
        clip=obs_cfg.clip,
    )
    return torch.cat((heights, validity), dim=-1)
