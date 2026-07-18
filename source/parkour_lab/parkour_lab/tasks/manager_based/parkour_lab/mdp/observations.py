import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse

from . import config
from ._shared import contact, terrain
from .commands import get_target_speed
from .navigation import geometry


def base_clearance_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Base/root clearance above the support surface underneath the robot.

    Returns:
        [num_envs, 1]
    """

    return terrain._base_clearance(env, asset_cfg).unsqueeze(-1)


def desired_speed_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Per-environment desired forward speed observation.

    Returns:
        [num_envs, 1]
    """

    return get_target_speed(env).unsqueeze(-1)


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


def goal_direction_body_xy(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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

    goal_vec_xy = geometry._goal_vector_xy(env, goal_cfg, asset_cfg)

    goal_vec_w = torch.zeros((goal_vec_xy.shape[0], 3), device=goal_vec_xy.device, dtype=goal_vec_xy.dtype)
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
    goal_vec_b = quat_apply_inverse(asset.data.root_quat_w, goal_vec_w)
    goal_dir_b_xy = goal_vec_b[:, :2]

    return goal_dir_b_xy / torch.linalg.norm(goal_dir_b_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)


def goal_direction_yaw_xy(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Wrap-safe oracle heading target in the robot's yaw-aligned body frame.

    The unit vector is ``[forward, left]``. It is training-only supervision
    for the student's heading head and must never be concatenated into the
    student policy input.

    Returns:
        [num_envs, 2]
    """

    return geometry._goal_direction_yaw_xy(env, goal_cfg, asset_cfg)


def goal_distance_xy_w(
    env: ManagerBasedRLEnv,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    XY distance from robot root to goal.

    Returns:
        [num_envs, 1]
    """

    return geometry._goal_distance_xy(env, goal_cfg, asset_cfg).unsqueeze(-1)


def student_exteroception_stub(env: ManagerBasedRLEnv, feature_dim: int = 64) -> torch.Tensor:
    """
    Return an information-free placeholder for a future depth embedding.

    This configurable-width zero tensor establishes the student exteroception
    API without exposing ray hits or other simulator geometry. It is suitable
    only for testing the pipeline. The future depth encoder may deliberately
    choose a different feature width, which will create a new student model
    interface while leaving the action contract unchanged.

    Returns:
        [num_envs, feature_dim]
    """

    if feature_dim <= 0:
        raise ValueError("feature_dim must be positive.")
    return torch.zeros((env.num_envs, feature_dim), device=env.device)


def terrain_height_scan(
    env: ManagerBasedRLEnv,
    obs_cfg: config.HeightScanObservationCfg = config.DEFAULT_HEIGHT_SCAN_OBSERVATION,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Fixed-size privileged terrain-height scan for the Phase 1 teacher.

    The configured ray caster is required. Failing when it is absent prevents
    training a supposedly terrain-aware teacher on an accidental all-zero
    terrain input. Heights are clipped in metres using ``obs_cfg.clip`` and
    then divided by that fixed bound, producing values in ``[-1, 1]``.

    A value of zero places the surface at ``root_z - vertical_offset``.
    Negative values represent surfaces above that reference plane; positive
    values represent surfaces below it. Missing hits use the deterministic
    value ``+1``; consume :func:`terrain_height_scan_validity` alongside this
    term to distinguish them from genuinely clipped-low surfaces.

    Returns:
        Normalized heights with shape ``[num_envs, obs_cfg.num_rays]``.
    """

    heights, _ = _terrain_height_scan_components(
        env,
        obs_cfg=obs_cfg,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    )
    return heights


def terrain_height_scan_validity(
    env: ManagerBasedRLEnv,
    obs_cfg: config.HeightScanObservationCfg = config.DEFAULT_HEIGHT_SCAN_OBSERVATION,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return a fixed-size floating mask identifying finite terrain-ray hits.

    ``1`` denotes a valid surface hit and ``0`` denotes a missing or otherwise
    non-finite hit. The mask has the same stable ray ordering as
    :func:`terrain_height_scan`.

    Returns:
        Validity mask with shape ``[num_envs, obs_cfg.num_rays]``.
    """

    _, validity = _terrain_height_scan_components(
        env,
        obs_cfg=obs_cfg,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
    )
    return validity


def _terrain_height_scan_components(
    env: ManagerBasedRLEnv,
    obs_cfg: config.HeightScanObservationCfg = config.DEFAULT_HEIGHT_SCAN_OBSERVATION,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read and process the configured privileged terrain ray caster."""

    sensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    if not isinstance(sensor, RayCaster):
        raise TypeError(f"Expected '{sensor_cfg.name}' to be a RayCaster, got {type(sensor).__name__}.")

    return terrain._terrain_height_components(
        asset.data.root_pos_w[:, 2],
        sensor.data.ray_hits_w,
        num_rays=obs_cfg.num_rays,
        vertical_offset=obs_cfg.vertical_offset,
        clip=obs_cfg.clip,
    )
