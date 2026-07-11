import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_apply_inverse

from . import config
from ._shared import contact, navigation, runtime, terrain
from .commands import get_target_speed


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

    return navigation._goal_distance_xy(env, goal_cfg, asset_cfg).unsqueeze(-1)


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

    goal_vec_xy = navigation._goal_vector_xy(env, goal_cfg, asset_cfg)

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


def base_clearance_obs(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Base/root clearance above the support surface underneath the robot.

    Returns:
        [num_envs, 1]
    """

    return terrain._base_clearance(env, asset_cfg).unsqueeze(-1)


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


def desired_speed_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Per-environment desired forward speed observation.

    Returns:
        [num_envs, 1]
    """

    return get_target_speed(env).unsqueeze(-1)


def height_scan_or_zeros(
    env: ManagerBasedRLEnv,
    obs_cfg: config.HeightScanObservationCfg = config.DEFAULT_HEIGHT_SCAN_OBSERVATION,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Fixed-size terrain/obstacle height scan.

    If the height scanner is not present, returns zeros with the expected size.
    This keeps observation dimensions stable across simple and terrain-rich
    environments.

    Intended for critic/teacher use, not the first deployable actor.

    Returns:
        [num_envs, obs_cfg.num_rays]
    """

    sensor = runtime._get_scene_entity_or_none(env, sensor_cfg.name)
    asset: Articulation = env.scene[asset_cfg.name]

    if sensor is None:
        return torch.zeros(
            (env.num_envs, obs_cfg.num_rays), device=asset.data.root_pos_w.device, dtype=asset.data.root_pos_w.dtype
        )

    if not isinstance(sensor, RayCaster):
        raise TypeError(f"Expected '{sensor_cfg.name}' to be a RayCaster, got {type(sensor).__name__}.")

    root_z = asset.data.root_pos_w[:, 2].unsqueeze(-1)
    hit_z = sensor.data.ray_hits_w[..., 2]

    if hit_z.shape[-1] != obs_cfg.num_rays:
        raise RuntimeError(f"Height scan expected {obs_cfg.num_rays} rays, but sensor returned {hit_z.shape[-1]} rays.")

    heights = root_z - obs_cfg.vertical_offset - hit_z

    heights = torch.nan_to_num(heights, nan=0.0, posinf=obs_cfg.clip, neginf=-obs_cfg.clip)

    return torch.clamp(heights, min=-obs_cfg.clip, max=obs_cfg.clip)
