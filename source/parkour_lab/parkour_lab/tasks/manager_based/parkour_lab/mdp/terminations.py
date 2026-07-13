import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ._shared import contact, navigation, terrain
from .commands import get_min_clearance
from .curriculums import episode_outcomes


def base_contact_done(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_contact", body_names="trunk"),
) -> torch.Tensor:
    """
    Base/trunk contact termination.

    Also records a per-env base-contact flag for curriculum updates.
    """

    force_norm = contact._force_norm_mask(env, sensor_cfg=sensor_cfg)

    # [num_envs]
    base_contact = torch.any(force_norm > threshold, dim=(1, 2))

    episode_outcomes.mark_base_contact(env, base_contact)

    return base_contact


def reached_goal_xy_done(
    env: ManagerBasedRLEnv,
    threshold: float,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Success termination based on XY distance to goal,
    while requiring the robot not to be collapsed.

    Returns:
        [num_envs]
    """

    dist_to_goal = navigation._goal_distance_xy(env, goal_cfg, asset_cfg)
    clearance = terrain._base_clearance(env, asset_cfg)

    min_clearance = get_min_clearance(env).to(
        device=clearance.device, dtype=clearance.dtype
    )

    success = torch.logical_and(dist_to_goal < threshold, clearance > min_clearance)

    episode_outcomes.mark_success(env, success)

    return success
