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
    reach_hold_s: float,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Advance active waypoints and terminate only after the safe final one.

    Returns:
        [num_envs]
    """

    return advance_active_waypoints(
        env,
        reach_threshold=reach_threshold,
        reach_hold_s=reach_hold_s,
        goal_cfg=goal_cfg,
        asset_cfg=asset_cfg,
    )
