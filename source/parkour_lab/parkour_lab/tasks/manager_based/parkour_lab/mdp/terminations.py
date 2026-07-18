import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ._shared import contact
from .curriculums import episode_outcomes
from .navigation.route import advance_active_waypoints


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

    success = advance_active_waypoints(
        env,
        reach_threshold=reach_threshold,
        reach_hold_s=reach_hold_s,
        goal_cfg=goal_cfg,
        asset_cfg=asset_cfg,
    )
    episode_outcomes.mark_success(env, success)
    return success
