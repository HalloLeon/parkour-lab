import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .. import config
from ..commands import get_min_clearance
from .terrain import _base_clearance


def _root_stability_mask(
    env: ManagerBasedRLEnv, stability_cfg: config.RootStabilityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Check whether the robot root is stable and sufficiently clear of the
    support surface underneath it.

    Stability requires:
      - limited roll/pitch angular velocity,
      - limited roll/pitch tilt,
      - enough base/root clearance above the current support surface.

    The support surface may be:
      - flat ground,
      - obstacle top,
      - later another terrain/platform surface.

    Returns:
        [num_envs]
    """

    asset: Articulation = env.scene[asset_cfg.name]

    # Roll/pitch angular speed.
    # [num_envs]
    roll_pitch_ang_speed = torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=-1)

    # Projected gravity x/y norm is small when the base is upright.
    # [num_envs]
    projected_gravity_xy_norm = torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=-1)

    # Clearance above whatever support surface is underneath the base.
    # This is not raw world height.
    # [num_envs]
    base_clearance = _base_clearance(env, asset_cfg=asset_cfg)

    ang_vel_stable = roll_pitch_ang_speed < stability_cfg.max_roll_pitch_ang_speed

    orientation_stable = projected_gravity_xy_norm < stability_cfg.max_projected_gravity_xy_norm

    min_clearance = get_min_clearance(env).to(device=base_clearance.device, dtype=base_clearance.dtype)

    clearance_stable = base_clearance > min_clearance

    return torch.logical_and(torch.logical_and(ang_vel_stable, orientation_stable), clearance_stable)
