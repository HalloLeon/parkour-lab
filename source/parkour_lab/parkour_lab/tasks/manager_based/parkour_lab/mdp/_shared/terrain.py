from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch

from .runtime import _get_scene_entity_or_none
from .state import _root_pos_env
from .state import _root_height_env
from .. import config


def _base_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Vertical clearance between robot base/root and the surface underneath it.

    Returns:
        [num_envs]
    """

    base_height = _root_height_env(env, asset_cfg)
    surface_height = _support_surface_height_under_base(env, asset_cfg)

    return base_height - surface_height


def _box_surface_height_under_xy(
    env: ManagerBasedRLEnv,
    xy: torch.Tensor,
    box_cfg: config.BoxSurfaceCfg
) -> torch.Tensor:
    """
    Height of a box top surface under a given XY position.

    If the XY position is outside the box footprint, returns -inf.

    Args:
        env: The RL environment.
        xy: Query positions in environment-local XY coordinates, shape [num_envs, 2].
        box_cfg: Box surface configuration.

    Returns:
        [num_envs]
    """

    box = _get_scene_entity_or_none(env, box_cfg.name)

    if box is None:
        return torch.full(
            (xy.shape[0],),
            -torch.inf,
            device=xy.device,
            dtype=xy.dtype
        )

    box_pos_env = box.data.root_pos_w - env.scene.env_origins

    half_size_x = 0.5 * box_cfg.size[0] + box_cfg.xy_margin
    half_size_y = 0.5 * box_cfg.size[1] + box_cfg.xy_margin
    half_size_z = 0.5 * box_cfg.size[2]

    dx = torch.abs(xy[:, 0] - box_pos_env[:, 0])
    dy = torch.abs(xy[:, 1] - box_pos_env[:, 1])

    above_footprint = torch.logical_and(
        dx <= half_size_x,
        dy <= half_size_y
    )

    box_top_height = box_pos_env[:, 2] + half_size_z

    return torch.where(
        above_footprint,
        box_top_height,
        torch.full_like(box_top_height, -torch.inf)
    )


def _support_surface_height_under_base(
    env: ManagerBasedRLEnv,
    asset_cfg=SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Highest support surface directly underneath the robot base/root.

    Currently considers:
      - flat ground
      - optional box obstacle

    Returns:
        [num_envs]
    """

    root_pos = _root_pos_env(env, asset_cfg)
    base_xy = root_pos[:, :2]

    ground_height = torch.full(
        (root_pos.shape[0],),
        config.GROUND_HEIGHT,
        device=root_pos.device,
        dtype=root_pos.dtype
    )

    obstacle_height = _box_surface_height_under_xy(
        env=env,
        xy=base_xy,
        box_cfg=config.OBSTACLE_SURFACE
    )

    return torch.maximum(ground_height, obstacle_height)
