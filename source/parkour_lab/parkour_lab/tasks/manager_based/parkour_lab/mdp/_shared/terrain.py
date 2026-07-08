from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch

from .state import _root_pos_env
from .state import _root_height_env
from .. import config
from ..commands import get_obstacle_pos
from ..commands import get_obstacle_size


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
    xy_margin: float = 0.02
) -> torch.Tensor:
    """
    Height of the active curriculum obstacle under XY.

    The obstacle is baked into terrain, but its active metadata is stored in
    per-env command buffers.
    """

    obstacle_pos = get_obstacle_pos(env).to(device=xy.device, dtype=xy.dtype)
    obstacle_size = get_obstacle_size(env).to(device=xy.device, dtype=xy.dtype)

    half_size_x = 0.5 * obstacle_size[:, 0] + xy_margin
    half_size_y = 0.5 * obstacle_size[:, 1] + xy_margin
    half_size_z = 0.5 * obstacle_size[:, 2]

    dx = torch.abs(xy[:, 0] - obstacle_pos[:, 0])
    dy = torch.abs(xy[:, 1] - obstacle_pos[:, 1])

    above_footprint = torch.logical_and(
        dx <= half_size_x,
        dy <= half_size_y
    )

    obstacle_top_height = obstacle_pos[:, 2] + half_size_z

    return torch.where(
        above_footprint,
        obstacle_top_height,
        torch.full_like(obstacle_top_height, -torch.inf)
    )


def _support_surface_height_under_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Support-surface height under the robot base.

    This mirrors the active curriculum terrain metadata.
    """

    root_pos_env = _root_pos_env(env, asset_cfg)

    ground_height = torch.full(
        (env.num_envs,),
        config.GROUND_HEIGHT,
        device=root_pos_env.device,
        dtype=root_pos_env.dtype
    )

    obstacle_height = _box_surface_height_under_xy(
        env,
        xy=root_pos_env[:, :2],
        xy_margin=0.02
    )

    return torch.maximum(ground_height, obstacle_height)
