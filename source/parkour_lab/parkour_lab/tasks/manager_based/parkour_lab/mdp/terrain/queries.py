# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Terrain queries and observation preprocessing."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from .._shared.robot import _root_height_env


def _base_clearance_components(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_height_scanner"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite base clearance and the downward-ray validity mask."""

    sensor = env.scene[sensor_cfg.name]
    if not isinstance(sensor, RayCaster):
        raise TypeError(
            f"Expected '{sensor_cfg.name}' to be a RayCaster, got {type(sensor).__name__}."
        )

    ray_hits_w = sensor.data.ray_hits_w
    if ray_hits_w.shape[1] != 1:
        raise RuntimeError(
            f"'{sensor_cfg.name}' must contain exactly one downward ray."
        )

    base_height = _root_height_env(env, asset_cfg)
    surface_height = ray_hits_w[:, 0, 2] - env.scene.env_origins[:, 2]
    clearance = base_height - surface_height
    valid_hit = torch.isfinite(ray_hits_w[:, 0, :]).all(dim=-1) & torch.isfinite(
        clearance
    )

    finite_clearance = torch.where(valid_hit, clearance, torch.zeros_like(clearance))
    return finite_clearance, valid_hit


def _terrain_height_components(
    root_z: torch.Tensor,
    ray_hits_w: torch.Tensor,
    *,
    num_rays: int,
    vertical_offset: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized heights and validity, each shaped ``[E, R]``.

    Missing hits use normalized height ``+1`` and validity ``0``. Numeric
    configuration is validated once by :class:`HeightScanObservationCfg`.
    """

    expected_shape = (root_z.shape[0], num_rays, 3) if root_z.ndim == 1 else None
    if tuple(ray_hits_w.shape) != expected_shape:
        raise ValueError(
            f"Expected root_z [E] and ray_hits_w [E, {num_rays}, 3], got "
            f"{tuple(root_z.shape)} and {tuple(ray_hits_w.shape)}."
        )

    valid_hits = torch.isfinite(ray_hits_w).all(dim=-1)
    heights_m = root_z.unsqueeze(-1) - vertical_offset - ray_hits_w[..., 2]
    valid_hits &= torch.isfinite(heights_m)
    finite_heights_m = torch.where(valid_hits, heights_m, clip)
    normalized_heights = torch.clamp(finite_heights_m, min=-clip, max=clip) / clip
    return normalized_heights, valid_hits.to(dtype=normalized_heights.dtype)
