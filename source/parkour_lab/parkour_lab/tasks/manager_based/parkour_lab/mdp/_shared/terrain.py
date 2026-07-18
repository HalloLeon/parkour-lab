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

from .robot import _root_height_env


def _base_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_height_scanner"),
) -> torch.Tensor:
    """Return vertical root clearance above the underlying surface.

    Returns:
        Clearance with shape ``[num_envs]``.
    """

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

    # A missed ray is unsafe, but it should not inject non-finite values into
    # rewards or observations during a temporary out-of-range state.
    return torch.where(
        torch.isfinite(clearance), clearance, torch.zeros_like(clearance)
    )


def _terrain_height_components(
    root_z: torch.Tensor,
    ray_hits_w: torch.Tensor,
    *,
    num_rays: int,
    vertical_offset: float,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ray hits into normalized robot-relative heights and validity.

    Missing hits use the deterministic normalized height value ``+1`` and are
    distinguished from valid, clipped-low surfaces by a zero in the mask.

    Args:
        root_z: Robot-root world height with shape ``[num_envs]``.
        ray_hits_w: World-frame ray-hit positions with shape
            ``[num_envs, num_rays, 3]``.
        num_rays: Required fixed number of rays.
        vertical_offset: Distance from the root to the height reference plane.
        clip: Symmetric metric clipping bound and normalization divisor.

    Returns:
        A pair ``(normalized_heights, validity_mask)``, each with shape
        ``[num_envs, num_rays]``. Heights lie in ``[-1, 1]`` and the floating
        mask contains only zero or one.
    """

    if num_rays <= 0:
        raise ValueError("num_rays must be positive.")
    if clip <= 0.0:
        raise ValueError("clip must be positive.")
    if root_z.ndim != 1:
        raise ValueError(
            f"root_z must have shape [num_envs], got {tuple(root_z.shape)}."
        )
    if ray_hits_w.ndim != 3 or ray_hits_w.shape[-1] != 3:
        raise ValueError(
            "ray_hits_w must have shape [num_envs, num_rays, 3], "
            f"got {tuple(ray_hits_w.shape)}."
        )
    if ray_hits_w.shape[0] != root_z.shape[0]:
        raise ValueError(
            "root_z and ray_hits_w must have the same environment dimension, "
            f"got {root_z.shape[0]} and {ray_hits_w.shape[0]}."
        )
    if ray_hits_w.shape[1] != num_rays:
        raise RuntimeError(
            f"Height scan expected {num_rays} rays, but sensor returned {ray_hits_w.shape[1]} rays."
        )

    # A hit is valid only when all three world-frame coordinates are finite.
    # Ray casters use non-finite coordinates to represent missing hits.
    valid_hits = torch.isfinite(ray_hits_w).all(dim=-1)

    # Positive heights represent surfaces below the reference plane, while
    # negative heights represent surfaces above it.
    heights_m = root_z.unsqueeze(-1) - vertical_offset - ray_hits_w[..., 2]

    # This also invalidates every ray associated with a non-finite root height
    # and catches non-finite results caused by floating-point overflow.
    valid_hits &= torch.isfinite(heights_m)

    # Encode a missing hit at the lower clipping boundary while preserving the
    # distinction through the validity mask.
    missing_height_m = torch.full_like(heights_m, clip)
    finite_heights_m = torch.where(valid_hits, heights_m, missing_height_m)
    normalized_heights = torch.clamp(finite_heights_m, min=-clip, max=clip) / clip
    validity_mask = valid_hits.to(dtype=normalized_heights.dtype)

    return normalized_heights, validity_mask
