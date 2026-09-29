"""Privileged terrain scan preprocessing for locomotion training."""

from __future__ import annotations

import math
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass

from .terrain import _terrain_height_components


@configclass
class HeightScanObservationCfg:
    """
    Configuration for the critic's privileged terrain-height scan.

    This simulator ray cast is a privileged critic input, not a causal student
    sensor.
    """

    num_rays: int = 132
    """Fixed number of ray samples in each flattened height and validity term."""

    vertical_offset: float = 0.3
    """Reference-plane distance below the robot root, in metres."""

    clip: float = 1.0
    """Symmetric metric clipping bound in metres, also used as the fixed normalization divisor."""

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_rays, bool)
            or not isinstance(self.num_rays, int)
            or self.num_rays <= 0
        ):
            raise ValueError("num_rays must be a positive integer.")
        if not math.isfinite(self.vertical_offset):
            raise ValueError("vertical_offset must be finite.")
        if not math.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("clip must be finite and positive.")


DEFAULT_HEIGHT_SCAN_OBSERVATION = HeightScanObservationCfg()


def terrain_height_scan(
    env: ManagerBasedRLEnv,
    obs_cfg: HeightScanObservationCfg = DEFAULT_HEIGHT_SCAN_OBSERVATION,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return one fixed-size privileged terrain scan for the critic.

    The configured ray caster is required. Failing when it is absent prevents
    training a supposedly terrain-aware critic on an accidental all-zero
    terrain input. The first ``num_rays`` entries are normalized heights and
    the remaining entries are their floating validity mask. Concatenating them
    here reads and preprocesses the ray caster only once per observation.

    Returns:
        Heights followed by validity with shape ``[num_envs, 2 * num_rays]``.
    """

    sensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if not isinstance(sensor, RayCaster):
        raise TypeError(
            f"Expected '{sensor_cfg.name}' to be a RayCaster, got {type(sensor).__name__}."
        )

    heights, validity = _terrain_height_components(
        asset.data.root_pos_w[:, 2],
        sensor.data.ray_hits_w,
        num_rays=obs_cfg.num_rays,
        vertical_offset=obs_cfg.vertical_offset,
        clip=obs_cfg.clip,
    )
    return torch.cat((heights, validity), dim=-1)
