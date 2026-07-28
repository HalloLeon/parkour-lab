# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Framework-independent model components for the privileged teacher.

This module owns the complete PyTorch teacher composition. The RSL-RL
``ActorCritic`` adapter and runner registration live separately in
``rsl_rl.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..architecture import (
    DEFAULT_TERRAIN_LATENT_DIM,
    MotorActor,
    MotorInterfaceCfg,
    _build_mlp,
    _validate_input,
)

__all__ = [
    "PrivilegedScanEncoder",
    "PrivilegedTeacherActor",
    "PrivilegedTeacherModelCfg",
    "PrivilegedTeacherPolicy",
]


class PrivilegedScanEncoder(nn.Module):
    """Compress simulator-only terrain scans into the shared terrain latent."""

    def __init__(
        self,
        scan_dim: int,
        latent_dim: int = DEFAULT_TERRAIN_LATENT_DIM,
        hidden_dims: tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()

        if scan_dim <= 0 or latent_dim <= 0:
            raise ValueError("Scan and terrain-latent dimensions must be positive.")
        if not hidden_dims or any(width <= 0 for width in hidden_dims):
            raise ValueError("Scan-encoder hidden dimensions must be positive.")
        self.scan_dim = scan_dim
        self.latent_dim = latent_dim
        self.network = _build_mlp(scan_dim, latent_dim, hidden_dims)

    def forward(self, terrain_scan: torch.Tensor) -> torch.Tensor:
        """Return one fixed-width terrain latent per environment."""

        _validate_input(terrain_scan, self.scan_dim, "terrain_scan")
        return self.network(terrain_scan)


@dataclass(frozen=True)
class PrivilegedTeacherModelCfg:
    """Transfer-facing architecture of the modular Phase-1 teacher."""

    motor: MotorInterfaceCfg
    terrain_scan_dim: int
    scan_hidden_dims: tuple[int, ...] = (128, 64)

    def validate(self) -> None:
        """Validate the scan encoder and shared motor interface."""

        self.motor.validate()
        if self.terrain_scan_dim <= 0:
            raise ValueError("terrain_scan_dim must be positive.")
        if not self.scan_hidden_dims or any(
            width <= 0 for width in self.scan_hidden_dims
        ):
            raise ValueError("Scan-encoder hidden dimensions must be positive.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible teacher architecture contract."""

        return {
            "terrain_encoder": {
                "class_name": PrivilegedScanEncoder.__name__,
                "input_dimension": self.terrain_scan_dim,
                "hidden_dimensions": list(self.scan_hidden_dims),
                "output_dimension": self.motor.terrain_latent_dim,
                "activation": "elu",
            },
            "motor_actor": {
                "class_name": MotorActor.__name__,
                "activation": "elu",
                **self.motor.to_dict(),
            },
        }


class PrivilegedTeacherPolicy(nn.Module):
    """Reference Phase-1 actor exposing a directly copyable motor module."""

    def __init__(self, cfg: PrivilegedTeacherModelCfg) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.terrain_encoder = PrivilegedScanEncoder(
            cfg.terrain_scan_dim,
            cfg.motor.terrain_latent_dim,
            cfg.scan_hidden_dims,
        )
        self.motor = MotorActor(cfg.motor)

    def forward(
        self,
        deployable_state: torch.Tensor,
        oracle_heading: torch.Tensor,
        terrain_scan: torch.Tensor,
        adaptation_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode privileged geometry and return deterministic motor actions."""

        terrain_latent = self.terrain_encoder(terrain_scan)
        return self.motor(
            deployable_state,
            oracle_heading,
            terrain_latent,
            adaptation_latent,
        )


class PrivilegedTeacherActor(nn.Module):
    """Adapt a concatenated actor input to the transferable teacher modules."""

    def __init__(
        self,
        motor_cfg: MotorInterfaceCfg,
        terrain_scan_dim: int | None,
        scan_encoder_hidden_dims: Sequence[int],
    ) -> None:
        super().__init__()
        self.motor = MotorActor(motor_cfg)
        self.terrain_encoder = (
            PrivilegedScanEncoder(
                terrain_scan_dim,
                motor_cfg.terrain_latent_dim,
                tuple(scan_encoder_hidden_dims),
            )
            if terrain_scan_dim is not None
            else None
        )
        self._input_dims = (
            motor_cfg.state_dim,
            motor_cfg.heading_dim,
            *((terrain_scan_dim,) if terrain_scan_dim is not None else ()),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Encode the optional scan and produce the shared motor action."""

        components = torch.split(observations, self._input_dims, dim=-1)
        deployable_state, oracle_heading = components[:2]
        terrain_latent = (
            self.terrain_encoder(components[2])
            if self.terrain_encoder is not None
            else deployable_state.new_zeros(
                (deployable_state.shape[0], self.motor.cfg.terrain_latent_dim)
            )
        )
        return self.motor(
            deployable_state,
            oracle_heading,
            terrain_latent,
        )
