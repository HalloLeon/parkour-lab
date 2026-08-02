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

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

from ..adaptation import DeployableHistoryEncoder, PrivilegedDynamicsEncoder
from ..architecture import (
    DEFAULT_TERRAIN_LATENT_DIM,
    MotorActor,
    MotorInterfaceCfg,
    _build_mlp,
    _validate_input,
)

__all__ = [
    "DeployableHistoryEncoder",
    "PrivilegedDynamicsEncoder",
    "PrivilegedScanEncoder",
    "PrivilegedTeacherActor",
    "PrivilegedTeacherModelCfg",
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
    terrain_scan_dim: int | None
    privileged_dynamics_dim: int
    history_dim: int
    dynamics_hidden_dims: tuple[int, ...] = (128, 64)
    history_hidden_dims: tuple[int, ...] = (256, 128)
    scan_hidden_dims: tuple[int, ...] = (128, 64)

    def validate(self) -> None:
        """Validate the scan encoder and shared motor interface."""

        self.motor.validate()
        if self.terrain_scan_dim is not None and self.terrain_scan_dim <= 0:
            raise ValueError("terrain_scan_dim must be positive.")
        if self.privileged_dynamics_dim <= 0:
            raise ValueError("privileged_dynamics_dim must be positive.")
        if self.history_dim <= 0:
            raise ValueError("history_dim must be positive.")
        for name in (
            "dynamics_hidden_dims",
            "history_hidden_dims",
            "scan_hidden_dims",
        ):
            dimensions = getattr(self, name)
            if not dimensions or any(width <= 0 for width in dimensions):
                raise ValueError(f"{name} must contain positive dimensions.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible teacher architecture contract."""

        return {
            "dynamics_encoder": {
                "class_name": PrivilegedDynamicsEncoder.__name__,
                "input_dimension": self.privileged_dynamics_dim,
                "hidden_dimensions": list(self.dynamics_hidden_dims),
                "output_dimension": self.motor.adaptation_latent_dim,
                "activation": "elu",
            },
            "history_encoder": {
                "class_name": DeployableHistoryEncoder.__name__,
                "input_dimension": self.history_dim,
                "hidden_dimensions": list(self.history_hidden_dims),
                "output_dimension": self.motor.adaptation_latent_dim,
                "activation": "elu",
            },
            "terrain_encoder": (
                {
                    "class_name": PrivilegedScanEncoder.__name__,
                    "input_dimension": self.terrain_scan_dim,
                    "hidden_dimensions": list(self.scan_hidden_dims),
                    "output_dimension": self.motor.terrain_latent_dim,
                    "activation": "elu",
                }
                if self.terrain_scan_dim is not None
                else None
            ),
            "motor_actor": {
                "class_name": MotorActor.__name__,
                "activation": "elu",
                **self.motor.to_dict(),
            },
        }


class PrivilegedTeacherActor(nn.Module):
    """Canonical teacher model with component-wise and RSL-RL input paths."""

    def __init__(self, cfg: PrivilegedTeacherModelCfg) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.dynamics_encoder = PrivilegedDynamicsEncoder(
            cfg.privileged_dynamics_dim,
            cfg.motor.adaptation_latent_dim,
            cfg.dynamics_hidden_dims,
        )
        self.history_encoder = DeployableHistoryEncoder(
            cfg.history_dim,
            cfg.motor.adaptation_latent_dim,
            cfg.history_hidden_dims,
        )
        self.motor = MotorActor(cfg.motor)
        self.terrain_encoder = (
            PrivilegedScanEncoder(
                cfg.terrain_scan_dim,
                cfg.motor.terrain_latent_dim,
                cfg.scan_hidden_dims,
            )
            if cfg.terrain_scan_dim is not None
            else None
        )
        self._input_dims = (
            cfg.motor.state_dim,
            cfg.motor.heading_dim,
            *((cfg.terrain_scan_dim,) if cfg.terrain_scan_dim is not None else ()),
            cfg.privileged_dynamics_dim,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Split RSL-RL's concatenated observations and produce motor actions."""

        components = torch.split(observations, self._input_dims, dim=-1)
        deployable_state, oracle_heading = components[:2]
        terrain_scan = components[2] if self.terrain_encoder is not None else None
        privileged_dynamics = components[-1]
        adaptation_latent = self.dynamics_encoder(privileged_dynamics)
        terrain_latent = self._encode_terrain(deployable_state, terrain_scan)
        return self.motor(
            deployable_state,
            oracle_heading,
            terrain_latent,
            adaptation_latent,
        )

    def forward_from_history(
        self,
        deployable_state: torch.Tensor,
        oracle_heading: torch.Tensor,
        terrain_scan: torch.Tensor | None,
        deployable_history: torch.Tensor,
    ) -> torch.Tensor:
        """Return actions without reading privileged dynamics."""

        adaptation_latent = self.history_encoder(deployable_history)
        terrain_latent = self._encode_terrain(deployable_state, terrain_scan)
        return self.motor(
            deployable_state,
            oracle_heading,
            terrain_latent,
            adaptation_latent,
        )

    def roa_losses(
        self,
        deployable_history: torch.Tensor,
        privileged_dynamics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two stop-gradient losses used by full ROA.

        The adaptation loss updates only the deployable history encoder. The
        privileged regularization loss applies the reverse gradient boundary,
        updating only the dynamics encoder so its latent remains reproducible
        from information available on the robot.
        """

        history_latent = self.history_encoder(deployable_history)
        privileged_latent = self.dynamics_encoder(privileged_dynamics)
        adaptation_loss = functional.smooth_l1_loss(
            history_latent,
            privileged_latent.detach(),
        )
        privileged_regularization_loss = functional.smooth_l1_loss(
            privileged_latent,
            history_latent.detach(),
        )
        return adaptation_loss, privileged_regularization_loss

    def _encode_terrain(
        self,
        deployable_state: torch.Tensor,
        terrain_scan: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode terrain or supply the no-terrain ablation's zero latent."""

        if self.terrain_encoder is None:
            if terrain_scan is not None:
                raise ValueError("terrain_scan must be omitted when terrain is disabled.")
            return deployable_state.new_zeros((deployable_state.shape[0], self.motor.cfg.terrain_latent_dim))
        if terrain_scan is None:
            raise ValueError("terrain_scan is required when terrain is enabled.")
        return self.terrain_encoder(terrain_scan)
