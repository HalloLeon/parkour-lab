# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Privileged and deployable encoders for online dynamics adaptation.

The teacher compresses simulator-known dynamics parameters into the motor
actor's adaptation latent. The deployable student predicts the same latent
from a causal history of robot states, whose final values already contain the
previous low-level action.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .architecture import (
    DEFAULT_ADAPTATION_LATENT_DIM,
    _build_mlp,
    _validate_input,
)

DEPLOYABLE_HISTORY_LENGTH = 10
"""Number of current-and-past deployable states used for adaptation."""

__all__ = [
    "DEPLOYABLE_HISTORY_LENGTH",
    "DeployableHistoryEncoder",
    "PrivilegedDynamicsEncoder",
]


class DeployableHistoryEncoder(nn.Module):
    """Estimate the privileged adaptation latent from causal state history."""

    def __init__(
        self,
        history_dim: int,
        latent_dim: int = DEFAULT_ADAPTATION_LATENT_DIM,
        hidden_dims: tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        if (
            history_dim <= 0
            or latent_dim <= 0
            or not hidden_dims
            or any(width <= 0 for width in hidden_dims)
        ):
            raise ValueError("History-encoder dimensions must be positive.")

        self.history_dim = history_dim
        self.latent_dim = latent_dim
        self.network = _build_mlp(
            history_dim,
            latent_dim,
            hidden_dims,
        )

    def forward(self, deployable_history: torch.Tensor) -> torch.Tensor:
        """Encode one flattened causal history per sample."""

        _validate_input(
            deployable_history,
            self.history_dim,
            "deployable_history",
        )
        return self.network(deployable_history)


class PrivilegedDynamicsEncoder(nn.Module):
    """Compress simulator-known dynamics into the shared adaptation latent."""

    def __init__(
        self,
        dynamics_dim: int,
        latent_dim: int = DEFAULT_ADAPTATION_LATENT_DIM,
        hidden_dims: tuple[int, ...] = (128, 64),
    ) -> None:
        super().__init__()
        if (
            dynamics_dim <= 0
            or latent_dim <= 0
            or not hidden_dims
            or any(width <= 0 for width in hidden_dims)
        ):
            raise ValueError("Dynamics-encoder dimensions must be positive.")

        self.dynamics_dim = dynamics_dim
        self.latent_dim = latent_dim
        self.network = _build_mlp(
            dynamics_dim,
            latent_dim,
            hidden_dims,
        )

    def forward(self, privileged_dynamics: torch.Tensor) -> torch.Tensor:
        """Return one privileged adaptation target per environment."""

        _validate_input(
            privileged_dynamics,
            self.dynamics_dim,
            "privileged_dynamics",
        )
        return self.network(privileged_dynamics)
