# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motor interface shared by the privileged teacher's inference paths.

The privileged and history-policy paths present the same ordered tensors to
``MotorActor``. Keeping that boundary explicit makes checkpoint semantics
independent of how terrain and dynamics latents are produced.

Teacher-specific composition lives in ``teacher/model.py``. Its RSL-RL
adapter lives in ``teacher/rsl_rl.py``. Both adaptation encoders produce the
latent width recorded by ``MotorInterfaceCfg``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

DEFAULT_ADAPTATION_LATENT_DIM = 20
"""Width shared by privileged and history-based dynamics encoders."""

DEFAULT_TERRAIN_LATENT_DIM = 32
"""Default width shared by the privileged scan and future depth encoders."""

TRAVEL_DIRECTION_DIM = 2
"""Width of the yaw-aligned ``[forward, left]`` travel-direction unit vector."""

MOTOR_INPUT_COMPONENTS = (
    "deployable_state",
    "travel_direction",
    "terrain_latent",
    "adaptation_latent",
)
"""Tensor concatenation order consumed by every transferable motor actor."""


@dataclass(frozen=True)
class MotorInterfaceCfg:
    """Dimensions and hidden layers that uniquely define a motor actor."""

    # Number of deployable robot-state values supplied for each environment,
    # excluding the travel-direction and terrain/adaptation latent vectors below.
    state_dim: int

    # Number of low-level action values produced by the actor, normally one
    # joint-position command per controlled joint.
    action_dim: int

    # Width of the fixed-size terrain representation produced by either the
    # privileged scan encoder or, later, the deployable perception encoder.
    terrain_latent_dim: int = DEFAULT_TERRAIN_LATENT_DIM

    # Width of the yaw-aligned local travel command, represented as the
    # two-component unit vector ``[forward, left]``.
    travel_direction_dim: int = TRAVEL_DIRECTION_DIM

    # Width of the dynamics representation. The privileged path derives it from
    # simulator parameters; the history path estimates it from proprioception.
    adaptation_latent_dim: int = DEFAULT_ADAPTATION_LATENT_DIM

    # Output widths of the motor MLP's hidden linear layers, in network order.
    hidden_dims: tuple[int, ...] = (512, 256, 128)

    @property
    def input_dim(self) -> int:
        """Return the complete concatenated motor-input width."""

        return (
            self.state_dim
            + self.travel_direction_dim
            + self.terrain_latent_dim
            + self.adaptation_latent_dim
        )

    def validate(self) -> None:
        """Validate the fixed interface dimensions and network widths."""

        for name in ("state_dim", "action_dim", "terrain_latent_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.travel_direction_dim != TRAVEL_DIRECTION_DIM:
            raise ValueError(
                "travel_direction_dim must be "
                f"{TRAVEL_DIRECTION_DIM} for a yaw-aligned unit vector."
            )
        if self.adaptation_latent_dim <= 0:
            raise ValueError("adaptation_latent_dim must be positive.")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("Motor hidden dimensions must be positive.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of the frozen motor contract."""

        return {
            "state_dim": self.state_dim,
            "travel_direction_dim": self.travel_direction_dim,
            "terrain_latent_dim": self.terrain_latent_dim,
            "adaptation_latent_dim": self.adaptation_latent_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "input_dim": self.input_dim,
            "input_order": list(MOTOR_INPUT_COMPONENTS),
        }


class MotorActor(nn.Module):
    """Map the ordered shared motor interface to low-level joint actions."""

    def __init__(self, cfg: MotorInterfaceCfg) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.network = _build_mlp(cfg.input_dim, cfg.action_dim, cfg.hidden_dims)

    def forward(
        self,
        deployable_state: torch.Tensor,
        travel_direction: torch.Tensor,
        terrain_latent: torch.Tensor,
        adaptation_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Return actions from tensors supplied in the frozen interface order."""
        motor_input = torch.cat(
            (
                deployable_state,
                travel_direction,
                terrain_latent,
                adaptation_latent,
            ),
            dim=-1,
        )
        return self.network(motor_input)


def _build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dims: tuple[int, ...],
) -> nn.Sequential:
    """Build an ELU MLP with a linear output layer."""

    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)
