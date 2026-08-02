# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Transfer contract shared by privileged and depth-based motor policies.

Phase 1 and Phase 2 may use different terrain encoders and heading sources,
but both must present the same ordered tensors to ``MotorActor``. Keeping that
boundary in one module makes the motor weights directly copyable instead of
depending on two independently assembled input vectors.

Teacher-specific composition lives in ``teacher/model.py``. Its RSL-RL
adapter lives in ``teacher/rsl_rl.py``. Privileged and deployable adaptation
encoders live on opposite sides of this contract but produce the same latent
width recorded by ``MotorInterfaceCfg``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

DEFAULT_ADAPTATION_LATENT_DIM = 20
"""Width shared by privileged and history-based dynamics encoders."""

DEFAULT_TERRAIN_LATENT_DIM = 32
"""Default width shared by the privileged scan and future depth encoders."""

HEADING_DIM = 2
"""Width of the wrap-safe yaw-aligned ``[cos(error), sin(error)]`` heading."""

MOTOR_INPUT_COMPONENTS = (
    "deployable_state",
    "heading",
    "terrain_latent",
    "adaptation_latent",
)
"""Tensor concatenation order consumed by every transferable motor actor."""


@dataclass(frozen=True)
class MotorInterfaceCfg:
    """Dimensions and hidden layers that uniquely define a motor actor."""

    # Number of deployable robot-state values supplied for each environment,
    # excluding the heading and terrain/adaptation latent vectors below.
    state_dim: int

    # Number of low-level action values produced by the actor, normally one
    # joint-position command per controlled joint.
    action_dim: int

    # Width of the fixed-size terrain representation produced by either the
    # privileged scan encoder or, later, the deployable perception encoder.
    terrain_latent_dim: int = DEFAULT_TERRAIN_LATENT_DIM

    # Width of the yaw-aligned heading command. This project represents it as
    # the two-component unit vector ``[cos(error), sin(error)]``.
    heading_dim: int = HEADING_DIM

    # Width of the dynamics representation. The teacher derives it from
    # privileged simulator parameters; the student estimates it from history.
    adaptation_latent_dim: int = DEFAULT_ADAPTATION_LATENT_DIM

    # Output widths of the motor MLP's hidden linear layers, in network order.
    hidden_dims: tuple[int, ...] = (512, 256, 128)

    @property
    def input_dim(self) -> int:
        """Return the complete concatenated motor-input width."""

        return (
            self.state_dim
            + self.heading_dim
            + self.terrain_latent_dim
            + self.adaptation_latent_dim
        )

    def validate(self) -> None:
        """Validate the fixed interface dimensions and network widths."""

        for name in ("state_dim", "action_dim", "terrain_latent_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.heading_dim != HEADING_DIM:
            raise ValueError(
                f"heading_dim must be {HEADING_DIM} for a yaw-aligned unit vector."
            )
        if self.adaptation_latent_dim <= 0:
            raise ValueError("adaptation_latent_dim must be positive.")
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("Motor hidden dimensions must be positive.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible description of the frozen motor contract."""

        return {
            "state_dim": self.state_dim,
            "heading_dim": self.heading_dim,
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

    @property
    def output_layer(self) -> nn.Linear:
        """Return the final action projection, primarily for initialization."""

        output = self.network[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("Motor actor must end in a linear layer.")
        return output

    def forward(
        self,
        deployable_state: torch.Tensor,
        heading: torch.Tensor,
        terrain_latent: torch.Tensor,
        adaptation_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Return actions from tensors supplied in the frozen interface order."""

        _validate_input(deployable_state, self.cfg.state_dim, "deployable_state")
        _validate_input(heading, self.cfg.heading_dim, "heading")
        _validate_input(
            terrain_latent,
            self.cfg.terrain_latent_dim,
            "terrain_latent",
        )

        _validate_input(
            adaptation_latent,
            self.cfg.adaptation_latent_dim,
            "adaptation_latent",
        )

        _validate_matching_batches(
            deployable_state,
            heading,
            terrain_latent,
            adaptation_latent,
        )

        # Concatenate features for each sample along the last dimension. The
        # resulting shape is ``[batch_size, cfg.input_dim]``, where
        # ``cfg.input_dim = state_dim + heading_dim + terrain_latent_dim +
        # adaptation_latent_dim``.
        motor_input = torch.cat(
            (
                deployable_state,
                heading,
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


def _validate_input(
    tensor: torch.Tensor,
    expected_dim: int,
    name: str,
) -> None:
    """Validate one two-dimensional batched tensor's feature width."""

    if tensor.ndim != 2 or tensor.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} must have shape [batch, {expected_dim}], "
            f"got {tuple(tensor.shape)}."
        )


def _validate_matching_batches(*tensors: torch.Tensor) -> None:
    """Require every motor component to describe the same batch."""

    batch_sizes = {tensor.shape[0] for tensor in tensors}
    if len(batch_sizes) != 1:
        raise ValueError("All motor-input batch dimensions must match.")
