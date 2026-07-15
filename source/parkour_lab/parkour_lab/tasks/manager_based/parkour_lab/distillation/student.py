# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Restricted student model and initial supervised distillation losses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as functional

STUDENT_OBSERVATION_GROUPS = ("student_policy", "student_exteroception")
"""Ordered environment groups concatenated for the restricted student."""


@dataclass(frozen=True)
class StudentModelCfg:
    """Architecture and loss configuration for the first restricted student."""

    state_dim: int
    exteroception_dim: int
    action_dim: int
    heading_hidden_dims: tuple[int, ...] = (256, 128)
    motor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    motor_loss_weight: float = 1.0
    heading_direction_loss_weight: float = 0.2
    heading_norm_loss_weight: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible model configuration."""

        return asdict(self)

    def validate(self) -> None:
        """Validate dimensions and non-negative loss weights."""

        for name in ("state_dim", "exteroception_dim", "action_dim"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive.")
        if not self.heading_hidden_dims or not self.motor_hidden_dims:
            raise ValueError("Heading and motor networks must each have hidden layers.")
        for name in (
            "motor_loss_weight",
            "heading_direction_loss_weight",
            "heading_norm_loss_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


class StudentPolicy(nn.Module):
    """Heading predictor and motor policy using only restricted information."""

    def __init__(self, cfg: StudentModelCfg) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        observation_dim = cfg.state_dim + cfg.exteroception_dim

        # Consume the complete restricted student observation and predict two
        # components representing the desired heading direction in the XY
        # plane. ``forward`` later normalizes this raw vector to unit length.
        self.heading = _build_mlp(observation_dim, 2, cfg.heading_hidden_dims)

        # Give the motor network the same student observation followed by the
        # two-component heading command. It produces one value for every
        # action dimension expected by the robot's low-level controller.
        self.motor = _build_mlp(
            observation_dim + 2, cfg.action_dim, cfg.motor_hidden_dims
        )

        # Zero the final projection so random hidden features cannot affect the
        # initial heading. Setting its first bias component to one immediately
        # afterwards makes every initial raw heading equal to ``(1, 0)``.
        heading_output = _last_linear(self.heading)
        nn.init.zeros_(heading_output.weight)
        nn.init.zeros_(heading_output.bias)
        heading_output.bias.data[0] = 1.0

        # Begin the motor policy with a zero joint offset.
        motor_output = _last_linear(self.motor)
        nn.init.zeros_(motor_output.weight)
        nn.init.zeros_(motor_output.bias)

    def act_inference(
        self, student_state: torch.Tensor, exteroception: torch.Tensor
    ) -> torch.Tensor:
        """Return deterministic motor actions executed during online rollout."""

        actions, _, _ = self.forward(student_state, exteroception)
        return actions

    def forward(
        self, student_state: torch.Tensor, exteroception: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return motor actions, unit heading commands, and raw heading vectors."""

        _validate_input(student_state, self.cfg.state_dim, "student_state")
        _validate_input(exteroception, self.cfg.exteroception_dim, "exteroception")
        if student_state.shape[0] != exteroception.shape[0]:
            raise ValueError(
                "Student state and exteroception batch dimensions must match."
            )

        observations = torch.cat((student_state, exteroception), dim=-1)
        raw_heading = self.heading(observations)

        # Normalize each final-dimension XY pair with its L2 norm so the motor
        # network receives a unit direction rather than an arbitrary magnitude.
        # ``eps`` prevents division by zero for a near-zero prediction.
        heading_command = functional.normalize(raw_heading, p=2.0, dim=-1, eps=1.0e-6)
        motor_input = torch.cat((observations, heading_command), dim=-1)
        actions = self.motor(motor_input)
        return actions, heading_command, raw_heading


def compute_distillation_losses(
    student: StudentPolicy,
    student_state: torch.Tensor,
    exteroception: torch.Tensor,
    oracle_heading: torch.Tensor,
    teacher_action: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute robust action imitation and wrap-safe heading supervision."""

    _validate_input(oracle_heading, 2, "oracle_heading")
    _validate_input(teacher_action, student.cfg.action_dim, "teacher_action")
    actions, heading_command, raw_heading = student(student_state, exteroception)
    # Normalize each final-dimension XY pair with its L2 norm so the motor
    # network receives a unit direction rather than an arbitrary magnitude.
    # ``eps`` prevents division by zero for a near-zero prediction.
    unit_heading_target = functional.normalize(
        oracle_heading, p=2.0, dim=-1, eps=1.0e-6
    )

    # Train the motor output to imitate the teacher action. Smooth L1 averages
    # a quadratic penalty for small errors and a less outlier-sensitive linear
    # penalty for large errors across all samples and action dimensions.
    motor = functional.smooth_l1_loss(actions, teacher_action)

    # Since both XY vectors have unit length, their dot product is their cosine
    # similarity, which should be maximized. Training minimizes losses, so
    # ``1 - similarity`` converts perfect alignment into the minimum loss of
    # zero, perpendicular headings into one, and opposite headings into two.
    heading_direction = (
        1.0 - torch.sum(heading_command * unit_heading_target, dim=-1)
    ).mean()

    # Keep each raw heading close to unit length before normalization. The
    # direction loss alone ignores magnitude, so this squared mean penalty
    # discourages unstable near-zero or unnecessarily large raw predictions.
    heading_norm = (torch.linalg.norm(raw_heading, dim=-1) - 1.0).square().mean()
    total = (
        student.cfg.motor_loss_weight * motor
        + student.cfg.heading_direction_loss_weight * heading_direction
        + student.cfg.heading_norm_loss_weight * heading_norm
    )

    heading_cosine = torch.sum(heading_command * unit_heading_target, dim=-1).clamp(
        -1.0, 1.0
    )
    heading_error_deg = torch.rad2deg(torch.acos(heading_cosine)).mean()
    return {
        "total": total,
        "motor_huber": motor,
        "heading_direction": heading_direction,
        "heading_norm": heading_norm,
        "heading_error_deg": heading_error_deg,
    }


def _build_mlp(
    input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]
) -> nn.Sequential:
    """Build an ELU MLP with a linear output layer."""

    # Accumulate the modules in execution order before combining them into one
    # feed-forward network.
    layers: list[nn.Module] = []

    # The first hidden layer consumes the complete input vector. After each
    # iteration, this tracks the width produced for the following layer.
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        if hidden_dim <= 0:
            raise ValueError("Hidden dimensions must be positive.")

        # Transform from the preceding width to this hidden width, then apply
        # the nonlinear ELU activation before feeding the next layer.
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ELU()))
        previous_dim = hidden_dim

    # Map the final hidden representation to the requested output dimension.
    # No activation is applied so callers can interpret or normalize the raw
    # heading and motor outputs as appropriate.
    layers.append(nn.Linear(previous_dim, output_dim))

    # ``*layers`` passes the collected modules as individual positional
    # arguments, and ``Sequential`` executes them in the listed order.
    return nn.Sequential(*layers)


def _last_linear(module: nn.Sequential) -> nn.Linear:
    """Return the final linear layer of an MLP."""

    output = module[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("Student MLP must end in a linear layer.")
    return output


def _validate_input(tensor: torch.Tensor, expected_dim: int, name: str) -> None:
    """Validate one two-dimensional batched tensor's static feature width."""

    if tensor.ndim != 2 or tensor.shape[-1] != expected_dim:
        raise ValueError(
            f"{name} must have shape [num_envs, {expected_dim}], got {tuple(tensor.shape)}."
        )
