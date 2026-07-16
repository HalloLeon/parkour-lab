# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Restricted student model built around the transferable motor contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional

from .architecture import MotorActor, MotorInterfaceCfg, _build_mlp, _validate_input

STUDENT_OBSERVATION_GROUPS = ("policy", "student_exteroception")
"""Deployable state and temporary terrain-latent groups used by the student."""


@dataclass(frozen=True)
class StudentModelCfg:
    """Heading-prediction and loss settings around one shared motor actor."""

    # Shared teacher/student motor contract. It fixes the deployable-state,
    # heading, terrain-latent, optional adaptation-latent, and action widths as
    # well as the motor MLP hidden layers and tensor concatenation order.
    motor: MotorInterfaceCfg

    # Hidden-layer widths of the MLP that predicts the two-component heading
    # from deployable state and the terrain latent.
    heading_hidden_dims: tuple[int, ...] = (256, 128)

    # Multiplier for the Smooth L1 loss between student and teacher actions.
    motor_loss_weight: float = 1.0

    # Multiplier for the cosine-direction loss that aligns the predicted
    # heading with the oracle heading, independently of vector magnitude.
    heading_direction_loss_weight: float = 0.2

    # Multiplier for the auxiliary penalty that keeps the raw heading vector's
    # length close to one before it is explicitly normalized.
    heading_norm_loss_weight: float = 0.01

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible model configuration."""

        return {
            "motor": self.motor.to_dict(),
            "heading_hidden_dims": list(self.heading_hidden_dims),
            "motor_loss_weight": self.motor_loss_weight,
            "heading_direction_loss_weight": self.heading_direction_loss_weight,
            "heading_norm_loss_weight": self.heading_norm_loss_weight,
        }

    def validate(self) -> None:
        """Validate dimensions and non-negative loss weights."""

        self.motor.validate()
        if not self.heading_hidden_dims or any(
            width <= 0 for width in self.heading_hidden_dims
        ):
            raise ValueError("Heading hidden dimensions must be positive.")
        for name in (
            "motor_loss_weight",
            "heading_direction_loss_weight",
            "heading_norm_loss_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative.")


class HeadingPredictor(nn.Module):
    """Predict a wrap-safe two-component heading from deployable inputs."""

    def __init__(self, cfg: StudentModelCfg) -> None:
        super().__init__()
        input_dim = cfg.motor.state_dim + cfg.motor.terrain_latent_dim
        self.network = _build_mlp(
            input_dim,
            cfg.motor.heading_dim,
            cfg.heading_hidden_dims,
        )

        # Begin with a defined forward heading instead of normalizing a random
        # near-zero vector. Later supervised updates learn obstacle-dependent
        # directions from the oracle heading target.
        output = _last_linear(self.network)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        output.bias.data[0] = 1.0

    def forward(
        self,
        deployable_state: torch.Tensor,
        terrain_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized heading and its unnormalized network output."""

        observations = torch.cat((deployable_state, terrain_latent), dim=-1)
        raw_heading = self.network(observations)

        # Convert the unconstrained network output into a unit XY direction.
        # This prevents its arbitrary magnitude from changing the motor input:
        # only the predicted direction should influence the action. Keep the
        # raw vector as well so the auxiliary loss can teach it a stable norm.
        heading = functional.normalize(
            raw_heading,
            p=2.0,
            dim=-1,
            eps=1.0e-6,
        )
        return heading, raw_heading


class StudentPolicy(nn.Module):
    """Heading predictor and motor policy using only restricted information."""

    def __init__(self, cfg: StudentModelCfg) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.heading = HeadingPredictor(cfg)
        self.motor = MotorActor(cfg.motor)

        # Preserve the current smoke-test behavior until a later stage copies
        # the trained teacher motor weights into this exact shared module.
        motor_output = self.motor.output_layer
        nn.init.zeros_(motor_output.weight)
        nn.init.zeros_(motor_output.bias)

    def act_inference(
        self,
        deployable_state: torch.Tensor,
        terrain_latent: torch.Tensor,
        adaptation_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return deterministic motor actions executed during online rollout."""

        actions, _, _ = self.forward(
            deployable_state,
            terrain_latent,
            adaptation_latent,
        )
        return actions

    def forward(
        self,
        deployable_state: torch.Tensor,
        terrain_latent: torch.Tensor,
        adaptation_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return motor actions, unit heading commands, and raw heading vectors."""

        _validate_input(
            deployable_state,
            self.cfg.motor.state_dim,
            "deployable_state",
        )
        _validate_input(
            terrain_latent,
            self.cfg.motor.terrain_latent_dim,
            "terrain_latent",
        )
        if deployable_state.shape[0] != terrain_latent.shape[0]:
            raise ValueError(
                "Deployable state and terrain-latent batch dimensions must match."
            )

        heading_command, raw_heading = self.heading(
            deployable_state,
            terrain_latent,
        )
        actions = self.motor(
            deployable_state,
            heading_command,
            terrain_latent,
            adaptation_latent,
        )
        return actions, heading_command, raw_heading


def compute_distillation_losses(
    student: StudentPolicy,
    deployable_state: torch.Tensor,
    terrain_latent: torch.Tensor,
    oracle_heading: torch.Tensor,
    teacher_action: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute robust action imitation and wrap-safe heading supervision."""

    _validate_input(
        oracle_heading,
        student.cfg.motor.heading_dim,
        "oracle_heading",
    )
    _validate_input(
        teacher_action,
        student.cfg.motor.action_dim,
        "teacher_action",
    )
    actions, heading_command, raw_heading = student(
        deployable_state,
        terrain_latent,
    )
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


def _last_linear(module: nn.Sequential) -> nn.Linear:
    """Return the final linear layer of an MLP."""

    output = module[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("Student MLP must end in a linear layer.")
    return output
