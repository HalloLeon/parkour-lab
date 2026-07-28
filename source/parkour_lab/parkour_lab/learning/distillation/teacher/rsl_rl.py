# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL integration for the modular privileged Phase-1 teacher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from rsl_rl.modules import ActorCritic

from ..architecture import (
    DEFAULT_TERRAIN_LATENT_DIM,
    MotorInterfaceCfg,
)
from ..contracts import TEACHER_OBSERVATION_GROUPS
from .model import PrivilegedTeacherActor

__all__ = [
    "PrivilegedTeacherActorCritic",
    "register_rsl_rl_teacher_actor_critic",
]


class PrivilegedTeacherActorCritic(ActorCritic):
    """RSL-RL actor-critic with a modular scan encoder and shared motor actor."""

    def __init__(
        self,
        obs: Mapping[str, torch.Tensor],
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        scan_encoder_hidden_dims: Sequence[int] = (128, 64),
        terrain_latent_dim: int = DEFAULT_TERRAIN_LATENT_DIM,
        **kwargs: object,
    ) -> None:
        actor_groups = tuple(obs_groups["policy"])
        groups_without_terrain = TEACHER_OBSERVATION_GROUPS[:-1]
        if actor_groups not in (groups_without_terrain, TEACHER_OBSERVATION_GROUPS):
            raise ValueError(
                "The modular teacher actor requires observation groups "
                f"{list(groups_without_terrain)} with optional "
                f"{TEACHER_OBSERVATION_GROUPS[-1]!r}, got {list(actor_groups)}."
            )
        if activation != "elu":
            raise ValueError("The shared teacher/student motor actor requires ELU.")

        super().__init__(
            obs,
            obs_groups,
            num_actions,
            actor_obs_normalization=actor_obs_normalization,
            critic_obs_normalization=critic_obs_normalization,
            actor_hidden_dims=list(actor_hidden_dims),
            critic_hidden_dims=list(critic_hidden_dims),
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        terrain_scan_dim = (
            int(obs[TEACHER_OBSERVATION_GROUPS[-1]].shape[-1])
            if actor_groups == TEACHER_OBSERVATION_GROUPS
            else None
        )
        motor_cfg = MotorInterfaceCfg(
            state_dim=int(obs[TEACHER_OBSERVATION_GROUPS[0]].shape[-1]),
            heading_dim=int(obs[TEACHER_OBSERVATION_GROUPS[1]].shape[-1]),
            terrain_latent_dim=terrain_latent_dim,
            action_dim=num_actions,
            hidden_dims=tuple(actor_hidden_dims),
        )
        self.actor = PrivilegedTeacherActor(
            motor_cfg,
            terrain_scan_dim,
            scan_encoder_hidden_dims,
        )
        print(f"Modular teacher actor: {self.actor}")


def register_rsl_rl_teacher_actor_critic() -> None:
    """Expose the custom class to RSL-RL 3.0.1's runner-local lookup."""

    # RSL-RL 3.0.1 resolves ``policy.class_name`` with ``eval`` inside this
    # module rather than accepting an external class directly. Registering the
    # class in that namespace preserves the stock runner and PPO implementation.
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    setattr(
        on_policy_runner,
        PrivilegedTeacherActorCritic.__name__,
        PrivilegedTeacherActorCritic,
    )
