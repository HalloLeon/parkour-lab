# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour-specific RSL-RL environment adapter."""

from __future__ import annotations

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from tensordict import TensorDict

from .distillation.contracts import (
    ADAPTATION_HISTORY_GROUP,
    DEPLOYABLE_HISTORY_LENGTH,
    DEPLOYABLE_STATE_GROUP,
)

__all__ = ["RslRlHistoryWrapper"]


class RslRlHistoryWrapper(RslRlVecEnvWrapper):
    """Derive causal adaptation history from the exact policy observations delivered."""

    def __init__(self, env: gym.Env, clip_actions: float | None = None) -> None:
        super().__init__(env, clip_actions)
        self._history: torch.Tensor | None = None
        self._observations = self._with_history(super().get_observations(), reset=True)

    def reset(self) -> tuple[TensorDict, dict]:
        observations, extras = super().reset()
        return self._with_history(observations, reset=True), extras

    def get_observations(self) -> TensorDict:
        """Return the last delivered observations without sampling corruption again."""

        return self._observations

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        observations, rewards, dones, extras = super().step(actions)
        return self._with_history(observations, dones=dones), rewards, dones, extras

    def _with_history(
        self,
        observations: TensorDict,
        *,
        reset: bool = False,
        dones: torch.Tensor | None = None,
    ) -> TensorDict:
        policy_observation = observations[DEPLOYABLE_STATE_GROUP]
        if reset or self._history is None:
            self._history = policy_observation.unsqueeze(1).repeat(
                1, DEPLOYABLE_HISTORY_LENGTH, 1
            )
        else:
            self._history = torch.cat(
                (self._history[:, 1:], policy_observation.unsqueeze(1)), dim=1
            )
            if dones is not None:
                done = dones.bool()
                self._history[done] = policy_observation[done].unsqueeze(1)
        observations[ADAPTATION_HISTORY_GROUP] = self._history.flatten(1)
        self._observations = observations
        return observations
