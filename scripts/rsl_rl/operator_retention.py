# PPO update portions adapted from RSL-RL 3.1.2 (BSD-3-Clause).
# Copyright (c) 2021-2025, ETH Zurich and NVIDIA Corporation.
# Copyright (c) 2025, ETH Zurich
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Opt-in source-mean retention on learner-visited moving-command states.

This is a soft training loss, not action blending or a behavioral guarantee.
The supported PPO update is deliberately restricted to the pinned stock path.
"""

from __future__ import annotations

import copy
import importlib.metadata

import torch
from torch import nn


VERSION = "moving_anchor_v1"
COEFFICIENT = 0.1
ACTION_SCALE = 0.1  # Raw policy actions; 0.025 rad after the stock action scale.


def retention_manifest():
    return {
        "version": VERSION,
        "reference": "exact frozen source actor from this repair's initial checkpoint",
        "observations": "current pre-action 48-D learner rollout observations",
        "mask": "either delivered planar velocity command is nonzero; both yaw signs",
        "loss": "coefficient * moving-row mean over 12 joints of squared mean-action difference / (2 * action_scale**2)",
        "coefficient": COEFFICIENT,
        "action_scale_raw": ACTION_SCALE,
        "action_scale_joint_target_rad": 0.25 * ACTION_SCALE,
        "reference_gradient": False,
        "action_noise_regularized": False,
        "reference_selects_actions": False,
        "hard_constraint": False,
        "reward_changed": False,
        "rsl_rl_version": "3.1.2",
    }


def moving_anchor_loss(mean, reference_mean, observation):
    """Conditional moving-row mean, with a differentiable zero for all holds.

    Commands are indices 9:12 of the validated stock observation; the gate
    uses commanded, not measured, velocity. A stalled moving robot still counts.
    """
    if (
        mean.ndim != 2
        or len(mean) == 0
        or mean.shape[1] != 12
        or reference_mean.shape != mean.shape
        or observation.shape != (len(mean), 48)
        or any(
            x.device != mean.device or x.dtype != mean.dtype
            for x in (reference_mean, observation)
        )
        or not all(torch.isfinite(x).all() for x in (mean, reference_mean, observation))
    ):
        raise ValueError(
            "Require matching finite 48-D observations and 12-D action means"
        )
    moving = (observation.detach()[:, 9:11] != 0).any(dim=1)
    squared = (mean - reference_mean.detach()).square().mean(dim=1)
    mean_squared = (squared * moving).sum() / moving.sum().clamp_min(1)
    return (
        mean_squared / (2 * ACTION_SCALE**2),
        mean_squared.detach(),
        moving.float().mean(),
    )


class MovingRetentionUpdate:
    """Pinned single-GPU, fixed-rate, feed-forward PPO plus one explicit loss.

    Installed after exact source restoration; only ``alg.update`` is replaced.
    The normal rollout, value bootstrap, optimizer, policy state and exporter
    stay unchanged. The reference is not registered as a policy submodule.

    PPO equations/loop adapted from rsl_rl/algorithms/ppo.py v3.1.2:
    Copyright (c) 2021-2025 ETH Zurich and NVIDIA Corporation.
    SPDX-License-Identifier: BSD-3-Clause
    """

    def __init__(self, algorithm):
        from rsl_rl.algorithms import PPO
        from rsl_rl.modules import ActorCritic

        if importlib.metadata.version("rsl-rl-lib") != "3.1.2":
            raise ValueError("Moving retention supports only RSL-RL 3.1.2")
        policy = algorithm.policy
        if (
            type(algorithm) is not PPO
            or type(policy) is not ActorCritic
            or policy.is_recurrent
            or policy.actor_obs_normalization
            or policy.critic_obs_normalization
            or policy.state_dependent_std
            or policy.noise_std_type != "scalar"
            or policy.obs_groups != {"policy": ["policy"], "critic": ["policy"]}
            or algorithm.schedule != "fixed"
            or algorithm.is_multi_gpu
            or algorithm.rnd is not None
            or algorithm.symmetry is not None
            or algorithm.optimizer.state
            or policy.actor[0].in_features != 48
            or policy.actor[-1].out_features != 12
        ):
            raise ValueError(
                "Moving retention requires the fresh stock fixed-rate PPO path"
            )
        self.algorithm = algorithm
        self.reference = copy.deepcopy(policy.actor).eval().requires_grad_(False)
        self.coefficient = COEFFICIENT
        self.updates = 0

    def __call__(self):
        alg = self.algorithm
        totals = dict(
            value_function=0.0,
            surrogate=0.0,
            entropy=0.0,
            moving_anchor=0.0,
            moving_action_mse=0.0,
            moving_fraction=0.0,
        )
        generator = alg.storage.mini_batch_generator(
            alg.num_mini_batches, alg.num_learning_epochs
        )
        count = 0
        for (
            obs,
            actions,
            target_values,
            advantages,
            returns,
            old_log_prob,
            _old_mu,
            _old_sigma,
            hidden_states,
            masks,
        ) in generator:
            if alg.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )
            # Preserve upstream sampling/RNG order even though PPO uses its mean.
            alg.policy.act(obs, masks=masks, hidden_state=hidden_states[0])
            log_prob = alg.policy.get_actions_log_prob(actions)
            values = alg.policy.evaluate(
                obs, masks=masks, hidden_state=hidden_states[1]
            )
            mean, entropy = alg.policy.action_mean, alg.policy.entropy
            ratio = torch.exp(log_prob - torch.squeeze(old_log_prob))
            surrogate = -torch.squeeze(advantages) * ratio
            clipped = -torch.squeeze(advantages) * ratio.clamp(
                1 - alg.clip_param, 1 + alg.clip_param
            )
            surrogate_loss = torch.max(surrogate, clipped).mean()
            if alg.use_clipped_value_loss:
                clipped_value = target_values + (values - target_values).clamp(
                    -alg.clip_param, alg.clip_param
                )
                value_loss = torch.max(
                    (values - returns).square(), (clipped_value - returns).square()
                ).mean()
            else:
                value_loss = (returns - values).square().mean()
            observation = obs["policy"].detach()
            with torch.no_grad():
                reference_mean = self.reference(observation)
            anchor, mse, fraction = moving_anchor_loss(
                mean, reference_mean, observation
            )
            loss = (
                surrogate_loss
                + alg.value_loss_coef * value_loss
                - alg.entropy_coef * entropy.mean()
            )
            loss = loss + self.coefficient * anchor
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite retention PPO loss")
            alg.optimizer.zero_grad()
            loss.backward()
            # One shared backward/clip/Adam step; never add unbounded gradients
            # after clipping, and never optimize the reference, labels or inputs.
            norm = nn.utils.clip_grad_norm_(alg.policy.parameters(), alg.max_grad_norm)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite retention PPO gradient")
            alg.optimizer.step()
            for name, value in zip(
                totals,
                (value_loss, surrogate_loss, entropy.mean(), anchor, mse, fraction),
                strict=True,
            ):
                totals[name] += float(value.detach())
            count += 1
        if count != alg.num_mini_batches * alg.num_learning_epochs:
            raise RuntimeError("Incomplete retention PPO minibatch update")
        alg.storage.clear()
        self.updates += 1
        return {name: value / count for name, value in totals.items()}


def install_moving_retention(algorithm):
    update = MovingRetentionUpdate(algorithm)
    algorithm.update = update
    return update
