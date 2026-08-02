# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL integration for the modular privileged Phase-1 teacher."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from ..architecture import (
    DEFAULT_TERRAIN_LATENT_DIM,
    MotorInterfaceCfg,
)
from ..contracts import (
    ADAPTATION_HISTORY_GROUP,
    PRIVILEGED_DYNAMICS_GROUP,
    PRIVILEGED_TERRAIN_GROUP,
    TEACHER_OBSERVATION_GROUPS,
)
from .model import (
    PrivilegedTeacherActor,
    PrivilegedTeacherModelCfg,
)

__all__ = [
    "PrivilegedTeacherActorCritic",
    "RegularizedPPO",
    "register_rsl_rl_teacher_actor_critic",
]


class PrivilegedTeacherActorCritic(ActorCritic):
    """RSL-RL actor-critic with a modular scan encoder and shared motor actor."""

    def __init__(
        self,
        obs: Mapping[str, torch.Tensor],
        obs_groups: dict[str, list[str]],
        num_actions: int,
        *,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: Literal["scalar", "log"] = "scalar",
        dynamics_encoder_hidden_dims: Sequence[int] = (128, 64),
        history_encoder_hidden_dims: Sequence[int] = (256, 128),
        scan_encoder_hidden_dims: Sequence[int] = (128, 64),
        terrain_latent_dim: int = DEFAULT_TERRAIN_LATENT_DIM,
        **kwargs: object,
    ) -> None:
        actor_groups = tuple(obs_groups["policy"])
        groups_without_terrain = tuple(
            group for group in TEACHER_OBSERVATION_GROUPS if group != PRIVILEGED_TERRAIN_GROUP
        )
        if actor_groups not in (groups_without_terrain, TEACHER_OBSERVATION_GROUPS):
            raise ValueError(
                "The modular teacher actor requires observation groups "
                f"{list(groups_without_terrain)} with optional "
                f"{PRIVILEGED_TERRAIN_GROUP!r}, got {list(actor_groups)}."
            )
        if activation != "elu":
            raise ValueError("The shared teacher/student motor actor requires ELU.")
        if actor_obs_normalization:
            raise ValueError(
                "The adaptation encoders require raw actor observations; "
                "actor_obs_normalization must remain disabled."
            )

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
            int(obs[PRIVILEGED_TERRAIN_GROUP].shape[-1]) if actor_groups == TEACHER_OBSERVATION_GROUPS else None
        )
        motor_cfg = MotorInterfaceCfg(
            state_dim=int(obs[TEACHER_OBSERVATION_GROUPS[0]].shape[-1]),
            heading_dim=int(obs[TEACHER_OBSERVATION_GROUPS[1]].shape[-1]),
            terrain_latent_dim=terrain_latent_dim,
            action_dim=num_actions,
            hidden_dims=tuple(actor_hidden_dims),
        )
        self.actor = PrivilegedTeacherActor(
            PrivilegedTeacherModelCfg(
                motor=motor_cfg,
                terrain_scan_dim=terrain_scan_dim,
                privileged_dynamics_dim=int(obs[PRIVILEGED_DYNAMICS_GROUP].shape[-1]),
                history_dim=int(obs[ADAPTATION_HISTORY_GROUP].shape[-1]),
                dynamics_hidden_dims=tuple(dynamics_encoder_hidden_dims),
                history_hidden_dims=tuple(history_encoder_hidden_dims),
                scan_hidden_dims=tuple(scan_encoder_hidden_dims),
            )
        )

        # Persist the ROA schedule position in normal policy checkpoints so a
        # resumed run does not restart its regularization ramp or rollout cycle.
        self.register_buffer("roa_update_count", torch.zeros((), dtype=torch.long))
        print(f"Modular teacher actor: {self.actor}")

    # Extend ActorCritic.act with ROA's history-conditioned path.
    def act(
        self,
        obs: Mapping[str, torch.Tensor],
        *,
        use_history: bool = False,
        **kwargs: object,
    ) -> torch.Tensor:
        """Sample from the privileged or history-conditioned actor."""

        if not use_history:
            return super().act(obs, **kwargs)

        mean = self.act_inference_from_history(obs)
        std: torch.Tensor
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)  # Direct per-action standard deviation.
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)  # Convert log standard deviation.
        else:
            raise ValueError(f"Unsupported noise_std_type: {self.noise_std_type!r}.")
        self.distribution = torch.distributions.Normal(mean, std)
        return self.distribution.sample()

    def act_inference_from_history(
        self,
        obs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return deterministic actions without using privileged dynamics."""

        terrain_scan = obs[PRIVILEGED_TERRAIN_GROUP] if PRIVILEGED_TERRAIN_GROUP in self.obs_groups["policy"] else None
        return self.actor.forward_from_history(
            obs[TEACHER_OBSERVATION_GROUPS[0]],
            obs[TEACHER_OBSERVATION_GROUPS[1]],
            terrain_scan,
            obs[ADAPTATION_HISTORY_GROUP],
        )


class RegularizedPPO(PPO):
    """PPO with bidirectional ROA losses and history-driven rollouts."""

    policy: PrivilegedTeacherActorCritic

    def __init__(
        self,
        policy: PrivilegedTeacherActorCritic,
        *,
        adaptation_loss_coef: float = 1.0,
        history_rollout_interval: int = 20,
        privileged_regularization_coef_end: float = 0.1,
        privileged_regularization_coef_start: float = 0.0,
        privileged_regularization_ramp_iterations: int = 300,
        privileged_regularization_warmup_iterations: int = 200,
        **kwargs: object,
    ) -> None:
        super().__init__(policy, **kwargs)
        if self.rnd is not None or self.symmetry is not None:
            raise ValueError("RegularizedPPO does not support RND or symmetry objectives.")
        if history_rollout_interval <= 0:
            raise ValueError("history_rollout_interval must be positive.")
        if privileged_regularization_ramp_iterations <= 0:
            raise ValueError("privileged_regularization_ramp_iterations must be positive.")
        if privileged_regularization_warmup_iterations < 0:
            raise ValueError("privileged_regularization_warmup_iterations cannot be negative.")
        self.adaptation_loss_coef = adaptation_loss_coef
        self.history_rollout_interval = history_rollout_interval
        self.privileged_regularization_coef_end = privileged_regularization_coef_end
        self.privileged_regularization_coef_start = privileged_regularization_coef_start
        self.privileged_regularization_ramp_iterations = privileged_regularization_ramp_iterations
        self.privileged_regularization_warmup_iterations = privileged_regularization_warmup_iterations
        self._history_rollout = False
        self._update_count: int | None = None

    # Override PPO.act to schedule rollouts through the history encoder.
    def act(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Collect a privileged- or history-conditioned rollout step."""

        if self._update_count is None:
            # Read the checkpointed counter once; subsequent iterations remain
            # in Python to avoid synchronizing the GPU at every environment step.
            self._update_count = int(self.policy.roa_update_count.item())
        self._history_rollout = (self._update_count + 1) % self.history_rollout_interval == 0
        self.transition.actions = self.policy.act(
            obs,
            use_history=self._history_rollout,
        ).detach()
        self.transition.values = self.policy.evaluate(obs).detach()  # Critic estimate for returns.
        self.transition.actions_log_prob = (  # Old log probability for the PPO ratio.
            self.policy.get_actions_log_prob(self.transition.actions).detach()
        )
        self.transition.action_mean = self.policy.action_mean.detach()  # Old Gaussian mean for KL.
        self.transition.action_sigma = self.policy.action_std.detach()  # Old Gaussian std for KL.
        self.transition.observations = obs
        return self.transition.actions

    # Override PPO.update to add ROA losses before the shared optimizer step.
    def update(self) -> dict[str, float]:
        """Optimize PPO together with both directed ROA objectives."""

        if self._update_count is None:
            self._update_count = int(self.policy.roa_update_count.item())
        regularization_coef = self._regularization_coefficient(self._update_count)

        mean_adaptation_loss = 0.0
        mean_entropy = 0.0
        mean_privileged_regularization_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_value_loss = 0.0
        generator = self.storage.mini_batch_generator(  # Yield shuffled rollout splits for each PPO epoch.
            self.num_mini_batches,
            self.num_learning_epochs,
        )
        # B: mini-batch size; A: action dimension.
        for (
            obs_batch,  # Observation groups, each shaped (B, group_dim).
            actions_batch,  # Rollout actions, shaped (B, A).
            target_values_batch,  # Old critic values, shaped (B, 1).
            advantages_batch,  # GAE advantages, shaped (B, 1).
            returns_batch,  # Bootstrapped critic targets, shaped (B, 1).
            old_actions_log_prob_batch,  # Rollout log probabilities, shaped (B, 1).
            old_mu_batch,  # Rollout Gaussian means, shaped (B, A).
            old_sigma_batch,  # Rollout Gaussian standard deviations, shaped (B, A).
            _,  # Unused recurrent-state placeholder.
            _,  # Unused recurrent-mask placeholder.
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1.0e-8)

            # Recompute the same actor path used to collect this rollout. This
            # keeps PPO's old and current action distributions comparable.
            self.policy.act(
                obs_batch,
                use_history=self._history_rollout,
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy
            self._adapt_learning_rate(
                mu_batch,
                sigma_batch,
                old_mu_batch,
                old_sigma_batch,
            )
            surrogate_loss, value_loss = self._ppo_losses(
                actions_log_prob_batch,
                advantages_batch,
                old_actions_log_prob_batch,
                returns_batch,
                target_values_batch,
                value_batch,
            )

            adaptation_loss, privileged_regularization_loss = self.policy.actor.roa_losses(
                obs_batch[ADAPTATION_HISTORY_GROUP],
                obs_batch[PRIVILEGED_DYNAMICS_GROUP],
            )
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.adaptation_loss_coef * adaptation_loss
                + regularization_coef * privileged_regularization_loss
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
            )
            self.optimizer.step()

            mean_adaptation_loss += float(adaptation_loss.detach().item())
            mean_entropy += float(entropy_batch.mean().detach().item())
            mean_privileged_regularization_loss += float(privileged_regularization_loss.detach().item())
            mean_surrogate_loss += float(surrogate_loss.detach().item())
            mean_value_loss += float(value_loss.detach().item())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        self.storage.clear()
        self._update_count += 1
        self.policy.roa_update_count.fill_(self._update_count)
        return {
            "adaptation": mean_adaptation_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "history_rollout": float(self._history_rollout),
            "privileged_regularization": mean_privileged_regularization_loss / num_updates,
            "privileged_regularization_coefficient": regularization_coef,
            "surrogate": mean_surrogate_loss / num_updates,
            "value_function": mean_value_loss / num_updates,
        }

    def _adapt_learning_rate(
        self,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        old_action_mean: torch.Tensor,
        old_action_std: torch.Tensor,
    ) -> None:
        """Apply RSL-RL's adaptive KL learning-rate schedule."""

        if self.desired_kl is None or self.schedule != "adaptive":
            return

        with torch.inference_mode():
            kl = torch.sum(
                torch.log(action_std / old_action_std + 1.0e-5)
                + (torch.square(old_action_std) + torch.square(old_action_mean - action_mean))
                / (2.0 * torch.square(action_std))
                - 0.5,
                dim=-1,
            )
            kl_mean = torch.mean(kl)
            # The policy moved too far from the rollout policy, so take smaller optimization steps.
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            # The policy update was conservative, so allow larger steps while keeping the rate bounded.
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
                self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = self.learning_rate

    def _ppo_losses(
        self,
        actions_log_prob: torch.Tensor,
        advantages: torch.Tensor,
        old_actions_log_prob: torch.Tensor,
        returns: torch.Tensor,
        target_values: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the clipped policy-surrogate and value-function losses."""

        ratio = torch.exp(actions_log_prob - torch.squeeze(old_actions_log_prob))
        surrogate = -torch.squeeze(advantages) * ratio
        surrogate_clipped = -torch.squeeze(advantages) * torch.clamp(
            ratio,
            1.0 - self.clip_param,
            1.0 + self.clip_param,
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        if not self.use_clipped_value_loss:
            return surrogate_loss, torch.square(returns - values).mean()

        values_clipped = target_values + (values - target_values).clamp(
            -self.clip_param,
            self.clip_param,
        )
        value_loss = torch.max(
            torch.square(values - returns),
            torch.square(values_clipped - returns),
        ).mean()
        return surrogate_loss, value_loss

    def _regularization_coefficient(self, update_count: int) -> float:
        """Return the warmup-and-ramp coefficient for privileged ROA."""

        progress = min(
            max(
                update_count - self.privileged_regularization_warmup_iterations,
                0,
            )
            / self.privileged_regularization_ramp_iterations,
            1.0,
        )
        return self.privileged_regularization_coef_start + progress * (
            self.privileged_regularization_coef_end - self.privileged_regularization_coef_start
        )


def register_rsl_rl_teacher_actor_critic() -> None:
    """Expose the custom policy and algorithm to RSL-RL's runner lookup."""

    # RSL-RL 3.0.1 resolves policy and algorithm class names with ``eval``
    # inside this module rather than accepting external classes directly.
    # Registering both names preserves the stock runner.
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    setattr(
        on_policy_runner,
        PrivilegedTeacherActorCritic.__name__,
        PrivilegedTeacherActorCritic,
    )
    setattr(on_policy_runner, RegularizedPPO.__name__, RegularizedPPO)
