# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL integration for the modular privileged Phase-1 teacher."""

from __future__ import annotations

import math
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
    SUPPORTED_TEACHER_OBSERVATION_GROUPS,
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
        state_dependent_std: bool = False,
        min_noise_std: float = 0.05,
        max_noise_std: float = 1.5,
        dynamics_encoder_hidden_dims: Sequence[int] = (128, 64),
        history_encoder_hidden_dims: Sequence[int] = (256, 128),
        scan_encoder_hidden_dims: Sequence[int] = (128, 64),
        terrain_latent_dim: int = DEFAULT_TERRAIN_LATENT_DIM,
        **kwargs: object,
    ) -> None:
        actor_groups = tuple(obs_groups["policy"])
        if actor_groups not in SUPPORTED_TEACHER_OBSERVATION_GROUPS:
            raise ValueError(
                "The modular teacher actor requires one of the supported observation routes "
                f"{[list(route) for route in SUPPORTED_TEACHER_OBSERVATION_GROUPS]}, "
                f"got {list(actor_groups)}."
            )
        if activation != "elu":
            raise ValueError("The modular teacher motor actor requires ELU.")
        if actor_obs_normalization:
            raise ValueError(
                "The adaptation encoders require raw actor observations; actor_obs_normalization must remain disabled."
            )
        if state_dependent_std:
            raise ValueError(
                "The modular teacher requires state-independent action noise."
            )
        if (
            not math.isfinite(min_noise_std)
            or not math.isfinite(max_noise_std)
            or min_noise_std <= 0.0
            or max_noise_std < min_noise_std
        ):
            raise ValueError(
                "Action-noise bounds must be finite and satisfy 0 < min_noise_std <= max_noise_std."
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
            state_dependent_std=False,
            **kwargs,
        )
        self.min_noise_std = min_noise_std
        self.max_noise_std = max_noise_std

        terrain_scan_dim = (
            int(obs[PRIVILEGED_TERRAIN_GROUP].shape[-1])
            if PRIVILEGED_TERRAIN_GROUP in actor_groups
            else None
        )
        motor_cfg = MotorInterfaceCfg(
            state_dim=int(obs[TEACHER_OBSERVATION_GROUPS[0]].shape[-1]),
            travel_direction_dim=int(obs[TEACHER_OBSERVATION_GROUPS[1]].shape[-1]),
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
        self.enforce_action_std_bounds_()
        print(f"Modular teacher actor: {self.actor}")

    def act(
        self,
        obs: Mapping[str, torch.Tensor],
        *,
        use_history: bool = False,
        **_: object,
    ) -> torch.Tensor:
        """Sample from the privileged or history-conditioned actor."""

        mean = self._actor_mean(obs, use_history=use_history)
        self.distribution = torch.distributions.Normal(
            mean,
            self._bounded_action_std(mean),
        )
        return self.distribution.sample()

    def act_inference_from_history(
        self,
        obs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Return deterministic actions without using privileged dynamics."""

        return self._actor_mean(obs, use_history=True)

    def _actor_mean(
        self,
        obs: Mapping[str, torch.Tensor],
        *,
        use_history: bool,
        actor: PrivilegedTeacherActor | None = None,
    ) -> torch.Tensor:
        """Evaluate either actor path, optionally with a frozen actor snapshot."""

        actor = self.actor if actor is None else actor
        if not use_history:
            actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
            return actor(actor_obs)
        terrain_scan = (
            obs[PRIVILEGED_TERRAIN_GROUP]
            if PRIVILEGED_TERRAIN_GROUP in self.obs_groups["policy"]
            else None
        )
        return actor.forward_from_history(
            obs[TEACHER_OBSERVATION_GROUPS[0]],
            obs[TEACHER_OBSERVATION_GROUPS[1]],
            terrain_scan,
            obs[ADAPTATION_HISTORY_GROUP],
        )

    def enforce_action_std_bounds_(self) -> None:
        """Project the learned noise parameter into its configured safe range."""

        with torch.no_grad():
            if self.noise_std_type == "scalar":
                parameter = self.std
                lower_bound = self.min_noise_std
                upper_bound = self.max_noise_std
            elif self.noise_std_type == "log":
                parameter = self.log_std
                lower_bound = math.log(self.min_noise_std)
                upper_bound = math.log(self.max_noise_std)
            else:
                raise ValueError(
                    f"Unsupported noise_std_type: {self.noise_std_type!r}."
                )
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(
                    "The learned action-noise parameter became non-finite."
                )
            parameter.clamp_(min=lower_bound, max=upper_bound)

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
    ) -> bool:
        """Load a compatible checkpoint and project its action noise safely."""

        result = super().load_state_dict(state_dict, strict=strict)
        self.enforce_action_std_bounds_()
        return result

    def _bounded_action_std(self, mean: torch.Tensor) -> torch.Tensor:
        """Return the bounded positive per-action standard deviation."""

        if self.noise_std_type == "scalar":
            std = self.std.clamp(
                min=self.min_noise_std,
                max=self.max_noise_std,
            )
        elif self.noise_std_type == "log":
            log_std = self.log_std.clamp(
                min=math.log(self.min_noise_std),
                max=math.log(self.max_noise_std),
            )
            std = torch.exp(log_std)
        else:
            raise ValueError(f"Unsupported noise_std_type: {self.noise_std_type!r}.")
        return std.expand_as(mean)


class RegularizedPPO(PPO):
    """PPO with reflected transitions, bidirectional ROA, and history rollouts."""

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
        max_learning_rate: float = 1.0e-3,
        **kwargs: object,
    ) -> None:
        super().__init__(policy, **kwargs)
        if self.is_multi_gpu:
            raise ValueError("RegularizedPPO supports only single-process training.")
        if self.rnd is not None:
            raise ValueError("RegularizedPPO does not support RND objectives.")
        if self.symmetry is not None:
            if not self.symmetry["use_data_augmentation"]:
                raise ValueError("RegularizedPPO symmetry requires data augmentation.")
            if self.symmetry["use_mirror_loss"]:
                raise ValueError(
                    "RegularizedPPO does not support a symmetry mirror loss."
                )
        if history_rollout_interval <= 0:
            raise ValueError("history_rollout_interval must be positive.")
        if privileged_regularization_ramp_iterations <= 0:
            raise ValueError(
                "privileged_regularization_ramp_iterations must be positive."
            )
        if privileged_regularization_warmup_iterations < 0:
            raise ValueError(
                "privileged_regularization_warmup_iterations cannot be negative."
            )
        if max_learning_rate <= 0.0 or max_learning_rate < self.learning_rate:
            raise ValueError(
                "max_learning_rate must be positive and no smaller than the initial learning rate."
            )
        if (
            not math.isfinite(self.clip_param)
            or self.clip_param <= 0.0
            or self.clip_param >= 1.0
        ):
            raise ValueError(
                "clip_param must be finite and lie strictly between 0 and 1."
            )
        self.adaptation_loss_coef = adaptation_loss_coef
        self.history_rollout_interval = history_rollout_interval
        self.privileged_regularization_coef_end = privileged_regularization_coef_end
        self.privileged_regularization_coef_start = privileged_regularization_coef_start
        self.privileged_regularization_ramp_iterations = (
            privileged_regularization_ramp_iterations
        )
        self.privileged_regularization_warmup_iterations = (
            privileged_regularization_warmup_iterations
        )
        self.max_learning_rate = max_learning_rate
        self._history_rollout = False
        self._update_count: int | None = None

    # Override PPO.act to schedule rollouts through the history encoder.
    def act(self, obs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Collect a privileged- or history-conditioned rollout step."""

        if self._update_count is None:
            # Read the checkpointed counter once; subsequent iterations remain
            # in Python to avoid synchronizing the GPU at every environment step.
            self._update_count = int(self.policy.roa_update_count.item())
        self._history_rollout = (
            self._update_count + 1
        ) % self.history_rollout_interval == 0
        self.transition.actions = self.policy.act(
            obs,
            use_history=self._history_rollout,
        ).detach()
        self.transition.values = self.policy.evaluate(
            obs
        ).detach()  # Critic estimate for returns.
        self.transition.actions_log_prob = (
            self.policy.get_actions_log_prob(  # Old log probability for the PPO ratio.
                self.transition.actions
            ).detach()
        )
        self.transition.action_mean = (
            self.policy.action_mean.detach()
        )  # Old Gaussian mean for KL.
        self.transition.action_sigma = (
            self.policy.action_std.detach()
        )  # Old Gaussian std for KL.
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
            original_batch_size = obs_batch.batch_size[0]
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (
                        advantages_batch.std(unbiased=False) + 1.0e-8
                    )

            # Append reflected transitions after the untouched mini-batch. The
            # augmentation function transforms observations and rollout actions;
            # scalar PPO targets describe the same reflected transition and are
            # therefore repeated. Old Gaussian parameters remain unaugmented
            # because adaptive KL is deliberately measured on collected samples.
            if self.symmetry is not None:
                obs_batch, actions_batch = self.symmetry["data_augmentation_func"](
                    env=self.symmetry["_env"],
                    obs=obs_batch,
                    actions=actions_batch,
                )
                if obs_batch is None or actions_batch is None:
                    raise RuntimeError(
                        "Symmetry augmentation must return observations and actions."
                    )
                augmented_batch_size = obs_batch.batch_size[0]
                if (
                    augmented_batch_size <= original_batch_size
                    or augmented_batch_size % original_batch_size != 0
                ):
                    raise RuntimeError(
                        "Symmetry augmentation must append complete copies of the original mini-batch."
                    )
                if actions_batch.shape[0] != augmented_batch_size:
                    raise RuntimeError(
                        "Symmetry augmentation returned inconsistent observation and action batches."
                    )
                num_augmentations = augmented_batch_size // original_batch_size
                # Reflection pushes each rollout action through a signed
                # permutation with unit Jacobian. The reflected behavior density
                # is therefore the collected action's density, not the old
                # actor's generally different density at the reflected state.
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(
                    num_augmentations, 1
                )
                target_values_batch = target_values_batch.repeat(num_augmentations, 1)
                advantages_batch = advantages_batch.repeat(num_augmentations, 1)
                returns_batch = returns_batch.repeat(num_augmentations, 1)

            # Recompute the same actor path used to collect this rollout. This
            # keeps PPO's old and current action distributions comparable.
            self.policy.act(
                obs_batch,
                use_history=self._history_rollout,
            )
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
            # The first slice is the unmodified rollout batch by contract. Keep
            # KL scheduling and entropy pressure independent of augmentation
            # multiplicity, while PPO and ROA train on all reflected samples.
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]
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

            adaptation_loss, privileged_regularization_loss = (
                self.policy.actor.roa_losses(
                    obs_batch[ADAPTATION_HISTORY_GROUP],
                    obs_batch[PRIVILEGED_DYNAMICS_GROUP],
                )
            )
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.adaptation_loss_coef * adaptation_loss
                + regularization_coef * privileged_regularization_loss
            )

            loss_is_finite = torch.isfinite(loss).to(dtype=torch.int32)
            if not loss_is_finite:
                raise FloatingPointError(
                    "RegularizedPPO produced a non-finite loss before the "
                    "optimizer step: "
                    f"surrogate={surrogate_loss.detach().item():.6g}, "
                    f"value={value_loss.detach().item():.6g}, "
                    f"adaptation={adaptation_loss.detach().item():.6g}, "
                    "privileged_regularization="
                    f"{privileged_regularization_loss.detach().item():.6g}."
                )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                self.max_grad_norm,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            self.policy.enforce_action_std_bounds_()

            mean_adaptation_loss += float(adaptation_loss.detach().item())
            mean_entropy += float(entropy_batch.mean().detach().item())
            mean_privileged_regularization_loss += float(
                privileged_regularization_loss.detach().item()
            )
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
            "privileged_regularization": mean_privileged_regularization_loss
            / num_updates,
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
                + (
                    torch.square(old_action_std)
                    + torch.square(old_action_mean - action_mean)
                )
                / (2.0 * torch.square(action_std))
                - 0.5,
                dim=-1,
            )
            kl_mean = torch.mean(kl)
            if not torch.isfinite(kl_mean):
                raise FloatingPointError(
                    "The adaptive PPO KL divergence became non-finite."
                )
            # The policy moved too far from the rollout policy, so take smaller optimization steps.
            if kl_mean > self.desired_kl * 2.0:
                self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            # The policy update was conservative, so allow larger steps while keeping the rate bounded.
            elif 0.0 < kl_mean < self.desired_kl / 2.0:
                self.learning_rate = min(
                    self.max_learning_rate,
                    self.learning_rate * 1.5,
                )
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

        advantages = torch.squeeze(advantages)
        log_ratio = actions_log_prob - torch.squeeze(old_actions_log_prob)
        if not torch.isfinite(log_ratio).all():
            raise FloatingPointError(
                "RegularizedPPO received a non-finite action log-probability ratio."
            )
        # This is algebraically identical to max(-A*r, -A*clip(r)), but chooses
        # the active PPO branch in log space before exponentiation. That avoids
        # evaluating an overflowing exponential on an inactive clipped branch
        # without capping a ratio that the textbook objective leaves unbounded.
        effective_log_ratio = torch.where(
            advantages >= 0.0,
            log_ratio.clamp_max(math.log1p(self.clip_param)),
            log_ratio.clamp_min(math.log1p(-self.clip_param)),
        )
        surrogate_loss = (-advantages * torch.exp(effective_log_ratio)).mean()
        if not torch.isfinite(surrogate_loss):
            raise FloatingPointError(
                "RegularizedPPO produced a non-finite policy surrogate; "
                "the behavior-policy likelihood ratio left the numerically "
                "representable PPO domain."
            )

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
            self.privileged_regularization_coef_end
            - self.privileged_regularization_coef_start
        )


def register_rsl_rl_teacher_actor_critic() -> None:
    """Expose the custom policy and algorithm to RSL-RL's runner lookup."""

    # RSL-RL 3.1.2 resolves policy and algorithm class names with ``eval``
    # inside this module rather than accepting external classes directly.
    # Registering both names preserves the stock runner.
    import rsl_rl.runners.on_policy_runner as on_policy_runner

    setattr(
        on_policy_runner,
        PrivilegedTeacherActorCritic.__name__,
        PrivilegedTeacherActorCritic,
    )
    setattr(on_policy_runner, RegularizedPPO.__name__, RegularizedPPO)
