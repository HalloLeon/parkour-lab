"""Explicit fixed-RSL PPO objective for the bounded ROA training pilot."""

import math

import torch
from torch import nn
from rsl_rl.algorithms import PPO

from .operator_roa import ROAActor, gradient_norms, set_phase, state_sha256


class ROAPPO(PPO):
    """RSL-RL 3.1.2 storage/GAE with an explicit directed ROA PPO objective.

    No RND, symmetry, adaptive LR, recurrent storage or multi-GPU. Before any
    update, replay each original-size batch and require exact rollout log-probs.
    The history estimator is excluded from Adam, not just gradient-detached.
    """

    def __init__(self, policy, *, regularization_coef=0.1, **kwargs):
        if type(policy.actor) is not ROAActor or policy.is_recurrent:
            raise ValueError("ROAPPO requires the explicit feedforward ROA actor")
        for key in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg", "desired_kl"):
            if kwargs.get(key) is not None:
                raise ValueError(f"ROAPPO does not support {key}")
        if kwargs.get("schedule", "fixed") != "fixed":
            raise ValueError("ROAPPO requires fixed learning rate")
        kwargs.update(schedule="fixed", desired_kl=None)
        super().__init__(policy, **kwargs)
        self.regularization_coef = regularization_coef
        estimator = {id(p) for p in policy.actor.estimator.parameters()}
        self.ppo_parameters = tuple(
            p for p in policy.parameters() if id(p) not in estimator
        )
        self.optimizer = torch.optim.Adam(self.ppo_parameters, lr=self.learning_rate)
        set_phase(policy, "privileged")

    def _check_phase(self):
        if (
            self.policy.obs_groups
            != {
                "policy": ["policy", "dynamics", "history"],
                "critic": ["critic_state", "terrain"],
            }
            or self.policy.actor_obs_normalization
            or self.policy.critic_obs_normalization
            or any(p.requires_grad for p in self.policy.actor.estimator.parameters())
            or not all(p.requires_grad for p in self.ppo_parameters)
            or {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            != {id(p) for p in self.ppo_parameters}
        ):
            raise ValueError(
                "PPO requires privileged routing and exclusive optimizer ownership"
            )

    def act(self, obs):
        self._check_phase()
        return super().act(obs)

    @torch.no_grad()
    def verify_first_replay(self):
        self._check_phase()
        if (
            self.storage is None
            or self.storage.step != self.storage.num_transitions_per_env
        ):
            raise ValueError("Require a complete privileged rollout before PPO replay")
        for step in range(self.storage.step):
            self.policy._update_distribution(
                self.policy.get_actor_obs(self.storage.observations[step])
            )
            actual = self.policy.get_actions_log_prob(self.storage.actions[step])
            expected = self.storage.actions_log_prob[step].squeeze(-1)
            if not (
                torch.equal(actual, expected)
                and torch.equal(self.policy.action_mean, self.storage.mu[step])
                and torch.equal(self.policy.action_std, self.storage.sigma[step])
            ):
                raise RuntimeError(
                    "First PPO replay differs from collected policy before any update"
                )
        return {
            "samples": self.storage.step * self.storage.num_envs,
            "log_prob_abs_max": 0.0,
            "ratio_min": 1.0,
            "ratio_max": 1.0,
        }

    def update(self):
        coefficient = self.regularization_coef
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (float, int))
            or not math.isfinite(coefficient)
            or coefficient < 0
        ):
            raise ValueError("ROA coefficient must be finite and nonnegative")
        replay = self.verify_first_replay()
        total = self.storage.num_envs * self.storage.num_transitions_per_env
        if (
            min(self.num_mini_batches, self.num_learning_epochs) < 1
            or total % self.num_mini_batches
        ):
            raise ValueError("ROA minibatches must cover every rollout row exactly")
        sums = {
            name: 0.0
            for name in ("value_function", "surrogate", "entropy", "regularization")
        }
        gradient_max = {
            "encoder_gradient_l2_max": 0.0,
            "projection_gradient_l2_max": 0.0,
        }
        estimator_before = state_sha256(self.policy.actor.estimator)
        count = 0
        for (
            obs,
            actions,
            old_values,
            advantages,
            returns,
            old_log_prob,
            _,
            _,
            _,
            _,
        ) in self.storage.mini_batch_generator(
            self.num_mini_batches, self.num_learning_epochs
        ):
            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )
            self.policy._update_distribution(self.policy.get_actor_obs(obs))
            log_prob = self.policy.get_actions_log_prob(actions)
            values = self.policy.evaluate(obs)
            entropy = self.policy.entropy.mean()
            ratio = (log_prob - old_log_prob.squeeze(-1)).exp()
            advantage = advantages.squeeze(-1)
            surrogate = torch.maximum(
                -advantage * ratio,
                -advantage * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
            ).mean()
            value_error = (values - returns).square()
            if self.use_clipped_value_loss:
                clipped = old_values + (values - old_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_error = torch.maximum(value_error, (clipped - returns).square())
            value_loss = value_error.mean()
            regularization = self.policy.actor.regularization_loss(obs)
            loss = (
                surrogate
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
                + coefficient * regularization
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite ROA PPO objective")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, value in gradient_norms(self.policy.actor).items():
                if value is None or not math.isfinite(value):
                    raise RuntimeError("Missing or nonfinite ROA branch gradient")
                gradient_max[name + "_max"] = max(gradient_max[name + "_max"], value)
            nn.utils.clip_grad_norm_(
                self.ppo_parameters, self.max_grad_norm, error_if_nonfinite=True
            )
            self.optimizer.step()
            if (
                any(not torch.isfinite(p).all() for p in self.ppo_parameters)
                or (self.policy.std <= 0).any()
            ):
                raise RuntimeError(
                    "ROA update produced invalid parameters or action noise"
                )
            for name, value in zip(
                sums, (value_loss, surrogate, entropy, regularization), strict=True
            ):
                sums[name] += float(value.detach())
            count += 1
        if state_sha256(self.policy.actor.estimator) != estimator_before:
            raise RuntimeError("PPO unexpectedly changed the causal estimator")
        self.storage.clear()
        return {
            **{name: value / count for name, value in sums.items()},
            "regularization_coef": float(coefficient),
            "first_replay": replay,
            "optimizer_steps": count,
            **gradient_max,
        }
