"""ROA training method: PPO collection/update and teacher-to-history fitting."""

import math

import torch
from torch import nn
from rsl_rl.algorithms import PPO

from .operator_roa import (
    FRAME_DIM,
    HISTORY_LENGTH,
    ROAActor,
    gradient_norms,
    set_phase,
    state_sha256,
)


class ROATrainingMethod:
    """Real ROA collection/update/adaptation with the legacy checkpoint payload.

    The pilot still owns ROA admission, scheduling and diagnostic gates. This
    adapter isolates the learner lifecycle without claiming that its simulator
    provider or experiment runner can already host an arbitrary learner.
    """

    def __init__(
        self,
        policy,
        observations,
        *,
        num_envs,
        rollout_steps,
        history_steps,
        adaptation_epochs,
        adaptation_batches,
        adaptation_learning_rate,
        **ppo_options,
    ):
        self.policy = policy
        self.rollout_steps = rollout_steps
        self.history_options = {
            "history_steps": history_steps,
            "epochs": adaptation_epochs,
            "minibatches": adaptation_batches,
        }
        self.algorithm = ROAPPO(policy, **ppo_options)
        self.algorithm.init_storage("rl", num_envs, rollout_steps, observations, [12])
        self.adaptation_optimizer = torch.optim.Adam(
            policy.actor.estimator.parameters(), lr=adaptation_learning_rate
        )

    @torch.no_grad()
    def collect(self, environment, observations, *, before_step=None, after_step=None):
        """Collect one PPO batch; the caller configures its regularization."""
        algorithm = self.algorithm
        algorithm.start_phase()
        for _ in range(self.rollout_steps):
            if before_step is not None:
                before_step()
            action = algorithm.act(observations)
            command = (
                observations["policy"][:, 6:9].clone()
                if after_step is not None
                else None
            )
            observations, reward, done, extras = environment.step(
                action, algorithm.phase + "_ppo"
            )
            if after_step is not None:
                after_step(done, command)
            algorithm.process_env_step(observations, reward, done.long(), extras)
        algorithm.compute_returns(observations)
        return observations

    def update(self, observations, *, observer=None):
        if observer is not None and observer.active:
            observer.capture_rollout(self.algorithm, observations)
            return self.algorithm.update(after_update=observer.after_update)
        return self.algorithm.update()

    def adapt(self, environment, observations, record, report, **options):
        return adapt_history(
            environment,
            self.policy,
            self.adaptation_optimizer,
            observations,
            record,
            report,
            **self.history_options,
            **options,
        )

    def state_dict(self):
        """Snapshot at an update boundary; no simulator/RNG resume claim."""
        if self.algorithm.storage.step:
            raise ValueError("Cannot checkpoint an in-flight ROA rollout")
        self.algorithm.check_fixed_action_std()
        return {
            "policy_state": self.policy.state_dict(),
            "ppo_optimizer": self.algorithm.optimizer.state_dict(),
            "adaptation_optimizer": self.adaptation_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        """Restore a verified, same-configuration learning snapshot, not a rollout."""
        if self.algorithm.storage.step:
            raise ValueError("Cannot restore over an in-flight ROA rollout")
        if set(state) != {"policy_state", "ppo_optimizer", "adaptation_optimizer"}:
            raise ValueError("ROA learning state requires policy and both optimizers")
        parameters = state["policy_state"]
        std = parameters.get("std")
        if (
            not isinstance(std, torch.Tensor)
            or std.shape != self.policy.std.shape
            or (std <= 0).any()
            or any(not torch.isfinite(value).all() for value in parameters.values())
        ):
            raise ValueError(
                "ROA learning state requires finite weights and positive action std"
            )
        fixed = self.algorithm.fixed_action_std
        if fixed is not None and not torch.equal(std.to(fixed), fixed):
            raise ValueError("ROA learning state differs from the fixed action std")
        self.policy.load_state_dict(parameters, strict=True)
        self.algorithm.optimizer.load_state_dict(state["ppo_optimizer"])
        self.adaptation_optimizer.load_state_dict(state["adaptation_optimizer"])
        self.algorithm.learning_rate = self.algorithm.optimizer.param_groups[0]["lr"]
        self.algorithm.start_phase()


def adapt_history(
    host,
    policy,
    optimizer,
    obs,
    record,
    report,
    *,
    require_change,
    history_steps,
    epochs,
    minibatches,
    observe=None,
    after_step=None,
    velocity_head=None,
):
    """Collect with fixed causal weights, then fit only the history estimator."""
    actor, env = policy.actor, host.env

    def frames(observations):
        return observations["history"].reshape(-1, HISTORY_LENGTH, FRAME_DIM)

    def privileged_hashes():
        return {
            name: state_sha256(module)
            for name, module in (
                ("motor", actor.motor),
                ("encoder", actor.encoder),
                ("critic", policy.critic),
            )
        }

    # Collect the whole history-owned block with fixed weights. Fit only
    # afterwards, so no action can incorporate its own privileged label.
    set_phase(policy, "frozen")
    fixed_hashes = privileged_hashes()
    fixed_std = policy.std.detach().clone()
    fixed_estimator = state_sha256(actor.estimator)
    samples = []
    with torch.no_grad():
        for _ in range(history_steps):
            if observe is not None:
                observe()
            sample = (frames(obs).clone(), obs["critic_state"][:, :3].clone())
            if velocity_head is None:
                sample += (actor.privileged_input(obs).clone(),)
            samples.append(sample)
            action = actor.history_action(obs["policy"], frames(obs))
            command = obs["policy"][:, 6:9].clone() if after_step is not None else None
            obs, _, done, _ = host.step(action, "history_adaptation")
            if after_step is not None:
                after_step(done, command)
    record["estimator_unchanged_during_history_collection"] = (
        state_sha256(actor.estimator) == fixed_estimator
    )
    if not record["estimator_unchanged_during_history_collection"]:
        raise RuntimeError("History estimator changed before block collection ended")
    histories, velocities, *privilege = (
        torch.cat(values, dim=0) for values in zip(*samples)
    )
    del samples
    if velocity_head is not None:
        with torch.no_grad():
            features = actor.estimator[:-1](histories.flatten(1))
        trainable = velocity_head
    else:
        trainable = tuple(actor.estimator.parameters())

    def adaptation_loss(indices=slice(None)):
        if velocity_head is not None:
            predicted = torch.nn.functional.linear(features[indices], *velocity_head)
            return {
                "velocity": torch.nn.functional.mse_loss(predicted, velocities[indices])
            }
        return actor.adaptation_losses(
            histories[indices], privilege[0][indices], velocities[indices]
        )

    with torch.no_grad():
        before = adaptation_loss()
    record["adaptation_before"] = {name: float(value) for name, value in before.items()}
    if velocity_head is None:
        set_phase(policy, "history")
    for _ in range(epochs):
        for indices in torch.randperm(len(histories), device=env.device).chunk(
            minibatches
        ):
            optimizer.zero_grad(set_to_none=True)
            losses = adaptation_loss(indices)
            loss = (
                losses["latent"] + losses["velocity"]
                if velocity_head is None
                else losses["velocity"]
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite ROA adaptation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            optimizer.step()
            report["adaptation_optimizer_steps"] += 1
    with torch.no_grad():
        after = adaptation_loss()
        if velocity_head is not None:
            actor.estimator[-1].weight[:3].copy_(velocity_head[0])
            actor.estimator[-1].bias[:3].copy_(velocity_head[1])
    record["adaptation_after"] = {name: float(value) for name, value in after.items()}
    record["privileged_modules_unchanged_during_adaptation"] = (
        fixed_hashes == privileged_hashes() and torch.equal(fixed_std, policy.std)
    )
    record["estimator_changed_during_adaptation"] = (
        state_sha256(actor.estimator) != fixed_estimator
    )
    if not record["privileged_modules_unchanged_during_adaptation"] or (
        require_change and not record["estimator_changed_during_adaptation"]
    ):
        raise RuntimeError(
            "ROA adaptation violated optimizer ownership or did not learn"
        )
    if (
        not all(torch.isfinite(p).all() for p in policy.parameters())
        or (policy.std <= 0).any()
    ):
        raise RuntimeError("Invalid ROA parameters")
    return obs


class ROAPPO(PPO):
    """RSL-RL 3.1.2 storage/GAE with an explicit directed ROA PPO objective.

    No RND, symmetry, adaptive LR, recurrent storage or multi-GPU. Before any
    update, replay each original-size batch and require exact rollout log-probs.
    The history estimator is excluded from Adam, not just gradient-detached.
    """

    def __init__(
        self,
        policy,
        *,
        regularization_coef=0.1,
        causal=False,
        freeze_action_std=False,
        **kwargs,
    ):
        if type(policy.actor) is not ROAActor or policy.is_recurrent:
            raise ValueError("ROAPPO requires the explicit feedforward ROA actor")
        if type(causal) is not bool or (causal and regularization_coef != 0.0):
            raise ValueError(
                "Causal fine-tuning requires an explicit route and zero regularization"
            )
        if type(freeze_action_std) is not bool or (
            freeze_action_std and (causal or kwargs.get("entropy_coef") != 0.0)
        ):
            raise ValueError(
                "Fixed action std requires privileged PPO and zero entropy bonus"
            )
        for key in ("rnd_cfg", "symmetry_cfg", "multi_gpu_cfg", "desired_kl"):
            if kwargs.get(key) is not None:
                raise ValueError(f"ROAPPO does not support {key}")
        if kwargs.get("schedule", "fixed") != "fixed":
            raise ValueError("ROAPPO requires fixed learning rate")
        kwargs.update(schedule="fixed", desired_kl=None)
        super().__init__(policy, **kwargs)
        self.causal = policy.actor.causal_ppo = causal
        self.phase = "causal" if causal else "privileged"
        policy.obs_groups["policy"] = (
            ["policy", "history"] if causal else policy.actor.privileged_obs_groups
        )
        self.regularization_coef = regularization_coef
        self.fixed_action_std = (
            policy.std.detach().clone() if freeze_action_std else None
        )
        if freeze_action_std and (
            self.fixed_action_std.shape != (12,)
            or not torch.isfinite(self.fixed_action_std).all()
            or (self.fixed_action_std <= 0).any()
        ):
            raise ValueError(
                "Fixed action std must inherit twelve finite positive values"
            )
        frozen = self._frozen_parameter_ids()
        self.ppo_parameters = tuple(
            p for p in policy.parameters() if id(p) not in frozen
        )
        self.optimizer = torch.optim.Adam(self.ppo_parameters, lr=self.learning_rate)
        self.start_phase()

    def start_phase(self):
        """Re-enter PPO after history fitting without unfreezing inherited noise."""
        set_phase(self.policy, self.phase)
        if self.fixed_action_std is not None:
            self.policy.std.requires_grad_(False)
            self.check_fixed_action_std()

    def check_fixed_action_std(self):
        if self.fixed_action_std is not None and (
            self.policy.std.requires_grad
            or self.policy.std.grad is not None
            or not torch.equal(self.policy.std, self.fixed_action_std)
            or any(
                parameter is self.policy.std
                for group in self.optimizer.param_groups
                for parameter in group["params"]
            )
        ):
            raise ValueError("Frozen parent action std or optimizer ownership changed")

    def _frozen_parameter_ids(self):
        frozen = {
            id(p) for module in self._frozen_modules() for p in module.parameters()
        }
        if self.fixed_action_std is not None:
            frozen.add(id(self.policy.std))
        return frozen

    def _frozen_modules(self):
        actor = self.policy.actor
        return (actor.estimator, actor.encoder) if self.causal else (actor.estimator,)

    def _check_phase(self):
        self.check_fixed_action_std()
        frozen = self._frozen_parameter_ids()
        expected_ppo = {id(p) for p in self.policy.parameters()} - frozen
        if (
            self.policy.obs_groups
            != {
                "policy": (
                    ["policy", "history"]
                    if self.causal
                    else self.policy.actor.privileged_obs_groups
                ),
                "critic": ["critic_state", "terrain"],
            }
            or self.policy.actor_obs_normalization
            or self.policy.critic_obs_normalization
            or self.policy.actor.causal_ppo is not self.causal
            or any(p.requires_grad for p in self.policy.parameters() if id(p) in frozen)
            or not all(p.requires_grad for p in self.ppo_parameters)
            or {id(p) for p in self.ppo_parameters} != expected_ppo
            or {id(p) for group in self.optimizer.param_groups for p in group["params"]}
            != {id(p) for p in self.ppo_parameters}
        ):
            raise ValueError(
                "PPO requires its declared routing and exclusive optimizer ownership"
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
            raise ValueError("Require a complete rollout before PPO replay")
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

    def update(self, *, after_update=None):
        """Update normally; an optional read-only observer runs before buffer clear."""
        coefficient = self.regularization_coef
        if (
            isinstance(coefficient, bool)
            or not isinstance(coefficient, (float, int))
            or not math.isfinite(coefficient)
            or coefficient < 0
            or (self.causal and coefficient != 0.0)
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
            name + "_max": 0.0 for name in gradient_norms(self.policy.actor)
        }
        frozen_before = tuple(state_sha256(module) for module in self._frozen_modules())
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
            regularization = (
                surrogate.new_zeros(())
                if self.causal
                else self.policy.actor.regularization_loss(obs)
            )
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
                if self.causal and name != "projection_gradient_l2":
                    if value is not None:
                        raise RuntimeError("Causal PPO reached the frozen teacher")
                    continue
                if value is None or not math.isfinite(value):
                    raise RuntimeError("Missing or nonfinite ROA branch gradient")
                gradient_max[name + "_max"] = max(gradient_max[name + "_max"], value)
            nn.utils.clip_grad_norm_(
                self.ppo_parameters, self.max_grad_norm, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.check_fixed_action_std()
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
        if (
            tuple(state_sha256(module) for module in self._frozen_modules())
            != frozen_before
        ):
            raise RuntimeError("PPO unexpectedly changed a frozen estimator or teacher")
        if after_update is not None:
            after_update(self)
        self.storage.clear()
        return {
            **{name: value / count for name, value in sums.items()},
            "regularization_coef": float(coefficient),
            "first_replay": replay,
            "optimizer_steps": count,
            **gradient_max,
        }
