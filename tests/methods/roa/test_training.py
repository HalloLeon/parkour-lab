"""ROA PPO optimization, update scheduling and learning state restoration."""

import contextlib
import copy
import io
import unittest
from unittest.mock import patch

import torch

from parkour_lab.methods.roa import model as roa
from parkour_lab.methods.roa.training import ROAPPO, ROATrainingMethod
from tests.support.roa import TensorHost, observations


class ROAPPOTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.manual_seed(823)
        self.obs = observations()
        with contextlib.redirect_stdout(io.StringIO()):
            self.policy, self.reference = roa.build_policy(self.obs)
        self.algorithm = ROAPPO(
            self.policy,
            num_learning_epochs=2,
            num_mini_batches=2,
            learning_rate=1e-4,
            device="cpu",
        )

    def collect(self):
        self.algorithm.init_storage("rl", 8, 4, self.obs, [12])
        with torch.inference_mode():
            for _ in range(4):
                self.algorithm.act(self.obs)
                self.algorithm.process_env_step(
                    self.obs, torch.randn(8), torch.zeros(8, dtype=torch.bool), {}
                )
            self.algorithm.compute_returns(self.obs)

    def test_factory_preserves_value_std_and_predicted_input_motor_parity(self):
        self.assertTrue(
            torch.equal(
                self.policy.evaluate(self.obs), self.reference.evaluate(self.obs)
            )
        )
        self.assertTrue(torch.equal(self.policy.std, self.reference.std))
        estimated = self.policy.actor.estimate(self.obs["history"].reshape(8, 25, 49))
        stock = self.reference.actor(
            torch.cat((estimated[:, :3], self.obs["policy"]), dim=-1)
        )
        self.assertTrue(torch.equal(stock, self.policy.act_inference(self.obs)))
        self.assertFalse(any(p.requires_grad for p in self.reference.parameters()))

    def test_timeout_bootstrap_uses_final_value_once_and_cuts_episode_returns(self):
        self.algorithm.init_storage("rl", 8, 1, self.obs, [12])
        terminated = torch.tensor(
            [True, False, True, False, False, False, False, False]
        )
        truncated = torch.tensor([False, True, True, False, False, False, False, False])
        done = terminated | truncated
        final = self.obs[:3].clone()
        final["critic_state"][:, 0] = torch.tensor([10.0, 20.0, 30.0])
        self.obs["critic_state"][:, 0] = 2.0
        reset = self.obs.clone()
        reset["critic_state"][:, 0] = 100.0
        reward = torch.ones(8)
        with patch.object(
            self.policy, "evaluate", side_effect=lambda obs: obs["critic_state"][:, :1]
        ):
            self.algorithm.act(self.obs)
            self.algorithm.process_env_step(
                reset,
                reward,
                done,
                {
                    "time_outs": truncated & ~terminated,
                    "final_env_ids": torch.tensor([0, 1, 2]),
                    "final_observation": final,
                },
            )
            self.algorithm.compute_returns(reset)
        expected = reward.clone()
        expected[1] += self.algorithm.gamma * 20
        torch.testing.assert_close(self.algorithm.storage.rewards[0, :, 0], expected)
        expected[~done] += self.algorithm.gamma * 100
        torch.testing.assert_close(self.algorithm.storage.returns[0, :, 0], expected)
        self.assertEqual(self.algorithm.timeout_bootstrap_rows, 1)
        self.assertTrue(torch.equal(reward, torch.ones(8)))
        self.assertTrue(torch.equal(self.algorithm.storage.dones[0, :, 0].bool(), done))

    def test_timeout_without_final_observation_cannot_enter_rollout(self):
        self.algorithm.init_storage("rl", 8, 1, self.obs, [12])
        self.algorithm.act(self.obs)
        with self.assertRaises(KeyError):
            self.algorithm.process_env_step(
                self.obs,
                torch.ones(8),
                torch.ones(8).bool(),
                {
                    "time_outs": torch.ones(8).bool(),
                },
            )
        self.assertEqual(self.algorithm.storage.step, 0)

    def test_exact_first_replay_and_exclusive_ppo_updates(self):
        self.collect()
        estimator_sha = roa.state_sha256(self.policy.actor.estimator)
        actor_sha = roa.state_sha256(self.policy.actor.motor)
        check = self.algorithm.verify_first_replay()
        self.assertEqual(check["ratio_min"], 1)
        self.assertEqual(check["ratio_max"], 1)
        result = self.algorithm.update()
        self.assertEqual(result["optimizer_steps"], 4)
        self.assertEqual(result["first_replay"]["samples"], 32)
        self.assertGreater(result["encoder_gradient_l2_max"], 0)
        self.assertGreater(result["projection_gradient_l2_max"], 0)
        self.assertEqual(roa.state_sha256(self.policy.actor.estimator), estimator_sha)
        self.assertNotEqual(roa.state_sha256(self.policy.actor.motor), actor_sha)
        excluded = {id(p) for p in self.policy.actor.estimator.parameters()}
        self.assertTrue(
            excluded.isdisjoint(
                id(p)
                for group in self.algorithm.optimizer.param_groups
                for p in group["params"]
            )
        )

    def test_policy_mutation_rejected_before_any_optimizer_update(self):
        self.collect()
        with torch.no_grad():
            self.policy.actor.motor.tail[-1].bias.add_(0.1)
        snapshot = roa.state_sha256(self.policy)
        with self.assertRaisesRegex(RuntimeError, "First PPO replay"):
            self.algorithm.update()
        self.assertEqual(roa.state_sha256(self.policy), snapshot)
        self.assertFalse(self.algorithm.optimizer.state)

    def test_history_blocks_change_only_estimator(self):
        roa.set_phase(self.policy, "history")
        frozen = {
            name: value.detach().clone()
            for name, value in self.policy.state_dict().items()
            if not name.startswith("actor.estimator.")
        }
        before = roa.state_sha256(self.policy.actor.estimator)
        optimizer = torch.optim.Adam(self.policy.actor.estimator.parameters(), lr=1e-3)
        losses = self.policy.actor.adaptation_losses(
            self.obs["history"].reshape(8, 25, 49),
            self.obs["dynamics"],
            self.obs["critic_state"][:, :3],
        )
        optimizer.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        optimizer.step()
        self.assertNotEqual(roa.state_sha256(self.policy.actor.estimator), before)
        for name, value in frozen.items():
            self.assertTrue(torch.equal(self.policy.state_dict()[name], value), name)
        with self.assertRaisesRegex(ValueError, "ownership"):
            self.algorithm.act(self.obs)

    def test_regularization_is_added_to_objective_with_no_policy_gradient(self):
        self.collect()
        self.algorithm.storage.advantages = torch.zeros_like(
            self.algorithm.storage.advantages
        )
        self.algorithm.value_loss_coef = 0
        self.algorithm.entropy_coef = 0
        self.algorithm.regularization_coef = 1
        encoder_before = roa.state_sha256(self.policy.actor.encoder)
        motor_before = roa.state_sha256(self.policy.actor.motor)
        self.algorithm.update()
        self.assertNotEqual(roa.state_sha256(self.policy.actor.encoder), encoder_before)
        self.assertEqual(roa.state_sha256(self.policy.actor.motor), motor_before)

    def test_unsupported_algorithms_and_bad_phase_rejected(self):
        for kwargs in (
            {"schedule": "adaptive"},
            {"desired_kl": 0.01},
            {"rnd_cfg": {}},
            {"symmetry_cfg": {}},
            {"multi_gpu_cfg": {}},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ROAPPO(self.policy, **kwargs)
        self.policy.actor.estimator.requires_grad_(True)
        with self.assertRaises(ValueError):
            self.algorithm.act(self.obs)
        with self.assertRaises(ValueError):
            roa.set_phase(self.policy, "unknown")

    def enable_contacts(self):
        self.obs["contacts"] = torch.randn(8, 12)
        self.policy.actor.enable_contact_conditioning()
        self.policy.obs_groups["policy"] = self.policy.actor.privileged_obs_groups
        with torch.no_grad():
            self.policy.actor.motor.latent_projection.weight.normal_(std=0.03)
        self.algorithm = ROAPPO(
            self.policy,
            regularization_coef=0,
            num_learning_epochs=2,
            num_mini_batches=2,
            learning_rate=1e-4,
            device="cpu",
        )

    def test_contact_ppo_replay_and_history_optimizer_ownership(self):
        self.enable_contacts()
        self.collect()
        self.assertEqual(
            tuple(self.algorithm.storage.observations["contacts"].shape), (4, 8, 12)
        )
        self.assertEqual(self.algorithm.verify_first_replay()["ratio_min"], 1)
        estimator = roa.state_sha256(self.policy.actor.estimator)
        projection = self.policy.actor.encoder[0].contact_projection
        before = projection.weight.detach().clone()
        result = self.algorithm.update()
        self.assertGreater(result["contact_projection_gradient_l2_max"], 0)
        self.assertFalse(torch.equal(before, projection.weight))
        self.assertEqual(estimator, roa.state_sha256(self.policy.actor.estimator))
        roa.set_phase(self.policy, "history")
        fixed = {
            name: value.clone()
            for name, value in self.policy.state_dict().items()
            if not name.startswith("actor.estimator.")
        }
        optimizer = torch.optim.Adam(self.policy.actor.estimator.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        sum(
            self.policy.actor.adaptation_losses(
                self.obs["history"].reshape(8, 25, 49),
                self.policy.actor.privileged_input(self.obs),
                self.obs["critic_state"][:, :3],
            ).values()
        ).backward()
        optimizer.step()
        self.assertNotEqual(estimator, roa.state_sha256(self.policy.actor.estimator))
        for name, value in fixed.items():
            self.assertTrue(torch.equal(value, self.policy.state_dict()[name]), name)

    def test_contact_rollout_label_mutation_and_stale_routing_rejected(self):
        self.enable_contacts()
        self.policy.obs_groups["policy"] = ["policy", "dynamics", "history"]
        with self.assertRaisesRegex(ValueError, "ownership"):
            self.algorithm.act(self.obs)
        self.policy.obs_groups["policy"] = self.policy.actor.privileged_obs_groups
        with torch.no_grad():
            self.policy.actor.encoder[0].contact_projection.weight.normal_(std=0.1)
        self.collect()
        self.algorithm.storage.observations["contacts"][0].add_(0.1)
        with self.assertRaisesRegex(RuntimeError, "First PPO replay"):
            self.algorithm.update()
        self.assertFalse(self.algorithm.optimizer.state)

    def test_contact_enable_after_optimizer_construction_rejected(self):
        self.obs["contacts"] = torch.randn(8, 12)
        self.policy.actor.enable_contact_conditioning()
        self.policy.obs_groups["policy"] = self.policy.actor.privileged_obs_groups
        with self.assertRaisesRegex(ValueError, "ownership"):
            self.algorithm.act(self.obs)

    def test_contact_factory_explicit_routing_and_validation(self):
        self.obs["contacts"] = torch.randn(8, 12)
        with contextlib.redirect_stdout(io.StringIO()):
            policy, _ = roa.build_policy(
                self.obs,
                contact_conditioned=True,
            )
        self.assertTrue(policy.actor.contact_conditioned)
        self.assertEqual(
            policy.obs_groups["policy"], policy.actor.privileged_obs_groups
        )
        self.assertTrue(torch.isfinite(policy.act_inference(self.obs)).all())
        self.obs["contacts"] = torch.zeros(8, 11)
        with self.assertRaises(ValueError):
            roa.build_policy(self.obs, contact_conditioned=True)


class ROAMethodTests(unittest.TestCase):
    def setUp(self):
        fork = torch.random.fork_rng(devices=[])
        fork.__enter__()
        self.addCleanup(fork.__exit__, None, None, None)
        torch.manual_seed(821)
        self.obs = observations(8)

    def policy(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return roa.build_policy(self.obs, **kwargs)

    def method(self, policy=None, **kwargs):
        policy = self.policy()[0] if policy is None else policy
        options = dict(
            num_envs=8,
            rollout_steps=2,
            history_steps=2,
            adaptation_epochs=2,
            adaptation_batches=2,
            adaptation_learning_rate=1e-3,
            history_interval=2,
            regularization_coef=0.2,
            regularization_start_update=0,
            regularization_end_update=2,
            num_learning_epochs=2,
            num_mini_batches=2,
            learning_rate=1e-3,
            device="cpu",
        )
        options.update(kwargs)
        return ROATrainingMethod(policy, self.obs, **options)

    def test_fresh_policy_requires_no_artifact_and_preserves_initial_arithmetic(self):
        policy, reference = self.policy(initial_action_std=0.4)
        self.assertTrue(torch.equal(policy.std, torch.full((12,), 0.4)))
        self.assertTrue(
            torch.equal(policy.evaluate(self.obs), reference.evaluate(self.obs))
        )
        estimate = policy.actor.estimate(self.obs["history"].reshape(8, 25, 49))
        expected = reference.actor(torch.cat((estimate[:, :3], self.obs["policy"]), -1))
        self.assertTrue(torch.equal(policy.act_inference(self.obs), expected))
        self.assertFalse(any(p.requires_grad for p in reference.parameters()))
        self.assertTrue(
            torch.equal(
                policy.actor.motor.latent_projection.weight, torch.zeros(128, 8)
            )
        )
        for value in (0, -1, True, float("nan"), float("inf"), "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.policy(initial_action_std=value)

    def test_advance_owns_schedule_and_runs_history_only_when_due(self):
        method = self.method()
        host = TensorHost(self.obs)
        estimator = roa.state_sha256(method.policy.actor.estimator)
        obs, first = method.advance(host, self.obs)
        self.assertEqual(first["update"], 1)
        self.assertEqual(first["regularization_coefficient"], 0.0)
        self.assertIsNone(first["adaptation"])
        self.assertEqual(host.phases, ["privileged_ppo"] * 2)
        self.assertEqual(roa.state_sha256(method.policy.actor.estimator), estimator)
        obs, second = method.advance(host, obs)
        self.assertEqual(second["update"], method.updates)
        self.assertEqual(second["regularization_coefficient"], 0.1)
        self.assertEqual(second["adaptation_optimizer_steps"], 4)
        self.assertTrue(
            second["adaptation"]["privileged_modules_unchanged_during_adaptation"]
        )
        self.assertEqual(
            host.phases, ["privileged_ppo"] * 4 + ["history_adaptation"] * 2
        )
        self.assertNotEqual(roa.state_sha256(method.policy.actor.estimator), estimator)
        _, third = method.advance(host, obs)
        self.assertEqual(third["regularization_coefficient"], 0.2)

    def test_advance_matches_the_existing_manual_numerical_operations_exactly(self):
        policy = self.policy()[0]
        manual = self.method(copy.deepcopy(policy))
        scheduled = self.method(copy.deepcopy(policy))
        manual_host, scheduled_host = TensorHost(self.obs), TensorHost(self.obs)
        rng = torch.get_rng_state().clone()
        manual_obs = self.obs
        losses = []
        for coefficient in (0.0, 0.1):
            manual.algorithm.regularization_coef = coefficient
            manual_obs = manual.collect(manual_host, manual_obs)
            losses.append(manual.update(manual_obs))
        record, report = {}, {"adaptation_optimizer_steps": 0}
        manual_obs = manual.adapt(
            manual_host, manual_obs, record, report, require_change=False
        )
        final_rng = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        scheduled_obs = self.obs
        for expected_loss in losses:
            scheduled_obs, metrics = scheduled.advance(scheduled_host, scheduled_obs)
            self.assertEqual(metrics["ppo_losses"], expected_loss)
        self.assertEqual(metrics["adaptation"], record)
        self.assertEqual(
            roa.state_sha256(scheduled.policy), roa.state_sha256(manual.policy)
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), final_rng))
        self.assertEqual(scheduled_host.phases, manual_host.phases)

    def test_regularization_warmup_endpoints_plateau_and_constant_mode(self):
        for start, end, coefficients in (
            (2, 4, [0, 0, 0, 0.1, 0.2, 0.2]),
            (0, 0, [0.2, 0.2]),
            (1, 1, [0, 0.2, 0.2]),
        ):
            with self.subTest(start=start, end=end):
                method = self.method(
                    regularization_start_update=start, regularization_end_update=end
                )
                obs, host = self.obs, TensorHost(self.obs)
                for expected in coefficients:
                    obs, metrics = method.advance(host, obs)
                    self.assertAlmostEqual(
                        metrics["regularization_coefficient"], expected
                    )

    def test_snapshot_restores_weights_optimizers_and_schedule_position(self):
        method = self.method()
        host = TensorHost(self.obs)
        obs, _ = method.advance(host, self.obs)
        snapshot = copy.deepcopy(method.state_dict())
        self.assertEqual(
            set(snapshot),
            {
                "policy_state",
                "ppo_optimizer",
                "adaptation_optimizer",
                "updates",
                "adaptation_optimizer_steps",
            },
        )
        restored = self.method()
        restored.load_state_dict(snapshot)
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.adaptation_optimizer_steps, 0)
        self.assertEqual(
            roa.state_sha256(restored.policy), roa.state_sha256(method.policy)
        )
        _, metrics = restored.advance(TensorHost(obs), obs)
        self.assertEqual(metrics["update"], 2)
        self.assertEqual(metrics["regularization_coefficient"], 0.1)
        self.assertIsNotNone(metrics["adaptation"])
        for key, value in (
            ("updates", True),
            ("updates", -1),
            ("adaptation_optimizer_steps", -1),
        ):
            bad = copy.deepcopy(snapshot)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.method().load_state_dict(bad)

    def test_causal_ppo_keeps_teacher_fixed_while_history_adapts(self):
        method = self.method(causal=True, regularization_coef=0.0)
        teacher = roa.state_sha256(method.policy.actor.encoder)
        estimator = roa.state_sha256(method.policy.actor.estimator)
        host = TensorHost(self.obs)
        obs, _ = method.advance(host, self.obs)
        _, metrics = method.advance(host, obs)
        self.assertEqual(roa.state_sha256(method.policy.actor.encoder), teacher)
        self.assertNotEqual(roa.state_sha256(method.policy.actor.estimator), estimator)
        self.assertEqual(host.phases, ["causal_ppo"] * 4 + ["history_adaptation"] * 2)
        self.assertIsNotNone(metrics["adaptation"])

    def test_fixed_action_std_stays_outside_both_optimizers(self):
        method = self.method(freeze_action_std=True, entropy_coef=0.0)
        std = method.policy.std.detach().clone()
        host = TensorHost(self.obs)
        obs, _ = method.advance(host, self.obs)
        method.advance(host, obs)
        self.assertTrue(torch.equal(method.policy.std, std))
        self.assertFalse(method.policy.std.requires_grad)
        self.assertIsNone(method.policy.std.grad)
        for optimizer in (method.algorithm.optimizer, method.adaptation_optimizer):
            for group in optimizer.param_groups:
                self.assertTrue(
                    all(
                        parameter is not method.policy.std
                        for parameter in group["params"]
                    )
                )
        method.state_dict()

    def test_invalid_collection_and_optimizer_options_fail_before_rollout(self):
        invalid = (
            {"history_interval": 0},
            {"history_interval": True},
            {"history_steps": 0},
            {"rollout_steps": 0},
            {"adaptation_epochs": 1.5},
            {"adaptation_batches": 17},
            {"adaptation_learning_rate": float("nan")},
            {"regularization_start_update": -1},
            {"regularization_end_update": -1},
            {"regularization_start_update": True},
            {"regularization_end_update": 1.5},
            {"regularization_start_update": 3, "regularization_end_update": 2},
            {"regularization_coef": float("inf")},
            {"num_learning_epochs": True},
            {"num_mini_batches": 0},
            {"learning_rate": 0},
            {"causal": True, "regularization_coef": 0.1},
        )
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.method(**options)

    def test_interrupted_update_cannot_be_continued_or_snapshotted(self):
        method = self.method()
        with patch.object(
            method, "update", side_effect=RuntimeError("test interruption")
        ):
            with self.assertRaisesRegex(RuntimeError, "test interruption"):
                method.advance(TensorHost(self.obs), self.obs)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            method.advance(TensorHost(self.obs), self.obs)
        with self.assertRaisesRegex(ValueError, "in-flight"):
            method.state_dict()


if __name__ == "__main__":
    unittest.main()
