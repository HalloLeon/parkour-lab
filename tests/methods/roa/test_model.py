"""Causal routing, latent conditioning and directed model gradients."""

import copy
import unittest
from unittest import mock

import torch
from parkour_lab.methods.roa import model as roa
from torch import nn

from tests.support.roa import actor_input, observations, stock_motor


class ROAModelTests(unittest.TestCase):
    def setUp(self):
        fork = torch.random.fork_rng(devices=[])
        fork.__enter__()
        self.addCleanup(fork.__exit__, None, None, None)
        torch.manual_seed(138)
        self.stock = stock_motor()
        self.actor = roa.ROAActor(self.stock)
        self.obs = observations()
        self.history = self.obs["history"].reshape(8, 25, 49)

    def test_motor_oracle_parity_separate_from_actual_predicted_velocity_actor(self):
        velocity = self.obs["critic_state"][:, :3]
        code = torch.cat((velocity, self.actor.encode(self.obs["dynamics"])), dim=-1)
        oracle = self.stock(torch.cat((velocity, self.obs["policy"]), dim=-1))
        self.assertTrue(torch.equal(self.actor.motor(self.obs["policy"], code), oracle))
        predicted = self.actor.estimate(self.history)
        actual_expected = self.stock(
            torch.cat((predicted[:, :3], self.obs["policy"]), dim=-1)
        )
        self.assertTrue(torch.equal(self.actor(actor_input(self.obs)), actual_expected))
        self.assertFalse(torch.equal(actual_expected, oracle))
        self.assertTrue(
            torch.equal(
                self.actor.history_action(self.obs["policy"], self.history),
                actual_expected,
            )
        )

    def test_directed_losses_are_mean_unsquared_l2_with_correct_gradients(self):
        mu = self.actor.encode(self.obs["dynamics"])
        phi = self.actor.estimate(self.history)[:, 3:]
        expected = (mu - phi).norm(p=2, dim=1).mean()
        regularization = self.actor.regularization_loss(self.obs)
        self.assertTrue(torch.equal(regularization, expected))
        self.assertFalse(torch.isclose(regularization, (mu - phi).square().mean()))
        regularization.backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in self.actor.encoder.parameters()
            )
        )
        self.assertTrue(all(p.grad is None for p in self.actor.estimator.parameters()))
        self.assertTrue(all(p.grad is None for p in self.actor.motor.parameters()))
        self.actor.zero_grad(set_to_none=True)
        losses = self.actor.adaptation_losses(
            self.history, self.obs["dynamics"], self.obs["critic_state"][:, :3]
        )
        self.assertTrue(torch.equal(losses["latent"], expected))
        (losses["latent"] + losses["velocity"]).backward()
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in self.actor.estimator.parameters()
            )
        )
        self.assertTrue(all(p.grad is None for p in self.actor.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in self.actor.motor.parameters()))

    def test_policy_gradient_never_trains_velocity_estimator(self):
        self.actor(actor_input(self.obs)).sum().backward()
        self.assertTrue(all(p.grad is None for p in self.actor.estimator.parameters()))
        self.assertGreater(roa.gradient_norms(self.actor)["projection_gradient_l2"], 0)
        self.actor.zero_grad(set_to_none=True)
        self.actor.history_action(self.obs["policy"], self.history).sum().backward()
        last = self.actor.estimator[-1]
        self.assertTrue(
            torch.equal(last.weight.grad[:3], torch.zeros_like(last.weight.grad[:3]))
        )
        with torch.no_grad():
            self.actor.motor.latent_projection.weight.fill_(0.05)
        self.actor.zero_grad(set_to_none=True)
        self.actor.history_action(self.obs["policy"], self.history).sum().backward()
        self.assertTrue(
            torch.equal(last.weight.grad[:3], torch.zeros_like(last.weight.grad[:3]))
        )
        self.assertGreater(last.weight.grad[3:].abs().sum(), 0)

    def test_true_velocity_and_terrain_do_not_affect_actor_paths(self):
        before = self.actor(actor_input(self.obs))
        history_before = self.actor.history_action(self.obs["policy"], self.history)
        changed = self.obs.clone()
        changed["critic_state"] *= 100
        changed["terrain"] *= -70
        self.assertTrue(torch.equal(before, self.actor(actor_input(changed))))
        self.assertTrue(
            torch.equal(
                history_before,
                self.actor.history_action(changed["policy"], self.history),
            )
        )
        with self.assertRaises(TypeError):
            self.actor.history_action(
                self.obs["policy"], self.history, changed["critic_state"][:, :3]
            )

    def test_adaptation_does_not_backpropagate_to_labels_or_history(self):
        history = self.history.clone().requires_grad_()
        dynamics = self.obs["dynamics"].clone().requires_grad_()
        velocity = self.obs["critic_state"][:, :3].clone().requires_grad_()
        losses = self.actor.adaptation_losses(history, dynamics, velocity)
        sum(losses.values()).backward()
        self.assertIsNone(history.grad)
        self.assertIsNone(dynamics.grad)
        self.assertIsNone(velocity.grad)
        with torch.inference_mode():
            history = history.clone()
            dynamics = dynamics.clone()
            velocity = velocity.clone()
        self.actor.zero_grad(set_to_none=True)
        sum(
            self.actor.adaptation_losses(history, dynamics, velocity).values()
        ).backward()

    def test_shared_motor_routes_and_diagnostics(self):
        self.assertEqual(
            set(dict(self.actor.named_children())), {"motor", "encoder", "estimator"}
        )
        diagnostics = roa.branch_diagnostics(self.actor, self.obs)
        self.assertEqual(diagnostics["latent_permutation_action_abs_max"], 0)
        self.assertGreater(diagnostics["latent_batch_std_max"], 0)
        with torch.no_grad():
            self.actor.motor.latent_projection.weight.fill_(0.05)
        self.assertGreater(
            roa.branch_diagnostics(self.actor, self.obs)[
                "latent_permutation_action_abs_max"
            ],
            0,
        )

    def test_history_order_reset_padding_and_no_alias_or_gradient(self):
        history = roa.CausalHistory()
        first = torch.ones(2, 49, requires_grad=True)
        delivered = history.push(first, torch.ones(2, dtype=torch.bool))
        self.assertFalse(delivered.requires_grad)
        self.assertTrue(torch.equal(delivered, first[:, None].repeat(1, 25, 1)))
        delivered.zero_()
        with torch.no_grad():
            first.zero_()
        second = torch.full((2, 49), 2.0)
        delivered = history.push(second, torch.tensor([False, True]))
        self.assertTrue(torch.equal(delivered[0, :-1], torch.ones(24, 49)))
        self.assertTrue(torch.equal(delivered[1], second[1, None].repeat(25, 1)))
        for step in range(3, 29):
            delivered = history.push(
                torch.full((2, 49), float(step)), torch.zeros(2, dtype=torch.bool)
            )
        self.assertEqual(delivered[0, :, 0].tolist(), list(range(4, 29)))

    def test_invalid_history_motor_and_actor_inputs_rejected(self):
        history = roa.CausalHistory()
        with self.assertRaises(ValueError):
            history.push(torch.zeros(2, 49), torch.tensor([True, False]))
        history.push(torch.zeros(2, 49), torch.ones(2, dtype=torch.bool))
        for frame, reset in (
            (torch.zeros(1, 49), torch.ones(1, dtype=torch.bool)),
            (torch.zeros(2, 49), torch.ones(2)),
            (torch.zeros(2, 49).double(), torch.ones(2, dtype=torch.bool)),
        ):
            with self.assertRaises(ValueError):
                history.push(frame, reset)
        bad = stock_motor()
        bad[1] = nn.ReLU()
        with self.assertRaises(ValueError):
            roa.ROAActor(bad)
        with self.assertRaisesRegex(ValueError, "Newest history"):
            self.actor.history_action(torch.zeros(8, 49), self.history)
        with self.assertRaises(ValueError):
            self.actor(torch.zeros(8, 55))

    def test_contact_enable_preserves_loaded_source_exactly_and_rng(self):
        with torch.no_grad():
            self.actor.motor.latent_projection.weight.normal_(std=0.03)
        source = copy.deepcopy(self.actor.state_dict())
        layer = self.actor.encoder[0]
        latent = self.actor.encode(self.obs["dynamics"])
        privileged = self.actor(actor_input(self.obs))
        causal = self.actor.history_action(self.obs["policy"], self.history)
        self.obs["contacts"] = torch.randn(8, 12)
        rng = torch.get_rng_state().clone()
        self.actor.enable_contact_conditioning()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(self.actor.contact_conditioned)
        self.assertEqual(self.actor.privileged_dim, 19)
        self.assertEqual(
            self.actor.privileged_obs_groups,
            ["policy", "dynamics", "contacts", "history"],
        )
        self.assertIs(self.actor.encoder[0].reference, layer)
        expected = {
            name.replace("encoder.0.", "encoder.0.reference."): value
            for name, value in source.items()
        }
        expected["encoder.0.contact_projection.weight"] = torch.zeros(64, 12)
        self.assertEqual(set(self.actor.state_dict()), set(expected))
        for name, value in self.actor.state_dict().items():
            self.assertTrue(torch.equal(value, expected[name]), name)
        self.assertTrue(
            torch.equal(
                latent, self.actor.encode(self.actor.privileged_input(self.obs))
            )
        )
        self.assertTrue(
            torch.equal(privileged, self.actor(actor_input(self.obs, self.actor)))
        )
        self.assertTrue(
            torch.equal(
                causal, self.actor.history_action(self.obs["policy"], self.history)
            )
        )
        restored = roa.ROAActor(self.stock, contact_conditioned=True)
        restored.load_state_dict(self.actor.state_dict(), strict=True)
        self.assertEqual(roa.state_sha256(restored), roa.state_sha256(self.actor))
        with self.assertRaisesRegex(ValueError, "already enabled"):
            self.actor.enable_contact_conditioning()

    def test_contact_projection_gradients_intervention_and_causal_isolation(self):
        self.obs["contacts"] = torch.randn(8, 12)
        self.actor.enable_contact_conditioning()
        with torch.no_grad():
            self.actor.motor.latent_projection.weight.normal_(std=0.03)
        projection = self.actor.encoder[0].contact_projection
        initial = roa.branch_diagnostics(self.actor, self.obs)
        self.assertEqual(initial["contact_projection_weight_l2"], 0)
        self.assertEqual(initial["contact_zero_action_abs_max"], 0)
        self.assertIsNone(initial["contact_projection_gradient_l2"])
        self.actor(actor_input(self.obs, self.actor)).square().sum().backward()
        self.assertGreater(projection.weight.grad.abs().sum(), 0)
        self.assertTrue(all(p.grad is None for p in self.actor.estimator.parameters()))
        self.assertGreater(
            roa.branch_diagnostics(self.actor, self.obs)[
                "contact_projection_gradient_l2"
            ],
            0,
        )
        causal = self.actor.history_action(self.obs["policy"], self.history)
        with torch.no_grad():
            projection.weight.normal_(std=0.1)
        diagnostics = roa.branch_diagnostics(self.actor, self.obs)
        self.assertGreater(diagnostics["contact_zero_action_abs_max"], 0)
        predicted = self.actor.estimate(self.history)
        code = torch.cat(
            (
                predicted[:, :3],
                self.actor.encode(self.actor.privileged_input(self.obs)),
            ),
            dim=-1,
        )
        zero = self.actor.privileged_input(self.obs)
        zero[:, 7:] = 0
        expected = (
            self.actor.motor(self.obs["policy"], code)
            - self.actor.motor(
                self.obs["policy"],
                torch.cat((predicted[:, :3], self.actor.encode(zero)), dim=-1),
            )
        ).abs()
        self.assertEqual(
            diagnostics["contact_zero_action_abs_mean"], float(expected.mean())
        )
        self.assertEqual(
            diagnostics["contact_zero_action_abs_max"], float(expected.max())
        )
        with mock.patch.object(
            self.actor, "encode", side_effect=AssertionError("oracle")
        ):
            self.assertTrue(
                torch.equal(
                    causal, self.actor.history_action(self.obs["policy"], self.history)
                )
            )
        self.actor.zero_grad(set_to_none=True)
        privilege = self.actor.privileged_input(self.obs).detach().requires_grad_()
        sum(
            self.actor.adaptation_losses(
                self.history, privilege, self.obs["critic_state"][:, :3]
            ).values()
        ).backward()
        self.assertIsNone(privilege.grad)
        self.assertTrue(all(p.grad is None for p in self.actor.encoder.parameters()))

    def test_contact_mode_validation_and_default_initialization(self):
        rng = torch.get_rng_state().clone()
        default = roa.ROAActor(self.stock)
        after = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        explicit = roa.ROAActor(self.stock, contact_conditioned=False)
        self.assertEqual(roa.state_sha256(default), roa.state_sha256(explicit))
        self.assertTrue(torch.equal(torch.get_rng_state(), after))
        torch.set_rng_state(rng)
        conditioned = roa.ROAActor(self.stock, contact_conditioned=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), after))
        self.assertEqual(default.privileged_dim, 7)
        self.assertIs(default.privileged_input(self.obs), self.obs["dynamics"])
        with self.assertRaises(KeyError):
            conditioned.privileged_input(self.obs)
        for contacts in (
            torch.zeros(8, 11),
            torch.zeros(8, 12).double(),
            torch.full((8, 12), float("nan")),
        ):
            self.obs["contacts"] = contacts
            with self.assertRaises(ValueError):
                conditioned.privileged_input(self.obs)
        with self.assertRaises(ValueError):
            conditioned.encode(self.obs["dynamics"])
        with self.assertRaises(ValueError):
            default.encode(torch.zeros(8, 19))
        with self.assertRaises(ValueError):
            roa.ROAActor(self.stock, contact_conditioned=1)


if __name__ == "__main__":
    unittest.main()
