"""L2 clipping preserves finite updates and rejects invalid gradients."""

import unittest
from types import SimpleNamespace

import torch

from parkour_lab.methods.roa.gradients import clip_grad_norm_


class GradientClippingTests(unittest.TestCase):
    def test_actor_branch_norms_do_not_overflow_before_global_clipping(self):
        from parkour_lab.methods.roa.model import gradient_norms

        encoder = torch.nn.Linear(2, 3)
        projection = torch.nn.Linear(3, 4, bias=False)
        for module in (encoder, projection):
            for parameter in module.parameters():
                parameter.grad = torch.full_like(parameter, 1e20)
        actor = SimpleNamespace(
            encoder=encoder,
            motor=SimpleNamespace(latent_projection=projection),
            contact_conditioned=False,
        )
        norms = gradient_norms(actor)
        for name, module in (("encoder", encoder), ("projection", projection)):
            expected = torch.cat(
                [p.grad.flatten().double() for p in module.parameters()]
            ).norm()
            self.assertAlmostEqual(norms[name + "_gradient_l2"] / float(expected), 1.0)

    @staticmethod
    def parameters(gradients):
        parameters = [
            torch.nn.Parameter(torch.zeros_like(value)) for value in gradients
        ]
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient.clone()
        return parameters

    def test_ordinary_clipped_and_unclipped_gradients_agree_with_torch(self):
        generator = torch.Generator().manual_seed(601)
        for magnitude in (0.0, 0.001, 1.0):
            with self.subTest(magnitude=magnitude):
                values = [
                    magnitude * torch.randn(shape, generator=generator)
                    for shape in ((5, 7), (12,))
                ]
                actual, expected = self.parameters(values), self.parameters(values)
                no_gradient = torch.nn.Parameter(torch.ones(1))
                actual.append(no_gradient)
                before = [parameter.detach().clone() for parameter in actual]
                rng = torch.get_rng_state().clone()
                norm = clip_grad_norm_(iter(actual), 1.0)
                expected_norm = torch.nn.utils.clip_grad_norm_(
                    expected, 1.0, error_if_nonfinite=True
                )
                torch.testing.assert_close(
                    norm, expected_norm.double(), rtol=2e-7, atol=1e-8
                )
                for parameter, reference in zip(actual, expected):
                    torch.testing.assert_close(
                        parameter.grad, reference.grad, rtol=2e-7, atol=1e-8
                    )
                    self.assertEqual(parameter.grad.dtype, torch.float32)
                self.assertIsNone(no_gradient.grad)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                for parameter, original in zip(actual, before, strict=True):
                    self.assertTrue(torch.equal(parameter, original))

    def test_large_finite_gradients_are_clipped_instead_of_overflowing(self):
        values = [torch.tensor([1e20, -1e20]), torch.tensor([3e19])]
        parameters = self.parameters(values)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        norm = clip_grad_norm_(parameters, 1.0)
        expected_norm = torch.linalg.vector_norm(torch.cat(values).double())
        torch.testing.assert_close(norm, expected_norm)
        clipped = torch.cat([parameter.grad for parameter in parameters]).double()
        self.assertTrue(torch.isfinite(clipped).all())
        self.assertLessEqual(float(clipped.norm()), 1.0 + 1e-7)
        torch.testing.assert_close(
            clipped, torch.cat(values).double() / expected_norm, rtol=1e-7, atol=0
        )

    def test_nonfinite_entries_reject_before_any_gradient_is_modified(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                values = [torch.tensor([3.0, 4.0]), torch.tensor([value])]
                parameters = self.parameters(values)
                with self.assertRaisesRegex(RuntimeError, "non-finite"):
                    clip_grad_norm_(parameters, 1.0)
                for parameter, original in zip(parameters, values, strict=True):
                    torch.testing.assert_close(
                        parameter.grad, original, rtol=0, atol=0, equal_nan=True
                    )


if __name__ == "__main__":
    unittest.main()
