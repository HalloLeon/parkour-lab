"""Mixed command probabilities, durations and private generator progression."""

import unittest

import torch

from parkour_lab.environments.commands.sampling import OperatorCommandSampler


def reference_draw(count, generator, *, previous=None, reset=None, rough=False):
    """Independent specification of the sampler's probabilities and durations."""
    background = torch.tensor([0.20, 0.15, 0.15, 0.20, 0.075, 0.075, 0.05, 0.05, 0.05])
    target = torch.tensor([0.12, 0.06, 0.06, 0.18, 0.06, 0.06, 0.28, 0.09, 0.09])
    if rough:
        background = target = torch.tensor(
            [0.15, 0.075, 0.075, 0.40, 0.15, 0.15, 0, 0, 0]
        )
    low = torch.tensor(
        [
            (0, 0, 0),
            (0, 0, 0.3),
            (0, 0, -0.8),
            (0.2, 0, 0),
            (0.2, 0, 0.2),
            (0.2, 0, -0.8),
            (-0.3, 0, 0),
            (0, 0.1, 0),
            (0, -0.2, 0),
        ]
    )
    high = torch.tensor(
        [
            (0, 0, 0),
            (0, 0, 0.8),
            (0, 0, -0.3),
            (0.7, 0, 0),
            (0.7, 0, 0.8),
            (0.7, 0, -0.2),
            (-0.1, 0, 0),
            (0, 0.2, 0),
            (0, -0.1, 0),
        ]
    )
    background_duration = torch.tensor(
        [(4, 12), (4, 8), (4, 8), (2, 6), (2, 6), (2, 6), (3, 6), (3, 6), (3, 6)]
    )
    target_duration = torch.tensor(
        [(2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4), (3, 5), (2, 4), (2, 4)]
    )
    category = torch.multinomial(background, count, True, generator=generator)
    coverage = torch.rand(count, generator=generator) < 0.25
    targeted = torch.multinomial(target, count, True, generator=generator)
    category = torch.where(coverage, category, targeted)
    if previous is not None:
        braking = torch.rand(count, generator=generator) < 0.40
        category = torch.where(
            ~coverage & (previous != 0) & ~reset & braking, 0, category
        )
    uniform = torch.rand(count, 4, generator=generator)
    commands = low[category] + (high[category] - low[category]) * uniform[:, :3]
    duration = torch.where(
        coverage[:, None], background_duration[category], target_duration[category]
    )
    seconds = duration[:, 0] + (duration[:, 1] - duration[:, 0]) * uniform[:, 3]
    return category, commands, seconds


class CommandSamplingTests(unittest.TestCase):
    def test_arithmetic_and_rng_state_match_explicit_distribution(self):
        global_rng = torch.get_rng_state().clone()
        for seed in (0, 42, 971):
            for rough in (False, True):
                actual_rng = torch.Generator().manual_seed(seed)
                expected_rng = torch.Generator().manual_seed(seed)
                sampler = OperatorCommandSampler()
                if rough:
                    probability = torch.tensor(
                        (0.15, 0.075, 0.075, 0.40, 0.15, 0.15, 0, 0, 0)
                    )
                    sampler.probability.copy_(probability)
                    sampler.target_probability.copy_(probability)
                for count in (1, 17, 128):
                    for history in (False, True):
                        previous = torch.arange(count) % 9 if history else None
                        reset = torch.arange(count) % 3 == 0 if history else None
                        expected = reference_draw(
                            count,
                            expected_rng,
                            previous=previous,
                            reset=reset,
                            rough=rough,
                        )
                        actual = sampler.sample(
                            count,
                            previous_category=previous,
                            new_episode=reset,
                            generator=actual_rng,
                        )
                        for value, reference in zip(actual, expected, strict=True):
                            self.assertTrue(torch.equal(value, reference))
                        self.assertTrue(
                            torch.equal(
                                actual_rng.get_state(), expected_rng.get_state()
                            )
                        )
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))

    def test_physical_reset_cannot_inherit_live_braking(self):
        sampler = OperatorCommandSampler()
        count = 4096

        def draw(previous, reset):
            return sampler.sample(
                count,
                previous_category=previous,
                new_episode=reset,
                generator=torch.Generator().manual_seed(19),
            )

        nonzero = torch.ones(count, dtype=torch.int64)
        zeros = torch.zeros(count, dtype=torch.int64)
        live = torch.zeros(count, dtype=torch.bool)
        fresh = draw(nonzero, ~live)
        stopped = draw(zeros, live)
        for left, right in zip(fresh, stopped, strict=True):
            self.assertTrue(torch.equal(left, right))
        braking = draw(nonzero, live)
        self.assertGreater(int((braking[0] == 0).sum()), int((fresh[0] == 0).sum()))
        changed = braking[0] != fresh[0]
        self.assertTrue((braking[0][changed] == 0).all())
        self.assertTrue((braking[1][changed] == 0).all())

    def test_history_validation_rejects_mismatched_shapes(self):
        sampler = OperatorCommandSampler()
        for kwargs in (
            {"previous_category": torch.zeros(2, dtype=torch.int64)},
            {"new_episode": torch.zeros(2, dtype=torch.bool)},
            {
                "previous_category": torch.zeros(2),
                "new_episode": torch.zeros(2, dtype=torch.bool),
            },
            {
                "previous_category": torch.zeros(3, dtype=torch.int64),
                "new_episode": torch.zeros(2, dtype=torch.bool),
            },
        ):
            with self.assertRaises(ValueError):
                sampler.sample(2, **kwargs)
        with self.assertRaises(ValueError):
            sampler.sample(0)


if __name__ == "__main__":
    unittest.main()
