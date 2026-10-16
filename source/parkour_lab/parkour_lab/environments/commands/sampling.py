"""Explicit operator command sampling.

Pure PyTorch: no simulator imports and no inference-time command assistance.
Durations and magnitudes are randomized, not copied from the benchmark sequence.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CommandMode:
    name: str
    probability: float  # Per resample, NOT a rollout-time fraction.
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    duration_s: tuple[float, float]


COVERAGE_MODES = (
    CommandMode("stand", 0.20, (0, 0, 0), (0, 0, 0), (4, 12)),
    CommandMode("pivot_positive", 0.15, (0, 0, 0.3), (0, 0, 0.8), (4, 8)),
    CommandMode("pivot_negative", 0.15, (0, 0, -0.8), (0, 0, -0.3), (4, 8)),
    CommandMode("forward", 0.20, (0.2, 0, 0), (0.7, 0, 0), (2, 6)),
    CommandMode("arc_positive", 0.075, (0.2, 0, 0.2), (0.7, 0, 0.8), (2, 6)),
    CommandMode("arc_negative", 0.075, (0.2, 0, -0.8), (0.7, 0, -0.2), (2, 6)),
    CommandMode("reverse", 0.05, (-0.3, 0, 0), (-0.1, 0, 0), (3, 6)),
    CommandMode("lateral_positive", 0.05, (0, 0.1, 0), (0, 0.2, 0), (3, 6)),
    CommandMode("lateral_negative", 0.05, (0, -0.2, 0), (0, -0.1, 0), (3, 6)),
)


# A quarter of draws use broad coverage, including long standing/pivot windows.
# Targeted draws emphasize reverse motion and live braking between commands.
TARGET_MODES = (
    CommandMode("stand", 0.12, (0, 0, 0), (0, 0, 0), (2, 4)),
    CommandMode("pivot_positive", 0.06, (0, 0, 0.3), (0, 0, 0.8), (2, 4)),
    CommandMode("pivot_negative", 0.06, (0, 0, -0.8), (0, 0, -0.3), (2, 4)),
    CommandMode("forward", 0.18, (0.2, 0, 0), (0.7, 0, 0), (2, 4)),
    CommandMode("arc_positive", 0.06, (0.2, 0, 0.2), (0.7, 0, 0.8), (2, 4)),
    CommandMode("arc_negative", 0.06, (0.2, 0, -0.8), (0.7, 0, -0.2), (2, 4)),
    CommandMode("reverse", 0.28, (-0.3, 0, 0), (-0.1, 0, 0), (3, 5)),
    CommandMode("lateral_positive", 0.09, (0, 0.1, 0), (0, 0.2, 0), (2, 4)),
    CommandMode("lateral_negative", 0.09, (0, -0.2, 0), (0, -0.1, 0), (2, 4)),
)
COVERAGE_PROBABILITY = 0.25
BRAKING_PROBABILITY = 0.40


class OperatorCommandSampler:
    def __init__(self, device="cpu"):
        self.probability = torch.tensor(
            [m.probability for m in COVERAGE_MODES], device=device
        )
        self.minimum = torch.tensor([m.minimum for m in COVERAGE_MODES], device=device)
        self.maximum = torch.tensor([m.maximum for m in COVERAGE_MODES], device=device)
        self.duration = torch.tensor(
            [m.duration_s for m in COVERAGE_MODES], device=device
        )
        self.target_probability = torch.tensor(
            [m.probability for m in TARGET_MODES], device=device
        )
        self.target_duration = torch.tensor(
            [m.duration_s for m in TARGET_MODES], device=device
        )

    def sample(
        self, count, *, previous_category=None, new_episode=None, generator=None
    ):
        if count < 1:
            raise ValueError("At least one command must be sampled")
        if (previous_category is None) != (new_episode is None):
            raise ValueError(
                "Previous categories and reset mask must be supplied together"
            )
        if previous_category is not None and (
            previous_category.shape != (count,)
            or new_episode.shape != (count,)
            or previous_category.dtype != torch.int64
            or new_episode.dtype != torch.bool
            or previous_category.device != self.probability.device
            or new_episode.device != self.probability.device
        ):
            raise ValueError(
                "Command history must be matching int64/bool device vectors"
            )
        category = torch.multinomial(self.probability, count, True, generator=generator)
        coverage = (
            torch.rand(count, device=self.probability.device, generator=generator)
            < COVERAGE_PROBABILITY
        )
        target = torch.multinomial(
            self.target_probability, count, True, generator=generator
        )
        category = torch.where(coverage, category, target)
        if previous_category is not None:
            braking = (
                torch.rand(count, device=self.probability.device, generator=generator)
                < BRAKING_PROBABILITY
            )
            live_nonzero_command = (previous_category != 0) & ~new_episode
            category = torch.where(
                ~coverage & live_nonzero_command & braking, 0, category
            )
        uniform = torch.rand(
            count, 4, device=self.probability.device, generator=generator
        )
        low, high = self.minimum[category], self.maximum[category]
        command = low + (high - low) * uniform[:, :3]
        duration = self.duration[category]
        duration = torch.where(
            coverage[:, None], duration, self.target_duration[category]
        )
        seconds = duration[:, 0] + (duration[:, 1] - duration[:, 0]) * uniform[:, 3]
        return category, command, seconds
