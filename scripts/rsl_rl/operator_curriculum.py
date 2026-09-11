"""Explicit operator command sampling and executed-exposure accounting.

Pure PyTorch: no simulator imports and no inference-time command assistance.
Durations and magnitudes are randomized, not copied from the benchmark sequence.
"""

from dataclasses import asdict, dataclass

import torch


VERSION = "operator_modes_v1"


@dataclass(frozen=True)
class CommandMode:
    name: str
    probability: float  # Per resample, NOT a rollout-time fraction.
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]
    duration_s: tuple[float, float]


MODES = (
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


# Keep v1 reproducible. The new profile spends less time in already-mastered
# holds, gives reverse substantial coverage, and samples live braking events.
# A quarter of draws still use v1, including its long standing/pivot windows.
TRANSITION_VERSION = "operator_transitions_v2"
TRANSITION_MODES = (
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
VERSIONS = (VERSION, TRANSITION_VERSION)
COVERAGE_PROBABILITY = 0.25
BRAKING_PROBABILITY = 0.40


def curriculum_manifest(version=VERSION):
    if version not in VERSIONS:
        raise ValueError(f"Unsupported operator curriculum: {version}")
    report = {
        "version": version,
        "units": ["m/s", "m/s", "rad/s"],
        "frame": "body",
        "modes": [asdict(mode) for mode in MODES],
        "transitions": "independent mode draws, including direct yaw reversal",
        "probability_unit": "command resamples, not rollout transitions",
        "episode_censoring": "windows truncate at the unchanged physical termination/20-s timeout",
        "inference_assistance": False,
    }
    if version == TRANSITION_VERSION:
        report.update(
            modes=[asdict(mode) for mode in TRANSITION_MODES],
            coverage_modes=[asdict(mode) for mode in MODES],
            coverage_probability=COVERAGE_PROBABILITY,
            targeted_probability=1.0 - COVERAGE_PROBABILITY,
            braking_probability=BRAKING_PROBABILITY,
            transitions=(
                "25% independent v1 coverage draws; 75% targeted draws. On targeted "
                "live resamples after any nonzero mode, replace the next draw with "
                "zero twist with probability 0.4. Never bridge physical resets."
            ),
            probability_unit=(
                "base probabilities conditional on component, BEFORE live-stop "
                "replacement; not marginal mode probabilities or rollout fractions"
            ),
        )
    return report


class OperatorCommandSampler:
    def __init__(self, device="cpu", version=VERSION):
        curriculum_manifest(version)  # Fail closed on unknown recipes.
        self.version = version
        self.probability = torch.tensor([m.probability for m in MODES], device=device)
        self.minimum = torch.tensor([m.minimum for m in MODES], device=device)
        self.maximum = torch.tensor([m.maximum for m in MODES], device=device)
        self.duration = torch.tensor([m.duration_s for m in MODES], device=device)
        self.target_probability = torch.tensor(
            [m.probability for m in TRANSITION_MODES], device=device
        )
        self.target_duration = torch.tensor(
            [m.duration_s for m in TRANSITION_MODES], device=device
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
        coverage = None
        if self.version == TRANSITION_VERSION:
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
                    torch.rand(
                        count, device=self.probability.device, generator=generator
                    )
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
        if coverage is not None:
            duration = torch.where(
                coverage[:, None], duration, self.target_duration[category]
            )
        seconds = duration[:, 0] + (duration[:, 1] - duration[:, 0]) * uniform[:, 3]
        return category, command, seconds


class CommandExposure:
    """Count commands actually applied to physics, including failed episodes.

    Call before each action and pass the previous action's done mask. Resampling
    the same mode counts as a command window; resets are not live transitions.
    """

    def __init__(self, device="cpu", version=VERSION):
        self.version = curriculum_manifest(version)["version"]
        n = len(MODES)
        self.steps = torch.zeros(n, dtype=torch.int64, device=device)
        self.starts = torch.zeros_like(self.steps)
        self.transitions = torch.zeros(n, n, dtype=torch.int64, device=device)
        self.episode_starts = torch.zeros_like(self.steps)
        self.previous_category = None
        self.previous_generation = None
        self.control_steps = 0

    def observe(self, category, generation, new_episode):
        counts = torch.bincount(category, minlength=len(MODES))
        self.steps += counts
        if self.previous_category is None:
            new_episode = torch.ones_like(category, dtype=torch.bool)
            new_window = new_episode
        else:
            new_window = new_episode | (generation != self.previous_generation)
            live = new_window & ~new_episode
            pairs = self.previous_category[live] * len(MODES) + category[live]
            self.transitions += torch.bincount(
                pairs, minlength=len(MODES) ** 2
            ).reshape(len(MODES), len(MODES))
        self.starts += torch.bincount(category[new_window], minlength=len(MODES))
        self.episode_starts += torch.bincount(
            category[new_episode], minlength=len(MODES)
        )
        self.previous_category = category.clone()
        self.previous_generation = generation.clone()
        self.control_steps += 1
        return counts / category.numel()

    def report(self):
        steps = self.steps.cpu().tolist()
        total = sum(steps)
        return {
            "version": self.version,
            "mode_order": [m.name for m in MODES],
            "control_steps": self.control_steps,
            "environment_transitions": total,
            "executed_transitions": dict(zip((m.name for m in MODES), steps)),
            "executed_fractions": {
                m.name: count / total if total else 0.0
                for m, count in zip(MODES, steps)
            },
            "command_window_starts": self.starts.cpu().tolist(),
            "episode_starts": self.episode_starts.cpu().tolist(),
            "live_command_transitions_from_to": self.transitions.cpu().tolist(),
        }


class OperatorExposureWrapper:
    """Instrument the existing RSL wrapper; never alter actions or observations."""

    def __init__(self, env):
        self.env = env
        self.command_term = env.unwrapped.command_manager.get_term("base_velocity")
        self.exposure = CommandExposure(
            env.device, getattr(self.command_term, "curriculum_version", VERSION)
        )
        self.previous_done = torch.ones(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def step(self, actions):
        # Snapshot BEFORE env.step resamples commands or resets terminated rows.
        category = self.command_term.category.clone()
        generation = self.command_term.command_counter.clone()
        observation, reward, done, extras = self.env.step(actions)
        fractions = self.exposure.observe(category, generation, self.previous_done)
        self.previous_done = done.bool().clone()
        extras = dict(extras)
        # RSL prefers "episode" if present, otherwise "log". Preserve every key.
        key = "episode" if "episode" in extras else "log"
        extras[key] = dict(extras.get(key, {}))
        extras[key].update(
            {
                f"Operator/exposure/{mode.name}": fraction
                for mode, fraction in zip(MODES, fractions)
            }
        )
        return observation, reward, done, extras
