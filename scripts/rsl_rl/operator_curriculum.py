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


class PivotTransitionExposure:
    """Bounded, passive accounting of commands actually delivered to physics.

    Call begin before native stepping/resampling, then finish after stepping.
    Native auto-reset destroys terminal velocities: count those command frames,
    but explicitly exclude their errors. No trajectory storage or RNG calls.
    """

    pivot_order = tuple(
        f"{sign}_{band}"
        for sign in ("positive", "negative")
        for band in ("below_0.3", "0.3_to_0.4", "at_least_0.4")
    )
    source_order = ("episode_start", "stand", "translation", *pivot_order)
    age_edges_s = (0.6, 1.0, 2.0)
    error_order = (
        "root_link_planar_speed_m_s",
        "max_body_world_yaw_abs_error_rad_s",
        "body_yaw_directional_shortfall_rad_s",
    )
    interruption_order = ("command_change", "physical", "workspace", "other_timeout")

    def __init__(self, group_ids, group_count, dt):
        if (
            type(group_count) is not int
            or group_count < 1
            or not isinstance(group_ids, torch.Tensor)
            or group_ids.ndim != 1
            or not group_ids.numel()
            or group_ids.dtype != torch.int64
            or ((group_ids < 0) | (group_ids >= group_count)).any()
            or dt != 0.02
        ):
            raise ValueError("Pivot exposure requires static int64 groups and 50 Hz")
        self.groups = group_ids.detach().clone()
        self.group_count = group_count
        self.age_edges = group_ids.new_tensor([30, 50, 100])
        self.magnitude_edges = torch.tensor([0.3, 0.4], device=group_ids.device)
        self.histogram_edges = torch.tensor(
            [0.05, 0.1, 0.15, 0.2, 0.3], dtype=torch.float64, device=group_ids.device
        )
        size = group_count * len(self.source_order) * len(self.pivot_order)
        self.entered = group_ids.new_zeros(size)
        self.frames = group_ids.new_zeros((size, 4))
        self.error_samples = torch.zeros_like(self.frames)
        self.error_sums = torch.zeros(
            size, 4, 3, dtype=torch.float64, device=group_ids.device
        )
        self.excluded_terminal_samples = group_ids.new_zeros((size, 3))
        self.incomplete_acquisitions = group_ids.new_zeros((size, 4))
        self.acquisition_histogram = group_ids.new_zeros((size, 6))
        self.acquisition_error_sums = torch.zeros(
            size, 3, dtype=torch.float64, device=group_ids.device
        )
        self.key = torch.full_like(group_ids, -1)
        self.age = torch.zeros_like(group_ids)
        self.acquisition_samples = torch.zeros_like(group_ids)
        self.window_error = torch.zeros(
            len(group_ids), 3, dtype=torch.float64, device=group_ids.device
        )
        self.previous_command = self.previous_generation = None
        self.previous_kind = torch.zeros_like(group_ids)
        self.previous_done = torch.ones_like(group_ids, dtype=torch.bool)
        self.pending = False
        self.control_steps = 0

    def _tensor(self, value, shape, dtype=None):
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != shape
            or value.device != self.groups.device
            or (
                value.dtype != dtype
                if dtype is not None
                else not value.is_floating_point()
            )
        ):
            raise ValueError("Invalid pivot-exposure tensor shape, device or dtype")

    def _count(self, destination, keys):
        destination += torch.bincount(keys, minlength=destination.numel()).reshape_as(
            destination
        )

    @torch.no_grad()
    def begin(self, command, generation, new_episode):
        if self.pending:
            raise RuntimeError("Finish the previous exposure step first")
        shape = self.groups.shape
        self._tensor(command, (len(self.groups), 3), torch.float32)
        self._tensor(generation, shape, torch.int64)
        self._tensor(new_episode, shape, torch.bool)
        if not torch.isfinite(command).all() or not torch.equal(
            new_episode, self.previous_done
        ):
            raise ValueError(
                "Require finite commands and the previous physics reset mask"
            )
        fresh = new_episode.clone()
        if self.previous_generation is not None:
            fresh |= generation != self.previous_generation
            if ((command != self.previous_command).any(-1) & ~fresh).any():
                raise ValueError("Command changed without native resampling or reset")
        # Exact zero planar commands only: arcs are translation, never pivots.
        pivot = (command[:, :2] == 0).all(-1) & (command[:, 2] != 0)
        target = torch.bucketize(
            command[:, 2].abs().contiguous(), self.magnitude_edges, right=True
        )
        target += (command[:, 2] < 0).long() * 3
        kind = torch.where(pivot, target + 3, torch.where((command == 0).all(-1), 1, 2))
        closing = fresh & (self.key >= 0) & (self.acquisition_samples < 20)
        self._count(self.incomplete_acquisitions[:, 0], self.key[closing])
        source = torch.where(new_episode, 0, self.previous_kind)
        key = (self.groups * len(self.source_order) + source) * 6 + target
        self.key = torch.where(fresh, torch.where(pivot, key, -1), self.key)
        self.age = torch.where(fresh, 0, self.age + 1)
        self.acquisition_samples[fresh] = 0
        self.window_error[fresh] = 0
        self._count(self.entered, self.key[fresh & pivot])
        self.previous_kind = kind
        # These native buffers are mutated in place by reset/resampling.
        self.previous_command = command.detach().clone()
        self.previous_generation = generation.detach().clone()
        self.pending = True

    @torch.no_grad()
    def finish(
        self, planar_velocity, body_yaw, world_yaw, terminated, timeouts, workspace
    ):
        if not self.pending:
            raise RuntimeError("Begin exposure before stepping physics")
        shape = self.groups.shape
        self._tensor(planar_velocity, (len(self.groups), 2))
        for value in (body_yaw, world_yaw):
            self._tensor(value, shape)
        for value in (terminated, timeouts, workspace):
            self._tensor(value, shape, torch.bool)
        done = terminated | timeouts
        if (workspace & ~done).any():
            raise ValueError("Workspace censoring requires a terminal step")
        pivot = self.key >= 0
        valid = pivot & ~done
        values = torch.stack(
            (
                planar_velocity[valid].norm(dim=-1),
                torch.maximum(
                    (body_yaw[valid] - self.previous_command[valid, 2]).abs(),
                    (world_yaw[valid] - self.previous_command[valid, 2]).abs(),
                ),
                self.previous_command[valid, 2].sign()
                * (self.previous_command[valid, 2] - body_yaw[valid]),
            ),
            dim=-1,
        ).double()
        if not torch.isfinite(values).all():
            raise ValueError("Nonfinite surviving pivot velocity")
        age_bin = torch.bucketize(self.age, self.age_edges, right=True)
        self._count(self.frames.view(-1), self.key[pivot] * 4 + age_bin[pivot])
        keys = self.key[valid] * 4 + age_bin[valid]
        self._count(self.error_samples.view(-1), keys)
        self.error_sums.view(-1, 3).index_add_(0, keys, values)
        acquisition = valid & (self.age >= 30) & (self.age < 50)
        self.acquisition_samples += acquisition.long()
        self.window_error[acquisition] += values[
            (self.age[valid] >= 30) & (self.age[valid] < 50)
        ]
        complete = valid & (self.age == 49) & (self.acquisition_samples == 20)
        means = self.window_error[complete] / 20
        self.acquisition_error_sums.index_add_(0, self.key[complete], means)
        bins = torch.bucketize(
            means[:, 1].contiguous(), self.histogram_edges, right=True
        )
        self._count(self.acquisition_histogram.view(-1), self.key[complete] * 6 + bins)
        for index, mask in enumerate(
            (
                terminated,
                ~terminated & timeouts & workspace,
                ~terminated & timeouts & ~workspace,
            )
        ):
            selected = pivot & mask
            self._count(self.excluded_terminal_samples[:, index], self.key[selected])
            self._count(
                self.incomplete_acquisitions[:, index + 1],
                self.key[selected & (self.acquisition_samples < 20)],
            )
        self.key[done] = -1
        self.previous_done = done.clone()
        self.control_steps += 1
        self.pending = False

    def report(self):
        if self.pending:
            raise RuntimeError("Cannot publish unfinished physics exposure")
        active = torch.bincount(
            self.key[(self.key >= 0) & (self.acquisition_samples < 20)],
            minlength=len(self.entered),
        )
        fields = {
            "entered_windows": self.entered,
            "executed_frames_by_age": self.frames,
            "surviving_error_samples_by_age": self.error_samples,
            "surviving_error_sums_by_age": self.error_sums,
            "excluded_terminal_samples": self.excluded_terminal_samples,
            "incomplete_acquisitions": self.incomplete_acquisitions,
            "active_incomplete_acquisitions": active,
            "complete_acquisition_yaw_error_histogram": self.acquisition_histogram,
            "complete_acquisition_mean_error_sums": self.acquisition_error_sums,
        }
        data = {name: value.cpu().tolist() for name, value in fields.items()}
        rows = []
        for key, count in enumerate(data["entered_windows"]):
            if count:
                group, pair = divmod(key, len(self.source_order) * 6)
                source, target = divmod(pair, 6)
                rows.append(
                    {
                        "group": group,
                        "from": self.source_order[source],
                        "to": self.pivot_order[target],
                        **{name: values[key] for name, values in data.items()},
                    }
                )
        return {
            "version": "operator_pivot_transition_exposure_v1",
            "scope": "current session only; sampled training actions, not policy acceptance; missing historical counters cannot be reconstructed",
            "period_s": 0.02,
            "control_steps": self.control_steps,
            "environment_transitions": self.control_steps * len(self.groups),
            "group_count": self.group_count,
            "source_order": list(self.source_order),
            "pivot_order": list(self.pivot_order),
            "magnitude_edges_rad_s": [0.3, 0.4],
            "age_edges_s": list(self.age_edges_s),
            "bins": "left-closed, right-open; final bin unbounded",
            "error_order": list(self.error_order),
            "terminal_exclusion_order": list(self.interruption_order[1:]),
            "incomplete_acquisition_order": list(self.interruption_order),
            "acquisition_yaw_error_edges_rad_s": self.histogram_edges.cpu().tolist(),
            "measurement": "pre-step native float32 command snapshot; post-physics surviving-row errors only, never auto-reset velocities; planar speed is root-link origin in body axes, not the historical COM scoring metric; all executed pivot frames include terminals; acquisition uses indices 30..49 ([.6,1) s), complete only with all 20 error samples; histogram contains per-window mean max(body,world) yaw error; directional shortfall positive means underturn; episode starts are not live edges; active incomplete windows are not failures",
            "rows": rows,
        }


class OperatorExposureWrapper:
    """Instrument the existing RSL wrapper; never alter actions or observations."""

    def __init__(self, env, *, operator_only=False):
        self.env = env
        self.command_term = env.unwrapped.command_manager.get_term("base_velocity")
        self.operator_only = operator_only
        self.workspace_censored = torch.zeros((), dtype=torch.long, device=env.device)
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
        if self.operator_only:
            self.workspace_censored += self.env.unwrapped.termination_manager.get_term(
                "operator_workspace"
            ).sum()
        rows = self.command_term.operator_role if self.operator_only else slice(None)
        fractions = self.exposure.observe(
            category[rows], generation[rows], self.previous_done[rows]
        )
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
