"""Shared episode categories with sustained-translation curriculum eligibility."""

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand

from parkour_lab.environments.terrains.curriculum import initialize_levels

VERSION = "go2_episode_twist_v1"
CATEGORIES = (
    "translation",
    "stand",
    "pivot_left",
    "pivot_right",
    "arc_left",
    "arc_right",
)
PROBABILITIES = (0.5, 0.1, 0.1, 0.1, 0.1, 0.1)
PERIOD_STEPS = 1000


class EpisodeVelocityCommand(UniformVelocityCommand):
    """Hold one independently sampled body twist until the physical episode ends."""

    def __init__(self, cfg, env):
        if cfg.heading_command or cfg.rel_heading_envs or cfg.rel_standing_envs:
            raise ValueError("Episode commands own stops; disable heading overrides")
        self.lesson = getattr(env.cfg, "benchmark_command_lesson", "full")
        if type(self.lesson) is not str or self.lesson not in ("full", "forward-stand"):
            raise ValueError("Unknown command lesson")
        if self.lesson != "full" and not env.cfg.benchmark_training:
            raise ValueError("Command lessons are training-only")
        super().__init__(cfg, env)
        from ..dynamics import get_randomization

        self.randomization = get_randomization(env)
        self.remaining_ticks = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.category = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.episode_command = torch.zeros_like(self.vel_command_b)
        self.curriculum_eligible = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.sampling_report = dict(
            version=VERSION,
            command_lesson=self.lesson,
            period_steps=PERIOD_STEPS,
            samples=0,
            translations=0,
            stops=0,
            pivots=0,
            arcs=0,
        )
        initialize_levels(env)

    def compute(self, dt):
        self.curriculum_eligible &= (self.vel_command_b == self.episode_command).all(-1)
        if dt != 0.02:
            raise ValueError("Episode commands require 50 Hz control")
        self._update_metrics()
        # Same-step autoreset rows have not executed their new command yet.
        sampled = torch.isfinite(self.time_left)
        self.remaining_ticks -= (sampled & (self._env.episode_length_buf > 0)).long()
        self._resample((sampled & (self.remaining_ticks <= 0)).nonzero().flatten())
        self.time_left[sampled] = self.remaining_ticks[sampled] * dt
        self._update_command()

    def _resample(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if len(env_ids):
            self.remaining_ticks[env_ids] = PERIOD_STEPS
            self.time_left[env_ids] = 20.0
            self._resample_command(env_ids)
            self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids):
        draws = torch.as_tensor(
            self.randomization.uniform(env_ids.cpu().numpy(), 4, "commands"),
            device=self.device,
            dtype=self.vel_command_b.dtype,
        )
        command = torch.zeros_like(draws[:, :3])
        if self.lesson == "forward-stand":
            category = (draws[:, 0] >= 0.9).long()
            command[category == 0, 0] = 0.35
        else:
            boundaries = draws.new_tensor(PROBABILITIES).cumsum(0)[:-1]
            category = torch.bucketize(draws[:, 0].contiguous(), boundaries, right=True)
            translation, arc = category == 0, category >= 4
            command[translation | arc, 0] = 0.2 + 0.3 * draws[translation | arc, 1]
            command[translation, 1] = -0.2 + 0.4 * draws[translation, 2]
            pivot = (category == 2) | (category == 3)
            sign = torch.where((category == 2) | (category == 4), 1.0, -1.0)
            command[pivot, 2] = sign[pivot] * (0.3 + 0.2 * draws[pivot, 3])
            command[arc, 2] = sign[arc] * (0.2 + 0.3 * draws[arc, 3])
        translation, arc = category == 0, category >= 4
        pivot = (category == 2) | (category == 3)
        fresh = self.command_counter[env_ids] == 0
        unchanged = (command == self.episode_command[env_ids]).all(-1)
        self.curriculum_eligible[env_ids] = torch.where(
            fresh, translation, self.curriculum_eligible[env_ids] & unchanged
        )
        self.episode_command[env_ids] = torch.where(
            fresh[:, None], command, self.episode_command[env_ids]
        )
        self.vel_command_b[env_ids] = command
        self.category[env_ids] = category
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = category == 1
        for name, mask in (
            ("translations", translation),
            ("stops", category == 1),
            ("pivots", pivot),
            ("arcs", arc),
        ):
            self.sampling_report[name] += int(mask.sum())
        self.sampling_report["samples"] += len(env_ids)
