"""Shared episode categories with sustained-translation curriculum eligibility."""

import torch

from parkour_lab.environments.terrains.families.curriculum import initialize_levels

from .native import FlatVelocityCommand

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


class EpisodeVelocityCommand(FlatVelocityCommand):
    """Hold one independently sampled body twist until the physical episode ends."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.category = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.episode_command = torch.zeros_like(self.vel_command_b)
        self.curriculum_eligible = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.sampling_report = dict(
            version=VERSION,
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
        super().compute(dt)

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
        boundaries = draws.new_tensor(PROBABILITIES).cumsum(0)[:-1]
        category = torch.bucketize(draws[:, 0].contiguous(), boundaries, right=True)
        command = torch.zeros_like(draws[:, :3])
        translation, arc = category == 0, category >= 4
        command[translation | arc, 0] = 0.2 + 0.3 * draws[translation | arc, 1]
        command[translation, 1] = -0.2 + 0.4 * draws[translation, 2]
        pivot = (category == 2) | (category == 3)
        sign = torch.where((category == 2) | (category == 4), 1.0, -1.0)
        command[pivot, 2] = sign[pivot] * (0.3 + 0.2 * draws[pivot, 3])
        command[arc, 2] = sign[arc] * (0.2 + 0.3 * draws[arc, 3])
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
