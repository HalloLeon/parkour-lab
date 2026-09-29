"""Current training-only yaw reversal, hold and restart sequence sampler."""

import torch

PROBABILITY = 0.25
PHASES = ("first_yaw", "opposite_yaw", "hold", "restart")
DURATIONS = ((3.0, 5.0), (3.0, 5.0), (3.0, 5.0), (2.0, 4.0))


class ReversalSequencePlan:
    """Per-environment schedule advanced only by that environment's resampling."""

    phase_names = PHASES
    orientation_names = ("positive_first", "negative_first")
    durations = DURATIONS

    def __init__(self, num_envs, device="cpu"):
        self.phase = torch.full((num_envs,), -1, dtype=torch.int64, device=device)
        self.orientation = torch.zeros_like(
            self.phase
        )  # 0: positive first, 1: negative first.
        self.intended = torch.zeros(
            len(self.orientation_names), dtype=torch.int64, device=device
        )
        self.episode_draws = 0

    def resample(self, env_ids, new_episode, *, generator=None):
        ids = torch.as_tensor(env_ids, device=self.phase.device, dtype=torch.int64)
        if (
            ids.ndim != 1
            or new_episode.shape != ids.shape
            or new_episode.dtype != torch.bool
        ):
            raise ValueError(
                "Require environment indices and a matching physical-reset mask"
            )
        if len(ids) == 0:
            return (
                ids,
                torch.empty((0, 3), device=self.phase.device),
                self.phase[ids],
                torch.empty(0, device=self.phase.device),
            )
        if (
            (ids < 0).any()
            or (ids >= len(self.phase)).any()
            or len(ids.unique()) != len(ids)
        ):
            raise ValueError("Environment indices must be unique and in range")
        phase = self.phase[ids]
        phase = torch.where((phase >= 0) & (phase < 3), phase + 1, -1)
        fresh = ids[new_episode]
        self.episode_draws += len(fresh)
        if len(fresh):
            draws = torch.rand(
                (len(fresh), 2), device=self.phase.device, generator=generator
            )
            selected = draws[:, 0] < PROBABILITY
            orientation = self._orientation(draws[:, 1])
            self.orientation[fresh] = orientation
            phase[new_episode] = torch.where(selected, 0, -1)
            self.intended += torch.bincount(
                orientation[selected], minlength=len(self.intended)
            )
        self.phase[ids] = phase
        active = ids[phase >= 0]
        phase = self.phase[active]
        draws = torch.rand(
            (len(active), 2), device=self.phase.device, generator=generator
        )
        durations = torch.tensor(self.durations, device=self.phase.device)[phase]
        seconds = durations[:, 0] + draws[:, 0] * (durations[:, 1] - durations[:, 0])
        commands, category = self._commands(active, phase, draws[:, 1], generator)
        return active, commands, category, seconds

    def _orientation(self, draw):
        return (draw < 0.5).long()

    def _commands(self, active, phase, magnitude, generator):
        commands = torch.zeros((len(active), 3), device=self.phase.device)
        sign = 1 - 2 * self.orientation[active]
        yaw = phase < 2
        sign = torch.where(phase == 1, -sign, sign)
        commands[yaw, 2] = sign[yaw] * (0.3 + 0.5 * magnitude[yaw])
        commands[phase == 3, 0] = 0.2 + 0.5 * magnitude[phase == 3]
        category = torch.where(
            yaw, torch.where(sign > 0, 1, 2), torch.where(phase == 2, 0, 3)
        )
        return commands, category
