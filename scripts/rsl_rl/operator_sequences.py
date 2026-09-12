"""Training-only reversal/hold/restart coverage; no policy input or action changes."""

import json

import torch

try:
    from .operator_curriculum import (
        OperatorExposureWrapper,
        TRANSITION_VERSION,
        curriculum_manifest,
    )
except ImportError:
    from operator_curriculum import (
        OperatorExposureWrapper,
        TRANSITION_VERSION,
        curriculum_manifest,
    )


VERSION = "operator_reversal_sequences_v3"
PROBABILITY = 0.25  # Per physical episode selection, NOT a time fraction.
PHASES = ("first_yaw", "opposite_yaw", "hold", "restart")
DURATIONS = ((3.0, 5.0), (3.0, 5.0), (3.0, 5.0), (2.0, 4.0))


def sequence_manifest():
    return {
        "version": VERSION,
        "base": json.loads(json.dumps(curriculum_manifest(TRANSITION_VERSION))),
        "sequence_episode_probability": PROBABILITY,
        "selection": "once per physical reset; independent equiprobable first yaw signs",
        "phases": list(PHASES),
        "duration_s": [list(x) for x in DURATIONS],
        "yaw_magnitude_rad_s": [0.3, 0.8],
        "restart_forward_m_s": [0.2, 0.7],
        "hold_command": [0.0, 0.0, 0.0],
        "sequence_end": "return to v2 for the remainder of the same episode",
        "maximum_planned_duration_s": sum(x[1] for x in DURATIONS),
        "reset_semantics": "discard pending stages; new episode makes a new component draw",
        "exposure": "intended reset selections, entered/completed stages, executed frames, physical censoring and training-end-active chains separately",
        "phase_observed_by_policy": False,
        "inference_assistance": False,
        "scope": "single command-distribution intervention; no demonstrated recovery or standalone acceptance",
    }


class ReversalSequencePlan:
    """Per-environment schedule advanced only by that environment's resampling."""

    def __init__(self, num_envs, device="cpu"):
        self.phase = torch.full((num_envs,), -1, dtype=torch.int64, device=device)
        self.orientation = torch.zeros_like(
            self.phase
        )  # 0: positive first, 1: negative first.
        self.intended = torch.zeros(2, dtype=torch.int64, device=device)
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
            orientation = (draws[:, 1] < 0.5).long()
            self.orientation[fresh] = orientation
            phase[new_episode] = torch.where(selected, 0, -1)
            self.intended += torch.bincount(orientation[selected], minlength=2)
        self.phase[ids] = phase
        active = ids[phase >= 0]
        phase = self.phase[active]
        draws = torch.rand(
            (len(active), 2), device=self.phase.device, generator=generator
        )
        durations = torch.tensor(DURATIONS, device=self.phase.device)[phase]
        seconds = durations[:, 0] + draws[:, 0] * (durations[:, 1] - durations[:, 0])
        commands = torch.zeros((len(active), 3), device=self.phase.device)
        sign = 1 - 2 * self.orientation[active]
        yaw = phase < 2
        sign = torch.where(phase == 1, -sign, sign)
        commands[yaw, 2] = sign[yaw] * (0.3 + 0.5 * draws[yaw, 1])
        commands[phase == 3, 0] = 0.2 + 0.5 * draws[phase == 3, 1]
        category = torch.where(
            yaw, torch.where(sign > 0, 1, 2), torch.where(phase == 2, 0, 3)
        )
        return active, commands, category, seconds


class SequenceExposure:
    """Measure actually executed phases, including interrupted episodes.

    Post-step resampling marks completion only after the final action executes.
    A termination/timeout on that action takes precedence over completion.
    """

    def __init__(self, plan):
        self.plan = plan
        self.previous = torch.full_like(plan.phase, -1)
        self.active = torch.zeros_like(plan.phase, dtype=torch.bool)
        self.active_orientation = plan.orientation.clone()
        self.entered = torch.zeros((2, 4), dtype=torch.int64, device=plan.phase.device)
        self.frames = torch.zeros_like(self.entered)
        self.completed = torch.zeros_like(self.entered)
        self.censored = torch.zeros_like(self.entered)

    def observe(self, phase, orientation, post_phase, done):
        selected = phase >= 0
        entering = selected & (phase != self.previous)
        ending = selected & ~done & (phase != post_phase)
        if (ending & (post_phase != torch.where(phase == 3, -1, phase + 1))).any():
            raise ValueError("Sequence advanced without the declared next stage")
        for counts, mask in (
            (self.entered, entering),
            (self.frames, selected),
            (self.completed, ending),
            (self.censored, selected & done),
        ):
            keys = orientation[mask] * 4 + phase[mask]
            counts += torch.bincount(keys, minlength=8).reshape(2, 4)
        self.active = selected & ~done & (post_phase >= 0)
        self.active_orientation = orientation.clone()
        self.previous = torch.where(done, -1, phase)

    def report(self):
        active = torch.bincount(self.active_orientation[self.active], minlength=2)
        entered = self.entered[:, 0]
        finished = self.completed[:, 3]
        censored = self.censored.sum(dim=1)
        if not torch.equal(entered, finished + censored + active):
            raise ValueError("Executed sequence accounting does not balance")
        return {
            "orientation_order": ["positive_first", "negative_first"],
            "phase_order": list(PHASES),
            "episode_draws": self.plan.episode_draws,
            "intended_sequences": self.plan.intended.cpu().tolist(),
            "entered_sequences": entered.cpu().tolist(),
            "entered_phases": self.entered.cpu().tolist(),
            "executed_frames": self.frames.cpu().tolist(),
            "completed_phases": self.completed.cpu().tolist(),
            "completed_sequences": finished.cpu().tolist(),
            "physically_censored_by_phase": self.censored.cpu().tolist(),
            "active_at_training_end": active.cpu().tolist(),
            "scope": "exposure only; a completed schedule is not successful robot behavior; intended reset selections may not receive an action before another reset/end",
        }


class SequenceExposureWrapper:
    def __init__(self, env):
        self.env = OperatorExposureWrapper(env)
        self.plan = self.env.command_term.sequence_plan
        self.sequence_exposure = SequenceExposure(self.plan)
        self.exposure = self  # Same report interface used by checkpoint publication.

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def episode_length_buf(self):
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.env.episode_length_buf = value

    def step(self, actions):
        phase, orientation = self.plan.phase.clone(), self.plan.orientation.clone()
        result = self.env.step(actions)
        self.sequence_exposure.observe(
            phase, orientation, self.plan.phase, result[2].bool()
        )
        return result

    def report(self):
        result = self.env.exposure.report()
        result.update(
            version=VERSION,
            base_version=TRANSITION_VERSION,
            reversal_sequences=self.sequence_exposure.report(),
        )
        return result
