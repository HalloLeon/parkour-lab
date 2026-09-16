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


class ArrivalHoldPlan(ReversalSequencePlan):
    """Training-only randomized approach, arc/pivot, long zero and restart."""

    phase_names = ("forward", "arrival", "hold", "restart")
    orientation_names = (
        "arc_positive",
        "arc_negative",
        "pivot_positive",
        "pivot_negative",
    )
    durations = ((3.0, 4.0), (1.5, 2.0), (10.0, 11.5), (1.5, 2.0))

    def _orientation(self, draw):
        return (draw * 4).long()

    def _commands(self, active, phase, magnitude, generator):
        kind = self.orientation[active]
        turning = phase == 1
        translating = (phase == 0) | (phase == 3) | (turning & (kind < 2))
        command = torch.zeros((len(active), 3), device=self.phase.device)
        command[translating, 0] = 0.2 + 0.5 * magnitude[translating]
        yaw = torch.rand(len(active), device=self.phase.device, generator=generator)
        command[turning, 2] = (1 - 2 * (kind[turning] % 2)) * (0.2 + 0.6 * yaw[turning])
        category = torch.where(
            turning,
            torch.where(kind < 2, 4 + kind, kind - 1),
            torch.where(phase == 2, 0, 3),
        )
        return command, category


def arrival_hold_manifest():
    return {
        "version": "operator_arrival_hold_sequences_v1",
        "sequence_episode_probability": PROBABILITY,
        "selection": "once per physical reset, all profiles; four equiprobable arrival kinds",
        "arrival_kinds": list(ArrivalHoldPlan.orientation_names),
        "phases": list(ArrivalHoldPlan.phase_names),
        "duration_s": [list(x) for x in ArrivalHoldPlan.durations],
        "forward_m_s": [0.2, 0.7],
        "yaw_magnitude_rad_s": [0.2, 0.8],
        "hold_command": [0.0, 0.0, 0.0],
        "maximum_planned_duration_s": 19.5,
        "background": "unchanged procedural sampler on unselected episodes and after restart",
        "phase_observed_by_policy": False,
        "inference_assistance": False,
        "scope": "command-distribution candidate, not causal attribution without a matched fresh-Adam control; no terrain, reward, motor, policy or episode-duration change",
    }


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
        self.entered = torch.zeros(
            (len(plan.orientation_names), 4),
            dtype=torch.int64,
            device=plan.phase.device,
        )
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
            counts += torch.bincount(keys, minlength=counts.numel()).reshape_as(counts)
        self.active = selected & ~done & (post_phase >= 0)
        self.active_orientation = orientation.clone()
        self.previous = torch.where(done, -1, phase)

    def report(self):
        active = torch.bincount(
            self.active_orientation[self.active], minlength=len(self.entered)
        )
        entered = self.entered[:, 0]
        finished = self.completed[:, 3]
        censored = self.censored.sum(dim=1)
        if not torch.equal(entered, finished + censored + active):
            raise ValueError("Executed sequence accounting does not balance")
        return {
            "orientation_order": list(self.plan.orientation_names),
            "phase_order": list(self.plan.phase_names),
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


class ArrivalHoldExposure(SequenceExposure):
    """Bounded counters for joint arrival/hold coverage, not success metrics."""

    hold_age_edges_s = (0.6, 2.0, 5.0, 10.0)

    def __init__(self, plan, group_ids, group_count, dt):
        super().__init__(plan)
        self.group_ids = group_ids
        self.group_count = group_count
        self.age = torch.zeros_like(plan.phase)
        self.arrival_motion = torch.zeros_like(plan.phase, dtype=torch.bool)
        self.uneven_arrival = torch.zeros_like(self.arrival_motion)
        self.age_edges = self.age.new_tensor(
            [round(t / dt) for t in self.hold_age_edges_s]
        )
        self.holds = {
            name: self.age.new_zeros((group_count, 4, 5))
            for name in (
                "all_frames",
                "nonflat_region",
                "nonflat_with_scan_relief",
                "after_measured_arrival_motion",
                "after_moving_nonflat_relief_entry",
                "four_feet_in_contact",
            )
        }
        self.hold_entries = self.age.new_zeros((group_count, 4))
        self.interruptions = {
            name: torch.zeros_like(self.entered)
            for name in ("physical", "workspace", "other_timeout")
        }

    def observe(
        self,
        phase,
        orientation,
        post_phase,
        terminated,
        timeouts,
        workspace,
        nonflat,
        relief,
        moving,
        four_contacts,
    ):
        done = terminated | timeouts
        entering = (phase == 2) & (self.previous != 2)
        hold = phase == 2
        self.uneven_arrival = torch.where(
            entering, nonflat & relief & self.arrival_motion, self.uneven_arrival
        )
        self.age = torch.where(hold & ~entering, self.age, 0)
        self.hold_entries += torch.bincount(
            self.group_ids[entering] * 4 + orientation[entering],
            minlength=self.hold_entries.numel(),
        ).reshape_as(self.hold_entries)
        bins = torch.bucketize(self.age, self.age_edges, right=True)
        keys = (self.group_ids * 4 + orientation) * 5 + bins
        for name, mask in (
            ("all_frames", hold),
            ("nonflat_region", hold & nonflat),
            ("nonflat_with_scan_relief", hold & nonflat & relief),
            ("after_measured_arrival_motion", hold & self.arrival_motion),
            ("after_moving_nonflat_relief_entry", hold & self.uneven_arrival),
            ("four_feet_in_contact", hold & four_contacts),
        ):
            counts = self.holds[name]
            counts += torch.bincount(keys[mask], minlength=counts.numel()).reshape_as(
                counts
            )
        for name, mask in (
            ("physical", terminated),
            ("workspace", ~terminated & timeouts & workspace),
            ("other_timeout", ~terminated & timeouts & ~workspace),
        ):
            selected = (phase >= 0) & mask
            self.interruptions[name] += torch.bincount(
                orientation[selected] * 4 + phase[selected],
                minlength=16,
            ).reshape(4, 4)
        super().observe(phase, orientation, post_phase, done)
        self.age = torch.where(hold & ~done, self.age + 1, 0)
        self.arrival_motion = (
            torch.where(phase == 1, moving, self.arrival_motion) & ~done
        )
        self.uneven_arrival &= ~done

    def report(self):
        result = super().report()
        result["interrupted_by_phase"] = result.pop("physically_censored_by_phase")
        result.update(
            interruption_causes={
                name: value.cpu().tolist() for name, value in self.interruptions.items()
            },
            hold_entries_by_group_and_kind=self.hold_entries.cpu().tolist(),
            hold_age_edges_s=list(self.hold_age_edges_s),
            hold_frame_axes=["row_profile_group", "arrival_kind", "hold_age_bin"],
            hold_frames={
                name: value.cpu().tolist() for name, value in self.holds.items()
            },
            measurement="pre-action frames; age bins [0,.6), [.6,2), [2,5), [5,10), [10,infinity); scan relief >2cm with all rays valid is neighborhood geometry, NOT confirmed under-foot support; four foot force norms >1N; arrival motion uses the last arrival frame (planar speed or absolute yaw rate >0.1); interruption takes precedence over phase completion",
        )
        return result


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
