"""Completed-update save barriers and a small index of native SB3 checkpoints."""

import json
import math
from pathlib import Path
from time import perf_counter

from parkour_lab.provenance import write_json
from parkour_lab.runtime.console import log_event

from .checkpoint import save_checkpoint


def validate_schedule(model, algorithm, initial, target, interval, milestones):
    """Reject unsupported schedules and unaligned targets before collection."""
    if type(interval) is not int or interval <= 0:
        raise ValueError("Checkpoint update interval must be a positive integer")
    if type(model._n_updates) is not int or model._n_updates < 0:
        raise ValueError("Require a nonnegative completed learner update counter")
    for name in ("lr_schedule", "clip_range", "clip_range_vf"):
        schedule = getattr(model, name, None)
        if callable(schedule):
            values = [float(schedule(point)) for point in (0.0, 0.5, 1.0)]
            if not all(math.isfinite(value) and value == values[0] for value in values):
                raise ValueError(
                    "Checkpoint segments require constant learning schedules"
                )
    if type(milestones) not in (tuple, list) or any(
        type(value) is not int for value in milestones
    ):
        raise ValueError("Milestones must be an ordered integer sequence")
    milestones = tuple(milestones)
    if list(milestones) != sorted(set(milestones)) or any(
        not initial < value <= target for value in milestones
    ):
        raise ValueError(
            "Milestones must be unique, ordered and within the remaining allowance"
        )
    if algorithm == "ppo":
        if (
            type(model.n_epochs) is not int
            or model.n_epochs <= 0
            or model.target_kl is not None
            or interval % model.n_epochs
            or model._n_updates % model.n_epochs
        ):
            raise ValueError(
                "PPO saves require complete fixed epochs and an aligned update interval"
            )
        quantum = model.n_envs * model.n_steps
    else:
        if (
            model.train_freq.frequency != 1
            or model.train_freq.unit.value != "step"
            or model.gradient_steps != 1
            or type(model.learning_starts) is not int
            or model.learning_starts < 0
        ):
            raise ValueError(
                "SAC saves require one collection step and one gradient step"
            )
        quantum = model.n_envs
        if (
            model._n_updates == 0
            and min((*milestones, target)) <= model.learning_starts
        ):
            raise ValueError(
                "Checkpoint targets must include a completed SAC update after warm-up"
            )
    if target <= initial or any(
        (value - initial) % quantum for value in (*milestones, target)
    ):
        raise ValueError(
            f"Remaining allowance and milestones must be positive multiples of {quantum} transitions"
        )
    return milestones


def next_snapshot(model, algorithm, *, interval, milestones, target):
    """Compute the next whole-update barrier without stepping one batch at a time."""
    updates = interval - model._n_updates % interval
    if algorithm == "ppo":
        transitions = updates // model.n_epochs * model.n_envs * model.n_steps
    else:
        warm_up = max(0, (model.learning_starts - model.num_timesteps) // model.n_envs)
        transitions = (updates + warm_up) * model.n_envs
    periodic = model.num_timesteps + transitions
    future = [value for value in milestones if value > model.num_timesteps]
    point = min(periodic, target, *future)
    reasons = [
        reason
        for applies, reason in (
            (point == periodic, "periodic"),
            (point in milestones, "milestone"),
            (point == target, "terminal"),
        )
        if applies
    ]
    return point, reasons


class SnapshotIndex:
    """Publish only completed bundles; saves never reset the live environment."""

    def __init__(
        self, task, *, output, run_id, algorithm, target, interval, milestones, resume
    ):
        self.output, self.run_id = Path(output), run_id
        self.started = perf_counter()
        self.completed = set()
        if resume is not None:
            progress = json.loads((Path(resume) / "progress.json").read_text())
            self.completed.update(progress["milestones"])
        self.record = dict(
            version="sb3_go2_checkpoint_index_v1",
            algorithm=algorithm,
            run_id=run_id,
            training_seed=task.metadata["seed"],
            save_every_updates=interval,
            update_counter_meaning=(
                "SB3 PPO epochs" if algorithm == "ppo" else "SB3 SAC gradient steps"
            ),
            target_interactions=target,
            requested_milestones=list(milestones),
            initial_interactions=dict(task.interactions),
            timing_scope="current training call, excluding environment/model setup and separate evaluation",
            complete=False,
            checkpoints=[],
        )

    def save(self, model, task, reasons):
        point = task.interactions["cumulative"]
        terminal = "terminal" in reasons
        relative = (
            Path("checkpoint") if terminal else Path("checkpoints") / f"step_{point}"
        )
        if terminal or "milestone" in reasons:
            self.completed.add(point)
        receipt = save_checkpoint(
            model,
            task,
            self.output / relative,
            run_id=self.run_id,
            milestones=tuple(sorted(self.completed)),
        )
        self.record["checkpoints"].append(
            dict(
                path=relative.as_posix(),
                reasons=list(reasons),
                training_interactions=point,
                learner_timesteps=model.num_timesteps,
                external_update_counter=model._n_updates,
                reached_milestones=sorted(self.completed),
                checkpoint_id=receipt["checkpoint_id"],
                checkpoint_sha256=receipt["checkpoint_sha256"],
                training_loop_wall_time_s=perf_counter() - self.started,
            )
        )
        self.record["complete"] = terminal
        temporary = self.output / "checkpoint_index.json.tmp"
        write_json(temporary, self.record)
        temporary.replace(self.output / "checkpoint_index.json")
        log_event(
            "checkpoint_saved",
            algorithm=self.record["algorithm"],
            training_seed=self.record["training_seed"],
            interactions=dict(task.interactions),
            completed_updates=int(model._n_updates),
            checkpoint_path=str(self.output / relative),
            reasons=list(reasons),
        )
        return receipt
