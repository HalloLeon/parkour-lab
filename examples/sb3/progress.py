"""Read-only, throttled console visibility for an external learner."""

import math
from time import perf_counter

from stable_baselines3.common.callbacks import BaseCallback

from parkour_lab.runtime.console import is_enabled, log_event


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _cached_diagnostics(model, task):
    result = {}
    episodes = model.ep_info_buffer
    if episodes:
        result["recent_episode_return"] = _finite(
            sum(row["r"] for row in episodes) / len(episodes)
        )
        result["recent_episode_length_steps"] = _finite(
            sum(row["l"] for row in episodes) / len(episodes)
        )
    result["learner"] = {
        key.removeprefix("train/"): _finite(value)
        for key, value in model.logger.name_to_value.items()
        if key.startswith("train/")
    }
    native = getattr(task, "_env", None)
    if native is not None:
        from parkour_lab.environments.terrains.curriculum import (
            curriculum_diagnostics,
        )

        result["curriculum"] = curriculum_diagnostics(native)
        result["command_samples"] = dict(
            native.command_manager.get_term("base_velocity").sampling_report
        )
    return result


class TrainingProgress(BaseCallback):
    """Log existing counters/statistics every ten seconds; never sense or step."""

    def __init__(self, task, *, algorithm, target):
        super().__init__()
        self.task, self.algorithm, self.target = task, algorithm, target
        self.started = self.last = perf_counter()
        self.initial = task.interactions["cumulative"]

    def emit(self, event, model):
        if not is_enabled():
            return
        now = perf_counter()
        self.last = now
        elapsed = now - self.started
        counts = dict(self.task.interactions)
        log_event(
            event,
            algorithm=self.algorithm,
            training_seed=self.task.metadata["seed"],
            interactions=counts,
            target_interactions=self.target,
            learner_timesteps=int(model.num_timesteps),
            completed_updates=int(model._n_updates),
            elapsed_s=round(elapsed, 3),
            interactions_per_s=round(
                (counts["cumulative"] - self.initial) / max(elapsed, 1e-9), 2
            ),
            **_cached_diagnostics(model, self.task),
        )

    def _on_step(self):
        if is_enabled() and perf_counter() - self.last >= 10.0:
            self.emit("training_progress", self.model)
        return True
