"""Learner-independent Go2 tasks with blind observations and native rewards."""

from .config import TASK_IDS, Go2Config, build_config

__all__ = ["TASK_IDS", "Go2Config", "build_config", "make"]


def make(config=None, *, training=None, render_mode=None):
    """Create after AppLauncher, optionally using a customized native config.

    A Go2Config defaults to training. A native configuration keeps the role set
    by build_config; supplying a conflicting role is an error.
    """
    from .task import Go2Task

    if training is not None and type(training) is not bool:
        raise ValueError("Training role must be a boolean")
    config = Go2Config() if config is None else config
    if isinstance(config, Go2Config):
        config = build_config(config, training=True if training is None else training)
    elif training is not None and training != getattr(
        config, "benchmark_training", None
    ):
        raise ValueError("Training role differs from the native task configuration")
    return Go2Task(config, render_mode=render_mode)
