"""Training workflow configuration; task-only settings live with environments."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from parkour_lab.environments.task_config import TaskConfig, construct, positive


@dataclass(frozen=True)
class MethodConfig:
    name: str = "roa"
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        from parkour_lab.methods import get_backend

        if not isinstance(self.options, dict):
            raise ValueError("Method options must be an object")
        object.__setattr__(
            self, "options", get_backend(self.name).configure(self.options)
        )


@dataclass(frozen=True)
class ExperimentConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    updates: int = 1000
    save_interval: int = 100
    action_mode: str = "unbounded"

    def __post_init__(self):
        if not isinstance(self.task, TaskConfig) or not isinstance(
            self.method, MethodConfig
        ):
            raise ValueError("Require typed task and method settings")
        positive(self.updates, "updates", integer=True)
        positive(self.save_interval, "save_interval", integer=True)
        if self.action_mode not in ("unbounded", "joint_limits_v1"):
            raise ValueError("Unknown action_mode")

    def validate_training(self):
        from parkour_lab.methods import get_backend

        if self.task.terrain == "connected":
            raise ValueError("Connected terrain is currently evaluation-only")
        if self.task.bank_profile is not None:
            raise ValueError("Development bank attempts cannot be used for training")
        if self.task.terrain == "flat" and self.task.dynamics != "randomized":
            raise ValueError("Flat baseline training requires randomized dynamics")
        get_backend(self.method.name).validate_training(
            self.method.options, self.task.num_envs
        )

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Experiment configuration must be an object")
        return construct(
            cls,
            {
                **value,
                "task": construct(TaskConfig, value.get("task", {})),
                "method": construct(MethodConfig, value.get("method", {})),
            },
        )

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
