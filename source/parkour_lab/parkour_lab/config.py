"""Current experiment settings, independent of the simulator and CLI.

Old run manifests and checkpoint recipes are intentionally not configuration
formats. Unknown fields fail early instead of silently changing an experiment.
"""

from dataclasses import asdict, dataclass, field, fields
import json
import math
from pathlib import Path


def positive(value, name, *, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be a positive {'integer' if integer else 'number'}"
        )


@dataclass(frozen=True)
class TaskConfig:
    terrain: str = "procedural"
    num_envs: int = 160
    seed: int = 42
    device: str = "cuda:0"
    difficulty_range: tuple[float, float] | None = None
    num_rows: int | None = None
    episode_length_s: float = 20.0
    traversal_layout: str = "standard"

    def __post_init__(self):
        if self.terrain not in ("flat", "procedural", "steps", "traversal"):
            raise ValueError("terrain must be flat, procedural, steps or traversal")
        if self.difficulty_range is None:
            object.__setattr__(
                self,
                "difficulty_range",
                {"steps": (0.15, 0.55), "traversal": (1.0, 1.0)}.get(
                    self.terrain, (0.05, 0.15)
                ),
            )
        if self.num_rows is None:
            object.__setattr__(self, "num_rows", 3 if self.terrain == "steps" else 1)
        positive(self.num_envs, "num_envs", integer=True)
        positive(self.num_rows, "num_rows", integer=True)
        positive(self.episode_length_s, "episode_length_s")
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or not isinstance(self.device, str)
            or not self.device
        ):
            raise ValueError("Require a uint32 seed and an explicit device")
        bounds = self.difficulty_range
        if (
            not isinstance(bounds, (list, tuple))
            or len(bounds) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
            or not 0 <= bounds[0] <= bounds[1] <= 1
        ):
            raise ValueError("difficulty_range must satisfy 0 <= low <= high <= 1")
        object.__setattr__(self, "difficulty_range", tuple(bounds))
        if self.traversal_layout not in ("standard", "step_ladder"):
            raise ValueError("Unknown traversal layout")


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
    task: TaskConfig = TaskConfig()
    method: MethodConfig = field(default_factory=MethodConfig)
    updates: int = 1000
    save_interval: int = 100

    def __post_init__(self):
        if not isinstance(self.task, TaskConfig) or not isinstance(
            self.method, MethodConfig
        ):
            raise ValueError("Require typed task and method settings")
        positive(self.updates, "updates", integer=True)
        positive(self.save_interval, "save_interval", integer=True)

    def validate_training(self):
        from parkour_lab.methods import get_backend

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


def construct(kind, data):
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(kind)}:
        raise ValueError(f"Unknown or malformed {kind.__name__} settings")
    return kind(**data)
