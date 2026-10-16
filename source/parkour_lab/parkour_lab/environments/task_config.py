"""Task settings independent of learning methods, simulators and command-line tools."""

import math
from dataclasses import dataclass, field, fields


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
    physics_hz: int = 200
    difficulty_range: tuple[float, float] | None = None
    num_rows: int | None = None
    episode_length_s: float = 20.0
    traversal_layout: str = "standard"
    linear_tracking_std: float = 0.5  # m/s; exp(-squared xy error / std**2)
    angular_tracking_std: float = 0.5  # rad/s; same kernel for body-z yaw
    rough_pivot_fraction: float = 0.0
    dynamics: str | None = None
    bank_profile: str | None = None
    terrain_group: str | None = None
    terrain_attempt_index: int | None = None
    rewards: dict = field(default_factory=dict)
    world: dict | None = None

    def __post_init__(self):
        from parkour_lab.environments.rewards import validate_rewards

        object.__setattr__(self, "rewards", validate_rewards(self.rewards))
        if self.terrain not in (
            "flat",
            "rough",
            "procedural",
            "steps",
            "traversal",
            "connected",
        ):
            raise ValueError("Unknown terrain")
        if self.difficulty_range is None:
            object.__setattr__(
                self,
                "difficulty_range",
                {
                    "rough": (0.0, 1.0),
                    "steps": (0.15, 0.55),
                    "traversal": (1.0, 1.0),
                    "connected": (1.0, 1.0),
                }.get(self.terrain, (0.05, 0.15)),
            )
        if self.num_rows is None:
            object.__setattr__(
                self, "num_rows", {"rough": 10, "steps": 3}.get(self.terrain, 1)
            )
        positive(self.num_envs, "num_envs", integer=True)
        positive(self.num_rows, "num_rows", integer=True)
        positive(self.episode_length_s, "episode_length_s")
        positive(self.linear_tracking_std, "linear_tracking_std")
        positive(self.angular_tracking_std, "angular_tracking_std")
        if (
            type(self.rough_pivot_fraction) not in (int, float)
            or not math.isfinite(self.rough_pivot_fraction)
            or not 0 <= self.rough_pivot_fraction <= 0.9
        ):
            raise ValueError("rough_pivot_fraction must be finite in [0, 0.9]")
        if type(self.physics_hz) is not int or self.physics_hz not in (200, 400):
            raise ValueError("physics_hz must be the integer 200 or 400")
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
        if self.terrain_group is not None or self.terrain_attempt_index is not None:
            from parkour_lab.environments.randomization import (
                terrain_development_attempt,
            )

            assignment = terrain_development_attempt(
                self.terrain_group, self.terrain_attempt_index
            )
            if (
                self.terrain != "connected"
                or self.bank_profile is not None
                or self.episode_length_s <= 30
            ):
                raise ValueError(
                    "A terrain development attempt requires connected playback, no flat profile and an episode longer than 30 s"
                )
            if self.world is not None and self.world != assignment["world"]:
                raise ValueError("World differs from the canonical terrain assignment")
            if self.dynamics is not None and self.dynamics != assignment["stratum"]:
                raise ValueError("Dynamics conflict with the terrain attempt stratum")
            object.__setattr__(self, "world", dict(assignment["world"]))
            object.__setattr__(self, "dynamics", assignment["stratum"])
        elif self.dynamics is None:
            object.__setattr__(self, "dynamics", "randomized")
        if self.dynamics not in ("nominal", "randomized"):
            raise ValueError("dynamics must be nominal or randomized")
        if self.terrain not in ("flat", "connected") and self.dynamics != "randomized":
            raise ValueError(
                "Nominal dynamics selection is supported only for flat or connected tasks"
            )
        if self.terrain == "connected":
            if self.num_envs != 1 or self.num_rows != 1 or tuple(bounds) != (1.0, 1.0):
                raise ValueError(
                    "Connected playback requires one environment, one row and difficulty (1, 1)"
                )
            if not isinstance(self.world, dict):
                raise ValueError("Connected playback requires a world object")
            from inspect import signature

            from parkour_lab.environments.terrains.worlds import build_world

            try:
                signature(build_world).bind(**self.world)
            except TypeError as error:
                raise ValueError(
                    f"Invalid connected world settings: {error}"
                ) from error
            object.__setattr__(self, "world", dict(self.world))
        elif self.world is not None:
            raise ValueError("world settings require connected terrain")
        if self.bank_profile is not None:
            from parkour_lab.evaluation.flat import PROFILES

            if (
                not isinstance(self.bank_profile, str)
                or self.bank_profile not in PROFILES
                or self.terrain != "flat"
                or self.num_envs != 100
                or self.episode_length_s <= 11
            ):
                raise ValueError(
                    "A flat development profile requires 100 rows and an episode longer than 11 s"
                )


def construct(kind, data):
    if not isinstance(data, dict) or set(data) - {f.name for f in fields(kind)}:
        raise ValueError(f"Unknown or malformed {kind.__name__} settings")
    return kind(**data)
