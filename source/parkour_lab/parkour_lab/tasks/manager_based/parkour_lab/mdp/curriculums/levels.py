"""Pure curriculum-level configuration and Hydra normalization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class ParkourObstacleLevelCfg:
    """Task geometry and training targets for one curriculum level."""

    name: str
    obstacle_pos: tuple[float, float, float]
    obstacle_size: tuple[float, float, float]
    goal_pos: tuple[float, float, float]
    target_speed: float
    min_clearance: float

    def __post_init__(self) -> None:
        if len(self.obstacle_pos) != 3:
            raise ValueError(f"{self.name}: obstacle_pos must have length 3.")

        if len(self.obstacle_size) != 3:
            raise ValueError(f"{self.name}: obstacle_size must have length 3.")

        if len(self.goal_pos) != 3:
            raise ValueError(f"{self.name}: goal_pos must have length 3.")

        if any(size <= 0.0 for size in self.obstacle_size):
            raise ValueError(f"{self.name}: obstacle_size entries must be positive.")

        if self.target_speed < 0.0:
            raise ValueError(f"{self.name}: target_speed must be non-negative.")

        if self.min_clearance < 0.0:
            raise ValueError(f"{self.name}: min_clearance must be non-negative.")

        expected_center_z = 0.5 * self.obstacle_size[2]
        if abs(self.obstacle_pos[2] - expected_center_z) > 1.0e-6:
            raise ValueError(f"{self.name}: obstacle_pos.z should be obstacle_size.z / 2 for a box resting on ground.")


def coerce_level_cfg(level: ParkourObstacleLevelCfg | Mapping[str, object]) -> ParkourObstacleLevelCfg:
    """Return a typed level config from either Python or Hydra's mapping representation.

    Hydra may serialize nested dataclasses as dictionaries while composing the
    environment config. Normalizing at this boundary lets the rest of the code
    consistently use attribute access.
    """

    if isinstance(level, ParkourObstacleLevelCfg):
        return level
    if not isinstance(level, Mapping):
        raise TypeError(f"Curriculum level must be ParkourObstacleLevelCfg or a mapping, got {type(level).__name__}.")

    required_fields = {
        "name",
        "obstacle_pos",
        "obstacle_size",
        "goal_pos",
        "target_speed",
        "min_clearance",
    }
    missing_fields = required_fields.difference(level)
    if missing_fields:
        raise ValueError(f"Curriculum level is missing fields: {', '.join(sorted(missing_fields))}.")

    name = level["name"]
    if not isinstance(name, str):
        raise TypeError(f"Curriculum level name must be a string, got {type(name).__name__}.")

    return ParkourObstacleLevelCfg(
        name=name,
        obstacle_pos=_float_triplet(level["obstacle_pos"], field_name="obstacle_pos"),
        obstacle_size=_float_triplet(level["obstacle_size"], field_name="obstacle_size"),
        goal_pos=_float_triplet(level["goal_pos"], field_name="goal_pos"),
        target_speed=_float_value(level["target_speed"], field_name="target_speed"),
        min_clearance=_float_value(level["min_clearance"], field_name="min_clearance"),
    )


def _float_triplet(value: object, *, field_name: str) -> tuple[float, float, float]:
    """Convert a Hydra/Python sequence into a fixed three-float tuple."""

    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of three numbers.")

    try:
        values = tuple(_float_value(component, field_name=field_name) for component in cast(Iterable[object], value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a sequence of three numbers.") from error

    if len(values) != 3:
        raise ValueError(f"{field_name} must have length 3, got {len(values)}.")

    return values[0], values[1], values[2]


def _float_value(value: object, *, field_name: str) -> float:
    """Convert a numeric Hydra value to float with a field-specific error."""

    try:
        # Hydra values are dynamically typed at this serialization boundary.
        return float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must contain numeric values.") from error
