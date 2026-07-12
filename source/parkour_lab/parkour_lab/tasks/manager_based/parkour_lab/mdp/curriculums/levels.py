"""Pure curriculum-level configuration and Hydra normalization helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast


@dataclass(frozen=True)
class ParkourStructureCfg:
    """A mesh factory invocation and its terrain-local rigid transform.

    `orientation_rpy` is expressed as roll, pitch, and yaw in radians.
    """

    mesh_factory: Callable[..., object]
    mesh_kwargs: dict[str, Any]
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not callable(self.mesh_factory):
            raise TypeError("mesh_factory must be callable.")
        if not isinstance(self.mesh_kwargs, Mapping) or not all(isinstance(name, str) for name in self.mesh_kwargs):
            raise TypeError("mesh_kwargs must be a mapping with string keys.")
        if len(self.position) != 3:
            raise ValueError("structure position must have length 3.")
        if len(self.orientation_rpy) != 3:
            raise ValueError("structure orientation_rpy must have length 3.")

    def metadata(self) -> dict[str, object]:
        """Return a JSON-compatible description of this structure."""

        return {
            "mesh_factory": (
                f"{self.mesh_factory.__module__}:{self.mesh_factory.__qualname__}"
                if hasattr(self.mesh_factory, "__module__") and hasattr(self.mesh_factory, "__qualname__")
                else repr(self.mesh_factory)
            ),
            "mesh_kwargs": dict(self.mesh_kwargs),
            "position": list(self.position),
            "orientation_rpy": list(self.orientation_rpy),
        }


@dataclass(frozen=True)
class ParkourLevelCfg:
    """Composable terrain geometry and training targets for one level."""

    name: str
    structures: tuple[ParkourStructureCfg, ...]
    goal_pos: tuple[float, float, float]
    target_speed: float
    min_clearance: float

    def __post_init__(self) -> None:
        if len(self.goal_pos) != 3:
            raise ValueError(f"{self.name}: goal_pos must have length 3.")

        if not all(isinstance(structure, ParkourStructureCfg) for structure in self.structures):
            raise TypeError(f"{self.name}: structures must contain supported parkour structure configurations.")

        if self.target_speed <= 0.0:
            raise ValueError(f"{self.name}: target_speed must be positive.")

        if self.min_clearance < 0.0:
            raise ValueError(f"{self.name}: min_clearance must be non-negative.")


def coerce_level_cfg(level: ParkourLevelCfg | Mapping[str, object]) -> ParkourLevelCfg:
    """Return a typed level config from either Python or Hydra's mapping representation.

    Hydra may serialize nested dataclasses as dictionaries while composing the
    environment config. Normalizing at this boundary lets the rest of the code
    consistently use attribute access.
    """

    if isinstance(level, ParkourLevelCfg):
        return level
    if not isinstance(level, Mapping):
        raise TypeError(f"Curriculum level must be ParkourLevelCfg or a mapping, got {type(level).__name__}.")

    required_fields = {
        "name",
        "structures",
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

    structures = level["structures"]
    if isinstance(structures, (str, bytes)):
        raise TypeError("structures must be a sequence of parkour structure configurations.")

    try:
        normalized_structures = tuple(
            coerce_structure_cfg(structure) for structure in cast(Iterable[object], structures)
        )
    except TypeError as error:
        raise TypeError("structures must be a sequence of parkour structure configurations.") from error

    return ParkourLevelCfg(
        name=name,
        structures=normalized_structures,
        goal_pos=_float_triplet(level["goal_pos"], field_name="goal_pos"),
        target_speed=_float_value(level["target_speed"], field_name="target_speed"),
        min_clearance=_float_value(level["min_clearance"], field_name="min_clearance"),
    )


def coerce_structure_cfg(
    structure: ParkourStructureCfg | Mapping[str, object],
) -> ParkourStructureCfg:
    """Normalize one structure from its typed or Hydra mapping representation."""

    if isinstance(structure, ParkourStructureCfg):
        return structure
    if not isinstance(structure, Mapping):
        raise TypeError(f"Parkour structure must be a configuration or mapping, got {type(structure).__name__}.")

    missing_fields = {"mesh_factory", "mesh_kwargs"}.difference(structure)
    if missing_fields:
        raise ValueError(f"Parkour structure is missing fields: {', '.join(sorted(missing_fields))}.")

    mesh_factory = _resolve_mesh_factory(structure["mesh_factory"])

    mesh_kwargs = structure["mesh_kwargs"]
    if not isinstance(mesh_kwargs, Mapping):
        raise TypeError(f"mesh_kwargs must be a mapping, got {type(mesh_kwargs).__name__}.")

    return ParkourStructureCfg(
        mesh_factory=mesh_factory,
        mesh_kwargs=dict(mesh_kwargs),
        position=_float_triplet(structure.get("position", (0.0, 0.0, 0.0)), field_name="structure position"),
        orientation_rpy=_float_triplet(
            structure.get("orientation_rpy", (0.0, 0.0, 0.0)),
            field_name="structure orientation_rpy",
        ),
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


def _resolve_mesh_factory(value: object) -> Callable[..., object]:
    """Resolve Isaac Lab's Hydra callable representation at the config boundary."""

    if callable(value):
        return cast(Callable[..., object], value)
    if not isinstance(value, str):
        raise TypeError(f"mesh_factory must be callable or a 'module:attribute' string, got {type(value).__name__}.")

    try:
        module_name, attribute_path = value.split(":", maxsplit=1)
        factory: object = import_module(module_name)
        for attribute_name in attribute_path.split("."):
            factory = getattr(factory, attribute_name)
    except (AttributeError, ImportError, ValueError) as error:
        raise ValueError(f"Could not resolve mesh_factory '{value}'. Expected 'module:attribute'.") from error

    if not callable(factory):
        raise TypeError(f"Resolved mesh_factory '{value}' is not callable.")
    return cast(Callable[..., object], factory)
