"""Pure curriculum-level configuration and Hydra reconstruction helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
import math
from typing import Any, cast

WAYPOINT_SURFACE_TOLERANCE_M = 0.05
"""Maximum marker-height offset accepted above a supporting surface."""


@dataclass(frozen=True)
class ParkourStructureCfg:
    """Physical course geometry created by a terrain-local mesh factory.

    A structure determines what is actually added to the terrain mesh. It does
    not by itself identify which faces are traversable. A
    :class:`ParkourSupportRegionCfg` can reference this structure by name to
    provide an explicit traversable-surface annotation for waypoint validation.
    The generic structure does not interpret the mesh's shape.
    """

    # Stable course-local identifier used by metadata and diagnostics.
    name: str

    # Callable used to create the structure's ``Trimesh`` object or objects.
    # It receives the keyword arguments stored in ``mesh_kwargs``.
    mesh_factory: Callable[..., object]

    # JSON-compatible keyword arguments forwarded directly to ``mesh_factory``.
    # Shape-specific values such as box extents or cylinder radii belong here.
    mesh_kwargs: dict[str, Any]

    # XYZ translation in meters relative to the center of the terrain tile.
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Roll, pitch, and yaw rotation in radians about the X, Y, and Z axes.
    orientation_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        _validate_name(self.name, field_name="structure name")
        if not callable(self.mesh_factory):
            raise TypeError("mesh_factory must be callable.")
        mesh_kwargs = _json_mapping(
            self.mesh_kwargs,
            field_name=f"{self.name} mesh_kwargs",
        )
        # Store normalized plain-Python data so later mesh creation and
        # metadata serialization do not need to normalize it again.
        object.__setattr__(self, "mesh_kwargs", mesh_kwargs)
        object.__setattr__(
            self,
            "position",
            _float_triplet(self.position, field_name="structure position"),
        )
        object.__setattr__(
            self,
            "orientation_rpy",
            _float_triplet(
                self.orientation_rpy,
                field_name="structure orientation_rpy",
            ),
        )

    def metadata(self) -> dict[str, object]:
        """Return a JSON-compatible description of this structure."""

        return {
            "name": self.name,
            "mesh_factory": (
                f"{self.mesh_factory.__module__}:{self.mesh_factory.__qualname__}"
                if hasattr(self.mesh_factory, "__module__")
                and hasattr(self.mesh_factory, "__qualname__")
                else repr(self.mesh_factory)
            ),
            "mesh_kwargs": dict(self.mesh_kwargs),
            "position": list(self.position),
            "orientation_rpy": list(self.orientation_rpy),
        }


@dataclass(frozen=True)
class ParkourWaypointCfg:
    """One ordered course waypoint in terrain-local XYZ coordinates."""

    position: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            _float_triplet(self.position, field_name="waypoint position"),
        )

    def metadata(self) -> dict[str, object]:
        """Return a JSON-compatible waypoint description."""

        return {"position": list(self.position)}


@dataclass(frozen=True)
class ParkourSupportRegionCfg:
    """Explicit horizontal surface on which a course waypoint may be placed.

    A :class:`ParkourStructureCfg` describes how to create physical mesh
    geometry, but an arbitrary mesh factory does not say which faces are safe
    course surfaces. Inferring that meaning from Trimesh geometry would couple
    configuration validation to individual shapes and mesh-inspection rules.
    A support region therefore records the intended traversable rectangle
    explicitly. The level uses it to reject a final waypoint that is outside a
    configured surface, and terrain validation uses the ground region to check
    that its footprint agrees with the generated tile.

    ``structure_name`` associates an elevated region with its physical mesh;
    ``None`` denotes the base ground created by the terrain generator. The
    annotation creates no geometry and is currently limited to horizontal,
    axis-aligned rectangles.
    """

    name: str

    # Name of the physical structure whose top surface this annotation
    # describes. ``None`` refers to the generated base ground instead.
    structure_name: str | None

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    surface_z: float = 0.0

    def __post_init__(self) -> None:
        _validate_name(self.name, field_name="support-region name")
        if self.structure_name is not None:
            _validate_name(
                self.structure_name,
                field_name=f"{self.name} structure_name",
            )
        x_range = _float_pair(self.x_range, field_name=f"{self.name} x_range")
        y_range = _float_pair(self.y_range, field_name=f"{self.name} y_range")
        if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
            raise ValueError(
                f"{self.name}: support-region ranges must have positive width."
            )
        object.__setattr__(self, "x_range", x_range)
        object.__setattr__(self, "y_range", y_range)
        object.__setattr__(
            self,
            "surface_z",
            _float_value(self.surface_z, field_name=f"{self.name} surface_z"),
        )

    def contains_xy(self, point: tuple[float, ...]) -> bool:
        """Return whether a point lies over this support's closed XY footprint."""

        return (
            self.x_range[0] <= point[0] <= self.x_range[1]
            and self.y_range[0] <= point[1] <= self.y_range[1]
        )

    def supports_waypoint(self, position: tuple[float, float, float]) -> bool:
        """Return whether a waypoint lies on this support within marker tolerance."""

        return self.contains_xy(position) and (
            abs(position[2] - self.surface_z) <= WAYPOINT_SURFACE_TOLERANCE_M
        )

    def metadata(self) -> dict[str, object]:
        """Return a JSON-compatible support-region description."""

        return {
            "name": self.name,
            "structure_name": self.structure_name,
            "x_range": list(self.x_range),
            "y_range": list(self.y_range),
            "surface_z": self.surface_z,
        }


@dataclass(frozen=True)
class ParkourDifficultyCfg:
    """Explicit curriculum rank and family-specific numeric parameters."""

    # Sortable curriculum rank used to arrange levels from easiest to hardest.
    # Larger values represent harder levels; equal values represent the same
    # rank and are allowed when two courses have comparable difficulty.
    # This is not Isaac Lab's normalized terrain difficulty in ``[0.0, 1.0]``;
    # values such as 0.0, 1.0, 2.0, and intermediate ranks are therefore valid.
    order: float

    # Obstacle-family-specific values that describe what makes this level hard.
    parameters: dict[str, float]

    def __post_init__(self) -> None:
        order = _float_value(self.order, field_name="curriculum difficulty rank")
        if order < 0.0:
            raise ValueError("Curriculum difficulty rank must be non-negative.")
        object.__setattr__(self, "order", order)
        object.__setattr__(
            self,
            "parameters",
            _float_mapping(self.parameters, field_name="difficulty parameters"),
        )

    def metadata(self) -> dict[str, object]:
        """Return a JSON-compatible difficulty description."""

        return {"order": self.order, "parameters": dict(self.parameters)}


@dataclass(frozen=True)
class ParkourLevelCfg:
    """Declarative geometry, route, and training targets for one course level."""

    # Stable logical identifier used in metadata, diagnostics, and validation
    # messages. It must be unique within one curriculum.
    name: str

    # Declarative category shared by geometrically related courses, such as
    # ``"step"`` or ``"gap"``. Runtime code must not branch on this label.
    obstacle_family: str

    # Route through the course in traversal order. Positions are XYZ offsets
    # in the terrain-local coordinate system; the final entry is the current
    # compatibility goal until waypoint progression is implemented.
    waypoints: tuple[ParkourWaypointCfg, ...]

    # Mesh-producing obstacles and platforms placed relative to the terrain
    # tile center. A flat course may leave this tuple empty.
    structures: tuple[ParkourStructureCfg, ...]

    # Traversable horizontal surfaces supporting waypoints. Each region
    # describes either the base ground or a surface associated with a named
    # structure, without assuming a particular mesh shape.
    support_regions: tuple[ParkourSupportRegionCfg, ...]

    # Desired velocity toward the active goal in meters per second. It is
    # exposed as a command and used to normalize goal-directed velocity reward.
    target_speed: float

    # Required robot-base clearance above the terrain in meters. Safety,
    # stability, and successful-goal checks consume this level-specific value.
    min_clearance: float

    # Explicit easiest-to-hardest rank plus obstacle-family-specific values.
    difficulty: ParkourDifficultyCfg

    def __post_init__(self) -> None:
        _validate_name(self.name, field_name="level name")
        _validate_name(self.obstacle_family, field_name="obstacle_family")

        _validate_typed_sequence(
            self.waypoints,
            ParkourWaypointCfg,
            owner=self.name,
            item_name="waypoint",
            required=True,
        )
        _validate_typed_sequence(
            self.structures,
            ParkourStructureCfg,
            owner=self.name,
            item_name="structure",
        )
        _validate_typed_sequence(
            self.support_regions,
            ParkourSupportRegionCfg,
            owner=self.name,
            item_name="support region",
            required=True,
        )
        if not isinstance(self.difficulty, ParkourDifficultyCfg):
            raise TypeError(f"{self.name}: difficulty must be a ParkourDifficultyCfg.")

        structure_by_name = {structure.name: structure for structure in self.structures}
        support_names = {region.name for region in self.support_regions}
        if len(structure_by_name) != len(self.structures):
            raise ValueError(f"{self.name}: structure names must be unique.")
        if len(support_names) != len(self.support_regions):
            raise ValueError(f"{self.name}: support-region names must be unique.")

        self._validate_support_references(structure_by_name)
        self._validate_final_waypoint()
        self._validate_training_targets()

    def _validate_support_references(
        self,
        structure_by_name: dict[str, ParkourStructureCfg],
    ) -> None:
        """Require every elevated support to reference physical course geometry."""

        for region in self.support_regions:
            # ``None`` represents the separately generated base ground, whose
            # footprint is checked later by ``validate_terrain_size``.
            if region.structure_name is None:
                continue

            # The support annotation is authoritative for arbitrary meshes, but
            # it must still identify physical geometry belonging to this level.
            if region.structure_name not in structure_by_name:
                raise ValueError(
                    f"{self.name}: support region {region.name!r} refers to unknown "
                    f"structure {region.structure_name!r}."
                )

    def _validate_final_waypoint(self) -> None:
        """Require the final waypoint to lie on a configured support region."""

        final_position = self.waypoints[-1].position
        if not any(
            region.supports_waypoint(final_position) for region in self.support_regions
        ):
            raise ValueError(
                f"{self.name}: final waypoint must lie on a valid support region."
            )

    def _validate_training_targets(self) -> None:
        """Validate scalar training targets and store them as finite floats."""

        target_speed = _float_value(self.target_speed, field_name="target_speed")
        if target_speed <= 0.0:
            raise ValueError(f"{self.name}: target_speed must be positive.")
        object.__setattr__(self, "target_speed", target_speed)

        min_clearance = _float_value(
            self.min_clearance,
            field_name="min_clearance",
        )
        if min_clearance < 0.0:
            raise ValueError(f"{self.name}: min_clearance must be non-negative.")
        object.__setattr__(self, "min_clearance", min_clearance)

    def validate_terrain_size(self, size: tuple[float, float]) -> None:
        """Validate base-ground annotations against the generated terrain tile."""

        size_x, size_y = _float_pair(size, field_name="terrain size")
        if size_x <= 0.0 or size_y <= 0.0:
            raise ValueError("terrain size must be positive.")

        ground_x_range = (-0.5 * size_x, 0.5 * size_x)
        ground_y_range = (-0.5 * size_y, 0.5 * size_y)
        base_regions = [
            region for region in self.support_regions if region.structure_name is None
        ]
        if len(base_regions) > 1:
            raise ValueError(
                f"{self.name}: the generated base ground may have at most one "
                "support-region annotation."
            )
        for region in base_regions:
            if not (
                _pairs_close(region.x_range, ground_x_range)
                and _pairs_close(region.y_range, ground_y_range)
                and math.isclose(region.surface_z, 0.0, abs_tol=1.0e-9)
            ):
                raise ValueError(
                    f"{self.name}: base-ground support region {region.name!r} "
                    "must match the generated terrain tile at z=0."
                )

    @property
    def goal_pos(self) -> tuple[float, float, float]:
        """Return the final waypoint for the single-goal runtime."""

        return self.waypoints[-1].position

    def metadata(self) -> dict[str, object]:
        """Return the complete JSON-compatible course description."""

        return {
            "name": self.name,
            "obstacle_family": self.obstacle_family,
            "waypoints": [waypoint.metadata() for waypoint in self.waypoints],
            "structures": [structure.metadata() for structure in self.structures],
            "support_regions": [region.metadata() for region in self.support_regions],
            "target_speed": self.target_speed,
            "min_clearance": self.min_clearance,
            "difficulty": self.difficulty.metadata(),
            # Preserve the old evaluation-report field until the introduction of
            # an active waypoint. It is derived, not independently configured.
            "goal_pos": list(self.goal_pos),
        }


def coerce_level_cfg(level: ParkourLevelCfg | Mapping[str, object]) -> ParkourLevelCfg:
    """Return a typed level config from either Python or Hydra's mapping representation.

    Hydra may serialize nested dataclasses as dictionaries while composing the
    environment config. This function only reconstructs those nested objects;
    each dataclass remains responsible for normalizing and validating its own
    primitive values.
    """

    if isinstance(level, ParkourLevelCfg):
        return level
    if not isinstance(level, Mapping):
        raise TypeError(
            f"Curriculum level must be ParkourLevelCfg or a mapping, got {type(level).__name__}."
        )

    required_fields = {
        "name",
        "obstacle_family",
        "waypoints",
        "structures",
        "support_regions",
        "target_speed",
        "min_clearance",
        "difficulty",
    }
    missing_fields = required_fields.difference(level)
    if missing_fields:
        raise ValueError(
            f"Curriculum level is missing fields: {', '.join(sorted(missing_fields))}."
        )

    return ParkourLevelCfg(
        name=cast(str, level["name"]),
        obstacle_family=cast(str, level["obstacle_family"]),
        waypoints=tuple(
            coerce_waypoint_cfg(waypoint)
            for waypoint in _sequence_value(
                level["waypoints"],
                field_name="waypoints",
            )
        ),
        structures=tuple(
            coerce_structure_cfg(structure)
            for structure in _sequence_value(
                level["structures"],
                field_name="structures",
            )
        ),
        support_regions=tuple(
            coerce_support_region_cfg(region)
            for region in _sequence_value(
                level["support_regions"],
                field_name="support_regions",
            )
        ),
        target_speed=cast(float, level["target_speed"]),
        min_clearance=cast(float, level["min_clearance"]),
        difficulty=coerce_difficulty_cfg(level["difficulty"]),
    )


def coerce_structure_cfg(
    structure: ParkourStructureCfg | Mapping[str, object],
) -> ParkourStructureCfg:
    """Reconstruct one structure from its typed or Hydra mapping representation."""

    if isinstance(structure, ParkourStructureCfg):
        return structure
    if not isinstance(structure, Mapping):
        raise TypeError(
            f"Parkour structure must be a configuration or mapping, got {type(structure).__name__}."
        )

    missing_fields = {
        "name",
        "mesh_factory",
        "mesh_kwargs",
    }.difference(structure)
    if missing_fields:
        raise ValueError(
            f"Parkour structure is missing fields: {', '.join(sorted(missing_fields))}."
        )

    mesh_factory = _resolve_mesh_factory(structure["mesh_factory"])

    return ParkourStructureCfg(
        name=cast(str, structure["name"]),
        mesh_factory=mesh_factory,
        mesh_kwargs=cast(dict[str, Any], structure["mesh_kwargs"]),
        position=cast(
            tuple[float, float, float],
            structure.get("position", (0.0, 0.0, 0.0)),
        ),
        orientation_rpy=cast(
            tuple[float, float, float],
            structure.get("orientation_rpy", (0.0, 0.0, 0.0)),
        ),
    )


def coerce_waypoint_cfg(
    waypoint: ParkourWaypointCfg | Mapping[str, object],
) -> ParkourWaypointCfg:
    """Reconstruct one ordered waypoint from Python or Hydra data."""

    if isinstance(waypoint, ParkourWaypointCfg):
        return waypoint
    if not isinstance(waypoint, Mapping) or "position" not in waypoint:
        raise TypeError("Waypoint must be a mapping containing 'position'.")
    return ParkourWaypointCfg(
        position=cast(tuple[float, float, float], waypoint["position"])
    )


def coerce_support_region_cfg(
    region: ParkourSupportRegionCfg | Mapping[str, object],
) -> ParkourSupportRegionCfg:
    """Reconstruct one support-region annotation from Python or Hydra data."""

    if isinstance(region, ParkourSupportRegionCfg):
        return region
    if not isinstance(region, Mapping):
        raise TypeError("Support region must be a configuration or mapping.")
    missing_fields = {"name", "structure_name", "x_range", "y_range"}.difference(region)
    if missing_fields:
        raise ValueError(
            "Support region is missing fields: " f"{', '.join(sorted(missing_fields))}."
        )
    return ParkourSupportRegionCfg(
        name=cast(str, region["name"]),
        structure_name=cast(str | None, region["structure_name"]),
        x_range=cast(tuple[float, float], region["x_range"]),
        y_range=cast(tuple[float, float], region["y_range"]),
        surface_z=cast(float, region.get("surface_z", 0.0)),
    )


def coerce_difficulty_cfg(difficulty: object) -> ParkourDifficultyCfg:
    """Reconstruct a difficulty config from typed or Hydra data."""

    if isinstance(difficulty, ParkourDifficultyCfg):
        return difficulty
    if not isinstance(difficulty, Mapping):
        raise TypeError("difficulty must be a ParkourDifficultyCfg or mapping.")
    missing_fields = {"order", "parameters"}.difference(difficulty)
    if missing_fields:
        raise ValueError(
            f"Difficulty is missing fields: {', '.join(sorted(missing_fields))}."
        )
    return ParkourDifficultyCfg(
        order=cast(float, difficulty["order"]),
        parameters=cast(dict[str, float], difficulty["parameters"]),
    )


def coerce_and_validate_levels(
    levels: Iterable[ParkourLevelCfg | Mapping[str, object]],
) -> tuple[ParkourLevelCfg, ...]:
    """Normalize a curriculum and require an easiest-to-hardest ordering."""

    normalized = tuple(coerce_level_cfg(level) for level in levels)
    if not normalized:
        raise ValueError("Parkour curriculum levels must not be empty.")

    names = [level.name for level in normalized]
    if len(names) != len(set(names)):
        raise ValueError("Parkour curriculum level names must be unique.")

    for field_name, values in (
        ("difficulty", [level.difficulty.order for level in normalized]),
        ("target speed", [level.target_speed for level in normalized]),
        ("minimum clearance", [level.min_clearance for level in normalized]),
    ):
        if any(current > following for current, following in zip(values, values[1:])):
            raise ValueError(f"Parkour curriculum {field_name} must be non-decreasing.")

    return normalized


def _validate_typed_sequence(
    values: tuple[object, ...],
    expected_type: type[object],
    *,
    owner: str,
    item_name: str,
    required: bool = False,
) -> None:
    """Validate the contents and optional non-emptiness of a config tuple."""

    if required and not values:
        raise ValueError(f"{owner}: at least one {item_name} is required.")
    if not all(isinstance(value, expected_type) for value in values):
        raise TypeError(
            f"{owner}: {item_name}s must contain {expected_type.__name__} values."
        )


def _float_triplet(value: object, *, field_name: str) -> tuple[float, float, float]:
    """Convert a Hydra/Python sequence into a fixed three-float tuple."""

    values = tuple(
        _float_value(component, field_name=field_name)
        for component in _sequence_value(value, field_name=field_name)
    )
    if len(values) != 3:
        raise ValueError(f"{field_name} must have length 3, got {len(values)}.")

    return values[0], values[1], values[2]


def _float_pair(value: object, *, field_name: str) -> tuple[float, float]:
    """Convert a Hydra/Python sequence into a fixed two-float tuple."""

    values = tuple(
        _float_value(component, field_name=field_name)
        for component in _sequence_value(value, field_name=field_name)
    )
    if len(values) != 2:
        raise ValueError(f"{field_name} must have length 2, got {len(values)}.")
    return values[0], values[1]


def _float_value(value: object, *, field_name: str) -> float:
    """Convert a numeric Hydra value to float with a field-specific error."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must contain numeric values, not bool.")
    try:
        # Hydra values are dynamically typed at this serialization boundary.
        converted = float(cast(Any, value))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must contain numeric values.") from error
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must contain finite numeric values.")
    return converted


def _float_mapping(value: object, *, field_name: str) -> dict[str, float]:
    """Convert a string-keyed Hydra mapping into finite float values."""

    if not isinstance(value, Mapping) or not all(
        isinstance(name, str) and name.strip() for name in value
    ):
        raise TypeError(f"{field_name} must be a mapping with non-empty string keys.")
    return {
        name: _float_value(component, field_name=f"{field_name}.{name}")
        for name, component in value.items()
    }


def _json_mapping(value: object, *, field_name: str) -> dict[str, object]:
    """Return a string-keyed mapping containing only JSON-compatible values."""

    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must be a mapping with string keys.")
    return {
        key: _json_value(item, field_name=f"{field_name}.{key}")
        for key, item in value.items()
    }


def _json_value(value: object, *, field_name: str) -> object:
    """Normalize one declarative value to Hydra/JSON-compatible Python data."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must be finite.")
        return value
    if isinstance(value, Mapping):
        return _json_mapping(value, field_name=field_name)
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]

    # NumPy arrays/scalars and similar numeric containers commonly appear in
    # mesh arguments. Convert them through their public Python-data methods.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist(), field_name=field_name)
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item(), field_name=field_name)
    raise TypeError(f"{field_name} is not Hydra/JSON-compatible.")


def _pairs_close(
    actual: tuple[float, float],
    expected: tuple[float, float],
) -> bool:
    """Return whether two coordinate ranges agree within numeric tolerance."""

    return all(
        math.isclose(actual_value, expected_value, abs_tol=1.0e-9)
        for actual_value, expected_value in zip(actual, expected)
    )


def _sequence_value(value: object, *, field_name: str) -> tuple[object, ...]:
    """Return a non-string Hydra/Python iterable as a tuple."""

    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{field_name} must be a sequence.")
    try:
        return tuple(cast(Iterable[object], value))
    except TypeError as error:
        raise TypeError(f"{field_name} must be a sequence.") from error


def _validate_name(value: str, *, field_name: str) -> None:
    """Require a non-empty identifier-like metadata string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


def _resolve_mesh_factory(value: object) -> Callable[..., object]:
    """Resolve Isaac Lab's Hydra callable representation at the config boundary."""

    if callable(value):
        return cast(Callable[..., object], value)
    if not isinstance(value, str):
        raise TypeError(
            f"mesh_factory must be callable or a 'module:attribute' string, got {type(value).__name__}."
        )

    try:
        module_name, attribute_path = value.split(":", maxsplit=1)
        factory: object = import_module(module_name)
        for attribute_name in attribute_path.split("."):
            factory = getattr(factory, attribute_name)
    except (AttributeError, ImportError, ValueError) as error:
        raise ValueError(
            f"Could not resolve mesh_factory '{value}'. Expected 'module:attribute'."
        ) from error

    if not callable(factory):
        raise TypeError(f"Resolved mesh_factory '{value}' is not callable.")
    return cast(Callable[..., object], factory)
