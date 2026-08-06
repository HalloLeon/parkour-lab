"""Declarative parkour curriculum models, support geometry, and Hydra reconstruction helpers."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

WAYPOINT_SURFACE_TOLERANCE_M = 0.05
"""Maximum perpendicular waypoint distance accepted from a supporting plane."""

_GEOMETRY_TOLERANCE = 1.0e-9
"""Floating-point tolerance used by geometry predicates."""


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
        """Return a JSON-compatible description of this difficulty."""

        return {"order": self.order, "parameters": dict(self.parameters)}


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

    # Hydra/JSON-compatible keyword arguments forwarded directly to
    # ``mesh_factory``. Shape-specific values such as box extents or cylinder
    # radii belong here.
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
                if hasattr(self.mesh_factory, "__module__") and hasattr(self.mesh_factory, "__qualname__")
                else repr(self.mesh_factory)
            ),
            "mesh_kwargs": dict(self.mesh_kwargs),
            "position": list(self.position),
            "orientation_rpy": list(self.orientation_rpy),
        }


@dataclass(frozen=True)
class ParkourSupportRegionCfg:
    """Ordered convex planar surface on which a waypoint may be placed.

    A :class:`ParkourStructureCfg` describes how to create physical mesh
    geometry, but an arbitrary mesh factory does not say which faces are safe
    course surfaces. Inferring that meaning from Trimesh geometry would couple
    configuration validation to individual shapes and mesh-inspection rules.
    A support region therefore records the intended traversable polygon
    explicitly. Vertices use terrain-local XYZ coordinates and must be ordered
    counter-clockwise when viewed from the upward-facing side of the plane.

    ``structure_name`` associates an elevated region with its separately
    configured physical mesh; ``None`` denotes one generated base-ground
    patch. Named regions remain annotations and do not duplicate their
    structure geometry. Base-ground regions are deliberately restricted to
    horizontal, axis-aligned rectangles because they are converted to boxes by
    :func:`base_ground_structures`.
    """

    # Stable course-local identifier used by metadata and diagnostics.
    name: str

    # Name of the physical structure whose top surface this annotation
    # describes. ``None`` makes this region a generated base-ground patch.
    structure_name: str | None

    # Polygon boundary in terrain-local XYZ coordinates. This is the single
    # canonical support representation; ranges and the plane normal are
    # derived from it rather than stored independently.
    vertices: tuple[tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        _validate_name(self.name, field_name="support-region name")
        if self.structure_name is not None:
            _validate_name(
                self.structure_name,
                field_name=f"{self.name} structure_name",
            )
        vertices = tuple(
            _float_triplet(vertex, field_name=f"{self.name} vertices[{index}]")
            for index, vertex in enumerate(_sequence_value(self.vertices, field_name=f"{self.name} vertices"))
        )
        _validate_support_polygon(self.name, vertices)
        object.__setattr__(self, "vertices", vertices)

    @classmethod
    def horizontal_rectangle(
        cls,
        *,
        name: str,
        structure_name: str | None,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        surface_z: float = 0.0,
    ) -> ParkourSupportRegionCfg:
        """Create a horizontal axis-aligned support with CCW vertex winding."""

        normalized_x = _float_pair(x_range, field_name=f"{name} x_range")
        normalized_y = _float_pair(y_range, field_name=f"{name} y_range")
        normalized_z = _float_value(surface_z, field_name=f"{name} surface_z")
        if normalized_x[0] >= normalized_x[1] or normalized_y[0] >= normalized_y[1]:
            raise ValueError(f"{name}: support-region ranges must have positive width.")
        x_min, x_max = normalized_x
        y_min, y_max = normalized_y
        return cls(
            name=name,
            structure_name=structure_name,
            vertices=(
                (x_min, y_min, normalized_z),
                (x_max, y_min, normalized_z),
                (x_max, y_max, normalized_z),
                (x_min, y_max, normalized_z),
            ),
        )

    @property
    def normal(self) -> tuple[float, float, float]:
        """Return the polygon's upward-facing unit plane normal."""

        return _unit_polygon_normal(self.vertices)

    @property
    def x_range(self) -> tuple[float, float]:
        """Return the polygon's derived closed X extent."""

        coordinates = tuple(vertex[0] for vertex in self.vertices)
        return min(coordinates), max(coordinates)

    @property
    def y_range(self) -> tuple[float, float]:
        """Return the polygon's derived closed Y extent."""

        coordinates = tuple(vertex[1] for vertex in self.vertices)
        return min(coordinates), max(coordinates)

    def boundary_segments_xyz(
        self,
    ) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
        """Return the ordered XYZ boundary segments of this support.

        The runtime edge penalty consumes these segments directly. Keeping the
        representation in metric course coordinates avoids a resolution-bound
        height-field mask and gives every environment the same exact geometry.
        """

        return tuple(
            (vertex, self.vertices[(index + 1) % len(self.vertices)]) for index, vertex in enumerate(self.vertices)
        )

    def metadata(self) -> dict[str, object]:
        """Return a JSON-compatible description of this support region."""

        return {
            "name": self.name,
            "structure_name": self.structure_name,
            "vertices": [list(vertex) for vertex in self.vertices],
        }

    def supports_waypoint(self, position: tuple[float, float, float]) -> bool:
        """Return whether a waypoint lies on this support within surface tolerance."""

        # Projecting the waypoint's displacement onto the unit normal gives its
        # signed perpendicular distance from the support plane.
        point = _float_triplet(position, field_name="waypoint position")
        normal = self.normal
        relative = _subtract_xyz(point, self.vertices[0])
        plane_distance = _dot_xyz(relative, normal)
        if abs(plane_distance) > WAYPOINT_SURFACE_TOLERANCE_M:
            return False

        # Remove the normal component to project the waypoint orthogonally onto
        # the support plane before testing polygon containment.
        projected = (
            point[0] - plane_distance * normal[0],
            point[1] - plane_distance * normal[1],
            point[2] - plane_distance * normal[2],
        )

        # A convex CCW polygon is the intersection of the left half-planes of
        # its directed boundary edges. The cross and dot products below are
        # non-negative exactly when the projected waypoint is left of or on an
        # edge. The small negative tolerance preserves boundary points despite
        # floating-point error.
        return all(
            _dot_xyz(
                _cross_xyz(
                    _subtract_xyz(end, start),
                    _subtract_xyz(projected, start),
                ),
                normal,
            )
            >= -_GEOMETRY_TOLERANCE
            for start, end in self.boundary_segments_xyz()
        )


@dataclass(frozen=True)
class ParkourWaypointCfg:
    """One ordered course waypoint in terrain-local XYZ coordinates.

    ``support_region_name`` distinguishes an immediate route-control target
    from a physical target that requires both root proximity and a contacted
    foot on one exact configured support. ``root_reach_radius`` overrides the
    curriculum default for this waypoint only. This keeps landing semantics
    declarative and independent of obstacle-family labels.

    ``is_rewarded_milestone`` distinguishes physical obstacle progress from a
    waypoint used only to align or redirect the route. Final waypoints must
    leave it false because safe course completion has its own reward.

    ``is_terminal_landing`` marks the exceptional case where the final target
    is itself the obstacle landing rather than a post-landing exit target.
    """

    position: tuple[float, float, float]
    support_region_name: str | None = None
    is_rewarded_milestone: bool = False
    is_terminal_landing: bool = False
    root_reach_radius: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position",
            _float_triplet(self.position, field_name="waypoint position"),
        )
        if self.support_region_name is not None:
            _validate_name(
                self.support_region_name,
                field_name="waypoint support_region_name",
            )
        if not isinstance(self.is_rewarded_milestone, bool):
            raise TypeError("is_rewarded_milestone must be a Boolean.")
        if not isinstance(self.is_terminal_landing, bool):
            raise TypeError("is_terminal_landing must be a Boolean.")
        if self.root_reach_radius is not None:
            root_reach_radius = _float_value(
                self.root_reach_radius,
                field_name="waypoint root_reach_radius",
            )
            if root_reach_radius <= 0.0:
                raise ValueError("waypoint root_reach_radius must be positive.")
            object.__setattr__(self, "root_reach_radius", root_reach_radius)

    def metadata(self) -> dict[str, object]:
        """Return a JSON-compatible description of this waypoint."""

        return {
            "position": list(self.position),
            "support_region_name": self.support_region_name,
            "is_rewarded_milestone": self.is_rewarded_milestone,
            "is_terminal_landing": self.is_terminal_landing,
            "root_reach_radius": self.root_reach_radius,
        }


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
    # in the terrain-local coordinate system. Intermediate entries may be
    # directional guides; only the final entry must lie on a support region.
    waypoints: tuple[ParkourWaypointCfg, ...]

    # Mesh-producing obstacles and platforms placed relative to the terrain
    # tile center. A flat course may leave this tuple empty.
    structures: tuple[ParkourStructureCfg, ...]

    # Traversable planar surfaces supporting waypoints. A base-ground region
    # (``structure_name=None``) must be a horizontal axis-aligned rectangle
    # and also produces one collision patch; named regions annotate a surface
    # of existing structure geometry and may be inclined convex polygons.
    support_regions: tuple[ParkourSupportRegionCfg, ...]

    # Desired velocity toward the active waypoint in meters per second. It is
    # exposed as a command and normalizes waypoint-directed velocity reward.
    target_speed: float

    # Required robot-base clearance above the terrain in meters. Safety,
    # stability and final-waypoint completion consume this level-specific value.
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
        self._validate_waypoint_supports()
        self._validate_rewarded_milestones()
        self._validate_final_waypoint()
        self._validate_training_targets()

    @property
    def final_waypoint_pos(self) -> tuple[float, float, float]:
        """Return the final course waypoint."""

        return self.waypoints[-1].position

    def metadata(self) -> dict[str, object]:
        """Return a JSON-compatible description of this course."""

        return {
            "name": self.name,
            "obstacle_family": self.obstacle_family,
            "waypoints": [waypoint.metadata() for waypoint in self.waypoints],
            "structures": [structure.metadata() for structure in self.structures],
            "support_regions": [region.metadata() for region in self.support_regions],
            "target_speed": self.target_speed,
            "min_clearance": self.min_clearance,
            "difficulty": self.difficulty.metadata(),
            # Keep the derived endpoint convenient for reports and external
            # tooling; runtime navigation uses the ordered ``waypoints`` above.
            "final_waypoint_pos": list(self.final_waypoint_pos),
        }

    def validate_terrain_size(self, size: tuple[float, float]) -> None:
        """Validate base-ground patches against the generated terrain tile."""

        size_x, size_y = _float_pair(size, field_name="terrain size")
        if size_x <= 0.0 or size_y <= 0.0:
            raise ValueError("terrain size must be positive.")

        ground_x_range = (-0.5 * size_x, 0.5 * size_x)
        ground_y_range = (-0.5 * size_y, 0.5 * size_y)
        base_regions = [region for region in self.support_regions if region.structure_name is None]
        if not base_regions:
            raise ValueError(f"{self.name}: at least one base-ground support region is required.")

        for region in base_regions:
            try:
                x_range, y_range, surface_z = _horizontal_rectangle_geometry(region)
            except ValueError as error:
                raise ValueError(
                    f"{self.name}: base-ground support region {region.name!r} "
                    "must be a horizontal axis-aligned rectangle."
                ) from error
            if not (
                _pair_within(x_range, ground_x_range)
                and _pair_within(y_range, ground_y_range)
                and math.isclose(surface_z, 0.0, abs_tol=_GEOMETRY_TOLERANCE)
            ):
                raise ValueError(
                    f"{self.name}: base-ground support region {region.name!r} "
                    "must lie inside the generated terrain tile at z=0."
                )

        # Positive-area overlap would create duplicate coplanar collision
        # meshes and ambiguous internal edges. Touching boundaries are allowed
        # because adjacent patches can meet without overlapping volume.
        for index, region in enumerate(base_regions):
            for other in base_regions[index + 1 :]:
                if _rectangles_overlap(region, other):
                    raise ValueError(
                        f"{self.name}: base-ground support regions {region.name!r} and {other.name!r} overlap."
                    )

    def _validate_final_waypoint(self) -> None:
        """Require the final waypoint to identify its intended support."""

        if any(waypoint.is_terminal_landing for waypoint in self.waypoints[:-1]):
            raise ValueError(f"{self.name}: only the final waypoint may be marked as a terminal landing.")
        if self.waypoints[-1].is_rewarded_milestone:
            raise ValueError(
                f"{self.name}: the final waypoint uses the course-completion "
                "reward and cannot also be an intermediate rewarded milestone."
            )
        if self.waypoints[-1].support_region_name is None:
            raise ValueError(f"{self.name}: final waypoint must identify its intended support region.")

    def _validate_rewarded_milestones(self) -> None:
        """Require every rewarded intermediate milestone to be physical."""

        for index, waypoint in enumerate(self.waypoints[:-1]):
            if waypoint.is_rewarded_milestone and waypoint.support_region_name is None:
                raise ValueError(
                    f"{self.name}: rewarded waypoint {index} must identify its " "intended support region."
                )

    def _validate_support_references(
        self,
        structure_by_name: dict[str, ParkourStructureCfg],
    ) -> None:
        """Require every elevated support to reference physical course geometry."""

        for region in self.support_regions:
            # ``None`` represents one generated base-ground patch, whose
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

    def _validate_waypoint_supports(self) -> None:
        """Validate each physical waypoint against its named support polygon."""

        support_by_name = {region.name: region for region in self.support_regions}
        for index, waypoint in enumerate(self.waypoints):
            support_name = waypoint.support_region_name
            if support_name is None:
                continue
            waypoint_label = "final waypoint" if index == len(self.waypoints) - 1 else f"waypoint {index}"
            support = support_by_name.get(support_name)
            if support is None:
                raise ValueError(f"{self.name}: {waypoint_label} refers to unknown support region {support_name!r}.")
            if not support.supports_waypoint(waypoint.position):
                raise ValueError(
                    f"{self.name}: {waypoint_label} must lie on its intended support region {support_name!r}."
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


@dataclass(frozen=True)
class ParkourFamilyCfg:
    """One obstacle family sampled independently across difficulty rows.

    Every entry in ``levels`` describes the same obstacle family. Its tuple
    position is the terrain-row difficulty index, so row changes make that
    family harder without silently changing the kind of obstacle.
    """

    # Stable public identifier used by terrain columns, fixed-evaluation CLI
    # selection, metadata, and runtime family lookup.
    name: str

    # Ordered easiest-to-hardest courses generated down this family's terrain
    # columns, one entry for every shared difficulty row.
    levels: tuple[ParkourLevelCfg, ...]

    def __post_init__(self) -> None:
        _validate_name(self.name, field_name="obstacle-family name")
        levels = tuple(self.levels)
        _validate_typed_sequence(
            levels,
            ParkourLevelCfg,
            owner=f"Obstacle family {self.name!r}",
            item_name="level",
            required=True,
        )

        names = [level.name for level in levels]
        if len(names) != len(set(names)):
            raise ValueError("Parkour curriculum level names must be unique.")

        for field_name, values in (
            ("difficulty", [level.difficulty.order for level in levels]),
            ("target speed", [level.target_speed for level in levels]),
            ("minimum clearance", [level.min_clearance for level in levels]),
        ):
            if any(current > following for current, following in zip(values, values[1:])):
                raise ValueError(f"Parkour curriculum {field_name} must be non-decreasing.")

        if any(level.obstacle_family != self.name for level in levels):
            raise ValueError(
                f"Obstacle family {self.name!r} may contain only levels whose obstacle_family has the same name."
            )
        object.__setattr__(self, "levels", levels)

    def metadata(self) -> dict[str, object]:
        """Return a JSON-compatible description of this obstacle family."""

        return {
            "name": self.name,
            "levels": [level.metadata() for level in self.levels],
        }


# Terrain-construction API.


def base_ground_structures(
    level: ParkourLevelCfg,
    *,
    mesh_factory: Callable[..., object],
    ground_thickness: float,
) -> tuple[ParkourStructureCfg, ...]:
    """Convert a level's base supports into physical box structures.

    This is the single bridge between the declarative support layout and
    terrain collision geometry. A level with two separated base supports
    therefore produces two boxes with empty space between them.
    """

    thickness = _float_value(ground_thickness, field_name="ground_thickness")
    if thickness <= 0.0:
        raise ValueError("ground_thickness must be positive.")

    base_regions = tuple(region for region in level.support_regions if region.structure_name is None)
    if not base_regions:
        raise ValueError(f"{level.name}: at least one base-ground support region is required.")

    structures: list[ParkourStructureCfg] = []
    for region in base_regions:
        try:
            x_range, y_range, surface_z = _horizontal_rectangle_geometry(region)
        except ValueError as error:
            raise ValueError(
                f"{level.name}: base-ground support region {region.name!r} "
                "must be a horizontal axis-aligned rectangle."
            ) from error
        if not math.isclose(surface_z, 0.0, abs_tol=_GEOMETRY_TOLERANCE):
            raise ValueError(f"{level.name}: base-ground support region {region.name!r} must lie at z=0.")
        structures.append(
            ParkourStructureCfg(
                name=f"base_ground_{region.name}",
                mesh_factory=mesh_factory,
                mesh_kwargs={
                    "extents": (
                        x_range[1] - x_range[0],
                        y_range[1] - y_range[0],
                        thickness,
                    )
                },
                position=(
                    0.5 * (x_range[0] + x_range[1]),
                    0.5 * (y_range[0] + y_range[1]),
                    -0.5 * thickness,
                ),
            )
        )
    return tuple(structures)


# Configuration-reconstruction API.


def coerce_difficulty_cfg(difficulty: object) -> ParkourDifficultyCfg:
    """Return a typed difficulty configuration from an existing object or Hydra mapping."""

    if isinstance(difficulty, ParkourDifficultyCfg):
        return difficulty
    if not isinstance(difficulty, Mapping):
        raise TypeError("difficulty must be a ParkourDifficultyCfg or mapping.")
    missing_fields = {"order", "parameters"}.difference(difficulty)
    if missing_fields:
        raise ValueError(f"Difficulty is missing fields: {', '.join(sorted(missing_fields))}.")
    return ParkourDifficultyCfg(
        order=cast(float, difficulty["order"]),
        parameters=cast(dict[str, float], difficulty["parameters"]),
    )


def coerce_family_cfg(
    family: ParkourFamilyCfg | Mapping[str, object],
) -> ParkourFamilyCfg:
    """Return a typed obstacle-family configuration from an existing object or Hydra mapping."""

    if isinstance(family, ParkourFamilyCfg):
        return family
    if not isinstance(family, Mapping):
        raise TypeError("Obstacle family must be a ParkourFamilyCfg or mapping.")
    missing_fields = {"name", "levels"}.difference(family)
    if missing_fields:
        raise ValueError(f"Obstacle family is missing fields: {', '.join(sorted(missing_fields))}.")
    return ParkourFamilyCfg(
        name=cast(str, family["name"]),
        levels=tuple(
            coerce_level_cfg(level)
            for level in _sequence_value(
                family["levels"],
                field_name="family levels",
            )
        ),
    )


def coerce_level_cfg(level: ParkourLevelCfg | Mapping[str, object]) -> ParkourLevelCfg:
    """Return a typed level configuration from an existing object or Hydra mapping.

    Hydra may serialize nested dataclasses as dictionaries while composing the
    environment config. This function only reconstructs those nested objects;
    each dataclass remains responsible for normalizing and validating its own
    primitive values.
    """

    if isinstance(level, ParkourLevelCfg):
        return level
    if not isinstance(level, Mapping):
        raise TypeError(f"Curriculum level must be ParkourLevelCfg or a mapping, got {type(level).__name__}.")

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
        raise ValueError(f"Curriculum level is missing fields: {', '.join(sorted(missing_fields))}.")

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
    """Return a typed structure configuration from an existing object or Hydra mapping."""

    if isinstance(structure, ParkourStructureCfg):
        return structure
    if not isinstance(structure, Mapping):
        raise TypeError(f"Parkour structure must be a configuration or mapping, got {type(structure).__name__}.")

    missing_fields = {
        "name",
        "mesh_factory",
        "mesh_kwargs",
    }.difference(structure)
    if missing_fields:
        raise ValueError(f"Parkour structure is missing fields: {', '.join(sorted(missing_fields))}.")

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


def coerce_support_region_cfg(
    region: ParkourSupportRegionCfg | Mapping[str, object],
) -> ParkourSupportRegionCfg:
    """Return a typed support-region configuration from an existing object or Hydra mapping."""

    if isinstance(region, ParkourSupportRegionCfg):
        return region
    if not isinstance(region, Mapping):
        raise TypeError("Support region must be a configuration or mapping.")
    missing_fields = {"name", "structure_name", "vertices"}.difference(region)
    if missing_fields:
        raise ValueError(f"Support region is missing fields: {', '.join(sorted(missing_fields))}.")
    return ParkourSupportRegionCfg(
        name=cast(str, region["name"]),
        structure_name=cast(str | None, region["structure_name"]),
        vertices=cast(tuple[tuple[float, float, float], ...], region["vertices"]),
    )


def coerce_waypoint_cfg(
    waypoint: ParkourWaypointCfg | Mapping[str, object],
) -> ParkourWaypointCfg:
    """Return a typed waypoint configuration from an existing object or Hydra mapping."""

    if isinstance(waypoint, ParkourWaypointCfg):
        return waypoint
    if not isinstance(waypoint, Mapping) or "position" not in waypoint:
        raise TypeError("Waypoint must be a mapping containing 'position'.")
    return ParkourWaypointCfg(
        position=cast(tuple[float, float, float], waypoint["position"]),
        support_region_name=cast(
            str | None,
            waypoint.get("support_region_name"),
        ),
        is_rewarded_milestone=cast(
            bool,
            waypoint.get("is_rewarded_milestone", False),
        ),
        is_terminal_landing=cast(
            bool,
            waypoint.get("is_terminal_landing", False),
        ),
        root_reach_radius=cast(
            float | None,
            waypoint.get("root_reach_radius"),
        ),
    )


# Polygon-geometry helpers.


def _horizontal_rectangle_geometry(
    region: ParkourSupportRegionCfg,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return the X/Y extents and surface height of a horizontal rectangular support."""

    # A rectangle must first be a four-vertex polygon. Its unit normal must be
    # world up ``(0, 0, 1)``: zero X and Y components mean the plane has no
    # tilt, while positive unit Z confirms the required upward CCW winding.
    # These conditions establish a horizontal quadrilateral; the coordinate
    # checks below additionally prove that its sides form an axis-aligned
    # rectangle rather than another four-sided shape.
    if len(region.vertices) != 4 or not (
        math.isclose(region.normal[0], 0.0, abs_tol=_GEOMETRY_TOLERANCE)
        and math.isclose(region.normal[1], 0.0, abs_tol=_GEOMETRY_TOLERANCE)
        and math.isclose(region.normal[2], 1.0, abs_tol=_GEOMETRY_TOLERANCE)
    ):
        raise ValueError("Support is not a horizontal rectangle.")

    # Previous support-region validation already guarantees a planar, strictly
    # convex polygon with unique CCW vertices. For a four-sided polygon, it is
    # therefore sufficient to require every boundary edge to follow exactly
    # one world axis. An X-aligned edge has no Y displacement, while a
    # Y-aligned edge has no X displacement.
    for start, end in region.boundary_segments_xyz():
        delta_x = end[0] - start[0]
        delta_y = end[1] - start[1]
        runs_along_x = math.isclose(
            delta_y,
            0.0,
            abs_tol=_GEOMETRY_TOLERANCE,
        )
        runs_along_y = math.isclose(
            delta_x,
            0.0,
            abs_tol=_GEOMETRY_TOLERANCE,
        )

        # Exactly one condition must hold. Both false describes a diagonal
        # edge; both true describes an effectively zero-length edge.
        if runs_along_x == runs_along_y:
            raise ValueError("Support is not axis-aligned.")

    return region.x_range, region.y_range, region.vertices[0][2]


def _unit_polygon_normal(
    vertices: tuple[tuple[float, float, float], ...],
) -> tuple[float, float, float]:
    """Return the polygon's oriented unit normal from its first three vertices.

    The first two consecutive boundary vectors are ``v1 - v0`` and
    ``v2 - v1``. Their cross product is perpendicular to both vectors and
    therefore to the polygon plane. Its sign follows the right-hand rule, so
    counter-clockwise vertices viewed from above produce an upward-facing
    normal while reversed vertex order produces the opposite normal.

    Dividing the cross product by its Euclidean magnitude removes the area
    scale and leaves a vector of length one. A near-zero magnitude means that
    the first three vertices are collinear or repeated and cannot establish a
    plane, in which case the function raises ``ValueError``. The caller
    separately verifies that all remaining vertices lie in the same plane.
    """

    first_edge = _subtract_xyz(vertices[1], vertices[0])
    second_edge = _subtract_xyz(vertices[2], vertices[1])
    normal = _cross_xyz(first_edge, second_edge)
    magnitude = math.sqrt(_dot_xyz(normal, normal))
    if magnitude <= _GEOMETRY_TOLERANCE:
        raise ValueError("Support-region vertices must define a nondegenerate plane.")
    return (
        normal[0] / magnitude,
        normal[1] / magnitude,
        normal[2] / magnitude,
    )


def _validate_support_polygon(
    name: str,
    vertices: tuple[tuple[float, float, float], ...],
) -> None:
    """Require one finite, planar, strictly convex, upward-wound polygon."""

    if len(vertices) < 3:
        raise ValueError(f"{name}: a support region requires at least three vertices.")

    for index, vertex in enumerate(vertices):
        for other in vertices[index + 1 :]:
            separation = _subtract_xyz(vertex, other)
            if _dot_xyz(separation, separation) <= _GEOMETRY_TOLERANCE**2:
                raise ValueError(f"{name}: support-region vertices must be unique.")

    normal = _unit_polygon_normal(vertices)
    if normal[2] <= _GEOMETRY_TOLERANCE:
        raise ValueError(f"{name}: support-region vertices must have counter-clockwise, upward-facing winding.")

    origin = vertices[0]

    # Subtracting ``origin`` expresses each remaining vertex relative to a
    # known point on the plane. Projecting that displacement onto the unit
    # normal gives its signed perpendicular distance from the plane. Every
    # coplanar vertex must have distance zero within the geometry tolerance.
    if any(abs(_dot_xyz(_subtract_xyz(vertex, origin), normal)) > _GEOMETRY_TOLERANCE for vertex in vertices[1:]):
        raise ValueError(f"{name}: support-region vertices must be coplanar.")

    # A strictly convex CCW polygon places every non-edge vertex strictly to
    # the left of each directed boundary edge. Unlike checking only adjacent
    # turns, this also rejects self-intersecting star-shaped vertex orders.
    for edge_index, start in enumerate(vertices):
        end_index = (edge_index + 1) % len(vertices)
        end = vertices[end_index]
        edge = _subtract_xyz(end, start)
        for vertex_index, vertex in enumerate(vertices):
            if vertex_index in (edge_index, end_index):
                continue

            # Crossing the directed edge with the vector from its start to
            # this vertex produces a vector perpendicular to the polygon. Its
            # dot product with the polygon normal is positive when the vertex
            # lies to the left of the edge, negative when it lies to the
            # right, and zero when it lies on the edge's line. Every non-edge
            # vertex of a strictly convex CCW polygon must be strictly left of
            # every directed boundary edge.
            signed_side = _dot_xyz(
                _cross_xyz(edge, _subtract_xyz(vertex, start)),
                normal,
            )
            if signed_side <= _GEOMETRY_TOLERANCE:
                raise ValueError(f"{name}: support-region vertices must form a strictly convex CCW polygon.")


# XYZ-vector helpers.


def _cross_xyz(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the right-handed cross product of two XYZ vectors."""

    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _dot_xyz(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    """Return the dot product of two XYZ vectors."""

    return sum(first[axis] * second[axis] for axis in range(3))


def _subtract_xyz(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return the difference of two XYZ vectors without a numerical dependency."""

    return (
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    )


# Range-geometry helpers.


def _pair_within(
    inner: tuple[float, float],
    outer: tuple[float, float],
) -> bool:
    """Return whether one closed coordinate range lies inside another."""

    tolerance = 1.0e-9
    return inner[0] >= outer[0] - tolerance and inner[1] <= outer[1] + tolerance


def _rectangles_overlap(
    first: ParkourSupportRegionCfg,
    second: ParkourSupportRegionCfg,
) -> bool:
    """Return whether two support rectangles overlap with positive area."""

    tolerance = 1.0e-9
    overlap_x = min(first.x_range[1], second.x_range[1]) - max(first.x_range[0], second.x_range[0])
    overlap_y = min(first.y_range[1], second.y_range[1]) - max(first.y_range[0], second.y_range[0])
    return overlap_x > tolerance and overlap_y > tolerance


# Configuration-value helpers.


def _float_mapping(value: object, *, field_name: str) -> dict[str, float]:
    """Convert a string-keyed Python or Hydra mapping into finite float values."""

    if not isinstance(value, Mapping) or not all(isinstance(name, str) and name.strip() for name in value):
        raise TypeError(f"{field_name} must be a mapping with non-empty string keys.")
    return {name: _float_value(component, field_name=f"{field_name}.{name}") for name, component in value.items()}


def _float_pair(value: object, *, field_name: str) -> tuple[float, float]:
    """Convert a Python or Hydra sequence into a fixed two-float tuple."""

    values = tuple(
        _float_value(component, field_name=field_name) for component in _sequence_value(value, field_name=field_name)
    )
    if len(values) != 2:
        raise ValueError(f"{field_name} must have length 2, got {len(values)}.")
    return values[0], values[1]


def _float_triplet(value: object, *, field_name: str) -> tuple[float, float, float]:
    """Convert a Python or Hydra sequence into a fixed three-float tuple."""

    values = tuple(
        _float_value(component, field_name=field_name) for component in _sequence_value(value, field_name=field_name)
    )
    if len(values) != 3:
        raise ValueError(f"{field_name} must have length 3, got {len(values)}.")

    return values[0], values[1], values[2]


def _float_value(value: object, *, field_name: str) -> float:
    """Convert a numeric Python or Hydra value to float with a field-specific error."""

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


def _json_mapping(value: object, *, field_name: str) -> dict[str, object]:
    """Return a string-keyed mapping containing only Hydra/JSON-compatible values."""

    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} must be a mapping with string keys.")
    return {key: _json_value(item, field_name=f"{field_name}.{key}") for key, item in value.items()}


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
        return [_json_value(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)]

    # NumPy arrays/scalars and similar numeric containers commonly appear in
    # mesh arguments. Convert them through their public Python-data methods.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_value(tolist(), field_name=field_name)
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item(), field_name=field_name)
    raise TypeError(f"{field_name} is not Hydra/JSON-compatible.")


def _resolve_mesh_factory(value: object) -> Callable[..., object]:
    """Resolve a callable or Hydra's serialized callable representation."""

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


def _sequence_value(value: object, *, field_name: str) -> tuple[object, ...]:
    """Return a non-string Python or Hydra iterable as a tuple."""

    if isinstance(value, (str, bytes, Mapping)):
        raise TypeError(f"{field_name} must be a sequence.")
    try:
        return tuple(cast(Iterable[object], value))
    except TypeError as error:
        raise TypeError(f"{field_name} must be a sequence.") from error


def _validate_name(value: str, *, field_name: str) -> None:
    """Require a non-empty metadata string."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty.")


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
        raise TypeError(f"{owner}: {item_name}s must contain {expected_type.__name__} values.")
