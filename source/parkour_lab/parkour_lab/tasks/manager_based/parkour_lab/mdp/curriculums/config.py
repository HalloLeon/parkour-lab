from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .difficulty import difficulty_to_level
from .levels import (
    ParkourDifficultyCfg,
    ParkourLevelCfg,
    ParkourStructureCfg,
    ParkourSupportRegionCfg,
    ParkourWaypointCfg,
    base_ground_structures,
    coerce_and_validate_levels,
    coerce_level_cfg,
)

_TERRAIN_X_RANGE_M = (-4.0, 4.0)
_TERRAIN_Y_RANGE_M = (-2.0, 2.0)
_GAP_CENTER_X_M = 2.0
_GAP_WIDTH_M = 0.4
_HURDLE_LANDING_MARGIN_M = 0.8
_GAP_X_RANGE_M = (
    _GAP_CENTER_X_M - 0.5 * _GAP_WIDTH_M,
    _GAP_CENTER_X_M + 0.5 * _GAP_WIDTH_M,
)


def _flat_level() -> ParkourLevelCfg:
    """Create the obstacle-free entry level of the default curriculum."""

    return ParkourLevelCfg(
        name="level_0_flat",
        obstacle_family="flat",
        waypoints=(ParkourWaypointCfg(position=(3.8, 0.0, 0.01)),),
        structures=(),
        support_regions=(
            ParkourSupportRegionCfg(
                name="ground",
                structure_name=None,
                x_range=_TERRAIN_X_RANGE_M,
                y_range=_TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
        ),
        target_speed=0.60,
        min_clearance=0.24,
        difficulty=ParkourDifficultyCfg(order=0.0, parameters={}),
    )


def _gap_level() -> ParkourLevelCfg:
    """Create the first course whose base supports leave a physical gap."""

    return ParkourLevelCfg(
        name="level_3_gap",
        obstacle_family="gap",
        waypoints=(
            ParkourWaypointCfg(position=(1.5, 0.0, 0.01)),
            ParkourWaypointCfg(position=(2.5, 0.0, 0.01)),
            ParkourWaypointCfg(position=(3.8, 0.0, 0.01)),
        ),
        structures=(),
        support_regions=(
            ParkourSupportRegionCfg(
                name="approach_ground",
                structure_name=None,
                x_range=(_TERRAIN_X_RANGE_M[0], _GAP_X_RANGE_M[0]),
                y_range=_TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
            ParkourSupportRegionCfg(
                name="landing_ground",
                structure_name=None,
                x_range=(_GAP_X_RANGE_M[1], _TERRAIN_X_RANGE_M[1]),
                y_range=_TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
        ),
        target_speed=0.80,
        min_clearance=0.27,
        difficulty=ParkourDifficultyCfg(
            order=3.0,
            parameters={"gap_width_m": _GAP_WIDTH_M},
        ),
    )


def _box_obstacle(
    *,
    name: str,
    obstacle_height: float,
    obstacle_width: float,
    obstacle_depth: float,
    obstacle_position_xy: tuple[float, float],
) -> tuple[
    ParkourStructureCfg,
    tuple[float, float],
    tuple[float, float],
]:
    """Create one ground-mounted box and return its exact XY footprint."""

    dimensions = (obstacle_depth, obstacle_width, obstacle_height)
    if any(not math.isfinite(value) or value <= 0.0 for value in dimensions):
        raise ValueError("Box-obstacle height, width, and depth must be positive.")
    if len(obstacle_position_xy) != 2 or any(
        not math.isfinite(value) for value in obstacle_position_xy
    ):
        raise ValueError("Box-obstacle XY position must contain two finite values.")

    center_x, center_y = obstacle_position_xy
    x_range = (
        center_x - 0.5 * obstacle_depth,
        center_x + 0.5 * obstacle_depth,
    )
    y_range = (
        center_y - 0.5 * obstacle_width,
        center_y + 0.5 * obstacle_width,
    )
    if not (
        _TERRAIN_X_RANGE_M[0] <= x_range[0] < x_range[1] <= _TERRAIN_X_RANGE_M[1]
        and _TERRAIN_Y_RANGE_M[0] <= y_range[0] < y_range[1] <= _TERRAIN_Y_RANGE_M[1]
    ):
        raise ValueError("Box-obstacle footprint must lie inside the terrain tile.")

    return (
        ParkourStructureCfg(
            name=name,
            mesh_factory=trimesh.creation.box,
            # Trimesh box extents are ordered X (depth), Y (width), Z (height).
            mesh_kwargs={"extents": dimensions},
            # Center the box vertically so its base is exactly on ground z=0.
            position=(center_x, center_y, 0.5 * obstacle_height),
        ),
        x_range,
        y_range,
    )


def _box_difficulty_parameters(
    *,
    obstacle_height: float,
    obstacle_width: float,
    obstacle_depth: float,
    obstacle_position_xy: tuple[float, float],
) -> dict[str, float]:
    """Record the box geometry varied by one curriculum level."""

    return {
        "obstacle_height_m": obstacle_height,
        "obstacle_width_m": obstacle_width,
        "obstacle_depth_m": obstacle_depth,
        "obstacle_position_x_m": obstacle_position_xy[0],
        "obstacle_position_y_m": obstacle_position_xy[1],
    }


def _high_step_level(
    *,
    name: str,
    difficulty_order: float,
    obstacle_height: float,
    obstacle_width: float,
    obstacle_depth: float,
    obstacle_position_xy: tuple[float, float],
    target_speed: float,
    min_clearance: float,
) -> ParkourLevelCfg:
    """Create a climbable face followed by an elevated landing platform."""

    platform, platform_x_range, platform_y_range = _box_obstacle(
        name="elevated_platform",
        obstacle_height=obstacle_height,
        obstacle_width=obstacle_width,
        obstacle_depth=obstacle_depth,
        obstacle_position_xy=obstacle_position_xy,
    )
    platform_center_x, platform_center_y = obstacle_position_xy
    approach_x = platform_x_range[0] - 0.5

    return ParkourLevelCfg(
        name=name,
        obstacle_family="high_step",
        waypoints=(
            ParkourWaypointCfg(position=(approach_x, platform_center_y, 0.01)),
            # The final target lies well inside the top footprint. Reaching it
            # requires climbing the front face and landing on the platform.
            ParkourWaypointCfg(
                position=(
                    platform_center_x,
                    platform_center_y,
                    obstacle_height + 0.01,
                )
            ),
        ),
        structures=(platform,),
        support_regions=(
            ParkourSupportRegionCfg(
                name="ground",
                structure_name=None,
                x_range=_TERRAIN_X_RANGE_M,
                y_range=_TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
            ParkourSupportRegionCfg(
                name="platform_top",
                structure_name=platform.name,
                x_range=platform_x_range,
                y_range=platform_y_range,
                surface_z=obstacle_height,
            ),
        ),
        target_speed=target_speed,
        min_clearance=min_clearance,
        difficulty=ParkourDifficultyCfg(
            order=difficulty_order,
            parameters=_box_difficulty_parameters(
                obstacle_height=obstacle_height,
                obstacle_width=obstacle_width,
                obstacle_depth=obstacle_depth,
                obstacle_position_xy=obstacle_position_xy,
            ),
        ),
    )


def _hurdle_level(
    *,
    name: str,
    difficulty_order: float,
    obstacle_height: float,
    obstacle_width: float,
    obstacle_depth: float,
    obstacle_position_xy: tuple[float, float],
    target_speed: float,
    min_clearance: float,
) -> ParkourLevelCfg:
    """Create a narrow obstacle cleared onto continuous ground beyond it."""

    hurdle, hurdle_x_range, _ = _box_obstacle(
        name="hurdle",
        obstacle_height=obstacle_height,
        obstacle_width=obstacle_width,
        obstacle_depth=obstacle_depth,
        obstacle_position_xy=obstacle_position_xy,
    )
    landing_x = hurdle_x_range[1] + _HURDLE_LANDING_MARGIN_M
    landing_y = obstacle_position_xy[1]

    return ParkourLevelCfg(
        name=name,
        obstacle_family="hurdle",
        # The only target is on ground beyond the rear face. Unlike the high
        # step, the hurdle top is never declared as a traversable support.
        waypoints=(ParkourWaypointCfg(position=(landing_x, landing_y, 0.01)),),
        structures=(hurdle,),
        support_regions=(
            ParkourSupportRegionCfg(
                name="ground",
                structure_name=None,
                x_range=_TERRAIN_X_RANGE_M,
                y_range=_TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
        ),
        target_speed=target_speed,
        min_clearance=min_clearance,
        difficulty=ParkourDifficultyCfg(
            order=difficulty_order,
            parameters=_box_difficulty_parameters(
                obstacle_height=obstacle_height,
                obstacle_width=obstacle_width,
                obstacle_depth=obstacle_depth,
                obstacle_position_xy=obstacle_position_xy,
            ),
        ),
    )


_DEFAULT_PARKOUR_LEVELS = (
    _flat_level(),
    _high_step_level(
        name="level_1_high_step",
        difficulty_order=1.0,
        obstacle_height=0.16,
        obstacle_width=1.8,
        obstacle_depth=1.6,
        obstacle_position_xy=(2.8, 0.0),
        target_speed=0.70,
        min_clearance=0.25,
    ),
    _hurdle_level(
        name="level_2_hurdle",
        difficulty_order=2.0,
        obstacle_height=0.12,
        obstacle_width=1.8,
        obstacle_depth=0.18,
        obstacle_position_xy=(2.0, 0.0),
        target_speed=0.75,
        min_clearance=0.26,
    ),
    _gap_level(),
)


@configclass
class ParkourCurriculumCfg:
    """
    Curriculum definition for the simplified parkour task.

    Levels should go from easiest to hardest.
    """

    levels: tuple[ParkourLevelCfg, ...] = _DEFAULT_PARKOUR_LEVELS

    initial_level: int = 1
    # Balance the initial population over levels 0..initial_level. This gives
    # PPO easy examples while avoiding a synchronized single-level population.
    distribute_initial_levels: bool = True
    max_level: int = 3

    # Adaptive curriculum.
    promote_on_success: bool = True
    demote_on_failure: bool = True

    # A waypoint changes only after the root remains within this XY radius for
    # ``waypoint_reach_hold_s``.
    waypoint_reach_threshold: float = 0.20
    waypoint_reach_hold_s: float = 0.10
    successes_to_promote: int = 2  # Avoids promotion from one lucky success
    failures_to_demote: int = (
        2  # Hysteresis prevents oscillating after one poor episode
    )

    base_contact_threshold: float = 1.0

    # A contacted foot within this metric distance of a support boundary is
    # counted by the edge penalty.
    edge_width_threshold: float = 0.03
    foot_edge_contact_threshold: float = 1.0

    def __post_init__(self) -> None:
        self.validate_configuration()

    def validate_configuration(self) -> None:
        """Validate ordering, bounds, and curriculum transition settings."""

        # Hydra can turn nested dataclasses into dictionaries. Convert them
        # back once and validate explicit easiest-to-hardest ordering so all
        # downstream consumers receive one representation.
        self.levels = coerce_and_validate_levels(self.levels)

        if self.initial_level < 0 or self.initial_level >= len(self.levels):
            raise ValueError("initial_level is out of range.")

        if self.max_level < self.initial_level or self.max_level >= len(self.levels):
            raise ValueError("max_level is out of range.")

        if (
            not np.isfinite(self.waypoint_reach_threshold)
            or self.waypoint_reach_threshold <= 0.0
        ):
            raise ValueError("waypoint_reach_threshold must be positive.")

        if (
            not np.isfinite(self.waypoint_reach_hold_s)
            or self.waypoint_reach_hold_s < 0.0
        ):
            raise ValueError("waypoint_reach_hold_s must be non-negative.")

        if self.successes_to_promote <= 0:
            raise ValueError("successes_to_promote must be positive.")

        if self.failures_to_demote <= 0:
            raise ValueError("failures_to_demote must be positive.")

        if self.base_contact_threshold < 0.0:
            raise ValueError("base_contact_threshold must be non-negative.")

        if (
            not np.isfinite(self.edge_width_threshold)
            or self.edge_width_threshold <= 0.0
        ):
            raise ValueError("edge_width_threshold must be positive.")

        if (
            not np.isfinite(self.foot_edge_contact_threshold)
            or self.foot_edge_contact_threshold < 0.0
        ):
            raise ValueError("foot_edge_contact_threshold must be non-negative.")


DEFAULT_PARKOUR_CURRICULUM = ParkourCurriculumCfg()


def parkour_terrain(
    difficulty: float, cfg: ParkourTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain tile from base-support patches and structures."""

    level = coerce_level_cfg(
        cfg.levels[difficulty_to_level(difficulty, len(cfg.levels))]
    )
    level.validate_terrain_size(cfg.size)
    terrain_center = _terrain_local_center(cfg)
    ground_structures = base_ground_structures(
        level,
        mesh_factory=trimesh.creation.box,
        ground_thickness=cfg.ground_thickness,
    )

    meshes: list[trimesh.Trimesh] = []
    for structure in (*ground_structures, *level.structures):
        meshes.extend(_structure_meshes(structure, terrain_center))

    return meshes, terrain_center.copy()


@configclass
class ParkourTerrainCfg(SubTerrainBaseCfg):
    """Terrain config for courses composed from reusable structures."""

    function = parkour_terrain

    levels: tuple[ParkourLevelCfg, ...] = DEFAULT_PARKOUR_CURRICULUM.levels

    ground_thickness: float = 0.05


PARKOUR_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    # Enable Isaac Lab's terrain-curriculum layout.
    #
    # With curriculum=True, terrain rows correspond to difficulty levels.
    # In our case, each row is one parkour curriculum level.
    curriculum=True,
    # Physical size of one terrain tile in meters: (x_size, y_size).
    size=(8.0, 4.0),
    # Extra terrain border around the whole generated terrain.
    border_width=5.0,
    # One terrain row per curriculum level.
    num_rows=len(DEFAULT_PARKOUR_CURRICULUM.levels),
    # Number of terrain columns per curriculum row.
    num_cols=40,
    # Horizontal resolution used by height-field/mesh terrain utilities.
    #
    # For this custom trimesh terrain, this is not the main geometric control;
    # the actual geometry comes from the configured mesh factories.
    #
    # Keep it reasonably small and standard.
    horizontal_scale=0.05,
    # Vertical resolution used by terrain utilities.
    #
    # Geometry is defined directly by each mesh factory. This value is still
    # required by TerrainGeneratorCfg.
    vertical_scale=0.005,
    # Slope threshold used by some terrain-generation utilities to correct or
    # simplify steep surfaces.
    #
    # Custom meshes may contain steep or vertical faces, so this is not their
    # primary geometry control. Keep it at a conservative default.
    slope_threshold=0.75,
    # Disable terrain cache.
    #
    # use_cache=False is useful while actively developing terrain code, because
    # changes take effect immediately.
    use_cache=False,
    # Dictionary of sub-terrain types.
    #
    # We only define one sub-terrain type, "parkour_course".
    # Since its proportion is 1.0, every terrain tile is generated by
    # ParkourTerrainCfg.
    sub_terrains={
        "parkour_course": ParkourTerrainCfg(
            proportion=1.0,
            levels=DEFAULT_PARKOUR_CURRICULUM.levels,
            ground_thickness=0.05,
        )
    },
)


def _normalize_mesh_result(result: object, factory: object) -> list[trimesh.Trimesh]:
    """Normalize common Trimesh factory outputs into independent meshes."""

    # Wrap a single mesh in a list so all callers can process one consistent
    # return type. Copy it because the caller subsequently transforms it.
    if isinstance(result, trimesh.Trimesh):
        return [result.copy()]

    # A scene may contain several positioned geometries. ``dump`` resolves its
    # scene graph and returns the geometries as transformed mesh objects.
    if isinstance(result, trimesh.Scene):
        result = result.dump(concatenate=False)
        if isinstance(result, trimesh.Trimesh):
            return [result.copy()]

    # Factories may also return a sequence or generator of meshes. Strings and
    # bytes are iterable too, but cannot represent a valid collection of meshes.
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        meshes = list(result)
        if all(isinstance(mesh, trimesh.Trimesh) for mesh in meshes):
            # Return independent copies so applying a structure transform does
            # not mutate meshes retained or reused by the factory.
            return [mesh.copy() for mesh in meshes]

    # Report the factory as well as the accepted return types to make malformed
    # custom structure factories straightforward to identify.
    raise TypeError(
        f"Mesh factory {factory!r} must return a Trimesh, Scene, or iterable of Trimesh objects."
    )


def _structure_meshes(
    structure: ParkourStructureCfg, terrain_center: np.ndarray
) -> list[trimesh.Trimesh]:
    """Create and rigidly transform all meshes produced by one structure."""

    # Call the configured factory with its declarative keyword arguments.
    # Normalize its possible mesh, scene, or iterable output into independent
    # meshes that are safe to transform.
    meshes = _normalize_mesh_result(
        structure.mesh_factory(**structure.mesh_kwargs),
        structure.mesh_factory,
    )

    # Structure positions are expressed relative to the terrain tile center.
    # Adding the center converts that local offset into tile mesh coordinates.
    translation = terrain_center + np.asarray(structure.position, dtype=np.float64)

    # Build one homogeneous transform containing the structure's XYZ
    # translation and roll-pitch-yaw rotation.
    transform = trimesh.transformations.compose_matrix(
        translate=translation,
        angles=structure.orientation_rpy,
    )

    # Apply the same rigid pose to every mesh returned for this structure.
    for mesh in meshes:
        mesh.apply_transform(transform)
    return meshes


def _terrain_local_center(cfg: ParkourTerrainCfg) -> np.ndarray:
    """Return the terrain tile center used as its environment origin."""

    size_x, size_y = cfg.size
    return np.array([0.5 * size_x, 0.5 * size_y, 0.0], dtype=np.float32)
