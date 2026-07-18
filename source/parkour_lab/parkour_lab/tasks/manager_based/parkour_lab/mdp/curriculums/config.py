from __future__ import annotations

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
    coerce_and_validate_levels,
    coerce_level_cfg,
)

_OBSTACLE_LENGTH_M = 0.5
_OBSTACLE_WIDTH_M = 1.8
_OBSTACLE_CENTER_X_M = 2.0
_OBSTACLE_X_RANGE_M = (
    _OBSTACLE_CENTER_X_M - 0.5 * _OBSTACLE_LENGTH_M,
    _OBSTACLE_CENTER_X_M + 0.5 * _OBSTACLE_LENGTH_M,
)
_OBSTACLE_Y_RANGE_M = (-0.5 * _OBSTACLE_WIDTH_M, 0.5 * _OBSTACLE_WIDTH_M)

# Name, obstacle family, difficulty rank, obstacle height, target speed, and
# minimum clearance for the shared default course layout below.
_DEFAULT_LEVEL_PARAMETERS = (
    ("level_0_flat_marker", "flat_marker", 0.0, 0.02, 0.60, 0.24),
    ("level_1_low_step", "step", 1.0, 0.05, 0.70, 0.25),
    ("level_2_medium_step", "step", 2.0, 0.08, 0.75, 0.26),
    ("level_3_higher_step", "step", 3.0, 0.12, 0.80, 0.27),
)


@configclass
class ParkourCurriculumCfg:
    """
    Curriculum definition for the simplified parkour task.

    Levels should go from easiest to hardest.
    """

    levels: tuple[ParkourLevelCfg, ...] = tuple(
        ParkourLevelCfg(
            name=name,
            obstacle_family=obstacle_family,
            waypoints=(
                ParkourWaypointCfg(position=(1.0, 0.0, 0.01)),
                ParkourWaypointCfg(
                    position=(
                        _OBSTACLE_CENTER_X_M,
                        0.0,
                        obstacle_height + 0.01,
                    )
                ),
                ParkourWaypointCfg(position=(3.8, 0.0, 0.01)),
            ),
            structures=(
                ParkourStructureCfg(
                    name="center_obstacle",
                    mesh_factory=trimesh.creation.box,
                    mesh_kwargs={
                        "extents": (
                            _OBSTACLE_LENGTH_M,
                            _OBSTACLE_WIDTH_M,
                            obstacle_height,
                        )
                    },
                    position=(
                        _OBSTACLE_CENTER_X_M,
                        0.0,
                        0.5 * obstacle_height,
                    ),
                ),
            ),
            # This annotation describes the traversable top of the box;
            # generic terrain code does not inspect the mesh's shape.
            support_regions=(
                ParkourSupportRegionCfg(
                    name="ground",
                    structure_name=None,
                    x_range=(-4.0, 4.0),
                    y_range=(-2.0, 2.0),
                    surface_z=0.0,
                ),
                ParkourSupportRegionCfg(
                    name="center_obstacle_top",
                    structure_name="center_obstacle",
                    x_range=_OBSTACLE_X_RANGE_M,
                    y_range=_OBSTACLE_Y_RANGE_M,
                    surface_z=obstacle_height,
                ),
            ),
            target_speed=target_speed,
            min_clearance=min_clearance,
            difficulty=ParkourDifficultyCfg(
                order=difficulty_order,
                parameters={"obstacle_height_m": obstacle_height},
            ),
        )
        for (
            name,
            obstacle_family,
            difficulty_order,
            obstacle_height,
            target_speed,
            min_clearance,
        ) in _DEFAULT_LEVEL_PARAMETERS
    )

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


DEFAULT_PARKOUR_CURRICULUM = ParkourCurriculumCfg()


def parkour_terrain(
    difficulty: float, cfg: ParkourTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain tile by composing configured mesh factories."""

    level = coerce_level_cfg(
        cfg.levels[difficulty_to_level(difficulty, len(cfg.levels))]
    )
    terrain_center = _terrain_local_center(cfg)
    ground = ParkourStructureCfg(
        name="ground",
        mesh_factory=trimesh.creation.box,
        mesh_kwargs={"extents": (*cfg.size, cfg.ground_thickness)},
        position=(0.0, 0.0, -0.5 * cfg.ground_thickness),
    )

    meshes: list[trimesh.Trimesh] = []
    for structure in (ground, *level.structures):
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
