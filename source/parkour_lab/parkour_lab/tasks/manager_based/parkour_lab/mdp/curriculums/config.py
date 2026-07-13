from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .difficulty import difficulty_to_level
from .levels import ParkourLevelCfg, ParkourStructureCfg, coerce_level_cfg


@configclass
class ParkourCurriculumCfg:
    """
    Curriculum definition for the simplified parkour task.

    Levels should go from easiest to hardest.
    """

    levels: tuple[ParkourLevelCfg, ...] = (
        ParkourLevelCfg(
            name="level_0_flat_marker",
            structures=(
                ParkourStructureCfg(
                    mesh_factory=trimesh.creation.box,
                    mesh_kwargs={"extents": (0.5, 1.8, 0.02)},
                    position=(2.0, 0.0, 0.01),
                ),
            ),
            goal_pos=(4.0, 0.0, 0.01),
            target_speed=0.60,
            min_clearance=0.24,
        ),
        ParkourLevelCfg(
            name="level_1_low_step",
            structures=(
                ParkourStructureCfg(
                    mesh_factory=trimesh.creation.box,
                    mesh_kwargs={"extents": (0.5, 1.8, 0.05)},
                    position=(2.0, 0.0, 0.025),
                ),
            ),
            goal_pos=(4.0, 0.0, 0.01),
            target_speed=0.70,
            min_clearance=0.25,
        ),
        ParkourLevelCfg(
            name="level_2_medium_step",
            structures=(
                ParkourStructureCfg(
                    mesh_factory=trimesh.creation.box,
                    mesh_kwargs={"extents": (0.5, 1.8, 0.08)},
                    position=(2.0, 0.0, 0.04),
                ),
            ),
            goal_pos=(4.0, 0.0, 0.01),
            target_speed=0.75,
            min_clearance=0.26,
        ),
        ParkourLevelCfg(
            name="level_3_higher_step",
            structures=(
                ParkourStructureCfg(
                    mesh_factory=trimesh.creation.box,
                    mesh_kwargs={"extents": (0.5, 1.8, 0.12)},
                    position=(2.0, 0.0, 0.06),
                ),
            ),
            goal_pos=(4.2, 0.0, 0.01),
            target_speed=0.80,
            min_clearance=0.27,
        ),
    )

    initial_level: int = 1
    # Balance the initial population over levels 0..initial_level. This gives
    # PPO easy examples while avoiding a synchronized single-level population.
    distribute_initial_levels: bool = True
    max_level: int = 3

    # Adaptive curriculum.
    promote_on_success: bool = True
    demote_on_failure: bool = True

    success_threshold: float = 0.30
    successes_to_promote: int = 2  # Avoids promotion from one lucky success
    failures_to_demote: int = (
        2  # Hysteresis prevents oscillating after one poor episode
    )

    base_contact_threshold: float = 1.0

    def __post_init__(self) -> None:
        self.validate_configuration()

    def validate_configuration(self) -> None:
        """Validate ordering, bounds, and curriculum transition settings."""

        if len(self.levels) == 0:
            raise ValueError("ParkourCurriculumCfg.levels must not be empty.")

        # Hydra can turn nested dataclasses into dictionaries. Convert them
        # back once so all downstream consumers receive one representation.
        self.levels = tuple(coerce_level_cfg(level) for level in self.levels)

        names = [level.name for level in self.levels]
        if len(names) != len(set(names)):
            raise ValueError("Parkour curriculum level names must be unique.")

        target_speeds = [level.target_speed for level in self.levels]
        min_clearances = [level.min_clearance for level in self.levels]

        for field_name, values in (
            ("target speed", target_speeds),
            ("minimum clearance", min_clearances),
        ):
            if any(
                current > following for current, following in zip(values, values[1:])
            ):
                raise ValueError(
                    f"Parkour curriculum {field_name} must be non-decreasing."
                )

        if self.initial_level < 0 or self.initial_level >= len(self.levels):
            raise ValueError("initial_level is out of range.")

        if self.max_level < self.initial_level or self.max_level >= len(self.levels):
            raise ValueError("max_level is out of range.")

        if self.success_threshold <= 0.0:
            raise ValueError("success_threshold must be positive.")

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

    if isinstance(result, trimesh.Trimesh):
        return [result.copy()]
    if isinstance(result, trimesh.Scene):
        result = result.dump(concatenate=False)
        if isinstance(result, trimesh.Trimesh):
            return [result.copy()]
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        meshes = list(result)
        if all(isinstance(mesh, trimesh.Trimesh) for mesh in meshes):
            return [mesh.copy() for mesh in meshes]
    raise TypeError(
        f"Mesh factory {factory!r} must return a Trimesh, Scene, or iterable of Trimesh objects."
    )


def _structure_meshes(
    structure: ParkourStructureCfg, terrain_center: np.ndarray
) -> list[trimesh.Trimesh]:
    """Create and rigidly transform all meshes produced by one structure."""

    meshes = _normalize_mesh_result(
        structure.mesh_factory(**dict(structure.mesh_kwargs)),
        structure.mesh_factory,
    )
    translation = terrain_center + np.asarray(structure.position, dtype=np.float64)
    transform = trimesh.transformations.compose_matrix(
        translate=translation,
        angles=structure.orientation_rpy,
    )

    for mesh in meshes:
        mesh.apply_transform(transform)
    return meshes


def _terrain_local_center(cfg: ParkourTerrainCfg) -> np.ndarray:
    """Return the terrain tile center used as its environment origin."""

    size_x, size_y = cfg.size
    return np.array([0.5 * size_x, 0.5 * size_y, 0.0], dtype=np.float32)
