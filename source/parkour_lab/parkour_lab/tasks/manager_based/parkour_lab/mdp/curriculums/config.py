from __future__ import annotations

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .difficulty import difficulty_to_level
from .levels import ParkourObstacleLevelCfg, coerce_level_cfg


@configclass
class ParkourCurriculumCfg:
    """
    Curriculum definition for the simplified parkour task.

    Levels should go from easiest to hardest.
    """

    levels: tuple[ParkourObstacleLevelCfg, ...] = (
        ParkourObstacleLevelCfg(
            name="level_0_flat_marker",
            obstacle_pos=(2.0, 0.0, 0.01),
            obstacle_size=(0.5, 1.8, 0.02),
            goal_pos=(4.0, 0.0, 0.01),
            target_speed=0.60,
            min_clearance=0.24,
        ),
        ParkourObstacleLevelCfg(
            name="level_1_low_step",
            obstacle_pos=(2.0, 0.0, 0.025),
            obstacle_size=(0.5, 1.8, 0.05),
            goal_pos=(4.0, 0.0, 0.01),
            target_speed=0.70,
            min_clearance=0.25,
        ),
        ParkourObstacleLevelCfg(
            name="level_2_medium_step",
            obstacle_pos=(2.0, 0.0, 0.04),
            obstacle_size=(0.5, 1.8, 0.08),
            goal_pos=(4.0, 0.0, 0.01),
            target_speed=0.75,
            min_clearance=0.26,
        ),
        ParkourObstacleLevelCfg(
            name="level_3_higher_step",
            obstacle_pos=(2.0, 0.0, 0.06),
            obstacle_size=(0.5, 1.8, 0.12),
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
    failures_to_demote: int = 2  # Hysteresis prevents oscillating after one poor episode

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

        obstacle_heights = [level.obstacle_size[2] for level in self.levels]
        target_speeds = [level.target_speed for level in self.levels]
        min_clearances = [level.min_clearance for level in self.levels]

        for field_name, values in (
            ("obstacle height", obstacle_heights),
            ("target speed", target_speeds),
            ("minimum clearance", min_clearances),
        ):
            if any(current > following for current, following in zip(values, values[1:])):
                raise ValueError(f"Parkour curriculum {field_name} must be non-decreasing.")

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


def parkour_box_terrain(difficulty: float, cfg: ParkourBoxTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    Generate one terrain tile from the authoritative discrete level table.

    TerrainGenerator adds small within-row difficulty jitter. Converting that
    value back to a discrete bin ensures every tile in row N has exactly the
    geometry and command metadata of logical level N.
    """

    level = coerce_level_cfg(cfg.levels[difficulty_to_level(difficulty, len(cfg.levels))])
    obstacle_size = level.obstacle_size
    obstacle_pos = level.obstacle_pos

    size_x, size_y = cfg.size

    terrain_local_center = np.array([0.5 * size_x, 0.5 * size_y, 0.0], dtype=np.float32)

    meshes: list[trimesh.Trimesh] = []

    # Create the ground as a rectangular box spanning the complete terrain
    # tile. ``extents`` contains the box's full dimensions along X, Y, and Z:
    #   [tile length, tile width, ground thickness].
    #
    # Trimesh creates a box around its center. Move that center to the tile's
    # XY center and to z = -ground_thickness / 2. This places the box entirely
    # below z = 0, with its top surface exactly at z = 0 for the robot to stand
    # on. Without the negative half-thickness offset, half of the ground would
    # protrude above the intended terrain height.
    ground_mesh = trimesh.creation.box(
        extents=(size_x, size_y, cfg.ground_thickness),
        transform=trimesh.transformations.translation_matrix(
            (terrain_local_center[0], terrain_local_center[1], -0.5 * cfg.ground_thickness)
        ),
    )
    meshes.append(ground_mesh)

    obstacle_mesh = trimesh.creation.box(
        extents=obstacle_size,
        transform=trimesh.transformations.translation_matrix(
            (terrain_local_center[0] + obstacle_pos[0], terrain_local_center[1] + obstacle_pos[1], obstacle_pos[2])
        ),
    )
    meshes.append(obstacle_mesh)

    origin = terrain_local_center.copy()

    return meshes, origin


@configclass
class ParkourBoxTerrainCfg(SubTerrainBaseCfg):
    """Terrain config for one parkour box/hurdle terrain."""

    function = parkour_box_terrain

    levels: tuple[ParkourObstacleLevelCfg, ...] = DEFAULT_PARKOUR_CURRICULUM.levels

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
    # the actual obstacle size comes from ParkourBoxTerrainCfg.
    #
    # Keep it reasonably small and standard.
    horizontal_scale=0.05,
    # Vertical resolution used by terrain utilities.
    #
    # Again, for this custom box mesh, the obstacle heights are defined directly
    # by obstacle_size. This value is still required by TerrainGeneratorCfg.
    vertical_scale=0.005,
    # Slope threshold used by some terrain-generation utilities to correct or
    # simplify steep surfaces.
    #
    # Our terrain has flat ground and vertical box sides, so this is not the
    # primary control of obstacle geometry. Keep it at a conservative default.
    slope_threshold=0.75,
    # Disable terrain cache.
    #
    # use_cache=False is useful while actively developing terrain code, because
    # changes take effect immediately.
    use_cache=False,
    # Dictionary of sub-terrain types.
    #
    # We only define one sub-terrain type, "parkour_box".
    # Since its proportion is 1.0, every terrain tile is generated by
    # ParkourBoxTerrainCfg.
    sub_terrains={
        "parkour_box": ParkourBoxTerrainCfg(
            proportion=1.0, levels=DEFAULT_PARKOUR_CURRICULUM.levels, ground_thickness=0.05
        )
    },
)
