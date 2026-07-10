from __future__ import annotations
from dataclasses import dataclass

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
import numpy as np
import trimesh


@dataclass(frozen=True)
class ParkourObstacleLevelCfg:
    """
    One obstacle-curriculum level.

    This describes the task geometry and a few training targets for that level.
    """

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
            raise ValueError(
                f"{self.name}: obstacle_pos.z should be obstacle_size.z / 2 "
                "for a box resting on ground."
            )


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
            min_clearance=0.24
        ),
        ParkourObstacleLevelCfg(
            name="level_1_low_step",
            obstacle_pos=(2.0, 0.0, 0.025),
            obstacle_size=(0.5, 1.8, 0.05),
            goal_pos=(4.0, 0.0, 0.01),
            target_speed=0.70,
            min_clearance=0.25
        ),
        ParkourObstacleLevelCfg(
            name="level_2_medium_step",
            obstacle_pos=(2.0, 0.0, 0.04),
            obstacle_size=(0.5, 1.8, 0.08),
            goal_pos=(4.0, 0.0, 0.01),
            target_speed=0.75,
            min_clearance=0.26
        ),
        ParkourObstacleLevelCfg(
            name="level_3_higher_step",
            obstacle_pos=(2.0, 0.0, 0.06),
            obstacle_size=(0.5, 1.8, 0.12),
            goal_pos=(4.2, 0.0, 0.01),
            target_speed=0.80,
            min_clearance=0.27
        )
    )

    initial_level: int = 1
    distribute_initial_levels: bool = False
    max_level: int = 3

    # Adaptive curriculum.
    promote_on_success: bool = True
    demote_on_base_contact: bool = True

    success_threshold: float = 0.30
    successes_to_promote: int = 2  # Avoids promotion from one lucky success
    failures_to_demote: int = 1  # Quickly makes the task easier after trunk-contact failure

    base_contact_threshold: float = 1.0

    def __post_init__(self) -> None:
        if len(self.levels) == 0:
            raise ValueError("ParkourCurriculumCfg.levels must not be empty.")

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
DEFAULT_PARKOUR_LEVEL = DEFAULT_PARKOUR_CURRICULUM.levels[
    DEFAULT_PARKOUR_CURRICULUM.initial_level
]


def parkour_box_terrain(
    difficulty: float,
    cfg: ParkourBoxTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    Generate one terrain tile with one obstacle.

    difficulty in [0, 1]:
        0 -> easiest obstacle height
        1 -> hardest obstacle height
    """

    difficulty = float(np.clip(difficulty, 0.0, 1.0))

    obstacle_height = (
        cfg.min_obstacle_height
        + difficulty * (cfg.max_obstacle_height - cfg.min_obstacle_height)
    )

    obstacle_size = (
        cfg.obstacle_length,
        cfg.obstacle_width,
        obstacle_height
    )

    obstacle_pos = (
        cfg.obstacle_x,
        cfg.obstacle_y,
        0.5 * obstacle_height
    )

    size_x, size_y = cfg.size

    terrain_local_center = np.array(
        [0.5 * size_x, 0.5 * size_y, 0.0],
        dtype=np.float32
    )

    meshes: list[trimesh.Trimesh] = []

    ground_mesh = trimesh.creation.box(
        extents=(size_x, size_y, cfg.ground_thickness),
        transform=trimesh.transformations.translation_matrix(
            (
                terrain_local_center[0],
                terrain_local_center[1],
                -0.5 * cfg.ground_thickness
            )
        )
    )
    meshes.append(ground_mesh)

    obstacle_mesh = trimesh.creation.box(
        extents=obstacle_size,
        transform=trimesh.transformations.translation_matrix(
            (
                terrain_local_center[0] + obstacle_pos[0],
                terrain_local_center[1] + obstacle_pos[1],
                obstacle_pos[2]
            )
        )
    )
    meshes.append(obstacle_mesh)

    origin = terrain_local_center.copy()

    return meshes, origin


@configclass
class ParkourBoxTerrainCfg(SubTerrainBaseCfg):
    """Terrain config for one parkour box/hurdle terrain."""

    function = parkour_box_terrain

    min_obstacle_height: float = 0.02
    max_obstacle_height: float = 0.12

    obstacle_x: float = 2.0
    obstacle_y: float = 0.0
    obstacle_length: float = 0.5
    obstacle_width: float = 1.8

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
    num_rows=10,

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
            proportion=1.0,
            min_obstacle_height=0.02,
            max_obstacle_height=0.12,
            obstacle_x=2.0,
            obstacle_y=0.0,
            obstacle_length=0.5,
            obstacle_width=1.8,
            ground_thickness=0.05
        )
    }
)
