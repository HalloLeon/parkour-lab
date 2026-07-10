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
