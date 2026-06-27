from dataclasses import dataclass

from isaaclab.utils import configclass


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
    max_level: int = 3

    def __post_init__(self) -> None:
        if len(self.levels) == 0:
            raise ValueError("ParkourCurriculumCfg.levels must not be empty.")

        if self.initial_level < 0 or self.initial_level >= len(self.levels):
            raise ValueError("initial_level is out of range.")

        if self.max_level < self.initial_level or self.max_level >= len(self.levels):
            raise ValueError("max_level is out of range.")


DEFAULT_PARKOUR_CURRICULUM = ParkourCurriculumCfg()
DEFAULT_PARKOUR_LEVEL = DEFAULT_PARKOUR_CURRICULUM.levels[
    DEFAULT_PARKOUR_CURRICULUM.initial_level
]
