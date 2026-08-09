"""Hurdle curriculum levels."""

from __future__ import annotations

from ..levels import ParkourDifficultyCfg, ParkourLevelCfg, ParkourSupportRegionCfg, ParkourWaypointCfg
from . import _shared

# Required ground distance from a hurdle's rear face to its landing waypoint.
LANDING_MARGIN_M = 0.8
APPROACH_MARGIN_M = 0.5
FINAL_EXIT_X_M = 3.8
_MIN_OBSTACLE_HEIGHT_M = 0.03
_MAX_OBSTACLE_HEIGHT_M = 0.18


def build_default_levels() -> tuple[ParkourLevelCfg, ...]:
    """Build the flat bootstrap and all default hurdle stages."""

    return (_shared.build_bootstrap_level("hurdle"),) + tuple(
        build_level(
            name=f"hurdle_difficulty_{obstacle_stage_index}",
            difficulty_order=float(obstacle_stage_index + 1),
            # Start with a barrier that the bootstrap gait can discover, then
            # increase its height uniformly through the final difficulty.
            obstacle_height=round(
                _MIN_OBSTACLE_HEIGHT_M
                + (_MAX_OBSTACLE_HEIGHT_M - _MIN_OBSTACLE_HEIGHT_M)
                * obstacle_stage_index
                / (_shared.NUM_OBSTACLE_STAGES - 1),
                2,
            ),
            # Span the full tile so a policy cannot walk around either end while
            # remaining inside its assigned course tile.
            obstacle_width=_shared.TERRAIN_Y_RANGE_M[1] - _shared.TERRAIN_Y_RANGE_M[0],
            obstacle_depth=0.18,
            obstacle_position_xy=(2.0, 0.0),
            target_speed=0.65,
            min_clearance=0.28,
        )
        for obstacle_stage_index in range(_shared.NUM_OBSTACLE_STAGES)
    )


def build_level(
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

    hurdle, hurdle_x_range, _ = _shared.build_box_obstacle(
        name="hurdle",
        obstacle_height=obstacle_height,
        obstacle_width=obstacle_width,
        obstacle_depth=obstacle_depth,
        obstacle_position_xy=obstacle_position_xy,
    )
    approach_x = hurdle_x_range[0] - APPROACH_MARGIN_M
    landing_x = hurdle_x_range[1] + LANDING_MARGIN_M
    landing_y = obstacle_position_xy[1]

    return ParkourLevelCfg(
        name=name,
        obstacle_family="hurdle",
        # The hurdle top is never declared as traversable support. The landing
        # event provides credit for clearing it; completion remains farther
        # down-course so contact with the front face is a genuine stalled run.
        waypoints=(
            ParkourWaypointCfg(position=(approach_x, landing_y, 0.01)),
            ParkourWaypointCfg(
                position=(landing_x, landing_y, 0.01),
                support_region_name="ground",
                is_rewarded_milestone=True,
            ),
            ParkourWaypointCfg(
                position=(FINAL_EXIT_X_M, landing_y, 0.01),
                support_region_name="ground",
            ),
        ),
        structures=(hurdle,),
        support_regions=(
            ParkourSupportRegionCfg.horizontal_rectangle(
                name="ground",
                structure_name=None,
                x_range=_shared.TERRAIN_X_RANGE_M,
                y_range=_shared.TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
        ),
        target_speed=target_speed,
        min_clearance=min_clearance,
        difficulty=ParkourDifficultyCfg(
            order=difficulty_order,
            parameters=_shared.box_difficulty_parameters(
                obstacle_height=obstacle_height,
                obstacle_width=obstacle_width,
                obstacle_depth=obstacle_depth,
                obstacle_position_xy=obstacle_position_xy,
            ),
        ),
    )
