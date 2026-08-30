"""Hurdle curriculum levels."""

from __future__ import annotations

from ..levels import (
    ParkourDifficultyCfg,
    ParkourLevelCfg,
    ParkourSupportRegionCfg,
    ParkourWaypointCfg,
)
from . import _shared

# Required ground distances around the hurdle's crossing waypoints.
APPROACH_MARGIN_M = 0.5
LANDING_MARGIN_M = 0.8
FINAL_EXIT_X_M = 3.8
_MIN_OBSTACLE_HEIGHT_M = 0.03
_MAX_OBSTACLE_HEIGHT_M = 0.18


def build_default_levels(
    geometry_variant_index: int = 0,
) -> tuple[ParkourLevelCfg, ...]:
    """Build the flat bootstrap and one deterministic hurdle ladder."""

    variant_offset = _shared.geometry_variant_offset(geometry_variant_index)

    return (_shared.build_bootstrap_level("hurdle", geometry_variant_index),) + tuple(
        build_level(
            name=(
                f"hurdle_difficulty_{obstacle_stage_index}"
                if geometry_variant_index == 0
                else f"hurdle_variant_{geometry_variant_index}_difficulty_{obstacle_stage_index}"
            ),
            difficulty_order=_shared.normalized_level_difficulty(
                obstacle_stage_index + 1
            )
            * _shared.NUM_OBSTACLE_STAGES,
            # Start with a barrier that the bootstrap gait can discover, then
            # increase its height uniformly through the final difficulty.
            obstacle_height=round(
                _shared.lerp(
                    _MIN_OBSTACLE_HEIGHT_M,
                    _MAX_OBSTACLE_HEIGHT_M,
                    _shared.obstacle_progress(
                        _shared.normalized_level_difficulty(obstacle_stage_index + 1)
                    ),
                )
                * (1.0 + 0.05 * variant_offset),
                4,
            ),
            # Span the full tile so a policy cannot walk around either end while
            # remaining inside its assigned course tile.
            obstacle_width=_shared.TERRAIN_Y_RANGE_M[1] - _shared.TERRAIN_Y_RANGE_M[0],
            obstacle_depth=0.18 * (1.0 + 0.05 * variant_offset),
            obstacle_position_xy=(2.0 - 0.05 * variant_offset, 0.0),
            target_speed=0.65,
            min_clearance=_shared.DEFAULT_MIN_BASE_CLEARANCE_M,
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
