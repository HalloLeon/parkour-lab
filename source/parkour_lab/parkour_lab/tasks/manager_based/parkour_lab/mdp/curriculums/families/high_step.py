"""High-step curriculum levels."""

from __future__ import annotations

from ..levels import (
    ParkourDifficultyCfg,
    ParkourLevelCfg,
    ParkourSupportRegionCfg,
    ParkourWaypointCfg,
)
from . import _shared

# Keep the final target safely inside the rear platform edge while extending
# the route beyond the first supported landing.
PLATFORM_EXIT_INSET_M = 0.15
_MIN_OBSTACLE_HEIGHT_M = 0.04
_MAX_OBSTACLE_HEIGHT_M = 0.24


def build_default_levels(
    geometry_variant_index: int = 0,
) -> tuple[ParkourLevelCfg, ...]:
    """Build the flat bootstrap and one deterministic high-step ladder."""

    variant_offset = _shared.geometry_variant_offset(geometry_variant_index)

    return (_shared.build_bootstrap_level("high_step"),) + tuple(
        build_level(
            name=(
                f"high_step_difficulty_{obstacle_stage_index}"
                if geometry_variant_index == 0
                else f"high_step_variant_{geometry_variant_index}_difficulty_{obstacle_stage_index}"
            ),
            difficulty_order=_shared.normalized_level_difficulty(obstacle_stage_index + 1)
            * _shared.NUM_OBSTACLE_STAGES,
            # Begin below ordinary swing clearance so accidental early
            # successes teach progressively higher foot placement.
            obstacle_height=round(
                _shared.lerp(
                    _MIN_OBSTACLE_HEIGHT_M,
                    _MAX_OBSTACLE_HEIGHT_M,
                    _shared.obstacle_progress(_shared.normalized_level_difficulty(obstacle_stage_index + 1)),
                )
                * (1.0 + 0.05 * variant_offset),
                4,
            ),
            obstacle_width=1.8,
            obstacle_depth=1.6 * (1.0 + 0.03 * variant_offset),
            obstacle_position_xy=(
                2.8 - 0.04 * variant_offset,
                0.03 * variant_offset,
            ),
            target_speed=0.55,
            min_clearance=0.24,
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
    """Create a climbable face followed by an elevated landing platform."""

    platform, platform_x_range, platform_y_range = _shared.build_box_obstacle(
        name="elevated_platform",
        obstacle_height=obstacle_height,
        obstacle_width=obstacle_width,
        obstacle_depth=obstacle_depth,
        obstacle_position_xy=obstacle_position_xy,
    )
    platform_center_x, platform_center_y = obstacle_position_xy
    approach_x = platform_x_range[0] - 0.5
    platform_exit_inset = min(PLATFORM_EXIT_INSET_M, 0.25 * obstacle_depth)
    platform_exit_x = platform_x_range[1] - platform_exit_inset

    return ParkourLevelCfg(
        name=name,
        obstacle_family="high_step",
        waypoints=(
            ParkourWaypointCfg(position=(approach_x, platform_center_y, 0.01)),
            ParkourWaypointCfg(
                position=(
                    platform_center_x,
                    platform_center_y,
                    obstacle_height + 0.01,
                ),
                support_region_name="platform_top",
                is_rewarded_milestone=True,
            ),
            # Completion lies later on the same support. This separates credit
            # for the difficult climb/landing from stable platform traversal.
            ParkourWaypointCfg(
                position=(
                    platform_exit_x,
                    platform_center_y,
                    obstacle_height + 0.01,
                ),
                support_region_name="platform_top",
            ),
        ),
        structures=(platform,),
        support_regions=(
            ParkourSupportRegionCfg.horizontal_rectangle(
                name="ground",
                structure_name=None,
                x_range=_shared.TERRAIN_X_RANGE_M,
                y_range=_shared.TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
            ParkourSupportRegionCfg.horizontal_rectangle(
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
            parameters=_shared.box_difficulty_parameters(
                obstacle_height=obstacle_height,
                obstacle_width=obstacle_width,
                obstacle_depth=obstacle_depth,
                obstacle_position_xy=obstacle_position_xy,
            ),
        ),
    )
