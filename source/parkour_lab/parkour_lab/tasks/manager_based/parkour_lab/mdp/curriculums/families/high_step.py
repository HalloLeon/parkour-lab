"""High-step curriculum levels."""

from __future__ import annotations

from ..levels import ParkourDifficultyCfg, ParkourLevelCfg, ParkourSupportRegionCfg, ParkourWaypointCfg
from . import _shared


def build_default_levels() -> tuple[ParkourLevelCfg, ...]:
    """Build the flat bootstrap and all default high-step stages."""

    return (_shared.build_bootstrap_level("high_step"),) + tuple(
        build_level(
            name=f"high_step_difficulty_{obstacle_stage_index}",
            difficulty_order=float(obstacle_stage_index + 1),
            obstacle_height=round(0.08 + 0.04 * obstacle_stage_index, 2),
            obstacle_width=1.8,
            obstacle_depth=1.6,
            obstacle_position_xy=(2.8, 0.0),
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
