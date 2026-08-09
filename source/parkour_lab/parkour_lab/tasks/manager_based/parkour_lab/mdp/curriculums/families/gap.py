"""Physical-gap curriculum levels."""

from __future__ import annotations

import math

from ..levels import ParkourDifficultyCfg, ParkourLevelCfg, ParkourSupportRegionCfg, ParkourWaypointCfg
from . import _shared

# Terrain-local X coordinate around which each physical gap expands
# symmetrically as its curriculum-controlled width changes.
_GAP_CENTER_X_M = 2.0
_MIN_GAP_WIDTH_M = 0.10
_MAX_GAP_WIDTH_M = 0.50


def build_default_levels() -> tuple[ParkourLevelCfg, ...]:
    """Build the flat bootstrap and all default physical-gap stages."""

    return (_shared.build_bootstrap_level("gap"),) + tuple(
        build_level(
            name=f"gap_difficulty_{obstacle_stage_index}",
            difficulty_order=float(obstacle_stage_index + 1),
            gap_width=round(
                _MIN_GAP_WIDTH_M
                + (_MAX_GAP_WIDTH_M - _MIN_GAP_WIDTH_M) * obstacle_stage_index / (_shared.NUM_OBSTACLE_STAGES - 1),
                2,
            ),
            target_speed=0.60,
            min_clearance=0.28,
        )
        for obstacle_stage_index in range(_shared.NUM_OBSTACLE_STAGES)
    )


def build_level(
    *,
    name: str,
    difficulty_order: float,
    gap_width: float,
    target_speed: float,
    min_clearance: float,
) -> ParkourLevelCfg:
    """Create a course whose base supports leave a physical gap."""

    if not math.isfinite(gap_width) or gap_width <= 0.0:
        raise ValueError("Gap width must be positive and finite.")
    gap_x_range = (
        _GAP_CENTER_X_M - 0.5 * gap_width,
        _GAP_CENTER_X_M + 0.5 * gap_width,
    )

    return ParkourLevelCfg(
        name=name,
        obstacle_family="gap",
        waypoints=(
            ParkourWaypointCfg(position=(1.5, 0.0, 0.01)),
            ParkourWaypointCfg(
                position=(2.5, 0.0, 0.01),
                support_region_name="landing_ground",
                is_rewarded_milestone=True,
            ),
            ParkourWaypointCfg(
                position=(3.8, 0.0, 0.01),
                support_region_name="landing_ground",
            ),
        ),
        structures=(),
        support_regions=(
            ParkourSupportRegionCfg.horizontal_rectangle(
                name="approach_ground",
                structure_name=None,
                x_range=(_shared.TERRAIN_X_RANGE_M[0], gap_x_range[0]),
                y_range=_shared.TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
            ParkourSupportRegionCfg.horizontal_rectangle(
                name="landing_ground",
                structure_name=None,
                x_range=(gap_x_range[1], _shared.TERRAIN_X_RANGE_M[1]),
                y_range=_shared.TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
        ),
        target_speed=target_speed,
        min_clearance=min_clearance,
        difficulty=ParkourDifficultyCfg(
            order=difficulty_order,
            parameters={"gap_width_m": gap_width},
        ),
    )
