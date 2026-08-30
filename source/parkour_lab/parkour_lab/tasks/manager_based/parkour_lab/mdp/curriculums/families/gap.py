"""Physical-gap curriculum levels."""

from __future__ import annotations

import math

from ..levels import (
    ParkourDifficultyCfg,
    ParkourLevelCfg,
    ParkourSupportRegionCfg,
    ParkourWaypointCfg,
)
from . import _shared

# Terrain-local X coordinate around which each physical gap expands
# symmetrically as its curriculum-controlled width changes.
_GAP_CENTER_X_M = 2.0
_MIN_GAP_WIDTH_M = 0.10
_MAX_GAP_WIDTH_M = 0.50


def build_default_levels(
    geometry_variant_index: int = 0,
) -> tuple[ParkourLevelCfg, ...]:
    """Build the flat bootstrap and one deterministic physical-gap ladder."""

    variant_offset = _shared.geometry_variant_offset(geometry_variant_index)

    return (_shared.build_bootstrap_level("gap", geometry_variant_index),) + tuple(
        build_level(
            name=(
                f"gap_difficulty_{obstacle_stage_index}"
                if geometry_variant_index == 0
                else f"gap_variant_{geometry_variant_index}_difficulty_{obstacle_stage_index}"
            ),
            difficulty_order=_shared.normalized_level_difficulty(
                obstacle_stage_index + 1
            )
            * _shared.NUM_OBSTACLE_STAGES,
            gap_center_x=_GAP_CENTER_X_M + 0.05 * variant_offset,
            gap_width=round(
                _shared.lerp(
                    _MIN_GAP_WIDTH_M,
                    _MAX_GAP_WIDTH_M,
                    _shared.obstacle_progress(
                        _shared.normalized_level_difficulty(obstacle_stage_index + 1)
                    ),
                )
                * (1.0 + 0.05 * variant_offset),
                4,
            ),
            target_speed=0.60,
            min_clearance=_shared.DEFAULT_MIN_BASE_CLEARANCE_M,
        )
        for obstacle_stage_index in range(_shared.NUM_OBSTACLE_STAGES)
    )


def build_level(
    *,
    name: str,
    difficulty_order: float,
    gap_width: float,
    gap_center_x: float = _GAP_CENTER_X_M,
    target_speed: float,
    min_clearance: float,
) -> ParkourLevelCfg:
    """Create a course whose base supports leave a physical gap."""

    if not math.isfinite(gap_width) or gap_width <= 0.0:
        raise ValueError("Gap width must be positive and finite.")
    if not math.isfinite(gap_center_x):
        raise ValueError("Gap center must be finite.")
    gap_x_range = (gap_center_x - 0.5 * gap_width, gap_center_x + 0.5 * gap_width)

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
            parameters={
                "gap_center_x_m": gap_center_x,
                "gap_width_m": gap_width,
            },
        ),
    )
