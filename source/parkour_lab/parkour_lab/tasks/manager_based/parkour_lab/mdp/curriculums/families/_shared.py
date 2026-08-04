"""Private construction helpers shared by the default obstacle families."""

from __future__ import annotations

import math

import trimesh

from ..levels import (
    ParkourDifficultyCfg,
    ParkourLevelCfg,
    ParkourStructureCfg,
    ParkourSupportRegionCfg,
    ParkourWaypointCfg,
)

# Terrain-local bounds of each 8 m by 4 m tile. Every default family uses these
# bounds when defining support regions and validating obstacle geometry.
TERRAIN_X_RANGE_M = (-4.0, 4.0)
TERRAIN_Y_RANGE_M = (-2.0, 2.0)

# Every family has five obstacle-bearing rows after the shared flat bootstrap.
NUM_OBSTACLE_STAGES = 5

_SUPPORTED_OBSTACLE_FAMILIES = frozenset({"gap", "high_step", "hurdle", "tilted_ramps"})
_BOOTSTRAP_ROUTE = (
    (1.25, 0.0, 0.01),
    (2.50, 0.0, 0.01),
    (3.80, 0.0, 0.01),
)
_BOOTSTRAP_TARGET_SPEED = 0.55


def build_bootstrap_level(obstacle_family: str) -> ParkourLevelCfg:
    """Create the obstacle-free row-zero course for one eventual family.

    Every terrain column retains its assigned family while row zero shares one
    flat route and target speed. This keeps the acquisition task identical
    across cohorts before promotion introduces family-specific obstacles.
    """

    if obstacle_family not in _SUPPORTED_OBSTACLE_FAMILIES:
        raise ValueError(f"Unsupported flat-bootstrap family: {obstacle_family!r}.")

    return ParkourLevelCfg(
        name=f"{obstacle_family}_flat_entry",
        obstacle_family=obstacle_family,
        waypoints=tuple(
            ParkourWaypointCfg(
                position=position,
                support_region_name=("ground" if index == len(_BOOTSTRAP_ROUTE) - 1 else None),
            )
            for index, position in enumerate(_BOOTSTRAP_ROUTE)
        ),
        structures=(),
        support_regions=(
            ParkourSupportRegionCfg.horizontal_rectangle(
                name="ground",
                structure_name=None,
                x_range=TERRAIN_X_RANGE_M,
                y_range=TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
        ),
        target_speed=_BOOTSTRAP_TARGET_SPEED,
        min_clearance=0.24,
        difficulty=ParkourDifficultyCfg(order=0.0, parameters={}),
    )


def build_box_obstacle(
    *,
    name: str,
    obstacle_height: float,
    obstacle_width: float,
    obstacle_depth: float,
    obstacle_position_xy: tuple[float, float],
) -> tuple[
    ParkourStructureCfg,
    tuple[float, float],
    tuple[float, float],
]:
    """Create one ground-mounted box and return its exact XY footprint."""

    dimensions = (obstacle_depth, obstacle_width, obstacle_height)
    if any(not math.isfinite(value) or value <= 0.0 for value in dimensions):
        raise ValueError("Box-obstacle height, width, and depth must be positive.")
    if len(obstacle_position_xy) != 2 or any(not math.isfinite(value) for value in obstacle_position_xy):
        raise ValueError("Box-obstacle XY position must contain two finite values.")

    center_x, center_y = obstacle_position_xy
    x_range = (
        center_x - 0.5 * obstacle_depth,
        center_x + 0.5 * obstacle_depth,
    )
    y_range = (
        center_y - 0.5 * obstacle_width,
        center_y + 0.5 * obstacle_width,
    )
    if not (
        TERRAIN_X_RANGE_M[0] <= x_range[0] < x_range[1] <= TERRAIN_X_RANGE_M[1]
        and TERRAIN_Y_RANGE_M[0] <= y_range[0] < y_range[1] <= TERRAIN_Y_RANGE_M[1]
    ):
        raise ValueError("Box-obstacle footprint must lie inside the terrain tile.")

    return (
        ParkourStructureCfg(
            name=name,
            mesh_factory=trimesh.creation.box,
            mesh_kwargs={"extents": dimensions},
            # Center the box vertically so its base is exactly on ground z=0.
            position=(center_x, center_y, 0.5 * obstacle_height),
        ),
        x_range,
        y_range,
    )


def box_difficulty_parameters(
    *,
    obstacle_height: float,
    obstacle_width: float,
    obstacle_depth: float,
    obstacle_position_xy: tuple[float, float],
) -> dict[str, float]:
    """Record the box geometry varied by one curriculum level."""

    return {
        "obstacle_height_m": obstacle_height,
        "obstacle_width_m": obstacle_width,
        "obstacle_depth_m": obstacle_depth,
        "obstacle_position_x_m": obstacle_position_xy[0],
        "obstacle_position_y_m": obstacle_position_xy[1],
    }
