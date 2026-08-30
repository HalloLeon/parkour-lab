"""Private construction helpers shared by the default obstacle families."""

from __future__ import annotations

import math

from ..levels import (
    ParkourDifficultyCfg,
    ParkourLevelCfg,
    ParkourStructureCfg,
    ParkourSupportRegionCfg,
    ParkourWaypointCfg,
)
from ._meshes import box_mesh

# Terrain-local bounds of each 8 m by 4 m tile. Every default family uses these
# bounds when defining support regions and validating obstacle geometry.
TERRAIN_X_RANGE_M = (-4.0, 4.0)
TERRAIN_Y_RANGE_M = (-2.0, 2.0)

# Every family has six obstacle-bearing rows after the shared flat bootstrap.
NUM_OBSTACLE_STAGES = 6

# Shared Go2 base-clearance floor. This remains below the nominal home-pose
# clearance while reducing the slack that allowed a crouched solution.
DEFAULT_MIN_BASE_CLEARANCE_M = 0.27

# Variant zero is the nominal course used for fixed evaluation. Adjacent pairs
# share one of five zero-mean severity offsets so the tilted-ramp family can
# assign the original and its exact left-right reflection to equally many
# training columns. Other families retain the same five severity values and
# duplicate each one to preserve the shared rectangular curriculum matrix.
GEOMETRY_VARIANT_OFFSETS = (0.0, 0.0, -1.0, -1.0, -0.5, -0.5, 0.5, 0.5, 1.0, 1.0)

_BOOTSTRAP_ROUTE = (
    (1.25, 0.0, 0.01),
    (2.50, 0.0, 0.01),
    (3.80, 0.0, 0.01),
)
_ROTATED_BOOTSTRAP_ROUTE = (
    (1.25, 0.25, 0.01),
    (2.50, 0.50, 0.01),
    (3.80, 0.76, 0.01),
)
_TURNING_BOOTSTRAP_ROUTE = (
    (1.20, 0.22, 0.01),
    (2.40, 0.65, 0.01),
    (3.65, 1.20, 0.01),
)
_BOOTSTRAP_TARGET_SPEED = 0.55
_SUPPORTED_OBSTACLE_FAMILIES = frozenset({"gap", "high_step", "hurdle", "tilted_ramps"})


# Curriculum-scalar helpers.


def geometry_variant_handedness(variant_index: int) -> float:
    """Return the paired lateral handedness for one geometry variant.

    Every even variant retains the original geometry and the following odd
    variant reflects it. Resolving the offset first applies the shared index
    validation without maintaining a second definition of the valid range.
    """

    geometry_variant_offset(variant_index)
    return 1.0 if variant_index % 2 == 0 else -1.0


def geometry_variant_offset(variant_index: int) -> float:
    """Return the bounded severity offset for one deterministic variant."""

    if (
        isinstance(variant_index, bool)
        or not isinstance(variant_index, int)
        or not 0 <= variant_index < len(GEOMETRY_VARIANT_OFFSETS)
    ):
        raise ValueError("variant_index must select a configured geometry variant.")
    return GEOMETRY_VARIANT_OFFSETS[variant_index]


def lerp(start: float, end: float, fraction: float) -> float:
    """Linearly interpolate finite endpoints with a normalized fraction."""

    values = (float(start), float(end), float(fraction))
    if any(not math.isfinite(value) for value in values) or not 0.0 <= values[2] <= 1.0:
        raise ValueError(
            "Interpolation requires finite endpoints and a fraction in [0, 1]."
        )
    return values[0] + (values[1] - values[0]) * values[2]


def normalized_level_difficulty(level_index: int) -> float:
    """Map the flat-plus-obstacle row index onto the public ``[0, 1]`` scale."""

    if (
        isinstance(level_index, bool)
        or not isinstance(level_index, int)
        or not 0 <= level_index <= NUM_OBSTACLE_STAGES
    ):
        raise ValueError("level_index must select the flat row or an obstacle stage.")
    return level_index / NUM_OBSTACLE_STAGES


def obstacle_progress(normalized_difficulty: float) -> float:
    """Normalize obstacle rows 1..N to interpolation progress 0..1."""

    normalized_difficulty = float(normalized_difficulty)
    first_obstacle = 1.0 / NUM_OBSTACLE_STAGES
    if (
        not math.isfinite(normalized_difficulty)
        or not first_obstacle <= normalized_difficulty <= 1.0
    ):
        raise ValueError("normalized_difficulty must select an obstacle-bearing row.")
    return (normalized_difficulty - first_obstacle) / (1.0 - first_obstacle)


# Shared course builders.


def build_bootstrap_level(
    obstacle_family: str,
    geometry_variant_index: int = 0,
) -> ParkourLevelCfg:
    """Create the obstacle-free row-zero course for one eventual family.

    Variants zero through five retain the original straight route. Variants
    six/seven rotate that line left/right, and eight/nine add mirrored gentle
    turns, without changing any obstacle row.
    """

    if obstacle_family not in _SUPPORTED_OBSTACLE_FAMILIES:
        raise ValueError(f"Unsupported flat-bootstrap family: {obstacle_family!r}.")
    handedness = geometry_variant_handedness(geometry_variant_index)
    if geometry_variant_index < 6:
        route = _BOOTSTRAP_ROUTE
    elif geometry_variant_index < 8:
        route = _ROTATED_BOOTSTRAP_ROUTE
    else:
        route = _TURNING_BOOTSTRAP_ROUTE

    return ParkourLevelCfg(
        name=f"{obstacle_family}_flat_entry",
        obstacle_family=obstacle_family,
        waypoints=tuple(
            ParkourWaypointCfg(
                position=(position[0], handedness * position[1], position[2]),
                support_region_name="ground",
                # Split the existing intermediate-milestone budget across the
                # two supported acquisition checkpoints.  The final waypoint
                # retains the separate course-completion bonus.
                is_rewarded_milestone=index < len(route) - 1,
            )
            for index, position in enumerate(route)
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
        min_clearance=DEFAULT_MIN_BASE_CLEARANCE_M,
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
    if len(obstacle_position_xy) != 2 or any(
        not math.isfinite(value) for value in obstacle_position_xy
    ):
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
            mesh_factory=box_mesh,
            mesh_kwargs={"extents": list(dimensions)},
            # Center the box vertically so its base is exactly on ground z=0.
            position=(center_x, center_y, 0.5 * obstacle_height),
        ),
        x_range,
        y_range,
    )


# Course metadata helpers.


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
