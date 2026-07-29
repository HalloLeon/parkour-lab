from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from ..terrain.ramps import TiltedRampGeometry
from .difficulty import difficulty_to_level
from .levels import (
    ParkourDifficultyCfg,
    ParkourFamilyCfg,
    ParkourLevelCfg,
    ParkourStructureCfg,
    ParkourSupportRegionCfg,
    ParkourWaypointCfg,
    base_ground_structures,
    coerce_family_cfg,
)

# Terrain-local bounds of each 8 m by 4 m tile. They define ground-support
# regions and constrain generated obstacle, ramp, and landing geometry.
_TERRAIN_X_RANGE_M = (-4.0, 4.0)
_TERRAIN_Y_RANGE_M = (-2.0, 2.0)

# Terrain-local X coordinate around which each physical gap expands
# symmetrically as its curriculum-controlled width changes.
_GAP_CENTER_X_M = 2.0

# Required ground distance from a hurdle's rear face to its landing waypoint.
_HURDLE_LANDING_MARGIN_M = 0.8

# Terrain-local X boundary at which the tilted-ramp approach support region
# ends; final landing regions must begin beyond it.
_RAMP_APPROACH_END_X_M = 0.55

# Longitudinal distance that ramp entry and exit waypoints are moved inside the
# top-centerline endpoints, keeping targets away from exact support edges.
_RAMP_WAYPOINT_INSET_M = 0.14


def _box_difficulty_parameters(
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


def _box_obstacle(
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
        _TERRAIN_X_RANGE_M[0] <= x_range[0] < x_range[1] <= _TERRAIN_X_RANGE_M[1]
        and _TERRAIN_Y_RANGE_M[0] <= y_range[0] < y_range[1] <= _TERRAIN_Y_RANGE_M[1]
    ):
        raise ValueError("Box-obstacle footprint must lie inside the terrain tile.")

    return (
        ParkourStructureCfg(
            name=name,
            mesh_factory=trimesh.creation.box,
            # Trimesh box extents are ordered X (depth), Y (width), Z (height).
            mesh_kwargs={"extents": dimensions},
            # Center the box vertically so its base is exactly on ground z=0.
            position=(center_x, center_y, 0.5 * obstacle_height),
        ),
        x_range,
        y_range,
    )


def _flat_level() -> ParkourLevelCfg:
    """Create the obstacle-free entry level of the default curriculum."""

    return ParkourLevelCfg(
        name="level_0_flat",
        obstacle_family="flat",
        waypoints=(ParkourWaypointCfg(position=(3.8, 0.0, 0.01)),),
        structures=(),
        support_regions=(
            ParkourSupportRegionCfg.horizontal_rectangle(
                name="ground",
                structure_name=None,
                x_range=_TERRAIN_X_RANGE_M,
                y_range=_TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
        ),
        target_speed=0.60,
        min_clearance=0.24,
        difficulty=ParkourDifficultyCfg(order=0.0, parameters={}),
    )


def _gap_level(
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
            ParkourWaypointCfg(position=(2.5, 0.0, 0.01)),
            ParkourWaypointCfg(position=(3.8, 0.0, 0.01)),
        ),
        structures=(),
        support_regions=(
            ParkourSupportRegionCfg.horizontal_rectangle(
                name="approach_ground",
                structure_name=None,
                x_range=(_TERRAIN_X_RANGE_M[0], gap_x_range[0]),
                y_range=_TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
            ParkourSupportRegionCfg.horizontal_rectangle(
                name="landing_ground",
                structure_name=None,
                x_range=(gap_x_range[1], _TERRAIN_X_RANGE_M[1]),
                y_range=_TERRAIN_Y_RANGE_M,
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


def _high_step_level(
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

    platform, platform_x_range, platform_y_range = _box_obstacle(
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
                )
            ),
        ),
        structures=(platform,),
        support_regions=(
            ParkourSupportRegionCfg.horizontal_rectangle(
                name="ground",
                structure_name=None,
                x_range=_TERRAIN_X_RANGE_M,
                y_range=_TERRAIN_Y_RANGE_M,
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
            parameters=_box_difficulty_parameters(
                obstacle_height=obstacle_height,
                obstacle_width=obstacle_width,
                obstacle_depth=obstacle_depth,
                obstacle_position_xy=obstacle_position_xy,
            ),
        ),
    )


def _hurdle_level(
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

    hurdle, hurdle_x_range, _ = _box_obstacle(
        name="hurdle",
        obstacle_height=obstacle_height,
        obstacle_width=obstacle_width,
        obstacle_depth=obstacle_depth,
        obstacle_position_xy=obstacle_position_xy,
    )
    landing_x = hurdle_x_range[1] + _HURDLE_LANDING_MARGIN_M
    landing_y = obstacle_position_xy[1]

    return ParkourLevelCfg(
        name=name,
        obstacle_family="hurdle",
        # The only target is on ground beyond the rear face. Unlike the high
        # step, the hurdle top is never declared as a traversable support.
        waypoints=(ParkourWaypointCfg(position=(landing_x, landing_y, 0.01)),),
        structures=(hurdle,),
        support_regions=(
            ParkourSupportRegionCfg.horizontal_rectangle(
                name="ground",
                structure_name=None,
                x_range=_TERRAIN_X_RANGE_M,
                y_range=_TERRAIN_Y_RANGE_M,
                surface_z=0.0,
            ),
        ),
        target_speed=target_speed,
        min_clearance=min_clearance,
        difficulty=ParkourDifficultyCfg(
            order=difficulty_order,
            parameters=_box_difficulty_parameters(
                obstacle_height=obstacle_height,
                obstacle_width=obstacle_width,
                obstacle_depth=obstacle_depth,
                obstacle_position_xy=obstacle_position_xy,
            ),
        ),
    )


def _ramp_structure(
    name: str,
    geometry: TiltedRampGeometry,
) -> ParkourStructureCfg:
    """Convert exact banked-ramp geometry into one collision box structure."""

    return ParkourStructureCfg(
        name=name,
        mesh_factory=trimesh.creation.box,
        mesh_kwargs={
            "extents": (
                geometry.length,
                geometry.width,
                geometry.thickness,
            )
        },
        # ``mesh_position`` compensates for rotating the slab thickness, so
        # ``center_xy`` continues to identify the center of the top surface.
        position=geometry.mesh_position,
        # Local +X is travel, roll banks the surface across its width, and yaw
        # sets the travel direction in terrain-local XY.
        orientation_rpy=geometry.orientation_rpy,
    )


def _tilted_ramp_difficulty_parameters(
    ramps: tuple[TiltedRampGeometry, ...],
    *,
    landing_x_range: tuple[float, float],
    landing_y_range: tuple[float, float],
) -> dict[str, float]:
    """Describe the exact ramp sequence and landing as flat numeric metadata.

    For every ramp after the first, ``gap_from_previous_m`` is the displacement
    from the previous ramp's top-centerline end to the current ramp's
    top-centerline start, projected onto the previous ramp's travel direction.
    It is therefore a longitudinal separation in the previous ramp's local
    frame, not the endpoints' Euclidean distance or the minimum clearance
    between the rotated collision meshes. The perpendicular projection in that
    same local frame is reported as ``lateral_offset_from_previous_m``.
    """

    parameters: dict[str, float] = {}
    for index, ramp in enumerate(ramps, start=1):
        prefix = f"ramp_{index}"
        parameters.update(
            {
                f"{prefix}_incline_deg": round(math.degrees(ramp.incline_radians), 12),
                f"{prefix}_yaw_deg": round(math.degrees(ramp.yaw_radians), 12),
                f"{prefix}_length_m": ramp.length,
                f"{prefix}_width_m": ramp.width,
                f"{prefix}_thickness_m": ramp.thickness,
                f"{prefix}_center_x_m": ramp.center_xy[0],
                f"{prefix}_center_y_m": ramp.center_xy[1],
            }
        )
        if index > 1:
            previous = ramps[index - 2]
            delta_x = ramp.centerline_start[0] - previous.centerline_end[0]
            delta_y = ramp.centerline_start[1] - previous.centerline_end[1]
            previous_travel_x, previous_travel_y = previous.travel_direction_xy
            previous_left_x, previous_left_y = previous.left_direction_xy

            # Decompose the endpoint displacement in the previous ramp's
            # orthonormal travel/left basis:
            # delta = gap * travel + lateral_offset * left.
            parameters[f"{prefix}_gap_from_previous_m"] = round(
                delta_x * previous_travel_x + delta_y * previous_travel_y,
                12,
            )
            parameters[f"{prefix}_lateral_offset_from_previous_m"] = round(
                delta_x * previous_left_x + delta_y * previous_left_y,
                12,
            )

    landing_x_min, landing_x_max = landing_x_range
    landing_y_min, landing_y_max = landing_y_range
    parameters.update(
        {
            "landing_length_m": landing_x_max - landing_x_min,
            "landing_width_m": landing_y_max - landing_y_min,
            "landing_center_x_m": 0.5 * (landing_x_min + landing_x_max),
            "landing_center_y_m": 0.5 * (landing_y_min + landing_y_max),
        }
    )
    return parameters


def _tilted_ramp_level(
    *,
    name: str,
    difficulty_order: float,
    ramps: tuple[TiltedRampGeometry, ...],
    landing_x_range: tuple[float, float],
    landing_y_range: tuple[float, float],
    target_speed: float,
    min_clearance: float,
    difficulty_parameters: dict[str, float],
) -> ParkourLevelCfg:
    """Create a laterally banked ramp sequence with an explicit redirected route."""

    ramps = tuple(ramps)
    if not ramps:
        raise ValueError("Tilted-ramp courses require at least one ramp.")
    if not all(isinstance(ramp, TiltedRampGeometry) for ramp in ramps):
        raise TypeError("Every tilted-ramp course entry must be TiltedRampGeometry.")
    if any(ramp.length <= 2.0 * _RAMP_WAYPOINT_INSET_M for ramp in ramps):
        raise ValueError("Each ramp must be long enough for inset route waypoints.")

    # Collision geometry, rather than only its centerline, must stay inside the
    # generated tile after applying both the bank angle and yaw.
    for ramp in ramps:
        x_bounds, y_bounds = ramp.collision_bounds_xy
        if not (
            _TERRAIN_X_RANGE_M[0] <= x_bounds[0] < x_bounds[1] <= _TERRAIN_X_RANGE_M[1]
            and _TERRAIN_Y_RANGE_M[0] <= y_bounds[0] < y_bounds[1] <= _TERRAIN_Y_RANGE_M[1]
        ):
            raise ValueError("Tilted-ramp collision geometry must lie inside the terrain tile.")

    marker_offset_z = 0.01
    approach_region = ParkourSupportRegionCfg.horizontal_rectangle(
        name="approach_ground",
        structure_name=None,
        x_range=(_TERRAIN_X_RANGE_M[0], _RAMP_APPROACH_END_X_M),
        y_range=_TERRAIN_Y_RANGE_M,
        surface_z=0.0,
    )
    landing_region = ParkourSupportRegionCfg.horizontal_rectangle(
        name="final_landing",
        structure_name=None,
        x_range=landing_x_range,
        y_range=landing_y_range,
        surface_z=0.0,
    )
    if not (
        _RAMP_APPROACH_END_X_M < landing_region.x_range[0]
        and _TERRAIN_X_RANGE_M[0] <= landing_region.x_range[0]
        and landing_region.x_range[1] <= _TERRAIN_X_RANGE_M[1]
        and _TERRAIN_Y_RANGE_M[0] <= landing_region.y_range[0]
        and landing_region.y_range[1] <= _TERRAIN_Y_RANGE_M[1]
    ):
        raise ValueError("The final ramp landing must fit inside the terrain tile.")

    structures = tuple(_ramp_structure(f"tilted_ramp_{index}", ramp) for index, ramp in enumerate(ramps, start=1))
    ramp_supports = tuple(
        ParkourSupportRegionCfg(
            name=f"tilted_ramp_{index}_top",
            structure_name=structure.name,
            vertices=ramp.top_corners,
        )
        for index, (ramp, structure) in enumerate(
            zip(ramps, structures),
            start=1,
        )
    )

    first_ramp = ramps[0]
    first_direction_x, first_direction_y = first_ramp.travel_direction_xy
    first_start = first_ramp.centerline_start

    # The 0.25 m value is a fixed route-design offset, not a quantity derived
    # from the ramp dimensions. Moving opposite the first ramp's unit travel
    # direction places the initial target on flat approach ground, giving the
    # route time to align with the ramp yaw before the robot reaches its edge.
    # The support-region check below rejects layouts for which that assumption
    # does not hold.
    waypoints = [
        ParkourWaypointCfg(
            position=(
                first_start[0] - 0.25 * first_direction_x,
                first_start[1] - 0.25 * first_direction_y,
                marker_offset_z,
            )
        )
    ]
    if not approach_region.supports_waypoint(waypoints[0].position):
        raise ValueError("The first tilted-ramp waypoint must lie on approach ground.")

    # The first approach waypoint already aligns the route with ramp one. Every
    # later ramp receives an entry waypoint so a gap or direction change is
    # explicit, and every ramp receives an inset exit waypoint.
    for index, (ramp, support) in enumerate(zip(ramps, ramp_supports)):
        direction_x, direction_y = ramp.travel_direction_xy
        ramp_waypoints: list[ParkourWaypointCfg] = []
        if index > 0:
            start = ramp.centerline_start
            ramp_waypoints.append(
                ParkourWaypointCfg(
                    position=(
                        start[0] + _RAMP_WAYPOINT_INSET_M * direction_x,
                        start[1] + _RAMP_WAYPOINT_INSET_M * direction_y,
                        ramp.top_center_z + marker_offset_z,
                    )
                )
            )
        end = ramp.centerline_end
        ramp_waypoints.append(
            ParkourWaypointCfg(
                position=(
                    end[0] - _RAMP_WAYPOINT_INSET_M * direction_x,
                    end[1] - _RAMP_WAYPOINT_INSET_M * direction_y,
                    ramp.top_center_z + marker_offset_z,
                )
            )
        )
        if not all(support.supports_waypoint(waypoint.position) for waypoint in ramp_waypoints):
            raise ValueError(f"Ramp {index + 1} waypoints must lie on its top surface.")
        waypoints.extend(ramp_waypoints)

    final_waypoint = ParkourWaypointCfg(
        position=(
            0.5 * (landing_region.x_range[0] + landing_region.x_range[1]),
            0.5 * (landing_region.y_range[0] + landing_region.y_range[1]),
            marker_offset_z,
        )
    )
    if not landing_region.supports_waypoint(final_waypoint.position):
        raise ValueError("The final tilted-ramp waypoint must lie on its landing.")
    waypoints.append(final_waypoint)

    return ParkourLevelCfg(
        name=name,
        obstacle_family="tilted_ramps",
        waypoints=tuple(waypoints),
        structures=structures,
        # Annotate the exact top faces so the same terrain-independent edge
        # reward covers horizontal ground and every banked ramp surface.
        support_regions=(
            approach_region,
            *ramp_supports,
            landing_region,
        ),
        target_speed=target_speed,
        min_clearance=min_clearance,
        difficulty=ParkourDifficultyCfg(
            order=difficulty_order,
            parameters=difficulty_parameters,
        ),
    )


def _default_tilted_ramp_level(index: int) -> ParkourLevelCfg:
    """Create one default two-ramp course at the requested difficulty row."""

    first_length = round(1.0 + 0.025 * index, 3)
    second_length = round(1.1 + 0.05 * index, 2)
    first_incline_degrees = round(4.0 + 2.0 * index, 1)
    second_incline_degrees = -first_incline_degrees
    first_yaw_degrees = round(2.5 * index, 1)
    second_yaw_degrees = round(12.0 + 5.0 * index, 1)
    ramp_width = round(1.2 - 0.05 * index, 2)
    ramp_thickness = 0.06
    inter_ramp_gap = round(0.08 + 0.03 * index, 2)
    inter_ramp_lateral_offset = round(0.08 + 0.08 * index, 2)
    landing_length = round(1.10 - 0.05 * index, 2)
    landing_width = 2.0
    landing_center_y = 0.0

    first_ramp = TiltedRampGeometry(
        center_xy=(1.15, -0.45),
        length=first_length,
        width=ramp_width,
        thickness=ramp_thickness,
        incline_radians=math.radians(first_incline_degrees),
        yaw_radians=math.radians(first_yaw_degrees),
    )
    second_ramp = first_ramp.placed_after(
        length=second_length,
        width=ramp_width,
        thickness=ramp_thickness,
        incline_radians=math.radians(second_incline_degrees),
        yaw_radians=math.radians(second_yaw_degrees),
        gap=inter_ramp_gap,
        lateral_offset=inter_ramp_lateral_offset,
    )
    ramps = (first_ramp, second_ramp)
    landing_x_range = (
        _TERRAIN_X_RANGE_M[1] - landing_length,
        _TERRAIN_X_RANGE_M[1],
    )
    landing_y_range = (
        landing_center_y - 0.5 * landing_width,
        landing_center_y + 0.5 * landing_width,
    )

    return _tilted_ramp_level(
        name=f"tilted_ramps_difficulty_{index}",
        difficulty_order=float(index),
        ramps=ramps,
        landing_x_range=landing_x_range,
        landing_y_range=landing_y_range,
        target_speed=round(0.65 + 0.05 * index, 2),
        min_clearance=round(0.24 + 0.01 * index, 2),
        difficulty_parameters=_tilted_ramp_difficulty_parameters(
            ramps,
            landing_x_range=landing_x_range,
            landing_y_range=landing_y_range,
        ),
    )


_NUM_DIFFICULTIES = 5

_DEFAULT_GAP_LEVELS = tuple(
    _gap_level(
        name=f"gap_difficulty_{index}",
        difficulty_order=float(index),
        gap_width=round(0.10 + 0.10 * index, 2),
        target_speed=round(0.60 + 0.05 * index, 2),
        min_clearance=round(0.24 + 0.01 * index, 2),
    )
    for index in range(_NUM_DIFFICULTIES)
)

_DEFAULT_HIGH_STEP_LEVELS = tuple(
    _high_step_level(
        name=f"high_step_difficulty_{index}",
        difficulty_order=float(index),
        obstacle_height=round(0.08 + 0.04 * index, 2),
        obstacle_width=1.8,
        obstacle_depth=1.6,
        obstacle_position_xy=(2.8, 0.0),
        target_speed=round(0.60 + 0.04 * index, 2),
        min_clearance=round(0.24 + 0.01 * index, 2),
    )
    for index in range(_NUM_DIFFICULTIES)
)

_DEFAULT_HURDLE_LEVELS = tuple(
    _hurdle_level(
        name=f"hurdle_difficulty_{index}",
        difficulty_order=float(index),
        obstacle_height=round(0.06 + 0.03 * index, 2),
        obstacle_width=1.8,
        obstacle_depth=0.18,
        obstacle_position_xy=(2.0, 0.0),
        target_speed=round(0.60 + 0.05 * index, 2),
        min_clearance=round(0.24 + 0.01 * index, 2),
    )
    for index in range(_NUM_DIFFICULTIES)
)

_DEFAULT_TILTED_RAMP_LEVELS = tuple(_default_tilted_ramp_level(index) for index in range(_NUM_DIFFICULTIES))

_DEFAULT_PARKOUR_FAMILIES = (
    ParkourFamilyCfg(name="gap", levels=_DEFAULT_GAP_LEVELS),
    ParkourFamilyCfg(name="high_step", levels=_DEFAULT_HIGH_STEP_LEVELS),
    ParkourFamilyCfg(name="hurdle", levels=_DEFAULT_HURDLE_LEVELS),
    ParkourFamilyCfg(
        name="tilted_ramps",
        levels=_DEFAULT_TILTED_RAMP_LEVELS,
    ),
)


@configclass
class ParkourTerrainLayout:
    """Map Isaac Lab's physical terrain grid to the curriculum matrix.

    Isaac Lab stores generated tile origins in a grid whose shape is
    ``(num_rows, num_columns, 3)``. This project gives those physical axes the
    following semantic meaning:

    * Physical row ``r`` represents curriculum difficulty ``r``. There must be
      exactly one row for every shared difficulty in the curriculum.
    * Physical columns are sampling slots rather than unique obstacle
      families. Multiple columns can contain the same family so training can
      allocate an equal number of environments to every family.
      ``family_index_by_column[c]`` identifies which curriculum family is
      generated in physical column ``c``.

    Keeping both parts in one value object makes the complete grid contract
    explicit wherever terrain generation and runtime indexing are connected.
    For example, a five-difficulty, four-family training layout with 40 columns
    has five rows and a mapping containing ten copies of each family index.
    Fixed-family evaluation instead maps every column to the selected family
    while retaining the same row-to-difficulty relationship.

    The layout is a mutable configclass because it is stored in an event term's
    ``params`` mapping. Isaac Lab's Hydra bridge reconstructs that mapping by
    assigning fields back onto the existing object, even without CLI overrides.

    Attributes:
        num_difficulty_rows: Number of physical terrain rows, equal to the
            number of shared curriculum difficulties.
        family_index_by_column: One curriculum-family index for every physical
            terrain column, ordered by column index.
    """

    num_difficulty_rows: int = MISSING
    family_index_by_column: tuple[int, ...] = MISSING

    @property
    def num_columns(self) -> int:
        """Return the number of physical terrain columns described."""

        return len(self.family_index_by_column)

    def validate_grid(
        self,
        *,
        curriculum_difficulties: int,
        curriculum_families: int,
        terrain_columns: int,
        terrain_rows: int,
    ) -> None:
        """Require the semantic layout and generated terrain grid to agree."""

        if self.num_difficulty_rows != curriculum_difficulties:
            raise ValueError(
                "The terrain layout and curriculum must describe the same "
                "number of difficulty rows: "
                f"got {self.num_difficulty_rows} and {curriculum_difficulties}."
            )
        if terrain_rows != self.num_difficulty_rows:
            raise ValueError(
                "Parkour terrain rows and difficulty levels must match "
                f"one-to-one: got {terrain_rows} rows and "
                f"{self.num_difficulty_rows} difficulties."
            )
        if terrain_columns != self.num_columns:
            raise ValueError(
                "The terrain layout must contain one family index per physical "
                f"column: got {self.num_columns} indices and "
                f"{terrain_columns} columns."
            )
        if any(
            family_index < 0 or family_index >= curriculum_families
            for family_index in self.family_index_by_column
        ):
            raise ValueError(
                "The terrain layout contains an out-of-range family index."
            )


@configclass
class ParkourCurriculumCfg:
    """Balanced obstacle-family by difficulty curriculum matrix.

    Terrain rows are shared difficulty indices. Terrain columns are split
    equally among ``families`` so PPO receives the same number of samples from
    gaps, high steps, hurdles, and tilted ramps at every active difficulty.
    """

    families: tuple[ParkourFamilyCfg, ...] = _DEFAULT_PARKOUR_FAMILIES

    initial_level: int = 1
    # Balance the initial population over levels 0..initial_level. This gives
    # PPO easy examples while avoiding a synchronized single-level population.
    distribute_initial_levels: bool = True
    max_level: int = 4

    # A waypoint changes only after the root remains within this XY radius for
    # ``waypoint_reach_hold_s``.
    waypoint_reach_threshold: float = 0.20
    waypoint_reach_hold_s: float = 0.10

    # Progress thresholds. Promotion requires more than half of
    # the configured route length; demotion requires less than half of the
    # distance commanded during the completed episode.
    promotion_course_fraction: float = 0.50
    demotion_expected_distance_fraction: float = 0.50

    base_contact_threshold: float = 1.0

    # A contacted foot within this metric distance of a support boundary is
    # counted by the edge penalty.
    edge_width_threshold: float = 0.05
    foot_edge_contact_threshold: float = 1.0

    def __post_init__(self) -> None:
        self.validate_configuration()

    def course(self, family_index: int, difficulty_index: int) -> ParkourLevelCfg:
        """Return one cell of the family-by-difficulty matrix."""

        if not 0 <= family_index < len(self.families):
            raise IndexError("family_index is out of range.")
        if not 0 <= difficulty_index < self.num_difficulties:
            raise IndexError("difficulty_index is out of range.")
        return self.families[family_index].levels[difficulty_index]

    def course_index(self, family_index: int, difficulty_index: int) -> int:
        """Flatten one matrix cell for vectorized runtime table indexing."""

        self.course(family_index, difficulty_index)
        return family_index * self.num_difficulties + difficulty_index

    @property
    def courses(self) -> tuple[ParkourLevelCfg, ...]:
        """Return family-major matrix cells for runtime lookup tables."""

        return tuple(level for family in self.families for level in family.levels)

    def family_index(self, family_name: str) -> int:
        """Return the stable index of a configured obstacle family."""

        try:
            return self.family_names.index(family_name)
        except ValueError as error:
            raise ValueError(
                f"Unknown terrain family {family_name!r}; choose one of "
                f"{list(self.family_names)}."
            ) from error

    @property
    def family_names(self) -> tuple[str, ...]:
        """Return obstacle-family names in their stable terrain-column order."""

        return tuple(family.name for family in self.families)

    def metadata(self) -> dict[str, object]:
        """Return the complete JSON-compatible curriculum matrix."""

        return {
            "family_order": list(self.family_names),
            "num_difficulties": self.num_difficulties,
            "families": [family.metadata() for family in self.families],
        }

    @property
    def num_difficulties(self) -> int:
        """Return the shared number of terrain difficulty rows."""

        return len(self.families[0].levels)

    def terrain_layout(
        self,
        num_columns: int,
        *,
        family_name: str | None = None,
    ) -> ParkourTerrainLayout:
        """Describe how physical terrain rows and columns encode the matrix.

        Training divides columns equally into contiguous family blocks,
        matching Isaac Lab's equal-proportion sub-terrain layout. Supplying
        ``family_name`` instead maps every column to that family for fixed
        evaluation. Rows always retain their one-to-one correspondence with
        the shared curriculum difficulties.
        """

        if (
            isinstance(num_columns, bool)
            or not isinstance(num_columns, int)
            or num_columns <= 0
        ):
            raise ValueError("num_columns must be a positive integer.")

        if family_name is not None:
            family_index_by_column = (self.family_index(family_name),) * num_columns
        else:
            num_families = len(self.families)
            if num_columns % num_families != 0:
                raise ValueError(
                    "Balanced family sampling requires num_columns to be "
                    f"divisible by the {num_families} obstacle families."
                )
            columns_per_family = num_columns // num_families
            family_index_by_column = tuple(
                column // columns_per_family for column in range(num_columns)
            )

        return ParkourTerrainLayout(
            num_difficulty_rows=self.num_difficulties,
            family_index_by_column=family_index_by_column,
        )

    def validate_configuration(self) -> None:
        """Validate matrix shape, balance assumptions, and transition settings."""

        # Hydra can turn nested dataclasses into dictionaries. Reconstruct each
        # family once so all downstream consumers receive one typed matrix.
        self.families = tuple(coerce_family_cfg(family) for family in self.families)
        if not self.families:
            raise ValueError("Parkour curriculum families must not be empty.")

        family_names = self.family_names
        if len(family_names) != len(set(family_names)):
            raise ValueError("Parkour curriculum family names must be unique.")

        difficulty_counts = {len(family.levels) for family in self.families}
        if len(difficulty_counts) != 1:
            raise ValueError(
                "Every obstacle family must define the same difficulty rows."
            )

        difficulty_orders = tuple(
            level.difficulty.order for level in self.families[0].levels
        )
        if any(
            tuple(level.difficulty.order for level in family.levels)
            != difficulty_orders
            for family in self.families[1:]
        ):
            raise ValueError(
                "Every obstacle family must use the same difficulty ranks by row."
            )

        if self.initial_level < 0 or self.initial_level >= self.num_difficulties:
            raise ValueError("initial_level is out of range.")

        if (
            self.max_level < self.initial_level
            or self.max_level >= self.num_difficulties
        ):
            raise ValueError("max_level is out of range.")

        if (
            not np.isfinite(self.waypoint_reach_threshold)
            or self.waypoint_reach_threshold <= 0.0
        ):
            raise ValueError("waypoint_reach_threshold must be positive.")

        if (
            not np.isfinite(self.waypoint_reach_hold_s)
            or self.waypoint_reach_hold_s < 0.0
        ):
            raise ValueError("waypoint_reach_hold_s must be non-negative.")

        if not 0.0 < self.promotion_course_fraction <= 1.0:
            raise ValueError("promotion_course_fraction must be in (0, 1].")

        if not 0.0 < self.demotion_expected_distance_fraction <= 1.0:
            raise ValueError("demotion_expected_distance_fraction must be in (0, 1].")

        if self.base_contact_threshold < 0.0:
            raise ValueError("base_contact_threshold must be non-negative.")

        if (
            not np.isfinite(self.edge_width_threshold)
            or self.edge_width_threshold <= 0.0
        ):
            raise ValueError("edge_width_threshold must be positive.")

        if (
            not np.isfinite(self.foot_edge_contact_threshold)
            or self.foot_edge_contact_threshold < 0.0
        ):
            raise ValueError("foot_edge_contact_threshold must be non-negative.")


DEFAULT_PARKOUR_CURRICULUM = ParkourCurriculumCfg()


def parkour_terrain(
    difficulty: float, cfg: ParkourTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain tile from base-support patches and structures."""

    levels = cfg.levels
    level = levels[difficulty_to_level(difficulty, len(levels))]
    level.validate_terrain_size(cfg.size)
    terrain_center = _terrain_local_center(cfg)
    ground_structures = base_ground_structures(
        level,
        mesh_factory=trimesh.creation.box,
        ground_thickness=cfg.ground_thickness,
    )

    meshes: list[trimesh.Trimesh] = []
    for structure in (*ground_structures, *level.structures):
        meshes.extend(_structure_meshes(structure, terrain_center))

    return meshes, terrain_center.copy()


@configclass
class ParkourTerrainCfg(SubTerrainBaseCfg):
    """Terrain config for courses composed from reusable structures."""

    function = parkour_terrain

    levels: tuple[ParkourLevelCfg, ...] = DEFAULT_PARKOUR_CURRICULUM.families[0].levels

    ground_thickness: float = 0.05


PARKOUR_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    # Enable Isaac Lab's terrain-curriculum layout.
    #
    # With curriculum=True, terrain rows correspond to difficulty levels.
    # In our case, each row is one parkour curriculum level.
    curriculum=True,
    # Physical size of one terrain tile in meters: (x_size, y_size).
    size=(8.0, 4.0),
    # Extra terrain border around the whole generated terrain.
    border_width=5.0,
    # One terrain row per shared difficulty level.
    num_rows=DEFAULT_PARKOUR_CURRICULUM.num_difficulties,
    # Number of terrain columns per curriculum row.
    num_cols=40,
    # Horizontal resolution used by height-field/mesh terrain utilities.
    #
    # For this custom trimesh terrain, this is not the main geometric control;
    # the actual geometry comes from the configured mesh factories.
    #
    # Keep it reasonably small and standard.
    horizontal_scale=0.05,
    # Vertical resolution used by terrain utilities.
    #
    # Geometry is defined directly by each mesh factory. This value is still
    # required by TerrainGeneratorCfg.
    vertical_scale=0.005,
    # Slope threshold used by some terrain-generation utilities to correct or
    # simplify steep surfaces.
    #
    # Custom meshes may contain steep or vertical faces, so this is not their
    # primary geometry control. Keep it at a conservative default.
    slope_threshold=0.75,
    # Disable terrain cache.
    #
    # use_cache=False is useful while actively developing terrain code, because
    # changes take effect immediately.
    use_cache=False,
    # Isaac Lab assigns sub-terrain types to contiguous column blocks in this
    # stable dictionary order. Equal proportions and 40 divisible columns give
    # every family exactly ten columns at every difficulty row.
    sub_terrains={
        family.name: ParkourTerrainCfg(
            proportion=1.0 / len(DEFAULT_PARKOUR_CURRICULUM.families),
            levels=family.levels,
            ground_thickness=0.05,
        )
        for family in DEFAULT_PARKOUR_CURRICULUM.families
    },
)


def _normalize_mesh_result(result: object, factory: object) -> list[trimesh.Trimesh]:
    """Normalize common Trimesh factory outputs into independent meshes."""

    # Wrap a single mesh in a list so all callers can process one consistent
    # return type. Copy it because the caller subsequently transforms it.
    if isinstance(result, trimesh.Trimesh):
        return [result.copy()]

    # A scene may contain several positioned geometries. ``dump`` resolves its
    # scene graph and returns the geometries as transformed mesh objects.
    if isinstance(result, trimesh.Scene):
        result = result.dump(concatenate=False)
        if isinstance(result, trimesh.Trimesh):
            return [result.copy()]

    # Factories may also return a sequence or generator of meshes. Strings and
    # bytes are iterable too, but cannot represent a valid collection of meshes.
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        meshes = list(result)
        if all(isinstance(mesh, trimesh.Trimesh) for mesh in meshes):
            # Return independent copies so applying a structure transform does
            # not mutate meshes retained or reused by the factory.
            return [mesh.copy() for mesh in meshes]

    # Report the factory as well as the accepted return types to make malformed
    # custom structure factories straightforward to identify.
    raise TypeError(
        f"Mesh factory {factory!r} must return a Trimesh, Scene, or iterable of Trimesh objects."
    )


def _structure_meshes(
    structure: ParkourStructureCfg, terrain_center: np.ndarray
) -> list[trimesh.Trimesh]:
    """Create and rigidly transform all meshes produced by one structure."""

    # Call the configured factory with its declarative keyword arguments.
    # Normalize its possible mesh, scene, or iterable output into independent
    # meshes that are safe to transform.
    meshes = _normalize_mesh_result(
        structure.mesh_factory(**structure.mesh_kwargs),
        structure.mesh_factory,
    )

    # Structure positions are expressed relative to the terrain tile center.
    # Adding the center converts that local offset into tile mesh coordinates.
    translation = terrain_center + np.asarray(structure.position, dtype=np.float64)

    # Build one homogeneous transform containing the structure's XYZ
    # translation and roll-pitch-yaw rotation.
    transform = trimesh.transformations.compose_matrix(
        translate=translation,
        angles=structure.orientation_rpy,
    )

    # Apply the same rigid pose to every mesh returned for this structure.
    for mesh in meshes:
        mesh.apply_transform(transform)
    return meshes


def _terrain_local_center(cfg: ParkourTerrainCfg) -> np.ndarray:
    """Return the terrain tile center used as its environment origin."""

    size_x, size_y = cfg.size
    return np.array([0.5 * size_x, 0.5 * size_y, 0.0], dtype=np.float32)
