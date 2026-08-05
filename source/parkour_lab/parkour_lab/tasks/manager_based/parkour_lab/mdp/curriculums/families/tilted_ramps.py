"""Tilted-ramp curriculum specifications and course construction.

This module owns the complete curriculum-facing tilted-ramp pipeline:

* declarative ramp-sequence and curriculum-stage specifications;
* conversion of those specifications into exact collision geometry;
* support-region, waypoint, and difficulty-metadata construction; and
* the default non-flat tilted-ramp acquisition ladder.

The terrain package retains only :class:`TiltedRampGeometry`, the reusable
geometric primitive that is independent of curriculum policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import trimesh

from ...terrain.ramps import TiltedRampGeometry
from ..levels import (
    ParkourDifficultyCfg,
    ParkourLevelCfg,
    ParkourStructureCfg,
    ParkourSupportRegionCfg,
    ParkourWaypointCfg,
)
from . import _shared

# Terrain-local X boundary at which the tilted-ramp approach support region
# ends; final landing regions must begin beyond it.
_APPROACH_END_X_M = 0.55

# Longitudinal distance that ramp entry and exit waypoints are moved inside the
# top-centerline endpoints, keeping targets away from exact support edges.
_WAYPOINT_INSET_M = 0.14


# Specification models


@dataclass(frozen=True)
class RampSpec:
    """Human-readable shape and orientation for one cross-slope ramp."""

    length: float
    width: float
    incline_degrees: float
    yaw_degrees: float
    thickness: float = 0.06


@dataclass(frozen=True)
class RampTransitionSpec:
    """Placement of one ramp relative to the preceding ramp."""

    gap: float
    lateral_offset: float


@dataclass(frozen=True)
class TiltedRampStageSpec:
    """Declarative geometry for one non-flat tilted-ramp curriculum stage.

    ``sequence_anchor_xy`` places the sequence's anchor ramp, ``ramps[0]``.
    Every remaining ramp is placed from its predecessor using the corresponding
    entry in ``transitions``. Consequently, a stage containing ``n`` ramps must
    contain exactly ``n - 1`` transitions.

    The stage describes the entire ramp sequence uniformly: geometry lives in
    ``ramps`` and only the relationships between adjacent ramps live in
    ``transitions``.
    """

    sequence_anchor_xy: tuple[float, float]
    ramps: tuple[RampSpec, ...]
    transitions: tuple[RampTransitionSpec, ...]
    landing_start_x: float
    landing_y_range: tuple[float, float]

    def __post_init__(self) -> None:
        ramps = tuple(self.ramps)
        transitions = tuple(self.transitions)
        object.__setattr__(self, "ramps", ramps)
        object.__setattr__(self, "transitions", transitions)

        if not ramps:
            raise ValueError("A tilted-ramp stage must contain at least one ramp specification.")
        if len(transitions) != len(ramps) - 1:
            raise ValueError("A tilted-ramp stage must contain exactly one transition between adjacent ramps.")


# Public course builders


def build_default_level(
    obstacle_stage_index: int,
) -> ParkourLevelCfg:
    """Create one non-flat stage of the default tilted-ramp curriculum.

    ``obstacle_stage_index`` is a zero-based index into
    :data:`DEFAULT_STAGE_SPECS`; it is not the physical terrain-row index used
    at runtime. Curriculum row zero is the shared flat bootstrap, so obstacle
    stage ``n`` becomes terrain row and difficulty order ``n + 1``. For the
    default five-stage ladder, indices 0 through 4 map to rows 1 through 5.

    All default stages are constructed once while the curriculum configuration
    initializes. Runtime promotion selects one of these prebuilt terrain rows;
    it does not call this function with mutable episode state.
    """

    if not 0 <= obstacle_stage_index < len(DEFAULT_STAGE_SPECS):
        raise ValueError("Tilted-ramp obstacle index must select one configured obstacle row.")

    stage_spec = DEFAULT_STAGE_SPECS[obstacle_stage_index]
    ramps = build_ramp_sequence(stage_spec)
    return build_level(
        name=f"tilted_ramps_difficulty_{obstacle_stage_index}",
        difficulty_order=float(obstacle_stage_index + 1),
        ramps=ramps,
        landing_x_range=(stage_spec.landing_start_x, _shared.TERRAIN_X_RANGE_M[1]),
        landing_y_range=stage_spec.landing_y_range,
        target_speed=0.60,
        min_clearance=0.24,
    )


def build_default_levels() -> tuple[ParkourLevelCfg, ...]:
    """Build the flat bootstrap and all default tilted-ramp stages."""

    if len(DEFAULT_STAGE_SPECS) != _shared.NUM_OBSTACLE_STAGES:
        raise ValueError("The tilted-ramp family must define the shared number of obstacle stages.")

    return (_shared.build_bootstrap_level("tilted_ramps"),) + tuple(
        build_default_level(obstacle_stage_index) for obstacle_stage_index in range(len(DEFAULT_STAGE_SPECS))
    )


def build_level(
    *,
    name: str,
    difficulty_order: float,
    ramps: tuple[TiltedRampGeometry, ...],
    landing_x_range: tuple[float, float],
    landing_y_range: tuple[float, float],
    target_speed: float,
    min_clearance: float,
) -> ParkourLevelCfg:
    """Create a laterally banked ramp sequence with an explicit redirected route."""

    ramps = tuple(ramps)
    if not ramps:
        raise ValueError("Tilted-ramp courses require at least one ramp.")
    if not all(isinstance(ramp, TiltedRampGeometry) for ramp in ramps):
        raise TypeError("Every tilted-ramp course entry must be TiltedRampGeometry.")
    if any(ramp.length <= 2.0 * _WAYPOINT_INSET_M for ramp in ramps):
        raise ValueError("Each ramp must be long enough for inset route waypoints.")

    # Collision geometry, rather than only its centerline, must stay inside the
    # generated tile after applying both the bank angle and yaw.
    for ramp in ramps:
        x_bounds, y_bounds = ramp.collision_bounds_xy
        if not (
            _shared.TERRAIN_X_RANGE_M[0] <= x_bounds[0] < x_bounds[1] <= _shared.TERRAIN_X_RANGE_M[1]
            and _shared.TERRAIN_Y_RANGE_M[0] <= y_bounds[0] < y_bounds[1] <= _shared.TERRAIN_Y_RANGE_M[1]
        ):
            raise ValueError("Tilted-ramp collision geometry must lie inside the terrain tile.")

    marker_offset_z = 0.01
    approach_region = ParkourSupportRegionCfg.horizontal_rectangle(
        name="approach_ground",
        structure_name=None,
        x_range=(_shared.TERRAIN_X_RANGE_M[0], _APPROACH_END_X_M),
        y_range=_shared.TERRAIN_Y_RANGE_M,
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
        _APPROACH_END_X_M < landing_region.x_range[0]
        and _shared.TERRAIN_X_RANGE_M[0] <= landing_region.x_range[0]
        and landing_region.x_range[1] <= _shared.TERRAIN_X_RANGE_M[1]
        and _shared.TERRAIN_Y_RANGE_M[0] <= landing_region.y_range[0]
        and landing_region.y_range[1] <= _shared.TERRAIN_Y_RANGE_M[1]
    ):
        raise ValueError("The final ramp landing must fit inside the terrain tile.")
    final_ramp = ramps[-1]
    ray_entry = 0.0
    ray_exit = math.inf
    for origin, direction, bounds in zip(
        final_ramp.centerline_end[:2],
        final_ramp.travel_direction_xy,
        (landing_region.x_range, landing_region.y_range),
    ):
        if math.isclose(direction, 0.0, abs_tol=1.0e-12):
            if not bounds[0] <= origin <= bounds[1]:
                raise ValueError("The final ramp centerline must point into the landing.")
            continue
        distances = (
            (bounds[0] - origin) / direction,
            (bounds[1] - origin) / direction,
        )
        ray_entry = max(ray_entry, min(distances))
        ray_exit = min(ray_exit, max(distances))
    if ray_entry > ray_exit:
        raise ValueError("The final ramp centerline must point into the landing.")

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

    anchor_ramp = ramps[0]
    anchor_direction_x, anchor_direction_y = anchor_ramp.travel_direction_xy
    anchor_start = anchor_ramp.centerline_start
    if anchor_start[0] <= _APPROACH_END_X_M:
        raise ValueError("The first ramp centerline must begin beyond approach ground.")

    # The 0.25 m value is a fixed route-design offset, not a quantity derived
    # from the ramp dimensions. Moving opposite the anchor ramp's unit travel
    # direction places the initial target on flat approach ground, giving the
    # route time to align with the ramp yaw before the robot reaches its edge.
    # The support-region check below rejects layouts for which that assumption
    # does not hold.
    waypoints = [
        ParkourWaypointCfg(
            position=(
                anchor_start[0] - 0.25 * anchor_direction_x,
                anchor_start[1] - 0.25 * anchor_direction_y,
                marker_offset_z,
            )
        )
    ]
    if not approach_region.supports_waypoint(waypoints[0].position):
        raise ValueError("The initial tilted-ramp waypoint must lie on approach ground.")

    # The initial approach waypoint already aligns the route with ramp one.
    # Every later ramp receives an entry waypoint so a gap or direction change
    # is explicit, and every ramp receives an inset exit waypoint.
    for index, (ramp, support) in enumerate(zip(ramps, ramp_supports)):
        direction_x, direction_y = ramp.travel_direction_xy
        ramp_waypoints: list[ParkourWaypointCfg] = []
        if index > 0:
            start = ramp.centerline_start
            ramp_waypoints.append(
                ParkourWaypointCfg(
                    position=(
                        start[0] + _WAYPOINT_INSET_M * direction_x,
                        start[1] + _WAYPOINT_INSET_M * direction_y,
                        ramp.top_center_z + marker_offset_z,
                    ),
                    support_region_name=support.name,
                    is_rewarded_milestone=True,
                )
            )
        end = ramp.centerline_end
        ramp_waypoints.append(
            ParkourWaypointCfg(
                position=(
                    end[0] - _WAYPOINT_INSET_M * direction_x,
                    end[1] - _WAYPOINT_INSET_M * direction_y,
                    ramp.top_center_z + marker_offset_z,
                ),
                support_region_name=support.name,
                is_rewarded_milestone=True,
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
        ),
        support_region_name=landing_region.name,
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
            parameters=_difficulty_parameters(
                ramps,
                landing_x_range=landing_x_range,
                landing_y_range=landing_y_range,
            ),
        ),
    )


def build_ramp_sequence(
    stage_spec: TiltedRampStageSpec,
) -> tuple[TiltedRampGeometry, ...]:
    """Resolve one declarative stage into absolute collision geometry."""

    anchor_spec = stage_spec.ramps[0]
    ramps = [
        TiltedRampGeometry(
            center_xy=stage_spec.sequence_anchor_xy,
            length=anchor_spec.length,
            width=anchor_spec.width,
            thickness=anchor_spec.thickness,
            incline_radians=math.radians(anchor_spec.incline_degrees),
            yaw_radians=math.radians(anchor_spec.yaw_degrees),
        )
    ]
    for ramp_spec, transition in zip(stage_spec.ramps[1:], stage_spec.transitions):
        ramps.append(
            ramps[-1].placed_after(
                length=ramp_spec.length,
                width=ramp_spec.width,
                thickness=ramp_spec.thickness,
                incline_radians=math.radians(ramp_spec.incline_degrees),
                yaw_radians=math.radians(ramp_spec.yaw_degrees),
                gap=transition.gap,
                lateral_offset=transition.lateral_offset,
            )
        )
    return tuple(ramps)


# Course-construction helpers


def _difficulty_parameters(
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

    ``ramp_1_entry_lip_height_m`` is the highest point on the first ramp's
    leading top edge. ``ramp_1_approach_overlap_m`` records how far that edge
    extends back over the approach patch in world X. Together these values
    make the asymmetric first contact of a banked, yawed ramp explicit in
    evaluation metadata.
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

            # Project the XY displacement from the previous centerline end to
            # the current centerline start onto the previous ramp's local axes.
            # ``gap`` is the signed longitudinal separation, not the Euclidean
            # endpoint distance or mesh clearance; positive lateral offset is
            # to the previous ramp's left:
            # delta_xy = gap * travel_xy + lateral_offset * left_xy.
            parameters[f"{prefix}_gap_from_previous_m"] = round(
                delta_x * previous_travel_x + delta_y * previous_travel_y,
                12,
            )
            parameters[f"{prefix}_lateral_offset_from_previous_m"] = round(
                delta_x * previous_left_x + delta_y * previous_left_y,
                12,
            )

    first_entry_edge = (ramps[0].top_corners[0], ramps[0].top_corners[-1])
    parameters.update(
        {
            "ramp_1_entry_lip_height_m": round(
                max(corner[2] for corner in first_entry_edge),
                12,
            ),
            "ramp_1_approach_overlap_m": round(
                max(0.0, _APPROACH_END_X_M - min(corner[0] for corner in first_entry_edge)),
                12,
            ),
        }
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


# Default curriculum

DEFAULT_STAGE_SPECS = (
    # Obstacle stage 0 / curriculum row 1: acquire one wide, straight,
    # gently banked support.
    TiltedRampStageSpec(
        sequence_anchor_xy=(1.50, 0.0),
        ramps=(
            RampSpec(
                length=1.80,
                width=1.60,
                incline_degrees=3.0,
                yaw_degrees=0.0,
            ),
        ),
        transitions=(),
        landing_start_x=2.40,
        landing_y_range=(-1.0, 1.0),
    ),
    # Obstacle stage 1 / curriculum row 2: traverse two contiguous,
    # aligned supports.
    TiltedRampStageSpec(
        sequence_anchor_xy=(1.05, 0.0),
        ramps=(
            RampSpec(
                length=0.90,
                width=1.60,
                incline_degrees=3.0,
                yaw_degrees=0.0,
            ),
            RampSpec(
                length=1.10,
                width=1.60,
                incline_degrees=3.0,
                yaw_degrees=0.0,
            ),
        ),
        transitions=(
            RampTransitionSpec(
                gap=0.0,
                lateral_offset=0.0,
            ),
        ),
        landing_start_x=2.60,
        landing_y_range=(-1.0, 1.0),
    ),
    # Obstacle stage 2 / curriculum row 3: introduce a mild opposing bank,
    # inter-ramp gap, and lateral redirection together at wide support width.
    TiltedRampStageSpec(
        sequence_anchor_xy=(1.05, -0.15),
        ramps=(
            RampSpec(
                length=0.90,
                width=1.60,
                incline_degrees=4.0,
                yaw_degrees=4.0,
            ),
            RampSpec(
                length=1.10,
                width=1.60,
                incline_degrees=-4.0,
                yaw_degrees=14.0,
            ),
        ),
        transitions=(
            RampTransitionSpec(
                gap=0.12,
                lateral_offset=0.18,
            ),
        ),
        landing_start_x=2.72,
        landing_y_range=(-1.0, 1.20),
    ),
    # Obstacle stage 3 / curriculum row 4: bridge the wide mild row to the
    # hardest row by increasing the opposing banks and redirection while only
    # moderately narrowing both supports.
    TiltedRampStageSpec(
        sequence_anchor_xy=(1.10, -0.10),
        ramps=(
            RampSpec(
                length=0.90,
                width=1.25,
                incline_degrees=7.0,
                yaw_degrees=7.0,
            ),
            RampSpec(
                length=1.10,
                width=1.25,
                incline_degrees=-7.0,
                yaw_degrees=23.0,
            ),
        ),
        transitions=(
            RampTransitionSpec(
                gap=0.16,
                lateral_offset=0.28,
            ),
        ),
        landing_start_x=2.88,
        landing_y_range=(0.0, 1.60),
    ),
    # Obstacle stage 4 / curriculum row 5: combine narrow supports, stronger
    # opposing banks, a larger gap, and a sharper inter-ramp redirection. Keep
    # the first approach aligned with the successful preceding row.
    TiltedRampStageSpec(
        sequence_anchor_xy=(1.15, 0.0),
        ramps=(
            RampSpec(
                length=1.10,
                width=1.00,
                incline_degrees=12.0,
                yaw_degrees=10.0,
            ),
            RampSpec(
                length=1.30,
                width=1.00,
                incline_degrees=-12.0,
                yaw_degrees=32.0,
            ),
        ),
        transitions=(
            RampTransitionSpec(
                gap=0.20,
                lateral_offset=0.40,
            ),
        ),
        landing_start_x=3.10,
        # Follow ramp two's redirected exit while retaining a margin from the
        # terrain boundary. The previous centered landing forced an almost
        # right-angle turn after the ramp and only touched one of its corners.
        landing_y_range=(0.40, 1.80),
    ),
)
