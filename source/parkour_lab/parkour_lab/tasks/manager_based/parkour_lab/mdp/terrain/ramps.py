# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-free geometry for laterally tilted rectangular ramps.

The ramp is rolled laterally about its local X axis, which points in the
configured travel direction. Following the right-hand rotation convention, a
positive incline raises the local-left edge and lowers the local-right edge; a
negative incline does the opposite. The ramp therefore slopes across its width
rather than upward or downward along its travel direction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Module-private Cartesian XYZ point used by the geometry properties below.
_Point3 = tuple[float, float, float]

# Module-private type alias for a 3D quadrilateral: exactly four ordered XYZ
# corners. Their order defines the face's perimeter winding; ``top_corners``
# uses counterclockwise order. This adds type information but no runtime class.
_Quad3 = tuple[_Point3, _Point3, _Point3, _Point3]


@dataclass(frozen=True, slots=True)
class TiltedRampGeometry:
    """Exact geometry for a laterally rolled rectangular collision slab.

    ``center_xy`` is the world-frame XY projection of the top face's center.
    Local X points along travel and local Y points left. ``incline_radians``
    rolls about local X, with positive values raising the left edge;
    ``yaw_radians`` then rotates counterclockwise about world Z. ``low_edge_z``
    anchors the lower top edge, while :attr:`mesh_position` converts the
    top-face anchor into the collision box's volume-center position.
    """

    center_xy: tuple[float, float]
    length: float
    width: float
    thickness: float
    incline_radians: float
    yaw_radians: float
    low_edge_z: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.center_xy, (tuple, list)) or len(self.center_xy) != 2:
            raise TypeError("center_xy must contain exactly two numeric values.")

        center_xy = (
            _finite_float(self.center_xy[0], field_name="center_xy[0]"),
            _finite_float(self.center_xy[1], field_name="center_xy[1]"),
        )
        length = _positive_float(self.length, field_name="length")
        width = _positive_float(self.width, field_name="width")
        thickness = _positive_float(self.thickness, field_name="thickness")
        incline_radians = _finite_float(
            self.incline_radians,
            field_name="incline_radians",
        )
        yaw_radians = _finite_float(
            self.yaw_radians,
            field_name="yaw_radians",
        )
        low_edge_z = _finite_float(self.low_edge_z, field_name="low_edge_z")

        if abs(incline_radians) >= 0.5 * math.pi:
            raise ValueError("abs(incline_radians) must be less than pi / 2.")

        object.__setattr__(self, "center_xy", center_xy)
        object.__setattr__(self, "length", length)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "incline_radians", incline_radians)
        object.__setattr__(self, "yaw_radians", yaw_radians)
        object.__setattr__(self, "low_edge_z", low_edge_z)

    @property
    def travel_direction_xy(self) -> tuple[float, float]:
        """Return the unit XY direction of the ramp's local positive X axis."""

        return (math.cos(self.yaw_radians), math.sin(self.yaw_radians))

    @property
    def left_direction_xy(self) -> tuple[float, float]:
        """Return the unit XY direction pointing left of ramp travel."""

        travel_x, travel_y = self.travel_direction_xy
        return (-travel_y, travel_x)

    @property
    def top_center_z(self) -> float:
        """Return the height at the center of the ramp's top surface."""

        # Width is measured along the surface, so its vertical projection uses
        # sine rather than tangent.
        return self.low_edge_z + 0.5 * self.width * abs(math.sin(self.incline_radians))

    @property
    def centerline_start(self) -> _Point3:
        """Return the top-centerline endpoint opposite the travel direction."""

        return self._top_surface_point(-0.5 * self.length, 0.0)

    @property
    def centerline_end(self) -> _Point3:
        """Return the top-centerline endpoint in the travel direction."""

        return self._top_surface_point(0.5 * self.length, 0.0)

    @property
    def surface_normal(self) -> _Point3:
        """Return the upward unit normal of the tilted top surface."""

        left_x, left_y = self.left_direction_xy
        sin_incline = math.sin(self.incline_radians)
        cos_incline = math.cos(self.incline_radians)

        # n_world = R_z(psi) R_x(phi) (0, 0, 1)^T: roll tilts the local top
        # normal, then yaw rotates its horizontal component into world XY.
        return (
            -sin_incline * left_x,
            -sin_incline * left_y,
            cos_incline,
        )

    @property
    def mesh_position(self) -> _Point3:
        """Return the XYZ center required by the rotated collision box."""

        normal_x, normal_y, normal_z = self.surface_normal
        half_thickness = 0.5 * self.thickness

        # Move inward from the anchored top face to the box's volume center.
        return (
            self.center_xy[0] - half_thickness * normal_x,
            self.center_xy[1] - half_thickness * normal_y,
            self.top_center_z - half_thickness * normal_z,
        )

    @property
    def orientation_rpy(self) -> _Point3:
        """Return roll, pitch, and yaw for ``ParkourStructureCfg``."""

        return (self.incline_radians, 0.0, self.yaw_radians)

    @property
    def top_corners(self) -> _Quad3:
        """Return the four top corners in counterclockwise local-XY order."""

        half_length = 0.5 * self.length
        half_width = 0.5 * self.width
        return (
            self._top_surface_point(-half_length, -half_width),
            self._top_surface_point(half_length, -half_width),
            self._top_surface_point(half_length, half_width),
            self._top_surface_point(-half_length, half_width),
        )

    @property
    def collision_corners(self) -> tuple[_Point3, ...]:
        """Return the four top and four bottom collision-box corners."""

        top_corners = self.top_corners
        normal_x, normal_y, normal_z = self.surface_normal
        bottom_corners = tuple(
            (
                x - self.thickness * normal_x,
                y - self.thickness * normal_y,
                z - self.thickness * normal_z,
            )
            for x, y, z in top_corners
        )
        return (*top_corners, *bottom_corners)

    @property
    def collision_bounds_xy(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return inclusive X and Y bounds of the rotated collision slab."""

        corners = self.collision_corners
        x_coordinates = tuple(corner[0] for corner in corners)
        y_coordinates = tuple(corner[1] for corner in corners)
        return (
            (min(x_coordinates), max(x_coordinates)),
            (min(y_coordinates), max(y_coordinates)),
        )

    def placed_after(
        self,
        *,
        length: float,
        width: float,
        thickness: float,
        incline_radians: float,
        yaw_radians: float,
        gap: float,
        lateral_offset: float,
        low_edge_z: float | None = None,
    ) -> TiltedRampGeometry:
        """Construct the next ramp after this ramp's centerline end.

        ``gap`` follows the current travel direction and ``lateral_offset``
        follows its left direction. The next center then advances half its own
        length along its own yaw. An omitted ``low_edge_z`` is inherited.
        """

        gap = _finite_float(gap, field_name="gap")
        lateral_offset = _finite_float(
            lateral_offset,
            field_name="lateral_offset",
        )
        if gap < 0.0:
            raise ValueError("gap must be non-negative.")

        next_low_edge_z = (
            self.low_edge_z
            if low_edge_z is None
            else _finite_float(low_edge_z, field_name="low_edge_z")
        )

        # Validate the new geometry before its normalized values define center.
        next_ramp = TiltedRampGeometry(
            center_xy=(0.0, 0.0),
            length=length,
            width=width,
            thickness=thickness,
            incline_radians=incline_radians,
            yaw_radians=yaw_radians,
            low_edge_z=next_low_edge_z,
        )

        current_travel_x, current_travel_y = self.travel_direction_xy
        current_left_x, current_left_y = self.left_direction_xy
        end_x, end_y, _ = self.centerline_end
        next_start_xy = (
            end_x + gap * current_travel_x + lateral_offset * current_left_x,
            end_y + gap * current_travel_y + lateral_offset * current_left_y,
        )

        next_travel_x, next_travel_y = next_ramp.travel_direction_xy
        next_center_xy = (
            next_start_xy[0] + 0.5 * next_ramp.length * next_travel_x,
            next_start_xy[1] + 0.5 * next_ramp.length * next_travel_y,
        )
        return TiltedRampGeometry(
            center_xy=next_center_xy,
            length=next_ramp.length,
            width=next_ramp.width,
            thickness=next_ramp.thickness,
            incline_radians=next_ramp.incline_radians,
            yaw_radians=next_ramp.yaw_radians,
            low_edge_z=next_ramp.low_edge_z,
        )

    def _top_surface_point(
        self,
        longitudinal_offset: float,
        lateral_offset: float,
    ) -> _Point3:
        """Map local top-surface offsets to world-frame XYZ."""

        travel_x, travel_y = self.travel_direction_xy
        left_x, left_y = self.left_direction_xy
        cos_incline = math.cos(self.incline_radians)
        sin_incline = math.sin(self.incline_radians)

        # p_world = c + R_z(psi) R_x(phi) p_local, where R_x applies the
        # incline about local X and R_z applies the yaw about world Z.
        return (
            self.center_xy[0]
            + longitudinal_offset * travel_x
            + lateral_offset * cos_incline * left_x,
            self.center_xy[1]
            + longitudinal_offset * travel_y
            + lateral_offset * cos_incline * left_y,
            self.top_center_z + lateral_offset * sin_incline,
        )


def _finite_float(value: object, *, field_name: str) -> float:
    """Convert a configuration value to a finite non-Boolean float."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a finite number, not bool.")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{field_name} must be a finite number.") from error
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite.")
    return converted


def _positive_float(value: object, *, field_name: str) -> float:
    """Convert a configuration value to a finite positive float."""

    converted = _finite_float(value, field_name=field_name)
    if converted <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return converted
