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


@dataclass(frozen=True)
class TiltedRampGeometry:
    """Represent a rectangular ramp whose top is cross-sloped by a roll.

    ``center_xy`` is the world-frame XY projection of the top face's center.
    It is not generally the XY position of the collision box's volume center.
    With zero incline the two centers coincide in XY. After a lateral roll,
    however, the top normal has a horizontal component, so moving half the
    slab thickness inward from the top face to the volume center also shifts
    XY. :attr:`mesh_position` computes that displacement.

    The ramp's local X axis is its travel direction, its local Y axis points
    left, and ``incline_radians`` rotates the slab about local X before
    ``yaw_radians`` rotates it about world Z. A positive incline therefore
    raises the ramp's left edge.

    ``low_edge_z`` fixes the lower of the two top-surface side edges. The
    derived collision-box position accounts for both the lateral roll and the
    horizontal displacement between the box center and its top-face center.

    Attributes:
        center_xy: World-frame XY projection of the top-face center.
        length: Distance along the ramp's travel direction.
        width: Distance across the tilted top surface.
        thickness: Collision-slab thickness measured normal to the top surface.
        incline_radians: Roll about the local X axis. Positive values raise the
            local-left edge.
        yaw_radians: Counterclockwise angle from world positive X to the local
            positive X axis.
        low_edge_z: Height of the lower top-surface side edge.
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

        # Yaw is measured counterclockwise from world positive X. Its unit-circle
        # coordinates ``(cos(yaw), sin(yaw))`` are therefore already normalized.
        return (math.cos(self.yaw_radians), math.sin(self.yaw_radians))

    @property
    def left_direction_xy(self) -> tuple[float, float]:
        """Return the unit XY direction pointing left of ramp travel."""

        travel_x, travel_y = self.travel_direction_xy

        # A positive 90-degree rotation maps ``(x, y)`` to ``(-y, x)``. In the
        # right-handed XY plane, this produces the unit direction to the left.
        return (-travel_y, travel_x)

    @property
    def top_center_z(self) -> float:
        """Return the height at the center of the ramp's top surface."""

        # Half the surface width has vertical projection
        # ``0.5 * width * sin(incline)``. The left and right edges receive
        # opposite signed projections, so the lower edge lies their absolute
        # magnitude below the top center. Adding that magnitude to
        # ``low_edge_z`` recovers the center height. This uses sine rather than
        # tangent because ``width`` is measured along the tilted surface.
        return self.low_edge_z + 0.5 * self.width * abs(math.sin(self.incline_radians))

    @property
    def centerline_start(self) -> tuple[float, float, float]:
        """Return the top-centerline endpoint opposite the travel direction."""

        travel_x, travel_y = self.travel_direction_xy
        half_length = 0.5 * self.length
        return (
            self.center_xy[0] - half_length * travel_x,
            self.center_xy[1] - half_length * travel_y,
            self.top_center_z,
        )

    @property
    def centerline_end(self) -> tuple[float, float, float]:
        """Return the top-centerline endpoint in the travel direction."""

        travel_x, travel_y = self.travel_direction_xy
        half_length = 0.5 * self.length
        return (
            self.center_xy[0] + half_length * travel_x,
            self.center_xy[1] + half_length * travel_y,
            self.top_center_z,
        )

    @property
    def surface_normal(self) -> tuple[float, float, float]:
        """Return the upward unit normal of the tilted top surface."""

        left_x, left_y = self.left_direction_xy
        sin_incline = math.sin(self.incline_radians)
        cos_incline = math.cos(self.incline_radians)

        # In the ``(forward, left, up)`` basis, rolling about forward changes
        # the top normal from ``(0, 0, 1)`` to
        # ``(0, -sin(incline), cos(incline))``. Yaw maps local left to
        # ``(left_x, left_y, 0)`` while leaving world up unchanged. Expanding
        # those basis coefficients gives the components below. The positive
        # cosine term keeps the normal upward-facing over the permitted incline
        # range.
        return (
            -sin_incline * left_x,
            -sin_incline * left_y,
            cos_incline,
        )

    @property
    def mesh_position(self) -> tuple[float, float, float]:
        """Return the XYZ center required by the rotated collision box."""

        normal_x, normal_y, normal_z = self.surface_normal
        half_thickness = 0.5 * self.thickness

        # A box factory positions its volume center. Move half a thickness
        # opposite the top normal so the configured top center remains at
        # ``(*center_xy, top_center_z)`` after rotation.
        return (
            self.center_xy[0] - half_thickness * normal_x,
            self.center_xy[1] - half_thickness * normal_y,
            self.top_center_z - half_thickness * normal_z,
        )

    @property
    def orientation_rpy(self) -> tuple[float, float, float]:
        """Return roll, pitch, and yaw for ``ParkourStructureCfg``."""

        return (self.incline_radians, 0.0, self.yaw_radians)

    @property
    def top_corners(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
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
    def collision_corners(
        self,
    ) -> tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
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

        The next centerline starts ``gap`` meters beyond this ramp along this
        ramp's travel direction, with ``lateral_offset`` measured along this
        ramp's left direction. From that start point, half of the next ramp's
        length is applied along the next ramp's own yaw to obtain its center.
        Omitting ``low_edge_z`` retains this ramp's lower-edge height.

        Args:
            length: Length of the next ramp.
            width: Surface width of the next ramp.
            thickness: Collision-slab thickness of the next ramp.
            incline_radians: Roll of the next ramp about its local X axis.
            yaw_radians: World-frame yaw of the next ramp.
            gap: Non-negative distance beyond this ramp's centerline end.
            lateral_offset: Signed offset along this ramp's left direction.
            low_edge_z: Lower-edge height of the next ramp. If omitted, reuse
                this ramp's value.

        Returns:
            Validated geometry for the positioned next ramp.

        Raises:
            ValueError: If ``gap`` is negative or any numeric constraint is
                invalid.
            TypeError: If any numeric argument cannot be converted to a finite
                non-Boolean float.
        """

        gap = _finite_float(gap, field_name="gap")
        lateral_offset = _finite_float(
            lateral_offset,
            field_name="lateral_offset",
        )
        if gap < 0.0:
            raise ValueError("gap must be non-negative.")

        next_low_edge_z = self.low_edge_z if low_edge_z is None else _finite_float(low_edge_z, field_name="low_edge_z")

        # Construct once at the origin to normalize and validate all new-ramp
        # parameters before using its own yaw to calculate its final center.
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
        next_start_x = self.centerline_end[0] + gap * current_travel_x + lateral_offset * current_left_x
        next_start_y = self.centerline_end[1] + gap * current_travel_y + lateral_offset * current_left_y

        next_travel_x, next_travel_y = next_ramp.travel_direction_xy
        next_center_xy = (
            next_start_x + 0.5 * next_ramp.length * next_travel_x,
            next_start_y + 0.5 * next_ramp.length * next_travel_y,
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
    ) -> tuple[float, float, float]:
        """Map local top-surface offsets to world-frame XYZ coordinates.

        Args:
            longitudinal_offset: Signed distance along the travel direction.
            lateral_offset: Signed surface distance toward the local left.

        Returns:
            World-frame XYZ coordinates of the requested top-surface point.
        """

        travel_x, travel_y = self.travel_direction_xy
        left_x, left_y = self.left_direction_xy
        cos_incline = math.cos(self.incline_radians)
        sin_incline = math.sin(self.incline_radians)

        # Yaw maps the longitudinal basis to
        # ``travel_w = (travel_x, travel_y, 0)``. Roll leaves that axis
        # unchanged and maps the lateral basis to
        # ``left_w = (cos(incline) * left_x,
        #             cos(incline) * left_y,
        #             sin(incline))``.
        #
        # The two rotated basis vectors remain perpendicular unit vectors, so
        # both offsets remain exact signed distances along the top surface.
        # Adding their scaled components to the top-face center produces the
        # world point returned below.
        return (
            self.center_xy[0] + longitudinal_offset * travel_x + lateral_offset * cos_incline * left_x,
            self.center_xy[1] + longitudinal_offset * travel_y + lateral_offset * cos_incline * left_y,
            self.top_center_z + lateral_offset * sin_incline,
        )


def _finite_float(value: object, *, field_name: str) -> float:
    """Convert a configuration value to a finite, non-Boolean float.

    Args:
        value: Candidate numeric value.
        field_name: Configuration field named in validation errors.

    Returns:
        The converted finite float.

    Raises:
        TypeError: If ``value`` is Boolean or cannot be converted to a float.
        ValueError: If the converted value is not finite.
    """

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
    """Convert a configuration value to a finite positive float.

    Args:
        value: Candidate numeric value.
        field_name: Configuration field named in validation errors.

    Returns:
        The converted finite positive float.

    Raises:
        TypeError: If ``value`` is Boolean or cannot be converted to a float.
        ValueError: If the converted value is not finite and positive.
    """

    converted = _finite_float(value, field_name=field_name)
    if converted <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return converted
