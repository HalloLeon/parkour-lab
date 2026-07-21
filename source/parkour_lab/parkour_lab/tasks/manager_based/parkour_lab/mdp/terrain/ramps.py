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
    """Describe one rectangular ramp whose top is cross-sloped by a roll.

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

        # Yaw is the counterclockwise angle from world +X in the XY plane, so
        # its point on the unit circle has X coordinate ``cos(yaw)`` and Y
        # coordinate ``sin(yaw)``. Since ``cos(yaw)^2 + sin(yaw)^2 = 1``, the
        # returned travel direction is already normalized.
        return (math.cos(self.yaw_radians), math.sin(self.yaw_radians))

    @property
    def left_direction_xy(self) -> tuple[float, float]:
        """Return the unit XY direction pointing left of ramp travel."""

        travel_x, travel_y = self.travel_direction_xy

        # Rotating a 2D vector ``(x, y)`` by +90 degrees about world Z gives
        # ``(-y, x)``. With the usual right-handed XY convention, this is left
        # of the travel direction. The rotation preserves length and produces
        # a perpendicular vector, so the result remains a unit direction.
        return (-travel_y, travel_x)

    @property
    def top_center_z(self) -> float:
        """Return the height at the center of the ramp's top surface."""

        # The configured width is measured along the top surface. Before roll,
        # the center-to-left-edge vector is ``(0, 0.5 * width, 0)`` in local
        # coordinates. Rotating it about local X produces
        # ``(0, 0.5 * width * cos(incline),
        #       0.5 * width * sin(incline))``. The right-edge vector is its
        # negative, so if ``z_center`` denotes the top-center height, then
        #
        # ``z_left  = z_center + 0.5 * width * sin(incline)`` and
        # ``z_right = z_center - 0.5 * width * sin(incline)``.
        #
        # Whichever edge is lower therefore has height
        # ``z_center - abs(0.5 * width * sin(incline))``. Setting this equal to
        # ``low_edge_z`` and solving for ``z_center`` yields the expression
        # below. Sine is used rather than tangent because ``width`` is the
        # distance along the tilted surface, whose vertical projection is
        # ``width * sin(incline)``.
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

        # In the ramp-local basis, the rolled normal is
        # ``(0, -sin(incline), cos(incline))``. These numbers are coefficients
        # of the local forward, left, and up basis vectors; they are not yet
        # world-X, world-Y, and world-Z coordinates. Its zero forward
        # coefficient removes any component along the travel direction.
        #
        # Before yaw, local left is ``(0, 1, 0)``. Rotating this vector about
        # world Z by ``yaw`` gives
        # ``R_z(yaw) * (0, 1, 0) = (-sin(yaw), cos(yaw), 0)``, which is exactly
        # ``left_w = (left_x, left_y, 0)`` from ``left_direction_xy``. Yaw
        # rotates only the horizontal plane: its rotation axis is world Z, so
        # ``R_z(yaw) * (0, 0, 1) = (0, 0, 1)`` and world up is unchanged. The
        # world normal can therefore be assembled directly from this yawed
        # left basis vector and the unchanged up basis vector:
        #
        # ``n_w = 0 * travel_w
        #        - sin(incline) * left_w
        #        + cos(incline) * up_w``.
        #
        # Expanding this basis sum component by component gives
        # ``n_w = (-sin * left_x, -sin * left_y, cos)``. A yawed local-left
        # vector generally has both world-X and world-Y components, so the
        # first returned value need not be zero even though the local-forward
        # coefficient is zero. The final value comes solely from world up and
        # remains positive for the permitted incline range, making this the
        # upward-facing normal.
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
        """Map local top-surface offsets to world-aligned XYZ coordinates."""

        travel_x, travel_y = self.travel_direction_xy
        left_x, left_y = self.left_direction_xy
        cos_incline = math.cos(self.incline_radians)
        sin_incline = math.sin(self.incline_radians)

        # Map local top-face coordinates to world coordinates.
        #
        # Let the top-face center C be the local origin, with signed offsets in meters:
        #
        #   u = longitudinal_offset  # +forward, -backward
        #   v = lateral_offset       # +left,    -right
        #
        # The ramp orientation first yaws about fixed world Z, then rolls about the
        # resulting local X-axis (the travel direction):
        #
        #   R = Rz(yaw) @ Rx(incline)
        #
        # For column vectors, the rightmost matrix acts first. Using
        #
        #   cy = cos(yaw),  sy = sin(yaw)
        #   ci = cos(incline),  si = sin(incline)
        #
        # the rotation matrices are
        #
        #        [ cy -sy  0 ]         [ 1  0   0 ]
        #   Rz = [ sy  cy  0 ],   Rx = [ 0  ci -si ].
        #        [  0   0  1 ]         [ 0  si  ci ]
        #
        # Their product is
        #
        #       [ cy -sy*ci   sy*si ]
        #   R = [ sy  cy*ci  -cy*si ].
        #       [  0     si      ci ]
        #
        # The local top surface is the z = 0 plane, with basis vectors
        #
        #   e_x = (1, 0, 0)  # forward
        #   e_y = (0, 1, 0)  # left
        #
        # Applying R gives their world-space directions:
        #
        #   travel_w = R @ e_x
        #            = (cy, sy, 0)
        #            = (travel_x, travel_y, 0)
        #
        #   left_w = Rz @ e_y
        #          = (-sy, cy, 0)
        #          = (left_x, left_y, 0)
        #
        #   rolled_left_w = R @ e_y
        #                 = (ci*left_x, ci*left_y, si)
        #
        # Thus roll leaves travel_w unchanged and tilts left_w toward world up. Since
        # travel_w and rolled_left_w are perpendicular unit vectors, u and v remain
        # exact signed distances along the tilted top surface.
        #
        # For q = (u, v, 0), the world point is
        #
        #   P = C + R @ q
        #     = C + u*travel_w + v*rolled_left_w
        #
        # With
        #
        #   C = (self.center_xy[0], self.center_xy[1], self.top_center_z)
        #
        # this expands component-wise to
        #
        #   P.x = self.center_xy[0] + u*travel_x + v*ci*left_x
        #   P.y = self.center_xy[1] + u*travel_y + v*ci*left_y
        #   P.z = self.top_center_z + v*si
        return (
            self.center_xy[0] + longitudinal_offset * travel_x + lateral_offset * cos_incline * left_x,
            self.center_xy[1] + longitudinal_offset * travel_y + lateral_offset * cos_incline * left_y,
            self.top_center_z + lateral_offset * sin_incline,
        )


def _finite_float(value: object, *, field_name: str) -> float:
    """Return one finite float without accepting Boolean configuration values."""

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
    """Return one finite positive float."""

    converted = _finite_float(value, field_name=field_name)
    if converted <= 0.0:
        raise ValueError(f"{field_name} must be positive.")
    return converted
