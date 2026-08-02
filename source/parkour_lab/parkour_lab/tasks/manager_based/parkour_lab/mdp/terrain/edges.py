# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Derive exposed support edges and query contact near them at runtime.

A course is one family/difficulty cell of the curriculum matrix. Its support
polygons contribute directed XYZ boundary segments. Shared, oppositely wound
portions are removed as internal seams, and the remaining edges are packed into
padded tensors by course.

Runtime queries combine exact 3D point-to-segment distance with recent
foot-contact history so a swing foot passing over an edge is not treated as an
edge contact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

    from ..curriculums.config import ParkourCurriculumCfg
    from ..curriculums.levels import ParkourLevelCfg

_GEOMETRY_TOLERANCE = 1.0e-9


@dataclass(frozen=True)
class _SupportEdgeCache:
    """Padded device-side edge geometry for one curriculum configuration.

    ``segment_table`` has shape ``(num_courses, max_segments, 2, 3)`` and stores
    two XYZ endpoints for each exposed edge. ``valid_segment_mask`` has shape
    ``(num_courses, max_segments)`` and distinguishes real edges from zero-filled
    padding. Retaining ``curriculum_cfg`` by identity prevents geometry derived
    from one configuration object from being reused for another.
    """

    curriculum_cfg: ParkourCurriculumCfg
    segment_table: torch.Tensor
    valid_segment_mask: torch.Tensor

    def matches(
        self,
        curriculum_cfg: ParkourCurriculumCfg,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> bool:
        """Return whether the cache matches the configuration and tensor layout.

        Reuse requires the identical curriculum object, the requested tensor
        device, and the requested floating-point dtype. The Boolean validity
        mask is device-sensitive but has a fixed Boolean dtype.
        """

        return (
            self.curriculum_cfg is curriculum_cfg
            and self.segment_table.device == device
            and self.segment_table.dtype == dtype
            and self.valid_segment_mask.device == device
        )


def foot_edge_contact_mask(
    env: ManagerBasedRLEnv,
    *,
    curriculum_cfg: ParkourCurriculumCfg,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Identify recently contacted feet near exposed edges of their course.

    Each environment selects the edge table for its active curriculum course.
    A foot is marked only when its recent contact-force history exceeds
    ``foot_edge_contact_threshold`` and its current XYZ position lies within
    ``edge_width_threshold`` of an exposed finite segment. Internal seams and
    padded table rows are excluded before this query.

    Args:
        env: Vectorized manager-based environment containing route state,
            robot bodies, and the configured contact sensor.
        curriculum_cfg: Curriculum courses and metric contact/edge thresholds.
        asset_cfg: Robot entity and body selection identifying the queried
            feet.
        sensor_cfg: Contact sensor entity and body selection matching those
            feet.

    Returns:
        Boolean tensor of shape ``(num_envs, num_feet)``. ``True`` means that
        the corresponding foot is both recently contacting terrain and close
        to an exposed edge.

    Raises:
        RuntimeError: If active per-environment course indices are unavailable
            or malformed.
    """

    import torch

    from .._shared import contact, robot, runtime
    from ..navigation import route

    course_indices = route.active_course_indices(env)

    foot_positions = robot._selected_body_pos_env(env, asset_cfg)

    # Reduce the history axis, retaining one recent-contact flag per foot.
    recent_contact = torch.any(
        contact._force_norm_mask(env, sensor_cfg=sensor_cfg) > curriculum_cfg.foot_edge_contact_threshold,
        dim=1,
    )
    runtime._validate_matching_shape(
        recent_contact,
        foot_positions[..., 0],
        lhs_name="foot contact mask",
        rhs_name="foot positions",
    )

    edge_cache = _get_support_edge_cache(
        env,
        curriculum_cfg,
        device=foot_positions.device,
        dtype=foot_positions.dtype,
    )
    edge_distance = _minimum_distance_to_course_edges(
        foot_positions,
        edge_cache.segment_table,
        edge_cache.valid_segment_mask,
        course_indices,
    )
    return recent_contact & (edge_distance <= curriculum_cfg.edge_width_threshold)


def _exposed_segment_fragments(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    all_segments: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...],
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    """Return the exposed pieces of one directed 3D support boundary.

    The candidate is parameterized by signed metric distance from ``start``
    along its unit direction, making its complete interval ``[0, length]``.
    Every oppositely directed segment that overlaps the same infinite 3D line
    removes its shared interval. This suppresses both coplanar seams and exact
    folds between adjacent support polygons while retaining partially exposed
    prefixes and suffixes. The remaining scalar intervals are finally mapped
    back to directed XYZ endpoint pairs.

    Args:
        start: First XYZ endpoint of the candidate boundary.
        end: Second XYZ endpoint, establishing its direction.
        all_segments: All directed support boundaries in the course, including
            the candidate itself.

    Returns:
        Exposed directed XYZ fragments in candidate order. The result is empty
        for a degenerate candidate or one covered completely by internal
        seams.
    """

    vector = tuple(end[axis] - start[axis] for axis in range(3))
    length = math.sqrt(sum(component * component for component in vector))
    if length <= _GEOMETRY_TOLERANCE:
        return ()
    unit_direction = tuple(component / length for component in vector)

    # Parameterize the candidate by metric distance from ``start``. This works
    # for horizontal, rotated, and sloped edges while retaining the simple
    # interval subtraction used to split partially covered boundaries.
    fragments = ((0.0, length),)

    # Oppositely directed overlap on the same 3D line is the neighboring side
    # of either a coplanar seam or a fold between two differently tilted faces.
    for other_start, other_end in all_segments:
        blocker = _shared_collinear_interval(
            start,
            end,
            other_start,
            other_end,
        )
        # Offset, crossing, same-direction, and endpoint-only boundaries do not
        # hide a positive-length part of this edge.
        if blocker is None:
            continue

        # Remove the shared one-dimensional interval from every piece that is
        # still exposed. One blocker may shorten a piece, split it in two, or
        # remove it completely; the flattened tuple becomes the input for the
        # next comparison and supports multiple partial neighboring seams.
        fragments = tuple(remaining for fragment in fragments for remaining in _subtract_interval(fragment, blocker))

    def point_at(distance: float) -> tuple[float, float, float]:
        """Map a metric line coordinate back to an XYZ candidate point.

        Exact candidate endpoints are returned unchanged when the coordinate
        lies within the geometry tolerance. Interior coordinates use
        ``start + distance * unit_direction``.
        """

        if math.isclose(distance, 0.0, abs_tol=_GEOMETRY_TOLERANCE):
            return start
        if math.isclose(distance, length, abs_tol=_GEOMETRY_TOLERANCE):
            return end
        return tuple(start[axis] + distance * unit_direction[axis] for axis in range(3))

    return tuple(
        (
            point_at(fragment_start),
            point_at(fragment_end),
        )
        for fragment_start, fragment_end in fragments
    )


def _get_support_edge_cache(
    env: ManagerBasedRLEnv,
    curriculum_cfg: ParkourCurriculumCfg,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _SupportEdgeCache:
    """Return padded exposed-edge tensors on the requested device.

    The cache is reused only when :meth:`_SupportEdgeCache.matches` accepts the
    curriculum identity, device, and dtype.

    On a cache miss, exposed segments are derived independently for every course
    and padded to the maximum edge count across courses. Zero-filled rows make
    the tensor rectangular, while a Boolean mask prevents those rows from
    participating in distance minima. The rebuilt cache is stored on ``env``
    for subsequent control steps.

    Args:
        env: Environment that owns the runtime cache.
        curriculum_cfg: Curriculum whose ordered courses supply the exposed
            edges.
        device: Device on which the edge and validity tensors must reside.
        dtype: Floating-point dtype used for XYZ edge coordinates.

    Returns:
        Cache containing a segment tensor of shape
        ``(num_courses, max_segments, 2, 3)`` and a validity mask of shape
        ``(num_courses, max_segments)``.
    """

    import torch

    cache = getattr(env, "_parkour_support_edge_cache", None)
    if isinstance(cache, _SupportEdgeCache) and cache.matches(
        curriculum_cfg,
        device=device,
        dtype=dtype,
    ):
        return cache

    # Each family/difficulty course may contain a different number of exposed
    # edges. Every edge consists of two XYZ endpoints, where Z identifies its
    # support surface.
    segments_by_course = tuple(_support_edge_segments(course) for course in curriculum_cfg.courses)
    max_segments = max(len(segments) for segments in segments_by_course)

    # Shape: ``(num_courses, max_segments, 2, 3)``. Shorter courses retain
    # zero-filled padding in their unused segment rows.
    segment_table = torch.zeros(
        (len(segments_by_course), max_segments, 2, 3),
        device=device,
        dtype=dtype,
    )

    # Shape: ``(num_courses, max_segments)``. True entries distinguish real
    # segments from the padded rows that the distance kernel must ignore.
    valid_segment_mask = torch.zeros(
        (len(segments_by_course), max_segments),
        device=device,
        dtype=torch.bool,
    )
    for course_index, segments in enumerate(segments_by_course):
        segment_count = len(segments)
        segment_table[course_index, :segment_count] = torch.tensor(
            segments,
            device=device,
            dtype=dtype,
        )
        valid_segment_mask[course_index, :segment_count] = True

    cache = _SupportEdgeCache(
        curriculum_cfg=curriculum_cfg,
        segment_table=segment_table,
        valid_segment_mask=valid_segment_mask,
    )
    env._parkour_support_edge_cache = cache
    return cache


def _minimum_distance_to_course_edges(
    points: torch.Tensor,
    segment_table: torch.Tensor,
    valid_segment_mask: torch.Tensor,
    course_indices: torch.Tensor,
) -> torch.Tensor:
    """Measure 3D points against the exposed edges of their course.

    Each environment selects the segment table for its course. Every query point
    is compared with every valid finite edge in that table using a clamped line
    projection, and the segment-axis minimum is returned. Distances include
    height, preventing an edge on one support surface from penalizing a foot
    contacting another surface at the same XY location. Padded rows are assigned
    infinite distance and cannot become minima.

    Args:
        points: Query positions with shape ``(num_envs, num_points, 3)``.
        segment_table: Padded edge endpoints with shape
            ``(num_courses, max_segments, 2, 3)``.
        valid_segment_mask: Valid rows of ``segment_table`` with shape
            ``(num_courses, max_segments)``.
        course_indices: Flattened family/difficulty index for each environment,
            with shape ``(num_envs,)``.

    Returns:
        Minimum distance for every query point, with shape
        ``(num_envs, num_points)``.
    """

    import torch

    # Shape legend: E = environments, P = query points, S = padded segments.
    #
    # Geometry roadmap for a point P and segment from A to B:
    #
    #     v = B - A
    #     w = P - A
    #     t = clamp(dot(w, v) / dot(v, v), 0, 1)
    #     Q = A + t * v
    #     distance = ||P - Q||_2
    #
    # Select one course table per environment: ``(E, S, 2, 3)``.
    selected_segments = segment_table[course_indices]

    # Insert a singleton point axis so every point can be compared with every
    # segment. Both tensors have shape ``(E, 1, S, 3)`` and broadcast across P.
    starts = selected_segments[:, None, :, 0, :]
    vectors = selected_segments[:, None, :, 1, :] - starts

    # Insert a singleton segment axis into the points. Broadcasting against
    # ``starts`` produces point offsets of shape ``(E, P, S, 3)``.
    point_offsets = points[:, :, None, :] - starts

    # Squared segment lengths have shape ``(E, 1, S)``.
    # Clamping also keeps zero-filled padding from causing division by zero.
    squared_lengths = torch.sum(vectors.square(), dim=-1).clamp_min(torch.finfo(points.dtype).eps)

    # The normalized scalar projection t says where the perpendicular projection
    # of P falls along the segment's infinite supporting line: t = 0 is A, t = 1
    # is B, t < 0 lies before A, and t > 1 lies beyond B. Summing over the XYZ
    # axis computes each dot product and produces shape ``(E, P, S)``.
    projection = torch.sum(point_offsets * vectors, dim=-1) / squared_lengths

    # Restrict t to the finite segment. A projection before A therefore uses A
    # as its closest point, while one beyond B uses B.
    projection = projection.clamp(0.0, 1.0)

    # Reconstruct the closest point Q = A + t * v. ``[..., None]`` restores an
    # XYZ axis to t so it broadcasts over the three vector components. The
    # resulting closest points have shape ``(E, P, S, 3)``.
    closest_points = starts + projection[..., None] * vectors

    # Finally, ||P - Q||_2 is the Euclidean distance from each point to each
    # segment. Reducing the XYZ axis leaves shape ``(E, P, S)``.
    distances = torch.linalg.norm(
        points[:, :, None, :] - closest_points,
        dim=-1,
    )

    # The selected mask has shape ``(E, 1, S)`` and broadcasts across P.
    # Infinite padded distances cannot become minima.
    distances.masked_fill_(
        ~valid_segment_mask[course_indices, None, :],
        torch.inf,
    )
    return distances.amin(dim=-1)


def _shared_collinear_interval(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    other_start: tuple[float, float, float],
    other_end: tuple[float, float, float],
) -> tuple[float, float] | None:
    """Return the blocking interval of an opposing collinear 3D boundary.

    This recognizes both a coplanar seam and the exact shared line at a fold
    between differently oriented support surfaces.

    The returned values are signed metric distances from ``start`` along the
    candidate direction. They may lie outside the candidate; interval
    subtraction determines the positive-length overlap. ``None`` means the
    other boundary cannot conceal any part of the candidate.

    Args:
        start: First XYZ endpoint of the candidate boundary.
        end: Second XYZ endpoint of the candidate boundary.
        other_start: First XYZ endpoint of the boundary being compared.
        other_end: Second XYZ endpoint of the boundary being compared.

    Returns:
        The other boundary's ascending interval along the candidate line, or
        ``None`` when the boundaries cannot form an internal seam.
    """

    candidate = tuple(end[axis] - start[axis] for axis in range(3))
    other = tuple(other_end[axis] - other_start[axis] for axis in range(3))
    candidate_length = math.sqrt(sum(component * component for component in candidate))
    other_length = math.sqrt(sum(component * component for component in other))
    if candidate_length <= _GEOMETRY_TOLERANCE or other_length <= _GEOMETRY_TOLERANCE:
        return None

    unit_direction = tuple(component / candidate_length for component in candidate)

    def projection_and_distance(
        point: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Return one point's signed line coordinate and perpendicular distance.

        The first result is metric distance from candidate ``start`` along
        ``unit_direction`` and may be negative or exceed the candidate length.
        The second is the non-negative shortest distance to the corresponding
        infinite 3D line.
        """

        # Let ``r = point - start`` and let ``u`` be the candidate line's unit
        # direction. The dot product ``r dot u`` is the signed scalar
        # projection of ``r`` onto the line: its magnitude is the distance
        # along the line from ``start``, and its sign identifies whether the
        # projected point lies with or against the candidate direction.
        offset = tuple(point[axis] - start[axis] for axis in range(3))
        projection = sum(offset[axis] * unit_direction[axis] for axis in range(3))

        # Multiplying the scalar projection by ``u`` reconstructs the parallel
        # component of ``r``. Removing that component leaves
        # ``r_perpendicular = r - (r dot u) * u``, the shortest displacement
        # from the infinite candidate line to ``point``. Its Euclidean norm is
        # therefore the point-to-line distance. A distance near zero means the
        # point is collinear with the candidate boundary; ``projection`` may
        # still fall outside the finite segment and is handled separately.
        perpendicular = tuple(offset[axis] - projection * unit_direction[axis] for axis in range(3))
        distance = math.sqrt(sum(component * component for component in perpendicular))
        return projection, distance

    # 1. Project both endpoints onto the candidate line.
    other_start_projection, other_start_distance = projection_and_distance(other_start)
    other_end_projection, other_end_distance = projection_and_distance(other_end)

    # 2. Reject boundaries whose endpoints are offset from the candidate line.
    if other_start_distance > _GEOMETRY_TOLERANCE or other_end_distance > _GEOMETRY_TOLERANCE:
        return None

    # 3. Require the shared boundary to run in the opposite direction.
    #
    # Each support polygon stores its boundary in CCW order when viewed from
    # its upward-facing normal. When two such polygons meet along an internal
    # seam, their interiors lie on opposite sides of that seam, so their CCW
    # boundary walks traverse the shared line in opposite directions. The dot
    # product below measures the other edge's signed component along the
    # candidate direction: a negative value means opposing traversal and can
    # conceal a seam, whereas a non-negative value is rejected as
    # equal-direction geometry. This includes the candidate edge itself when
    # ``all_segments`` is compared against itself. Since the preceding checks
    # already require a nondegenerate collinear segment, zero is retained only
    # as a defensive numerical boundary between the two directions.
    if sum(other[axis] * unit_direction[axis] for axis in range(3)) >= 0.0:
        return None

    return tuple(sorted((other_start_projection, other_end_projection)))


def _subtract_interval(
    interval: tuple[float, float],
    blocker: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    """Subtract one closed scalar blocker from an ascending interval.

    Both pairs belong to the same one-dimensional candidate-line coordinate
    system, but ``blocker`` may extend outside ``interval``. Touching only at
    an endpoint, or overlapping by no more than the geometry tolerance, leaves
    the interval unchanged. Positive-length overlap can shorten the interval,
    split it into two ordered fragments, or remove it completely.

    A central blocker produces two fragments::

        interval:  [----------------]
        blocker:         [------]
        result:    [-----]      [----]

    Args:
        interval: Ascending exposed interval to retain where possible.
        blocker: Ascending interval occupied by an internal seam.

    Returns:
        Zero, one, or two ascending fragments in their original order.
    """

    start, end = interval
    overlap_start = max(start, blocker[0])
    overlap_end = min(end, blocker[1])
    if overlap_end - overlap_start <= _GEOMETRY_TOLERANCE:
        return (interval,)

    fragments: list[tuple[float, float]] = []
    if overlap_start - start > _GEOMETRY_TOLERANCE:
        fragments.append((start, overlap_start))
    if end - overlap_end > _GEOMETRY_TOLERANCE:
        fragments.append((overlap_end, end))
    return tuple(fragments)


def _support_edge_segments(
    level: ParkourLevelCfg,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    """Derive every exposed directed XYZ boundary for one course level.

    Boundary segments are collected from all ordered support polygons.
    Oppositely directed, collinear 3D boundaries describe either a coplanar
    internal seam or the exact fold where traversable surfaces meet. Their
    shared intervals are removed from both candidates because crossing them
    does not leave the union of traversable supports. Partial overlap may split
    one raw boundary into multiple exposed fragments.

    Args:
        level: Course level containing the support-region polygons.

    Returns:
        Directed exposed segments, each represented by two XYZ endpoints. The
        order follows the level's support regions, their CCW boundaries, and
        the surviving fragment order along each boundary.
    """

    raw_segments = tuple(
        (start, end) for region in level.support_regions for start, end in region.boundary_segments_xyz()
    )
    return tuple(
        fragment for start, end in raw_segments for fragment in _exposed_segment_fragments(start, end, raw_segments)
    )
