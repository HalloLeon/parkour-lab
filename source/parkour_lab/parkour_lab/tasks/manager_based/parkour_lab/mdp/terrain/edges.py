# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Metric support-edge geometry and contact-gated runtime queries."""

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
    """Device-side edge geometry derived from one curriculum configuration."""

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
        """Return whether this cache can serve the requested runtime tensors."""

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
    """Return contacted feet near exposed edges of their current support.

    Returns:
        Boolean tensor with shape ``(num_envs, num_feet)``.
    """

    import torch

    from .._shared import contact, robot, runtime

    levels = getattr(env, "_parkour_waypoint_level", None)
    if levels is None or levels.shape != (env.num_envs,):
        raise RuntimeError(
            "Active course levels must be initialized before evaluating edges."
        )

    foot_positions = robot._selected_body_pos_env(env, asset_cfg)

    # Reduce the history axis, retaining one recent-contact flag per foot.
    recent_contact = torch.any(
        contact._force_norm_mask(env, sensor_cfg=sensor_cfg)
        > curriculum_cfg.foot_edge_contact_threshold,
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
    edge_distance = _minimum_distance_to_level_edges(
        foot_positions,
        edge_cache.segment_table,
        edge_cache.valid_segment_mask,
        levels,
    )
    return recent_contact & (edge_distance <= curriculum_cfg.edge_width_threshold)


def _exposed_segment_fragments(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    all_segments: tuple[
        tuple[tuple[float, float, float], tuple[float, float, float]], ...
    ],
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    """Remove internal seams from one segment and return its exposed pieces."""

    # Axis-aligned horizontal segments have equal start and end Y coordinates.
    # Use an absolute tolerance because course geometry is represented by
    # floating-point values.
    horizontal = math.isclose(
        start[1],
        end[1],
        rel_tol=0.0,
        abs_tol=_GEOMETRY_TOLERANCE,
    )

    # Coordinate index 0 is X and index 1 is Y. X varies along a horizontal
    # segment, whereas Y varies along a vertical segment.
    varying_axis = 0 if horizontal else 1

    # Record whether the directed boundary travels in the positive or negative
    # direction so reconstructed fragments preserve the original orientation.
    direction = 1 if end[varying_axis] > start[varying_axis] else -1

    # Begin with the segment's complete one-dimensional interval in ascending
    # order. The outer tuple holds the remaining pieces because removing
    # internal seams may split this interval into multiple fragments.
    fragments = (
        (
            min(start[varying_axis], end[varying_axis]),
            max(start[varying_axis], end[varying_axis]),
        ),
    )

    # Compare this boundary with every course boundary. An oppositely directed
    # segment on the same line and support height belongs to the neighboring
    # side of a coplanar region, so their overlap is an internal seam.
    for other_start, other_end in all_segments:
        blocker = _shared_coplanar_interval(
            start,
            end,
            other_start,
            other_end,
        )
        # Different axes, lines, heights, or directions cannot hide any part
        # of this boundary and therefore leave its fragments unchanged.
        if blocker is None:
            continue

        # Remove the shared one-dimensional interval from every piece that is
        # still exposed. One blocker may shorten a piece, split it in two, or
        # remove it completely; the flattened tuple becomes the input for the
        # next comparison and supports multiple partial neighboring seams.
        fragments = tuple(
            remaining
            for fragment in fragments
            for remaining in _subtract_interval(fragment, blocker)
        )

    exposed_segments: list[
        tuple[tuple[float, float, float], tuple[float, float, float]]
    ] = []
    for fragment_start, fragment_end in fragments:
        first = fragment_start if direction > 0 else fragment_end
        second = fragment_end if direction > 0 else fragment_start
        if horizontal:
            exposed_segments.append(
                (
                    (first, start[1], start[2]),
                    (second, start[1], start[2]),
                )
            )
        else:
            exposed_segments.append(
                (
                    (start[0], first, start[2]),
                    (start[0], second, start[2]),
                )
            )

    return tuple(exposed_segments)


def _get_support_edge_cache(
    env: ManagerBasedRLEnv,
    curriculum_cfg: ParkourCurriculumCfg,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _SupportEdgeCache:
    """Return cached, padded edge geometry on the requested device."""

    import torch

    cache = getattr(env, "_parkour_support_edge_cache", None)
    if isinstance(cache, _SupportEdgeCache) and cache.matches(
        curriculum_cfg,
        device=device,
        dtype=dtype,
    ):
        return cache

    # Each level may contain a different number of exposed edges. Every edge
    # consists of two XYZ endpoints, where Z identifies its support surface.
    segments_by_level = tuple(
        _support_edge_segments(level) for level in curriculum_cfg.levels
    )
    max_segments = max(len(segments) for segments in segments_by_level)

    # Shape: ``(num_levels, max_segments, 2, 3)``. Shorter levels retain
    # zero-filled padding in their unused segment rows.
    segment_table = torch.zeros(
        (len(segments_by_level), max_segments, 2, 3),
        device=device,
        dtype=dtype,
    )

    # Shape: ``(num_levels, max_segments)``. True entries distinguish real
    # segments from the padded rows that the distance kernel must ignore.
    valid_segment_mask = torch.zeros(
        (len(segments_by_level), max_segments),
        device=device,
        dtype=torch.bool,
    )
    for level_index, segments in enumerate(segments_by_level):
        segment_count = len(segments)
        segment_table[level_index, :segment_count] = torch.tensor(
            segments,
            device=device,
            dtype=dtype,
        )
        valid_segment_mask[level_index, :segment_count] = True

    cache = _SupportEdgeCache(
        curriculum_cfg=curriculum_cfg,
        segment_table=segment_table,
        valid_segment_mask=valid_segment_mask,
    )
    env._parkour_support_edge_cache = cache
    return cache


def _minimum_distance_to_level_edges(
    points: torch.Tensor,
    segment_table: torch.Tensor,
    valid_segment_mask: torch.Tensor,
    levels: torch.Tensor,
) -> torch.Tensor:
    """Measure 3D points against exposed edges for their environment level.

    Including height prevents an edge on one support surface from penalizing a
    foot contacting another surface at the same XY location.

    Args:
        points: Query positions with shape ``(num_envs, num_points, 3)``.
        segment_table: Padded edge endpoints with shape
            ``(num_levels, max_segments, 2, 3)``.
        valid_segment_mask: Valid rows of ``segment_table`` with shape
            ``(num_levels, max_segments)``.
        levels: Level index for each environment with shape ``(num_envs,)``.

    Returns:
        Minimum distance for every query point, with shape
        ``(num_envs, num_points)``.
    """

    import torch

    # Select one level table per environment:
    # ``(num_envs, max_segments, 2, 3)``.
    selected_segments = segment_table[levels]

    # Insert a singleton point axis so every point can be compared with every
    # segment. Both tensors have shape ``(num_envs, 1, max_segments, 3)``.
    starts = selected_segments[:, None, :, 0, :]
    vectors = selected_segments[:, None, :, 1, :] - starts

    # Insert a singleton segment axis into the points. Broadcasting against
    # ``starts`` produces offsets of shape
    # ``(num_envs, num_points, max_segments, 3)``.
    point_offsets = points[:, :, None, :] - starts

    # Squared segment lengths have shape ``(num_envs, 1, max_segments)``.
    # Clamping also keeps zero-filled padding from causing division by zero.
    squared_lengths = torch.sum(vectors.square(), dim=-1).clamp_min(
        torch.finfo(points.dtype).eps
    )

    # For a point P and segment from A to B, ``vectors`` is v = B - A and
    # ``point_offsets`` is w = P - A. The normalized scalar projection
    #
    #     t = dot(w, v) / dot(v, v)
    #
    # says where the perpendicular projection of P falls along the segment's
    # infinite supporting line: t = 0 is A, t = 1 is B, t < 0 lies before A,
    # and t > 1 lies beyond B. Summing over the XYZ axis computes each dot
    # product and produces shape ``(num_envs, num_points, max_segments)``.
    projection = torch.sum(point_offsets * vectors, dim=-1) / squared_lengths

    # Restrict t to the finite segment. A projection before A therefore uses A
    # as its closest point, while one beyond B uses B.
    projection = projection.clamp(0.0, 1.0)

    # Reconstruct the closest point Q = A + t * v. ``[..., None]`` restores an
    # XYZ axis to t so it broadcasts over the three vector components. The
    # resulting closest points have shape
    # ``(num_envs, num_points, max_segments, 3)``.
    closest_points = starts + projection[..., None] * vectors

    # Finally, ||P - Q||_2 is the Euclidean distance from each point to each
    # segment. Reducing the XYZ axis leaves shape
    # ``(num_envs, num_points, max_segments)``.
    distances = torch.linalg.norm(
        points[:, :, None, :] - closest_points,
        dim=-1,
    )

    # The selected mask has shape ``(num_envs, 1, max_segments)`` and
    # broadcasts across points. Infinite padded distances cannot become minima.
    distances.masked_fill_(~valid_segment_mask[levels, None, :], torch.inf)
    return distances.amin(dim=-1)


def _shared_coplanar_interval(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    other_start: tuple[float, float, float],
    other_end: tuple[float, float, float],
) -> tuple[float, float] | None:
    """Return the blocking interval of an opposing coplanar boundary.

    Two axis-aligned boundaries can describe the two sides of an internal seam
    only when they are both horizontal or both vertical, lie on the same XY
    line, have the same support-surface height, and point in opposite
    directions. The opposite direction follows from the consistent winding of
    neighboring support rectangles.

    When those conditions hold, the returned pair contains ``other_start`` and
    ``other_end`` projected onto the segment's varying axis and sorted in
    ascending order. It may extend beyond ``start`` and ``end``; the caller's
    interval-subtraction step computes their actual overlap. ``None`` means the
    other boundary cannot hide any part of the candidate as an internal seam.

    Args:
        start: First XYZ endpoint of the candidate boundary.
        end: Second XYZ endpoint of the candidate boundary.
        other_start: First XYZ endpoint of the boundary being compared.
        other_end: Second XYZ endpoint of the boundary being compared.

    Returns:
        The other boundary's ascending interval along the varying axis, or
        ``None`` when the boundaries cannot form an internal seam.
    """

    horizontal = math.isclose(
        start[1],
        end[1],
        rel_tol=0.0,
        abs_tol=_GEOMETRY_TOLERANCE,
    )
    other_horizontal = math.isclose(
        other_start[1],
        other_end[1],
        rel_tol=0.0,
        abs_tol=_GEOMETRY_TOLERANCE,
    )
    if horizontal != other_horizontal:
        return None

    # Coordinate index 0 is X and index 1 is Y. X varies along a horizontal
    # segment, whereas Y varies along a vertical segment.
    varying_axis = 0 if horizontal else 1
    constant_axis = 1 - varying_axis
    same_line = math.isclose(
        start[constant_axis],
        other_start[constant_axis],
        rel_tol=0.0,
        abs_tol=_GEOMETRY_TOLERANCE,
    )
    same_height = math.isclose(
        start[2],
        other_start[2],
        rel_tol=0.0,
        abs_tol=_GEOMETRY_TOLERANCE,
    )

    if not same_line or not same_height:
        return None

    direction = 1 if end[varying_axis] > start[varying_axis] else -1
    other_direction = 1 if other_end[varying_axis] > other_start[varying_axis] else -1
    # On a shared boundary, the right edge of the region on the left points
    # upward while the left edge of the region on the right points downward.
    # Opposite directions therefore identify the two sides of an internal
    # seam. Equal directions do not; this also ignores the segment when it is
    # compared with itself in ``all_segments``.
    if direction == other_direction:
        return None

    return (
        min(other_start[varying_axis], other_end[varying_axis]),
        max(other_start[varying_axis], other_end[varying_axis]),
    )


def _subtract_interval(
    interval: tuple[float, float],
    blocker: tuple[float, float],
) -> tuple[tuple[float, float], ...]:
    """Remove the positive-length overlap with ``blocker`` from an interval."""

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
    """Return exposed XYZ support boundaries for one course level.

    Oppositely directed, collinear boundaries at the same height describe an
    internal seam between coplanar support regions. Their shared interval is
    removed because traversing that seam does not risk stepping off a surface.
    """

    raw_segments = tuple(
        (
            (start[0], start[1], region.surface_z),
            (end[0], end[1], region.surface_z),
        )
        for region in level.support_regions
        for start, end in region.boundary_segments_xy()
    )
    return tuple(
        fragment
        for start, end in raw_segments
        for fragment in _exposed_segment_fragments(start, end, raw_segments)
    )
