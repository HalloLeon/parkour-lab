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
_Interval = tuple[float, float]
_Point3 = tuple[float, float, float]
_Segment3 = tuple[_Point3, _Point3]


@dataclass(frozen=True, slots=True)
class _SupportEdgeCache:
    """Padded device-side edge geometry shared by one environment."""

    # XYZ endpoints with shape [course, padded segment, start/end, coordinate].
    segment_table: torch.Tensor

    # Shape [course, padded segment]; True marks a real segment rather than padding.
    valid_segment_mask: torch.Tensor


def foot_edge_contact_mask(
    env: ManagerBasedRLEnv,
    *,
    curriculum_cfg: ParkourCurriculumCfg,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return recently contacting feet within the configured exposed-edge band."""

    import torch

    from .._shared import contact, robot, runtime
    from ..navigation import route

    foot_positions = robot._selected_body_pos_env(env, asset_cfg)
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
    edge_distance = _minimum_distance_to_course_edges(
        foot_positions,
        edge_cache.segment_table,
        edge_cache.valid_segment_mask,
        route.active_course_indices(env),
    )
    return recent_contact & (edge_distance <= curriculum_cfg.edge_width_threshold)


# Runtime device cache and distance query.


def _get_support_edge_cache(
    env: ManagerBasedRLEnv,
    curriculum_cfg: ParkourCurriculumCfg,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> _SupportEdgeCache:
    """Build the environment's padded course-edge tensors on first use."""

    import torch

    cache = getattr(env, "_parkour_support_edge_cache", None)
    if isinstance(cache, _SupportEdgeCache):
        return cache

    segments_by_course = tuple(
        _support_edge_segments(course) for course in curriculum_cfg.courses
    )
    max_segments = max(len(segments) for segments in segments_by_course)
    segment_table = torch.zeros(
        (len(segments_by_course), max_segments, 2, 3),
        device=device,
        dtype=dtype,
    )
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
    """Return each 3D point's distance to its course's nearest exposed edge."""

    import torch

    # E = environments, P = points, S = padded segments. For segment A->B,
    # t=clamp(dot(P-A, B-A)/||B-A||^2, 0, 1) and d=||(P-A)-t(B-A)||.
    selected_segments = segment_table[course_indices]
    starts = selected_segments[:, None, :, 0, :]
    vectors = selected_segments[:, None, :, 1, :] - starts
    offsets = points[:, :, None, :] - starts
    squared_lengths = torch.sum(vectors.square(), dim=-1).clamp_min(
        torch.finfo(points.dtype).eps
    )
    projection = (torch.sum(offsets * vectors, dim=-1) / squared_lengths).clamp(
        0.0, 1.0
    )
    distances = torch.linalg.norm(offsets - projection[..., None] * vectors, dim=-1)
    distances.masked_fill_(
        ~valid_segment_mask[course_indices, None, :],
        torch.inf,
    )
    return distances.amin(dim=-1)


# Dependency-free exposed-edge geometry.


def _exposed_segment_fragments(
    start: _Point3,
    end: _Point3,
    all_segments: tuple[_Segment3, ...],
) -> tuple[_Segment3, ...]:
    """Remove opposing collinear seams from one directed support boundary."""

    vector = tuple(end[axis] - start[axis] for axis in range(3))
    length = math.sqrt(sum(component * component for component in vector))
    if length <= _GEOMETRY_TOLERANCE:
        return ()
    unit_direction = tuple(component / length for component in vector)
    fragments: tuple[_Interval, ...] = ((0.0, length),)

    for other_start, other_end in all_segments:
        blocker = _shared_collinear_interval(
            start, unit_direction, other_start, other_end
        )
        if blocker is not None:
            fragments = tuple(
                remaining
                for fragment in fragments
                for remaining in _subtract_interval(fragment, blocker)
            )

    def point_at(distance: float) -> _Point3:
        if math.isclose(distance, 0.0, abs_tol=_GEOMETRY_TOLERANCE):
            return start
        if math.isclose(distance, length, abs_tol=_GEOMETRY_TOLERANCE):
            return end
        return tuple(start[axis] + distance * unit_direction[axis] for axis in range(3))

    return tuple((point_at(first), point_at(last)) for first, last in fragments)


def _projection_coefficient(vector: _Point3, direction: _Point3) -> float:
    """Return t = dot(v, d) / dot(d, d), where proj_d(v) = t * d."""

    squared_length = sum(component * component for component in direction)
    if squared_length <= _GEOMETRY_TOLERANCE**2:
        raise ValueError("Cannot project onto a zero-length direction.")
    return sum(vector[axis] * direction[axis] for axis in range(3)) / squared_length


def _shared_collinear_interval(
    start: _Point3,
    unit_direction: _Point3,
    other_start: _Point3,
    other_end: _Point3,
) -> _Interval | None:
    """Return an opposing collinear segment's interval along a candidate line.

    Args:
        start: XYZ origin of the candidate segment's line.
        unit_direction: Candidate segment direction, normalized to unit length.
        other_start: XYZ start of the segment being compared.
        other_end: XYZ end of the segment being compared.

    Returns:
        The compared segment's signed distances along the candidate line,
        measured from ``start``, or ``None`` if it is not oppositely directed
        and collinear.
    """

    other = tuple(other_end[axis] - other_start[axis] for axis in range(3))
    # CCW boundaries on adjacent support faces traverse an internal seam in
    # opposite directions; equal-direction and degenerate segments stay exposed.
    if _projection_coefficient(other, unit_direction) >= 0.0:
        return None

    projection_coefficients = []
    tolerance_squared = _GEOMETRY_TOLERANCE**2
    for point in (other_start, other_end):
        offset = tuple(point[axis] - start[axis] for axis in range(3))
        coefficient = _projection_coefficient(offset, unit_direction)
        perpendicular_squared = sum(
            (offset[axis] - coefficient * unit_direction[axis]) ** 2
            for axis in range(3)
        )
        if perpendicular_squared > tolerance_squared:
            return None
        projection_coefficients.append(coefficient)
    return (min(projection_coefficients), max(projection_coefficients))


def _subtract_interval(
    interval: _Interval, blocker: _Interval
) -> tuple[_Interval, ...]:
    """Subtract one closed blocker from an ascending scalar interval."""

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
) -> tuple[_Segment3, ...]:
    """Return exposed boundaries after removing shared seams between supports."""

    raw_segments = tuple(
        (start, end)
        for region in level.support_regions
        for start, end in region.boundary_segments_xyz()
    )
    return tuple(
        fragment
        for start, end in raw_segments
        for fragment in _exposed_segment_fragments(start, end, raw_segments)
    )
