# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Grouped immutable course tables and mutable per-environment route state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv

    from ..curriculums.config import ParkourCurriculumCfg


@dataclass(frozen=True, slots=True)
class CourseTables:
    """Immutable device-side course data shared by all environments.

    Waypoint and support geometry is padded for vectorized indexing; the count
    tensors delimit valid entries. Milestone fractions are pre-normalized so
    routes with more control points cannot gain a larger reward budget.
    """

    # Course-matrix metadata.
    num_difficulties: int

    # Route geometry.
    cumulative_distances_m: torch.Tensor
    waypoint_counts: torch.Tensor
    waypoints: torch.Tensor

    # Reward metadata.
    milestone_reward_fractions: torch.Tensor

    # Support geometry.
    support_normals: torch.Tensor
    support_vertex_counts: torch.Tensor
    support_vertices: torch.Tensor

    # Course commands.
    min_clearances: torch.Tensor
    target_speeds: torch.Tensor


@dataclass(slots=True)
class RouteState:
    """Mutable episode state with one row per parallel environment.

    ``course_indices`` retains the completed episode's course until reset, so
    curriculum progress is normalized against the right route. The two progress
    tensors distinguish the running maximum from the post-reset evaluation
    snapshot. ``previous_root_xy`` bridges manager ordering for plane crossings;
    the Boolean event tensors expose one-step retarget and completion events.
    """

    # Selected course and active cursor. The selected course determines the
    # valid cursor range.
    course_indices: torch.Tensor
    active_waypoint_indices: torch.Tensor

    # Cross-step crossing and progress history.
    previous_root_xy: torch.Tensor
    maximum_progress_m: torch.Tensor

    # Completed-episode evaluation snapshot.
    previous_episode_maximum_progress_m: torch.Tensor

    # One-step route-transition events.
    waypoint_changed: torch.Tensor
    course_completed: torch.Tensor


@dataclass(frozen=True, slots=True)
class ParkourRuntime:
    """All navigation-owned course data and live route state."""

    courses: CourseTables
    route: RouteState


# Runtime access and initialization.


def _ensure_parkour_runtime(
    env: ManagerBasedRLEnv,
    curriculum_cfg: ParkourCurriculumCfg,
    *,
    dtype: torch.dtype,
) -> ParkourRuntime:
    """Build the grouped runtime once and rebuild it only for a stale layout."""

    runtime = _parkour_runtime_or_none(env)
    if runtime is not None and _runtime_matches(
        runtime,
        env,
        curriculum_cfg,
        dtype=dtype,
    ):
        return runtime

    courses = _build_course_tables(
        env,
        curriculum_cfg,
        dtype=dtype,
    )
    route = _build_route_state(env, dtype=dtype)
    runtime = ParkourRuntime(courses=courses, route=route)

    # Assign the sole environment-level navigation attribute last. A failed
    # table build therefore leaves no partially initialized runtime behind.
    env._parkour_runtime = runtime
    return runtime


def _parkour_runtime(env: ManagerBasedRLEnv) -> ParkourRuntime:
    """Return initialized navigation state or fail at the ownership boundary."""

    runtime = _parkour_runtime_or_none(env)
    if runtime is None:
        raise RuntimeError("Active routes must be initialized before use.")
    return runtime


def _parkour_runtime_or_none(env: ManagerBasedRLEnv) -> ParkourRuntime | None:
    """Return initialized navigation state without creating environment state."""

    runtime = getattr(env, "_parkour_runtime", None)
    return runtime if isinstance(runtime, ParkourRuntime) else None


# Runtime construction.


def _build_course_tables(
    env: ManagerBasedRLEnv,
    curriculum_cfg: ParkourCurriculumCfg,
    *,
    dtype: torch.dtype,
) -> CourseTables:
    """Build rectangular vectorized tables from the validated course matrix."""

    import torch

    courses = curriculum_cfg.courses
    routes = tuple(
        tuple(waypoint.position for waypoint in course.waypoints) for course in courses
    )
    max_waypoints = max(len(route) for route in routes)
    max_support_vertices = max(
        len(region.vertices) for course in courses for region in course.support_regions
    )

    cumulative_distances_m = torch.zeros(
        (len(routes), max_waypoints),
        device=env.device,
        dtype=dtype,
    )
    waypoint_counts = torch.empty(
        len(routes),
        device=env.device,
        dtype=torch.long,
    )
    waypoints = torch.empty(
        (len(routes), max_waypoints, 3),
        device=env.device,
        dtype=dtype,
    )
    milestone_reward_fractions = torch.zeros(
        (len(routes), max_waypoints),
        device=env.device,
        dtype=dtype,
    )
    support_normals = torch.zeros(
        (len(routes), max_waypoints, 3),
        device=env.device,
        dtype=dtype,
    )
    support_vertex_counts = torch.zeros(
        (len(routes), max_waypoints),
        device=env.device,
        dtype=torch.long,
    )
    support_vertices = torch.zeros(
        (len(routes), max_waypoints, max_support_vertices, 3),
        device=env.device,
        dtype=dtype,
    )

    for course_index, (course, route) in enumerate(zip(courses, routes)):
        route_tensor = torch.tensor(route, device=env.device, dtype=dtype)
        waypoint_count = route_tensor.shape[0]
        waypoints[course_index, :waypoint_count] = route_tensor
        # Repeat the final target in padded rows. ``waypoint_counts`` remains
        # authoritative, so a route cursor cannot intentionally enter them.
        waypoints[course_index, waypoint_count:] = route_tensor[-1]
        waypoint_counts[course_index] = waypoint_count

        rewarded_milestones = tuple(
            waypoint.is_rewarded_milestone for waypoint in course.waypoints[:-1]
        )
        rewarded_milestone_count = sum(rewarded_milestones)
        if rewarded_milestone_count > 0:
            milestone_reward_fractions[
                course_index,
                : waypoint_count - 1,
            ] = torch.tensor(
                rewarded_milestones,
                device=env.device,
                dtype=dtype,
            ) / float(rewarded_milestone_count)

        support_by_name = {region.name: region for region in course.support_regions}
        for waypoint_index, waypoint in enumerate(course.waypoints):
            support_name = waypoint.support_region_name
            if support_name is None:
                continue

            support = support_by_name[support_name]
            vertices = torch.tensor(
                support.vertices,
                device=env.device,
                dtype=dtype,
            )
            vertex_count = vertices.shape[0]
            # Repeating vertex zero in the padded suffix makes the last valid
            # edge close back to the polygon start.
            support_vertices[course_index, waypoint_index, :] = vertices[0]
            support_vertices[
                course_index,
                waypoint_index,
                :vertex_count,
            ] = vertices
            support_vertex_counts[course_index, waypoint_index] = vertex_count
            support_normals[course_index, waypoint_index] = torch.tensor(
                support.normal,
                device=env.device,
                dtype=dtype,
            )

        previous_xy = torch.zeros(2, device=env.device, dtype=dtype)
        distance = torch.zeros((), device=env.device, dtype=dtype)
        for waypoint_index, position in enumerate(route):
            waypoint_xy = torch.tensor(
                position[:2],
                device=env.device,
                dtype=dtype,
            )
            distance = distance + torch.linalg.norm(waypoint_xy - previous_xy)
            cumulative_distances_m[course_index, waypoint_index] = distance
            previous_xy = waypoint_xy
        cumulative_distances_m[course_index, waypoint_count:] = distance

    return CourseTables(
        num_difficulties=curriculum_cfg.num_difficulties,
        cumulative_distances_m=cumulative_distances_m,
        waypoint_counts=waypoint_counts,
        waypoints=waypoints,
        milestone_reward_fractions=milestone_reward_fractions,
        support_normals=support_normals,
        support_vertex_counts=support_vertex_counts,
        support_vertices=support_vertices,
        min_clearances=torch.tensor(
            [course.min_clearance for course in courses],
            device=env.device,
            dtype=dtype,
        ),
        target_speeds=torch.tensor(
            [course.target_speed for course in courses],
            device=env.device,
            dtype=dtype,
        ),
    )


def _build_route_state(
    env: ManagerBasedRLEnv,
    *,
    dtype: torch.dtype,
) -> RouteState:
    """Allocate fresh mutable route state for every environment."""

    import torch

    integer_zeros = torch.zeros(
        env.num_envs,
        device=env.device,
        dtype=torch.long,
    )
    floating_zeros = torch.zeros(
        env.num_envs,
        device=env.device,
        dtype=dtype,
    )
    return RouteState(
        course_indices=integer_zeros,
        active_waypoint_indices=integer_zeros.clone(),
        previous_root_xy=torch.zeros(
            (env.num_envs, 2),
            device=env.device,
            dtype=dtype,
        ),
        maximum_progress_m=floating_zeros,
        previous_episode_maximum_progress_m=floating_zeros.clone(),
        waypoint_changed=torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        ),
        course_completed=torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.bool,
        ),
    )


# Runtime validation.


def _runtime_matches(
    runtime: ParkourRuntime,
    env: ManagerBasedRLEnv,
    curriculum_cfg: ParkourCurriculumCfg,
    *,
    dtype: torch.dtype,
) -> bool:
    """Return whether an existing runtime has the requested tensor layout."""

    import torch

    courses = runtime.courses
    route = runtime.route
    device = torch.device(env.device)
    waypoint_changed = getattr(route, "waypoint_changed", None)
    course_completed = getattr(route, "course_completed", None)
    return (
        courses.num_difficulties == curriculum_cfg.num_difficulties
        and courses.waypoints.shape[0] == len(curriculum_cfg.courses)
        and courses.waypoints.device == device
        and courses.waypoints.dtype == dtype
        and route.course_indices.shape == (env.num_envs,)
        and route.course_indices.device == device
        and route.previous_root_xy.shape == (env.num_envs, 2)
        and route.previous_root_xy.device == device
        and route.previous_root_xy.dtype == dtype
        and isinstance(waypoint_changed, torch.Tensor)
        and waypoint_changed.shape == (env.num_envs,)
        and waypoint_changed.device == device
        and waypoint_changed.dtype == torch.bool
        and isinstance(course_completed, torch.Tensor)
        and course_completed.shape == (env.num_envs,)
        and course_completed.device == device
        and course_completed.dtype == torch.bool
    )
