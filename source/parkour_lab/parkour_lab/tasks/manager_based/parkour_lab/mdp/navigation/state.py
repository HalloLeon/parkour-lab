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

    # Family-variant-difficulty matrix metadata.
    # Number of difficulty rows per family/variant ladder. Runtime queries use
    # it to recover a difficulty row from a flattened course index.
    num_difficulties: int

    # Route geometry.
    # [course, waypoint]: XY route length from the tile-local origin through
    # this waypoint. The padded suffix repeats the complete course length.
    cumulative_distances_m: torch.Tensor
    # [course]: number of valid waypoints before padding; bounds the route
    # cursor and identifies the final target.
    waypoint_counts: torch.Tensor
    # [course, waypoint, xyz]: terrain-local target positions. Padded entries
    # repeat the final waypoint to keep vectorized indexing safe.
    waypoints: torch.Tensor
    # [course, waypoint]: maximum root-to-target XY distance accepted for each
    # waypoint, after resolving per-waypoint overrides and the global default.
    root_reach_radii: torch.Tensor
    # [course, waypoint]: marks targets declared as terminal landings so speed
    # shaping can distinguish a landing target from an ordinary final exit.
    terminal_landing_masks: torch.Tensor

    # Reward metadata.
    # [course, waypoint]: normalized one-shot milestone credit. Rewarded
    # intermediate waypoints share a total budget of one; all others are zero.
    milestone_reward_fractions: torch.Tensor

    # Support geometry.
    # [course, waypoint, xyz]: normal of the required support plane, or zero
    # when the waypoint has no support-contact requirement.
    support_normals: torch.Tensor
    # [course, waypoint]: number of valid polygon vertices. Zero means that
    # route advancement does not require contact with a named support.
    support_vertex_counts: torch.Tensor
    # [course, waypoint, vertex, xyz]: terrain-local support polygons used for
    # contact containment and edge queries; padded suffixes repeat vertex zero.
    support_vertices: torch.Tensor

    # Course constraints.
    # [course]: minimum terrain-relative base clearance used by safety shaping
    # and the stable final-waypoint completion gate.
    min_clearances: torch.Tensor


@dataclass(slots=True)
class RouteState:
    """Mutable episode state with one row per parallel environment.

    ``course_indices`` retains the completed episode's course until reset, so
    curriculum progress is normalized against the right route. The two progress
    tensors distinguish the running maximum from the post-reset evaluation
    snapshot. The Boolean event tensors expose one-step retarget and completion
    events.
    """

    # Selected course and active cursor. The selected course determines the
    # valid cursor range.
    # [environment]: flattened family-variant-difficulty course-table row. It
    # remains unchanged at termination until reset consumes episode metrics.
    course_indices: torch.Tensor
    # [environment]: index of the waypoint currently targeted in that course.
    active_waypoint_indices: torch.Tensor

    # Progress and terminal-stability history.
    # [environment]: furthest safe route-projected distance reached in the
    # current episode; monotonic and not advanced during chassis contact.
    maximum_progress_m: torch.Tensor
    # [environment]: uninterrupted time satisfying the terminal-landing support
    # and whole-body stability gate. Non-terminal targets keep this at zero.
    terminal_landing_stable_time_s: torch.Tensor

    # Completed-episode evaluation snapshot.
    # [environment]: previous value of ``maximum_progress_m``, copied during
    # reset before the current episode accumulator is cleared.
    previous_episode_maximum_progress_m: torch.Tensor

    # One-step route-transition events.
    # [environment]: true on the step that the active cursor advances. Rewards
    # use it for milestone credit and to ignore the retargeting distance jump.
    waypoint_changed: torch.Tensor
    # [environment]: true on the step that the final waypoint passes all
    # configured support, clearance, contact, pose, and motion-stability gates.
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
    """Build the environment-owned navigation runtime on first use."""

    runtime = _parkour_runtime_or_none(env)
    if runtime is not None:
        return runtime

    courses = _build_course_tables(
        env,
        curriculum_cfg,
        dtype=dtype,
    )
    route = _build_route_state(
        env,
        dtype=dtype,
    )
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
    root_reach_radii = torch.empty(
        (len(routes), max_waypoints),
        device=env.device,
        dtype=dtype,
    )
    terminal_landing_masks = torch.zeros(
        (len(routes), max_waypoints),
        device=env.device,
        dtype=torch.bool,
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

        course_root_reach_radii = torch.tensor(
            [
                (
                    waypoint.root_reach_radius
                    if waypoint.root_reach_radius is not None
                    else curriculum_cfg.waypoint_reach_radius_m
                )
                for waypoint in course.waypoints
            ],
            device=env.device,
            dtype=dtype,
        )
        root_reach_radii[course_index, :waypoint_count] = course_root_reach_radii
        root_reach_radii[course_index, waypoint_count:] = course_root_reach_radii[-1]
        terminal_landing_masks[course_index, :waypoint_count] = torch.tensor(
            [waypoint.is_terminal_landing for waypoint in course.waypoints],
            device=env.device,
            dtype=torch.bool,
        )
        terminal_landing_masks[course_index, waypoint_count:] = course.waypoints[
            -1
        ].is_terminal_landing

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
        root_reach_radii=root_reach_radii,
        terminal_landing_masks=terminal_landing_masks,
        milestone_reward_fractions=milestone_reward_fractions,
        support_normals=support_normals,
        support_vertex_counts=support_vertex_counts,
        support_vertices=support_vertices,
        min_clearances=torch.tensor(
            [course.min_clearance for course in courses],
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
        maximum_progress_m=floating_zeros,
        terminal_landing_stable_time_s=floating_zeros.clone(),
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
