# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-environment ordered-route state and waypoint-marker synchronization.

Simulator dependencies stay inside the runtime functions so the pure route
transition can also be exercised by dependency-light NumPy tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

    from ..curriculums.config import ParkourCurriculumCfg


def active_waypoint_changed_this_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return which environments switched targets during the current step.

    The success termination term updates this event before rewards are
    evaluated. It therefore remains valid for the progress reward during the
    same step and is overwritten by the next route update.
    """

    import torch

    changed = getattr(env, "_parkour_active_waypoint_changed", None)
    if changed is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return changed


def active_waypoint_positions(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Derive active terrain-local waypoints, falling back before first reset."""

    table = getattr(env, "_parkour_waypoint_table", None)
    course_indices = getattr(env, "_parkour_course_index", None)
    active_indices = getattr(env, "_parkour_active_waypoint_index", None)
    if (
        table is not None
        and course_indices is not None
        and active_indices is not None
        and course_indices.shape == (env.num_envs,)
        and active_indices.shape == (env.num_envs,)
    ):
        return table[course_indices, active_indices]

    waypoint_marker = env.scene[waypoint_marker_cfg.name]
    return waypoint_marker.data.root_pos_w - env.scene.env_origins


def advance_active_waypoints(
    env: ManagerBasedRLEnv,
    *,
    reach_threshold: float,
    waypoint_marker_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Update every route once and return final-course completions.

    Isaac Lab evaluates termination terms before rewards and observations. This
    function is therefore called by the success termination term so every later
    consumer sees the newly selected waypoint during the same control step.
    """

    import torch

    from .._shared import robot
    from ..commands import get_min_clearance
    from ..terrain import queries

    if reach_threshold <= 0.0:
        raise ValueError("reach_threshold must be positive.")
    required_state = (
        "_parkour_waypoint_table",  # Terrain-local XYZ waypoints for every course.
        "_parkour_waypoint_count_by_course",  # Unpadded waypoint count per course.
        "_parkour_course_index",  # Selected course for each environment.
        "_parkour_active_waypoint_index",  # Current target waypoint per environment.
    )
    if not all(hasattr(env, name) for name in required_state):
        raise RuntimeError("Active waypoints must be initialized before stepping.")

    active_indices = env._parkour_active_waypoint_index
    course_indices = env._parkour_course_index
    waypoint_counts = env._parkour_waypoint_count_by_course[course_indices]
    active_positions = env._parkour_waypoint_table[course_indices, active_indices]
    robot_pos = robot._root_pos_env(env, asset_cfg)
    distance_xy = torch.linalg.norm(
        robot_pos[:, :2] - active_positions[:, :2],
        dim=-1,
    )
    within_radius = distance_xy < reach_threshold
    passed_waypoint_plane = _active_waypoint_plane_passed(
        env,
        robot_pos[:, :2],
        lateral_tolerance=reach_threshold,
    )

    # Intermediate waypoints only select a new direction. The final waypoint
    # retains the existing safety rule that a collapsed robot is not successful.
    clearance = queries._base_clearance(env, asset_cfg)
    min_clearance = get_min_clearance(env).to(
        device=clearance.device,
        dtype=clearance.dtype,
    )
    final_waypoint_eligible = clearance > min_clearance

    progress = _route_progress_m(env, robot_pos[:, :2])
    # Progress can decrease if the robot backtracks. Preserve the furthest point
    # reached so curriculum decisions and evaluation summarize its best advance.
    env._parkour_max_course_progress_m[:] = torch.maximum(
        env._parkour_max_course_progress_m,
        progress,
    )

    next_indices, completed_course = _advance_route_state(
        active_indices,
        waypoint_counts,
        within_radius,
        passed_waypoint_plane,
        final_waypoint_eligible,
    )

    advanced = next_indices != active_indices
    env._parkour_active_waypoint_index[:] = next_indices
    # Retargeting replaces the nearby reached waypoint with the farther next one,
    # making active-waypoint distance jump without robot motion. The progress
    # reward uses this event to ignore that artificial change for the current step.
    env._parkour_active_waypoint_changed[:] = advanced
    env._parkour_max_course_progress_m[:] = torch.where(
        completed_course,
        env._parkour_route_cumulative_m[course_indices, waypoint_counts - 1],
        env._parkour_max_course_progress_m,
    )

    advanced_env_ids = torch.nonzero(advanced, as_tuple=False).flatten()
    if advanced_env_ids.numel() > 0:
        next_waypoints = env._parkour_waypoint_table[
            course_indices[advanced_env_ids],
            next_indices[advanced_env_ids],
        ]
        _write_waypoint_marker(
            env,
            advanced_env_ids,
            next_waypoints,
            waypoint_marker_cfg,
        )

    return completed_course


def last_episode_max_course_progress_m(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return each environment's latest completed-episode progress in metres."""

    progress = getattr(env, "_parkour_last_max_course_progress_m", None)
    if progress is None:
        raise RuntimeError("Route progress must be initialized before evaluation.")
    return progress


def reset_routes(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    family_indices: torch.Tensor,
    difficulty_indices: torch.Tensor,
    curriculum_cfg: ParkourCurriculumCfg,
    waypoint_marker_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reset selected environments to the first waypoint of their new routes.

    This initializes the shared waypoint tables when necessary, assigns each
    selected environment its logical curriculum level, clears its route
    progress and per-step event flag, and moves its visible marker to waypoint
    zero. Passing ``None`` as ``env_ids`` resets every environment.

    Args:
        env: Vectorized manager-based environment that owns the route buffers
            and waypoint-marker scene entity.
        env_ids: Indices of the environments being reset, or ``None`` for all
            environments.
        family_indices: Obstacle-family index for each selected environment.
        difficulty_indices: Difficulty-row index for each selected environment.
        curriculum_cfg: Course definitions used to build the waypoint table.
        waypoint_marker_cfg: Scene-entity selection for the visible waypoint
            marker.

    Returns:
        Flattened family-major course index for each reset environment. The
        returned tensor is also stored in the authoritative route-state buffer.
    """

    import torch

    from .._shared.runtime import _all_env_ids

    env_ids = _all_env_ids(env, env_ids)
    family_indices = family_indices.to(device=env.device, dtype=torch.long)
    difficulty_indices = difficulty_indices.to(device=env.device, dtype=torch.long)
    if (
        family_indices.shape != env_ids.shape
        or difficulty_indices.shape != env_ids.shape
    ):
        raise ValueError(
            "family_indices and difficulty_indices must contain one value per reset environment."
        )

    waypoint_marker = env.scene[waypoint_marker_cfg.name]
    dtype = waypoint_marker.data.default_root_state.dtype
    _ensure_route_state(env, curriculum_cfg, dtype=dtype)

    if torch.any(
        (family_indices < 0) | (family_indices >= len(curriculum_cfg.families))
    ):
        raise ValueError("family_indices contains an out-of-range obstacle family.")
    if torch.any(
        (difficulty_indices < 0)
        | (difficulty_indices >= curriculum_cfg.num_difficulties)
    ):
        raise ValueError("difficulty_indices contains an out-of-range difficulty.")

    course_indices = (
        family_indices * curriculum_cfg.num_difficulties + difficulty_indices
    )
    env._parkour_last_max_course_progress_m[env_ids] = (
        env._parkour_max_course_progress_m[env_ids]
    )
    env._parkour_max_course_progress_m[env_ids] = 0.0

    env._parkour_course_index[env_ids] = course_indices
    env._parkour_active_waypoint_index[env_ids] = 0
    env._parkour_active_waypoint_changed[env_ids] = False

    first_waypoints = env._parkour_waypoint_table[course_indices, 0]
    _write_waypoint_marker(
        env,
        env_ids,
        first_waypoints,
        waypoint_marker_cfg,
    )
    return course_indices


def _active_waypoint_plane_passed(
    env: ManagerBasedRLEnv,
    robot_xy: torch.Tensor,
    *,
    lateral_tolerance: float,
) -> torch.Tensor:
    """Detect route-plane crossings without accepting lateral shortcuts."""

    import torch

    # [num_envs]: course and active-waypoint cursor for each environment.
    course_indices = env._parkour_course_index
    active_indices = env._parkour_active_waypoint_index
    # [num_envs]: waypoint cursor immediately before each active target.
    previous_indices = (active_indices - 1).clamp_min(0)
    # [num_envs, 2]: XY waypoint preceding each active target.
    previous_waypoints = env._parkour_waypoint_table[
        course_indices,
        previous_indices,
        :2,
    ]
    # [num_envs, 2]: segment start; the condition broadcasts over XY.
    segment_starts = torch.where(
        (active_indices > 0)[:, None],
        previous_waypoints,
        torch.zeros_like(previous_waypoints),
    )
    # [num_envs, 2]: active target and vector from segment start to target.
    segment_ends = env._parkour_waypoint_table[
        course_indices,
        active_indices,
        :2,
    ]
    segment_vectors = segment_ends - segment_starts
    segment_lengths = torch.linalg.norm(segment_vectors, dim=-1)
    valid_segment = segment_lengths > torch.finfo(robot_xy.dtype).eps
    unit_directions = (
        segment_vectors
        / segment_lengths.clamp_min(torch.finfo(robot_xy.dtype).eps)[:, None]
    )
    relative_position = robot_xy - segment_starts
    longitudinal = torch.sum(relative_position * unit_directions, dim=-1)
    lateral_vector = relative_position - longitudinal[:, None] * unit_directions
    lateral_distance = torch.linalg.norm(lateral_vector, dim=-1)
    return (
        valid_segment
        & (longitudinal >= segment_lengths)
        & (lateral_distance < lateral_tolerance)
    )


def _advance_route_state(
    active_indices: torch.Tensor,
    waypoint_counts: torch.Tensor,
    within_radius: torch.Tensor,
    passed_waypoint_plane: torch.Tensor,
    final_waypoint_eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance independently reached cursors without exceeding route lengths.

    Intermediate routing points advance immediately from proximity or a safe
    crossing of their route-normal plane. Final completion instead requires
    proximity and ``final_waypoint_eligible``.

    Args:
        active_indices: Current waypoint index for each parallel environment.
            Each value is a cursor into that environment's configured route.
        waypoint_counts: Number of valid waypoints in each environment's route.
        within_radius: Whether each environment is currently close enough to
            its active waypoint.
        passed_waypoint_plane: Whether each environment passed its active
            waypoint along the current route segment while remaining inside
            the configured lateral corridor.
        final_waypoint_eligible: Whether each environment satisfies the extra
            completion condition required at its final waypoint.

    Returns:
        The next active indices and completed-course mask.
    """

    # Each environment can follow a route of a different length, so determine
    # its final index from its own waypoint count rather than a shared constant.
    final_waypoint = active_indices == waypoint_counts - 1

    # Intermediate route markers are control targets rather than places where
    # the robot should stop. Retarget as soon as the robot enters their radius
    # or crosses the marker plane inside the same lateral tolerance.
    advance_cursor = (~final_waypoint) & (within_radius | passed_waypoint_plane)

    completed_course = final_waypoint & within_radius & final_waypoint_eligible

    # Adding a Boolean tensor increments selected cursors by exactly one. Final
    # cursors are excluded above, so no index can exceed its route length.
    next_active_indices = active_indices + advance_cursor
    return next_active_indices, completed_course


def _ensure_route_state(
    env: ManagerBasedRLEnv,
    curriculum_cfg: ParkourCurriculumCfg,
    *,
    dtype: torch.dtype,
) -> None:
    """Create immutable route tables once and ensure live per-environment state."""

    import torch

    from .._shared.runtime import _get_or_init_env_buffer

    if not hasattr(env, "_parkour_waypoint_table"):
        courses = curriculum_cfg.courses
        routes = tuple(
            tuple(waypoint.position for waypoint in course.waypoints)
            for course in courses
        )
        max_waypoints = max(len(route) for route in routes)
        table = torch.empty(
            (len(routes), max_waypoints, 3),
            device=env.device,
            dtype=dtype,
        )
        counts = torch.empty(
            len(routes),
            device=env.device,
            dtype=torch.long,
        )
        milestone_table = torch.zeros(
            (len(routes), max_waypoints),
            device=env.device,
            dtype=torch.bool,
        )
        milestone_counts = torch.zeros(
            len(routes),
            device=env.device,
            dtype=torch.long,
        )
        for course_index, (course, route) in enumerate(zip(courses, routes)):
            route_tensor = torch.tensor(route, device=env.device, dtype=dtype)
            milestone_flags = torch.tensor(
                [waypoint.is_rewarded_milestone for waypoint in course.waypoints],
                device=env.device,
                dtype=torch.bool,
            )

            # Routes may contain different numbers of waypoints. Store the
            # actual length separately because the table must be rectangular
            # for vectorized indexing across levels and environments.
            count = route_tensor.shape[0]
            table[course_index, :count] = route_tensor

            # Fill the unused suffix with the final waypoint instead of leaving
            # uninitialized memory. ``counts`` remains authoritative, so route
            # cursors never intentionally advance into these padding entries.
            table[course_index, count:] = route_tensor[-1]
            counts[course_index] = count
            milestone_table[course_index, :count] = milestone_flags
            milestone_counts[course_index] = milestone_flags[:-1].sum()
        cumulative = torch.zeros(
            (len(routes), max_waypoints),
            device=env.device,
            dtype=dtype,
        )
        for course_index, route in enumerate(routes):
            previous_xy = torch.zeros(2, device=env.device, dtype=dtype)
            distance = torch.zeros((), device=env.device, dtype=dtype)
            for waypoint_index, position in enumerate(route):
                waypoint_xy = torch.tensor(
                    position[:2],
                    device=env.device,
                    dtype=dtype,
                )
                distance = distance + torch.linalg.norm(waypoint_xy - previous_xy)
                cumulative[course_index, waypoint_index] = distance
                previous_xy = waypoint_xy
            cumulative[course_index, len(route) :] = distance
        target_speeds = torch.tensor(
            [course.target_speed for course in courses],
            device=env.device,
            dtype=dtype,
        )
        min_clearances = torch.tensor(
            [course.min_clearance for course in courses],
            device=env.device,
            dtype=dtype,
        )
        env._parkour_waypoint_count_by_course = counts
        env._parkour_waypoint_milestone_table = milestone_table
        env._parkour_milestone_count_by_course = milestone_counts
        env._parkour_route_cumulative_m = cumulative
        env._parkour_target_speed_by_course = target_speeds
        env._parkour_min_clearance_by_course = min_clearances
        # Assign the sentinel last so a failed build is retried on the next reset.
        env._parkour_waypoint_table = table

    buffer_groups = (
        (
            torch.long,
            (
                "_parkour_course_index",
                "_parkour_active_waypoint_index",
            ),
        ),
        (
            dtype,
            (
                "_parkour_last_max_course_progress_m",
                "_parkour_max_course_progress_m",
            ),
        ),
        (
            torch.bool,
            ("_parkour_active_waypoint_changed",),
        ),
    )
    for buffer_dtype, names in buffer_groups:
        initial_value = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=buffer_dtype,
        )
        for name in names:
            _get_or_init_env_buffer(env, name, initial_value)


def _route_progress_m(
    env: ManagerBasedRLEnv,
    robot_xy: torch.Tensor,
) -> torch.Tensor:
    """Project each robot onto its active route segment in metric XY distance."""

    import torch

    # robot_xy [num_envs, 2]: terrain-local robot XY positions.
    # [num_envs]: course and target-waypoint cursor for each environment.
    course_indices = env._parkour_course_index
    active_indices = env._parkour_active_waypoint_index
    # [num_envs]: start-waypoint cursor, clamped for the first route segment.
    previous_indices = (active_indices - 1).clamp_min(0)
    # [num_envs, 2]: XY waypoint preceding each active target.
    previous_waypoints = env._parkour_waypoint_table[
        course_indices,
        previous_indices,
        :2,
    ]
    # [num_envs, 2]: segment start; the first segment starts at local XY zero.
    segment_starts = torch.where(
        (active_indices > 0)[:, None],
        previous_waypoints,
        torch.zeros_like(previous_waypoints),
    )
    # [num_envs, 2]: active target and directed segment vector.
    segment_ends = env._parkour_waypoint_table[
        course_indices,
        active_indices,
        :2,
    ]
    segment_vectors = segment_ends - segment_starts
    # [num_envs]: squared length and clamped projection along each segment.
    squared_lengths = torch.sum(segment_vectors.square(), dim=-1).clamp_min(
        torch.finfo(robot_xy.dtype).eps
    )
    fraction = (
        torch.sum((robot_xy - segment_starts) * segment_vectors, dim=-1)
        / squared_lengths
    ).clamp(0.0, 1.0)
    # [num_envs]: active-segment length and distance completed before it.
    segment_lengths = torch.sqrt(squared_lengths)
    # The [num_courses, max_waypoints] table stores cumulative route distances.
    prior_progress = torch.where(
        active_indices > 0,
        env._parkour_route_cumulative_m[course_indices, previous_indices],
        torch.zeros_like(fraction),
    )
    # [num_envs]: total route distance reached by the projected robot position.
    progress = prior_progress + fraction * segment_lengths
    return progress


def _write_waypoint_marker(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    waypoint_pos: torch.Tensor,
    waypoint_marker_cfg: SceneEntityCfg,
) -> None:
    """Move selected kinematic markers from terrain-local to world positions."""

    import torch

    waypoint_marker = env.scene[waypoint_marker_cfg.name]

    # Each default root-state row contains 13 values:
    # ``[position_xyz(3), quaternion_wxyz(4), linear_velocity_xyz(3),
    # angular_velocity_xyz(3)]``. Select the requested environments and retain
    # only the first seven pose values because velocity is written separately.
    marker_pose = waypoint_marker.data.default_root_state[env_ids, :7].clone()
    marker_pose[:, :3] = waypoint_pos + env.scene.env_origins[env_ids]
    zero_velocity = torch.zeros(
        (env_ids.numel(), 6),
        device=env.device,
        dtype=marker_pose.dtype,
    )
    # ``marker_pose`` has shape ``[len(env_ids), 7]``. Each row contains the
    # world-frame position ``(x, y, z)`` followed by the root quaternion
    # ``(w, x, y, z)`` for one selected waypoint marker.
    waypoint_marker.write_root_pose_to_sim(marker_pose, env_ids=env_ids)

    # ``zero_velocity`` has shape ``[len(env_ids), 6]``. Each row contains
    # world-frame linear velocity ``(vx, vy, vz)`` followed by angular
    # velocity ``(wx, wy, wz)``, all zero so the kinematic marker stays still.
    waypoint_marker.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
