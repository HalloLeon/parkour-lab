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


# Read-only route queries.


def active_course_indices(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return selected family-major course indices."""

    from .._shared.runtime import _all_env_ids
    from .state import _parkour_runtime

    env_ids = _all_env_ids(env, env_ids)
    return _parkour_runtime(env).route.course_indices[env_ids]


def active_difficulty_indices(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return difficulty rows for the retained active courses."""

    from .state import _parkour_runtime

    runtime = _parkour_runtime(env)
    return active_course_indices(env, env_ids) % runtime.courses.num_difficulties


def active_waypoint_changed_this_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return which environments switched targets during the current step.

    The success termination term updates this event before rewards are
    evaluated. It therefore remains valid for the progress reward during the
    same step and is overwritten by the next route update.
    """

    import torch

    from .state import _parkour_runtime_or_none

    runtime = _parkour_runtime_or_none(env)
    if runtime is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return runtime.route.waypoint_changed


def course_completed_this_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return which environments safely completed a course this step."""

    import torch

    from .state import _parkour_runtime_or_none

    runtime = _parkour_runtime_or_none(env)
    if runtime is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return runtime.route.course_completed


def active_waypoint_positions(
    env: ManagerBasedRLEnv,
    waypoint_marker_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Derive active terrain-local waypoints, falling back before first reset."""

    from .state import _parkour_runtime_or_none

    runtime = _parkour_runtime_or_none(env)
    if runtime is not None:
        return runtime.courses.waypoints[
            runtime.route.course_indices,
            runtime.route.active_waypoint_indices,
        ]

    waypoint_marker = env.scene[waypoint_marker_cfg.name]
    return waypoint_marker.data.root_pos_w - env.scene.env_origins


def current_min_clearances(
    env: ManagerBasedRLEnv,
    default: float = 0.25,
) -> torch.Tensor:
    """Return per-environment clearance targets without materializing copies."""

    import torch

    from .state import _parkour_runtime_or_none

    runtime = _parkour_runtime_or_none(env)
    if runtime is None:
        return torch.full(
            (env.num_envs,),
            default,
            device=env.device,
            dtype=torch.float32,
        )
    return runtime.courses.min_clearances[runtime.route.course_indices]


def current_target_speeds(
    env: ManagerBasedRLEnv,
    default: float = 0.70,
) -> torch.Tensor:
    """Return per-environment speed targets without materializing copies."""

    import torch

    from .state import _parkour_runtime_or_none

    runtime = _parkour_runtime_or_none(env)
    if runtime is None:
        return torch.full(
            (env.num_envs,),
            default,
            device=env.device,
            dtype=torch.float32,
        )
    return runtime.courses.target_speeds[runtime.route.course_indices]


def last_episode_max_course_progress_m(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return each environment's latest completed-episode progress in metres."""

    from .state import _parkour_runtime

    return _parkour_runtime(env).route.previous_episode_maximum_progress_m


def normalized_course_progress(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return monotonic safe route progress normalized by selected course length."""

    import torch

    from .._shared.runtime import _all_env_ids
    from .state import _parkour_runtime_or_none

    env_ids = _all_env_ids(env, env_ids)
    runtime = _parkour_runtime_or_none(env)
    if runtime is None:
        return torch.zeros(
            env_ids.numel(),
            device=env.device,
            dtype=torch.float32,
        )

    course_indices = runtime.route.course_indices[env_ids]
    waypoint_counts = runtime.courses.waypoint_counts[course_indices]
    course_lengths = runtime.courses.cumulative_distances_m[
        course_indices,
        waypoint_counts - 1,
    ]
    maximum_progress = runtime.route.maximum_progress_m[env_ids]
    return (
        maximum_progress
        / course_lengths.to(dtype=maximum_progress.dtype).clamp_min(torch.finfo(maximum_progress.dtype).eps)
    ).clamp(min=0.0, max=1.0)


def reached_milestone_reward_fractions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the normalized one-step milestone reward for each environment."""

    import torch

    from .state import _parkour_runtime_or_none

    runtime = _parkour_runtime_or_none(env)
    if runtime is None:
        return torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.float32,
        )

    route_state = runtime.route
    reached_indices = (route_state.active_waypoint_indices - 1).clamp_min(0)
    fractions = runtime.courses.milestone_reward_fractions[
        route_state.course_indices,
        reached_indices,
    ]
    return torch.where(
        route_state.waypoint_changed,
        fractions,
        torch.zeros_like(fractions),
    )


def route_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return normalized active-cursor and safe-progress phase."""

    import torch

    from .state import _parkour_runtime_or_none

    runtime = _parkour_runtime_or_none(env)
    if runtime is None:
        return torch.zeros((env.num_envs, 2), device=env.device)

    course_indices = runtime.route.course_indices
    waypoint_counts = runtime.courses.waypoint_counts[course_indices]
    route_dtype = runtime.courses.cumulative_distances_m.dtype
    active_waypoint_fraction = (
        runtime.route.active_waypoint_indices.to(dtype=route_dtype)
        / (waypoint_counts - 1).clamp_min(1).to(dtype=route_dtype)
    ).clamp(min=0.0, max=1.0)
    return torch.stack(
        (
            active_waypoint_fraction,
            normalized_course_progress(env),
        ),
        dim=-1,
    )


# Route transitions.


def advance_active_waypoints(
    env: ManagerBasedRLEnv,
    *,
    reach_threshold: float,
    waypoint_marker_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    feet_asset_cfg: SceneEntityCfg,
    feet_contact_cfg: SceneEntityCfg,
    trunk_contact_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    max_completion_tilt: float = 0.5,
    max_completion_vertical_speed: float = 0.5,
    support_margin: float = 0.05,
    support_plane_tolerance: float = 0.12,
) -> torch.Tensor:
    """Update every route once and return final-course completions.

    Isaac Lab evaluates termination terms before rewards and observations. This
    function is therefore called by the success termination term so every later
    consumer sees the newly selected waypoint during the same control step.
    """

    import torch

    from .._shared import contact, robot
    from ..terrain import queries
    from .state import _parkour_runtime

    if reach_threshold <= 0.0:
        raise ValueError("reach_threshold must be positive.")
    if contact_threshold < 0.0:
        raise ValueError("contact_threshold must be non-negative.")
    if max_completion_tilt <= 0.0:
        raise ValueError("max_completion_tilt must be positive.")
    if max_completion_vertical_speed <= 0.0:
        raise ValueError("max_completion_vertical_speed must be positive.")
    if support_margin < 0.0:
        raise ValueError("support_margin must be non-negative.")
    if support_plane_tolerance <= 0.0:
        raise ValueError("support_plane_tolerance must be positive.")
    runtime = _parkour_runtime(env)
    courses = runtime.courses
    route_state = runtime.route
    active_indices = route_state.active_waypoint_indices
    course_indices = route_state.course_indices
    waypoint_counts = courses.waypoint_counts[course_indices]
    active_positions = courses.waypoints[course_indices, active_indices]
    robot_pos = robot._root_pos_env(env, asset_cfg)
    distance_xy = torch.linalg.norm(
        robot_pos[:, :2] - active_positions[:, :2],
        dim=-1,
    )
    within_radius = distance_xy < reach_threshold
    previous_robot_xy = route_state.previous_root_xy.clone()
    passed_waypoint_plane = _active_waypoint_plane_passed(
        env,
        previous_robot_xy,
        robot_pos[:, :2],
        lateral_tolerance=reach_threshold,
    )
    route_state.previous_root_xy[:] = robot_pos[:, :2]

    support_required = (
        courses.support_vertex_counts[
            course_indices,
            active_indices,
        ]
        > 0
    )
    supported = _active_waypoint_supported(
        env,
        feet_asset_cfg=feet_asset_cfg,
        feet_contact_cfg=feet_contact_cfg,
        contact_threshold=contact_threshold,
        support_margin=support_margin,
        support_plane_tolerance=support_plane_tolerance,
    )
    trunk_contact = torch.any(
        contact._force_norm_mask(env, sensor_cfg=trunk_contact_cfg) > contact_threshold,
        dim=(1, 2),
    )

    clearance = queries._base_clearance(env, asset_cfg)
    min_clearance = courses.min_clearances[course_indices].to(
        device=clearance.device,
        dtype=clearance.dtype,
    )
    final_waypoint_eligible = (
        supported
        & (~trunk_contact)
        & (clearance > min_clearance)
        & (torch.abs(robot._root_lin_vel_z(env, asset_cfg)) < max_completion_vertical_speed)
        & (
            torch.linalg.norm(
                robot._root_projected_gravity_xy(env, asset_cfg),
                dim=-1,
            )
            < max_completion_tilt
        )
    )

    progress = _route_progress_m(
        env,
        robot_pos[:, :2],
        lateral_tolerance=reach_threshold,
    )
    # Progress can decrease if the robot backtracks. Preserve the furthest point
    # reached safely so a crash or off-route shortcut cannot drive curriculum.
    route_state.maximum_progress_m[:] = torch.where(
        trunk_contact,
        route_state.maximum_progress_m,
        torch.maximum(
            route_state.maximum_progress_m,
            progress,
        ),
    )

    next_indices, completed_course = _advance_route_state(
        active_indices,
        waypoint_counts,
        within_radius,
        passed_waypoint_plane,
        support_required,
        supported,
        ~trunk_contact,
        final_waypoint_eligible,
    )

    advanced = next_indices != active_indices
    route_state.active_waypoint_indices[:] = next_indices
    # Retargeting replaces the nearby reached waypoint with the farther next one,
    # making active-waypoint distance jump without robot motion. The progress
    # reward uses this event to ignore that artificial change for the current step.
    route_state.waypoint_changed[:] = advanced
    route_state.course_completed[:] = completed_course
    route_state.maximum_progress_m[:] = torch.where(
        completed_course,
        courses.cumulative_distances_m[course_indices, waypoint_counts - 1],
        route_state.maximum_progress_m,
    )

    advanced_env_ids = torch.nonzero(advanced, as_tuple=False).flatten()
    if advanced_env_ids.numel() > 0:
        next_waypoints = courses.waypoints[
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


def reset_routes(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    family_indices: torch.Tensor,
    difficulty_indices: torch.Tensor,
    curriculum_cfg: ParkourCurriculumCfg,
    waypoint_marker_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Reset selected environments to the first waypoint of their new routes.

    This initializes the shared waypoint tables when necessary, assigns each
    selected environment its logical curriculum level, clears its route
    progress and per-step event flag, and moves its visible marker to waypoint
    zero. Passing ``None`` as ``env_ids`` resets every environment.

    Args:
        env: Vectorized manager-based environment that owns the grouped route
            runtime and waypoint-marker scene entity.
        env_ids: Indices of the environments being reset, or ``None`` for all
            environments.
        family_indices: Obstacle-family index for each selected environment.
        difficulty_indices: Difficulty-row index for each selected environment.
        curriculum_cfg: Course definitions used to build the waypoint table.
        waypoint_marker_cfg: Scene-entity selection for the visible waypoint
            marker.
        asset_cfg: Robot scene entity whose post-reset terrain-local root
            position seeds genuine route-plane crossing detection.

    Returns:
        Flattened family-major course index for each reset environment. The
        returned tensor is also stored in the authoritative grouped route state.
    """

    import torch

    from .._shared import robot
    from .._shared.runtime import _all_env_ids
    from .state import _ensure_parkour_runtime

    env_ids = _all_env_ids(env, env_ids)
    family_indices = family_indices.to(device=env.device, dtype=torch.long)
    difficulty_indices = difficulty_indices.to(device=env.device, dtype=torch.long)
    if family_indices.shape != env_ids.shape or difficulty_indices.shape != env_ids.shape:
        raise ValueError("family_indices and difficulty_indices must contain one value per reset environment.")

    waypoint_marker = env.scene[waypoint_marker_cfg.name]
    dtype = waypoint_marker.data.default_root_state.dtype
    runtime = _ensure_parkour_runtime(env, curriculum_cfg, dtype=dtype)

    if torch.any((family_indices < 0) | (family_indices >= len(curriculum_cfg.families))):
        raise ValueError("family_indices contains an out-of-range obstacle family.")
    if torch.any((difficulty_indices < 0) | (difficulty_indices >= curriculum_cfg.num_difficulties)):
        raise ValueError("difficulty_indices contains an out-of-range difficulty.")

    course_indices = family_indices * curriculum_cfg.num_difficulties + difficulty_indices
    route_state = runtime.route
    route_state.previous_episode_maximum_progress_m[env_ids] = route_state.maximum_progress_m[env_ids]
    route_state.maximum_progress_m[env_ids] = 0.0
    route_state.course_indices[env_ids] = course_indices
    route_state.active_waypoint_indices[env_ids] = 0
    route_state.waypoint_changed[env_ids] = False
    route_state.course_completed[env_ids] = False
    route_state.previous_root_xy[env_ids] = robot._root_pos_env(
        env,
        asset_cfg,
    )[env_ids, :2]

    first_waypoints = runtime.courses.waypoints[course_indices, 0]
    _write_waypoint_marker(
        env,
        env_ids,
        first_waypoints,
        waypoint_marker_cfg,
    )
    return course_indices


# Private route geometry and transition helpers.


def _active_waypoint_plane_passed(
    env: ManagerBasedRLEnv,
    previous_robot_xy: torch.Tensor,
    robot_xy: torch.Tensor,
    *,
    lateral_tolerance: float,
) -> torch.Tensor:
    """Detect a genuine previous-to-current crossing inside the route corridor."""

    import torch

    from .state import _parkour_runtime

    runtime = _parkour_runtime(env)
    # [num_envs]: course and active-waypoint cursor for each environment.
    course_indices = runtime.route.course_indices
    active_indices = runtime.route.active_waypoint_indices
    # [num_envs]: waypoint cursor immediately before each active target.
    previous_indices = (active_indices - 1).clamp_min(0)
    # [num_envs, 2]: XY waypoint preceding each active target.
    previous_waypoints = runtime.courses.waypoints[
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
    segment_ends = runtime.courses.waypoints[
        course_indices,
        active_indices,
        :2,
    ]
    segment_vectors = segment_ends - segment_starts
    segment_lengths = torch.linalg.norm(segment_vectors, dim=-1)
    valid_segment = segment_lengths > torch.finfo(robot_xy.dtype).eps
    unit_directions = segment_vectors / segment_lengths.clamp_min(torch.finfo(robot_xy.dtype).eps)[:, None]
    # Project both robot samples onto the route direction. The target plane is at
    # ``segment_lengths`` along this direction.
    previous_longitudinal = torch.sum(
        (previous_robot_xy - segment_starts) * unit_directions,
        dim=-1,
    )
    current_longitudinal = torch.sum(
        (robot_xy - segment_starts) * unit_directions,
        dim=-1,
    )
    # A crossing occurs only when the robot moves from behind the target plane to
    # the plane or beyond it during this step.
    crossed_forward = (previous_longitudinal < segment_lengths) & (current_longitudinal >= segment_lengths)

    # Treat the motion between samples as a straight line. This fraction says how
    # far through the step the robot reaches the target plane. The epsilon keeps
    # division safe for non-crossing rows, which are rejected below.
    longitudinal_delta = current_longitudinal - previous_longitudinal
    crossing_fraction = (
        (segment_lengths - previous_longitudinal) / longitudinal_delta.clamp_min(torch.finfo(robot_xy.dtype).eps)
    ).clamp(0.0, 1.0)
    # ``crossing_xy`` is the estimated XY position where that straight-line motion
    # intersects the target plane. ``[:, None]`` applies one fraction to both XY
    # coordinates for each environment.
    crossing_xy = previous_robot_xy + crossing_fraction[:, None] * (robot_xy - previous_robot_xy)

    # Remove the component parallel to the route; the remaining vector is the
    # lateral offset from the route centerline at the crossing point.
    target_relative = crossing_xy - segment_ends
    target_longitudinal = torch.sum(
        target_relative * unit_directions,
        dim=-1,
    )
    lateral_vector = target_relative - target_longitudinal[:, None] * unit_directions
    lateral_distance = torch.linalg.norm(lateral_vector, dim=-1)

    # Testing the interpolated crossing position prevents a robot from crossing
    # outside the corridor and then moving laterally inside before this sample.
    return valid_segment & crossed_forward & (lateral_distance <= lateral_tolerance)


def _active_waypoint_supported(
    env: ManagerBasedRLEnv,
    *,
    feet_asset_cfg: SceneEntityCfg,
    feet_contact_cfg: SceneEntityCfg,
    contact_threshold: float,
    support_margin: float,
    support_plane_tolerance: float,
) -> torch.Tensor:
    """Return whether a recently contacted foot is on the intended support."""

    import torch

    from .._shared import contact, robot, runtime
    from .state import _parkour_runtime

    parkour_runtime = _parkour_runtime(env)
    course_indices = parkour_runtime.route.course_indices
    active_indices = parkour_runtime.route.active_waypoint_indices
    support_vertices = parkour_runtime.courses.support_vertices[
        course_indices,
        active_indices,
    ]
    support_vertex_counts = parkour_runtime.courses.support_vertex_counts[
        course_indices,
        active_indices,
    ]
    support_normals = parkour_runtime.courses.support_normals[
        course_indices,
        active_indices,
    ]
    support_required = support_vertex_counts > 0

    foot_positions = robot._selected_body_pos_env(env, feet_asset_cfg)
    recent_contact = torch.any(
        contact._force_norm_mask(env, sensor_cfg=feet_contact_cfg) > contact_threshold,
        dim=1,
    )
    runtime._validate_matching_shape(
        recent_contact,
        foot_positions[..., 0],
        lhs_name="recent foot contact",
        rhs_name="foot positions",
    )

    # [num_envs, max_vertices, 3]: ordered polygon edges. Padded vertices
    # repeat the first vertex and are ignored by ``valid_edges`` below.
    edge_starts = support_vertices
    edge_ends = torch.roll(support_vertices, shifts=-1, dims=1)
    edge_vectors = edge_ends - edge_starts
    edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)
    edge_indices = torch.arange(
        support_vertices.shape[1],
        device=env.device,
    )
    valid_edges = edge_indices[None, :] < support_vertex_counts[:, None]

    # Vector from one point on the plane to each foot:
    # [num_envs, num_feet, 3] - [num_envs, 1, 3].
    relative_to_plane = foot_positions - support_vertices[:, None, 0, :]
    # d = (foot - plane_point) dot unit_normal is the signed plane distance.
    signed_plane_distance = torch.sum(
        relative_to_plane * support_normals[:, None, :],
        dim=-1,
    )
    # foot - d * unit_normal is its orthogonal projection onto the plane.
    projected_feet = foot_positions - signed_plane_distance[..., None] * support_normals[:, None, :]
    # Vector from every edge start to every projected foot.
    edge_to_foot = projected_feet[:, :, None, :] - edge_starts[:, None, :, :]
    # ((edge x edge_to_foot) dot normal) / |edge| is the signed distance
    # from the foot to that edge; CCW vertex order makes the inside positive.
    inward_distance = (
        torch.sum(
            torch.linalg.cross(
                edge_vectors[:, None, :, :],
                edge_to_foot,
                dim=-1,
            )
            * support_normals[:, None, None, :],
            dim=-1,
        )
        / edge_lengths.clamp_min(torch.finfo(foot_positions.dtype).eps)[:, None, :]
    )
    # A point is inside a convex polygon when it is inside every valid edge;
    # ``support_margin`` permits a small metric distance beyond the boundary.
    inside_polygon = torch.all(
        (~valid_edges[:, None, :]) | (inward_distance >= -support_margin),
        dim=-1,
    )
    # A foot supports the waypoint only if it contacts inside and near its plane.
    near_plane = torch.abs(signed_plane_distance) <= support_plane_tolerance
    supported_feet = recent_contact & inside_polygon & near_plane
    # The environment passes when support is required and at least one foot has it.
    return support_required & torch.any(supported_feet, dim=-1)


def _advance_route_state(
    active_indices: torch.Tensor,
    waypoint_counts: torch.Tensor,
    within_radius: torch.Tensor,
    passed_waypoint_plane: torch.Tensor,
    support_required: torch.Tensor,
    supported: torch.Tensor,
    route_state_eligible: torch.Tensor,
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
        support_required: Whether the active waypoint identifies a physical
            support instead of serving only as a route-control target.
        supported: Whether a recently contacted foot is currently on the
            active waypoint's intended support polygon.
        route_state_eligible: Whether the robot may advance route state. A
            trunk-contacting robot is ineligible.
        final_waypoint_eligible: Whether each environment satisfies the extra
            completion condition required at its final waypoint.

    Returns:
        The next active indices and completed-course mask.
    """

    # Each environment can follow a route of a different length, so determine
    # its final index from its own waypoint count rather than a shared constant.
    final_waypoint = active_indices == waypoint_counts - 1

    # Control targets retarget immediately from proximity or one genuine plane
    # crossing. A physical waypoint instead represents a landing and advances
    # only while the root is nearby and a contacted foot is on its support.
    control_reached = (~support_required) & (within_radius | passed_waypoint_plane)
    physical_reached = support_required & within_radius & supported
    advance_cursor = (~final_waypoint) & route_state_eligible & (control_reached | physical_reached)

    completed_course = final_waypoint & physical_reached & route_state_eligible & final_waypoint_eligible

    # Adding a Boolean tensor increments selected cursors by exactly one. Final
    # cursors are excluded above, so no index can exceed its route length.
    next_active_indices = active_indices + advance_cursor
    return next_active_indices, completed_course


def _route_progress_m(
    env: ManagerBasedRLEnv,
    robot_xy: torch.Tensor,
    *,
    lateral_tolerance: float,
) -> torch.Tensor:
    """Project onto the route without crediting motion outside its corridor."""

    import torch

    from .state import _parkour_runtime

    runtime = _parkour_runtime(env)
    # robot_xy [num_envs, 2]: terrain-local robot XY positions.
    # [num_envs]: course and target-waypoint cursor for each environment.
    course_indices = runtime.route.course_indices
    active_indices = runtime.route.active_waypoint_indices
    # [num_envs]: start-waypoint cursor, clamped for the first route segment.
    previous_indices = (active_indices - 1).clamp_min(0)
    # [num_envs, 2]: XY waypoint preceding each active target.
    previous_waypoints = runtime.courses.waypoints[
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
    segment_ends = runtime.courses.waypoints[
        course_indices,
        active_indices,
        :2,
    ]
    segment_vectors = segment_ends - segment_starts
    # [num_envs]: squared length and clamped projection along each segment.
    squared_lengths = torch.sum(segment_vectors.square(), dim=-1).clamp_min(torch.finfo(robot_xy.dtype).eps)
    relative_position = robot_xy - segment_starts
    fraction = (torch.sum(relative_position * segment_vectors, dim=-1) / squared_lengths).clamp(0.0, 1.0)
    # [num_envs]: active-segment length and distance completed before it.
    segment_lengths = torch.sqrt(squared_lengths)
    unit_directions = segment_vectors / segment_lengths[:, None]
    longitudinal = torch.sum(
        relative_position * unit_directions,
        dim=-1,
    )
    lateral_vector = relative_position - longitudinal[:, None] * unit_directions
    inside_corridor = torch.linalg.norm(lateral_vector, dim=-1) <= lateral_tolerance
    # The [num_courses, max_waypoints] table stores cumulative route distances.
    prior_progress = torch.where(
        active_indices > 0,
        runtime.courses.cumulative_distances_m[
            course_indices,
            previous_indices,
        ],
        torch.zeros_like(fraction),
    )
    # [num_envs]: total route distance reached by the projected robot position.
    progress = torch.where(
        inside_corridor,
        prior_progress + fraction * segment_lengths,
        prior_progress,
    )
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
