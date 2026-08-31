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


# Course-selection queries.


def active_course_indices(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return selected family-then-variant-major course indices."""

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


# Active-waypoint queries.


def has_active_routes(env: ManagerBasedRLEnv) -> bool:
    """Return whether the first curriculum reset initialized route state."""

    from .state import _parkour_runtime_or_none

    return _parkour_runtime_or_none(env) is not None


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


def active_waypoint_indices(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return each environment's active route cursor."""

    from .state import _parkour_runtime

    return _parkour_runtime(env).route.active_waypoint_indices.clone()


def active_waypoint_is_final(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether each active waypoint is the course's final target."""

    from .state import _parkour_runtime

    runtime = _parkour_runtime(env)
    course_indices = runtime.route.course_indices
    return (
        runtime.route.active_waypoint_indices
        == runtime.courses.waypoint_counts[course_indices] - 1
    )


def active_waypoint_is_rewarded_milestone(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether each active waypoint is a rewarded physical milestone."""

    from .state import _parkour_runtime

    runtime = _parkour_runtime(env)
    return (
        runtime.courses.milestone_reward_fractions[
            runtime.route.course_indices,
            runtime.route.active_waypoint_indices,
        ]
        > 0.0
    )


def active_waypoint_is_terminal_landing(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return whether each active target is itself the final landing."""

    from .state import _parkour_runtime

    runtime = _parkour_runtime(env)
    return runtime.courses.terminal_landing_masks[
        runtime.route.course_indices,
        runtime.route.active_waypoint_indices,
    ]


def active_waypoint_inbound_direction_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the fixed world-XY unit direction of each active route segment."""

    import torch

    from .state import _parkour_runtime

    runtime = _parkour_runtime(env)
    course_indices = runtime.route.course_indices
    active_indices = runtime.route.active_waypoint_indices
    previous_indices = (active_indices - 1).clamp_min(0)
    previous_waypoints = runtime.courses.waypoints[course_indices, previous_indices, :2]
    starts = torch.where(
        (active_indices > 0).unsqueeze(-1),
        previous_waypoints,
        torch.zeros_like(previous_waypoints),
    )
    vectors = runtime.courses.waypoints[course_indices, active_indices, :2] - starts
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)
    fallback = torch.zeros_like(vectors)
    fallback[:, 0] = 1.0
    return torch.where(
        lengths > 1.0e-6,
        vectors / lengths.clamp_min(1.0e-6),
        fallback,
    )


def terminal_landing_diagnostics(
    env: ManagerBasedRLEnv,
) -> dict[str, torch.Tensor]:
    """Return cached terminal-gate predicates and dwell without recomputation."""

    import torch

    from .state import TERMINAL_LANDING_PREDICATE_NAMES, _parkour_runtime

    route_state = _parkour_runtime(env).route
    diagnostics = {
        name: route_state.terminal_landing_predicates[:, index]
        for index, name in enumerate(TERMINAL_LANDING_PREDICATE_NAMES)
    }
    diagnostics["active"] = route_state.terminal_landing_active
    diagnostics["stable"] = route_state.terminal_landing_active & torch.all(
        route_state.terminal_landing_predicates,
        dim=-1,
    )
    diagnostics["dwell_s"] = route_state.terminal_landing_stable_time_s
    return diagnostics


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


def active_waypoint_root_reach_radii(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the configured root-reach radius of each active waypoint."""

    from .state import _parkour_runtime

    runtime = _parkour_runtime(env)
    return runtime.courses.root_reach_radii[
        runtime.route.course_indices,
        runtime.route.active_waypoint_indices,
    ]


# Route-state and geometry queries.


def course_completed_this_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return which environments safely completed a course this step."""

    import torch

    from .state import _parkour_runtime_or_none

    runtime = _parkour_runtime_or_none(env)
    if runtime is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return runtime.route.course_completed


def current_min_clearances(
    env: ManagerBasedRLEnv,
    default: float = 0.27,
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


def final_waypoint_positions(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return each selected course's final terrain-local waypoint."""

    from .state import _parkour_runtime

    runtime = _parkour_runtime(env)
    final_indices = runtime.courses.waypoint_counts[runtime.route.course_indices] - 1
    return runtime.courses.waypoints[runtime.route.course_indices, final_indices]


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
        / course_lengths.to(dtype=maximum_progress.dtype).clamp_min(
            torch.finfo(maximum_progress.dtype).eps
        )
    ).clamp(min=0.0, max=1.0)


def normalized_waypoint_progress(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the fraction of intermediate route waypoints already passed.

    Final completion is handled separately because it is a binary success
    outcome. Every intermediate cursor transition contributes equally, keeping
    curriculum evidence independent of the milestone reward budget.
    """

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

    route_state = runtime.route
    course_indices = route_state.course_indices[env_ids]
    dtype = runtime.courses.cumulative_distances_m.dtype
    passed_waypoints = route_state.active_waypoint_indices[env_ids].to(dtype=dtype)
    intermediate_counts = (
        runtime.courses.waypoint_counts[course_indices] - 1
    ).clamp_min(1)
    progress = passed_waypoints / intermediate_counts.to(dtype=dtype)
    return torch.where(
        route_state.course_completed[env_ids],
        torch.ones_like(progress),
        progress,
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


def route_cross_track_error_m(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return root XY distance to the nearest finite segment of the route.

    The approved route is the terrain-local polyline from the course origin to
    its valid waypoints. The fixed route envelope consumes this query without
    exposing either the distance or its limits to the actor.
    """

    from .._shared import robot
    from .state import _parkour_runtime

    runtime = _parkour_runtime(env)
    course_indices = runtime.route.course_indices
    return _finite_route_cross_track_error_m(
        robot._root_pos_env(env)[:, :2],
        runtime.courses.waypoints[course_indices, :, :2],
        runtime.courses.waypoint_counts[course_indices],
    )


def route_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return normalized active-cursor and safe-progress phase."""

    import torch

    from .state import _parkour_runtime_or_none

    runtime = _parkour_runtime_or_none(env)
    if runtime is None:
        return torch.zeros((env.num_envs, 2), device=env.device)

    return torch.stack(
        (
            normalized_waypoint_progress(env),
            normalized_course_progress(env),
        ),
        dim=-1,
    )


# Route transitions.


def advance_active_waypoints(
    env: ManagerBasedRLEnv,
    *,
    waypoint_marker_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    feet_asset_cfg: SceneEntityCfg,
    feet_contact_cfg: SceneEntityCfg,
    chassis_contact_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
    max_completion_tilt_sine: float = 0.5,
    max_completion_vertical_speed_m_s: float = 0.5,
    support_margin: float = 0.05,
    support_plane_tolerance: float = 0.12,
    terminal_support_load_threshold_n: float = 10.0,
    terminal_min_support_feet: int = 2,
    terminal_stability_dwell_s: float = 0.2,
    terminal_max_planar_speed_m_s: float = 0.2,
    terminal_max_vertical_speed_m_s: float = 0.2,
    terminal_max_yaw_rate_rad_s: float = 0.35,
    terminal_max_roll_pitch_rate_rad_s: float = 0.35,
    terminal_max_tilt_sine: float = 0.25,
    progress_route_half_width_m: float = 0.2,
    hard_route_half_width_m: float | None = None,
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

    runtime = _parkour_runtime(env)
    courses = runtime.courses
    route_state = runtime.route
    active_indices = route_state.active_waypoint_indices
    course_indices = route_state.course_indices
    waypoint_counts = courses.waypoint_counts[course_indices]
    active_positions = courses.waypoints[course_indices, active_indices]
    robot_pos = robot._root_pos_env(env, asset_cfg)
    if hard_route_half_width_m is None:
        inside_hard_route_envelope = torch.ones(
            env.num_envs, device=env.device, dtype=torch.bool
        )
    else:
        cross_track_error = _finite_route_cross_track_error_m(
            robot_pos[:, :2],
            courses.waypoints[course_indices, :, :2],
            waypoint_counts,
        )
        inside_hard_route_envelope = torch.isfinite(cross_track_error) & (
            cross_track_error <= hard_route_half_width_m
        )
    distance_xy = torch.linalg.norm(
        robot_pos[:, :2] - active_positions[:, :2],
        dim=-1,
    )
    active_root_reach_radii = courses.root_reach_radii[
        course_indices,
        active_indices,
    ]
    root_within_radius = distance_xy < active_root_reach_radii
    support_required = (
        courses.support_vertex_counts[
            course_indices,
            active_indices,
        ]
        > 0
    )
    supported, terminal_load_supported = _active_waypoint_support(
        env,
        feet_asset_cfg=feet_asset_cfg,
        feet_contact_cfg=feet_contact_cfg,
        contact_threshold=contact_threshold,
        support_margin=support_margin,
        support_plane_tolerance=support_plane_tolerance,
        terminal_support_load_threshold_n=terminal_support_load_threshold_n,
        terminal_min_support_feet=terminal_min_support_feet,
    )
    chassis_contact = torch.any(
        contact._force_norm_mask(env, sensor_cfg=chassis_contact_cfg)
        > contact_threshold,
        dim=(1, 2),
    )
    route_state_eligible = (~chassis_contact) & inside_hard_route_envelope

    clearance, clearance_valid = queries._base_clearance_components(env, asset_cfg)
    min_clearance = courses.min_clearances[course_indices].to(
        device=clearance.device,
        dtype=clearance.dtype,
    )
    completion_clearance = clearance_valid & (clearance > min_clearance)
    ordinary_final_eligible = (
        supported
        & (~chassis_contact)
        & completion_clearance
        & (
            torch.abs(robot._root_lin_vel_z(env, asset_cfg))
            < max_completion_vertical_speed_m_s
        )
        & (
            torch.linalg.norm(
                robot._root_projected_gravity_xy(env, asset_cfg),
                dim=-1,
            )
            < max_completion_tilt_sine
        )
    )
    terminal_landing = courses.terminal_landing_masks[course_indices, active_indices]
    terminal_predicates = torch.stack(
        (
            root_within_radius,
            route_state_eligible,
            supported,
            terminal_load_supported,
            completion_clearance,
            torch.linalg.norm(robot._root_lin_vel_xy(env, asset_cfg), dim=-1)
            < terminal_max_planar_speed_m_s,
            torch.abs(robot._root_lin_vel_z(env, asset_cfg))
            < terminal_max_vertical_speed_m_s,
            torch.abs(robot._root_ang_vel_z(env, asset_cfg))
            < terminal_max_yaw_rate_rad_s,
            torch.linalg.norm(robot._root_ang_vel_xy(env, asset_cfg), dim=-1)
            < terminal_max_roll_pitch_rate_rad_s,
            torch.linalg.norm(
                robot._root_projected_gravity_xy(env, asset_cfg),
                dim=-1,
            )
            < terminal_max_tilt_sine,
        ),
        dim=-1,
    )
    route_state.terminal_landing_active.copy_(terminal_landing)
    route_state.terminal_landing_predicates.copy_(terminal_predicates)
    terminal_stable = terminal_landing & torch.all(terminal_predicates, dim=-1)
    route_state.terminal_landing_stable_time_s[:] = torch.where(
        terminal_stable,
        route_state.terminal_landing_stable_time_s + float(env.step_dt),
        torch.zeros_like(route_state.terminal_landing_stable_time_s),
    )
    terminal_final_eligible = (
        route_state.terminal_landing_stable_time_s + 1.0e-6
        >= terminal_stability_dwell_s
    )
    final_waypoint_eligible = torch.where(
        terminal_landing,
        terminal_final_eligible,
        ordinary_final_eligible,
    )

    progress = _route_progress_m(
        env,
        robot_pos[:, :2],
        route_half_width_m=progress_route_half_width_m,
    )
    # Progress can decrease if the robot backtracks. Preserve the furthest point
    # reached safely so a crash or off-route shortcut cannot drive curriculum.
    route_state.maximum_progress_m[:] = torch.where(
        ~route_state_eligible,
        route_state.maximum_progress_m,
        torch.maximum(
            route_state.maximum_progress_m,
            progress,
        ),
    )

    next_indices, completed_course = _advance_route_state(
        active_indices,
        waypoint_counts,
        root_within_radius,
        support_required,
        supported,
        route_state_eligible,
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
    geometry_variant_indices: torch.Tensor,
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
        geometry_variant_indices: Deterministic geometry variant selected by
            each environment's physical terrain column.

    Returns:
        Flattened family-then-variant-major course index for each reset
        environment. The returned tensor is also stored in the authoritative
        grouped route state.
    """

    import torch

    from .._shared.runtime import _all_env_ids
    from .state import _ensure_parkour_runtime

    env_ids = _all_env_ids(env, env_ids)
    family_indices = family_indices.to(device=env.device, dtype=torch.long)
    difficulty_indices = difficulty_indices.to(device=env.device, dtype=torch.long)
    geometry_variant_indices = geometry_variant_indices.to(
        device=env.device,
        dtype=torch.long,
    )
    if any(
        indices.shape != env_ids.shape
        for indices in (
            family_indices,
            difficulty_indices,
            geometry_variant_indices,
        )
    ):
        raise ValueError(
            "family, geometry variant, and difficulty indices must contain one value per reset environment."
        )

    waypoint_marker = env.scene[waypoint_marker_cfg.name]
    dtype = waypoint_marker.data.default_root_state.dtype
    runtime = _ensure_parkour_runtime(
        env,
        curriculum_cfg,
        dtype=dtype,
    )

    if torch.any(
        (family_indices < 0) | (family_indices >= len(curriculum_cfg.families))
    ):
        raise ValueError("family_indices contains an out-of-range obstacle family.")
    if torch.any(
        (difficulty_indices < 0)
        | (difficulty_indices >= curriculum_cfg.num_difficulties)
    ):
        raise ValueError("difficulty_indices contains an out-of-range difficulty.")
    if torch.any(
        (geometry_variant_indices < 0)
        | (geometry_variant_indices >= curriculum_cfg.num_geometry_variants)
    ):
        raise ValueError("geometry_variant_indices contains an out-of-range variant.")

    course_indices = (
        family_indices * curriculum_cfg.num_geometry_variants + geometry_variant_indices
    ) * curriculum_cfg.num_difficulties + difficulty_indices
    route_state = runtime.route
    route_state.previous_episode_maximum_progress_m[env_ids] = (
        route_state.maximum_progress_m[env_ids]
    )
    route_state.maximum_progress_m[env_ids] = 0.0
    route_state.terminal_landing_stable_time_s[env_ids] = 0.0
    route_state.terminal_landing_active[env_ids] = False
    route_state.terminal_landing_predicates[env_ids] = False
    route_state.course_indices[env_ids] = course_indices
    route_state.active_waypoint_indices[env_ids] = 0
    route_state.waypoint_changed[env_ids] = False
    route_state.course_completed[env_ids] = False
    first_waypoints = runtime.courses.waypoints[course_indices, 0]
    _write_waypoint_marker(
        env,
        env_ids,
        first_waypoints,
        waypoint_marker_cfg,
    )
    return course_indices


def _active_waypoint_support(
    env: ManagerBasedRLEnv,
    *,
    feet_asset_cfg: SceneEntityCfg,
    feet_contact_cfg: SceneEntityCfg,
    contact_threshold: float,
    support_margin: float,
    support_plane_tolerance: float,
    terminal_support_load_threshold_n: float,
    terminal_min_support_feet: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ordinary contact and terminal load-bearing support masks."""

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
    projected_feet = (
        foot_positions - signed_plane_distance[..., None] * support_normals[:, None, :]
    )
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
    feet_on_support = support_required[:, None] & inside_polygon & near_plane
    ordinary_supported = torch.any(feet_on_support & recent_contact, dim=-1)

    current_forces_w = contact._selected_contact_forces_w(env, feet_contact_cfg)
    runtime._validate_matching_shape(
        current_forces_w[..., 0],
        foot_positions[..., 0],
        lhs_name="current foot contact force",
        rhs_name="foot positions",
    )
    normal_load = torch.sum(
        current_forces_w * support_normals[:, None, :],
        dim=-1,
    )
    load_bearing_feet = feet_on_support & (
        normal_load >= terminal_support_load_threshold_n
    )
    terminal_supported = (
        torch.sum(load_bearing_feet, dim=-1) >= terminal_min_support_feet
    )
    return ordinary_supported, terminal_supported


def _advance_route_state(
    active_indices: torch.Tensor,
    waypoint_counts: torch.Tensor,
    root_within_radius: torch.Tensor,
    support_required: torch.Tensor,
    supported: torch.Tensor,
    route_state_eligible: torch.Tensor,
    final_waypoint_eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance independently reached cursors without exceeding route lengths.

    Control targets advance from root proximity. Intermediate physical targets
    require root proximity and named support.
    Final completion requires root proximity and ``final_waypoint_eligible``.

    Args:
        active_indices: Current waypoint index for each parallel environment.
            Each value is a cursor into that environment's configured route.
        waypoint_counts: Number of valid waypoints in each environment's route.
        root_within_radius: Whether each robot root is currently close enough
            to its active waypoint.
        support_required: Whether the active waypoint identifies a physical
            support instead of serving only as a route-control target.
        supported: Whether a recently contacted foot is on the active
            waypoint's intended support.
        route_state_eligible: Whether the robot may advance route state. A
            chassis-contacting robot is ineligible.
        final_waypoint_eligible: Whether each environment has named-support
            contact and satisfies the whole-body stability conditions required
            at its final waypoint.

    Returns:
        The next active indices and completed-course mask.
    """

    # Each environment can follow a route of a different length, so determine
    # its final index from its own waypoint count rather than a shared constant.
    final_waypoint = active_indices == waypoint_counts - 1

    # Physical targets additionally require contact on their named support.
    # Final completion retains the stricter whole-body stability conditions
    # collected in ``final_waypoint_eligible``.
    control_reached = (~support_required) & root_within_radius
    physical_reached = support_required & root_within_radius & supported
    advance_cursor = (
        (~final_waypoint) & route_state_eligible & (control_reached | physical_reached)
    )

    completed_course = (
        final_waypoint
        & physical_reached
        & route_state_eligible
        & final_waypoint_eligible
    )

    # Adding a Boolean tensor increments selected cursors by exactly one. Final
    # cursors are excluded above, so no index can exceed its route length.
    next_active_indices = active_indices + advance_cursor
    return next_active_indices, completed_course


def _route_progress_m(
    env: ManagerBasedRLEnv,
    robot_xy: torch.Tensor,
    *,
    route_half_width_m: float,
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
    squared_lengths = torch.sum(segment_vectors.square(), dim=-1).clamp_min(
        torch.finfo(robot_xy.dtype).eps
    )
    relative_position = robot_xy - segment_starts
    fraction = (
        torch.sum(relative_position * segment_vectors, dim=-1) / squared_lengths
    ).clamp(0.0, 1.0)
    # [num_envs]: active-segment length and distance completed before it.
    segment_lengths = torch.sqrt(squared_lengths)
    unit_directions = segment_vectors / segment_lengths[:, None]
    longitudinal = torch.sum(
        relative_position * unit_directions,
        dim=-1,
    )
    lateral_vector = relative_position - longitudinal[:, None] * unit_directions
    inside_corridor = torch.linalg.norm(lateral_vector, dim=-1) <= route_half_width_m
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


# Route-envelope geometry.


def _finite_route_cross_track_error_m(
    root_xy: torch.Tensor,
    waypoints_xy: torch.Tensor,
    waypoint_counts: torch.Tensor,
) -> torch.Tensor:
    """Measure batched point-to-polyline distance using finite segments only."""

    import torch

    segment_starts = torch.cat(
        (torch.zeros_like(waypoints_xy[:, :1]), waypoints_xy[:, :-1]),
        dim=1,
    )
    segment_vectors = waypoints_xy - segment_starts
    squared_lengths = torch.sum(segment_vectors.square(), dim=-1)
    relative_positions = root_xy[:, None, :] - segment_starts
    fractions = (
        torch.sum(relative_positions * segment_vectors, dim=-1)
        / squared_lengths.clamp_min(torch.finfo(root_xy.dtype).eps)
    ).clamp(0.0, 1.0)
    closest_points = segment_starts + fractions[..., None] * segment_vectors
    distances = torch.linalg.norm(root_xy[:, None, :] - closest_points, dim=-1)

    segment_indices = torch.arange(waypoints_xy.shape[1], device=root_xy.device)
    valid_segments = segment_indices[None, :] < waypoint_counts[:, None]
    distances = distances.masked_fill(~valid_segments, torch.inf)
    minimum = torch.amin(distances, dim=1)
    return torch.where(
        torch.isfinite(minimum), minimum, torch.full_like(minimum, torch.inf)
    )


# Marker synchronization.


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
