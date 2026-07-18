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
    goal_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Derive active terrain-local waypoints, falling back before first reset."""

    table = getattr(env, "_parkour_waypoint_table", None)
    route_levels = getattr(env, "_parkour_waypoint_level", None)
    active_indices = getattr(env, "_parkour_active_waypoint_index", None)
    if (
        table is not None
        and route_levels is not None
        and active_indices is not None
        and route_levels.shape == (env.num_envs,)
        and active_indices.shape == (env.num_envs,)
    ):
        return table[route_levels, active_indices]

    goal = env.scene[goal_cfg.name]
    return goal.data.root_pos_w - env.scene.env_origins


def advance_active_waypoints(
    env: ManagerBasedRLEnv,
    *,
    reach_threshold: float,
    reach_hold_s: float,
    goal_cfg: SceneEntityCfg,
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
    if reach_hold_s < 0.0:
        raise ValueError("reach_hold_s must be non-negative.")
    required_state = (
        "_parkour_waypoint_table",
        "_parkour_waypoint_count_by_level",
        "_parkour_waypoint_level",
        "_parkour_active_waypoint_index",
    )
    if not all(hasattr(env, name) for name in required_state):
        raise RuntimeError("Active waypoints must be initialized before stepping.")

    active_indices = env._parkour_active_waypoint_index
    route_levels = env._parkour_waypoint_level
    waypoint_counts = env._parkour_waypoint_count_by_level[route_levels]
    active_positions = env._parkour_waypoint_table[route_levels, active_indices]
    robot_pos = robot._root_pos_env(env, asset_cfg)
    distance_xy = torch.linalg.norm(
        robot_pos[:, :2] - active_positions[:, :2],
        dim=-1,
    )
    # Completion is absorbing until the reset event selects a new route. This
    # prevents repeated helper calls from counting the final waypoint twice.
    already_completed = env._parkour_course_completed
    within_radius = (distance_xy < reach_threshold) & ~already_completed

    # Intermediate waypoints only select a new direction. The final waypoint
    # retains the existing safety rule that a collapsed robot is not successful.
    clearance = queries._base_clearance(env, asset_cfg)
    min_clearance = get_min_clearance(env).to(
        device=clearance.device,
        dtype=clearance.dtype,
    )
    final_waypoint_eligible = clearance > min_clearance

    (
        next_indices,
        next_hold_times_s,
        _,
        completed_course,
    ) = _advance_route_state(
        active_indices,
        waypoint_counts,
        env._parkour_waypoint_hold_time_s,
        within_radius,
        final_waypoint_eligible,
        step_dt_s=float(env.step_dt),
        reach_hold_s=reach_hold_s,
    )

    advanced = next_indices != active_indices
    env._parkour_active_waypoint_index[:] = next_indices
    env._parkour_waypoint_hold_time_s[:] = next_hold_times_s
    env._parkour_active_waypoint_changed[:] = advanced
    env._parkour_course_completed[:] = completed_course | already_completed

    advanced_env_ids = torch.nonzero(advanced, as_tuple=False).flatten()
    if advanced_env_ids.numel() > 0:
        next_waypoints = env._parkour_waypoint_table[
            route_levels[advanced_env_ids],
            next_indices[advanced_env_ids],
        ]
        _write_goal_marker(env, advanced_env_ids, next_waypoints, goal_cfg)

    return env._parkour_course_completed


def course_completed_this_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the final-waypoint event computed by the termination manager."""

    import torch

    completed = getattr(env, "_parkour_course_completed", None)
    if completed is None:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return completed


def reset_active_waypoints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    logical_levels: torch.Tensor,
    curriculum_cfg: ParkourCurriculumCfg,
    goal_cfg: SceneEntityCfg,
) -> None:
    """Reset selected environments to the first waypoint of their new route.

    This initializes the shared waypoint tables when necessary, assigns each
    selected environment its logical curriculum level, clears its route
    progress and per-step event flags, and moves its goal marker to waypoint
    zero. Passing ``None`` as ``env_ids`` resets every environment.

    Args:
        env: Vectorized manager-based environment that owns the route buffers
            and goal-marker scene entity.
        env_ids: Indices of the environments being reset, or ``None`` for all
            environments.
        logical_levels: Curriculum-level index for each selected environment.
            Its shape must match the resolved ``env_ids`` tensor.
        curriculum_cfg: Course definitions used to build the waypoint table.
        goal_cfg: Scene-entity selection for the visible goal marker.
    """

    import torch

    from .._shared.runtime import _all_env_ids

    env_ids = _all_env_ids(env, env_ids)
    logical_levels = logical_levels.to(device=env.device, dtype=torch.long)
    if logical_levels.shape != env_ids.shape:
        raise ValueError("logical_levels must contain one level per reset environment.")

    goal = env.scene[goal_cfg.name]
    dtype = goal.data.default_root_state.dtype
    _ensure_waypoint_state(env, curriculum_cfg, dtype=dtype)

    num_levels = env._parkour_waypoint_table.shape[0]
    if torch.any((logical_levels < 0) | (logical_levels >= num_levels)):
        raise ValueError("logical_levels contains an out-of-range course level.")

    env._parkour_waypoint_level[env_ids] = logical_levels
    env._parkour_active_waypoint_index[env_ids] = 0
    env._parkour_waypoint_hold_time_s[env_ids] = 0.0
    env._parkour_active_waypoint_changed[env_ids] = False
    env._parkour_course_completed[env_ids] = False

    first_waypoints = env._parkour_waypoint_table[logical_levels, 0]
    _write_goal_marker(env, env_ids, first_waypoints, goal_cfg)


def _advance_route_state(
    active_indices: torch.Tensor,
    waypoint_counts: torch.Tensor,
    hold_times_s: torch.Tensor,
    within_radius: torch.Tensor,
    final_waypoint_eligible: torch.Tensor,
    *,
    step_dt_s: float,
    reach_hold_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance independently reached cursors without exceeding route lengths.

    ``final_waypoint_eligible`` lets the runtime impose a stricter completion
    condition on the last waypoint while intermediate routing points advance
    from proximity alone.

    Args:
        active_indices: Current waypoint index for each parallel environment.
            Each value is a cursor into that environment's configured route.
        waypoint_counts: Number of valid waypoints in each environment's route.
        hold_times_s: Time each environment has continuously remained within
            its active waypoint radius.
        within_radius: Whether each environment is currently close enough to
            its active waypoint.
        final_waypoint_eligible: Whether each environment satisfies the extra
            completion condition required at its final waypoint.
        step_dt_s: Duration represented by this state update, in seconds.
        reach_hold_s: Continuous time required inside the waypoint radius.

    Returns:
        The next active indices, next dwell times, reached-waypoint mask, and
        completed-course mask.
    """

    # Accumulate dwell time only while an environment remains inside its active
    # waypoint radius. Multiplication by the Boolean mask resets the timer to
    # zero immediately after the robot leaves that radius.
    next_hold_times_s = (hold_times_s + step_dt_s) * within_radius

    # Account for float32 accumulation (for example, 0.08 + 0.02 may be stored
    # just below 0.10) without shortening the dwell by a meaningful duration.
    dwell_satisfied = within_radius & (next_hold_times_s >= reach_hold_s - step_dt_s * 1.0e-6)

    # Each environment can follow a route of a different length, so determine
    # its final index from its own waypoint count rather than a shared constant.
    final_waypoint = active_indices == waypoint_counts - 1

    # Proximity and dwell time are sufficient for an intermediate waypoint.
    # The final waypoint additionally requires the caller's completion gate,
    # such as the minimum-clearance check used by the runtime.
    reached_waypoint = dwell_satisfied & (~final_waypoint | final_waypoint_eligible)

    # Split a reached event into mutually exclusive outcomes: the final target
    # completes the course, while an intermediate target advances the cursor.
    completed_course = reached_waypoint & final_waypoint
    advance_cursor = reached_waypoint & ~final_waypoint

    # Adding a Boolean tensor increments selected cursors by exactly one. Final
    # cursors are excluded above, so no index can exceed its route length.
    next_active_indices = active_indices + advance_cursor

    # Begin a fresh dwell interval after reaching a waypoint; otherwise retain
    # the accumulated time for environments that are still approaching it.
    next_hold_times_s = next_hold_times_s * ~reached_waypoint
    return (
        next_active_indices,
        next_hold_times_s,
        reached_waypoint,
        completed_course,
    )


def _ensure_waypoint_state(
    env: ManagerBasedRLEnv,
    curriculum_cfg: ParkourCurriculumCfg,
    *,
    dtype: torch.dtype,
) -> None:
    """Create route constants and per-environment state on the runtime device."""

    import torch

    from .._shared.runtime import _get_or_init_env_buffer

    route_signature = tuple(tuple(waypoint.position for waypoint in level.waypoints) for level in curriculum_cfg.levels)
    expected_device = torch.device(env.device)

    # ``_parkour_waypoint_table`` stores every level's terrain-local XYZ
    # waypoints in one padded tensor shaped [num_levels, max_waypoints, 3].
    # ``_parkour_waypoint_count_by_level`` records each unpadded route length.
    table = getattr(env, "_parkour_waypoint_table", None)
    counts = getattr(env, "_parkour_waypoint_count_by_level", None)

    # The plain-Python signature acts as a cache key. Rebuild the device-side
    # constants if the configured routes, target device, or coordinate dtype
    # changed since the previous initialization.
    needs_route_table = (
        getattr(env, "_parkour_waypoint_route_signature", None) != route_signature
        or table is None
        or counts is None
        or table.device != expected_device
        or counts.device != expected_device
        or table.dtype != dtype
    )
    if needs_route_table:
        max_waypoints = max(len(route) for route in route_signature)
        table = torch.empty(
            (len(route_signature), max_waypoints, 3),
            device=env.device,
            dtype=dtype,
        )
        counts = torch.empty(
            len(route_signature),
            device=env.device,
            dtype=torch.long,
        )
        for level_index, route in enumerate(route_signature):
            route_tensor = torch.tensor(route, device=env.device, dtype=dtype)

            # Routes may contain different numbers of waypoints. Store the
            # actual length separately because the table must be rectangular
            # for vectorized indexing across levels and environments.
            count = route_tensor.shape[0]
            table[level_index, :count] = route_tensor

            # Fill the unused suffix with the final waypoint instead of leaving
            # uninitialized memory. ``counts`` remains authoritative, so route
            # cursors never intentionally advance into these padding entries.
            table[level_index, count:] = route_tensor[-1]
            counts[level_index] = count
        env._parkour_waypoint_table = table
        env._parkour_waypoint_count_by_level = counts

        # Remember which immutable route definition produced the cached table.
        env._parkour_waypoint_route_signature = route_signature

    # One integer per environment selecting a level row in the waypoint table.
    # Reset derives it from the environment's terrain level, after any
    # curriculum promotion or demotion. Keeping this selection with the route
    # state ensures later lookups use the same course chosen for that episode.
    _get_or_init_env_buffer(
        env,
        "_parkour_waypoint_level",
        torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
    )

    # One integer cursor per environment selecting a waypoint column within its
    # chosen route row. It starts at zero, advances by one after each reached
    # intermediate waypoint, and remains at the final index after completion.
    _get_or_init_env_buffer(
        env,
        "_parkour_active_waypoint_index",
        torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
    )

    # One floating-point dwell timer per environment, measured in seconds. It
    # accumulates over consecutive control steps inside the active waypoint's
    # reach radius. Leaving the radius, reaching the waypoint, or resetting the
    # environment clears it to zero, which prevents brief fly-bys from counting.
    _get_or_init_env_buffer(
        env,
        "_parkour_waypoint_hold_time_s",
        torch.zeros(env.num_envs, device=env.device, dtype=dtype),
    )

    # One Boolean event per environment. The success termination term calls
    # ``advance_active_waypoints`` once per control step, before rewards, and
    # overwrites this entire buffer with the environments whose intermediate
    # waypoint cursor advanced. The progress reward can therefore suppress the
    # invalid old-target/new-target distance comparison during that same step;
    # the next route update normally changes the event back to false. Episode
    # reset also clears it explicitly for the selected environments.
    _get_or_init_env_buffer(
        env,
        "_parkour_active_waypoint_changed",
        torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
    )

    # One Boolean completion state per environment. Reaching the final waypoint
    # with the required clearance sets it to true; it then remains true until
    # reset. This absorbing behavior prevents duplicate completion events and
    # supplies both the success termination and its sparse completion reward.
    _get_or_init_env_buffer(
        env,
        "_parkour_course_completed",
        torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
    )


def _write_goal_marker(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    waypoint_pos: torch.Tensor,
    goal_cfg: SceneEntityCfg,
) -> None:
    """Move selected kinematic markers from terrain-local to world positions."""

    import torch

    goal = env.scene[goal_cfg.name]

    # Each default root-state row contains 13 values:
    # ``[position_xyz(3), quaternion_wxyz(4), linear_velocity_xyz(3),
    # angular_velocity_xyz(3)]``. Select the requested environments and retain
    # only the first seven pose values because velocity is written separately.
    goal_pose = goal.data.default_root_state[env_ids, :7].clone()
    goal_pose[:, :3] = waypoint_pos + env.scene.env_origins[env_ids]
    zero_velocity = torch.zeros(
        (env_ids.numel(), 6),
        device=env.device,
        dtype=goal_pose.dtype,
    )
    # ``goal_pose`` has shape ``[len(env_ids), 7]``. Each row contains the
    # world-frame position ``(x, y, z)`` followed by the root quaternion
    # ``(w, x, y, z)`` for one selected goal marker.
    goal.write_root_pose_to_sim(goal_pose, env_ids=env_ids)

    # ``zero_velocity`` has shape ``[len(env_ids), 6]``. Each row contains
    # world-frame linear velocity ``(vx, vy, vz)`` followed by angular
    # velocity ``(wx, wy, wz)``, all zero so the kinematic marker stays still.
    goal.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
