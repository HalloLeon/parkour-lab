from typing import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from .._shared.runtime import _all_env_ids, _get_or_init_env_buffer
from ..commands import get_target_speed, set_commands
from ..navigation import route
from . import config


def initialize_parkour_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    terrain_layout: config.ParkourTerrainLayout,
    curriculum_cfg: config.ParkourCurriculumCfg = config.DEFAULT_PARKOUR_CURRICULUM,
    initial_level_override: int | None = None,
) -> None:
    """Place environments on exact terrain rows before the first reset.

    ``TerrainImporterCfg.max_init_terrain_level`` is an upper bound for random
    sampling, not an exact initial level. This startup event makes the training
    distribution explicit and is also what pins deterministic evaluation to a
    single difficulty.

    Args:
        env: Vectorized manager-based environment whose terrain assignments,
            origins, family indices, and curriculum statistics are initialized.
        env_ids: Environment indices to initialize. ``None`` selects every
            environment; otherwise only the specified environments are changed.
        terrain_layout: Authoritative relationship between physical terrain
            rows and difficulty indices, plus the family assigned to every
            physical column. Training uses balanced family blocks, while fixed
            evaluation maps every column to its selected family.
        curriculum_cfg: Ordered parkour families, difficulty levels, initial
            distribution settings, and curriculum bounds associated with the
            generated terrain layout.
        initial_level_override: Optional exact difficulty row assigned to every
            selected environment during startup. When omitted, levels are
            either balanced across rows zero through ``initial_level`` or fixed
            at ``initial_level``, as controlled by
            ``distribute_initial_levels``.
    """

    terrain: TerrainImporter = env.scene.terrain
    _validate_terrain_layout(terrain, terrain_layout, curriculum_cfg)

    env_ids = _all_env_ids(env, env_ids)
    levels = _initial_level_indices(
        env_ids,
        curriculum_cfg,
        initial_level_override,
    )
    family_indices = _family_indices_for_terrain_columns(
        env,
        terrain,
        env_ids,
        terrain_layout,
    )

    terrain.terrain_levels[env_ids] = levels
    _family_index_buffer(env)[env_ids] = family_indices
    # terrain_origins has shape [num_levels, num_terrain_types, 3] and is the
    # generated-tile lookup table (difficulty row, terrain column/type, XYZ).
    # env_origins has shape [num_envs, 3] and stores the selected tile origin
    # for each environment.
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        levels, terrain.terrain_types[env_ids]
    ]


def parkour_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    curriculum_cfg: config.ParkourCurriculumCfg = config.DEFAULT_PARKOUR_CURRICULUM,
) -> dict[str, torch.Tensor]:
    """Update selected environments' difficulty from terminal outcomes.

    Args:
        env: Vectorized environment containing terrain, route, and episode
            outcome state.
        env_ids: Environments whose completed episodes are being processed.
        curriculum_cfg: Course matrix and mastery/progress thresholds
            controlling row promotion and demotion.

    Returns:
        Scalar population and reset-batch metrics for experiment logging.
    """

    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    terrain: TerrainImporter = env.scene.terrain

    family_index_buffer = _family_index_buffer(env)
    success_streak_buffer = _success_streak_buffer(env)
    if env_ids.numel() == 0:
        empty_bool = torch.zeros(0, device=env.device, dtype=torch.bool)
        return _curriculum_stats(
            terrain=terrain,
            curriculum_cfg=curriculum_cfg,
            success_streaks=success_streak_buffer,
            family_indices=family_index_buffer[env_ids],
            maximum_progress=torch.zeros(0, device=env.device),
            promotion_ready=empty_bool,
            demotion_ready=empty_bool,
            success_event=empty_bool,
            failure_event=empty_bool,
            actual_change=torch.zeros(0, device=env.device, dtype=torch.long),
        )

    success_event, _, failure_event = _terminal_event_masks(
        env,
        env_ids,
    )
    updated_streaks, promotion_ready = _success_streak_transition(
        success_streak_buffer[env_ids],
        success_event,
        failure_event,
        required_successes=curriculum_cfg.promotion_successes_required,
    )
    success_streak_buffer[env_ids] = updated_streaks

    maximum_progress, demotion_ready = _progress_transition_decisions(
        env,
        env_ids,
        failure_event,
        curriculum_cfg,
    )

    old_levels = terrain.terrain_levels[env_ids].clone()
    move_up = promotion_ready & (old_levels < curriculum_cfg.max_level)
    move_down = demotion_ready & (old_levels > 0) & (~move_up)

    # Let TerrainImporter update both its authoritative row indices and the
    # corresponding per-environment origins so those two pieces of state cannot
    # diverge after promotion or demotion.
    terrain.update_env_origins(env_ids=env_ids, move_up=move_up, move_down=move_down)

    new_levels = terrain.terrain_levels[env_ids]
    actual_change = new_levels - old_levels
    level_changed = actual_change != 0
    success_streak_buffer[env_ids[level_changed]] = 0

    return _curriculum_stats(
        terrain=terrain,
        curriculum_cfg=curriculum_cfg,
        success_streaks=success_streak_buffer,
        family_indices=family_index_buffer[env_ids],
        maximum_progress=maximum_progress,
        promotion_ready=promotion_ready,
        demotion_ready=demotion_ready,
        success_event=success_event,
        failure_event=failure_event,
        actual_change=actual_change,
    )


def reset_routes_and_commands(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: config.ParkourCurriculumCfg = config.DEFAULT_PARKOUR_CURRICULUM,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> None:
    """Reset the route cursor, goal marker, and commands for selected environments.

    The terrain level has already been updated by the CurriculumManager before
    reset events are applied.

    Args:
        env: Vectorized environment containing terrain, route, goal-marker, and
            command state.
        env_ids: Environments beginning a new episode, or ``None`` for all
            environments.
        curriculum_cfg: Course matrix used to resolve the selected family and
            difficulty into route geometry and scalar commands.
        goal_cfg: Scene-entity selection for the visible waypoint marker.
    """

    env_ids = _all_env_ids(env, env_ids)

    terrain: TerrainImporter = env.scene.terrain

    # Terrain row N is generated directly from curriculum level N, so the
    # importer's row indices are already the logical course-level indices.
    difficulty_indices = terrain.terrain_levels[env_ids]
    family_indices = _family_index_buffer(env)[env_ids]

    # Curriculum updates run before reset events, so ``difficulty_indices``
    # already contains any promotion or demotion selected for this new episode.
    # Reset only these environments to waypoint zero of that newly selected
    # route. The route helper returns the same flattened course indices it stores
    # as authoritative per-environment route state.
    course_indices = route.reset_routes(
        env,
        env_ids,
        family_indices,
        difficulty_indices,
        curriculum_cfg,
        goal_cfg,
    )
    set_commands(
        env=env,
        env_ids=env_ids,
        target_speed=env._parkour_target_speed_by_course[course_indices],
        min_clearance=env._parkour_min_clearance_by_course[course_indices],
    )


def _curriculum_stats(
    terrain: TerrainImporter,
    curriculum_cfg: config.ParkourCurriculumCfg,
    success_streaks: torch.Tensor,
    family_indices: torch.Tensor,
    maximum_progress: torch.Tensor,
    promotion_ready: torch.Tensor,
    demotion_ready: torch.Tensor,
    success_event: torch.Tensor,
    failure_event: torch.Tensor,
    actual_change: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build scalar curriculum metrics for TensorBoard/W&B logging.

    Level-distribution metrics describe the complete parallel environment
    population. Event, transition, progress, and per-family metrics describe
    only the environments in the current reset batch.
    """

    population_levels = terrain.terrain_levels
    zero = torch.zeros((), device=population_levels.device, dtype=torch.float32)

    def _event_rate(event: torch.Tensor) -> torch.Tensor:
        return event.float().mean() if event.numel() > 0 else zero

    stats = {
        "mean_level": population_levels.float().mean(),
        "min_level": population_levels.min().float(),
        "max_level": population_levels.max().float(),
        "top_level_fraction": (
            (population_levels == curriculum_cfg.max_level).float().mean()
        ),
        "mean_success_streak": success_streaks.float().mean(),
        "success_rate": _event_rate(success_event),
        "failure_rate": _event_rate(failure_event),
        "mean_max_course_progress_m": (
            maximum_progress.float().mean() if maximum_progress.numel() > 0 else zero
        ),
        "promotion_ready_rate": _event_rate(promotion_ready),
        "demotion_ready_rate": _event_rate(demotion_ready),
        "promotion_rate": _event_rate(actual_change > 0),
        "demotion_rate": _event_rate(actual_change < 0),
    }
    for family_index, family_name in enumerate(curriculum_cfg.family_names):
        family_mask = family_indices == family_index
        stats[f"family/{family_name}/reset_fraction"] = _event_rate(family_mask)
        stats[f"family/{family_name}/success_rate"] = (
            _event_rate(success_event[family_mask])
            if torch.any(family_mask)
            else zero
        )
        stats[f"family/{family_name}/mean_max_course_progress_m"] = (
            maximum_progress[family_mask].float().mean()
            if torch.any(family_mask)
            else zero
        )
    return stats


def _family_index_buffer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the authoritative per-environment obstacle-family buffer."""

    return _get_or_init_env_buffer(
        env,
        "_parkour_family_index",
        torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
    )


def _family_indices_for_terrain_columns(
    env: ManagerBasedRLEnv,
    terrain: TerrainImporter,
    env_ids: torch.Tensor,
    terrain_layout: config.ParkourTerrainLayout,
) -> torch.Tensor:
    """Resolve selected environments' terrain columns to obstacle families.

    ``terrain_layout.family_index_by_column`` has one semantic family index for
    every physical terrain column. ``terrain_types`` selects a column for each
    environment; indexing the mapping with those column identifiers produces
    one family index per selected environment. The startup layout validation
    has already established that the mapping covers every generated column and
    refers only to configured families.
    """

    family_by_column = torch.as_tensor(
        terrain_layout.family_index_by_column,
        device=env.device,
        dtype=torch.long,
    )
    return family_by_column[terrain.terrain_types[env_ids]]


def _initial_level_indices(
    env_ids: torch.Tensor,
    curriculum_cfg: config.ParkourCurriculumCfg,
    initial_level_override: int | None,
) -> torch.Tensor:
    """Select one deterministic startup difficulty for each environment.

    Fixed evaluation fills every selected environment with the explicit
    override. Training either balances environments over rows zero through the
    configured initial level or places all environments on that initial level.
    """

    if initial_level_override is not None:
        if not 0 <= initial_level_override <= curriculum_cfg.max_level:
            raise ValueError(
                "initial_level_override must be in "
                f"[0, {curriculum_cfg.max_level}], got {initial_level_override}."
            )
        return torch.full_like(env_ids, initial_level_override)

    if curriculum_cfg.distribute_initial_levels:
        # Deterministically balanced, which is more reproducible than relying on
        # TerrainImporter's random 0..max initialization.
        return torch.remainder(env_ids, curriculum_cfg.initial_level + 1)
    return torch.full_like(env_ids, curriculum_cfg.initial_level)


def _progress_transition_decisions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    failure_event: torch.Tensor,
    curriculum_cfg: config.ParkourCurriculumCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return progress and the demotion decision for a reset batch.

    Initial construction may ask the curriculum manager to process a reset
    before route state exists. Such resets remain neutral and report zero
    progress. Once route state is available, demotion uses the greatest
    progress reached during the episode, configured target speed, and completed
    episode duration. Promotion is handled independently by successful-course
    streaks.

    The route subsystem supplies the monotonic
    ``_parkour_max_course_progress_m`` buffer, which stores the greatest
    cumulative XY route progress reached in the current episode and is cleared
    on reset.

    Target speeds come from the per-environment command buffer selected when
    each route was reset.
    """

    route_initialized = hasattr(env, "_parkour_max_course_progress_m")
    if not route_initialized:
        maximum_progress = torch.zeros(
            env_ids.numel(),
            device=env.device,
            dtype=torch.float32,
        )
        return maximum_progress, torch.zeros_like(failure_event)

    maximum_progress = env._parkour_max_course_progress_m[env_ids]
    target_speeds = get_target_speed(env)[env_ids].to(
        dtype=maximum_progress.dtype
    )
    episode_duration = env.episode_length_buf[env_ids].to(
        dtype=maximum_progress.dtype
    ) * float(env.step_dt)
    demotion_ready = _demotion_transition_mask(
        maximum_progress,
        target_speeds * episode_duration,
        failure_event,
        demotion_expected_distance_fraction=(
            curriculum_cfg.demotion_expected_distance_fraction
        ),
    )
    return maximum_progress, demotion_ready


def _success_streak_buffer(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return consecutive successful completions for every environment."""

    return _get_or_init_env_buffer(
        env,
        "_parkour_success_streak",
        torch.zeros(env.num_envs, device=env.device, dtype=torch.long),
    )


def _success_streak_transition(
    success_streaks: torch.Tensor,
    success_event: torch.Tensor,
    failure_event: torch.Tensor,
    *,
    required_successes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update consecutive-success counts and return promotion readiness.

    Manual resets preserve the current streak because they are neither success
    nor failure outcomes. Successful completions increment and saturate at the
    configured requirement; any terminal failure clears the streak.
    """

    incremented = torch.clamp(success_streaks + 1, max=required_successes)
    updated_streaks = torch.where(success_event, incremented, success_streaks)
    updated_streaks = torch.where(
        failure_event,
        torch.zeros_like(success_streaks),
        updated_streaks,
    )
    promotion_ready = success_event & (updated_streaks >= required_successes)
    return updated_streaks, promotion_ready


def _demotion_transition_mask(
    maximum_progress: torch.Tensor,
    expected_distances: torch.Tensor,
    failure_event: torch.Tensor,
    *,
    demotion_expected_distance_fraction: float,
) -> torch.Tensor:
    """Return which failed episodes made too little commanded progress.

    The strict comparison leaves exact-threshold outcomes unchanged. Successful
    completions, initial resets, and manual resets remain ineligible.
    """

    return failure_event & (
        maximum_progress < demotion_expected_distance_fraction * expected_distances
    )


def _terminal_event_masks(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return success, terminal, and failure masks for a reset batch.

    CurriculumManager runs only for reset environments. ``reset_buf`` therefore
    covers trunk contact, timeout, and future failure terminations. Requiring a
    positive episode length keeps initial and repeated manual resets neutral.
    Success wins if several termination terms fire during the same step.
    """

    has_completed_step = env.episode_length_buf[env_ids] > 0
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is None:
        terminal_event = torch.zeros_like(has_completed_step)
    else:
        terminal_event = reset_buf[env_ids].to(
            device=env.device,
            dtype=torch.bool,
        ) & has_completed_step
    success_event = (
        env.termination_manager.get_term("success")[env_ids].to(
            device=env.device,
            dtype=torch.bool,
        )
        & terminal_event
    )
    failure_event = terminal_event & (~success_event)
    return success_event, terminal_event, failure_event


def _validate_terrain_layout(
    terrain: TerrainImporter | None,
    terrain_layout: config.ParkourTerrainLayout,
    curriculum_cfg: config.ParkourCurriculumCfg,
) -> None:
    """Validate the physical terrain grid against its semantic matrix layout.

    Isaac Lab's ``terrain_origins`` table has one physical row for each
    generated difficulty and one physical column for each terrain sampling
    slot. ``ParkourTerrainLayout`` supplies the semantic interpretation that
    Isaac Lab itself does not store: row indices are curriculum difficulty
    indices, while each column maps to one obstacle-family index. Checking the
    complete relationship once during startup prevents mesh generation, route
    lookup, and per-environment family state from silently disagreeing.
    """

    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError(
            "The parkour curriculum requires TerrainImporterCfg with terrain_type='generator'."
        )

    terrain_layout.validate_grid(
        curriculum_difficulties=curriculum_cfg.num_difficulties,
        curriculum_families=len(curriculum_cfg.families),
        terrain_columns=int(terrain.terrain_origins.shape[1]),
        terrain_rows=int(terrain.terrain_origins.shape[0]),
    )
