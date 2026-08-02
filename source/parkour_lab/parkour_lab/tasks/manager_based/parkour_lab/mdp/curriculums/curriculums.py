from dataclasses import dataclass
from typing import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from .._shared.runtime import _all_env_ids
from ..navigation import route
from . import config


# Startup lifecycle.


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
        env: Vectorized manager-based environment whose terrain assignments
            and origins are initialized.
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

    terrain.terrain_levels[env_ids] = levels
    # terrain_origins has shape [num_levels, num_terrain_types, 3] and is the
    # generated-tile lookup table (difficulty row, terrain column/type, XYZ).
    # env_origins has shape [num_envs, 3] and stores the selected tile origin
    # for each environment.
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        levels, terrain.terrain_types[env_ids]
    ]


# Curriculum update lifecycle.


@dataclass(frozen=True, slots=True)
class CurriculumBatch:
    """Transient reset-batch evidence; ``completed`` selects valid outcomes."""

    # Outcome masks.
    completed: torch.Tensor
    poor_failure: torch.Tensor
    success: torch.Tensor

    # Numeric outcomes.
    level_change: torch.Tensor
    normalized_progress: torch.Tensor


class ParkourTerrainCurriculum(ManagerTermBase):
    """Update terrain difficulty while owning its per-environment memory."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        # These are the only values that must survive between episodes.
        # Family, level, progress, and outcomes remain derivable from their
        # authoritative terrain, route, and termination owners.
        self.consecutive_successes = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.long,
        )
        self.consecutive_poor_failures = torch.zeros_like(self.consecutive_successes)
        self.demotion_grace_episodes_remaining = torch.zeros_like(
            self.consecutive_successes
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Preserve curriculum evidence across ordinary episode resets."""

        # CurriculumManager invokes this immediately after every curriculum
        # computation. Streaks and grace intentionally span episodes, so only
        # constructing a new term should clear them.
        pass

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | slice,
        curriculum_cfg: config.ParkourCurriculumCfg,
        terrain_layout: config.ParkourTerrainLayout,
    ) -> dict[str, torch.Tensor]:
        """Update selected environments' difficulty from terminal outcomes."""

        env_ids = _all_env_ids(env, env_ids)
        terrain: TerrainImporter = env.scene.terrain
        population_env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.long,
        )
        population_family_indices = _family_indices_for_terrain_columns(
            env,
            terrain,
            population_env_ids,
            terrain_layout,
        )

        if env_ids.numel() == 0:
            empty_bool = torch.empty(0, device=env.device, dtype=torch.bool)
            return _curriculum_metrics(
                terrain.terrain_levels,
                population_family_indices,
                CurriculumBatch(
                    completed=empty_bool,
                    poor_failure=empty_bool,
                    success=empty_bool,
                    level_change=torch.empty(
                        0,
                        device=env.device,
                        dtype=torch.long,
                    ),
                    normalized_progress=torch.empty(
                        0,
                        device=env.device,
                        dtype=torch.float32,
                    ),
                ),
                curriculum_cfg,
            )

        success_event, terminal_event, failure_event = _terminal_event_masks(
            env,
            env_ids,
        )
        old_levels = terrain.terrain_levels[env_ids].clone()

        updated_successes, promotion_ready = _success_streak_transition(
            self.consecutive_successes[env_ids],
            success_event,
            failure_event,
            required_successes=curriculum_cfg.promotion_successes_required,
        )
        self.consecutive_successes[env_ids] = updated_successes

        # Read progress before changing terrain rows: route state still points
        # at the course whose episode just ended, and its length is therefore
        # the correct denominator for this terminal outcome.
        normalized_progress = route.normalized_course_progress(env, env_ids)
        poor_failure_event = _demotion_transition_mask(
            normalized_progress,
            failure_event,
            demotion_progress_fraction=curriculum_cfg.demotion_progress_fraction,
        )
        updated_grace, demotion_eligible = _demotion_grace_transition(
            self.demotion_grace_episodes_remaining[env_ids],
            terminal_event,
        )
        self.demotion_grace_episodes_remaining[env_ids] = updated_grace
        poor_failure_eligible = poor_failure_event & demotion_eligible
        # Flat terrain has no lower row. Do not accumulate a stale demotion
        # streak that cannot yet produce a level change.
        poor_failure_eligible = poor_failure_eligible & (old_levels > 0)
        updated_poor_failures, demotion_ready = _demotion_streak_transition(
            self.consecutive_poor_failures[env_ids],
            poor_failure_eligible,
            terminal_event,
            required_failures=curriculum_cfg.demotion_failures_required,
        )
        self.consecutive_poor_failures[env_ids] = updated_poor_failures

        move_up = promotion_ready & (old_levels < curriculum_cfg.max_level)
        move_down = demotion_ready & (old_levels > 0) & (~move_up)

        # TerrainImporter updates both authoritative row indices and origins.
        terrain.update_env_origins(
            env_ids=env_ids,
            move_up=move_up,
            move_down=move_down,
        )

        level_change = terrain.terrain_levels[env_ids] - old_levels
        changed_env_ids = env_ids[level_change != 0]
        self.consecutive_successes[changed_env_ids] = 0
        self.consecutive_poor_failures[changed_env_ids] = 0
        self.demotion_grace_episodes_remaining[env_ids[level_change > 0]] = (
            curriculum_cfg.post_promotion_grace_episodes
        )
        self.demotion_grace_episodes_remaining[env_ids[level_change < 0]] = 0

        return _curriculum_metrics(
            terrain.terrain_levels,
            population_family_indices,
            CurriculumBatch(
                completed=terminal_event,
                poor_failure=poor_failure_event,
                success=success_event,
                level_change=level_change,
                normalized_progress=normalized_progress,
            ),
            curriculum_cfg,
        )


# Episode-reset lifecycle.


def reset_routes(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    terrain_layout: config.ParkourTerrainLayout,
    curriculum_cfg: config.ParkourCurriculumCfg = config.DEFAULT_PARKOUR_CURRICULUM,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the route cursor and waypoint marker for new episodes.

    The terrain level has already been updated by the CurriculumManager before
    reset events are applied.

    Args:
        env: Vectorized environment containing terrain, route, and
            waypoint-marker state.
        env_ids: Environments beginning a new episode, or ``None`` for all
            environments.
        terrain_layout: Mapping from physical terrain columns to obstacle
            families.
        curriculum_cfg: Course matrix used to resolve the selected family and
            difficulty into route geometry.
        waypoint_marker_cfg: Scene-entity selection for the visible marker.
        asset_cfg: Scene-entity selection whose post-reset root position seeds
            route crossing history.
    """

    env_ids = _all_env_ids(env, env_ids)

    terrain: TerrainImporter = env.scene.terrain

    # Terrain row N is generated directly from curriculum level N, so the
    # importer's row indices are already the logical course-level indices.
    difficulty_indices = terrain.terrain_levels[env_ids]
    family_indices = _family_indices_for_terrain_columns(
        env,
        terrain,
        env_ids,
        terrain_layout,
    )

    # Curriculum updates run before reset events, so ``difficulty_indices``
    # already contains any promotion or demotion selected for this new episode.
    route.reset_routes(
        env=env,
        env_ids=env_ids,
        family_indices=family_indices,
        difficulty_indices=difficulty_indices,
        curriculum_cfg=curriculum_cfg,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
    )


# Curriculum reporting.


def _curriculum_metrics(
    population_levels: torch.Tensor,
    population_family_indices: torch.Tensor,
    batch: CurriculumBatch,
    curriculum_cfg: config.ParkourCurriculumCfg,
) -> dict[str, torch.Tensor]:
    """Build a compact curriculum health summary for experiment logging.

    Population metrics use every parallel environment. Episode and transition
    metrics are conditional on completed episodes, so initial and manual resets
    do not masquerade as failed or zero-progress episodes.
    """

    completed = batch.completed.to(dtype=torch.float32)
    completed_count = completed.sum().clamp_min(1.0)

    stats = {
        "level/mean": population_levels.float().mean(),
        "level/top_fraction": (population_levels == curriculum_cfg.max_level)
        .float()
        .mean(),
        "episode/mean_normalized_progress": (
            batch.normalized_progress.float() * completed
        ).sum()
        / completed_count,
        "episode/poor_failure_rate": batch.poor_failure.float().sum() / completed_count,
        "episode/success_rate": batch.success.float().sum() / completed_count,
        "transition/demotion_rate": (batch.level_change < 0).float().sum()
        / completed_count,
        "transition/promotion_rate": (batch.level_change > 0).float().sum()
        / completed_count,
    }
    for family_index, family_name in enumerate(curriculum_cfg.family_names):
        family_weights = (population_family_indices == family_index).float()
        family_count = family_weights.sum().clamp_min(1.0)
        stats[f"family/{family_name}/mean_level"] = (
            population_levels.float() * family_weights
        ).sum() / family_count
    return stats


# Terrain-selection helpers.


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
                f"initial_level_override must be in [0, {curriculum_cfg.max_level}], got {initial_level_override}."
            )
        return torch.full_like(env_ids, initial_level_override)

    if curriculum_cfg.distribute_initial_levels:
        # Deterministically balanced, which is more reproducible than relying on
        # TerrainImporter's random 0..max initialization.
        return torch.remainder(env_ids, curriculum_cfg.initial_level + 1)
    return torch.full_like(env_ids, curriculum_cfg.initial_level)


# Pure curriculum transitions.


def _demotion_grace_transition(
    grace_remaining: torch.Tensor,
    terminal_event: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume grace on completed episodes and return demotion eligibility.

    Eligibility is based on the value at the start of the completed episode,
    so a grace value of three excludes exactly three terminal outcomes. Manual
    and initial resets do not consume the allowance.
    """

    updated_grace = torch.where(
        terminal_event,
        torch.clamp(grace_remaining - 1, min=0),
        grace_remaining,
    )
    demotion_eligible = terminal_event & (grace_remaining == 0)
    return updated_grace, demotion_eligible


def _demotion_streak_transition(
    demotion_streaks: torch.Tensor,
    poor_failure_event: torch.Tensor,
    terminal_event: torch.Tensor,
    *,
    required_failures: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update poor-failure counts and return demotion readiness.

    Only a terminal failure below the progress threshold increments the streak.
    Any other completed episode clears it, while initial and manual resets
    preserve it because they provide no outcome evidence.
    """

    incremented = torch.clamp(demotion_streaks + 1, max=required_failures)
    updated_streaks = torch.where(
        poor_failure_event,
        incremented,
        demotion_streaks,
    )
    completed_without_poor_failure = terminal_event & (~poor_failure_event)
    updated_streaks = torch.where(
        completed_without_poor_failure,
        torch.zeros_like(demotion_streaks),
        updated_streaks,
    )
    demotion_ready = poor_failure_event & (updated_streaks >= required_failures)
    return updated_streaks, demotion_ready


def _demotion_transition_mask(
    normalized_progress: torch.Tensor,
    failure_event: torch.Tensor,
    *,
    demotion_progress_fraction: float,
) -> torch.Tensor:
    """Return failed episodes that completed too little of their route.

    The strict comparison leaves exact-threshold outcomes unchanged. Successful
    completions, initial resets, and manual resets remain ineligible.
    """

    return failure_event & (normalized_progress < demotion_progress_fraction)


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


def _terminal_event_masks(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return success, terminal, and failure masks for a reset batch.

    CurriculumManager runs only for reset environments. ``reset_buf`` therefore
    covers trunk contact, timeout, and future failure terminations. Requiring a
    positive episode length keeps initial and repeated manual resets neutral.
    Trunk contact wins if success and a crash fire during the same step: a
    robot cannot earn mastery by arriving at the exit gate while collapsed.
    """

    has_completed_step = env.episode_length_buf[env_ids] > 0
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is None:
        terminal_event = torch.zeros_like(has_completed_step)
    else:
        terminal_event = (
            reset_buf[env_ids].to(
                device=env.device,
                dtype=torch.bool,
            )
            & has_completed_step
        )
    raw_success_event = (
        env.termination_manager.get_term("success")[env_ids].to(
            device=env.device,
            dtype=torch.bool,
        )
        & terminal_event
    )
    trunk_contact_event = (
        env.termination_manager.get_term("trunk_contact")[env_ids].to(
            device=env.device,
            dtype=torch.bool,
        )
        & terminal_event
    )
    success_event = raw_success_event & (~trunk_contact_event)
    failure_event = terminal_event & (~success_event)
    return success_event, terminal_event, failure_event


# Terrain-layout validation.


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
