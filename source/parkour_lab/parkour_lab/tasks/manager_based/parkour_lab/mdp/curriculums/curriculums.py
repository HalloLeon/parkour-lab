from dataclasses import dataclass
from typing import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from .._shared.runtime import _all_env_ids
from ..navigation import route
from . import config
from .state import ParkourCurriculumState

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

    _set_terrain_levels(terrain, env_ids, levels)


# Curriculum update lifecycle.


@dataclass(frozen=True, slots=True)
class CurriculumBatch:
    """Transient reset-batch evidence for curriculum metrics."""

    # Outcome masks.
    frontier_attempt: torch.Tensor
    stalled_failure: torch.Tensor
    success: torch.Tensor

    # Numeric outcomes.
    frontier_change: torch.Tensor
    normalized_progress: torch.Tensor


class ParkourTerrainCurriculum(ManagerTermBase):
    """Apply terrain-difficulty transitions to grouped curriculum state."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        curriculum_cfg: config.ParkourCurriculumCfg = cfg.params["curriculum_cfg"]
        self.state = ParkourCurriculumState.allocate(
            env.num_envs,
            env.device,
            curriculum_cfg,
        )

    def state_dict(self) -> dict[str, object]:
        """Return portable curriculum memory for a training checkpoint."""

        return self.state.state_dict(self.cfg.params["curriculum_cfg"])

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore curriculum memory after validating its runtime layout."""

        self.state.load_state_dict(state, self.cfg.params["curriculum_cfg"])

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Preserve curriculum evidence across ordinary episode resets."""

        # CurriculumManager invokes this after every curriculum computation.
        # Frontier state and rolling evidence intentionally span episodes.
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
        state = self.state
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
        population_frontiers = torch.where(
            state.frontier_levels >= 0,
            state.frontier_levels,
            terrain.terrain_levels,
        )

        if env_ids.numel() == 0:
            empty_bool = torch.empty(0, device=env.device, dtype=torch.bool)
            return _curriculum_metrics(
                population_frontiers,
                terrain.terrain_levels,
                population_family_indices,
                CurriculumBatch(
                    frontier_attempt=empty_bool,
                    stalled_failure=empty_bool,
                    success=empty_bool,
                    frontier_change=torch.empty(
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

        attempted_levels = terrain.terrain_levels[env_ids].clone()
        uninitialized = state.frontier_levels[env_ids] < 0
        state.frontier_levels[env_ids] = torch.where(
            uninitialized,
            attempted_levels,
            state.frontier_levels[env_ids],
        )
        old_frontiers = state.frontier_levels[env_ids].clone()

        success_event, terminal_event, failure_event = _terminal_event_masks(
            env,
            env_ids,
        )
        # Read progress before changing terrain rows: route state still points
        # at the course whose episode just ended, and its length is therefore
        # the correct denominator for this terminal outcome.
        normalized_progress = route.normalized_course_progress(env, env_ids)
        stalled_failure_event = _demotion_transition_mask(
            normalized_progress,
            failure_event,
            demotion_progress_fraction=curriculum_cfg.demotion_progress_fraction,
        )
        frontier_attempt = terminal_event & (attempted_levels == old_frontiers)
        updated_grace, demotion_eligible = _demotion_grace_transition(
            state.demotion_grace_episodes_remaining[env_ids],
            frontier_attempt,
        )
        state.demotion_grace_episodes_remaining[env_ids] = updated_grace

        success_history = _rolling_evidence_transition(
            state.success_history[env_ids],
            success_event,
            frontier_attempt,
        )
        stalled_history = _rolling_evidence_transition(
            state.stalled_history[env_ids],
            stalled_failure_event,
            demotion_eligible,
        )
        state.success_history[env_ids] = success_history
        state.stalled_history[env_ids] = stalled_history

        move_up, move_down = _frontier_transition_masks(
            old_frontiers,
            success_history,
            stalled_history,
            success_event,
            stalled_failure_event,
            frontier_attempt,
            demotion_eligible,
            max_level=curriculum_cfg.max_level,
            required_successes=curriculum_cfg.promotion_successes_required,
            required_stalled_failures=curriculum_cfg.demotion_failures_required,
        )
        frontier_change = move_up.to(dtype=torch.long) - move_down.to(dtype=torch.long)
        new_frontiers = old_frontiers + frontier_change
        state.frontier_levels[env_ids] = new_frontiers

        changed = frontier_change != 0
        changed_env_ids = env_ids[changed]
        state.success_history[changed_env_ids] = False
        state.stalled_history[changed_env_ids] = False
        state.demotion_grace_episodes_remaining[env_ids[move_up]] = curriculum_cfg.post_promotion_grace_episodes
        state.demotion_grace_episodes_remaining[env_ids[move_down]] = 0

        next_levels, _ = _sample_episode_levels(
            new_frontiers,
            state.demotion_grace_episodes_remaining[env_ids],
            changed,
            bootstrap_replay_probability=curriculum_cfg.bootstrap_replay_probability,
            predecessor_replay_probability=curriculum_cfg.predecessor_replay_probability,
        )
        _set_terrain_levels(terrain, env_ids, next_levels)

        population_frontiers = torch.where(
            state.frontier_levels >= 0,
            state.frontier_levels,
            terrain.terrain_levels,
        )

        return _curriculum_metrics(
            population_frontiers,
            terrain.terrain_levels,
            population_family_indices,
            CurriculumBatch(
                frontier_attempt=frontier_attempt,
                stalled_failure=stalled_failure_event,
                success=success_event,
                frontier_change=frontier_change,
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
    population_frontier_levels: torch.Tensor,
    population_sampled_levels: torch.Tensor,
    population_family_indices: torch.Tensor,
    batch: CurriculumBatch,
    curriculum_cfg: config.ParkourCurriculumCfg,
) -> dict[str, torch.Tensor]:
    """Build a compact curriculum health summary for experiment logging.

    Population metrics use every parallel environment. Outcome and transition
    metrics use only frontier attempts, so replay episodes cannot overstate
    competence and manual resets remain neutral.
    """

    frontier_attempts = batch.frontier_attempt.to(dtype=torch.float32)
    frontier_attempt_count = frontier_attempts.sum().clamp_min(1.0)
    bootstrap_replay = (population_frontier_levels > 0) & (population_sampled_levels == 0)

    stats = {
        "frontier/mean": population_frontier_levels.float().mean(),
        "frontier/top_fraction": (population_frontier_levels == curriculum_cfg.max_level).float().mean(),
        "sampled/mean": population_sampled_levels.float().mean(),
        "sampled/replay_fraction": (population_sampled_levels < population_frontier_levels).float().mean(),
        "sampled/bootstrap_replay_fraction": bootstrap_replay.float().mean(),
        "frontier_episode/mean_normalized_progress": (batch.normalized_progress.float() * frontier_attempts).sum()
        / frontier_attempt_count,
        "frontier_episode/stalled_failure_rate": (batch.stalled_failure.float() * frontier_attempts).sum()
        / frontier_attempt_count,
        "frontier_episode/success_rate": (batch.success.float() * frontier_attempts).sum() / frontier_attempt_count,
        "transition/demotion_rate": (batch.frontier_change < 0).float().sum() / frontier_attempt_count,
        "transition/promotion_rate": (batch.frontier_change > 0).float().sum() / frontier_attempt_count,
    }
    for family_index, family_name in enumerate(curriculum_cfg.family_names):
        family_weights = (population_family_indices == family_index).float()
        family_count = family_weights.sum().clamp_min(1.0)
        stats[f"family/{family_name}/mean_frontier"] = (
            population_frontier_levels.float() * family_weights
        ).sum() / family_count
    return stats


# Terrain-selection helpers.


def _set_terrain_levels(
    terrain: TerrainImporter,
    env_ids: torch.Tensor,
    levels: torch.Tensor,
) -> None:
    """Assign exact terrain rows and their matching environment origins."""

    terrain.terrain_levels[env_ids] = levels
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        levels,
        terrain.terrain_types[env_ids],
    ]


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
    frontier_attempt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Consume grace on completed frontier attempts and return eligibility.

    Eligibility uses the value at the start of the episode, so a grace value of
    one excludes exactly the first harder attempt. Replay and manual resets do
    not consume the allowance.
    """

    updated_grace = torch.where(
        frontier_attempt,
        torch.clamp(grace_remaining - 1, min=0),
        grace_remaining,
    )
    demotion_eligible = frontier_attempt & (grace_remaining == 0)
    return updated_grace, demotion_eligible


def _rolling_evidence_transition(
    history: torch.Tensor,
    evidence: torch.Tensor,
    append_mask: torch.Tensor,
) -> torch.Tensor:
    """Append one Boolean observation to the selected histories."""

    shifted = torch.roll(history, shifts=-1, dims=1)
    shifted[:, -1] = evidence
    return torch.where(append_mask[:, None], shifted, history)


def _demotion_transition_mask(
    normalized_progress: torch.Tensor,
    failure_event: torch.Tensor,
    *,
    demotion_progress_fraction: float,
) -> torch.Tensor:
    """Return stalled failures that completed too little of their route.

    The strict comparison leaves exact-threshold outcomes unchanged. Successful
    completions, initial resets, and manual resets remain ineligible.
    """

    return failure_event & (normalized_progress < demotion_progress_fraction)


def _frontier_transition_masks(
    frontier_levels: torch.Tensor,
    success_history: torch.Tensor,
    stalled_history: torch.Tensor,
    success_event: torch.Tensor,
    stalled_failure_event: torch.Tensor,
    frontier_attempt: torch.Tensor,
    demotion_eligible: torch.Tensor,
    *,
    max_level: int,
    required_successes: int,
    required_stalled_failures: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mutually exclusive mastery promotion and stalled demotion masks."""

    move_up = (
        frontier_attempt
        & success_event
        & (torch.sum(success_history, dim=-1) >= required_successes)
        & (frontier_levels < max_level)
    )
    move_down = (
        frontier_attempt
        & stalled_failure_event
        & demotion_eligible
        & (torch.sum(stalled_history, dim=-1) >= required_stalled_failures)
        & (frontier_levels > 0)
        & (~move_up)
    )
    return move_up, move_down


def _sample_episode_levels(
    frontier_levels: torch.Tensor,
    grace_remaining: torch.Tensor,
    frontier_changed: torch.Tensor,
    *,
    bootstrap_replay_probability: float,
    predecessor_replay_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the bootstrap, immediate predecessor, or frontier."""

    replay_eligible = (frontier_levels > 0) & (~frontier_changed) & (grace_remaining == 0)
    draw = torch.rand_like(frontier_levels, dtype=torch.float32)
    replay = replay_eligible & (
        draw < bootstrap_replay_probability + predecessor_replay_probability
    )
    bootstrap_replay = replay & (draw < bootstrap_replay_probability)
    levels = torch.where(replay, frontier_levels - 1, frontier_levels)
    return torch.where(bootstrap_replay, torch.zeros_like(levels), levels), replay


def _terminal_event_masks(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return success, terminal, and failure masks for a reset batch.

    CurriculumManager runs only for reset environments. ``reset_buf`` therefore
    covers trunk contact, falls, timeout, and future failure terminations. Requiring a
    positive episode length keeps initial and repeated manual resets neutral.
    A trunk contact or fall wins if it fires with success during the same step:
    a robot cannot earn mastery by reaching the exit gate while crashing.
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
    fell_below_event = (
        env.termination_manager.get_term("fell_below_course")[env_ids].to(
            device=env.device,
            dtype=torch.bool,
        )
        & terminal_event
    )
    success_event = raw_success_event & (~trunk_contact_event) & (~fell_below_event)
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
        raise RuntimeError("The parkour curriculum requires TerrainImporterCfg with terrain_type='generator'.")

    terrain_layout.validate_grid(
        curriculum_difficulties=curriculum_cfg.num_difficulties,
        curriculum_families=len(curriculum_cfg.families),
        terrain_columns=int(terrain.terrain_origins.shape[1]),
        terrain_rows=int(terrain.terrain_origins.shape[0]),
    )
