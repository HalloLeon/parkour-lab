from typing import Sequence

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

from .._shared.runtime import _all_env_ids, _env_torch_device
from ..commands import set_commands
from . import config, episode_outcomes


def initialize_parkour_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | None,
    curriculum_cfg: config.ParkourCurriculumCfg = config.DEFAULT_PARKOUR_CURRICULUM,
    fixed_level: int | None = None,
) -> None:
    """Place environments on exact terrain rows before the first reset.

    ``TerrainImporterCfg.max_init_terrain_level`` is an upper bound for random
    sampling, not an exact initial level. This startup event makes the training
    distribution explicit and is also what pins deterministic evaluation to a
    single difficulty.
    """

    terrain: TerrainImporter = env.scene.terrain
    _validate_terrain_layout(terrain, curriculum_cfg)

    if fixed_level is not None and not 0 <= fixed_level <= curriculum_cfg.max_level:
        raise ValueError(
            f"fixed_level must be in [0, {curriculum_cfg.max_level}], got {fixed_level}."
        )

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if fixed_level is not None:
        levels = torch.full_like(env_ids, fixed_level)
    elif curriculum_cfg.distribute_initial_levels:
        # Deterministically balanced, which is more reproducible than relying on
        # TerrainImporter's random 0..max initialization.
        levels = torch.remainder(env_ids, curriculum_cfg.initial_level + 1)
    else:
        levels = torch.full_like(env_ids, curriculum_cfg.initial_level)

    terrain.terrain_levels[env_ids] = levels
    # terrain_origins has shape [num_levels, num_terrain_types, 3] and is the
    # generated-tile lookup table (difficulty row, terrain column/type, XYZ).
    # env_origins has shape [num_envs, 3] and stores the selected tile origin
    # for each environment.
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        levels, terrain.terrain_types[env_ids]
    ]

    _ensure_curriculum_stat_buffers(env)
    env._parkour_success_streak[env_ids] = 0
    env._parkour_failure_streak[env_ids] = 0
    env._parkour_last_level_change[env_ids] = 0
    episode_outcomes.clear_episode_outcomes(env, env_ids)


def parkour_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    curriculum_cfg: config.ParkourCurriculumCfg = config.DEFAULT_PARKOUR_CURRICULUM,
) -> dict[str, torch.Tensor]:
    """Update per-environment difficulty from terminal episode outcomes."""

    env_ids_tensor = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    terrain: TerrainImporter = env.scene.terrain
    _validate_terrain_layout(terrain, curriculum_cfg)

    if env_ids_tensor.numel() == 0:
        return _curriculum_stats(
            terrain=terrain,
            curriculum_cfg=curriculum_cfg,
            success_event=torch.zeros(0, device=env.device, dtype=torch.bool),
            failure_event=torch.zeros(0, device=env.device, dtype=torch.bool),
            actual_change=torch.zeros(0, device=env.device, dtype=torch.long),
        )

    _ensure_curriculum_stat_buffers(env)
    episode_outcomes.ensure_episode_outcome_buffers(env)

    success_event = episode_outcomes.get_success(env)[env_ids_tensor]

    # CurriculumManager runs only for reset environments. reset_buf therefore
    # covers trunk contact, timeout, and future failure terminations. Outcome
    # buffers keep the initial/manual reset neutral. Success wins if multiple
    # termination terms fire on the same step.
    if hasattr(env, "reset_buf"):
        terminal_event = env.reset_buf[env_ids_tensor].to(
            device=env.device, dtype=torch.bool
        )
    else:
        terminal_event = (
            episode_outcomes.get_base_contact(env)[env_ids_tensor] | success_event
        )
    failure_event = terminal_event & (~success_event)

    success_streak = env._parkour_success_streak[env_ids_tensor]
    failure_streak = env._parkour_failure_streak[env_ids_tensor]

    success_streak = torch.where(
        success_event,
        success_streak + 1,
        torch.where(failure_event, torch.zeros_like(success_streak), success_streak),
    )

    failure_streak = torch.where(
        failure_event,
        failure_streak + 1,
        torch.where(success_event, torch.zeros_like(failure_streak), failure_streak),
    )

    promotion_ready = success_streak >= curriculum_cfg.successes_to_promote
    demotion_ready = failure_streak >= curriculum_cfg.failures_to_demote
    if not curriculum_cfg.promote_on_success:
        promotion_ready = torch.zeros_like(promotion_ready)
    if not curriculum_cfg.demote_on_failure:
        demotion_ready = torch.zeros_like(demotion_ready)

    old_levels = terrain.terrain_levels[env_ids_tensor].clone()
    move_up = promotion_ready & (old_levels < curriculum_cfg.max_level)
    move_down = demotion_ready & (old_levels > 0) & (~move_up)

    # Important:
    # Let TerrainImporter own terrain_levels and env_origins.
    terrain.update_env_origins(
        env_ids=env_ids_tensor, move_up=move_up, move_down=move_down
    )

    new_levels = terrain.terrain_levels[env_ids_tensor]
    actual_change = new_levels - old_levels
    consumed_streak = promotion_ready | demotion_ready

    env._parkour_success_streak[env_ids_tensor] = torch.where(
        consumed_streak, torch.zeros_like(success_streak), success_streak
    )

    env._parkour_failure_streak[env_ids_tensor] = torch.where(
        consumed_streak, torch.zeros_like(failure_streak), failure_streak
    )

    env._parkour_last_level_change[env_ids_tensor] = actual_change

    episode_outcomes.clear_episode_outcomes(env, env_ids_tensor)

    return _curriculum_stats(
        terrain=terrain,
        curriculum_cfg=curriculum_cfg,
        success_event=success_event,
        failure_event=failure_event,
        actual_change=actual_change,
    )


def reset_goal_and_commands_from_terrain_level(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: config.ParkourCurriculumCfg = config.DEFAULT_PARKOUR_CURRICULUM,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> None:
    """
    Reset goal pose and per-env command buffers based on the current terrain level.

    The terrain level has already been updated by the CurriculumManager before
    reset events are applied.
    """

    def _level_tensor(name: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor(
            [
                getattr(config.coerce_level_cfg(level), name)
                for level in curriculum_cfg.levels
            ],
            device=env.device,
            dtype=dtype,
        )

    env_ids = _all_env_ids(env, env_ids)

    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError(
            "reset_goal_and_commands_from_terrain_level requires generated terrain."
        )

    goal: RigidObject = env.scene[goal_cfg.name]

    dtype = goal.data.default_root_state.dtype
    num_reset_envs = env_ids.numel()

    terrain_levels = terrain.terrain_levels[env_ids]

    levels = _logical_level_from_terrain_level(
        env=env,
        terrain_level=terrain_levels,
        curriculum_cfg=curriculum_cfg,
    )

    goal_pos_by_level = _level_tensor("goal_pos", dtype=dtype)
    target_speed_by_level = _level_tensor("target_speed")
    min_clearance_by_level = _level_tensor("min_clearance")

    level_goal_pos = goal_pos_by_level[levels]

    goal_pos_w = level_goal_pos + env.scene.env_origins[env_ids]

    # Root velocity has six values per environment:
    #   [linear_x, linear_y, linear_z, angular_x, angular_y, angular_z].
    # Therefore the batched tensor has shape [num_reset_envs, 6]. The goal is
    # kinematic and stationary, so all linear and angular velocities are zero.
    zero_velocity = torch.zeros((num_reset_envs, 6), device=env.device, dtype=dtype)

    # Root pose has seven values per environment:
    #   [position_x, position_y, position_z, quaternion_w, quaternion_x,
    #    quaternion_y, quaternion_z].
    # Select only the reset environments and clone to obtain a writable tensor
    # with shape [num_reset_envs, 7]. Keep the configured orientation in columns
    # 3:7 and replace only the XYZ position in columns 0:3.
    goal_pose = goal.data.default_root_state[env_ids, :7].clone()
    goal_pose[:, :3] = goal_pos_w

    goal.write_root_pose_to_sim(goal_pose, env_ids=env_ids)
    goal.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    set_commands(
        env=env,
        env_ids=env_ids,
        target_speed=target_speed_by_level[levels],
        min_clearance=min_clearance_by_level[levels],
    )


def _curriculum_stats(
    terrain: TerrainImporter,
    curriculum_cfg: config.ParkourCurriculumCfg,
    success_event: torch.Tensor,
    failure_event: torch.Tensor,
    actual_change: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build scalar curriculum metrics for TensorBoard/W&B logging."""

    levels = terrain.terrain_levels
    zero = torch.zeros((), device=levels.device, dtype=torch.float32)

    def _event_rate(event: torch.Tensor) -> torch.Tensor:
        return event.float().mean() if event.numel() > 0 else zero

    return {
        "mean_level": levels.float().mean(),
        "min_level": levels.min().float(),
        "max_level": levels.max().float(),
        "top_level_fraction": (levels == curriculum_cfg.max_level).float().mean(),
        "success_rate": _event_rate(success_event),
        "failure_rate": _event_rate(failure_event),
        "promotion_rate": _event_rate(actual_change > 0),
        "demotion_rate": _event_rate(actual_change < 0),
    }


def _ensure_curriculum_stat_buffers(env: ManagerBasedRLEnv) -> None:
    """
    Ensure curriculum statistic buffers exist.
    """

    device = _env_torch_device(env)

    if (
        not hasattr(env, "_parkour_success_streak")
        or env._parkour_success_streak.shape != (env.num_envs,)
        or env._parkour_success_streak.device != device
        or env._parkour_success_streak.dtype != torch.long
    ):
        env._parkour_success_streak = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    if (
        not hasattr(env, "_parkour_failure_streak")
        or env._parkour_failure_streak.shape != (env.num_envs,)
        or env._parkour_failure_streak.device != device
        or env._parkour_failure_streak.dtype != torch.long
    ):
        env._parkour_failure_streak = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )

    if (
        not hasattr(env, "_parkour_last_level_change")
        or env._parkour_last_level_change.shape != (env.num_envs,)
        or env._parkour_last_level_change.device != device
        or env._parkour_last_level_change.dtype != torch.long
    ):
        env._parkour_last_level_change = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.long
        )


def _logical_level_from_terrain_level(
    env: ManagerBasedRLEnv,
    terrain_level: torch.Tensor,
    curriculum_cfg: config.ParkourCurriculumCfg,
) -> torch.Tensor:
    """Map terrain rows one-to-one to the authoritative logical levels.

    ``terrain_level`` contains integer row indices maintained by
    ``TerrainImporter``; it is not the normalized floating-point difficulty
    passed to the terrain-generation function. The parkour configuration
    enforces exactly one terrain row per logical level, so the mapping is:

        terrain row 0 -> logical level 0
        terrain row 1 -> logical level 1
        ...

    Consequently, no division or proportional scaling is required. If the
    number of terrain rows and logical levels ever differs, the layout
    validation below raises instead of silently applying an ambiguous mapping.
    """

    terrain = env.scene.terrain

    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError("Generated terrain is required.")

    _validate_terrain_layout(terrain, curriculum_cfg)

    # This is already the logical level because physical row N is generated
    # directly from curriculum level N. Only normalize its device and dtype.
    logical_level = terrain_level.to(device=env.device, dtype=torch.long)

    # max_level may intentionally expose only a prefix of the defined levels.
    return torch.clamp(logical_level, min=0, max=curriculum_cfg.max_level)


def _validate_terrain_layout(
    terrain: TerrainImporter | None,
    curriculum_cfg: config.ParkourCurriculumCfg,
) -> None:
    """Require an exact physical-row to logical-level mapping."""

    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError(
            "The parkour curriculum requires TerrainImporterCfg with terrain_type='generator'."
        )

    num_rows = terrain.terrain_origins.shape[0]
    num_levels = len(curriculum_cfg.levels)
    if num_rows != num_levels:
        raise RuntimeError(
            "Parkour terrain rows and curriculum levels must match one-to-one: "
            f"got {num_rows} rows and {num_levels} levels."
        )
