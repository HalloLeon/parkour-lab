from typing import Sequence

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
import torch

from . import config
from . import episode_outcomes
from .._shared.runtime import _all_env_ids
from ..commands import set_commands


def parkour_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    curriculum_cfg: config.ParkourCurriculumCfg = config.DEFAULT_PARKOUR_CURRICULUM
) -> torch.Tensor:
    """
    Terrain curriculum term.

    This is the Isaac-Lab-style replacement for manually writing
    terrain_importer.terrain_levels and terrain_importer.env_origins.

    It:
      - reads episode outcome buffers,
      - computes promote/demote events,
      - calls terrain.update_env_origins(...),
      - clears consumed episode outcome flags,
      - returns the mean terrain level for logging.
    """

    env_ids = torch.as_tensor(
        env_ids,
        device=env.device,
        dtype=torch.long
    )

    if env_ids.numel() == 0:
        terrain: TerrainImporter = env.scene.terrain
        return torch.mean(terrain.terrain_levels.float())

    terrain: TerrainImporter = env.scene.terrain

    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError(
            "parkour_terrain_levels requires TerrainImporterCfg with "
            "terrain_type='generator'."
        )

    _ensure_curriculum_stat_buffers(env)
    episode_outcomes.ensure_episode_outcome_buffers(env)

    if hasattr(env, "episode_length_buf"):
        has_real_episode = env.episode_length_buf[env_ids] > 1
    else:
        has_real_episode = torch.ones(
            env_ids.numel(),
            device=env.device,
            dtype=torch.bool
        )

    success_event = episode_outcomes.get_success(env)[env_ids] & has_real_episode
    base_contact_event = episode_outcomes.get_base_contact(env)[env_ids] & has_real_episode

    promote_event = success_event
    demote_event = base_contact_event & (~success_event)

    success_streak = env._parkour_success_streak[env_ids]
    failure_streak = env._parkour_failure_streak[env_ids]

    success_streak = torch.where(
        promote_event,
        success_streak + 1,
        torch.zeros_like(success_streak)
    )

    failure_streak = torch.where(
        demote_event,
        failure_streak + 1,
        torch.zeros_like(failure_streak)
    )

    move_up = (
        curriculum_cfg.promote_on_success
        and success_streak >= curriculum_cfg.successes_to_promote
    )

    move_down = (
        curriculum_cfg.demote_on_base_contact
        and failure_streak >= curriculum_cfg.failures_to_demote
    )

    # Important:
    # Let TerrainImporter own terrain_levels and env_origins.
    terrain.update_env_origins(
        env_ids=env_ids,
        move_up=move_up,
        move_down=move_down
    )

    changed = move_up | move_down

    env._parkour_success_streak[env_ids] = torch.where(
        changed,
        torch.zeros_like(success_streak),
        success_streak
    )

    env._parkour_failure_streak[env_ids] = torch.where(
        changed,
        torch.zeros_like(failure_streak),
        failure_streak
    )

    env._parkour_last_level_change[env_ids] = (
        move_up.to(torch.long) - move_down.to(torch.long)
    )

    episode_outcomes.clear_episode_outcomes(env, env_ids)

    return torch.mean(terrain.terrain_levels.float())


def reset_goal_and_commands_from_terrain_level(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: config.ParkourCurriculumCfg = config.DEFAULT_PARKOUR_CURRICULUM,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal")
) -> None:
    """
    Reset goal pose and per-env command buffers based on the current terrain level.

    The terrain level has already been updated by the CurriculumManager before
    reset events are applied.
    """

    def level_field(level, name: str):
        return level[name] if isinstance(level, dict) else getattr(level, name)

    def level_tensor(name: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.tensor(
            [level_field(level, name) for level in curriculum_cfg.levels],
            device=env.device,
            dtype=dtype
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

    goal_pos_by_level = level_tensor("goal_pos", dtype=dtype)
    obstacle_pos_by_level = level_tensor("obstacle_pos", dtype=dtype)
    obstacle_size_by_level = level_tensor("obstacle_size", dtype=dtype)
    target_speed_by_level = level_tensor("target_speed")
    min_clearance_by_level = level_tensor("min_clearance")

    level_goal_pos = goal_pos_by_level[levels]
    level_obstacle_pos = obstacle_pos_by_level[levels]
    level_obstacle_size = obstacle_size_by_level[levels]

    goal_pos_env = level_goal_pos.clone()
    goal_pos_env[:, 2] = 0.01

    goal_pos_w = goal_pos_env + env.scene.env_origins[env_ids]

    zero_velocity = torch.zeros(
        (num_reset_envs, 6),
        device=env.device,
        dtype=dtype
    )

    goal_pose = goal.data.default_root_state[env_ids, :7].clone()
    goal_pose[:, :3] = goal_pos_w

    goal.write_root_pose_to_sim(goal_pose, env_ids=env_ids)
    goal.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    set_commands(
        env=env,
        env_ids=env_ids,
        target_speed=target_speed_by_level[levels],
        min_clearance=min_clearance_by_level[levels],
        obstacle_pos=level_obstacle_pos,
        obstacle_size=level_obstacle_size
    )


def _ensure_curriculum_stat_buffers(env: ManagerBasedRLEnv) -> None:
    """
    Ensure curriculum statistic buffers exist.
    """

    if (
        not hasattr(env, "_parkour_success_streak")
        or env._parkour_success_streak.shape != (env.num_envs,)
        or env._parkour_success_streak.device != env.device
    ):
        env._parkour_success_streak = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.long
        )

    if (
        not hasattr(env, "_parkour_failure_streak")
        or env._parkour_failure_streak.shape != (env.num_envs,)
        or env._parkour_failure_streak.device != env.device
    ):
        env._parkour_failure_streak = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.long
        )

    if (
        not hasattr(env, "_parkour_last_level_change")
        or env._parkour_last_level_change.shape != (env.num_envs,)
        or env._parkour_last_level_change.device != env.device
    ):
        env._parkour_last_level_change = torch.zeros(
            env.num_envs,
            device=env.device,
            dtype=torch.long
        )


def _logical_level_from_terrain_level(
    env: ManagerBasedRLEnv,
    terrain_level: torch.Tensor,
    curriculum_cfg: config.ParkourCurriculumCfg
) -> torch.Tensor:
    """
    Map TerrainImporter terrain rows to parkour command/curriculum levels.

    This supports both cases:

    Case 1:
        num_terrain_rows == num_logical_levels
        terrain row 0 -> level 0
        terrain row 1 -> level 1
        ...

    Case 2:
        num_terrain_rows > num_logical_levels
        terrain rows are grouped proportionally into logical levels.
    """

    terrain = env.scene.terrain

    if terrain is None or terrain.terrain_origins is None:
        raise RuntimeError("Generated terrain is required.")

    num_terrain_rows = terrain.terrain_origins.shape[0]
    num_logical_levels = len(curriculum_cfg.levels)

    logical_level = torch.div(
        terrain_level.to(device=env.device, dtype=torch.long) * num_logical_levels,
        num_terrain_rows,
        rounding_mode="floor"
    )

    return torch.clamp(
        logical_level,
        min=0,
        max=curriculum_cfg.max_level
    )
