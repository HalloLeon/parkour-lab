from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import torch

from . import config
from .._shared.runtime import _all_env_ids
from ..commands import set_commands


def reset_goal_and_obstacle_by_level(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: config.ParkourCurriculumCfg = config.DEFAULT_PARKOUR_CURRICULUM,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal")
) -> None:
    """
    Reset goal pose, obstacle pose, and per-environment parkour commands
    according to each environment's current curriculum level.

    This function assigns level-specific task geometry and command values.
    It does not promote or demote levels.
    """

    env_ids = utils._all_env_ids(env, env_ids)
    levels = utils._parkour_level_index(env, curriculum_cfg)[env_ids]

    obstacle: RigidObject = env.scene[obstacle_cfg.name]
    goal: RigidObject = env.scene[goal_cfg.name]

    device = env.device
    dtype = obstacle.data.default_root_state.dtype
    num_reset_envs = env_ids.numel()

    obstacle_pos_by_level = torch.tensor(
        [level.obstacle_pos for level in curriculum_cfg.levels],
        device=device,
        dtype=dtype
    )

    obstacle_size_by_level = torch.tensor(
        [level.obstacle_size for level in curriculum_cfg.levels],
        device=device,
        dtype=dtype
    )

    goal_pos_by_level = torch.tensor(
        [level.goal_pos for level in curriculum_cfg.levels],
        device=device,
        dtype=dtype
    )

    target_speed_by_level = torch.tensor(
        [level.target_speed for level in curriculum_cfg.levels],
        device=device,
        dtype=torch.float32
    )

    min_clearance_by_level = torch.tensor(
        [level.min_clearance for level in curriculum_cfg.levels],
        device=device,
        dtype=torch.float32
    )

    level_obstacle_pos = obstacle_pos_by_level[levels]
    level_obstacle_size = obstacle_size_by_level[levels]
    level_goal_pos = goal_pos_by_level[levels]

    # With a single spawned cuboid, the collider size is fixed.
    # We implement height curriculum by spawning the maximum-height cuboid
    # and moving it vertically so that the visible top surface matches the
    # active level's desired obstacle height.
    scene_obstacle_height = obstacle_size_by_level[:, 2].max()

    obstacle_pos_env = level_obstacle_pos.clone()
    obstacle_pos_env[:, 2] = level_obstacle_size[:, 2] - 0.5 * scene_obstacle_height

    # The goal is an XY target marker. Keep it visually on the ground.
    goal_pos_env = level_goal_pos.clone()
    goal_pos_env[:, 2] = 0.01

    obstacle_pos_w = obstacle_pos_env + env.scene.env_origins[env_ids]
    goal_pos_w = goal_pos_env + env.scene.env_origins[env_ids]

    zero_velocity = torch.zeros(
        (num_reset_envs, 6),
        device=device,
        dtype=dtype
    )

    obstacle_pose = obstacle.data.default_root_state[env_ids, :7].clone()
    obstacle_pose[:, :3] = obstacle_pos_w
    obstacle.write_root_pose_to_sim(obstacle_pose, env_ids=env_ids)
    obstacle.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    goal_pose = goal.data.default_root_state[env_ids, :7].clone()
    goal_pose[:, :3] = goal_pos_w
    goal.write_root_pose_to_sim(goal_pose, env_ids=env_ids)
    goal.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

    set_commands(
        env=env,
        env_ids=env_ids,
        target_speed=target_speed_by_level[levels],
        min_clearance=min_clearance_by_level[levels]
    )


def _ensure_parkour_level_buffer(
    env: ManagerBasedRLEnv,
    curriculum_cfg: config.ParkourCurriculumCfg
) -> torch.Tensor:
    """
    Ensure env._parkour_levels exists.

    Returns:
        [num_envs] long tensor of curriculum levels.
    """

    needs_init = (
        not hasattr(env, "_parkour_levels")
        or env._parkour_levels.shape != (env.num_envs,)
        or env._parkour_levels.device != env.device
    )

    if needs_init:
        if getattr(curriculum_cfg, "distribute_initial_levels", False):
            levels = torch.arange(
                env.num_envs,
                device=env.device,
                dtype=torch.long,
            ) % (curriculum_cfg.max_level + 1)
        else:
            levels = torch.full(
                (env.num_envs,),
                fill_value=curriculum_cfg.initial_level,
                device=env.device,
                dtype=torch.long
            )

        env._parkour_levels = levels

    return env._parkour_levels


def _parkour_level_index(
    env: ManagerBasedRLEnv,
    curriculum_cfg: config.ParkourCurriculumCfg
) -> torch.Tensor:
    """
    Current curriculum level per environment.

    Returns:
        [num_envs]
    """

    levels = _ensure_parkour_level_buffer(env, curriculum_cfg)

    return torch.clamp(
        levels,
        min=0,
        max=curriculum_cfg.max_level
    )
