from isaaclab.envs import ManagerBasedRLEnv

import torch

from . import config


def _all_env_ids(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None
) -> torch.Tensor:
    """
    Return all environment ids if env_ids is None.
    """

    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    return env_ids.to(device=env.device, dtype=torch.long)


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


def _parkour_level_scalar_tensor(
    env: ManagerBasedRLEnv,
    curriculum_cfg: config.ParkourCurriculumCfg,
    attr_name: str
) -> torch.Tensor:
    """
    Get a scalar per environment from the active curriculum level.

    Example:
        attr_name = "target_speed"

    Returns:
        [num_envs]
    """

    values = torch.tensor(
        [float(getattr(level, attr_name)) for level in curriculum_cfg.levels],
        device=env.device,
        dtype=torch.float32
    )

    levels = _parkour_level_index(env, curriculum_cfg)

    return values[levels]


def _parkour_level_vec3_tensor(
    env: ManagerBasedRLEnv,
    curriculum_cfg: config.ParkourCurriculumCfg,
    attr_name: str
) -> torch.Tensor:
    """
    Get a 3D vector per environment from the active curriculum level.

    Example:
        attr_name = "goal_pos"

    Returns:
        [num_envs, 3]
    """

    values = torch.tensor(
        [getattr(level, attr_name) for level in curriculum_cfg.levels],
        device=env.device,
        dtype=torch.float32
    )

    levels = _parkour_level_index(env, curriculum_cfg)

    return values[levels]
