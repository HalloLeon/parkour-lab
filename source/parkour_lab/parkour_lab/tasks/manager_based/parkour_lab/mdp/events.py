from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
import torch

from .commands import set_commands
from .commands import _all_env_ids
from .curriculums import curriculums
from .curriculums import curriculums_config


def reset_goal_and_obstacle_by_level(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: curriculums_config.ParkourCurriculumCfg = curriculums_config.DEFAULT_PARKOUR_CURRICULUM,
    obstacle_cfg: SceneEntityCfg = SceneEntityCfg("obstacle"),
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal")
) -> None:
    curriculums.reset_goal_and_obstacle_by_level(
        env=env,
        env_ids=env_ids,
        curriculum_cfg=curriculum_cfg,
        obstacle_cfg=obstacle_cfg,
        goal_cfg=goal_cfg
    )


def reset_constant_parkour_commands(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    target_speed: float = 0.75,
    min_clearance: float = 0.25
) -> None:
    env_ids = _all_env_ids(env, env_ids)

    target_speed_tensor = torch.full(
        (len(env_ids),),
        target_speed,
        device=env.device,
        dtype=torch.float32
    )

    min_clearance_tensor = torch.full(
        (len(env_ids),),
        min_clearance,
        device=env.device,
        dtype=torch.float32
    )

    set_commands(
        env,
        env_ids=env_ids,
        target_speed=target_speed_tensor,
        min_clearance=min_clearance_tensor
    )
