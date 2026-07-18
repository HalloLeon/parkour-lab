import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .curriculums import curriculums, curriculums_config


def initialize_parkour_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: curriculums_config.ParkourCurriculumCfg = curriculums_config.DEFAULT_PARKOUR_CURRICULUM,
    fixed_level: int | None = None,
) -> None:
    curriculums.initialize_parkour_terrain_levels(
        env=env,
        env_ids=env_ids,
        curriculum_cfg=curriculum_cfg,
        fixed_level=fixed_level,
    )


def reset_waypoints_and_commands_from_terrain_level(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: curriculums_config.ParkourCurriculumCfg = curriculums_config.DEFAULT_PARKOUR_CURRICULUM,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> None:
    curriculums.reset_waypoints_and_commands_from_terrain_level(
        env=env, env_ids=env_ids, curriculum_cfg=curriculum_cfg, goal_cfg=goal_cfg
    )
