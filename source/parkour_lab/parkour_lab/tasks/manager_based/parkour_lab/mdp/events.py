import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .curriculums import curriculums, curriculums_config


def initialize_parkour_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    terrain_layout: curriculums_config.ParkourTerrainLayout,
    curriculum_cfg: curriculums_config.ParkourCurriculumCfg = curriculums_config.DEFAULT_PARKOUR_CURRICULUM,
    initial_level_override: int | None = None,
) -> None:
    curriculums.initialize_parkour_terrain_levels(
        env=env,
        env_ids=env_ids,
        terrain_layout=terrain_layout,
        curriculum_cfg=curriculum_cfg,
        initial_level_override=initial_level_override,
    )


def reset_routes_and_commands(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    curriculum_cfg: curriculums_config.ParkourCurriculumCfg = curriculums_config.DEFAULT_PARKOUR_CURRICULUM,
    goal_cfg: SceneEntityCfg = SceneEntityCfg("goal"),
) -> None:
    curriculums.reset_routes_and_commands(
        env=env, env_ids=env_ids, curriculum_cfg=curriculum_cfg, goal_cfg=goal_cfg
    )
