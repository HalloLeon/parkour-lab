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


def reset_routes(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    terrain_layout: curriculums_config.ParkourTerrainLayout,
    curriculum_cfg: curriculums_config.ParkourCurriculumCfg = curriculums_config.DEFAULT_PARKOUR_CURRICULUM,
    waypoint_marker_cfg: SceneEntityCfg = SceneEntityCfg("waypoint_marker"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_speed_range: tuple[float, float] = (0.45, 0.70),
) -> None:
    curriculums.reset_routes(
        env=env,
        env_ids=env_ids,
        terrain_layout=terrain_layout,
        curriculum_cfg=curriculum_cfg,
        waypoint_marker_cfg=waypoint_marker_cfg,
        asset_cfg=asset_cfg,
        target_speed_range=target_speed_range,
    )
