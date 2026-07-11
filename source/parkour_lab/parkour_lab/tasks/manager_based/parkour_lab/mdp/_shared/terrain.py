import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from .state import _root_height_env


def _base_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_height_scanner"),
) -> torch.Tensor:
    """
    Vertical clearance between robot base/root and the surface underneath it.

    Returns:
        [num_envs]
    """

    sensor = env.scene[sensor_cfg.name]
    if not isinstance(sensor, RayCaster):
        raise TypeError(f"Expected '{sensor_cfg.name}' to be a RayCaster, got {type(sensor).__name__}.")

    ray_hits_w = sensor.data.ray_hits_w
    if ray_hits_w.shape[1] != 1:
        raise RuntimeError(f"'{sensor_cfg.name}' must contain exactly one downward ray.")

    base_height = _root_height_env(env, asset_cfg)
    surface_height = ray_hits_w[:, 0, 2] - env.scene.env_origins[:, 2]
    clearance = base_height - surface_height

    # A missed ray is unsafe, but it should not inject infinities into rewards
    # or observations if the simulator briefly places a robot out of range.
    return torch.where(torch.isfinite(clearance), clearance, torch.zeros_like(clearance))
