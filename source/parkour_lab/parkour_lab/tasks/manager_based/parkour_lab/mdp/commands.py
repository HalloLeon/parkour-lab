import torch
from isaaclab.envs import ManagerBasedRLEnv

from .navigation import route


def get_min_clearance(
    env: ManagerBasedRLEnv,
    default: float = 0.27,
) -> torch.Tensor:
    """Return the active course's minimum clearance for every environment."""

    return route.current_min_clearances(env, default=default)


def get_target_speed(
    env: ManagerBasedRLEnv,
    default: float = 0.70,
) -> torch.Tensor:
    """Return the episode's desired speed for every environment."""

    return route.current_target_speeds(env, default=default)
