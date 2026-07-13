import torch
from isaaclab.envs import ManagerBasedRLEnv

from ._shared.runtime import _all_env_ids, _env_torch_device

_TARGET_SPEED_BUFFER = "_cmd_target_speed"
_MIN_CLEARANCE_BUFFER = "_cmd_min_clearance"


def ensure_parkour_commands(
    env: ManagerBasedRLEnv,
    default_target_speed: float = 0.70,
    default_min_clearance: float = 0.25,
) -> None:
    """
    Ensure parkour command buffers exist.

    This may be called during ObservationManager construction, before reset
    events/curriculum have initialized per-env command values.
    """

    device = _env_torch_device(env)

    needs_target_speed = (
        not hasattr(env, _TARGET_SPEED_BUFFER)
        or getattr(env, _TARGET_SPEED_BUFFER).shape != (env.num_envs,)
        or getattr(env, _TARGET_SPEED_BUFFER).device != device
        or getattr(env, _TARGET_SPEED_BUFFER).dtype != torch.float32
    )

    if needs_target_speed:
        env._cmd_target_speed = torch.full(
            (env.num_envs,),
            default_target_speed,
            device=env.device,
            dtype=torch.float32,
        )

    needs_min_clearance = (
        not hasattr(env, _MIN_CLEARANCE_BUFFER)
        or getattr(env, _MIN_CLEARANCE_BUFFER).shape != (env.num_envs,)
        or getattr(env, _MIN_CLEARANCE_BUFFER).device != device
        or getattr(env, _MIN_CLEARANCE_BUFFER).dtype != torch.float32
    )

    if needs_min_clearance:
        env._cmd_min_clearance = torch.full(
            (env.num_envs,),
            default_min_clearance,
            device=env.device,
            dtype=torch.float32,
        )


def get_min_clearance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current per-environment minimum clearance with shape [num_envs]."""

    ensure_parkour_commands(env)
    return env._cmd_min_clearance


def get_target_speed(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current per-environment target speed with shape [num_envs]."""

    ensure_parkour_commands(env)
    return env._cmd_target_speed


def set_commands(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    target_speed: torch.Tensor,
    min_clearance: torch.Tensor,
) -> None:
    """Set per-environment parkour commands."""

    ensure_parkour_commands(env)

    env_ids = _all_env_ids(env, env_ids)

    env._cmd_target_speed[env_ids] = target_speed.to(
        device=env.device, dtype=torch.float32
    )
    env._cmd_min_clearance[env_ids] = min_clearance.to(
        device=env.device, dtype=torch.float32
    )
