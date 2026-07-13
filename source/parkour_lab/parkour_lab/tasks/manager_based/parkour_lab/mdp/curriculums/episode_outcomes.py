import torch
from isaaclab.envs import ManagerBasedRLEnv

from .._shared.runtime import _env_torch_device

_SUCCESS_BUFFER = "_parkour_episode_success"
_BASE_CONTACT_BUFFER = "_parkour_episode_base_contact"


def clear_episode_outcomes(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Clear outcome flags after curriculum has consumed them."""

    ensure_episode_outcome_buffers(env)

    env._parkour_episode_success[env_ids] = False
    env._parkour_episode_base_contact[env_ids] = False


def ensure_episode_outcome_buffers(env: ManagerBasedRLEnv) -> None:
    """Ensure per-env episode outcome buffers exist."""

    device = _env_torch_device(env)

    if (
        not hasattr(env, _SUCCESS_BUFFER)
        or getattr(env, _SUCCESS_BUFFER).shape != (env.num_envs,)
        or getattr(env, _SUCCESS_BUFFER).device != device
        or getattr(env, _SUCCESS_BUFFER).dtype != torch.bool
    ):
        env._parkour_episode_success = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )

    if (
        not hasattr(env, _BASE_CONTACT_BUFFER)
        or getattr(env, _BASE_CONTACT_BUFFER).shape != (env.num_envs,)
        or getattr(env, _BASE_CONTACT_BUFFER).device != device
        or getattr(env, _BASE_CONTACT_BUFFER).dtype != torch.bool
    ):
        env._parkour_episode_base_contact = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )


def get_base_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return current per-env base-contact flags."""

    ensure_episode_outcome_buffers(env)
    return env._parkour_episode_base_contact


def get_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return current per-env success flags."""

    ensure_episode_outcome_buffers(env)
    return env._parkour_episode_success


def mark_base_contact(env: ManagerBasedRLEnv, base_contact: torch.Tensor) -> None:
    """Accumulate base-contact failure flags for the current episode."""

    ensure_episode_outcome_buffers(env)
    env._parkour_episode_base_contact |= base_contact.to(
        device=env.device, dtype=torch.bool
    )


def mark_success(env: ManagerBasedRLEnv, success: torch.Tensor) -> None:
    """Accumulate success flags for the current episode."""

    ensure_episode_outcome_buffers(env)
    env._parkour_episode_success |= success.to(device=env.device, dtype=torch.bool)
