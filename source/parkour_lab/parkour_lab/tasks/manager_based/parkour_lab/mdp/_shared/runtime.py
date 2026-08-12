from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv


def _all_env_ids(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
) -> torch.Tensor:
    """Return selected environment indices as a device-local integer tensor."""

    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]

    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _validate_matching_shape(
    lhs: torch.Tensor, rhs: torch.Tensor, *, lhs_name: str, rhs_name: str
) -> None:
    """
    Validate that two tensors have identical shape.

    Raises:
        RuntimeError: If shapes differ.
    """

    if lhs.shape != rhs.shape:
        raise RuntimeError(
            f"{lhs_name} shape does not match {rhs_name} shape. "
            f"Got {lhs_name} shape {tuple(lhs.shape)} and "
            f"{rhs_name} shape {tuple(rhs.shape)}."
        )
