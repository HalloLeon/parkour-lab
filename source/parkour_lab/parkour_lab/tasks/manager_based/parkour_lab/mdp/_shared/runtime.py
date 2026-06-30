from isaaclab.assets import AssetBase
from isaaclab.envs import ManagerBasedRLEnv
import torch


def _all_env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    return env_ids.to(device=env.device, dtype=torch.long)


def _difference_from_previous_env_buffer(
    env: ManagerBasedRLEnv,
    *,
    buffer_name: str,
    current_value: torch.Tensor,
    reset_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Compute previous_value - current_value using an environment-level buffer.

    The buffer is always updated, even when reset_mask is true.

    Returns:
        [num_envs]
    """

    previous_value = _get_or_init_env_buffer(
        env=env,
        name=buffer_name,
        value=current_value
    )

    difference = previous_value - current_value

    if reset_mask is not None:
        difference = torch.where(
            reset_mask,
            torch.zeros_like(difference),
            difference
        )

    _set_env_buffer(
        env=env,
        name=buffer_name,
        value=current_value
    )

    return difference


def _episode_start_mask(
    env: ManagerBasedRLEnv,
    reference: torch.Tensor,
    grace_steps: int
) -> torch.Tensor:
    """
    Boolean mask for environments that have just reset.

    Returns:
        [num_envs]
    """

    if grace_steps <= 0 or not hasattr(env, "episode_length_buf"):
        return torch.zeros_like(reference, dtype=torch.bool)

    episode_length = env.episode_length_buf.to(device=reference.device)

    return episode_length <= grace_steps


def _gate_positive_values(
    values: torch.Tensor,
    gate: torch.Tensor
) -> torch.Tensor:
    """
    Keep negative values always, but allow positive values only when gate is true.

    This is useful for rewards where:
      - positive progress should require valid behavior,
      - negative progress should still be penalized.

    Returns:
        Tensor with the same shape as values.
    """

    keep_value = torch.logical_or(values <= 0.0, gate)

    return torch.where(
        keep_value,
        values,
        torch.zeros_like(values)
    )


def _get_or_init_env_buffer(
    env: ManagerBasedRLEnv,
    name: str,
    value: torch.Tensor
) -> torch.Tensor:
    """
    Get an environment-level tensor buffer, creating or resizing it if needed.

    Returns:
        Tensor with the same shape, device, and dtype as value.
    """

    needs_init = (
        not hasattr(env, name)
        or getattr(env, name).shape != value.shape
        or getattr(env, name).device != value.device
        or getattr(env, name).dtype != value.dtype
    )

    if needs_init:
        setattr(env, name, value.detach().clone())

    return getattr(env, name)


def _get_scene_entity_or_none(
    env: ManagerBasedRLEnv,
    name: str
) -> AssetBase | None:
    """
    Return a scene entity if it exists, otherwise None.

    This keeps optional scene objects, such as an obstacle, from making
    reward functions crash in simpler environments.
    """

    try:
        return env.scene[name]
    except KeyError:
        return None


def _linear_ramp(
    value: torch.Tensor,
    lower: float,
    upper: float
) -> torch.Tensor:
    """
    Smoothly map value from [lower, upper] to [0, 1].

    Values below lower become 0.
    Values above upper become 1.

    Returns:
        Tensor with same shape as value.
    """

    return torch.clamp(
        (value - lower) / (upper - lower),
        min=0.0,
        max=1.0
    )


def _private_buffer_name(
    prefix: str,
    *parts: str
) -> str:
    """
    Create a private environment-buffer name.

    Returns:
        str
    """

    safe_parts = [
        str(part).replace("/", "_").replace(" ", "_")
        for part in parts
    ]

    return "_" + "_".join((prefix, *safe_parts))


def _set_env_buffer(
    env: ManagerBasedRLEnv,
    name: str,
    value: torch.Tensor
) -> None:
    """
    Store a detached clone as an environment-level tensor buffer.
    """

    setattr(env, name, value.detach().clone())


def _validate_matching_shape(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    *,
    lhs_name: str,
    rhs_name: str
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
