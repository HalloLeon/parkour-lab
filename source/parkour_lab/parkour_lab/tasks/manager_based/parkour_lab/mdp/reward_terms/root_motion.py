# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stateful root-motion regularization."""

from dataclasses import dataclass

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from .. import config
from .._shared import runtime, state


def root_chatter_l2(
    env: ManagerBasedRLEnv,
    chatter_cfg: config.RootMotionChatterCfg = config.DEFAULT_ROOT_MOTION_CHATTER,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize small, rapid root/core oscillations.

    This targets high-frequency chatter:
      - small vertical bounces that quickly reverse,
      - small roll/pitch wiggles that quickly reverse.

    It does not penalize large vertical motion directly. Larger step-up,
    jump, or obstacle traversal motions are allowed as long as they are not
    small rapid reversals.

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    current = _RootChatterState.from_env(env, asset_cfg=asset_cfg)

    buffer_prefix = runtime._private_buffer_name("parkour_root_chatter", asset_cfg.name)

    previous = _RootChatterState.previous_from_env(env, buffer_prefix=buffer_prefix, current=current)

    vertical_penalty = _vertical_root_chatter_l2(current=current, previous=previous, chatter_cfg=chatter_cfg)

    angular_penalty = _angular_root_chatter_l2(current=current, previous=previous, chatter_cfg=chatter_cfg)

    penalty = vertical_penalty + chatter_cfg.angular_weight * angular_penalty

    just_reset = runtime._episode_start_mask(env, reference=penalty, grace_steps=chatter_cfg.reset_grace_steps)

    penalty = torch.where(just_reset, torch.zeros_like(penalty), penalty)

    current.write_to_env(env, buffer_prefix=buffer_prefix)

    return penalty


@dataclass(frozen=True)
class _RootChatterState:
    """
    Root/core signals used by root_chatter_l2.

    This groups tensors that are always used together, reducing argument-heavy
    helper functions without hiding the reward logic.
    """

    root_z: torch.Tensor
    root_z_vel: torch.Tensor
    projected_gravity_xy: torch.Tensor
    roll_pitch_rate: torch.Tensor

    @classmethod
    def from_env(cls, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> "_RootChatterState":
        """Read current root/core signals from the environment."""

        return cls(
            root_z=state._root_height_env(env, asset_cfg),
            root_z_vel=state._root_lin_vel_z(env, asset_cfg),
            projected_gravity_xy=state._root_projected_gravity_xy(env, asset_cfg),
            roll_pitch_rate=state._root_roll_pitch_rate(env, asset_cfg),
        )

    @classmethod
    def previous_from_env(
        cls, env: ManagerBasedRLEnv, *, buffer_prefix: str, current: "_RootChatterState"
    ) -> "_RootChatterState":
        """
        Read previous root/core signals from environment buffers.

        Missing or stale buffers are initialized from the current state.
        """

        return cls(
            root_z=runtime._get_or_init_env_buffer(env, f"{buffer_prefix}_root_z", current.root_z),
            root_z_vel=runtime._get_or_init_env_buffer(env, f"{buffer_prefix}_root_z_vel", current.root_z_vel),
            projected_gravity_xy=runtime._get_or_init_env_buffer(
                env, f"{buffer_prefix}_projected_gravity_xy", current.projected_gravity_xy
            ),
            roll_pitch_rate=runtime._get_or_init_env_buffer(
                env, f"{buffer_prefix}_roll_pitch_rate", current.roll_pitch_rate
            ),
        )

    def write_to_env(self, env: ManagerBasedRLEnv, *, buffer_prefix: str) -> None:
        """Store this state as the previous-step root/core state."""

        runtime._set_env_buffer(env, f"{buffer_prefix}_root_z", self.root_z)
        runtime._set_env_buffer(env, f"{buffer_prefix}_root_z_vel", self.root_z_vel)
        runtime._set_env_buffer(env, f"{buffer_prefix}_projected_gravity_xy", self.projected_gravity_xy)
        runtime._set_env_buffer(env, f"{buffer_prefix}_roll_pitch_rate", self.roll_pitch_rate)


def _reversal_excess(
    current: torch.Tensor, previous: torch.Tensor, min_magnitude: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Detect sign reversal and compute excess reversal magnitude.

    Returns:
        reversed_direction:
            Boolean tensor.

        excess:
            max(min(abs(current), abs(previous)) - min_magnitude, 0)
    """

    reversed_direction = current * previous < 0.0

    reversal_magnitude = torch.minimum(torch.abs(current), torch.abs(previous))

    excess = torch.clamp(reversal_magnitude - min_magnitude, min=0.0)

    return reversed_direction, excess


def _vertical_root_chatter_l2(
    current: _RootChatterState, previous: _RootChatterState, chatter_cfg: config.RootMotionChatterCfg
) -> torch.Tensor:
    """
    Penalize small vertical bounces that rapidly reverse direction.

    Returns:
        [num_envs]
    """

    z_displacement = torch.abs(current.root_z - previous.root_z)

    velocity_reversed, reversal_excess = _reversal_excess(
        current=current.root_z_vel, previous=previous.root_z_vel, min_magnitude=chatter_cfg.min_z_reversal_speed
    )

    small_displacement = z_displacement < chatter_cfg.small_z_displacement

    chatter_active = velocity_reversed & small_displacement

    return reversal_excess.square() * chatter_active.to(dtype=current.root_z.dtype)


def _angular_root_chatter_l2(
    current: _RootChatterState, previous: _RootChatterState, chatter_cfg: config.RootMotionChatterCfg
) -> torch.Tensor:
    """
    Penalize small roll/pitch wiggles that rapidly reverse direction.

    Returns:
        [num_envs]
    """

    tilt_change = torch.linalg.norm(current.projected_gravity_xy - previous.projected_gravity_xy, dim=-1)

    small_tilt_change = tilt_change < chatter_cfg.small_tilt_change

    angular_reversed, angular_excess = _reversal_excess(
        current=current.roll_pitch_rate,
        previous=previous.roll_pitch_rate,
        min_magnitude=chatter_cfg.min_roll_pitch_reversal_rate,
    )

    # angular_reversed has shape [num_envs, 2] for the roll and pitch axes,
    # while small_tilt_change has shape [num_envs]. `None` inserts a
    # singleton axis, producing [num_envs, 1], so PyTorch broadcasts the same
    # per-environment tilt gate across both angular axes.
    chatter_active = angular_reversed & small_tilt_change[:, None]

    penalty_per_axis = angular_excess.square() * chatter_active.to(dtype=current.roll_pitch_rate.dtype)

    return torch.sum(penalty_per_axis, dim=-1)
