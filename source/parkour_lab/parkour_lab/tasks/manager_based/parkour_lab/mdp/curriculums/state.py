# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mutable per-environment state owned by the parkour curriculum."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .config import ParkourCurriculumCfg

# Increment when the persistent dataclass field layout changes.
_CHECKPOINT_VERSION = 1


@dataclass(slots=True)
class ParkourCurriculumState:
    """All persistent, mutable curriculum memory.

    ``frontier_levels`` is each environment's current mastery target; ``-1``
    means it has not yet been initialized from its first sampled row.
    ``success_history`` records recent frontier completions used for promotion.
    ``stalled_history`` records eligible low-progress frontier failures used for
    demotion. ``demotion_grace_episodes_remaining`` counts protected frontier
    attempts after promotion. Replay and manual-reset episodes do not alter any
    of this evidence.

    The currently sampled episode row remains owned by
    ``TerrainImporter.terrain_levels`` because it is derived from the frontier
    and replay policy rather than curriculum memory.
    """

    frontier_levels: torch.Tensor
    success_history: torch.Tensor
    stalled_history: torch.Tensor
    demotion_grace_episodes_remaining: torch.Tensor

    @classmethod
    def allocate(
        cls,
        num_envs: int,
        device: str | torch.device,
        cfg: ParkourCurriculumCfg,
    ) -> ParkourCurriculumState:
        """Allocate empty curriculum memory on the environment device."""

        import torch

        return cls(
            frontier_levels=torch.full(
                (num_envs,),
                -1,
                device=device,
                dtype=torch.long,
            ),
            success_history=torch.zeros(
                (num_envs, cfg.promotion_window),
                device=device,
                dtype=torch.bool,
            ),
            stalled_history=torch.zeros(
                (num_envs, cfg.demotion_window),
                device=device,
                dtype=torch.bool,
            ),
            demotion_grace_episodes_remaining=torch.zeros(
                num_envs,
                device=device,
                dtype=torch.long,
            ),
        )

    def state_dict(self, cfg: ParkourCurriculumCfg) -> dict[str, object]:
        """Return portable curriculum memory for a training checkpoint."""

        return {
            "version": _CHECKPOINT_VERSION,
            "family_names": tuple(cfg.family_names),
            **{field.name: getattr(self, field.name).detach().cpu().clone() for field in fields(self)},
        }

    def load_state_dict(
        self,
        state: dict[str, object],
        cfg: ParkourCurriculumCfg,
    ) -> None:
        """Atomically restore checkpoint memory after layout validation."""

        import torch

        if state.get("version") != _CHECKPOINT_VERSION:
            raise ValueError(f"Unsupported parkour curriculum state version: {state.get('version')!r}.")
        if tuple(state.get("family_names", ())) != tuple(cfg.family_names):
            raise ValueError("Checkpoint and environment use different parkour families.")

        tensors: dict[str, torch.Tensor] = {}
        for field in fields(self):
            saved = state.get(field.name)
            target = getattr(self, field.name)
            if not isinstance(saved, torch.Tensor):
                raise TypeError(f"Checkpoint curriculum state {field.name!r} is not a tensor.")
            if saved.shape != target.shape or saved.dtype != target.dtype:
                raise ValueError(
                    f"Checkpoint curriculum state {field.name!r} has shape/dtype "
                    f"{tuple(saved.shape)}/{saved.dtype}; expected {tuple(target.shape)}/{target.dtype}."
                )
            tensors[field.name] = saved

        frontiers = tensors["frontier_levels"]
        if torch.any((frontiers < -1) | (frontiers > cfg.max_level)):
            raise ValueError(f"Checkpoint curriculum frontiers must be between -1 and {cfg.max_level}.")
        if torch.any(tensors["demotion_grace_episodes_remaining"] < 0):
            raise ValueError("Checkpoint curriculum grace counters must be non-negative.")

        for name, saved in tensors.items():
            target = getattr(self, name)
            target.copy_(saved.to(device=target.device))
