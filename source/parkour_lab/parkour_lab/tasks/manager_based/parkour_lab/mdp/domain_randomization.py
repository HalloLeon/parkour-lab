# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Staged dynamics randomization and control delays."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.modifiers import ModifierBase, ModifierCfg

##
# Configuration
##


@configclass
class DomainRandomizationCfg:
    """Ranges for the nominal, narrow, and wide training stages.

    ``off`` is the nominal-learning stage. ``narrow`` applies half of each
    configured range around its nominal value, and ``wide`` applies the complete
    range. A resumed run should progress in that order rather than enabling the
    widest perturbations before the nominal task has been learned.
    """

    stage: Literal["off", "narrow", "wide"] = "off"

    static_friction_range: tuple[float, float] = (0.7, 1.3)
    dynamic_friction_range: tuple[float, float] = (0.6, 1.2)
    restitution_range: tuple[float, float] = (0.0, 0.1)

    added_trunk_mass_range_kg: tuple[float, float] = (-0.5, 1.0)
    trunk_com_x_range_m: tuple[float, float] = (-0.02, 0.02)
    trunk_com_y_range_m: tuple[float, float] = (-0.02, 0.02)
    trunk_com_z_range_m: tuple[float, float] = (-0.01, 0.01)
    actuator_gain_scale_range: tuple[float, float] = (0.8, 1.2)

    initial_xy_range_m: tuple[float, float] = (-0.05, 0.05)
    initial_yaw_range_rad: tuple[float, float] = (-0.1, 0.1)
    initial_linear_velocity_range_m_s: tuple[float, float] = (-0.2, 0.2)
    initial_angular_velocity_range_rad_s: tuple[float, float] = (-0.2, 0.2)
    push_interval_range_s: tuple[float, float] = (8.0, 12.0)
    push_velocity_range_m_s: tuple[float, float] = (-0.3, 0.3)

    max_action_delay_steps: int = 1
    max_proprioception_delay_steps: int = 1
    angular_velocity_noise: float = 0.04
    gravity_noise: float = 0.02
    joint_position_noise_rad: float = 0.01
    joint_velocity_noise_rad_s: float = 0.1

    @property
    def stage_scale(self) -> float:
        """Return zero, half, or full range for the selected training stage."""

        try:
            return {"off": 0.0, "narrow": 0.5, "wide": 1.0}[self.stage]
        except KeyError as error:
            raise ValueError(
                "domain randomization stage must be 'off', 'narrow', or 'wide', "
                f"got {self.stage!r}."
            ) from error


##
# Runtime terms
##


class DelayedJointPositionAction(JointPositionAction):
    """Joint-position action with a per-environment control-step delay."""

    cfg: DelayedJointPositionActionCfg

    def __init__(
        self,
        cfg: DelayedJointPositionActionCfg,
        env: ManagerBasedEnv,
    ) -> None:
        super().__init__(cfg, env)
        self._delay_buffer = DelayBuffer(
            history_length=cfg.max_delay_steps,
            batch_size=self.num_envs,
            device=self.device,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Delay raw policy actions before the normal affine transformation."""

        super().process_actions(self._delay_buffer.compute(actions))

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Clear selected histories and sample their next episode delays."""

        super().reset(env_ids)
        batch_ids, count = _batch_selection(env_ids, self.num_envs)
        time_lags = _sample_lags(
            count,
            self.cfg.min_delay_steps,
            self.cfg.max_delay_steps,
            self.device,
        )
        self._delay_buffer.set_time_lag(time_lags, batch_ids=batch_ids)
        self._delay_buffer.reset(batch_ids=batch_ids)


class ProprioceptionDelay(ModifierBase):
    """Delay a proprioceptive tensor without adding a history dimension."""

    def __init__(
        self,
        cfg: ProprioceptionDelayCfg,
        data_dim: tuple[int, ...],
        device: str,
    ) -> None:
        super().__init__(cfg, data_dim, device)
        self._delay_buffer = DelayBuffer(
            history_length=cfg.max_delay_steps,
            batch_size=data_dim[0],
            device=device,
        )

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Return the selected current or past sample with unchanged shape."""

        return self._delay_buffer.compute(data)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear selected histories and sample their next episode delays."""

        batch_ids, count = _batch_selection(env_ids, self._data_dim[0])
        time_lags = _sample_lags(
            count,
            self._cfg.min_delay_steps,
            self._cfg.max_delay_steps,
            self._device,
        )
        self._delay_buffer.set_time_lag(time_lags, batch_ids=batch_ids)
        self._delay_buffer.reset(batch_ids=batch_ids)


class RecordPrivilegedDynamics(ManagerTermBase):
    """Record the actual persistent dynamics randomized for each environment."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        joint_ids = self.asset_cfg.joint_ids
        joint_names = (
            self.asset.joint_names[joint_ids]
            if isinstance(joint_ids, slice)
            else [self.asset.joint_names[index] for index in joint_ids]
        )
        self.component_names = privileged_dynamics_component_names(joint_names)

        # ObservationManager queries term dimensions before startup events run.
        # Allocate the final shape now; the startup call fills it after every
        # persistent physics randomizer has written its sampled values.
        self.values = torch.zeros(
            (env.num_envs, len(self.component_names)),
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
    ) -> None:
        """Read actual simulator properties after startup randomization."""

        del env, env_ids, asset_cfg
        body_ids = self.asset_cfg.body_ids
        joint_ids = self.asset_cfg.joint_ids

        # N: environments, B: selected bodies (B=1), S: collision shapes,
        # J: selected joints.
        # Post-randomization PhysX masses and nominal asset masses: (N, B).
        masses = self.asset.root_physx_view.get_masses()[:, body_ids]
        default_masses = self.asset.data.default_mass[:, body_ids]
        # Relative change from nominal (zero means unchanged): (N, B).
        mass_ratio = (
            masses.to(self.values) / default_masses.to(self.values).clamp_min(1.0e-6)
            - 1.0
        )
        # Post-randomization local COM xyz: (N, B, 3).
        centers_of_mass = self.asset.root_physx_view.get_coms()[:, body_ids, :3].to(
            self.values
        )
        # Per-shape (static friction, dynamic friction, restitution): (N, S, 3).
        # Mean across shapes: (N, 3).
        materials = self.asset.root_physx_view.get_material_properties().to(self.values)
        mean_material = materials.mean(dim=1)

        # Explicit actuator models such as A1's DC motors keep their operative
        # gains on the actuator objects; the PhysX joint buffers remain zero.
        # Assemble the current global gain tensors from those actuator-local
        # values so this vector records the gains that actually produce torque.
        # Full-robot operative gains: (N, number of robot joints).
        stiffness = self.asset.data.default_joint_stiffness.clone()
        damping = self.asset.data.default_joint_damping.clone()
        for actuator in self.asset.actuators.values():
            stiffness[:, actuator.joint_indices] = actuator.stiffness
            damping[:, actuator.joint_indices] = actuator.damping

        # Selected operative gains and relative changes from nominal: (N, J).
        stiffness = stiffness[:, joint_ids]
        default_stiffness = self.asset.data.default_joint_stiffness[:, joint_ids]
        stiffness_ratio = (stiffness / default_stiffness.clamp_min(1.0e-6) - 1.0).to(
            self.values
        )
        damping = damping[:, joint_ids]
        default_damping = self.asset.data.default_joint_damping[:, joint_ids]
        damping_ratio = (damping / default_damping.clamp_min(1.0e-6) - 1.0).to(
            self.values
        )

        # Concatenated feature vector: (N, 4B + 3 + 2J) = (N, 7 + 2J).
        self.values.copy_(
            torch.cat(
                (
                    mass_ratio,
                    centers_of_mass.flatten(start_dim=1),
                    mean_material,
                    stiffness_ratio,
                    damping_ratio,
                ),
                dim=-1,
            )
        )


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for delayed joint-position actions."""

    class_type: type[ActionTerm] = DelayedJointPositionAction
    min_delay_steps: int = 0
    max_delay_steps: int = 0


@configclass
class ProprioceptionDelayCfg(ModifierCfg):
    """Per-environment control-step delay for one proprioceptive signal."""

    func: type[ProprioceptionDelay] = ProprioceptionDelay
    min_delay_steps: int = 0
    max_delay_steps: int = 0


##
# Runtime helpers
##


def privileged_dynamics(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the persistent randomized dynamics recorded at startup."""

    recorder = env.event_manager.get_term_cfg("record_privileged_dynamics").func
    if not isinstance(recorder, RecordPrivilegedDynamics):
        raise TypeError("record_privileged_dynamics must use RecordPrivilegedDynamics.")
    return recorder.values.clone()


def privileged_dynamics_component_names(
    joint_names: Sequence[str],
) -> tuple[str, ...]:
    """Return the stable semantic order of the privileged dynamics vector."""

    return (
        "trunk_mass_ratio_minus_one",
        "trunk_com_x",
        "trunk_com_y",
        "trunk_com_z",
        "mean_static_friction",
        "mean_dynamic_friction",
        "mean_restitution",
        *(f"joint_stiffness_ratio_minus_one:{name}" for name in joint_names),
        *(f"joint_damping_ratio_minus_one:{name}" for name in joint_names),
    )


def scaled_delay(maximum: int, scale: float) -> int:
    """Scale an integer delay while retaining one step in the narrow stage."""

    return math.ceil(maximum * scale)


def scaled_range(
    bounds: tuple[float, float],
    scale: float,
    *,
    center: float = 0.0,
) -> tuple[float, float]:
    """Contract a configured interval around its nominal center."""

    return (
        center + scale * (bounds[0] - center),
        center + scale * (bounds[1] - center),
    )


def _batch_selection(
    env_ids: Sequence[int] | slice | None,
    num_envs: int,
) -> tuple[Sequence[int] | slice | None, int]:
    """Return DelayBuffer indices and the number of lags to sample."""

    if env_ids is None or isinstance(env_ids, slice):
        return env_ids, num_envs
    return env_ids, len(env_ids)


def _sample_lags(
    count: int,
    minimum: int,
    maximum: int,
    device: str,
) -> torch.Tensor:
    """Sample inclusive delays with DelayBuffer's integer dtype."""

    return torch.randint(
        minimum,
        maximum + 1,
        (count,),
        device=device,
        dtype=torch.int,
    )


__all__ = [
    "DelayedJointPositionAction",
    "DelayedJointPositionActionCfg",
    "DomainRandomizationCfg",
    "ProprioceptionDelay",
    "ProprioceptionDelayCfg",
    "RecordPrivilegedDynamics",
    "privileged_dynamics",
    "privileged_dynamics_component_names",
    "scaled_delay",
    "scaled_range",
]
