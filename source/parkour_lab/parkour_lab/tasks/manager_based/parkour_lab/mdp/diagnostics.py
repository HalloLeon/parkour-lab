# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior-neutral episode diagnostics for teacher training."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from ._shared.runtime import _all_env_ids
from .commands import get_target_speed
from .navigation import geometry, route

GO2_FOOT_NAMES = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
)
"""Canonical foot order used by named training diagnostics."""

GO2_LEG_NAMES = ("FL", "FR", "RL", "RR")
"""Canonical leg order corresponding to :data:`GO2_FOOT_NAMES`."""

GO2_JOINT_TYPES = ("hip", "thigh", "calf")
"""Per-leg joint order used by the Unitree Go2 asset."""

GO2_JOINT_NAMES = tuple(
    f"{leg_name}_{joint_type}_joint" for leg_name in GO2_LEG_NAMES for joint_type in GO2_JOINT_TYPES
)
"""Canonical leg-major order of the twelve Unitree Go2 joints."""


class TrainingDiagnostics(ManagerTermBase):
    """Accumulate gait, control, task, and body metrics without changing reward.

    Isaac Lab evaluates reward terms after physics and termination computation,
    but before resetting completed environments. Registering this stateful term
    with a nonzero manager weight therefore captures the terminal sample while
    returning an identically zero reward. The curriculum reporter reads the
    accumulated episode statistics before :class:`RewardManager` resets these
    buffers.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self._asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self._base_height_sensor_cfg: SceneEntityCfg = cfg.params["base_height_sensor_cfg"]
        self._feet_sensor_cfg: SceneEntityCfg = cfg.params["feet_sensor_cfg"]
        self._waypoint_marker_cfg: SceneEntityCfg = cfg.params["waypoint_marker_cfg"]
        self._contact_threshold = _finite_nonnegative(
            cfg.params["contact_threshold"],
            name="contact_threshold",
        )
        self._joint_velocity_limit_ratio = _finite_interval(
            cfg.params["joint_velocity_limit_ratio"],
            name="joint_velocity_limit_ratio",
            lower=0.0,
            upper=1.0,
        )
        self._reverse_speed_threshold_mps = _finite_nonnegative(
            cfg.params["reverse_speed_threshold_mps"],
            name="reverse_speed_threshold_mps",
        )
        self._torque_clip_tolerance_nm = _finite_nonnegative(
            cfg.params["torque_clip_tolerance_nm"],
            name="torque_clip_tolerance_nm",
        )

        asset = env.scene[self._asset_cfg.name]
        if not isinstance(asset, Articulation):
            raise TypeError(f"Expected '{self._asset_cfg.name}' to be an Articulation, got {type(asset).__name__}.")
        feet_sensor = env.scene[self._feet_sensor_cfg.name]
        if not isinstance(feet_sensor, ContactSensor):
            raise TypeError(
                f"Expected '{self._feet_sensor_cfg.name}' to be a ContactSensor, got {type(feet_sensor).__name__}."
            )
        if not feet_sensor.cfg.track_air_time:
            raise ValueError(
                f"Contact sensor '{self._feet_sensor_cfg.name}' must set track_air_time=True for touchdown diagnostics."
            )
        if not math.isclose(
            float(feet_sensor.cfg.force_threshold),
            self._contact_threshold,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError(
                f"Contact sensor '{self._feet_sensor_cfg.name}' force_threshold must match "
                f"the diagnostic contact threshold {self._contact_threshold}."
            )
        base_height_sensor = env.scene[self._base_height_sensor_cfg.name]
        if not isinstance(base_height_sensor, RayCaster):
            raise TypeError(
                f"Expected '{self._base_height_sensor_cfg.name}' to be a RayCaster, "
                f"got {type(base_height_sensor).__name__}."
            )

        _require_resolved_names(
            self._asset_cfg.body_names,
            GO2_FOOT_NAMES,
            role="articulation feet",
        )
        _require_resolved_names(
            self._feet_sensor_cfg.body_names,
            GO2_FOOT_NAMES,
            role="contact-sensor feet",
        )
        _require_resolved_names(
            self._asset_cfg.joint_names,
            GO2_JOINT_NAMES,
            role="articulation joints",
        )
        if self._asset_cfg.body_ids is None or self._asset_cfg.joint_ids is None:
            raise ValueError("Training diagnostics require resolved articulation body_ids and joint_ids.")
        if self._feet_sensor_cfg.body_ids is None:
            raise ValueError("Training diagnostics require resolved contact-sensor body_ids.")

        action_term_name = str(cfg.params["action_term_name"])
        action_term = env.action_manager.get_term(action_term_name)
        action_joint_names = tuple(getattr(action_term, "_joint_names", ()))
        if set(action_joint_names) != set(GO2_JOINT_NAMES) or len(action_joint_names) != len(GO2_JOINT_NAMES):
            raise RuntimeError(
                f"Action term '{action_term_name}' must control exactly the canonical Go2 joints; "
                f"got {action_joint_names}."
            )
        self._action_indices = torch.tensor(
            [action_joint_names.index(name) for name in GO2_JOINT_NAMES],
            device=env.device,
            dtype=torch.long,
        )
        self._action_term = action_term
        self._asset = asset
        self._base_height_sensor = base_height_sensor
        self._feet_sensor = feet_sensor

        self._buffers: list[torch.Tensor] = []
        self._step_count = self._buffer(env)
        self._flat_step_count = self._buffer(env)
        self._base_ray_valid_count = self._buffer(env)

        self._foot_air_time_sum = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_flat_air_time_sum = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_contact_step_count = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_flat_contact_step_count = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_touchdown_count = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_flat_touchdown_count = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_vertical_force_sum = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_flat_vertical_force_sum = self._buffer(env, len(GO2_FOOT_NAMES))

        self._last_action = self._buffer(env, len(GO2_JOINT_NAMES))
        self._action_delta_square_sum = self._buffer(env, len(GO2_JOINT_NAMES))
        self._action_square_sum = self._buffer(env, len(GO2_JOINT_NAMES))
        self._applied_torque_square_sum = self._buffer(env, len(GO2_JOINT_NAMES))
        self._default_deviation_square_sum = self._buffer(env, len(GO2_JOINT_NAMES))
        self._joint_tracking_error_square_sum = self._buffer(env, len(GO2_JOINT_NAMES))
        self._torque_clip_count = self._buffer(env, len(GO2_JOINT_NAMES))
        self._velocity_limit_count = self._buffer(env, len(GO2_JOINT_NAMES))

        self._abs_lateral_speed_sum = self._buffer(env)
        self._abs_speed_error_sum = self._buffer(env)
        self._base_clearance_sum = self._buffer(env)
        self._forward_speed_sum = self._buffer(env)
        self._projected_gravity_xy_square_sum = self._buffer(env)
        self._reverse_motion_count = self._buffer(env)
        self._vertical_speed_square_sum = self._buffer(env)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        base_height_sensor_cfg: SceneEntityCfg,
        feet_sensor_cfg: SceneEntityCfg,
        waypoint_marker_cfg: SceneEntityCfg,
        action_term_name: str,
        contact_threshold: float,
        joint_velocity_limit_ratio: float,
        reverse_speed_threshold_mps: float,
        torque_clip_tolerance_nm: float,
    ) -> torch.Tensor:
        """Record the current post-physics sample and return zero reward."""

        del (
            asset_cfg,
            base_height_sensor_cfg,
            feet_sensor_cfg,
            waypoint_marker_cfg,
            action_term_name,
            contact_threshold,
            joint_velocity_limit_ratio,
            reverse_speed_threshold_mps,
            torque_clip_tolerance_nm,
        )

        self._step_count += 1.0
        flat = route.active_difficulty_indices(env) == 0
        self._flat_step_count += flat.to(dtype=torch.float32)

        feet_ids = self._feet_sensor_cfg.body_ids
        foot_forces = self._feet_sensor.data.net_forces_w[:, feet_ids]
        foot_force_norm = torch.linalg.norm(foot_forces, dim=-1)
        in_contact = foot_force_norm > self._contact_threshold
        first_contact = self._feet_sensor.compute_first_contact(env.step_dt)[:, feet_ids]
        last_air_time = self._feet_sensor.data.last_air_time[:, feet_ids]
        flat_feet = flat.unsqueeze(-1)

        self._foot_air_time_sum += last_air_time * first_contact
        self._foot_flat_air_time_sum += last_air_time * first_contact * flat_feet
        self._foot_contact_step_count += in_contact.to(dtype=torch.float32)
        self._foot_flat_contact_step_count += (in_contact & flat_feet).to(dtype=torch.float32)
        self._foot_touchdown_count += first_contact.to(dtype=torch.float32)
        self._foot_flat_touchdown_count += (first_contact & flat_feet).to(dtype=torch.float32)
        self._foot_vertical_force_sum += torch.abs(foot_forces[..., 2])
        self._foot_flat_vertical_force_sum += torch.abs(foot_forces[..., 2]) * flat_feet

        joint_ids = self._asset_cfg.joint_ids
        action = self._action_term.raw_actions[:, self._action_indices]
        joint_pos = self._asset.data.joint_pos[:, joint_ids]
        joint_pos_target = self._asset.data.joint_pos_target[:, joint_ids]
        default_joint_pos = self._asset.data.default_joint_pos[:, joint_ids]
        applied_torque = self._asset.data.applied_torque[:, joint_ids]
        computed_torque = self._asset.data.computed_torque[:, joint_ids]
        joint_velocity = self._asset.data.joint_vel[:, joint_ids]
        joint_velocity_limit = self._asset.data.soft_joint_vel_limits[:, joint_ids]

        self._action_delta_square_sum += (action - self._last_action).square()
        self._action_square_sum += action.square()
        self._applied_torque_square_sum += applied_torque.square()
        self._default_deviation_square_sum += (joint_pos - default_joint_pos).square()
        self._joint_tracking_error_square_sum += (joint_pos_target - joint_pos).square()
        self._torque_clip_count += (torch.abs(computed_torque - applied_torque) > self._torque_clip_tolerance_nm).to(
            dtype=torch.float32
        )
        valid_velocity_limit = torch.isfinite(joint_velocity_limit) & (joint_velocity_limit > 0.0)
        self._velocity_limit_count += (
            valid_velocity_limit
            & (torch.abs(joint_velocity) >= self._joint_velocity_limit_ratio * joint_velocity_limit)
        ).to(dtype=torch.float32)
        self._last_action.copy_(action)

        waypoint_direction_xy = geometry._active_waypoint_direction_xy(
            env,
            waypoint_marker_cfg=self._waypoint_marker_cfg,
            asset_cfg=self._asset_cfg,
        )
        root_velocity = self._asset.data.root_lin_vel_w
        root_velocity_xy = root_velocity[:, :2]
        forward_speed = torch.sum(root_velocity_xy * waypoint_direction_xy, dim=-1)
        lateral_velocity = root_velocity_xy - forward_speed.unsqueeze(-1) * waypoint_direction_xy
        target_speed = get_target_speed(env).to(dtype=forward_speed.dtype)

        self._abs_lateral_speed_sum += torch.linalg.norm(lateral_velocity, dim=-1)
        self._abs_speed_error_sum += torch.abs(forward_speed - target_speed)
        self._forward_speed_sum += forward_speed
        self._projected_gravity_xy_square_sum += torch.sum(
            self._asset.data.projected_gravity_b[:, :2].square(),
            dim=-1,
        )
        self._reverse_motion_count += (forward_speed < -self._reverse_speed_threshold_mps).to(dtype=torch.float32)
        self._vertical_speed_square_sum += root_velocity[:, 2].square()

        ray_hits_w = self._base_height_sensor.data.ray_hits_w
        if ray_hits_w.ndim != 3 or ray_hits_w.shape[1:] != (1, 3):
            raise RuntimeError(
                f"'{self._base_height_sensor_cfg.name}' must return one XYZ ray hit per environment, "
                f"got shape {tuple(ray_hits_w.shape)}."
            )
        valid_base_hit = torch.isfinite(ray_hits_w[:, 0]).all(dim=-1)
        base_clearance = self._asset.data.root_pos_w[:, 2] - ray_hits_w[:, 0, 2]
        self._base_ray_valid_count += valid_base_hit.to(dtype=torch.float32)
        self._base_clearance_sum += torch.where(
            valid_base_hit,
            base_clearance,
            torch.zeros_like(base_clearance),
        )

        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    def episode_metrics(self, env_ids: Sequence[int] | slice | None) -> dict[str, torch.Tensor]:
        """Summarize completed episodes selected by ``env_ids``."""

        env_ids = _all_env_ids(self._env, env_ids)
        step_count = self._step_count[env_ids].sum()
        safe_step_count = step_count.clamp_min(1.0)
        flat_step_count = self._flat_step_count[env_ids].sum()
        safe_flat_step_count = flat_step_count.clamp_min(1.0)

        contact_steps = self._foot_contact_step_count[env_ids].sum(dim=0)
        flat_contact_steps = self._foot_flat_contact_step_count[env_ids].sum(dim=0)
        touchdowns = self._foot_touchdown_count[env_ids].sum(dim=0)
        flat_touchdowns = self._foot_flat_touchdown_count[env_ids].sum(dim=0)
        vertical_forces = self._foot_vertical_force_sum[env_ids].sum(dim=0)
        flat_vertical_forces = self._foot_flat_vertical_force_sum[env_ids].sum(dim=0)

        metrics: dict[str, torch.Tensor] = {}
        for foot_index, foot_name in enumerate(GO2_FOOT_NAMES):
            prefix = f"gait/{foot_name}"
            metrics[f"{prefix}/contact_fraction"] = contact_steps[foot_index] / safe_step_count
            metrics[f"{prefix}/flat_contact_fraction"] = flat_contact_steps[foot_index] / safe_flat_step_count
            metrics[f"{prefix}/flat_mean_completed_air_time_s"] = self._foot_flat_air_time_sum[
                env_ids, foot_index
            ].sum() / flat_touchdowns[foot_index].clamp_min(1.0)
            metrics[f"{prefix}/flat_touchdown_rate_hz"] = flat_touchdowns[foot_index] / (
                safe_flat_step_count * float(self._env.step_dt)
            )
            metrics[f"{prefix}/flat_vertical_load_fraction"] = flat_vertical_forces[
                foot_index
            ] / flat_vertical_forces.sum().clamp_min(torch.finfo(torch.float32).eps)
            metrics[f"{prefix}/mean_completed_air_time_s"] = self._foot_air_time_sum[
                env_ids, foot_index
            ].sum() / touchdowns[foot_index].clamp_min(1.0)
            metrics[f"{prefix}/touchdown_rate_hz"] = touchdowns[foot_index] / (
                safe_step_count * float(self._env.step_dt)
            )
            metrics[f"{prefix}/vertical_load_fraction"] = vertical_forces[foot_index] / vertical_forces.sum().clamp_min(
                torch.finfo(torch.float32).eps
            )

        action_delta_square_sum = self._action_delta_square_sum[env_ids].sum(dim=0).reshape(len(GO2_LEG_NAMES), -1)
        action_square_sum = self._action_square_sum[env_ids].sum(dim=0).reshape(len(GO2_LEG_NAMES), -1)
        applied_torque_square_sum = self._applied_torque_square_sum[env_ids].sum(dim=0).reshape(len(GO2_LEG_NAMES), -1)
        default_deviation_square_sum = (
            self._default_deviation_square_sum[env_ids].sum(dim=0).reshape(len(GO2_LEG_NAMES), -1)
        )
        tracking_error_square_sum = (
            self._joint_tracking_error_square_sum[env_ids].sum(dim=0).reshape(len(GO2_LEG_NAMES), -1)
        )
        torque_clip_count = self._torque_clip_count[env_ids].sum(dim=0).reshape(len(GO2_LEG_NAMES), -1)
        velocity_limit_count = self._velocity_limit_count[env_ids].sum(dim=0).reshape(len(GO2_LEG_NAMES), -1)
        joint_sample_count = safe_step_count * len(GO2_JOINT_TYPES)
        for leg_index, leg_name in enumerate(GO2_LEG_NAMES):
            prefix = f"control/{leg_name}"
            metrics[f"{prefix}/action_rms"] = torch.sqrt(action_square_sum[leg_index].sum() / joint_sample_count)
            metrics[f"{prefix}/action_rate_rms"] = torch.sqrt(
                action_delta_square_sum[leg_index].sum() / joint_sample_count
            )
            metrics[f"{prefix}/applied_torque_rms_nm"] = torch.sqrt(
                applied_torque_square_sum[leg_index].sum() / joint_sample_count
            )
            metrics[f"{prefix}/default_joint_deviation_rms_rad"] = torch.sqrt(
                default_deviation_square_sum[leg_index].sum() / joint_sample_count
            )
            metrics[f"{prefix}/joint_tracking_error_rms_rad"] = torch.sqrt(
                tracking_error_square_sum[leg_index].sum() / joint_sample_count
            )
            metrics[f"{prefix}/torque_clip_fraction"] = torque_clip_count[leg_index].sum() / joint_sample_count
            metrics[f"{prefix}/velocity_limit_fraction"] = velocity_limit_count[leg_index].sum() / joint_sample_count

        torque_clip_by_leg_and_type = torque_clip_count
        joint_type_sample_count = safe_step_count * len(GO2_LEG_NAMES)
        for joint_type_index, joint_type in enumerate(GO2_JOINT_TYPES):
            metrics[f"control/joint_type/{joint_type}/torque_clip_fraction"] = (
                torque_clip_by_leg_and_type[:, joint_type_index].sum() / joint_type_sample_count
            )

        metrics.update(
            {
                "body/base_ray_miss_fraction": 1.0 - self._base_ray_valid_count[env_ids].sum() / safe_step_count,
                "body/mean_valid_base_clearance_m": self._base_clearance_sum[env_ids].sum()
                / self._base_ray_valid_count[env_ids].sum().clamp_min(1.0),
                "body/projected_gravity_xy_rms": torch.sqrt(
                    self._projected_gravity_xy_square_sum[env_ids].sum() / safe_step_count
                ),
                "body/vertical_speed_rms_mps": torch.sqrt(
                    self._vertical_speed_square_sum[env_ids].sum() / safe_step_count
                ),
                "episode/flat_step_fraction": flat_step_count / safe_step_count,
                "episode/mean_duration_s": self._step_count[env_ids].mean() * float(self._env.step_dt),
                "episode/mean_final_geometric_progress": route.normalized_course_progress(
                    self._env,
                    env_ids,
                ).mean(),
                "episode/mean_final_waypoint_progress": route.normalized_waypoint_progress(
                    self._env,
                    env_ids,
                ).mean(),
                "task/mean_abs_lateral_speed_mps": self._abs_lateral_speed_sum[env_ids].sum() / safe_step_count,
                "task/mean_abs_speed_error_mps": self._abs_speed_error_sum[env_ids].sum() / safe_step_count,
                "task/mean_forward_speed_mps": self._forward_speed_sum[env_ids].sum() / safe_step_count,
                "task/reverse_motion_fraction": self._reverse_motion_count[env_ids].sum() / safe_step_count,
            }
        )
        return metrics

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Clear selected completed-episode accumulators."""

        env_ids = _all_env_ids(self._env, env_ids)
        for buffer in self._buffers:
            buffer[env_ids] = 0.0

    def _buffer(self, env: ManagerBasedRLEnv, width: int | None = None) -> torch.Tensor:
        shape = (env.num_envs,) if width is None else (env.num_envs, width)
        buffer = torch.zeros(shape, device=env.device, dtype=torch.float32)
        self._buffers.append(buffer)
        return buffer


def report_training_diagnostics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
    reward_term_name: str,
) -> dict[str, torch.Tensor]:
    """Expose the logging-only reward term's completed-episode metrics."""

    term = env.reward_manager.get_term_cfg(reward_term_name).func
    if not isinstance(term, TrainingDiagnostics):
        raise TypeError(
            f"Reward term '{reward_term_name}' must resolve to TrainingDiagnostics, got {type(term).__name__}."
        )
    return term.episode_metrics(env_ids)


def _finite_interval(value: object, *, name: str, lower: float, upper: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not lower < value <= upper:
        raise ValueError(f"{name} must be finite and in ({lower}, {upper}].")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return value


def _require_resolved_names(
    actual_names: Sequence[str] | None,
    expected_names: Sequence[str],
    *,
    role: str,
) -> None:
    actual = tuple(actual_names or ())
    expected = tuple(expected_names)
    if actual != expected:
        raise RuntimeError(f"Training-diagnostic {role} must resolve as {expected}, got {actual}.")
