# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Behavior-neutral episode diagnostics for teacher training."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

from ._shared.go2 import (
    GO2_FOOT_NAMES,
    GO2_JOINT_NAMES,
    GO2_JOINT_TYPES,
    GO2_LEG_NAMES,
)
from ._shared.runtime import _all_env_ids
from ._shared.robot import _root_forward_xy_w
from .commands import (
    PROVISIONAL_ORACLE_RESIDUAL_THRESHOLD_RAD,
    get_requested_travel_direction_yaw_xy,
    get_target_speed,
    get_target_yaw_rate,
    wrapped_heading_residual_rad,
)
from .navigation import geometry, route

_MIN_GAIT_DIAGNOSTIC_DURATION_S = 0.5
"""Ignore shorter episode fragments in per-episode gait distributions."""

_MOVEMENT_DIRECTION_MIN_PLANAR_SPEED_M_S = 0.10


@dataclass(frozen=True, slots=True)
class EvaluationStep:
    """One post-physics, pre-reset transition snapshot for fixed evaluation."""

    metrics: dict[str, torch.Tensor]
    root_position_xy: torch.Tensor
    active_waypoint_indices: torch.Tensor
    route_cross_track_error_m: torch.Tensor
    waypoint_changed: torch.Tensor
    foot_contact: torch.Tensor
    foot_touchdown: torch.Tensor
    foot_world_z_force: torch.Tensor


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
        self._base_height_sensor_cfg: SceneEntityCfg = cfg.params[
            "base_height_sensor_cfg"
        ]
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
        self._capture_evaluation = bool(
            cfg.params.get("capture_evaluation_step", False)
        )

        asset = env.scene[self._asset_cfg.name]
        if not isinstance(asset, Articulation):
            raise TypeError(
                f"Expected '{self._asset_cfg.name}' to be an Articulation, got {type(asset).__name__}."
            )
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
            raise ValueError(
                "Training diagnostics require resolved articulation body_ids and joint_ids."
            )
        if self._feet_sensor_cfg.body_ids is None:
            raise ValueError(
                "Training diagnostics require resolved contact-sensor body_ids."
            )

        action_term_name = str(cfg.params["action_term_name"])
        action_term = env.action_manager.get_term(action_term_name)
        action_joint_names = tuple(action_term.IO_descriptor.joint_names)
        if set(action_joint_names) != set(GO2_JOINT_NAMES) or len(
            action_joint_names
        ) != len(GO2_JOINT_NAMES):
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
        self._evaluation_step: EvaluationStep | None = None
        self._evaluation_reward_slots: dict[str, tuple[int, float]] | None = None

        self._buffers: list[torch.Tensor] = []
        self._step_count = self._buffer(env)
        self._bootstrap_flat_step_count = self._buffer(env)
        self._base_ray_valid_count = self._buffer(env)

        self._foot_air_time_sum = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_bootstrap_flat_air_time_sum = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_bootstrap_flat_contact_step_count = self._buffer(
            env, len(GO2_FOOT_NAMES)
        )
        self._foot_bootstrap_flat_current_noncontact_step_count = self._buffer(
            env, len(GO2_FOOT_NAMES)
        )
        self._foot_bootstrap_flat_max_noncontact_step_count = self._buffer(
            env, len(GO2_FOOT_NAMES)
        )
        self._foot_bootstrap_flat_touchdown_count = self._buffer(
            env, len(GO2_FOOT_NAMES)
        )
        self._foot_bootstrap_flat_world_z_force_sum = self._buffer(
            env, len(GO2_FOOT_NAMES)
        )
        self._foot_contact_step_count = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_current_noncontact_step_count = self._buffer(
            env, len(GO2_FOOT_NAMES)
        )
        self._foot_max_noncontact_step_count = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_touchdown_count = self._buffer(env, len(GO2_FOOT_NAMES))
        self._foot_world_z_force_sum = self._buffer(env, len(GO2_FOOT_NAMES))
        self._has_previous_contact_sample = torch.zeros(
            env.num_envs, device=env.device, dtype=torch.bool
        )
        self._buffers.append(self._has_previous_contact_sample)

        self._last_action = self._buffer(env, len(GO2_JOINT_NAMES))
        self._action_delta_square_sum = self._buffer(env, len(GO2_JOINT_NAMES))
        self._action_square_sum = self._buffer(env, len(GO2_JOINT_NAMES))
        self._applied_torque_square_sum = self._buffer(env, len(GO2_JOINT_NAMES))
        self._default_deviation_square_sum = self._buffer(env, len(GO2_JOINT_NAMES))
        self._joint_tracking_error_square_sum = self._buffer(env, len(GO2_JOINT_NAMES))
        self._joint_position_soft_limit_violation_count = self._buffer(
            env, len(GO2_JOINT_NAMES)
        )
        self._joint_soft_limit_valid_count = self._buffer(env, len(GO2_JOINT_NAMES))
        self._joint_target_soft_limit_violation_count = self._buffer(
            env, len(GO2_JOINT_NAMES)
        )
        self._torque_clip_count = self._buffer(env, len(GO2_JOINT_NAMES))
        self._velocity_limit_count = self._buffer(env, len(GO2_JOINT_NAMES))

        self._abs_lateral_speed_sum = self._buffer(env)
        self._abs_speed_error_sum = self._buffer(env)
        self._base_clearance_sum = self._buffer(env)
        self._bootstrap_flat_projected_gravity_xy_square_sum = self._buffer(env)
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
        capture_evaluation_step: bool = False,
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
            capture_evaluation_step,
        )

        self._step_count += 1.0
        # Difficulty row zero is the shared bootstrap course. Do not call
        # horizontal approach/landing phases inside obstacle courses "flat":
        # they currently have no explicit route-phase metadata.
        bootstrap_flat = route.active_difficulty_indices(env) == 0
        self._bootstrap_flat_step_count += bootstrap_flat.to(dtype=torch.float32)

        feet_ids = self._feet_sensor_cfg.body_ids
        foot_forces = self._feet_sensor.data.net_forces_w[:, feet_ids]
        foot_force_norm = torch.linalg.norm(foot_forces, dim=-1)
        in_contact = foot_force_norm > self._contact_threshold
        first_contact = self._feet_sensor.compute_first_contact(env.step_dt)[
            :, feet_ids
        ]
        # ContactSensor treats a planted reset stance as a newly established
        # contact. Exclude that first sample so a foot that lifts immediately
        # and is then carried for the entire episode still has zero touchdowns.
        valid_touchdown = first_contact & self._has_previous_contact_sample.unsqueeze(
            -1
        )
        last_air_time = self._feet_sensor.data.last_air_time[:, feet_ids]
        bootstrap_flat_feet = bootstrap_flat.unsqueeze(-1)

        current_noncontact_steps = torch.where(
            in_contact,
            torch.zeros_like(self._foot_current_noncontact_step_count),
            self._foot_current_noncontact_step_count + 1.0,
        )
        self._foot_current_noncontact_step_count.copy_(current_noncontact_steps)
        self._foot_max_noncontact_step_count.copy_(
            torch.maximum(
                self._foot_max_noncontact_step_count, current_noncontact_steps
            )
        )

        bootstrap_flat_noncontact = bootstrap_flat_feet & ~in_contact
        bootstrap_flat_current_noncontact_steps = torch.where(
            bootstrap_flat_noncontact,
            self._foot_bootstrap_flat_current_noncontact_step_count + 1.0,
            torch.zeros_like(self._foot_bootstrap_flat_current_noncontact_step_count),
        )
        self._foot_bootstrap_flat_current_noncontact_step_count.copy_(
            bootstrap_flat_current_noncontact_steps
        )
        self._foot_bootstrap_flat_max_noncontact_step_count.copy_(
            torch.maximum(
                self._foot_bootstrap_flat_max_noncontact_step_count,
                bootstrap_flat_current_noncontact_steps,
            )
        )

        self._foot_air_time_sum += last_air_time * valid_touchdown
        self._foot_bootstrap_flat_air_time_sum += (
            last_air_time * valid_touchdown * bootstrap_flat_feet
        )
        self._foot_contact_step_count += in_contact.to(dtype=torch.float32)
        self._foot_bootstrap_flat_contact_step_count += (
            in_contact & bootstrap_flat_feet
        ).to(dtype=torch.float32)
        self._foot_touchdown_count += valid_touchdown.to(dtype=torch.float32)
        self._foot_bootstrap_flat_touchdown_count += (
            valid_touchdown & bootstrap_flat_feet
        ).to(dtype=torch.float32)
        # This is absolute world-frame Fz, not support-normal force on ramps.
        foot_world_z_force = torch.abs(foot_forces[..., 2])
        self._foot_world_z_force_sum += foot_world_z_force
        self._foot_bootstrap_flat_world_z_force_sum += (
            foot_world_z_force * bootstrap_flat_feet
        )
        self._has_previous_contact_sample.fill_(True)

        joint_ids = self._asset_cfg.joint_ids
        action = self._action_term.raw_actions[:, self._action_indices]
        joint_pos = self._asset.data.joint_pos[:, joint_ids]
        joint_pos_target = self._asset.data.joint_pos_target[:, joint_ids]
        default_joint_pos = self._asset.data.default_joint_pos[:, joint_ids]
        applied_torque = self._asset.data.applied_torque[:, joint_ids]
        computed_torque = self._asset.data.computed_torque[:, joint_ids]
        joint_velocity = self._asset.data.joint_vel[:, joint_ids]
        joint_position_limits = self._asset.data.soft_joint_pos_limits[:, joint_ids]
        joint_velocity_limit = self._asset.data.soft_joint_vel_limits[:, joint_ids]

        self._action_delta_square_sum += (action - self._last_action).square()
        self._action_square_sum += action.square()
        self._applied_torque_square_sum += applied_torque.square()
        self._default_deviation_square_sum += (joint_pos - default_joint_pos).square()
        self._joint_tracking_error_square_sum += (joint_pos_target - joint_pos).square()
        valid_position_limit = torch.isfinite(joint_position_limits).all(dim=-1) & (
            joint_position_limits[..., 1] > joint_position_limits[..., 0]
        )
        self._joint_soft_limit_valid_count += valid_position_limit.to(
            dtype=torch.float32
        )
        self._joint_position_soft_limit_violation_count += (
            valid_position_limit
            & (
                (joint_pos < joint_position_limits[..., 0])
                | (joint_pos > joint_position_limits[..., 1])
            )
        ).to(dtype=torch.float32)
        self._joint_target_soft_limit_violation_count += (
            valid_position_limit
            & (
                (joint_pos_target < joint_position_limits[..., 0])
                | (joint_pos_target > joint_position_limits[..., 1])
            )
        ).to(dtype=torch.float32)
        self._torque_clip_count += (
            torch.abs(computed_torque - applied_torque) > self._torque_clip_tolerance_nm
        ).to(dtype=torch.float32)
        valid_velocity_limit = torch.isfinite(joint_velocity_limit) & (
            joint_velocity_limit > 0.0
        )
        self._velocity_limit_count += (
            valid_velocity_limit
            & (
                torch.abs(joint_velocity)
                >= self._joint_velocity_limit_ratio * joint_velocity_limit
            )
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
        lateral_velocity = (
            root_velocity_xy - forward_speed.unsqueeze(-1) * waypoint_direction_xy
        )
        target_speed = get_target_speed(env).to(dtype=forward_speed.dtype)

        self._abs_lateral_speed_sum += torch.linalg.norm(lateral_velocity, dim=-1)
        self._abs_speed_error_sum += torch.abs(forward_speed - target_speed)
        self._forward_speed_sum += forward_speed
        projected_gravity_xy_square = torch.sum(
            self._asset.data.projected_gravity_b[:, :2].square(),
            dim=-1,
        )
        self._bootstrap_flat_projected_gravity_xy_square_sum += (
            projected_gravity_xy_square * bootstrap_flat.to(dtype=torch.float32)
        )
        self._projected_gravity_xy_square_sum += projected_gravity_xy_square
        self._reverse_motion_count += (
            forward_speed < -self._reverse_speed_threshold_mps
        ).to(dtype=torch.float32)
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

        if self._capture_evaluation:
            self._evaluation_step = self._capture_evaluation_step(
                env,
                foot_contact=in_contact,
                foot_touchdown=valid_touchdown,
                foot_world_z_force=foot_world_z_force,
                forward_speed=forward_speed,
                root_velocity=root_velocity,
                target_speed=target_speed,
            )

        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)

    def evaluation_step(self) -> EvaluationStep:
        """Return the latest post-physics snapshot, retained across auto-reset."""

        if self._evaluation_step is None:
            raise RuntimeError(
                "Evaluation diagnostics are unavailable before the first environment step."
            )
        return self._evaluation_step

    def _capture_evaluation_step(
        self,
        env: ManagerBasedRLEnv,
        *,
        foot_contact: torch.Tensor,
        foot_touchdown: torch.Tensor,
        foot_world_z_force: torch.Tensor,
        forward_speed: torch.Tensor,
        root_velocity: torch.Tensor,
        target_speed: torch.Tensor,
    ) -> EvaluationStep:
        """Freeze evaluator signals before Isaac Lab resets completed rows."""

        target_yaw_rate = get_target_yaw_rate(env).to(
            device=forward_speed.device,
            dtype=forward_speed.dtype,
        )
        planar_speed = torch.linalg.norm(root_velocity[:, :2], dim=-1)
        yaw_rate = self._asset.data.root_ang_vel_w[:, 2]
        abs_yaw_rate = torch.abs(yaw_rate)
        oracle_travel_direction = geometry._active_waypoint_direction_yaw_xy(env)
        # Terminations advance the route before rewards run, while the command
        # manager updates afterward. Rebuild the scripted bearing at this
        # captured state so transition diagnostics are not one control tick
        # stale. Fixed evaluation never installs an external override.
        stored_requested_direction = get_requested_travel_direction_yaw_xy(env)
        final_reference, final_reference_valid = (
            geometry._final_waypoint_direction_yaw_xy_components(env)
        )
        flat = route.active_difficulty_indices(env) == 0
        requested_travel_direction = torch.where(
            flat.unsqueeze(-1),
            oracle_travel_direction,
            torch.where(
                final_reference_valid.unsqueeze(-1),
                final_reference,
                stored_requested_direction,
            ),
        )
        oracle_residual = wrapped_heading_residual_rad(
            requested_travel_direction, oracle_travel_direction
        )
        absolute_oracle_residual = torch.abs(oracle_residual)
        moving = target_speed > 0.0
        pivoting = (~moving) & (target_yaw_rate != 0.0)
        stationary = (~moving) & (~pivoting)
        pivot_yaw_rate_error = torch.abs(yaw_rate - target_yaw_rate)

        root_forward = _root_forward_xy_w(env, self._asset_cfg)
        root_left = torch.stack((-root_forward[:, 1], root_forward[:, 0]), dim=-1)
        requested_travel_direction_w = (
            requested_travel_direction[:, :1] * root_forward
            + requested_travel_direction[:, 1:] * root_left
        )
        movement_direction_error = torch.abs(
            wrapped_heading_residual_rad(
                requested_travel_direction_w,
                root_velocity[:, :2],
            )
        )
        waypoint_changed = route.active_waypoint_changed_this_step(env)
        movement_direction_valid = (
            moving
            & (~waypoint_changed)
            & (planar_speed > _MOVEMENT_DIRECTION_MIN_PLANAR_SPEED_M_S)
        )
        all_feet_airborne = torch.all(
            self._feet_sensor.data.current_air_time[:, self._feet_sensor_cfg.body_ids]
            > 0.0,
            dim=-1,
        )

        metrics = {
            "forward_speed_m_s": forward_speed,
            "planar_speed_m_s": planar_speed,
            "abs_yaw_rate_rad_s": abs_yaw_rate,
            "moving_speed_absolute_error_m_s": torch.where(
                moving,
                torch.abs(forward_speed - target_speed),
                torch.zeros_like(target_speed),
            ),
            "stopped_planar_speed_m_s": torch.where(
                stationary, planar_speed, torch.zeros_like(planar_speed)
            ),
            "stopped_abs_yaw_rate_rad_s": torch.where(
                stationary, abs_yaw_rate, torch.zeros_like(abs_yaw_rate)
            ),
            "pivot_planar_speed_m_s": torch.where(
                pivoting, planar_speed, torch.zeros_like(planar_speed)
            ),
            "pivot_yaw_rate_absolute_error_rad_s": torch.where(
                pivoting,
                pivot_yaw_rate_error,
                torch.zeros_like(pivot_yaw_rate_error),
            ),
            "pivot_wrong_way": (pivoting & ((yaw_rate * target_yaw_rate) < 0.0)).to(
                dtype=forward_speed.dtype
            ),
            "stationary_command": stationary.to(dtype=forward_speed.dtype),
            "pivot_command": pivoting.to(dtype=forward_speed.dtype),
            "movement_direction_error_rad": torch.where(
                movement_direction_valid,
                movement_direction_error,
                torch.zeros_like(movement_direction_error),
            ),
            "movement_direction_valid": movement_direction_valid.to(
                dtype=forward_speed.dtype
            ),
            "oracle_residual_rad": oracle_residual,
            "absolute_oracle_residual_rad": torch.where(
                moving,
                absolute_oracle_residual,
                torch.zeros_like(absolute_oracle_residual),
            ),
            "oracle_residual_threshold_exceedance": (
                (absolute_oracle_residual > PROVISIONAL_ORACLE_RESIDUAL_THRESHOLD_RAD)
                & moving
            ).to(dtype=forward_speed.dtype),
            "moving_command": moving.to(dtype=forward_speed.dtype),
            "overspeed_ratio": torch.where(
                moving,
                torch.relu(forward_speed - target_speed)
                / target_speed.clamp_min(torch.finfo(target_speed.dtype).eps),
                torch.zeros_like(target_speed),
            ),
            "vertical_velocity_squared_m2_s2": root_velocity[:, 2].square(),
            "all_feet_airborne": all_feet_airborne.to(dtype=forward_speed.dtype),
            "feet_edge_contacts": self._cached_raw_reward(env, "feet_edge"),
            "undesired_body_contacts": self._cached_raw_reward(
                env, "undesired_contact"
            ),
        }
        return EvaluationStep(
            metrics={name: values.clone() for name, values in metrics.items()},
            root_position_xy=self._asset.data.root_pos_w[:, :2].clone(),
            active_waypoint_indices=route.active_waypoint_indices(env).clone(),
            route_cross_track_error_m=route.route_cross_track_error_m(env).clone(),
            waypoint_changed=waypoint_changed.clone(),
            foot_contact=foot_contact.clone(),
            foot_touchdown=foot_touchdown.clone(),
            foot_world_z_force=foot_world_z_force.clone(),
        )

    def _cached_raw_reward(
        self, env: ManagerBasedRLEnv, reward_term_name: str
    ) -> torch.Tensor:
        """Read an already-evaluated reward term without invoking it twice."""

        if self._evaluation_reward_slots is None:
            term_names = tuple(env.reward_manager.active_terms)
            diagnostic_indices = [
                index
                for index, name in enumerate(term_names)
                if env.reward_manager.get_term_cfg(name).func is self
            ]
            if len(diagnostic_indices) != 1:
                raise RuntimeError(
                    "TrainingDiagnostics must be registered exactly once in the reward manager."
                )
            diagnostic_index = diagnostic_indices[0]
            slots: dict[str, tuple[int, float]] = {}
            for name in ("feet_edge", "undesired_contact"):
                if name not in term_names:
                    raise RuntimeError(
                        f"Evaluation requires active reward term '{name}'."
                    )
                index = term_names.index(name)
                weight = float(env.reward_manager.get_term_cfg(name).weight)
                if (
                    index >= diagnostic_index
                    or not math.isfinite(weight)
                    or weight == 0.0
                ):
                    raise RuntimeError(
                        f"Evaluation reward term '{name}' must precede TrainingDiagnostics and have a finite nonzero weight."
                    )
                slots[name] = (index, weight)
            self._evaluation_reward_slots = slots

        index, weight = self._evaluation_reward_slots[reward_term_name]
        try:
            weighted_step_reward = env.reward_manager._step_reward[:, index]
        except (AttributeError, IndexError) as error:
            raise RuntimeError(
                "Evaluation requires Isaac Lab RewardManager._step_reward to expose current per-term values."
            ) from error
        return weighted_step_reward / weight

    def episode_metrics(
        self, env_ids: Sequence[int] | slice | None
    ) -> dict[str, float]:
        """Summarize completed episodes selected by ``env_ids``.

        Conditional reset-batch previews are accompanied by raw sums and
        sample counts. Their ratios remain correctly weighted after RSL-RL
        averages log dictionaries across a rollout.
        """

        env_ids = _all_env_ids(self._env, env_ids)
        episode_step_counts = self._step_count[env_ids]
        episode_bootstrap_flat_step_counts = self._bootstrap_flat_step_count[env_ids]
        step_count = episode_step_counts.sum()
        safe_step_count = step_count.clamp_min(1.0)
        bootstrap_flat_step_count = episode_bootstrap_flat_step_counts.sum()
        safe_bootstrap_flat_step_count = bootstrap_flat_step_count.clamp_min(1.0)
        minimum_gait_steps = max(
            1, math.ceil(_MIN_GAIT_DIAGNOSTIC_DURATION_S / float(self._env.step_dt))
        )
        bootstrap_flat_episode_valid = (
            episode_bootstrap_flat_step_counts >= minimum_gait_steps
        )
        bootstrap_flat_projected_gravity_valid_step_count = (
            episode_bootstrap_flat_step_counts[bootstrap_flat_episode_valid].sum()
        )
        bootstrap_flat_projected_gravity_xy_square_sum = (
            self._bootstrap_flat_projected_gravity_xy_square_sum[env_ids][
                bootstrap_flat_episode_valid
            ].sum()
        )

        contact_steps = self._foot_contact_step_count[env_ids].sum(dim=0)
        bootstrap_flat_contact_steps = self._foot_bootstrap_flat_contact_step_count[
            env_ids
        ].sum(dim=0)
        touchdowns = self._foot_touchdown_count[env_ids].sum(dim=0)
        bootstrap_flat_touchdowns = self._foot_bootstrap_flat_touchdown_count[
            env_ids
        ].sum(dim=0)
        world_z_forces = self._foot_world_z_force_sum[env_ids].sum(dim=0)
        bootstrap_flat_world_z_forces = self._foot_bootstrap_flat_world_z_force_sum[
            env_ids
        ].sum(dim=0)

        metrics = _episode_foot_participation_metrics(
            contact_step_counts=self._foot_contact_step_count[env_ids],
            max_noncontact_step_counts=self._foot_max_noncontact_step_count[env_ids],
            prefix="gait/episode",
            step_counts=episode_step_counts,
            step_dt=float(self._env.step_dt),
            touchdown_counts=self._foot_touchdown_count[env_ids],
            world_z_force_sums=self._foot_world_z_force_sum[env_ids],
        )
        metrics.update(
            _episode_foot_participation_metrics(
                contact_step_counts=self._foot_bootstrap_flat_contact_step_count[
                    env_ids
                ],
                max_noncontact_step_counts=self._foot_bootstrap_flat_max_noncontact_step_count[
                    env_ids
                ],
                prefix="gait/bootstrap_flat_episode",
                step_counts=episode_bootstrap_flat_step_counts,
                step_dt=float(self._env.step_dt),
                touchdown_counts=self._foot_bootstrap_flat_touchdown_count[env_ids],
                world_z_force_sums=self._foot_bootstrap_flat_world_z_force_sum[env_ids],
            )
        )
        for foot_index, foot_name in enumerate(GO2_FOOT_NAMES):
            prefix = f"gait/{foot_name}"
            metrics[f"{prefix}/contact_fraction"] = (
                contact_steps[foot_index] / safe_step_count
            )
            bootstrap_flat_contact_fraction = (
                bootstrap_flat_contact_steps[foot_index]
                / safe_bootstrap_flat_step_count
            )
            bootstrap_flat_mean_completed_air_time = (
                self._foot_bootstrap_flat_air_time_sum[env_ids, foot_index].sum()
                / bootstrap_flat_touchdowns[foot_index].clamp_min(1.0)
            )
            bootstrap_flat_touchdown_rate = bootstrap_flat_touchdowns[foot_index] / (
                safe_bootstrap_flat_step_count * float(self._env.step_dt)
            )
            bootstrap_flat_world_z_load_fraction = bootstrap_flat_world_z_forces[
                foot_index
            ] / bootstrap_flat_world_z_forces.sum().clamp_min(
                torch.finfo(torch.float32).eps
            )

            metrics[f"{prefix}/bootstrap_flat_contact_fraction"] = (
                bootstrap_flat_contact_fraction
            )
            metrics[f"{prefix}/bootstrap_flat_mean_completed_air_time_s"] = (
                bootstrap_flat_mean_completed_air_time
            )
            metrics[f"{prefix}/bootstrap_flat_touchdown_rate_hz"] = (
                bootstrap_flat_touchdown_rate
            )
            metrics[f"{prefix}/bootstrap_flat_world_z_load_fraction"] = (
                bootstrap_flat_world_z_load_fraction
            )

            metrics[f"{prefix}/mean_completed_air_time_s"] = self._foot_air_time_sum[
                env_ids, foot_index
            ].sum() / touchdowns[foot_index].clamp_min(1.0)
            metrics[f"{prefix}/touchdown_rate_hz"] = touchdowns[foot_index] / (
                safe_step_count * float(self._env.step_dt)
            )
            metrics[f"{prefix}/world_z_load_fraction"] = world_z_forces[
                foot_index
            ] / world_z_forces.sum().clamp_min(torch.finfo(torch.float32).eps)

        action_delta_square_sum = (
            self._action_delta_square_sum[env_ids]
            .sum(dim=0)
            .reshape(len(GO2_LEG_NAMES), -1)
        )
        action_square_sum = (
            self._action_square_sum[env_ids].sum(dim=0).reshape(len(GO2_LEG_NAMES), -1)
        )
        applied_torque_square_sum = (
            self._applied_torque_square_sum[env_ids]
            .sum(dim=0)
            .reshape(len(GO2_LEG_NAMES), -1)
        )
        default_deviation_square_sum = (
            self._default_deviation_square_sum[env_ids]
            .sum(dim=0)
            .reshape(len(GO2_LEG_NAMES), -1)
        )
        tracking_error_square_sum = (
            self._joint_tracking_error_square_sum[env_ids]
            .sum(dim=0)
            .reshape(len(GO2_LEG_NAMES), -1)
        )
        joint_position_limit_violation_count = (
            self._joint_position_soft_limit_violation_count[env_ids]
            .sum(dim=0)
            .reshape(len(GO2_LEG_NAMES), -1)
        )
        joint_soft_limit_valid_count = (
            self._joint_soft_limit_valid_count[env_ids]
            .sum(dim=0)
            .reshape(len(GO2_LEG_NAMES), -1)
        )
        joint_target_limit_violation_count = (
            self._joint_target_soft_limit_violation_count[env_ids]
            .sum(dim=0)
            .reshape(len(GO2_LEG_NAMES), -1)
        )
        torque_clip_count = (
            self._torque_clip_count[env_ids].sum(dim=0).reshape(len(GO2_LEG_NAMES), -1)
        )
        velocity_limit_count = (
            self._velocity_limit_count[env_ids]
            .sum(dim=0)
            .reshape(len(GO2_LEG_NAMES), -1)
        )
        joint_sample_count = safe_step_count * len(GO2_JOINT_TYPES)
        for leg_index, leg_name in enumerate(GO2_LEG_NAMES):
            prefix = f"control/{leg_name}"
            leg_soft_limit_valid_count = joint_soft_limit_valid_count[leg_index].sum()
            metrics[f"{prefix}/action_rms"] = torch.sqrt(
                action_square_sum[leg_index].sum() / joint_sample_count
            )
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
            position_limit_violation_count = joint_position_limit_violation_count[
                leg_index
            ].sum()
            target_limit_violation_count = joint_target_limit_violation_count[
                leg_index
            ].sum()
            metrics[f"{prefix}/joint_position_soft_limit_valid_sample_count"] = (
                leg_soft_limit_valid_count
            )
            metrics[f"{prefix}/joint_position_soft_limit_violation_count"] = (
                position_limit_violation_count
            )
            metrics[f"{prefix}/joint_position_soft_limit_violation_fraction"] = (
                position_limit_violation_count
                / leg_soft_limit_valid_count.clamp_min(1.0)
            )
            metrics[f"{prefix}/joint_position_soft_limit_valid_fraction"] = (
                leg_soft_limit_valid_count / joint_sample_count
            )
            metrics[f"{prefix}/joint_position_target_soft_limit_violation_count"] = (
                target_limit_violation_count
            )
            metrics[f"{prefix}/joint_position_target_soft_limit_violation_fraction"] = (
                target_limit_violation_count / leg_soft_limit_valid_count.clamp_min(1.0)
            )
            metrics[f"{prefix}/torque_clip_fraction"] = (
                torque_clip_count[leg_index].sum() / joint_sample_count
            )
            metrics[f"{prefix}/velocity_limit_fraction"] = (
                velocity_limit_count[leg_index].sum() / joint_sample_count
            )

        torque_clip_by_leg_and_type = torque_clip_count
        joint_type_sample_count = safe_step_count * len(GO2_LEG_NAMES)
        for joint_type_index, joint_type in enumerate(GO2_JOINT_TYPES):
            prefix = f"control/joint_type/{joint_type}"
            joint_type_soft_limit_valid_count = joint_soft_limit_valid_count[
                :, joint_type_index
            ].sum()
            position_limit_violation_count = joint_position_limit_violation_count[
                :, joint_type_index
            ].sum()
            target_limit_violation_count = joint_target_limit_violation_count[
                :, joint_type_index
            ].sum()
            metrics[f"{prefix}/joint_position_soft_limit_valid_sample_count"] = (
                joint_type_soft_limit_valid_count
            )
            metrics[f"{prefix}/joint_position_soft_limit_violation_count"] = (
                position_limit_violation_count
            )
            metrics[f"{prefix}/joint_position_soft_limit_violation_fraction"] = (
                position_limit_violation_count
                / joint_type_soft_limit_valid_count.clamp_min(1.0)
            )
            metrics[f"{prefix}/joint_position_soft_limit_valid_fraction"] = (
                joint_type_soft_limit_valid_count / joint_type_sample_count
            )
            metrics[f"{prefix}/joint_position_target_soft_limit_violation_count"] = (
                target_limit_violation_count
            )
            metrics[f"{prefix}/joint_position_target_soft_limit_violation_fraction"] = (
                target_limit_violation_count
                / joint_type_soft_limit_valid_count.clamp_min(1.0)
            )
            metrics[f"{prefix}/torque_clip_fraction"] = (
                torque_clip_by_leg_and_type[:, joint_type_index].sum()
                / joint_type_sample_count
            )

        metrics.update(
            {
                "body/base_ray_miss_fraction": 1.0
                - self._base_ray_valid_count[env_ids].sum() / safe_step_count,
                "body/bootstrap_flat_projected_gravity_valid_step_count": (
                    bootstrap_flat_projected_gravity_valid_step_count
                ),
                "body/bootstrap_flat_projected_gravity_xy_rms": torch.sqrt(
                    bootstrap_flat_projected_gravity_xy_square_sum
                    / bootstrap_flat_projected_gravity_valid_step_count.clamp_min(1.0)
                ),
                "body/bootstrap_flat_projected_gravity_xy_square_sum": bootstrap_flat_projected_gravity_xy_square_sum,
                "body/mean_valid_base_clearance_m": self._base_clearance_sum[
                    env_ids
                ].sum()
                / self._base_ray_valid_count[env_ids].sum().clamp_min(1.0),
                "body/projected_gravity_xy_rms": torch.sqrt(
                    self._projected_gravity_xy_square_sum[env_ids].sum()
                    / safe_step_count
                ),
                "body/vertical_speed_rms_mps": torch.sqrt(
                    self._vertical_speed_square_sum[env_ids].sum() / safe_step_count
                ),
                "episode/bootstrap_flat_step_fraction": bootstrap_flat_step_count
                / safe_step_count,
                "episode/mean_duration_s": self._step_count[env_ids].mean()
                * float(self._env.step_dt),
                "episode/mean_final_geometric_progress": route.normalized_course_progress(
                    self._env,
                    env_ids,
                ).mean(),
                "episode/mean_final_waypoint_progress": route.normalized_waypoint_progress(
                    self._env,
                    env_ids,
                ).mean(),
                "task/mean_abs_lateral_speed_mps": self._abs_lateral_speed_sum[
                    env_ids
                ].sum()
                / safe_step_count,
                "task/mean_abs_speed_error_mps": self._abs_speed_error_sum[
                    env_ids
                ].sum()
                / safe_step_count,
                "task/mean_forward_speed_mps": self._forward_speed_sum[env_ids].sum()
                / safe_step_count,
                "task/reverse_motion_fraction": self._reverse_motion_count[
                    env_ids
                ].sum()
                / safe_step_count,
            }
        )
        return _metrics_to_python(metrics)

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
) -> dict[str, float]:
    """Expose the logging-only reward term's completed-episode metrics."""

    term = env.reward_manager.get_term_cfg(reward_term_name).func
    if not isinstance(term, TrainingDiagnostics):
        raise TypeError(
            f"Reward term '{reward_term_name}' must resolve to TrainingDiagnostics, got {type(term).__name__}."
        )
    return term.episode_metrics(env_ids)


def latest_evaluation_step(
    env: ManagerBasedRLEnv,
    reward_term_name: str = "training_diagnostics",
) -> EvaluationStep:
    """Read the transition snapshot retained by the logging reward term."""

    term = env.reward_manager.get_term_cfg(reward_term_name).func
    if not isinstance(term, TrainingDiagnostics):
        raise TypeError(
            f"Reward term '{reward_term_name}' must resolve to TrainingDiagnostics, got {type(term).__name__}."
        )
    return term.evaluation_step()


def _distribution_metrics(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, torch.Tensor]:
    """Return per-batch previews plus reconstructable sums and counts."""

    selected = values[valid]
    sample_count = valid.to(dtype=values.dtype).sum()
    if selected.numel() == 0:
        zero = values.new_zeros(())
        return {
            f"{prefix}/mean": zero,
            f"{prefix}/median": zero,
            f"{prefix}/median_weighted_sum": zero,
            f"{prefix}/p05": zero,
            f"{prefix}/p05_weighted_sum": zero,
            f"{prefix}/p95": zero,
            f"{prefix}/p95_weighted_sum": zero,
            f"{prefix}/sample_count": sample_count,
            f"{prefix}/sum": zero,
        }
    p05, median, p95 = torch.quantile(
        selected,
        selected.new_tensor((0.05, 0.50, 0.95)),
    ).unbind()
    return {
        f"{prefix}/mean": selected.mean(),
        f"{prefix}/median": median,
        f"{prefix}/median_weighted_sum": median * sample_count,
        f"{prefix}/p05": p05,
        f"{prefix}/p05_weighted_sum": p05 * sample_count,
        f"{prefix}/p95": p95,
        f"{prefix}/p95_weighted_sum": p95 * sample_count,
        f"{prefix}/sample_count": sample_count,
        f"{prefix}/sum": selected.sum(),
    }


def _episode_foot_participation_metrics(
    *,
    contact_step_counts: torch.Tensor,
    max_noncontact_step_counts: torch.Tensor,
    prefix: str,
    step_counts: torch.Tensor,
    step_dt: float,
    touchdown_counts: torch.Tensor,
    world_z_force_sums: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Summarize foot use and absolute world-Z force, not slope-normal load."""

    if step_counts.ndim != 1:
        raise RuntimeError(
            f"step_counts must be one-dimensional, got shape {tuple(step_counts.shape)}."
        )
    expected_shape = (step_counts.shape[0], len(GO2_FOOT_NAMES))
    named_matrices = {
        "contact_step_counts": contact_step_counts,
        "max_noncontact_step_counts": max_noncontact_step_counts,
        "touchdown_counts": touchdown_counts,
        "world_z_force_sums": world_z_force_sums,
    }
    for name, matrix in named_matrices.items():
        if matrix.shape != expected_shape:
            raise RuntimeError(
                f"{name} must have canonical FL/FR/RL/RR shape {expected_shape}, got {tuple(matrix.shape)}."
            )
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError("step_dt must be finite and positive.")

    minimum_steps = max(1, math.ceil(_MIN_GAIT_DIAGNOSTIC_DURATION_S / step_dt))
    valid = step_counts >= minimum_steps
    valid_count = valid.to(dtype=torch.float32).sum()
    safe_valid_count = valid_count.clamp_min(1.0)
    safe_step_counts = step_counts.clamp_min(1.0).unsqueeze(-1)

    contact_fractions = contact_step_counts / safe_step_counts
    total_contact_fraction = contact_fractions.sum(dim=-1)
    contact_balance_valid = valid & (total_contact_fraction > 0.0)

    total_world_z_force = world_z_force_sums.sum(dim=-1)
    world_z_load_valid = valid & (
        total_world_z_force > torch.finfo(world_z_force_sums.dtype).eps
    )
    world_z_load_fractions = world_z_force_sums / total_world_z_force.clamp_min(
        torch.finfo(world_z_force_sums.dtype).eps
    ).unsqueeze(-1)

    left_contact = contact_fractions[:, 0] + contact_fractions[:, 2]
    right_contact = contact_fractions[:, 1] + contact_fractions[:, 3]
    front_contact = contact_fractions[:, 0] + contact_fractions[:, 1]
    rear_contact = contact_fractions[:, 2] + contact_fractions[:, 3]
    abs_left_right_contact_imbalance = torch.abs(
        left_contact - right_contact
    ) / total_contact_fraction.clamp_min(torch.finfo(contact_fractions.dtype).eps)
    rear_contact_balance_valid = valid & (rear_contact > 0.0)
    abs_rear_left_right_contact_imbalance = torch.abs(
        contact_fractions[:, 2] - contact_fractions[:, 3]
    ) / rear_contact.clamp_min(torch.finfo(contact_fractions.dtype).eps)
    front_minus_rear_contact_imbalance = (
        front_contact - rear_contact
    ) / total_contact_fraction.clamp_min(torch.finfo(contact_fractions.dtype).eps)

    left_load = world_z_load_fractions[:, 0] + world_z_load_fractions[:, 2]
    right_load = world_z_load_fractions[:, 1] + world_z_load_fractions[:, 3]
    front_load = world_z_load_fractions[:, 0] + world_z_load_fractions[:, 1]
    rear_load = world_z_load_fractions[:, 2] + world_z_load_fractions[:, 3]
    abs_left_right_world_z_load_imbalance = torch.abs(left_load - right_load)
    rear_world_z_load_balance_valid = world_z_load_valid & (
        rear_load > torch.finfo(world_z_load_fractions.dtype).eps
    )
    abs_rear_left_right_world_z_load_imbalance = torch.abs(
        world_z_load_fractions[:, 2] - world_z_load_fractions[:, 3]
    ) / rear_load.clamp_min(torch.finfo(world_z_load_fractions.dtype).eps)
    front_minus_rear_world_z_load_imbalance = front_load - rear_load

    zero_touchdown = touchdown_counts <= 0.0
    never_contacted = contact_step_counts <= 0.0
    any_foot_never_contacted_count = (
        (never_contacted.any(dim=-1) & valid).to(dtype=torch.float32).sum()
    )
    any_foot_zero_touchdown_count = (
        (zero_touchdown.any(dim=-1) & valid).to(dtype=torch.float32).sum()
    )
    metrics = {
        f"{prefix}/any_foot_never_contacted_episode_count": any_foot_never_contacted_count,
        f"{prefix}/any_foot_never_contacted_fraction": any_foot_never_contacted_count
        / safe_valid_count,
        f"{prefix}/any_foot_zero_touchdown_episode_count": any_foot_zero_touchdown_count,
        f"{prefix}/any_foot_zero_touchdown_fraction": any_foot_zero_touchdown_count
        / safe_valid_count,
        f"{prefix}/contact_balance_valid_episode_count": contact_balance_valid.to(
            dtype=torch.float32
        ).sum(),
        f"{prefix}/rear_contact_balance_valid_episode_count": rear_contact_balance_valid.to(
            dtype=torch.float32
        ).sum(),
        f"{prefix}/rear_world_z_load_balance_valid_episode_count": rear_world_z_load_balance_valid.to(
            dtype=torch.float32
        ).sum(),
        f"{prefix}/valid_episode_count": valid_count,
        f"{prefix}/world_z_load_valid_episode_count": world_z_load_valid.to(
            dtype=torch.float32
        ).sum(),
    }
    for foot_index, foot_name in enumerate(GO2_FOOT_NAMES):
        foot_prefix = f"{prefix}/{foot_name}"
        contact_fraction_sum = (contact_fractions[:, foot_index] * valid).sum()
        world_z_load_fraction_sum = (
            world_z_load_fractions[:, foot_index] * world_z_load_valid
        ).sum()
        never_contacted_count = (
            (never_contacted[:, foot_index] & valid).to(dtype=torch.float32).sum()
        )
        zero_touchdown_count = (
            (zero_touchdown[:, foot_index] & valid).to(dtype=torch.float32).sum()
        )
        metrics[f"{foot_prefix}/contact_fraction_sum"] = contact_fraction_sum
        metrics[f"{foot_prefix}/mean_contact_fraction"] = (
            contact_fraction_sum / safe_valid_count
        )
        metrics[f"{foot_prefix}/mean_world_z_load_fraction"] = (
            world_z_load_fraction_sum
            / world_z_load_valid.to(dtype=torch.float32).sum().clamp_min(1.0)
        )
        metrics[f"{foot_prefix}/never_contacted_episode_count"] = never_contacted_count
        metrics[f"{foot_prefix}/never_contacted_fraction"] = (
            never_contacted_count / safe_valid_count
        )
        metrics[f"{foot_prefix}/world_z_load_fraction_sum"] = world_z_load_fraction_sum
        metrics[f"{foot_prefix}/zero_touchdown_episode_count"] = zero_touchdown_count
        metrics[f"{foot_prefix}/zero_touchdown_fraction"] = (
            zero_touchdown_count / safe_valid_count
        )

    distributions = {
        "abs_left_right_contact_imbalance": (
            abs_left_right_contact_imbalance,
            contact_balance_valid,
        ),
        "abs_left_right_world_z_load_imbalance": (
            abs_left_right_world_z_load_imbalance,
            world_z_load_valid,
        ),
        "abs_rear_left_right_contact_imbalance": (
            abs_rear_left_right_contact_imbalance,
            rear_contact_balance_valid,
        ),
        "abs_rear_left_right_world_z_load_imbalance": (
            abs_rear_left_right_world_z_load_imbalance,
            rear_world_z_load_balance_valid,
        ),
        "front_minus_rear_contact_imbalance": (
            front_minus_rear_contact_imbalance,
            contact_balance_valid,
        ),
        "front_minus_rear_world_z_load_imbalance": (
            front_minus_rear_world_z_load_imbalance,
            world_z_load_valid,
        ),
        "maximum_foot_noncontact_time_s": (
            max_noncontact_step_counts.max(dim=-1).values * step_dt,
            valid,
        ),
        "minimum_contact_fraction": (contact_fractions.min(dim=-1).values, valid),
        "minimum_world_z_load_fraction": (
            world_z_load_fractions.min(dim=-1).values,
            world_z_load_valid,
        ),
    }
    for metric_name, (values, metric_valid) in distributions.items():
        metrics.update(
            _distribution_metrics(
                values,
                metric_valid,
                prefix=f"{prefix}/{metric_name}",
            )
        )
    return metrics


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


def _metrics_to_python(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    """Transfer all scalar diagnostics to the logger with one device sync."""

    names = tuple(metrics)
    values = torch.stack(tuple(metrics.values())).detach().cpu().tolist()
    return dict(zip(names, values, strict=True))


def _require_resolved_names(
    actual_names: Sequence[str] | None,
    expected_names: Sequence[str],
    *,
    role: str,
) -> None:
    actual = tuple(actual_names or ())
    expected = tuple(expected_names)
    if actual != expected:
        raise RuntimeError(
            f"Training-diagnostic {role} must resolve as {expected}, got {actual}."
        )
