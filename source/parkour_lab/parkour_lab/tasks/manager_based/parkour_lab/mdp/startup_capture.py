"""Opt-in, device-local snapshots for single-environment startup diagnostics.

This is not a reward or a policy input. Samples own their storage so a terminal
post-physics sample survives the environment's automatic reset. Only raw scan
non-finites are sanitized, with explicit masks; invalid robot state is retained
so the report writer can reject it instead of silently hiding an instability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ._shared.go2 import GO2_FOOT_NAMES
from .commands import get_target_speed

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def validate_startup_capture(
    enabled: bool, *, capture_evaluation_step: bool, num_envs: int
) -> None:
    """Reject unsupported capture configurations without accessing simulation."""

    if enabled and not capture_evaluation_step:
        raise ValueError("Startup diagnostics require capture_evaluation_step=True.")
    if enabled and num_envs != 1:
        raise ValueError("Startup diagnostics require exactly one environment.")


def _action_context(env: ManagerBasedRLEnv):
    # IO_descriptor performs GPU-to-CPU copies in Isaac Lab; do not query it in
    # the reward loop. These are the resolved names used by JointPositionAction.
    action = env.action_manager.get_term("joint_pos")
    asset = env.scene["robot"]
    joint_names = list(action._joint_names)
    joint_ids = [asset.joint_names.index(name) for name in joint_names]
    return asset, action, joint_ids


def _finite_scan(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(values).all(dim=-1)
    return torch.where(valid.unsqueeze(-1), values, torch.zeros_like(values)), valid


def capture_startup_state(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    """Capture current robot, action and sensor buffers without CPU transfers.

    All joint vectors use resolved *action* joint order (metadata supplies the
    names), not canonical leg order. Root and foot local positions subtract the
    environment origin; they are not heights above support or body coordinates.
    Action-term raw values have already passed through its delay buffer, whereas
    ``environment_action`` is the manager input before that delay. Pre-limit
    targets reconstruct the affine transform of those delayed raw values.
    """

    if env.num_envs != 1:
        raise ValueError("Startup diagnostics require exactly one environment.")
    asset, action, joint_ids = _action_context(env)
    data = asset.data
    origins = env.scene.env_origins
    foot_ids = [asset.body_names.index(name) for name in GO2_FOOT_NAMES]
    feet_sensor = env.scene["feet_contact"]
    sensor_foot_ids = [feet_sensor.body_names.index(name) for name in GO2_FOOT_NAMES]
    affine_targets = action.raw_actions * action._scale + action._offset
    configured_targets = affine_targets
    if action.cfg.clip is not None:
        configured_targets = affine_targets.clamp(
            min=action._clip[..., 0], max=action._clip[..., 1]
        )
    safe_limits = action._safe_target_limits
    safe_targets = configured_targets.clamp(
        min=safe_limits[:, 0], max=safe_limits[:, 1]
    )

    values = {
        "environment_origin_w_m": origins,
        "root_position_w_m": data.root_pos_w,
        "root_position_env_m": data.root_pos_w - origins,
        "root_orientation_wxyz": data.root_quat_w,
        "linear_velocity_body_m_s": data.root_lin_vel_b,
        "linear_velocity_w_m_s": data.root_lin_vel_w,
        "angular_velocity_body_rad_s": data.root_ang_vel_b,
        "angular_velocity_w_rad_s": data.root_ang_vel_w,
        "projected_gravity_body": data.projected_gravity_b,
        "intent_command": env.command_manager.get_command("intent"),
        "target_speed_m_s": get_target_speed(env),
        "joint_position_rad": data.joint_pos[:, joint_ids],
        "joint_velocity_rad_s": data.joint_vel[:, joint_ids],
        "joint_default_position_rad": data.default_joint_pos[:, joint_ids],
        "joint_position_target_rad": data.joint_pos_target[:, joint_ids],
        "joint_soft_position_limits_rad": data.soft_joint_pos_limits[:, joint_ids],
        "joint_computed_torque_nm": data.computed_torque[:, joint_ids],
        "joint_applied_torque_nm": data.applied_torque[:, joint_ids],
        "environment_action": env.action_manager.action,
        "delayed_raw_action": action.raw_actions,
        "processed_joint_target_rad": action.processed_actions,
        "affine_joint_target_rad": affine_targets,
        "configured_clip_joint_target_rad": configured_targets,
        "reconstructed_safe_joint_target_rad": safe_targets,
        "configured_target_clipped": configured_targets != affine_targets,
        "safe_target_clipped": safe_targets != configured_targets,
        # Safe limits are a joint-by-2 table shared by all environments. Retain
        # the leading environment dimension used by every other snapshot field.
        "safe_joint_target_limits_rad": safe_limits.unsqueeze(0),
        "foot_position_w_m": data.body_pos_w[:, foot_ids],
        # body_lin_vel_w is COM velocity in Isaac Lab, whereas body_pos_w is
        # link-origin position. Use the matching link velocity for this trace.
        "foot_linear_velocity_w_m_s": data.body_link_lin_vel_w[:, foot_ids],
        "foot_position_env_m": data.body_pos_w[:, foot_ids] - origins.unsqueeze(1),
        "foot_force_w_n": feet_sensor.data.net_forces_w[:, sensor_foot_ids],
        "undesired_contact_force_w_n": env.scene["undesired_contact"].data.net_forces_w,
        "chassis_contact_force_w_n": env.scene["chassis_contact"].data.net_forces_w,
    }
    for name in ("height_scanner", "base_height_scanner"):
        scanner = env.scene[name]
        scan = scanner.data  # Update existing sensor buffers through its normal API.
        hits, valid = _finite_scan(scan.ray_hits_w)
        values[f"{name}_position_w_m"] = scan.pos_w
        values[f"{name}_orientation_wxyz"] = scan.quat_w
        values[f"{name}_ray_hits_w_m"] = hits
        values[f"{name}_ray_hit_valid"] = valid
        values[f"{name}_ray_hits_env_m"] = torch.where(
            valid.unsqueeze(-1), hits - origins.unsqueeze(1), torch.zeros_like(hits)
        )
        values[f"{name}_root_relative_height_m"] = torch.where(
            valid,
            data.root_pos_w[:, 2:3] - hits[..., 2],
            torch.zeros_like(hits[..., 2]),
        )
        values[f"{name}_ray_starts_pattern_m"] = scanner.ray_starts
        # Newer Isaac Lab exposes the exact world-space ray-start cache. Older
        # versions do not; omit it instead of inventing a reconstruction that
        # could silently disagree with ray alignment, offsets or drift.
        if hasattr(scanner, "_ray_starts_w"):
            starts, starts_valid = _finite_scan(scanner._ray_starts_w)
            values[f"{name}_ray_starts_w_m"] = starts
            values[f"{name}_ray_start_valid"] = starts_valid
    return {name: value.detach().clone() for name, value in values.items()}


def startup_capture_metadata(env: ManagerBasedRLEnv) -> dict:
    """Name every coordinate convention and vector ordering without GPU reads."""

    asset, action, _ = _action_context(env)
    del asset
    return {
        "joint_names": list(action._joint_names),
        "foot_names": list(GO2_FOOT_NAMES),
        "contact_body_names": {
            name: list(env.scene[name].body_names)
            for name in ("undesired_contact", "chassis_contact")
        },
        "intent_command_components": [
            "direction_yaw_x",
            "direction_yaw_y",
            "preferred_speed_m_s",
            "yaw_rate_rad_s",
        ],
        "frames": {
            "w": "world Cartesian axes",
            "env": "world axes with scene.env_origins subtracted; not ground clearance",
            "body": "robot root body frame",
            "wxyz": "quaternion scalar first",
            "root_relative_height_m": "root world z minus ray-hit world z; validity mask required",
            "foot_linear_velocity_w_m_s": "foot rigid-body origin velocity in world axes; not contact-point slip velocity",
            "ray_starts_pattern_m": "RayCaster.ray_starts including configured pattern offset, before world transform",
        },
        "action_semantics": {
            "joint_order": "All joint tensors use joint_names order, including clip flags and torques.",
            "environment_action": "ActionManager input, before action-term delay; may already be wrapper-clipped.",
            "delayed_raw_action": "JointPositionAction.raw_actions, after the action-term delay buffer.",
            "affine_joint_target_rad": "delayed_raw_action * action scale + offset, before either target clamp",
            "clip_order": "affine target -> optional configured target clip -> safe soft-limit target clip",
            "initial_sample": "Before the first action, processed/actual target buffers may not equal the reconstructed transform; both are recorded without applying an action.",
        },
        "scan_invalid_values": "A ray with any nonfinite hit component is zero-filled; use its ray_hit_valid mask.",
        "scanners": {
            name: {
                "ray_alignment": env.scene[name].cfg.ray_alignment,
                "exact_world_ray_starts_available": hasattr(
                    env.scene[name], "_ray_starts_w"
                ),
            }
            for name in ("height_scanner", "base_height_scanner")
        },
    }
