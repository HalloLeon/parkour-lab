# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-checkpoint identity and compact inference contracts.

Resolved environment and agent files archive the full training configuration.
This module records only checkpoint-facing semantics plus a terrain-domain hash.

Only checkpoint-facing semantics are frozen. Critic inputs, unused observation
groups, source code, and framework versions may change independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from parkour_lab.runtime_versions import REQUIRED_RUNTIME_VERSIONS

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
    from tensordict import TensorDict

# Increment when checkpoint-facing tensor semantics or model wiring change.
# Checkpoints from another interface version are intentionally incompatible.
TEACHER_INTERFACE_VERSION = 20

# Number of delivered deployable-state frames retained by the history wrapper.
DEPLOYABLE_HISTORY_LENGTH = 10

# TensorDict observation-group keys shared by Isaac Lab, RSL-RL, and the
# checkpoint manifest. Here, "policy" means observations supplied to the actor,
# not the policy network or the action it produces.
DEPLOYABLE_STATE_GROUP = "policy"  # Current hardware-available state and commands.
# Wrapper-derived state history used to estimate the dynamics latent.
ADAPTATION_HISTORY_GROUP = "adaptation_history"
# Active-waypoint travel direction supplied only to the privileged teacher.
ORACLE_TRAVEL_DIRECTION_GROUP = "oracle_travel_direction"
PRIVILEGED_DYNAMICS_GROUP = "dynamics"  # Randomized simulator physics properties.
PRIVILEGED_TERRAIN_GROUP = "terrain"  # Simulator ray-cast terrain geometry.

# These archived sections do not define the neural-network inference interface.
# Curriculum compatibility is checked separately; provenance is informational.
_NON_INFERENCE_SECTIONS = frozenset({"command_contract", "training_provenance"})

# Exact observation routes accepted by the teacher. RSL-RL concatenates each
# tuple from left to right, so changing the order changes checkpoint semantics.
TEACHER_OBSERVATION_GROUPS = (
    DEPLOYABLE_STATE_GROUP,
    ORACLE_TRAVEL_DIRECTION_GROUP,
    PRIVILEGED_TERRAIN_GROUP,
    PRIVILEGED_DYNAMICS_GROUP,
)
# Terrain-ablation route: retain the oracle direction and privileged dynamics,
# but omit ray-cast terrain input and its encoder.
TEACHER_NO_TERRAIN_OBSERVATION_GROUPS = (
    DEPLOYABLE_STATE_GROUP,
    ORACLE_TRAVEL_DIRECTION_GROUP,
    PRIVILEGED_DYNAMICS_GROUP,
)
# Whitelist of actor-group layouts for which the model and manifest are defined.
SUPPORTED_TEACHER_OBSERVATION_GROUPS = (
    TEACHER_OBSERVATION_GROUPS,
    TEACHER_NO_TERRAIN_OBSERVATION_GROUPS,
)

__all__ = [
    "ADAPTATION_HISTORY_GROUP",
    "DEPLOYABLE_HISTORY_LENGTH",
    "InterfaceMismatchError",
    "DEPLOYABLE_STATE_GROUP",
    "ORACLE_TRAVEL_DIRECTION_GROUP",
    "PRIVILEGED_DYNAMICS_GROUP",
    "PRIVILEGED_TERRAIN_GROUP",
    "TEACHER_INTERFACE_VERSION",
    "TEACHER_NO_TERRAIN_OBSERVATION_GROUPS",
    "TEACHER_OBSERVATION_GROUPS",
    "SUPPORTED_TEACHER_OBSERVATION_GROUPS",
    "TeacherCheckpoint",
    "assert_teacher_interface_matches",
    "build_teacher_interface",
    "interface_sha256",
    "load_teacher_checkpoint",
    "sha256_file",
    "terrain_curriculum_matches",
    "write_json",
]


class InterfaceMismatchError(RuntimeError):
    """Raised when a checkpoint would receive an incompatible actor interface."""


@dataclass(frozen=True)
class TeacherCheckpoint:
    """Identity and validated training interface of one teacher checkpoint.

    Fields are frozen, but the nested interface dictionary remains mutable.
    """

    checkpoint_path: str
    checkpoint_sha256: str
    teacher_interface: dict[str, object]
    teacher_interface_sha256: str


def assert_teacher_interface_matches(
    expected: dict[str, object], actual: dict[str, object], *, context: str
) -> None:
    """Raise when the checkpoint-facing subset of the interface changed."""

    _assert_matching(
        _inference_interface(expected),
        _inference_interface(actual),
        context=context,
        incompatibility="changed the frozen teacher interface",
    )


def build_teacher_interface(
    base_env: ManagerBasedRLEnv,
    observations: TensorDict,
    agent_cfg: RslRlBaseRunnerCfg,
) -> dict[str, object]:
    """Describe the checkpoint-facing teacher interface."""

    # RSL-RL concatenates these groups in order, so reordering equal-width
    # groups still changes the checkpoint interface.
    actor_groups: tuple[str, ...] = tuple(agent_cfg.obs_groups.get("policy", ()))
    if actor_groups not in SUPPORTED_TEACHER_OBSERVATION_GROUPS:
        raise InterfaceMismatchError(
            "The privileged teacher actor must use one of the supported observation routes "
            f"{[list(route) for route in SUPPORTED_TEACHER_OBSERVATION_GROUPS]}, "
            f"got {list(actor_groups)}."
        )
    uses_terrain = PRIVILEGED_TERRAIN_GROUP in actor_groups

    groups = _describe_observation_groups(base_env, observations, actor_groups)
    policy_group = next(
        group for group in groups if group["name"] == DEPLOYABLE_STATE_GROUP
    )
    history_dimension = _flat_dimension(observations[ADAPTATION_HISTORY_GROUP])
    expected_history_dimension = (
        int(policy_group["dimension"]) * DEPLOYABLE_HISTORY_LENGTH
    )
    if history_dimension != expected_history_dimension:
        raise InterfaceMismatchError(
            f"Derived adaptation history must have width {expected_history_dimension}, "
            f"got {history_dimension}."
        )
    history_group = {
        "name": ADAPTATION_HISTORY_GROUP,
        "dimension": history_dimension,
        "derived_from_group": DEPLOYABLE_STATE_GROUP,
        "history_length": DEPLOYABLE_HISTORY_LENGTH,
        "flattening_order": "frame_major_oldest_to_newest",
        "newest_frame": "exact_delivered_source_group",
    }
    action_manager = base_env.action_manager

    # The runtime descriptor contains the resolved joint order and transforms.
    action_descriptor = action_manager.get_term("joint_pos").IO_descriptor
    robot_identity = _describe_robot_asset(base_env.cfg.scene.robot)
    curriculum_cfg = base_env.cfg.parkour_curriculum
    intent_cfg = base_env.cfg.commands.intent
    success_params = base_env.cfg.terminations.success.params
    policy_cfg = agent_cfg.policy
    dynamics_recorder = base_env.event_manager.get_term_cfg(
        "record_privileged_dynamics"
    ).func
    dynamics_component_names = list(dynamics_recorder.component_names)
    terrain_generator_cfg = base_env.cfg.scene.ground.terrain_generator
    terrain_curriculum = {
        "demotion_progress_metric": "active_waypoint_cursor_fraction_v1",
        "demotion_progress_fraction": float(curriculum_cfg.demotion_progress_fraction),
        "route_envelope": {
            "metric": "finite_approved_route_polyline_cross_track_v1",
            "progress_half_width_m": float(curriculum_cfg.progress_route_half_width_m),
            "soft_half_width_m": float(curriculum_cfg.soft_route_half_width_m),
            "hard_half_width_m": float(curriculum_cfg.hard_route_half_width_m),
            "hard_violation": "non_timeout_failure_without_route_advancement",
        },
        "tile_size_m": _simple_value(terrain_generator_cfg.size),
        "matrix": curriculum_cfg.metadata(),
    }
    terrain_scan = None
    if uses_terrain:
        height_obs_cfg = base_env.cfg.observations.terrain.height_scan.params["obs_cfg"]
        scanner_cfg = base_env.cfg.scene.height_scanner
        terrain_scan = {
            "num_rays": int(height_obs_cfg.num_rays),
            "feature_order": ["normalized_heights", "validity_mask"],
            "vertical_offset_m": float(height_obs_cfg.vertical_offset),
            "metric_clip_m": float(height_obs_cfg.clip),
            "normalized_range": [-1.0, 1.0],
            "missing_height_value": 1.0,
            "validity_values": {"valid": 1.0, "missing": 0.0},
            "sensor_prim_path": scanner_cfg.prim_path,
            "sensor_offset_position_m": _simple_value(scanner_cfg.offset.pos),
            "ray_alignment": scanner_cfg.ray_alignment,
            "resolution_m": float(scanner_cfg.pattern_cfg.resolution),
            "size_m": _simple_value(scanner_cfg.pattern_cfg.size),
            "direction": _simple_value(scanner_cfg.pattern_cfg.direction),
            "flattening_order": scanner_cfg.pattern_cfg.ordering,
            "mesh_prim_paths": list(scanner_cfg.mesh_prim_paths),
            "max_distance_m": float(scanner_cfg.max_distance),
            "update_period_s": float(scanner_cfg.update_period),
        }

    # Import the tensor architecture only when building a live simulator
    # manifest, keeping checkpoint identity helpers usable without PyTorch.
    from .architecture import MotorInterfaceCfg
    from .teacher.model import PrivilegedTeacherModelCfg

    teacher_model_cfg = PrivilegedTeacherModelCfg(
        motor=MotorInterfaceCfg(
            state_dim=_flat_dimension(observations[DEPLOYABLE_STATE_GROUP]),
            travel_direction_dim=_flat_dimension(
                observations[ORACLE_TRAVEL_DIRECTION_GROUP]
            ),
            terrain_latent_dim=int(policy_cfg.terrain_latent_dim),
            action_dim=int(action_manager.total_action_dim),
            hidden_dims=tuple(policy_cfg.actor_hidden_dims),
        ),
        history_dim=_flat_dimension(observations[ADAPTATION_HISTORY_GROUP]),
        privileged_dynamics_dim=_flat_dimension(
            observations[PRIVILEGED_DYNAMICS_GROUP]
        ),
        terrain_scan_dim=(
            _flat_dimension(observations[PRIVILEGED_TERRAIN_GROUP])
            if uses_terrain
            else None
        ),
        dynamics_hidden_dims=tuple(policy_cfg.dynamics_encoder_hidden_dims),
        history_hidden_dims=tuple(policy_cfg.history_encoder_hidden_dims),
        scan_hidden_dims=tuple(policy_cfg.scan_encoder_hidden_dims),
    )
    teacher_model_cfg.validate()

    return {
        "interface_version": TEACHER_INTERFACE_VERSION,
        # A1 and Go2 expose compatible policy tensor widths and joint names but
        # must not share checkpoints. Freeze source-asset identity independently
        # of tensor shape; this identifies the path rather than hashing contents.
        "robot": robot_identity,
        # The command source is intentionally outside teacher inference
        # matching. Record only the boundary that is implemented today; future
        # joystick, corridor, and assistance logic owns a separate contract.
        "command_contract": {
            "schema_version": 3,
            "command_term": "intent",
            "stored_values": [
                "requested_travel_forward",
                "requested_travel_left",
                "preferred_speed_m_s",
                "preferred_yaw_rate_rad_s",
            ],
            "requested_travel_direction": {
                "source": "intent_command_columns_0_1",
                "frame": "robot_yaw_forward_left",
                "representation": "unit_xy",
            },
            "scripted_training_source": {
                "flat": "active_course_waypoint",
                "obstacle": (
                    "final_course_waypoint_until_terminal_then_active_terminal_guidance"
                ),
            },
            "ingress": {
                "stop_deadband_m_s": float(intent_cfg.stop_deadband_m_s),
                "max_external_speed_m_s": float(intent_cfg.max_external_speed_m_s),
                "yaw_rate_deadband_rad_s": float(intent_cfg.yaw_rate_deadband_rad_s),
                "max_external_yaw_rate_rad_s": float(
                    intent_cfg.max_external_yaw_rate_rad_s
                ),
                "terminal_slowdown_distance_m": float(
                    intent_cfg.terminal_slowdown_distance_m
                ),
                "terminal_min_approach_speed_m_s": float(
                    intent_cfg.terminal_min_approach_speed_m_s
                ),
                "zero_speed_direction": "preserve_last_valid",
                "invalid_or_stale": "exact_zero_speed_and_yaw_rate",
                "simultaneous_translation_and_yaw": "yaw_rate_forced_to_zero",
            },
        },
        # Preserve the information boundary even though PPO concatenates all groups.
        "information_contract": {
            "deployable_state_group": DEPLOYABLE_STATE_GROUP,
            "deployable_history_group": ADAPTATION_HISTORY_GROUP,
            "oracle_travel_direction_group": ORACLE_TRAVEL_DIRECTION_GROUP,
            "travel_direction_representation": "yaw_aligned_unit_xy",
            "translational_target_speed": {
                "state_term": "desired_speed",
                "raw_source": "intent.preferred_speed_m_s",
                "units": "m_s",
                "domain": "non_negative",
                "zero_semantics": (
                    "raw_stop_or_pivot_or_terminal_root_inside_reach_radius"
                ),
                "terminal_landing_semantics": (
                    "preferred_speed_tapered_to_minimum_approach_then_exact_zero_inside_reach_radius"
                ),
                "semantic_version": 2,
            },
            "preferred_yaw_rate": {
                "state_term": "desired_yaw_rate",
                "units": "rad_s",
                "domain": "signed",
                "axis": "gravity_aligned_yaw_z_flat_only_v1",
                "positive_direction": "yaw_left_counter_clockwise",
                "active_only_when_preferred_speed_is_zero": True,
                "zero_semantics": "no_body_yaw_request",
                "semantic_version": 1,
            },
            # Exact ordered routes are archived in the resolved environment.
            "oracle_travel_direction_source": {
                "kind": "active_course_waypoint",
                "default_root_reach_radius_m": float(
                    curriculum_cfg.waypoint_reach_radius_m
                ),
                "root_reach_radius_override_source": "course_waypoint.root_reach_radius",
                "control_waypoint_transition": "current_root_within_radius",
                "physical_waypoint_transition": "root_radius_and_named_support_contact",
                "terminal_direction": (
                    "non_reversing_inbound_segment_inside_reach_circle_direct_point_outside"
                ),
                "support_contact_threshold_n": float(
                    success_params["contact_threshold"]
                ),
                "terminal_support_load_threshold_n": float(
                    success_params["terminal_support_load_threshold_n"]
                ),
                "terminal_min_support_feet": int(
                    success_params["terminal_min_support_feet"]
                ),
                "terminal_stability_dwell_s": float(
                    success_params["terminal_stability_dwell_s"]
                ),
                "support_margin_m": float(success_params["support_margin"]),
                "support_plane_tolerance_m": float(
                    success_params["support_plane_tolerance"]
                ),
            },
            "privileged_dynamics_group": PRIVILEGED_DYNAMICS_GROUP,
            "privileged_terrain_group": (
                PRIVILEGED_TERRAIN_GROUP if uses_terrain else None
            ),
        },
        "actor": {
            "observation_groups": groups,
            "observation_normalization": bool(policy_cfg.actor_obs_normalization),
            "architecture": {
                "class_name": getattr(
                    policy_cfg, "class_name", type(policy_cfg).__name__
                ),
                "model": teacher_model_cfg.to_dict(),
            },
        },
        "adaptation": {
            "privileged_dynamics_components": dynamics_component_names,
            "deployable_history": history_group,
            "history_layout": "frame_major_flattened_oldest_to_newest",
        },
        "terrain_scan": terrain_scan,
        # ``env.yaml`` archives the full domain. Only its terrain identity is
        # needed here for curriculum restore and out-of-domain warnings.
        "training_provenance": {
            "schema_version": 1,
            "runtime_versions": dict(REQUIRED_RUNTIME_VERSIONS),
            "progress_route_half_width_m": float(
                curriculum_cfg.progress_route_half_width_m
            ),
            "terrain_curriculum_sha256": interface_sha256(terrain_curriculum),
        },
        "action": {
            "term_order": list(action_manager.active_terms),
            "term_dimensions": [
                int(dimension) for dimension in action_manager.action_term_dim
            ],
            "joint_names": list(action_descriptor.joint_names),
            "scale": _simple_value(action_descriptor.scale),
            "offset": _simple_value(action_descriptor.offset),
            "clip": _simple_value(action_descriptor.clip),
            "clip_semantics": action_descriptor.extras["clip_semantics"],
            "clip_source": action_descriptor.extras["clip_source"],
            "soft_joint_limit_margin_rad": float(
                action_descriptor.extras["soft_joint_limit_margin_rad"]
            ),
            "wrapper_clip": _simple_value(agent_cfg.clip_actions),
        },
        # Physics step and decimation determine integration and control rate.
        "timing": {
            "physics_dt_s": float(base_env.cfg.sim.dt),
            "decimation": int(base_env.cfg.decimation),
        },
    }


def interface_sha256(interface: Mapping[str, object]) -> str:
    """Return the SHA-256 identity of a compact interface manifest."""

    # Dict insertion order is irrelevant; list order remains significant.
    canonical_json = json.dumps(
        interface,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def load_teacher_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    *,
    checkpoint_sha256: str | None = None,
) -> TeacherCheckpoint:
    """Load one teacher checkpoint identity and its training interface."""

    resolved_checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(resolved_checkpoint_path):
        raise FileNotFoundError(
            f"Teacher checkpoint does not exist: {resolved_checkpoint_path}"
        )

    interface_path = os.path.join(
        os.path.dirname(resolved_checkpoint_path),
        "params",
        "teacher_interface.json",
    )
    if not os.path.isfile(interface_path):
        raise FileNotFoundError(
            "The teacher checkpoint has no training interface manifest. "
            f"Expected: {interface_path}."
        )

    with open(interface_path, encoding="utf-8") as interface_file:
        loaded_payload = json.load(interface_file)
    if not isinstance(loaded_payload, dict):
        raise ValueError(f"Teacher interface manifest is invalid: {interface_path}")
    payload = cast(dict[str, object], loaded_payload)

    interface = payload.get("teacher_interface")
    if not isinstance(interface, dict):
        raise ValueError(f"Teacher interface is missing or invalid: {interface_path}")
    teacher_interface = cast(dict[str, object], interface)
    if teacher_interface.get("interface_version") != TEACHER_INTERFACE_VERSION:
        raise ValueError(
            "Teacher interface uses an unsupported serialization version: "
            f"{interface_path}"
        )
    robot_identity = teacher_interface.get("robot")
    if not isinstance(robot_identity, dict) or any(
        not isinstance(robot_identity.get(field), str) or not robot_identity[field]
        for field in ("model", "asset_path")
    ):
        raise ValueError(
            f"Teacher interface robot identity is missing or invalid: {interface_path}"
        )
    teacher_interface_hash = interface_sha256(teacher_interface)
    if payload.get("teacher_interface_sha256") != teacher_interface_hash:
        raise ValueError(f"Teacher interface hash is invalid: {interface_path}")

    return TeacherCheckpoint(
        checkpoint_path=resolved_checkpoint_path,
        checkpoint_sha256=checkpoint_sha256 or sha256_file(resolved_checkpoint_path),
        teacher_interface=teacher_interface,
        teacher_interface_sha256=teacher_interface_hash,
    )


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return a stable SHA-256 identity for a file's contents."""

    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def terrain_curriculum_matches(
    expected: Mapping[str, object], actual: Mapping[str, object]
) -> bool:
    """Return whether two manifests identify the same training terrain."""

    expected_hash = _nested(
        expected, "training_provenance", "terrain_curriculum_sha256"
    )
    return isinstance(expected_hash, str) and expected_hash == _nested(
        actual, "training_provenance", "terrain_curriculum_sha256"
    )


def write_json(path: str | os.PathLike[str], value: object) -> None:
    """Write human-readable JSON and create its parent directory."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
        output_file.write("\n")


# Manifest-description helpers.


def _callable_name(value: Callable[..., object]) -> str:
    """Return a callable's qualified name without hashing its implementation."""

    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _describe_observation_groups(
    base_env: ManagerBasedRLEnv,
    observations: TensorDict,
    group_names: tuple[str, ...],
) -> list[dict[str, object]]:
    """Describe the resolved groups consumed by the teacher actor."""

    observation_manager = base_env.observation_manager
    groups: list[dict[str, object]] = []

    for group_name in group_names:
        if group_name not in observations:
            raise InterfaceMismatchError(
                f"The teacher routes missing observation group {group_name!r}."
            )

        group_cfg = getattr(base_env.cfg.observations, group_name)
        terms: list[dict[str, object]] = []

        # The manager supplies runtime term order and dimensions; the config
        # supplies each tensor slice's meaning and preprocessing.
        for term_name, term_shape in zip(
            observation_manager.active_terms[group_name],
            observation_manager.group_obs_term_dim[group_name],
            strict=True,
        ):
            term_cfg = getattr(group_cfg, term_name)
            terms.append(
                {
                    "name": term_name,
                    # Flattened history dimensions come from NumPy's
                    # ``prod`` in Isaac Lab and are therefore ``np.int64``.
                    # Normalize every resolved dimension at this boundary so
                    # the interface remains directly JSON serializable.
                    "shape": [int(dimension) for dimension in term_shape],
                    "function": _callable_name(term_cfg.func),
                    "simple_params": _simple_mapping(term_cfg.params),
                    "clip": _simple_value(term_cfg.clip),
                    "scale": _simple_value(term_cfg.scale),
                    "history_length": int(term_cfg.history_length),
                    "flatten_history_dim": bool(term_cfg.flatten_history_dim),
                }
            )

        groups.append(
            {
                "name": group_name,
                # Parallel-environment batch size is not part of the model.
                "dimension": _flat_dimension(observations[group_name]),
                "concatenate_terms": bool(
                    observation_manager.group_obs_concatenate[group_name]
                ),
                "history_length": _simple_value(group_cfg.history_length),
                "flatten_history_dim": bool(group_cfg.flatten_history_dim),
                "terms": terms,
            }
        )

    return groups


def _describe_robot_asset(robot_cfg: object) -> dict[str, str]:
    """Return the robot model and exact source-asset path identity."""

    spawn_cfg = getattr(robot_cfg, "spawn", None)
    asset_path: str | None = None
    for attribute in ("usd_path", "asset_path"):
        candidate = getattr(spawn_cfg, attribute, None)
        if isinstance(candidate, (str, os.PathLike)) and candidate:
            asset_path = os.fspath(candidate)
            break

    if asset_path is None:
        raise InterfaceMismatchError(
            "The teacher robot must expose a non-empty USD or URDF asset path."
        )

    # Official Unitree assets use model-specific filenames (``a1.usd``,
    # ``go2.usd``). Strip a possible URI query while preserving the exact path
    # separately for compatibility checks.
    model = Path(asset_path.split("?", 1)[0]).stem.lower()
    if not model:
        raise InterfaceMismatchError(
            f"Cannot derive the teacher robot model from asset path {asset_path!r}."
        )

    return {
        "model": model,
        "asset_path": asset_path,
    }


# Compatibility checks.


def _assert_matching(
    expected: dict[str, object],
    actual: dict[str, object],
    *,
    context: str,
    incompatibility: str,
) -> None:
    """Raise one consistently formatted interface mismatch."""

    if expected == actual:
        return
    changed = sorted(
        key
        for key in set(expected) | set(actual)
        if expected.get(key) != actual.get(key)
    )
    raise InterfaceMismatchError(
        f"{context} {incompatibility}; changed sections: {', '.join(changed)}."
    )


def _inference_interface(interface: dict[str, object]) -> dict[str, object]:
    """Return only tensor/semantic fields consumed by teacher inference."""

    return {
        key: value
        for key, value in interface.items()
        if key not in _NON_INFERENCE_SECTIONS
    }


def _nested(values: object, *keys: str) -> object:
    """Read one nested manifest value, returning ``None`` if malformed."""

    for key in keys:
        if not isinstance(values, Mapping):
            return None
        values = values.get(key)
    return values


# Simple-value serialization.


def _flat_dimension(tensor: torch.Tensor) -> int:
    """Return the flattened non-batch dimension of an observation tensor."""

    return math.prod(tensor.shape[1:])


def _scene_entity_selector(value: object) -> dict[str, object] | None:
    """Record entity selectors because they determine tensor meaning and order."""

    # Keep Isaac Lab optional when this module is used by simulator-free tools.
    if type(value).__name__ != "SceneEntityCfg":
        return None

    selector: dict[str, object] = {"name": getattr(value, "name")}
    for attribute in ("joint_names", "body_names", "preserve_order"):
        if hasattr(value, attribute):
            selector[attribute] = _simple_value(getattr(value, attribute))
    return selector


def _simple_mapping(values: object) -> dict[str, object]:
    """Keep simple parameters and explicit scene-entity selectors."""

    if not isinstance(values, Mapping):
        return {}
    result: dict[str, object] = {}
    for key, value in values.items():
        converted = _simple_value(value)
        if converted is None and value is not None:
            converted = _scene_entity_selector(value)
        if converted is not None or value is None:
            result[str(key)] = converted
    return result


def _simple_value(value: object) -> object:
    """Convert simple numeric configuration values to JSON without introspection."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        # ``None`` can mean JSON null or unsupported; inspect the source to tell.
        converted_items = [_simple_value(item) for item in value]
        return (
            converted_items
            if all(
                item is not None or source is None
                for item, source in zip(converted_items, value)
            )
            else None
        )
    if isinstance(value, dict):
        converted_mapping: dict[str, object] = {}
        for key, source in value.items():
            item = _simple_value(source)
            if item is None and source is not None:
                return None
            converted_mapping[str(key)] = item
        return converted_mapping

    # Convert tensor-like metadata without retaining device or gradient state.
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        value = cpu() if callable(cpu) else value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _simple_value(tolist())
    return None
