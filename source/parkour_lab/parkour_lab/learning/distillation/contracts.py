# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-checkpoint identity and inference-interface contracts.

Training records the observation, network, terrain, action, and timing metadata
that determines teacher inference. Playback and distillation rebuild and
compare that compact manifest before loading the checkpoint.

Only checkpoint-facing semantics are frozen. Critic inputs, unused observation
groups, source code, and framework versions may change independently.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
    from tensordict import TensorDict

TEACHER_INTERFACE_VERSION = 13

ADAPTATION_HISTORY_GROUP = "adaptation_history"
DEPLOYABLE_STATE_GROUP = "policy"
ORACLE_HEADING_GROUP = "heading_target"
PRIVILEGED_DYNAMICS_GROUP = "dynamics"
PRIVILEGED_TERRAIN_GROUP = "terrain"

# RSL-RL concatenates the deployable, oracle, and privileged inputs in this order.
TEACHER_OBSERVATION_GROUPS = (
    DEPLOYABLE_STATE_GROUP,
    ORACLE_HEADING_GROUP,
    PRIVILEGED_TERRAIN_GROUP,
    PRIVILEGED_DYNAMICS_GROUP,
)

__all__ = [
    "ADAPTATION_HISTORY_GROUP",
    "InterfaceMismatchError",
    "DEPLOYABLE_STATE_GROUP",
    "ORACLE_HEADING_GROUP",
    "PRIVILEGED_DYNAMICS_GROUP",
    "PRIVILEGED_TERRAIN_GROUP",
    "TEACHER_INTERFACE_VERSION",
    "TEACHER_OBSERVATION_GROUPS",
    "TeacherCheckpoint",
    "assert_teacher_interface_matches",
    "build_teacher_interface",
    "interface_sha256",
    "load_teacher_checkpoint",
    "sha256_file",
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

    def to_dict(self) -> dict[str, object]:
        """Return a detached, JSON-compatible representation."""

        return asdict(self)


def assert_teacher_interface_matches(expected: dict[str, object], actual: dict[str, object], *, context: str) -> None:
    """Raise a readable error when a checkpoint-facing interface changed."""

    differences = _find_differences(expected, actual)
    if not differences:
        return

    # Keep badly mismatched manifests readable while reporting the omitted count.
    shown = differences[:20]
    if len(differences) > len(shown):
        shown.append(f"... and {len(differences) - len(shown)} more differences.")

    detail = "\n".join(f"  - {difference}" for difference in shown)
    raise InterfaceMismatchError(f"{context} changed the frozen teacher interface:\n{detail}")


def build_teacher_interface(
    base_env: ManagerBasedRLEnv,
    observations: TensorDict,
    agent_cfg: RslRlBaseRunnerCfg,
) -> dict[str, object]:
    """Describe only the inputs and outputs that determine teacher inference."""

    # RSL-RL concatenates these groups in order, so reordering equal-width
    # groups still changes the checkpoint interface.
    actor_groups: tuple[str, ...] = tuple(agent_cfg.obs_groups.get("policy", ()))
    if actor_groups != TEACHER_OBSERVATION_GROUPS:
        raise InterfaceMismatchError(
            "The privileged teacher actor must use observation groups "
            f"{list(TEACHER_OBSERVATION_GROUPS)}, got {list(actor_groups)}."
        )

    groups = _describe_observation_groups(base_env, observations, actor_groups)
    history_group = _describe_observation_groups(
        base_env,
        observations,
        (ADAPTATION_HISTORY_GROUP,),
    )[0]
    action_manager = base_env.action_manager

    # The runtime descriptor contains the resolved joint order and transforms.
    action_descriptor = action_manager.get_term("joint_pos").IO_descriptor
    height_obs_cfg = base_env.cfg.observations.terrain.height_scan.params["obs_cfg"]
    scanner_cfg = base_env.cfg.scene.height_scanner
    curriculum_cfg = base_env.cfg.parkour_curriculum
    success_params = base_env.cfg.terminations.success.params
    terrain_generator_cfg = base_env.cfg.scene.ground.terrain_generator
    terrain_cfgs = terrain_generator_cfg.sub_terrains.values()
    ground_thicknesses = {float(terrain_cfg.ground_thickness) for terrain_cfg in terrain_cfgs}
    if len(ground_thicknesses) != 1:
        raise InterfaceMismatchError(
            f"Teacher sub-terrains must share one ground thickness; got {sorted(ground_thicknesses)}."
        )
    ground_thickness_m = ground_thicknesses.pop()
    policy_cfg = agent_cfg.policy
    dynamics_recorder = base_env.event_manager.get_term_cfg("record_privileged_dynamics").func
    dynamics_component_names = list(dynamics_recorder.component_names)

    # Import the tensor architecture only when building a live simulator
    # manifest, keeping checkpoint identity helpers usable without PyTorch.
    from .architecture import MotorInterfaceCfg
    from .teacher.model import PrivilegedTeacherModelCfg

    teacher_model_cfg = PrivilegedTeacherModelCfg(
        motor=MotorInterfaceCfg(
            state_dim=_flat_dimension(observations[DEPLOYABLE_STATE_GROUP]),
            heading_dim=_flat_dimension(observations[ORACLE_HEADING_GROUP]),
            terrain_latent_dim=int(policy_cfg.terrain_latent_dim),
            action_dim=int(action_manager.total_action_dim),
            hidden_dims=tuple(policy_cfg.actor_hidden_dims),
        ),
        history_dim=_flat_dimension(observations[ADAPTATION_HISTORY_GROUP]),
        privileged_dynamics_dim=_flat_dimension(observations[PRIVILEGED_DYNAMICS_GROUP]),
        terrain_scan_dim=_flat_dimension(observations[PRIVILEGED_TERRAIN_GROUP]),
        dynamics_hidden_dims=tuple(policy_cfg.dynamics_encoder_hidden_dims),
        history_hidden_dims=tuple(policy_cfg.history_encoder_hidden_dims),
        scan_hidden_dims=tuple(policy_cfg.scan_encoder_hidden_dims),
    )
    teacher_model_cfg.validate()

    return {
        "interface_version": TEACHER_INTERFACE_VERSION,
        # Preserve the information boundary even though PPO concatenates all groups.
        "information_contract": {
            "deployable_state_group": DEPLOYABLE_STATE_GROUP,
            "deployable_history_group": ADAPTATION_HISTORY_GROUP,
            "oracle_heading_group": ORACLE_HEADING_GROUP,
            "heading_representation": "yaw_aligned_unit_xy",
            # Ordered routes are already frozen by ``terrain_curriculum.matrix``.
            "oracle_heading_source": {
                "kind": "active_course_waypoint",
                "reach_threshold_m": float(curriculum_cfg.waypoint_reach_threshold),
                "control_waypoint_transition": "radius_or_true_route_plane_crossing",
                "physical_waypoint_transition": "radius_and_named_support_contact",
                "support_contact_threshold_n": float(success_params["contact_threshold"]),
                "support_margin_m": float(success_params["support_margin"]),
                "support_plane_tolerance_m": float(success_params["support_plane_tolerance"]),
                "final_transition": "radius_and_named_support_contact",
                "final_requires_no_trunk_contact": True,
                "final_requires_min_clearance": True,
                "final_max_projected_gravity_xy_norm": float(success_params["max_completion_tilt"]),
                "final_max_vertical_speed_m_s": float(success_params["max_completion_vertical_speed"]),
            },
            "privileged_dynamics_group": PRIVILEGED_DYNAMICS_GROUP,
            "privileged_terrain_group": PRIVILEGED_TERRAIN_GROUP,
        },
        "actor": {
            "observation_groups": groups,
            "observation_normalization": bool(policy_cfg.actor_obs_normalization),
            "architecture": {
                "class_name": getattr(policy_cfg, "class_name", type(policy_cfg).__name__),
                # These stable state-dictionary paths make the encoder and
                # directly transferable motor identifiable in RSL-RL checkpoints.
                "checkpoint_modules": {
                    "dynamics_encoder": "actor.dynamics_encoder",
                    "history_encoder": "actor.history_encoder",
                    "terrain_encoder": "actor.terrain_encoder",
                    "motor_actor": "actor.motor",
                },
                "model": teacher_model_cfg.to_dict(),
            },
        },
        "adaptation": {
            "privileged_dynamics_components": dynamics_component_names,
            "privileged_dynamics_dimension": _flat_dimension(observations[PRIVILEGED_DYNAMICS_GROUP]),
            "deployable_history": history_group,
            "history_layout": "term_major_flattened_oldest_to_newest",
            "history_length": int(base_env.cfg.observations.adaptation_history.history_length),
            "includes_previous_action": "last_action"
            in base_env.observation_manager.active_terms[ADAPTATION_HISTORY_GROUP],
        },
        "terrain_scan": {
            "num_rays": int(height_obs_cfg.num_rays),
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
        },
        # Support patches affect both collision geometry and privileged ray values.
        "terrain_curriculum": {
            "tile_size_m": _simple_value(terrain_generator_cfg.size),
            "ground_thickness_m": ground_thickness_m,
            "matrix": curriculum_cfg.metadata(),
        },
        "action": {
            "term_order": list(action_manager.active_terms),
            "term_dimensions": list(action_manager.action_term_dim),
            "joint_names": list(action_descriptor.joint_names),
            "scale": _simple_value(action_descriptor.scale),
            "offset": _simple_value(action_descriptor.offset),
            "clip": _simple_value(action_descriptor.clip),
            "wrapper_clip": _simple_value(agent_cfg.clip_actions),
        },
        # Physics step and decimation determine integration and control rate.
        "timing": {
            "physics_dt_s": float(base_env.cfg.sim.dt),
            "decimation": int(base_env.cfg.decimation),
        },
    }


def interface_sha256(interface: dict[str, object]) -> str:
    """Return the SHA-256 identity of a compact interface manifest."""

    # Dict insertion order is irrelevant; list order remains significant.
    canonical_json = json.dumps(interface, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def load_teacher_checkpoint(
    checkpoint_path: str | os.PathLike[str],
) -> TeacherCheckpoint:
    """Load one teacher checkpoint identity and its training interface."""

    resolved_checkpoint_path = os.path.abspath(os.path.expanduser(checkpoint_path))
    if not os.path.isfile(resolved_checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {resolved_checkpoint_path}")

    interface_path = os.path.join(
        os.path.dirname(resolved_checkpoint_path),
        "params",
        "teacher_interface.json",
    )
    if not os.path.isfile(interface_path):
        raise FileNotFoundError(
            "The teacher checkpoint has no training interface manifest. " f"Expected: {interface_path}."
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
        raise ValueError("Teacher interface uses an unsupported serialization version: " f"{interface_path}")
    teacher_interface_hash = interface_sha256(teacher_interface)
    if payload.get("teacher_interface_sha256") != teacher_interface_hash:
        raise ValueError(f"Teacher interface hash is invalid: {interface_path}")

    return TeacherCheckpoint(
        checkpoint_path=resolved_checkpoint_path,
        checkpoint_sha256=sha256_file(resolved_checkpoint_path),
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


def write_json(path: str | os.PathLike[str], value: object) -> None:
    """Write human-readable JSON and create its parent directory."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


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
            raise InterfaceMismatchError(f"The teacher routes missing observation group {group_name!r}.")

        group_cfg = getattr(base_env.cfg.observations, group_name)
        terms: list[dict[str, object]] = []

        # The manager supplies runtime term order and dimensions; the config
        # supplies each tensor slice's meaning and preprocessing.
        for term_name, term_shape in zip(
            observation_manager.active_terms[group_name],
            observation_manager.group_obs_term_dim[group_name],
        ):
            term_cfg = getattr(group_cfg, term_name)
            terms.append(
                {
                    "name": term_name,
                    "shape": list(term_shape),
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
                "concatenate_terms": bool(observation_manager.group_obs_concatenate[group_name]),
                "history_length": _simple_value(group_cfg.history_length),
                "flatten_history_dim": bool(group_cfg.flatten_history_dim),
                "terms": terms,
            }
        )

    return groups


def _find_differences(expected: object, actual: object, path: str = "interface") -> list[str]:
    """Return readable differences between nested JSON values."""

    if type(expected) is not type(actual):
        return [f"{path}: expected type {type(expected).__name__}, got {type(actual).__name__}"]
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child_path = f"{path}.{key}"
            if key not in expected:
                differences.append(f"{child_path}: unexpected value {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{child_path}: missing; expected {expected[key]!r}")
            else:
                differences.extend(_find_differences(expected[key], actual[key], child_path))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: expected length {len(expected)}, got {len(actual)}"]
        differences = []
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            differences.extend(_find_differences(expected_item, actual_item, f"{path}[{index}]"))
        return differences
    return [] if expected == actual else [f"{path}: expected {expected!r}, got {actual!r}"]


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
            if all(item is not None or source is None for item, source in zip(converted_items, value))
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
