# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Teacher-selection and inference-interface contracts for distillation.

This module connects teacher training, fixed evaluation, and online
distillation. It builds a compact JSON-compatible description of the inputs
and outputs that determine teacher inference: actor observation groups and
term order, terrain preprocessing, network settings, action interpretation,
and control timing. Training writes that description, fixed evaluation carries
it together with the selected checkpoint, and distillation reconstructs and
compares it before loading the frozen teacher.

Checkpoint and interface SHA-256 hashes identify the exact evaluated model and
detect files or metadata that changed afterwards. ``TeacherSelection`` retains
the evaluation evidence for the accepted checkpoint, while readable recursive
comparisons explain interface mismatches instead of reporting only a failed
hash comparison.

The contract intentionally covers only checkpoint-facing teacher semantics. It
does not freeze critic inputs, unused observation groups, implementation source
code, or framework versions, allowing unrelated extensions and
behavior-preserving refactors without invalidating a teacher.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
    from tensordict import TensorDict

TEACHER_INTERFACE_VERSION = 1
"""Serialization version of the compact teacher interface manifest."""

TEACHER_OBSERVATION_GROUPS = ("policy", "terrain")
"""Observation-group order consumed by the privileged teacher actor."""

__all__ = [
    "InterfaceMismatchError",
    "TEACHER_INTERFACE_VERSION",
    "TEACHER_OBSERVATION_GROUPS",
    "TeacherSelection",
    "assert_teacher_interface_matches",
    "build_teacher_interface",
    "interface_sha256",
    "load_teacher_selection",
    "sha256_file",
    "write_json",
]


class InterfaceMismatchError(RuntimeError):
    """Raised when a checkpoint would receive an incompatible actor interface."""


@dataclass(frozen=True)
class TeacherSelection:
    """Immutable handoff from fixed teacher evaluation to distillation.

    ``load_teacher_selection`` creates this record only after confirming that
    every evaluation completed, selected the same checkpoint contents, and
    used the same checkpoint-facing inference interface. Distillation then uses
    the record to load that exact teacher and retain the evidence explaining
    why it was selected.

    ``frozen=True`` prevents reassignment of the record's fields after
    validation. It does not recursively freeze nested dictionaries.
    """

    # Absolute path resolved from the evaluation report and used to load the
    # exact teacher checkpoint for distillation.
    checkpoint_path: str

    # SHA-256 of the checkpoint bytes. Unlike the path alone, this detects a
    # checkpoint file that was replaced or modified after evaluation.
    checkpoint_sha256: str

    # Absolute paths to all completed fixed-evaluation reports used as the
    # evidence for selecting this checkpoint.
    evaluation_metrics_paths: tuple[str, ...]

    # Curriculum level evaluated by each corresponding metrics report, kept in
    # the same order as ``evaluation_metrics_paths``.
    evaluated_levels: tuple[int, ...]

    # Aggregate results copied from each corresponding evaluation report, such
    # as success and failure rates, for provenance and later reporting.
    evaluation_summaries: tuple[dict[str, object], ...]

    # Compact description of the teacher actor's observation order and
    # preprocessing, terrain scan, action mapping, and control timing.
    teacher_interface: dict[str, object]

    # SHA-256 of the stable JSON representation of ``teacher_interface``, used
    # to verify that evaluation and distillation use the same interface.
    teacher_interface_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return all selection evidence as a JSON-compatible dictionary."""

        # ``asdict`` recursively copies the dataclass fields so callers can
        # serialize the result without modifying this record's attributes.
        return cast(dict[str, object], asdict(self))


def assert_teacher_interface_matches(
    expected: dict[str, object], actual: dict[str, object], *, context: str
) -> None:
    """Raise a readable error when a checkpoint-facing interface changed."""

    # Compare every nested dictionary value and list element while retaining
    # its path, such as ``interface.actor.observation_groups[0]``.
    differences = _differences(expected, actual)

    # An empty difference list means that the current runtime interface is
    # compatible with the interface recorded for the checkpoint.
    if not differences:
        return

    # Limit the displayed diagnostics so a severely incompatible interface
    # does not produce an unreadably large exception, while still reporting
    # how many additional differences were found.
    shown = differences[:20]
    detail = "\n".join(f"  - {difference}" for difference in shown)
    if len(differences) > len(shown):
        detail += f"\n  - ... and {len(differences) - len(shown)} more differences."

    # ``context`` identifies the operation performing the check, for example
    # teacher playback or distillation, in the final error message.
    raise InterfaceMismatchError(
        f"{context} changed the frozen teacher interface:\n{detail}"
    )


def build_teacher_interface(
    base_env: ManagerBasedRLEnv,
    observations: TensorDict,
    agent_cfg: RslRlBaseRunnerCfg,
) -> dict[str, object]:
    """Describe only the inputs and outputs that determine teacher inference."""

    # RSL-RL concatenates these groups to form the teacher actor input. Their
    # order is part of the checkpoint interface, even when the total dimension
    # would remain unchanged after an accidental reordering.
    actor_groups = tuple(agent_cfg.obs_groups.get("policy", ()))
    if actor_groups != TEACHER_OBSERVATION_GROUPS:
        raise InterfaceMismatchError(
            "The privileged teacher actor must use observation groups "
            f"{list(TEACHER_OBSERVATION_GROUPS)}, got {list(actor_groups)}."
        )

    # Use the instantiated manager so the manifest records the resolved term
    # order and dimensions actually supplied by the environment.
    observation_manager = base_env.observation_manager

    # Describe only groups consumed by the teacher actor. Each ``group_name``
    # is an observation-group key from the actor route, such as ``"policy"``
    # or ``"terrain"``. The same key identifies the group's runtime tensor in
    # ``observations`` and its declarative configuration in ``base_env.cfg``.
    # Additional critic, student, or diagnostic groups may evolve without
    # invalidating the teacher interface.
    groups: list[dict[str, object]] = []
    for group_name in actor_groups:
        # A configured route is unusable if the environment did not compute a
        # corresponding observation tensor.
        if group_name not in observations:
            raise InterfaceMismatchError(
                f"The teacher routes missing observation group {group_name!r}."
            )

        # ``base_env.cfg`` is the resolved ``ParkourLabEnvCfg`` retained by the
        # running environment. It contains the declarative observation terms
        # and preprocessing settings, while the manager contains their resolved
        # runtime order and dimensions.
        group_cfg = getattr(base_env.cfg.observations, group_name)
        terms: list[dict[str, object]] = []

        # ``active_terms[group_name]`` is the ordered list of enabled term
        # names, such as ``["joint_pos", "joint_vel", "last_action"]``.
        # ``group_obs_term_dim[group_name]`` contains the matching non-batch
        # output shape of each term. For example, a term producing a tensor of
        # shape ``[num_envs, 12]`` contributes the stored shape ``(12,)``.
        # Isaac Lab keeps both lists in the same order, so ``zip`` pairs every
        # term name with the shape of the tensor slice that it contributes.
        for term_name, term_shape in zip(
            observation_manager.active_terms[group_name],
            observation_manager.group_obs_term_dim[group_name],
        ):
            # Look up the declarative ``ObsTerm`` configuration associated with
            # this resolved runtime term name.
            term_cfg = getattr(group_cfg, term_name)

            # Store the meaning and shape of each tensor slice without hashing
            # function bodies or serializing complete framework objects.
            terms.append(
                {
                    "name": term_name,
                    "shape": list(term_shape),
                    "function": _callable_name(term_cfg.func),
                    "simple_params": _simple_mapping(term_cfg.params),
                    "clip": _simple_value(term_cfg.clip),
                    "scale": _simple_value(term_cfg.scale),
                }
            )

        # The batch size is intentionally excluded from ``dimension`` because
        # changing the number of parallel environments does not affect a model.
        groups.append(
            {
                "name": group_name,
                "dimension": _flat_dimension(observations[group_name]),
                "concatenate_terms": bool(
                    observation_manager.group_obs_concatenate[group_name]
                ),
                "enable_corruption": bool(group_cfg.enable_corruption),
                "terms": terms,
            }
        )

    # Resolve the remaining checkpoint-facing runtime objects and their
    # declarative settings from the environment configuration before
    # assembling the JSON-compatible manifest.
    action_manager = base_env.action_manager
    action_cfg = base_env.cfg.actions.joint_pos

    # The runtime action term resolves configured joint patterns into the exact
    # joint order and exposes the resulting scale, offset, and clipping metadata
    # through its IO descriptor. Record that resolved interface so a checkpoint
    # cannot be reused with a differently ordered or interpreted action tensor.
    action_descriptor = action_manager.get_term("joint_pos").IO_descriptor
    height_obs_cfg = base_env.cfg.observations.terrain.height_scan.params["obs_cfg"]
    scanner_cfg = base_env.cfg.scene.height_scanner
    policy_cfg = agent_cfg.policy

    return {
        "interface_version": TEACHER_INTERFACE_VERSION,
        # Record the actor input layout and preprocessing performed by RSL-RL.
        "actor": {
            "observation_groups": groups,
            "input_dimension": sum(
                _flat_dimension(observations[name]) for name in actor_groups
            ),
            "network": {
                "class_name": getattr(
                    policy_cfg, "class_name", type(policy_cfg).__name__
                ),
                "hidden_dimensions": list(policy_cfg.actor_hidden_dims),
                "activation": policy_cfg.activation,
                "observation_normalization": bool(policy_cfg.actor_obs_normalization),
            },
        },
        # Record the fixed geometric meaning of the privileged terrain tensor.
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
        # Record how network outputs map to ordered low-level joint commands.
        "action": {
            "dimension": int(action_manager.total_action_dim),
            "term_order": list(action_manager.active_terms),
            "term_dimensions": list(action_manager.action_term_dim),
            "joint_names": list(action_descriptor.joint_names),
            "scale": _simple_value(action_descriptor.scale),
            "offset": _simple_value(action_descriptor.offset),
            "clip": _simple_value(action_descriptor.clip),
            "use_default_offset": bool(action_cfg.use_default_offset),
            "preserve_order": bool(action_cfg.preserve_order),
            "wrapper_clip": _simple_value(agent_cfg.clip_actions),
        },
        # Record both simulation and control timing because changing either can
        # alter how the same action sequence affects the robot.
        "timing": {
            "physics_dt_s": float(base_env.cfg.sim.dt),
            "decimation": int(base_env.cfg.decimation),
            "control_dt_s": float(base_env.step_dt),
        },
    }


def interface_sha256(interface: dict[str, object]) -> str:
    """Return the SHA-256 identity of a compact interface manifest."""

    return hashlib.sha256(_stable_json(interface).encode("utf-8")).hexdigest()


def load_teacher_selection(
    evaluation_metrics_paths: Iterable[str], *, training_task: str
) -> TeacherSelection:
    """Load one evaluated teacher and verify its checkpoint identity."""

    metrics_paths = tuple(
        os.path.abspath(os.path.expanduser(path)) for path in evaluation_metrics_paths
    )
    if not metrics_paths:
        raise ValueError("At least one fixed-evaluation metrics path is required.")

    records: list[dict[str, object]] = []
    for metrics_path in metrics_paths:
        with open(metrics_path, encoding="utf-8") as metrics_file:
            record = json.load(metrics_file)
        if record.get("difficulty_level") is None:
            raise ValueError(
                f"Evaluation did not freeze a difficulty level: {metrics_path}"
            )
        if record.get("completed_episodes") != record.get("requested_episodes"):
            raise ValueError(
                f"Evaluation did not complete its requested episodes: {metrics_path}"
            )
        if _training_task_name(record.get("task")) != _training_task_name(
            training_task
        ):
            raise ValueError(
                f"Evaluation task {record.get('task')!r} is incompatible with training task {training_task!r}."
            )

        interface = record.get("teacher_interface")
        if not isinstance(interface, dict):
            raise ValueError(
                f"Evaluation lacks a teacher interface: {metrics_path}. Rerun fixed evaluation with this code."
            )
        if record.get("teacher_interface_sha256") != interface_sha256(interface):
            raise ValueError(f"Teacher interface hash is invalid in {metrics_path}.")
        records.append(record)

    first = records[0]
    checkpoint_path = os.path.abspath(cast(str, first["checkpoint"]))
    checkpoint_hash = cast(str, first["checkpoint_sha256"])
    teacher_interface = cast(dict[str, object], first["teacher_interface"])
    teacher_interface_hash = cast(str, first["teacher_interface_sha256"])

    for metrics_path, record in zip(metrics_paths[1:], records[1:]):
        if os.path.abspath(cast(str, record["checkpoint"])) != checkpoint_path:
            raise ValueError(
                f"Evaluation selects a different checkpoint path: {metrics_path}"
            )
        if record["checkpoint_sha256"] != checkpoint_hash:
            raise ValueError(
                f"Evaluation selects different checkpoint contents: {metrics_path}"
            )
        if record["teacher_interface_sha256"] != teacher_interface_hash:
            raise ValueError(
                f"Evaluation uses a different teacher interface: {metrics_path}"
            )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Selected teacher checkpoint does not exist: {checkpoint_path}"
        )
    actual_checkpoint_hash = sha256_file(checkpoint_path)
    if actual_checkpoint_hash != checkpoint_hash:
        raise ValueError(
            "Selected teacher checkpoint contents changed after evaluation: "
            f"expected {checkpoint_hash}, got {actual_checkpoint_hash}."
        )

    return TeacherSelection(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_hash,
        evaluation_metrics_paths=metrics_paths,
        evaluated_levels=tuple(cast(int, record["difficulty_level"]) for record in records),
        evaluation_summaries=tuple(
            cast(dict[str, object], record.get("summary", {})) for record in records
        ),
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


def _differences(expected: object, actual: object, path: str = "interface") -> list[str]:
    """Return readable differences between nested JSON values."""

    if type(expected) is not type(actual):
        return [
            f"{path}: expected type {type(expected).__name__}, got {type(actual).__name__}"
        ]
    if isinstance(expected, dict):
        expected_mapping = cast(dict[str, object], expected)
        actual_mapping = cast(dict[str, object], actual)
        differences: list[str] = []
        for key in sorted(set(expected_mapping) | set(actual_mapping)):
            child_path = f"{path}.{key}"
            if key not in expected_mapping:
                differences.append(
                    f"{child_path}: unexpected value {actual_mapping[key]!r}"
                )
            elif key not in actual_mapping:
                differences.append(
                    f"{child_path}: missing; expected {expected_mapping[key]!r}"
                )
            else:
                differences.extend(
                    _differences(
                        expected_mapping[key], actual_mapping[key], child_path
                    )
                )
        return differences
    if isinstance(expected, list):
        expected_list = cast(list[object], expected)
        actual_list = cast(list[object], actual)
        if len(expected_list) != len(actual_list):
            return [
                f"{path}: expected length {len(expected_list)}, got {len(actual_list)}"
            ]
        differences = []
        for index, (expected_item, actual_item) in enumerate(
            zip(expected_list, actual_list)
        ):
            differences.extend(
                _differences(expected_item, actual_item, f"{path}[{index}]")
            )
        return differences
    return (
        [] if expected == actual else [f"{path}: expected {expected!r}, got {actual!r}"]
    )


def _flat_dimension(tensor: torch.Tensor) -> int:
    """Return the flattened non-batch dimension of an observation tensor."""

    return math.prod(tensor.shape[1:])


def _scene_entity_selector(value: object) -> dict[str, object] | None:
    """Record selectors that influence an observation's meaning and ordering.

    This describes the requested entity, joint or body names, and ordering
    behavior for interface comparison. Isaac Lab still resolves and applies
    the actual runtime indices and order.
    """

    # A ``SceneEntityCfg`` is not directly JSON-serializable, but it determines
    # which entity, joints, or bodies supply an observation. Those selections
    # define the meaning of fixed policy-input indices and may change without
    # changing the observation dimension. Record only the relevant selectors
    # so the interface check detects such a semantic or ordering change.
    if type(value).__name__ != "SceneEntityCfg":
        return None

    # Start with the scene entity's name, such as ``"robot"`` or
    # ``"height_scanner"``.
    selector: dict[str, object] = {"name": getattr(value, "name")}

    # ``attribute`` is successively each string below. It becomes both the
    # dictionary key and the name of the field read from ``SceneEntityCfg``.
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
        # JSON has arrays, represented here as Python lists, but no distinct
        # tuple type. Recursively convert every element into JSON-safe data.
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
            # A non-``None`` source that converts to ``None`` is unsupported,
            # whereas an original ``None`` is a valid JSON value. Reject the
            # whole mapping rather than recording silently incomplete metadata.
            if item is None and source is not None:
                return None
            converted_mapping[str(key)] = item
        return converted_mapping

    # For tensor-like values, discard gradient history and copy this value to
    # CPU memory before converting it to ordinary Python data. This affects
    # only the value being serialized, not the environment or policy device.
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
        cpu = getattr(value, "cpu", None)
        value = cpu() if callable(cpu) else value
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _simple_value(tolist())
    return None


def _stable_json(value: object) -> str:
    """Serialize an already compact manifest deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _training_task_name(task_name: str | None) -> str | None:
    """Normalize a fixed-play task ID to its corresponding training ID."""

    if task_name is None:
        return None
    return task_name.split(":")[-1].replace("-Play", "")
