# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right data augmentation for the privileged parkour teacher."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]


_CACHE_ATTRIBUTE = "_parkour_lab_left_right_symmetry_layout"
_CRITIC_GROUP = "critic_privileged"
_DYNAMICS_GROUP = "dynamics"
_HEADING_GROUP = "heading_target"
_HISTORY_GROUP = "adaptation_history"
_POLICY_GROUP = "policy"
_TERRAIN_GROUP = "terrain"
_UNCHANGED_GROUPS = frozenset(("student_exteroception",))

_DEPLOYABLE_TERM_SPECS = {
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    "desired_speed": "identity",
    "joint_pos": "joint",
    "joint_vel": "joint",
    "last_action": "action",
}
_GROUP_TERM_SPECS = {
    _POLICY_GROUP: _DEPLOYABLE_TERM_SPECS,
    _HISTORY_GROUP: _DEPLOYABLE_TERM_SPECS,
    _HEADING_GROUP: {"active_waypoint_direction_yaw_xy": (1.0, -1.0)},
    _DYNAMICS_GROUP: {"properties": "dynamics"},
    _CRITIC_GROUP: {
        "base_lin_vel": (1.0, -1.0, 1.0),
        "base_clearance": "identity",
        "active_waypoint_distance_xy": "identity",
        "route_phase": "identity",
        "foot_contacts": "feet",
    },
    _TERRAIN_GROUP: {"height_scan": "terrain", "height_scan_validity": "terrain"},
}

_FOOT_PATTERN = re.compile(r"^(?P<end>[FR])(?P<side>[LR])_foot$")
_JOINT_PATTERN = re.compile(r"^(?P<end>[FR])(?P<side>[LR])_(?P<kind>hip|thigh|calf)_joint$")


@dataclass(frozen=True, slots=True)
class _IndexTransform:
    """Source indices and signs for one reflected feature vector."""

    permutation: tuple[int, ...]
    signs: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _ReflectionLayout:
    """Resolved semantic layout used by every augmentation call."""

    action: _IndexTransform
    groups: dict[str, _IndexTransform]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append one sagittal-plane reflection to observations and actions.

    The returned batch always keeps the original samples first and appends
    exactly one left-right-reflected copy. This is the callable contract used
    by Isaac Lab's :class:`RslRlSymmetryCfg` and RSL-RL's PPO implementation.

    Reflection metadata is resolved from the live environment once and cached
    on the unwrapped environment. Joint and foot mappings therefore follow
    semantic names rather than assuming a particular USD tensor order.
    """

    if obs is None and actions is None:
        return None, None
    if obs is not None and len(obs.batch_size) != 1:
        raise ValueError(f"Symmetry expects one TensorDict batch dimension, got {tuple(obs.batch_size)}.")
    if actions is not None and actions.ndim != 2:
        raise ValueError(f"Actions must have shape [batch, features], got {tuple(actions.shape)}.")

    if obs is not None and actions is not None and obs.batch_size[0] != actions.shape[0]:
        raise ValueError(
            "Observation and action batch sizes must match before symmetry "
            f"augmentation, got {obs.batch_size[0]} and {actions.shape[0]}."
        )

    base_env = getattr(env, "unwrapped", env)
    layout = _reflection_layout(base_env)
    obs_aug = _augment_observations(obs, layout) if obs is not None else None
    actions_aug = _augment_actions(actions, layout.action) if actions is not None else None
    return obs_aug, actions_aug


def _augment_actions(actions: torch.Tensor, transform: _IndexTransform) -> torch.Tensor:
    """Return original actions followed by their reflected copy."""

    if actions.shape[-1] != len(transform.permutation):
        raise ValueError(f"Actions must have width {len(transform.permutation)}, got shape {tuple(actions.shape)}.")
    return torch.cat((actions, _apply_index_transform(actions, transform)), dim=0)


def _augment_observations(
    obs: TensorDict,
    layout: _ReflectionLayout,
) -> TensorDict:
    """Return original observations followed by their reflected copy."""

    unsupported_groups = set(obs.keys()) - set(layout.groups) - _UNCHANGED_GROUPS
    if unsupported_groups:
        raise ValueError(
            f"Left-right symmetry has no declared transform for observation groups {sorted(unsupported_groups)}."
        )

    batch_size = obs.batch_size[0]
    obs_aug = obs.repeat(2)
    for group_name, transform in layout.groups.items():
        if group_name not in obs:
            continue
        values = obs[group_name]
        expected_width = len(transform.permutation)
        if values.ndim != 2 or values.shape[-1] != expected_width:
            raise ValueError(
                f"Observation group {group_name!r} must have shape [batch, {expected_width}], "
                f"got {tuple(values.shape)}."
            )
        obs_aug[group_name][batch_size:] = _apply_index_transform(values, transform)
    return obs_aug


def _reflection_layout(env: ManagerBasedRLEnv) -> _ReflectionLayout:
    """Return the cached reflection layout for one environment."""

    cached = getattr(env, _CACHE_ATTRIBUTE, None)
    if isinstance(cached, _ReflectionLayout):
        return cached

    layout = _build_reflection_layout(env)
    try:
        setattr(env, _CACHE_ATTRIBUTE, layout)
    except (AttributeError, TypeError):
        # Slot-restricted test doubles can still use the correct uncached path.
        pass
    return layout


def _build_reflection_layout(env: ManagerBasedRLEnv) -> _ReflectionLayout:
    """Resolve all tensor layouts and semantic reflection mappings."""

    active_groups = env.observation_manager.active_terms
    for group_name in _GROUP_TERM_SPECS:
        if group_name not in active_groups and group_name != _TERRAIN_GROUP:
            raise ValueError(f"Required symmetry observation group {group_name!r} is missing.")

    action = _action_transform(env)
    history_cfg = getattr(env.cfg.observations, _HISTORY_GROUP)
    history_length = int(history_cfg.history_length)
    if history_length <= 0 or not history_cfg.flatten_history_dim:
        raise ValueError("Adaptation history must use a positive, flattened term-major history.")

    named_transforms = {
        "action": action,
        "dynamics": _dynamics_transform(env),
        "feet": _foot_contact_transform(env),
    }
    if _TERRAIN_GROUP in active_groups:
        named_transforms["terrain"] = _terrain_transform(env)

    groups = {
        group_name: _group_transform(env, group_name, named_transforms, history_length)
        for group_name in _GROUP_TERM_SPECS
        if group_name in active_groups
    }
    return _ReflectionLayout(action=action, groups=groups)


def _group_transform(
    env: ManagerBasedRLEnv,
    group_name: str,
    named_transforms: dict[str, _IndexTransform],
    history_length: int,
) -> _IndexTransform:
    """Compile one validated observation group into a signed permutation."""

    manager = env.observation_manager
    if not manager.group_obs_concatenate[group_name]:
        raise ValueError(f"Observation group {group_name!r} must concatenate its terms.")
    term_specs = _GROUP_TERM_SPECS[group_name]
    actual_names = tuple(manager.active_terms[group_name])
    if set(actual_names) != set(term_specs) or len(actual_names) != len(term_specs):
        raise ValueError(
            f"Observation group {group_name!r} must contain exactly {list(term_specs)}, got {list(actual_names)}."
        )

    offset = 0
    permutation: list[int] = []
    signs: list[float] = []
    for term_name, term_shape in zip(
        actual_names,
        manager.group_obs_term_dim[group_name],
        strict=True,
    ):
        width = math.prod(int(dimension) for dimension in term_shape)
        transform = _term_transform(
            env,
            group_name,
            term_name,
            width,
            term_specs[term_name],
            named_transforms,
            history_length,
        )
        permutation.extend(offset + index for index in transform.permutation)
        signs.extend(transform.signs)
        offset += width
    transform = _IndexTransform(tuple(permutation), tuple(signs))
    _validate_involution(transform, f"observation group {group_name!r}")
    return transform


def _term_transform(
    env: ManagerBasedRLEnv,
    group_name: str,
    term_name: str,
    term_width: int,
    spec: str | tuple[float, ...],
    named_transforms: dict[str, _IndexTransform],
    history_length: int,
) -> _IndexTransform:
    """Resolve one term and expand it over flattened history when needed."""

    repetitions = history_length if group_name == _HISTORY_GROUP else 1
    if term_width % repetitions != 0:
        raise ValueError(
            f"Historical term {term_name!r} width {term_width} is not divisible by history length {repetitions}."
        )
    feature_width = term_width // repetitions
    if isinstance(spec, tuple):
        transform = _IndexTransform(tuple(range(len(spec))), spec)
    elif spec == "identity":
        transform = _IndexTransform(tuple(range(feature_width)), (1.0,) * feature_width)
    elif spec == "joint":
        transform = _joint_transform(_observation_joint_names(env, group_name, term_name))
    else:
        transform = named_transforms[spec]
    if len(transform.permutation) != feature_width:
        raise ValueError(
            f"Observation term {group_name!r}.{term_name!r} has feature width {feature_width}, "
            f"but its reflection resolves width {len(transform.permutation)}."
        )
    if repetitions == 1:
        return transform
    return _IndexTransform(
        tuple(
            repetition * feature_width + index for repetition in range(repetitions) for index in transform.permutation
        ),
        transform.signs * repetitions,
    )


def _action_transform(env: ManagerBasedRLEnv) -> _IndexTransform:
    """Resolve the single joint-position action's semantic reflection."""

    action_manager = env.action_manager
    if action_manager.active_terms != ["joint_pos"]:
        raise ValueError(
            "Left-right symmetry currently requires exactly the 'joint_pos' action term, "
            f"got {action_manager.active_terms}."
        )
    descriptor = action_manager.get_term("joint_pos").IO_descriptor
    return _joint_transform(tuple(descriptor.joint_names))


def _observation_joint_names(
    env: ManagerBasedRLEnv,
    group_name: str,
    term_name: str,
) -> tuple[str, ...]:
    """Resolve one joint observation's names in its actual tensor order."""

    term_cfg = getattr(getattr(env.cfg.observations, group_name), term_name)
    asset_cfg = term_cfg.params.get("asset_cfg")
    asset_name = getattr(asset_cfg, "name", "robot")
    asset = env.scene[asset_name]
    joint_ids = getattr(asset_cfg, "joint_ids", slice(None))
    return _selected_names(tuple(asset.joint_names), joint_ids)


def _foot_contact_transform(env: ManagerBasedRLEnv) -> _IndexTransform:
    """Resolve critic contact columns from the contact sensor's body order."""

    term_cfg = getattr(getattr(env.cfg.observations, _CRITIC_GROUP), "foot_contacts")
    sensor_cfg = term_cfg.params["sensor_cfg"]
    sensor = env.scene[sensor_cfg.name]
    body_names = _selected_names(tuple(sensor.body_names), sensor_cfg.body_ids)
    return _semantic_transform(body_names, _foot_reflection)


def _terrain_transform(env: ManagerBasedRLEnv) -> _IndexTransform:
    """Resolve lateral ray pairs from the scanner's actual local origins."""

    term_cfg = getattr(getattr(env.cfg.observations, _TERRAIN_GROUP), "height_scan")
    sensor_cfg = term_cfg.params["sensor_cfg"]
    ray_starts = env.scene[sensor_cfg.name].ray_starts
    if ray_starts.ndim == 3:
        ray_starts = ray_starts[0]
    if ray_starts.ndim != 2 or ray_starts.shape[-1] != 3:
        raise ValueError(f"Terrain ray origins must have shape [rays, 3], got {tuple(ray_starts.shape)}.")

    target_starts = ray_starts.clone()
    target_starts[:, 1] = -target_starts[:, 1]
    pairwise_error = (target_starts[:, None, :] - ray_starts[None, :, :]).abs().amax(dim=-1)
    minimum_error, permutation = pairwise_error.min(dim=-1)
    tolerance = max(1.0e-6, 100.0 * torch.finfo(ray_starts.dtype).eps)
    if bool(torch.any(minimum_error > tolerance).item()):
        raise ValueError("Terrain scanner rays are not closed under left-right reflection.")
    resolved = tuple(int(index) for index in permutation.detach().cpu().tolist())
    transform = _IndexTransform(resolved, (1.0,) * len(resolved))
    _validate_involution(transform, "terrain ray")
    return transform


def _dynamics_transform(env: ManagerBasedRLEnv) -> _IndexTransform:
    """Resolve privileged dynamics by their recorder component names."""

    recorder = env.event_manager.get_term_cfg("record_privileged_dynamics").func
    component_names = tuple(recorder.component_names)
    indices = {name: index for index, name in enumerate(component_names)}
    if len(indices) != len(component_names):
        raise ValueError("Privileged-dynamics component names must be unique.")

    invariant_names = {
        "base_mass_ratio_minus_one",
        "base_com_x",
        "base_com_z",
        "mean_static_friction",
        "mean_dynamic_friction",
        "mean_restitution",
    }
    joint_prefixes = {
        "joint_stiffness_ratio_minus_one",
        "joint_damping_ratio_minus_one",
    }
    permutation: list[int] = []
    signs: list[float] = []
    for name in component_names:
        if name == "base_com_y":
            source_name = name
            sign = -1.0
        elif name in invariant_names:
            source_name = name
            sign = 1.0
        elif ":" in name:
            prefix, joint_name = name.split(":", maxsplit=1)
            if prefix not in joint_prefixes:
                raise ValueError(f"Unsupported privileged-dynamics component {name!r}.")
            source_name = f"{prefix}:{_joint_reflection(joint_name)[0]}"
            sign = 1.0
        else:
            raise ValueError(f"Unsupported privileged-dynamics component {name!r}.")
        try:
            permutation.append(indices[source_name])
        except KeyError as error:
            raise ValueError(
                f"Privileged-dynamics component {name!r} has no reflected source {source_name!r}."
            ) from error
        signs.append(sign)
    transform = _IndexTransform(tuple(permutation), tuple(signs))
    _validate_involution(transform, "privileged dynamics")
    return transform


def _joint_transform(joint_names: tuple[str, ...]) -> _IndexTransform:
    """Return the name-derived transform for joint-like values."""

    return _semantic_transform(joint_names, _joint_reflection)


def _semantic_transform(
    names: tuple[str, ...],
    reflection_for_name: Callable[[str], tuple[str, float]],
) -> _IndexTransform:
    """Build and validate a semantic name permutation."""

    indices = {name: index for index, name in enumerate(names)}
    if len(indices) != len(names):
        raise ValueError("Symmetry feature names must be unique.")

    permutation: list[int] = []
    signs: list[float] = []
    for name in names:
        counterpart, sign = reflection_for_name(name)
        try:
            permutation.append(indices[counterpart])
        except KeyError as error:
            raise ValueError(f"Feature {name!r} has no reflected counterpart {counterpart!r}.") from error
        signs.append(sign)

    transform = _IndexTransform(tuple(permutation), tuple(signs))
    _validate_involution(transform, "semantic feature")
    return transform


def _validate_involution(transform: _IndexTransform, role: str) -> None:
    """Require applying an index/sign transform twice to be the identity."""

    size = len(transform.permutation)
    if len(transform.signs) != size or sorted(transform.permutation) != list(range(size)):
        raise ValueError(f"The {role} reflection must define a signed permutation.")
    for index, source_index in enumerate(transform.permutation):
        if transform.permutation[source_index] != index:
            raise ValueError(f"The {role} permutation must be an involution.")
        if transform.signs[index] * transform.signs[source_index] != 1.0:
            raise ValueError(f"The {role} signs must cancel after two reflections.")


def _joint_reflection(name: str) -> tuple[str, float]:
    """Return one Unitree joint's reflected name and sign."""

    match = _JOINT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported Unitree leg-joint name {name!r}.")
    side = "R" if match.group("side") == "L" else "L"
    kind = match.group("kind")
    return f"{match.group('end')}{side}_{kind}_joint", -1.0 if kind == "hip" else 1.0


def _foot_reflection(name: str) -> tuple[str, float]:
    """Return one Unitree foot's reflected name and sign."""

    match = _FOOT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported Unitree foot-body name {name!r}.")
    side = "R" if match.group("side") == "L" else "L"
    return f"{match.group('end')}{side}_foot", 1.0


def _selected_names(names: tuple[str, ...], indices: list[int] | slice) -> tuple[str, ...]:
    """Select semantic names using one resolved SceneEntityCfg index value."""

    if isinstance(indices, slice):
        return names[indices]
    return tuple(names[int(index)] for index in indices)


def _apply_index_transform(values: torch.Tensor, transform: _IndexTransform) -> torch.Tensor:
    """Apply a signed permutation along the final tensor dimension."""

    permutation = torch.tensor(transform.permutation, device=values.device, dtype=torch.long)
    signs = values.new_tensor(transform.signs)
    return values.index_select(-1, permutation) * signs
