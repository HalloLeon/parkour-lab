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

_FOOT_PATTERN = re.compile(r"^(?P<end>[FR])(?P<side>[LR])_foot$")
_JOINT_PATTERN = re.compile(r"^(?P<end>[FR])(?P<side>[LR])_(?P<kind>hip|thigh|calf)_joint$")


@dataclass(frozen=True, slots=True)
class _IndexTransform:
    """Source indices and signs for one reflected feature vector."""

    permutation: tuple[int, ...]
    signs: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _TermSlice:
    """One observation term's slice in a concatenated group."""

    name: str
    start: int
    stop: int

    @property
    def width(self) -> int:
        """Return the flattened feature width of this term."""

        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class _ReflectionLayout:
    """Resolved semantic layout used by every augmentation call."""

    action: _IndexTransform
    dynamics: _IndexTransform | None
    foot_contacts: _IndexTransform | None
    groups: dict[str, tuple[_TermSlice, ...]]
    history_length: int | None
    joint_terms: dict[tuple[str, str], _IndexTransform]
    terrain: _IndexTransform | None


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

    base_env = getattr(env, "unwrapped", env)
    layout = _reflection_layout(base_env)

    if obs is not None and actions is not None and obs.batch_size[0] != actions.shape[0]:
        raise ValueError(
            "Observation and action batch sizes must match before symmetry "
            f"augmentation, got {obs.batch_size[0]} and {actions.shape[0]}."
        )

    obs_aug = _augment_observations(obs, layout) if obs is not None else None
    actions_aug = _augment_actions(actions, layout.action) if actions is not None else None
    return obs_aug, actions_aug


def _augment_actions(actions: torch.Tensor, transform: _IndexTransform) -> torch.Tensor:
    """Return original actions followed by their reflected copy."""

    if actions.ndim != 2 or actions.shape[-1] != len(transform.permutation):
        raise ValueError(
            "Actions must have shape [batch, resolved joint-action dimension] with width "
            f"{len(transform.permutation)}, got shape {tuple(actions.shape)}."
        )
    return torch.cat((actions, _apply_index_transform(actions, transform)), dim=0)


def _augment_observations(
    obs: TensorDict,
    layout: _ReflectionLayout,
) -> TensorDict:
    """Return original observations followed by their reflected copy."""

    if len(obs.batch_size) != 1:
        raise ValueError(f"Symmetry expects one TensorDict batch dimension, got {tuple(obs.batch_size)}.")

    unsupported_groups = set(obs.keys()) - set(layout.groups) - _UNCHANGED_GROUPS
    if unsupported_groups:
        raise ValueError(
            "Left-right symmetry has no declared transform for observation " f"groups {sorted(unsupported_groups)}."
        )

    batch_size = obs.batch_size[0]
    obs_aug = obs.repeat(2)
    for group_name in layout.groups:
        if group_name not in obs:
            continue
        reflected = _reflect_group(group_name, obs[group_name], layout)
        obs_aug[group_name][batch_size:] = reflected
    return obs_aug


def _reflect_group(
    group_name: str,
    values: torch.Tensor,
    layout: _ReflectionLayout,
) -> torch.Tensor:
    """Reflect one concatenated observation group."""

    terms = layout.groups[group_name]
    expected_width = terms[-1].stop if terms else 0
    if values.ndim != 2 or values.shape[-1] != expected_width:
        raise ValueError(
            f"Observation group {group_name!r} must end in width {expected_width}, " f"got shape {tuple(values.shape)}."
        )

    reflected = values.clone()
    for term in terms:
        segment = values[..., term.start : term.stop]
        transformed = _reflect_term(group_name, term, segment, layout)
        reflected[..., term.start : term.stop] = transformed
    return reflected


def _reflect_term(
    group_name: str,
    term: _TermSlice,
    values: torch.Tensor,
    layout: _ReflectionLayout,
) -> torch.Tensor:
    """Reflect one named observation term."""

    if group_name in (_POLICY_GROUP, _HISTORY_GROUP):
        return _reflect_deployable_term(group_name, term, values, layout)
    if group_name == _HEADING_GROUP:
        if term.name != "active_waypoint_direction_yaw_xy":
            raise ValueError(f"Unsupported heading term {term.name!r}.")
        return _apply_signs(values, (1.0, -1.0))
    if group_name == _TERRAIN_GROUP:
        if term.name not in ("height_scan", "height_scan_validity") or layout.terrain is None:
            raise ValueError(f"Unsupported terrain term {term.name!r}.")
        return _apply_index_transform(values, layout.terrain)
    if group_name == _DYNAMICS_GROUP:
        if term.name != "properties" or layout.dynamics is None:
            raise ValueError(f"Unsupported privileged-dynamics term {term.name!r}.")
        return _apply_index_transform(values, layout.dynamics)
    if group_name == _CRITIC_GROUP:
        return _reflect_critic_term(term, values, layout)
    raise ValueError(f"Unsupported observation group {group_name!r}.")


def _reflect_deployable_term(
    group_name: str,
    term: _TermSlice,
    values: torch.Tensor,
    layout: _ReflectionLayout,
) -> torch.Tensor:
    """Reflect one current or historical deployable-state term."""

    history_length = layout.history_length if group_name == _HISTORY_GROUP else None
    feature_values = values
    if history_length is not None:
        if term.width % history_length != 0:
            raise ValueError(
                f"Historical term {term.name!r} width {term.width} is not divisible "
                f"by history length {history_length}."
            )
        feature_values = values.reshape(*values.shape[:-1], history_length, term.width // history_length)

    if term.name == "base_ang_vel":
        transformed = _apply_signs(feature_values, (-1.0, 1.0, -1.0))
    elif term.name == "projected_gravity":
        transformed = _apply_signs(feature_values, (1.0, -1.0, 1.0))
    elif term.name == "desired_speed":
        transformed = feature_values
    elif term.name in ("joint_pos", "joint_vel", "last_action"):
        transformed = _apply_index_transform(
            feature_values,
            layout.joint_terms[(group_name, term.name)],
        )
    else:
        raise ValueError(f"Unsupported deployable-state term {term.name!r}.")

    return transformed.reshape_as(values)


def _reflect_critic_term(
    term: _TermSlice,
    values: torch.Tensor,
    layout: _ReflectionLayout,
) -> torch.Tensor:
    """Reflect one critic-only term."""

    if term.name == "base_lin_vel":
        return _apply_signs(values, (1.0, -1.0, 1.0))
    if term.name == "foot_contacts":
        if layout.foot_contacts is None:
            raise ValueError("Foot-contact reflection metadata is unavailable.")
        return _apply_index_transform(values, layout.foot_contacts)
    if term.name in ("active_waypoint_distance_xy", "base_clearance", "route_phase"):
        return values
    raise ValueError(f"Unsupported critic-only term {term.name!r}.")


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

    groups = {
        group_name: _group_term_slices(env, group_name)
        for group_name in (
            _HISTORY_GROUP,
            _CRITIC_GROUP,
            _DYNAMICS_GROUP,
            _HEADING_GROUP,
            _POLICY_GROUP,
            _TERRAIN_GROUP,
        )
        if group_name in env.observation_manager.active_terms
    }

    _validate_group_terms(
        groups,
        _POLICY_GROUP,
        ("base_ang_vel", "projected_gravity", "desired_speed", "joint_pos", "joint_vel", "last_action"),
    )
    _validate_group_terms(
        groups,
        _HISTORY_GROUP,
        ("base_ang_vel", "projected_gravity", "desired_speed", "joint_pos", "joint_vel", "last_action"),
    )
    _validate_group_terms(groups, _HEADING_GROUP, ("active_waypoint_direction_yaw_xy",))
    _validate_group_terms(groups, _DYNAMICS_GROUP, ("properties",))
    _validate_group_terms(
        groups,
        _CRITIC_GROUP,
        ("base_lin_vel", "base_clearance", "active_waypoint_distance_xy", "route_phase", "foot_contacts"),
    )
    if _TERRAIN_GROUP in groups:
        _validate_group_terms(groups, _TERRAIN_GROUP, ("height_scan", "height_scan_validity"))

    action = _action_transform(env)
    joint_terms: dict[tuple[str, str], _IndexTransform] = {}
    for group_name in (_POLICY_GROUP, _HISTORY_GROUP):
        for term_name in ("joint_pos", "joint_vel"):
            names = _observation_joint_names(env, group_name, term_name)
            joint_terms[(group_name, term_name)] = _joint_transform(names)
        joint_terms[(group_name, "last_action")] = action

    history_length = None
    if _HISTORY_GROUP in groups:
        history_cfg = getattr(env.cfg.observations, _HISTORY_GROUP)
        history_length = int(history_cfg.history_length)
        if history_length <= 0 or not history_cfg.flatten_history_dim:
            raise ValueError("Adaptation history must use a positive, flattened term-major history.")

    dynamics = _dynamics_transform(env) if _DYNAMICS_GROUP in groups else None
    foot_contacts = _foot_contact_transform(env) if _CRITIC_GROUP in groups else None
    terrain = _terrain_transform(env) if _TERRAIN_GROUP in groups else None
    if terrain is not None:
        for term in groups[_TERRAIN_GROUP]:
            if term.width != len(terrain.permutation):
                raise ValueError(
                    f"Terrain term {term.name!r} has width {term.width}, but the "
                    f"scanner resolves {len(terrain.permutation)} rays."
                )

    return _ReflectionLayout(
        action=action,
        dynamics=dynamics,
        foot_contacts=foot_contacts,
        groups=groups,
        history_length=history_length,
        joint_terms=joint_terms,
        terrain=terrain,
    )


def _group_term_slices(env: ManagerBasedRLEnv, group_name: str) -> tuple[_TermSlice, ...]:
    """Return concatenated slices from ObservationManager runtime metadata."""

    manager = env.observation_manager
    if not manager.group_obs_concatenate[group_name]:
        raise ValueError(f"Observation group {group_name!r} must concatenate its terms.")

    start = 0
    terms: list[_TermSlice] = []
    for term_name, term_shape in zip(
        manager.active_terms[group_name],
        manager.group_obs_term_dim[group_name],
        strict=True,
    ):
        width = math.prod(int(dimension) for dimension in term_shape)
        terms.append(_TermSlice(term_name, start, start + width))
        start += width
    return tuple(terms)


def _validate_group_terms(
    groups: dict[str, tuple[_TermSlice, ...]],
    group_name: str,
    expected_names: tuple[str, ...],
) -> None:
    """Require every symmetry-sensitive group to have known semantics."""

    if group_name not in groups:
        raise ValueError(f"Required symmetry observation group {group_name!r} is missing.")
    actual_names = tuple(term.name for term in groups[group_name])
    if set(actual_names) != set(expected_names) or len(actual_names) != len(expected_names):
        raise ValueError(
            f"Observation group {group_name!r} must contain exactly {list(expected_names)}, "
            f"got {list(actual_names)}."
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
    return _semantic_transform(body_names, _mirrored_foot_name)


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
    if permutation.unique().numel() != permutation.numel():
        raise ValueError("Terrain reflection does not define a one-to-one ray permutation.")
    indices = torch.arange(permutation.numel(), device=permutation.device)
    if not torch.equal(permutation.index_select(0, permutation), indices):
        raise ValueError("Terrain ray reflection must be an involution.")
    resolved = tuple(int(index) for index in permutation.detach().cpu().tolist())
    return _IndexTransform(resolved, (1.0,) * len(resolved))


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
            source_name = f"{prefix}:{_mirrored_joint_name(joint_name)}"
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

    return _semantic_transform(
        joint_names,
        _mirrored_joint_name,
        sign_for_name=lambda name: -1.0 if _joint_kind(name) == "hip" else 1.0,
    )


def _semantic_transform(
    names: tuple[str, ...],
    mirrored_name: Callable[[str], str],
    *,
    sign_for_name: Callable[[str], float] | None = None,
) -> _IndexTransform:
    """Build and validate a semantic name permutation."""

    indices = {name: index for index, name in enumerate(names)}
    if len(indices) != len(names):
        raise ValueError("Symmetry feature names must be unique.")

    permutation: list[int] = []
    signs: list[float] = []
    for name in names:
        counterpart = mirrored_name(name)
        try:
            permutation.append(indices[counterpart])
        except KeyError as error:
            raise ValueError(f"Feature {name!r} has no reflected counterpart {counterpart!r}.") from error
        signs.append(1.0 if sign_for_name is None else float(sign_for_name(name)))

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


def _mirrored_joint_name(name: str) -> str:
    """Return the sagittal counterpart of one Unitree leg-joint name."""

    match = _JOINT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported Unitree leg-joint name {name!r}.")
    side = "R" if match.group("side") == "L" else "L"
    return f"{match.group('end')}{side}_{match.group('kind')}_joint"


def _joint_kind(name: str) -> str:
    """Return the semantic kind encoded by one Unitree leg-joint name."""

    match = _JOINT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported Unitree leg-joint name {name!r}.")
    return match.group("kind")


def _mirrored_foot_name(name: str) -> str:
    """Return the sagittal counterpart of one Unitree foot-body name."""

    match = _FOOT_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported Unitree foot-body name {name!r}.")
    side = "R" if match.group("side") == "L" else "L"
    return f"{match.group('end')}{side}_foot"


def _selected_names(names: tuple[str, ...], indices: list[int] | slice) -> tuple[str, ...]:
    """Select semantic names using one resolved SceneEntityCfg index value."""

    if isinstance(indices, slice):
        return names[indices]
    return tuple(names[int(index)] for index in indices)


def _apply_index_transform(values: torch.Tensor, transform: _IndexTransform) -> torch.Tensor:
    """Apply a signed permutation along the final tensor dimension."""

    if values.shape[-1] != len(transform.permutation):
        raise ValueError(f"Signed permutation expects width {len(transform.permutation)}, got {values.shape[-1]}.")
    permutation = torch.tensor(transform.permutation, device=values.device, dtype=torch.long)
    signs = values.new_tensor(transform.signs)
    return values.index_select(-1, permutation) * signs


def _apply_signs(values: torch.Tensor, signs: tuple[float, ...]) -> torch.Tensor:
    """Apply fixed reflection signs along the final tensor dimension."""

    if values.shape[-1] != len(signs):
        raise ValueError(f"Reflection expects width {len(signs)}, got {values.shape[-1]}.")
    return values * values.new_tensor(signs)
