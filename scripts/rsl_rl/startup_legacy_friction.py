"""Replay-only observation/intervention on the deprecated PhysX friction channel.

Isaac Lab's Isaac Sim >=5 path reads/writes the new friction triple. Its zero
readback does not independently establish the deprecated coefficient's value.
This diagnostic never changes the new model, actuator gains, scene, or state and
never steps physics. It is not a production fix or evidence of a simulator bug.
"""

from __future__ import annotations

import math
import re


_UNSUPPORTED = (AttributeError, NotImplementedError, RuntimeError)


def _joint_names(names, label):
    names = list(names)
    if (
        len(names) != 12
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != 12
    ):
        raise ValueError(f"{label} must contain exactly 12 unique joint names")
    return names


def _coefficient(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative coefficient")
    return float(value)


def _declared_friction(asset, action_names):
    """Resolve explicit actuator config values; never substitute USD defaults."""
    declared = {}
    actuators = getattr(asset, "actuators", None)
    if not isinstance(actuators, dict) or not actuators:
        raise ValueError("Runtime actuators are required to verify nominal friction")
    for group, actuator in actuators.items():
        names = list(actuator.joint_names)
        if (
            not names
            or len(set(names)) != len(names)
            or any(name not in action_names for name in names)
        ):
            raise ValueError(f"Actuator {group}: invalid joint coverage")
        value = getattr(actuator.cfg, "friction", None)
        if isinstance(value, dict):
            if not value or any(not isinstance(key, str) for key in value):
                raise ValueError(f"Actuator {group}: malformed friction mapping")
            values = {
                key: _coefficient(item, f"Actuator {group}.friction[{key}]")
                for key, item in value.items()
            }
            try:
                matches = {
                    name: [
                        item for key, item in values.items() if re.fullmatch(key, name)
                    ]
                    for name in names
                }
            except re.error as error:
                raise ValueError(
                    f"Actuator {group}: invalid friction expression"
                ) from error
            if any(len(items) != 1 for items in matches.values()):
                raise ValueError(
                    f"Actuator {group}: friction mapping must cover each joint once"
                )
            resolved = {name: items[0] for name, items in matches.items()}
        else:
            resolved = {
                name: _coefficient(value, f"Actuator {group}.cfg.friction")
                for name in names
            }
        for name, item in resolved.items():
            if name in declared:
                raise ValueError(
                    f"Overlapping actuator friction declaration for {name}"
                )
            declared[name] = item
    if set(declared) != set(action_names):
        raise ValueError(
            "Actuator friction declarations do not cover all action joints"
        )
    return [declared[name] for name in action_names]


def _read(view, method, shape):
    import torch

    getter = getattr(view, method, None)
    if not callable(getter):
        raise AttributeError(f"{method} is unavailable")
    value = getter()
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or not torch.is_floating_point(value)
    ):
        raise ValueError(f"{method}: expected floating tensor with shape {shape}")
    # PhysX getter buffers can be reused by later calls. Own the snapshot.
    value = value.detach().to(device="cpu").clone()
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError(f"{method}: coefficients must be finite and nonnegative")
    return value


def configure_legacy_friction_probe(env, mode, *, action_replay=False):
    """Observe or conditionally zero only the legacy coefficient for one Go2.

    The caller must explicitly establish the recorded-action replay scope and
    pass ``action_replay=True``. ``zero`` is refused unless every new-model
    parameter and explicit nominal actuator friction is zero. A supported but
    malformed getter fails closed. A failed setter/readback raises, so its
    uncertain result cannot be treated as an executed causal experiment.

    Returned statuses describe diagnostic setup only, never robot acceptance.
    ``UNAVAILABLE`` and ``NO_INTERVENTION_NEEDED`` mean no setter was called.
    """
    if action_replay is not True:
        raise ValueError(
            "Legacy friction probe is restricted to explicit action replay"
        )
    if mode not in ("observe", "zero"):
        raise ValueError("Legacy friction probe mode must be 'observe' or 'zero'")
    if type(env.num_envs) is not int or env.num_envs != 1:
        raise ValueError("Legacy friction probe requires exactly one environment")
    asset = env.scene["robot"]
    action = env.action_manager.get_term("joint_pos")
    if getattr(action, "_asset", asset) is not asset:
        raise ValueError("Action and robot must refer to the same articulation")
    raw_names = _joint_names(asset.joint_names, "Raw joint names")
    action_names = _joint_names(action._joint_names, "Action joint names")
    if set(raw_names) != set(action_names):
        raise ValueError("Action joints must be a permutation of all raw joints")
    joint_ids = [raw_names.index(name) for name in action_names]
    declared = _declared_friction(asset, action_names)
    result = {
        "kind": "startup_legacy_joint_friction",
        "schema_version": 1,
        "mode": mode,
        "status": "UNAVAILABLE",
        "action_joint_names": action_names,
        "raw_joint_names": raw_names,
        "legacy_before_action_order": None,
        "legacy_after_action_order": None,
        "new_params_before_action_order": None,
        "new_params_after_action_order": None,
        "declared_actuator_friction": declared,
        "setter_called": False,
        "reason": "",
        "limitations": [
            "Replay-only setup diagnostic; this is not policy acceptance or a production fix.",
            "Legacy readback is distinct from the static/dynamic/viscous friction triple.",
            "A behavioral difference would support a channel contribution, not establish a version-independent PhysX bug.",
        ],
    }
    view = asset.root_physx_view
    before = {}
    unavailable = []
    for name, method, shape in (
        ("legacy", "get_dof_friction_coefficients", (1, 12)),
        ("new_params", "get_dof_friction_properties", (1, 12, 3)),
    ):
        try:
            value = _read(view, method, shape)
        except _UNSUPPORTED as error:
            unavailable.append(f"{method}: {type(error).__name__}: {error}")
            continue
        before[name] = value
        result[f"{name}_before_action_order"] = value[0, joint_ids].tolist()
    if unavailable:
        result["reason"] = "; ".join(unavailable)
        return result
    if mode == "observe":
        # There was no setter; both labels intentionally describe this single
        # observational snapshot, not an independent post-intervention readback.
        for name, value in before.items():
            result[f"{name}_after_action_order"] = value[0, joint_ids].tolist()
        result["status"] = "OBSERVED"
        result["reason"] = (
            "Read-only: before/after labels use the same unmodified snapshot."
        )
        return result
    if not bool((before["legacy"] > 0).any()):
        for name, value in before.items():
            result[f"{name}_after_action_order"] = value[0, joint_ids].tolist()
        result["status"] = "NO_INTERVENTION_NEEDED"
        result["reason"] = "Legacy coefficients are already zero; no setter was called."
        return result
    if bool((before["new_params"] != 0).any()):
        raise ValueError(
            "Legacy-only zero probe requires every new friction parameter to be zero"
        )
    if any(value != 0 for value in declared):
        raise ValueError(
            "Legacy-only zero probe requires explicit nominal actuator friction zero"
        )
    setter = getattr(view, "set_dof_friction_coefficients", None)
    if not callable(setter):
        result["reason"] = (
            "set_dof_friction_coefficients is unavailable; no setter was called."
        )
        return result
    import torch

    zeros = torch.zeros_like(before["legacy"], device="cpu")
    indices = torch.tensor([0], dtype=torch.int32, device="cpu")
    try:
        setter(zeros, indices=indices)
    except Exception as error:
        raise RuntimeError(
            "Legacy friction setter failed; abort this replay"
        ) from error
    result["setter_called"] = True
    try:
        after_legacy = _read(view, "get_dof_friction_coefficients", (1, 12))
        after_new = _read(view, "get_dof_friction_properties", (1, 12, 3))
    except Exception as error:
        raise RuntimeError(
            "Legacy friction post-write readback failed; abort this replay"
        ) from error
    if bool((after_legacy != 0).any()):
        raise RuntimeError(
            "Legacy friction zero write did not take effect; abort this replay"
        )
    if not torch.equal(after_new, before["new_params"]):
        raise RuntimeError(
            "Legacy friction write changed new-model parameters; abort this replay"
        )
    result["legacy_after_action_order"] = after_legacy[0, joint_ids].tolist()
    result["new_params_after_action_order"] = after_new[0, joint_ids].tolist()
    result["status"] = "APPLIED"
    result["reason"] = (
        "Only legacy coefficients were zeroed; legacy and unchanged new-model readbacks verified."
    )
    return result
