"""Lossless batch normalization of the archived native motor binding.

Only identical repeated actuator rows may be collapsed. No tolerances, gain
overrides, joint reordering or inference from configuration defaults. This is
the recorded simulator motor contract, not a complete robot/physics certificate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

VERSION = "native_operator_motor_contract_v1"
PARAMETERS = (
    "stiffness",
    "damping",
    "effort_limit",
    "velocity_limit",
    "effort_limit_sim",
    "velocity_limit_sim",
    "armature",
    "friction",
)


def _encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def binding_sha256(binding):
    return hashlib.sha256(_encoded(binding)).hexdigest()


def _numbers(row, size):
    if (
        type(row) is not list
        or len(row) != size
        or any(type(v) not in (float, int) or not math.isfinite(v) for v in row)
    ):
        raise ValueError("Invalid finite motor-parameter row")


def _collapse(binding):
    """Preserve every field except provably redundant batch rows."""
    try:
        if type(binding) is not dict or set(binding) != {
            "joint_names",
            "default_position_rad",
            "action",
            "actuators",
            "step_dt_s",
            "physics_dt_s",
            "decimation",
        }:
            raise ValueError("Unknown native motor binding schema")
        names = binding["joint_names"]
        if (
            type(names) is not list
            or len(names) != 12
            or any(type(n) is not str or not n for n in names)
            or len(set(names)) != 12
        ):
            raise ValueError("Require twelve distinct named motor joints")
        _numbers(binding["default_position_rad"], 12)
        if (
            type(binding["decimation"]) is not int
            or binding["decimation"] != 4
            or type(binding["step_dt_s"]) is not float
            or binding["step_dt_s"] != 0.02
            or type(binding["physics_dt_s"]) is not float
            or binding["physics_dt_s"] != 0.005
            or type(binding["action"]) is not dict
            or type(binding["actuators"]) is not dict
            or not binding["actuators"]
        ):
            raise ValueError(
                "Motor contract requires the native 50 Hz/200 Hz interface"
            )
        compact, count, covered = copy.deepcopy(binding), None, []
        for name, actuator in binding["actuators"].items():
            if (
                type(name) is not str
                or not name
                or type(actuator) is not dict
                or set(actuator)
                != {
                    "joint_names",
                    "configuration",
                    "resolved_parameters",
                }
                or type(actuator["configuration"]) is not dict
            ):
                raise ValueError("Unknown native actuator schema")
            joints = actuator["joint_names"]
            if (
                type(joints) is not list
                or not joints
                or any(j not in names for j in joints)
            ):
                raise ValueError("Unknown actuator joints")
            covered.extend(joints)
            params = actuator["resolved_parameters"]
            if type(params) is not dict or set(params) != set(PARAMETERS):
                raise ValueError("Incomplete resolved actuator parameters")
            for key, rows in params.items():
                if type(rows) is not list or not 1 <= len(rows) <= 5120:
                    raise ValueError("Invalid motor binding batch size")
                if count is None:
                    count = len(rows)
                if len(rows) != count:
                    raise ValueError("Motor parameter batch dimensions differ")
                first = _encoded(rows[0])
                for row in rows:
                    _numbers(row, len(joints))
                    if _encoded(row) != first:
                        raise ValueError("Nonuniform motor rows cannot be collapsed")
                compact["actuators"][name]["resolved_parameters"][key] = [
                    copy.deepcopy(rows[0])
                ]
        if sorted(covered) != sorted(names):
            raise ValueError("Actuators must cover every joint exactly once")
        _encoded(compact)
        return compact, count
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError("Malformed native motor binding") from error


def _source_hash(profile):
    if type(profile) is not str or not profile.startswith("native_motor_sha256:"):
        raise ValueError("Missing archived native motor identity")
    digest = profile.removeprefix("native_motor_sha256:")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Invalid archived native motor identity")
    return digest


def make_motor_contract(binding, actuator_profile):
    if binding_sha256(binding) != _source_hash(actuator_profile):
        raise ValueError("Source motor binding does not match the checkpoint")
    compact, count = _collapse(binding)
    return {"version": VERSION, "source_num_envs": count, "binding": compact}


def validate_motor_contract(contract, manifest):
    """Expand the compact receipt and verify the original archive hash exactly."""
    try:
        if type(contract) is not dict or set(contract) != {
            "version",
            "source_num_envs",
            "binding",
        }:
            raise ValueError("Invalid motor contract schema")
        count = contract["source_num_envs"]
        if (
            contract["version"] != VERSION
            or type(count) is not int
            or not 1 <= count <= 5120
        ):
            raise ValueError("Invalid motor contract version or source batch size")
        compact, rows = _collapse(contract["binding"])
        if rows != 1 or (
            compact["joint_names"] != manifest["joint_names"]
            or compact["default_position_rad"]
            != manifest["configuration"]["default_position_rad"]
            or compact["step_dt_s"] != manifest["period_s"]
        ):
            raise ValueError("Motor contract differs from the actor interface")
        expanded = copy.deepcopy(compact)
        for actuator in expanded["actuators"].values():
            for name, values in actuator["resolved_parameters"].items():
                actuator["resolved_parameters"][name] = values * count
        if binding_sha256(expanded) != _source_hash(manifest["actuator_profile"]):
            raise ValueError(
                "Compact motor receipt does not reproduce the archived identity"
            )
        return compact
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed motor contract or controller manifest") from error


def verify_runtime_motor(contract, manifest, runtime_binding):
    """Verify actual resolved motors without changing the actor's archived identity."""
    expected = validate_motor_contract(contract, manifest)
    actual, count = _collapse(runtime_binding)
    if _encoded(actual) != _encoded(expected):
        raise ValueError("Runtime motor contract differs from the exported actor")
    return {
        "version": VERSION,
        "source_motor_binding_sha256": _source_hash(manifest["actuator_profile"]),
        "runtime_motor_binding_sha256": binding_sha256(runtime_binding),
        "portable_motor_sha256": binding_sha256(
            {"version": VERSION, "binding": expected}
        ),
        "source_num_envs": contract["source_num_envs"],
        "runtime_num_envs": count,
        "scope": "exact archived motor fields; only identical repeated batch rows collapsed",
    }
