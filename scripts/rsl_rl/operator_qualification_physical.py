"""Offline physical-evidence component for one native first-attempt trial.

This is not a simulator, a motor-contract verifier, or a safety certificate.
The native recorder samples joint positions/contact norms at control boundaries
and explicit actuator-model computed/applied torques at four physics substeps.
These are not measured joint torques or continuous collision/joint-limit proof.
An absent prospective physical contract blocks acceptance; observed failures are
still reported. No contact identities or acceptance thresholds are inferred.
"""

from __future__ import annotations

import math

import numpy as np


VERSION = "operator_qualification_physical_component_v1"
CONTRACT_VERSION = "operator_qualification_physical_contract_v1"
DESCRIPTIVE_CLIPPING_GAP_NM = 1e-5

_WIDTHS = {
    "position": 3,
    "pre_position": 3,
    "quaternion": 4,
    "pre_quaternion": 4,
    "linear_velocity_b": 3,
    "angular_velocity_b": 3,
    "angular_velocity_w": 3,
    "root_link_lin_vel_b": 3,
    "observation": 45,
    "action": 12,
    "joint_target": 12,
    "default_joint_position": 12,
    "joint_position_post": 12,
    "joint_velocity_post": 12,
}


def _names(value, label, count=None):
    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 1
        or value.dtype.kind != "U"
        or not len(value)
        or (count is not None and len(value) != count)
        or len(set(value)) != len(value)
        or any(not name for name in value)
    ):
        raise ValueError(f"Invalid recorded {label}")
    return value.tolist()


def _arrays(trace, expected_steps):
    if type(expected_steps) is not int or expected_steps <= 0:
        raise ValueError(
            "Expected steps must be an independently declared positive int"
        )
    index = trace.get("sample_index")
    if (
        not isinstance(index, np.ndarray)
        or index.ndim != 1
        or not np.issubdtype(index.dtype, np.integer)
        or len(index) > expected_steps
        or not np.array_equal(index, np.arange(len(index)))
    ):
        raise ValueError("Sample indices must be a contiguous prefix of the trial")
    count = len(index)
    names = _names(trace.get("contact_body_names"), "contact body names")
    if "joint_names" in trace:
        _names(trace["joint_names"], "joint names", 12)
    limits = trace.get("joint_pos_limits")
    if (
        not isinstance(limits, np.ndarray)
        or limits.shape != (12, 2)
        or not np.issubdtype(limits.dtype, np.floating)
        or not np.isfinite(limits).all()
        or np.any(limits[:, 0] >= limits[:, 1])
    ):
        raise ValueError("Missing or invalid measured native hard joint limits")
    shapes = {name: (count, width) for name, width in _WIDTHS.items()}
    shapes.update(
        computed_torque_substeps=(count, 4, 12),
        applied_torque_substeps=(count, 4, 12),
        contact_force_norm_n=(count, len(names)),
    )
    for name, shape in shapes.items():
        value = trace.get(name)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or not np.issubdtype(value.dtype, np.floating)
        ):
            raise ValueError(f"Missing or invalid native physical field: {name}")
    for name in (
        "terminated",
        "time_out",
        "procedural_workspace",
        "valid_first_attempt",
    ):
        value = trace.get(name)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (count,)
            or value.dtype != np.bool_
        ):
            raise ValueError(f"Invalid boolean mask: {name}")
    done = trace["terminated"] | trace["time_out"]
    if np.any(trace["procedural_workspace"] & ~trace["time_out"]):
        raise ValueError("Workspace censor flag is missing from native timeout history")
    if not np.array_equal(trace["valid_first_attempt"], np.cumsum(done) - done == 0):
        raise ValueError("First-attempt mask differs from native termination history")
    terminals = np.flatnonzero(done)
    terminal = int(terminals[0]) if len(terminals) else None
    return count, terminal, names, shapes


def _contract(trace, contract, names):
    if contract is None:
        return None
    fields = {
        "version",
        "joint_names",
        "joint_pos_limits_rad",
        "contact_body_names",
        "foot_names",
        "prohibited_contact_body_names",
        "prohibited_contact_force_threshold_n",
        "torque_semantics",
        "torque_clipping_gap_tolerance_nm",
        "maximum_torque_clipping_fraction_per_joint",
    }
    if type(contract) is not dict or set(contract) != fields:
        raise ValueError("Incomplete explicit physical contract")
    if contract["version"] != CONTRACT_VERSION:
        raise ValueError("Unknown physical contract version")
    for key, number in (("joint_names", 12), ("foot_names", 4)):
        value = contract[key]
        if (
            type(value) is not list
            or len(value) != number
            or any(type(item) is not str or not item for item in value)
            or len(set(value)) != number
        ):
            raise ValueError(f"Invalid physical contract {key}")
    actual_joints = _names(trace.get("joint_names"), "joint names", 12)
    if contract["joint_names"] != actual_joints:
        raise ValueError("Joint identity/order differs from the physical contract")
    if (
        type(contract["contact_body_names"]) is not list
        or contract["contact_body_names"] != names
    ):
        raise ValueError("Contact identity/order differs from the physical contract")
    limits = contract["joint_pos_limits_rad"]
    if (
        type(limits) is not list
        or len(limits) != 12
        or any(
            type(row) is not list
            or len(row) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
            for row in limits
        )
        or not np.array_equal(limits, trace["joint_pos_limits"])
    ):
        raise ValueError("Hard limits differ from the physical contract")
    feet = contract["foot_names"]
    prohibited = contract["prohibited_contact_body_names"]
    if (
        any(name not in names for name in feet)
        or type(prohibited) is not list
        or not prohibited
        or any(
            type(name) is not str or name not in names or name in feet
            for name in prohibited
        )
        or len(set(prohibited)) != len(prohibited)
    ):
        raise ValueError("Explicit distinct prohibited nonfoot identities required")
    # The rule is explicitly supplied, never inferred from a reward or termination.
    for key in (
        "prohibited_contact_force_threshold_n",
        "torque_clipping_gap_tolerance_nm",
        "maximum_torque_clipping_fraction_per_joint",
    ):
        value = contract[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid physical contract threshold: {key}")
    if contract["maximum_torque_clipping_fraction_per_joint"] > 1:
        raise ValueError("Clipping fraction must be between zero and one")
    if contract["torque_semantics"] != "explicit_actuator_model":
        raise ValueError(
            "Clipping acceptance requires explicit-actuator model evidence"
        )
    return contract


def score_trial_physical(trace, *, expected_steps, contract=None):
    """Audit one selected trial; missing prospective rules cannot produce a pass.

    Contract fields are validated against recorded identities/limits. The caller
    must independently establish their provenance and explicit-actuator semantics;
    this module cannot authenticate a caller-created dictionary. A full tape with
    even a final-row timeout fails. Terminal physical samples retain failure credit
    but post-reset housekeeping cannot repair or create a first-attempt failure.
    """
    count, terminal, names, shapes = _arrays(trace, expected_steps)
    policy = _contract(trace, contract, names)
    length = terminal + 1 if terminal is not None else count
    failures, blocked = [], []
    if count != expected_steps or terminal is not None:
        failures.append("trial_incomplete_or_interrupted")
    if terminal is not None:
        for key, failure in (
            ("terminated", "native_physical_termination"),
            ("time_out", "native_timeout"),
            ("procedural_workspace", "workspace_censored"),
        ):
            if bool(trace[key][terminal]):
                failures.append(failure)
    invalid = {
        name: np.flatnonzero(
            ~np.isfinite(trace[name][:length]).reshape(length, -1).all(axis=1)
        ).tolist()
        for name in shapes
        if length and not np.isfinite(trace[name][:length]).all()
    }
    if invalid:
        failures.append("nonfinite_required_physical_state")
    if length and np.any(trace["contact_force_norm_n"][:length] < 0):
        failures.append("negative_contact_force_norm")
    for name in ("pre_quaternion", "quaternion"):
        if (
            length
            and name not in invalid
            and not np.allclose(
                np.linalg.norm(trace[name][:length], axis=-1), 1, atol=1e-3, rtol=0
            )
        ):
            failures.append(f"invalid_{name}")
    joint = trace["joint_position_post"][:length].astype(np.float64)
    limits = trace["joint_pos_limits"].astype(np.float64)
    excess = None
    if length and "joint_position_post" not in invalid:
        with np.errstate(over="ignore", invalid="ignore"):
            excess = np.maximum(limits[:, 0] - joint, 0) + np.maximum(
                joint - limits[:, 1], 0
            )
        if np.any(excess > 0):
            failures.append("observed_hard_joint_limit_excess")
        if not np.isfinite(excess).all():
            failures.append("nonfinite_derived_hard_limit_excess")
            excess = None
    gap = None
    if (
        length
        and not {"computed_torque_substeps", "applied_torque_substeps"} & invalid.keys()
    ):
        with np.errstate(over="ignore", invalid="ignore"):
            gap = np.abs(
                trace["computed_torque_substeps"][:length].astype(np.float64)
                - trace["applied_torque_substeps"][:length].astype(np.float64)
            )
        if not np.isfinite(gap).all():
            failures.append("nonfinite_derived_torque_gap")
            gap = None
    tolerance = (
        policy["torque_clipping_gap_tolerance_nm"]
        if policy
        else DESCRIPTIVE_CLIPPING_GAP_NM
    )
    clipped = None if gap is None else (gap > tolerance).mean(axis=(0, 1))
    contact = trace["contact_force_norm_n"][:length]
    contact_max = (
        None
        if not length or "contact_force_norm_n" in invalid
        else contact.max(axis=0).astype(float).tolist()
    )
    prohibited_result = None
    undeclared_nonfoot = None
    if policy is None:
        blocked.extend(
            (
                "prohibited_nonfoot_contact_rule_not_declared",
                "torque_clipping_acceptance_rule_not_declared",
            )
        )
    else:
        undeclared_nonfoot = [
            name
            for name in names
            if name not in policy["foot_names"]
            and name not in policy["prohibited_contact_body_names"]
        ]
        if undeclared_nonfoot:
            blocked.append("some_nonfoot_contact_rules_not_declared")
        ids = [names.index(name) for name in policy["prohibited_contact_body_names"]]
        exceeded = (
            None
            if contact_max is None
            else [
                names[i]
                for i in ids
                if contact_max[i] > policy["prohibited_contact_force_threshold_n"]
            ]
        )
        prohibited_result = {
            "body_names": policy["prohibited_contact_body_names"],
            "threshold_n": float(policy["prohibited_contact_force_threshold_n"]),
            "exceedance_comparison": "strictly_greater",
            "observed_exceeding_bodies": exceeded,
        }
        if exceeded:
            failures.append("observed_prohibited_nonfoot_contact")
        if clipped is not None and np.any(
            clipped > policy["maximum_torque_clipping_fraction_per_joint"]
        ):
            failures.append("torque_clipping_fraction_exceeded")
    passed = not failures and not blocked
    return {
        "version": VERSION,
        "status": (
            "COMPONENT_FAIL"
            if failures
            else "COMPONENT_BLOCKED" if blocked else "COMPONENT_PASS"
        ),
        "physical_passed": passed,
        "qualification_passed": False,
        "exit_allowed": False,
        "expected_steps": expected_steps,
        "recorded_steps": count,
        "first_attempt_steps_including_terminal": length,
        "first_terminal_step": terminal,
        "ignored_housekeeping_steps": count - length,
        "failures": failures,
        "blocked_requirements": blocked,
        "nonfinite_first_attempt_samples": invalid,
        "joint_names": (
            trace["joint_names"].tolist() if "joint_names" in trace else None
        ),
        "joint_identity_scope": (
            "recorded_joint_order"
            if "joint_names" in trace
            else "indices_only_no_joint_identity_inferred"
        ),
        "hard_joint_limits": {
            "source": "recorded_native_joint_pos_limits_not_soft_limits",
            "maximum_observed_excess_rad_per_joint": (
                None if excess is None else excess.max(axis=0).tolist()
            ),
            "scope": "post_control_step_only_no_substep_extrema",
        },
        "contacts": {
            "body_names": names,
            "maximum_observed_force_norm_n": contact_max,
            "prohibited_nonfoot_rule": prohibited_result,
            "undeclared_nonfoot_body_names": undeclared_nonfoot,
            "scope": "post_control_step_net_normal_force_norm_not_full_wrench_or_collision_history",
        },
        "torque_clipping": {
            "scope": "four_native_actuator_model_substeps_per_control_step_not_measured_joint_torque",
            "gap_tolerance_nm": float(tolerance),
            "tolerance_scope": (
                "explicit_acceptance_contract"
                if policy
                else "descriptive_only_not_an_acceptance_rule"
            ),
            "maximum_gap_nm_per_joint": (
                None if gap is None else gap.max(axis=(0, 1)).tolist()
            ),
            "clipped_fraction_per_joint": None if clipped is None else clipped.tolist(),
            "maximum_allowed_fraction_per_joint": (
                None
                if policy is None
                else float(policy["maximum_torque_clipping_fraction_per_joint"])
            ),
        },
        "unscored_requirements": [
            "independent_physical_contract_provenance_and_motor_identity",
            "joint_limit_and_contact_extrema_between_control_samples",
            "collision_pairs_and_net_force_cancellation",
            "terrain_support_and_transitions",
            "kinematics_and_command_source",
            "heldout_case_matrix_and_streamed_input",
            "parent_observed_process_exit",
        ],
        "scope": "Declared observable physical rules only; not generic physical safety or qualification acceptance.",
    }
