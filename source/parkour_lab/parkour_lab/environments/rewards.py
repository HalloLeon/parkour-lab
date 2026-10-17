"""JSON-safe descriptions of actual native reward-manager terms."""

import math

# Formulas are annotations only; the native manager owns reward arithmetic.
FORMULAS = {
    "track_lin_vel_xy_exp": "exp(-sum((command_xy - root_COM_velocity_b_xy)^2) / std^2)",
    "track_ang_vel_z_exp": "exp(-(command_yaw - root_angular_velocity_b_z)^2 / std^2)",
    "lin_vel_z_l2": "root_COM_velocity_b_z^2",
    "ang_vel_xy_l2": "sum(root_angular_velocity_b_xy^2)",
    "dof_torques_l2": "sum(applied_joint_torque^2)",
    "dof_acc_l2": "sum(joint_acceleration^2)",
    "action_rate_l2": "sum((delivered_native_action - previous_delivered_native_action)^2)",
    "feet_air_time": "sum((last_air_time_s - threshold) * first_contact) * (norm(command_xy) > 0.1)",
    "flat_orientation_l2": "sum(projected_gravity_b_xy^2)",
    "dof_pos_limits": "sum(max(soft_lower - joint_position, 0) + max(joint_position - soft_upper, 0))",
}

# Formula annotations describe these pinned functions, not arbitrary terms that
# happen to reuse their configuration names. Keep this lookup simulator-free.
_FORMULA_FUNCTIONS = {
    name: f"isaaclab.envs.mdp.rewards:{function}"
    for name, function in (
        ("track_lin_vel_xy_exp", "track_lin_vel_xy_exp"),
        ("track_ang_vel_z_exp", "track_ang_vel_z_exp"),
        ("lin_vel_z_l2", "lin_vel_z_l2"),
        ("ang_vel_xy_l2", "ang_vel_xy_l2"),
        ("dof_torques_l2", "joint_torques_l2"),
        ("dof_acc_l2", "joint_acc_l2"),
        ("action_rate_l2", "action_rate_l2"),
        ("flat_orientation_l2", "flat_orientation_l2"),
        ("dof_pos_limits", "joint_pos_limits"),
    )
}
_FORMULA_FUNCTIONS.update(
    feet_air_time="isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards:feet_air_time",
)


def _callable_name(function):
    """Name native functions and initialized stateful term instances alike."""
    owner = function if hasattr(function, "__qualname__") else type(function)
    return f"{owner.__module__}:{owner.__qualname__}"


def _parameter_value(value):
    """Copy native selectors and numeric parameters into JSON-compatible values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Reward metadata requires finite parameters")
        return value
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Reward parameter mapping keys must be strings")
        return {key: _parameter_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_parameter_value(item) for item in value)
    if isinstance(value, slice):
        return {"start": value.start, "stop": value.stop, "step": value.step}
    if callable(value):
        return _callable_name(value)
    if hasattr(value, "tolist"):
        return _parameter_value(value.tolist())
    if hasattr(value, "__dict__"):
        return {
            key: _parameter_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    raise TypeError(f"Unsupported reward parameter metadata: {type(value).__name__}")


def reward_recipe(env):
    """Describe the actual manager terms, including zero-weight disabled terms.

    Scene selectors use resolved body names. Joint quantities use radians / SI
    units; functions without explicit selectors operate on all robot joints.
    """
    terms = {}
    for name in env.reward_manager.active_terms:
        term = env.reward_manager.get_term_cfg(name)
        function = _callable_name(term.func)
        params = dict(term.params)
        if "sensor_cfg" in params:
            sensor = params["sensor_cfg"]
            params["sensor_cfg"] = {
                "name": sensor.name,
                "body_names": sensor.body_names,
            }
        terms[name] = {
            "weight": term.weight,
            "function": function,
            "formula": (
                FORMULAS.get(name) if function == _FORMULA_FUNCTIONS.get(name) else None
            ),
            "params": _parameter_value(params),
        }
    return {
        "terms": terms,
        "control_dt_s": env.step_dt,
        "arithmetic": "reward = sum(weight * unweighted_term * control_dt_s); no clipping",
        "state": "clean simulator state; not causal actor input",
    }
