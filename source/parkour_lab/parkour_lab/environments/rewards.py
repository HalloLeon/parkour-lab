"""Bounded Go2 reward configuration; native functions retain their arithmetic.

Importing the schema requires neither Torch nor Isaac Lab. Only posture is a
project-defined reward; all other functions come from the stock Go2 task.
"""

import math
from copy import deepcopy

# Default weight and unweighted formula. The manager applies weight * control dt.
TERMS = {
    "track_lin_vel_xy_exp": (
        1.5,
        "exp(-sum((command_xy - root_COM_velocity_b_xy)^2) / std^2)",
    ),
    "track_ang_vel_z_exp": (
        0.75,
        "exp(-(command_yaw - root_angular_velocity_b_z)^2 / std^2)",
    ),
    "lin_vel_z_l2": (-2.0, "root_COM_velocity_b_z^2"),
    "ang_vel_xy_l2": (-0.05, "sum(root_angular_velocity_b_xy^2)"),
    "dof_torques_l2": (-0.0002, "sum(applied_joint_torque^2)"),
    "dof_acc_l2": (-2.5e-7, "sum(joint_acceleration^2)"),
    "action_rate_l2": (
        -0.01,
        "sum((delivered_native_action - previous_delivered_native_action)^2)",
    ),
    "feet_air_time": (
        0.01,
        "sum((last_air_time_s - threshold) * first_contact) * (norm(command_xy) > 0.1)",
    ),
    "flat_orientation_l2": (-2.5, "sum(projected_gravity_b_xy^2)"),
    "dof_pos_limits": (
        -10.0,
        "sum(max(soft_lower - joint_position, 0) + max(joint_position - soft_upper, 0))",
    ),
    "joint_posture": (
        0.0,
        "norm(joint_position - default_position) * (stand_still_scale if command_vx_vy_wz == 0 and norm(root_COM_velocity_b_xy) <= velocity_threshold else 1)",
    ),
}

# Each numeric parameter has a default and inclusive lower bound. Tracking stds
# remain solely in TaskConfig's linear_tracking_std/angular_tracking_std fields.
PARAMETERS = {
    "feet_air_time": {"threshold": (0.5, 0.0)},
    "joint_posture": {
        "stand_still_scale": (5.0, 1.0),
        "velocity_threshold": (0.3, 0.0),
    },
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
    joint_posture=f"{__name__}:joint_posture",
)


def validate_rewards(value):
    """Validate explicit overrides, without filling omitted defaults or aliases."""
    if not isinstance(value, dict) or set(value) - TERMS.keys():
        raise ValueError("rewards must map known native term names to overrides")
    for name, override in value.items():
        if not isinstance(override, dict) or set(override) - {"weight", "params"}:
            raise ValueError(f"rewards.{name} allows only weight and params")
        params = override.get("params", {})
        allowed = PARAMETERS.get(name, {})
        if not isinstance(params, dict) or set(params) - allowed.keys():
            raise ValueError(f"Unknown rewards.{name} params; allowed: {list(allowed)}")
        for key, number in {
            **params,
            **({"weight": override["weight"]} if "weight" in override else {}),
        }.items():
            if type(number) not in (int, float) or not math.isfinite(number):
                raise ValueError(f"rewards.{name}.{key} must be a finite number")
            if key == "weight":
                # The optional posture term is a penalty despite its zero default.
                positive = TERMS[name][0] > 0
                if (positive and number < 0) or (not positive and number > 0):
                    raise ValueError(f"rewards.{name}.weight has the wrong sign")
            elif number < allowed[key][1]:
                raise ValueError(f"rewards.{name}.{key} must be >= {allowed[key][1]}")
    return deepcopy(value)


def joint_posture(env, stand_still_scale=5.0, velocity_threshold=0.3):
    """Unitree Go2-style unsquared joint deviation; pure yaw is not a stop.

    Uses clean simulator state for training reward, never as causal actor input.
    https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py
    """
    import torch

    data = env.scene["robot"].data
    command = env.command_manager.get_command("base_velocity")
    stopped = (command == 0).all(dim=1) & (
        data.root_lin_vel_b[:, :2].norm(dim=1) <= velocity_threshold
    )
    deviation = (data.joint_pos - data.default_joint_pos).norm(dim=1)
    return deviation * torch.where(stopped, stand_still_scale, 1.0)


def configure_rewards(cfg, task):
    """Apply only the known reward recipe; never alter motors, task or learner."""
    from isaaclab.managers import RewardTermCfg

    cfg.rewards.joint_posture = RewardTermCfg(func=joint_posture, weight=0.0)
    for name, (weight, _) in TERMS.items():
        term = getattr(cfg.rewards, name)
        override = task.rewards.get(name, {})
        term.weight = override.get("weight", weight)
        term.params.update(
            {key: default for key, (default, _) in PARAMETERS.get(name, {}).items()}
        )
        term.params.update(override.get("params", {}))
    cfg.rewards.track_lin_vel_xy_exp.params["std"] = task.linear_tracking_std
    cfg.rewards.track_ang_vel_z_exp.params["std"] = task.angular_tracking_std


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
                TERMS[name][1] if function == _FORMULA_FUNCTIONS.get(name) else None
            ),
            "params": _parameter_value(params),
        }
    return {
        "terms": terms,
        "control_dt_s": env.step_dt,
        "arithmetic": "reward = sum(weight * unweighted_term * control_dt_s); no clipping",
        "state": "clean simulator state; not causal actor input",
    }
