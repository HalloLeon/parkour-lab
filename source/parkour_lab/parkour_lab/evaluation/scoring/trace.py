"""Validate the retained first-attempt prefix before computing any score."""

import numpy as np

from ..cases import PERIOD_S, STEPS, EvaluationCase


def _array(arrays, name, shape, *, boolean=False, finite=True):
    value = np.asarray(arrays.get(name))
    valid_type = value.dtype == np.bool_ if boolean else value.dtype.kind in "fiu"
    if value.shape != shape or not valid_type:
        raise ValueError(f"Invalid trace shape or dtype: {name}; expected {shape}")
    if finite and not np.isfinite(value).all():
        raise ValueError(f"Nonfinite trace data: {name}")
    return value if boolean else value.astype(np.float64, copy=False)


def _support(arrays, name, valid_name, shape):
    height = _array(arrays, name, shape, finite=False)
    valid = (
        _array(arrays, valid_name, shape, boolean=True)
        if valid_name in arrays
        else np.isfinite(height)
    )
    if np.isinf(height).any() or not np.isfinite(height[valid]).all():
        raise ValueError(f"Invalid supporting-surface heights: {name}")
    return height, valid


def validate_trace(case, arrays):
    """Tail rows after the first ending are excluded, including their rewards."""
    if not isinstance(case, EvaluationCase) or not hasattr(arrays, "get"):
        raise ValueError("Require a fixed case and named trace arrays")
    time = np.asarray(arrays.get("decision_time_s"))
    if time.ndim != 1:
        raise ValueError("Require one-dimensional recorded decision times")
    supplied = len(time)
    flags = {
        key: _array(arrays, key, (supplied,), boolean=True)
        for key in ("terminated", "truncated")
    }
    ended = np.flatnonzero(flags["terminated"] | flags["truncated"])
    count = int(ended[0]) + 1 if len(ended) else supplied
    if count > STEPS:
        raise ValueError("First attempt exceeds the fixed twenty-second horizon")
    trace = {}
    for name, shape in {
        "decision_time_s": (count,),
        "command": (count, 3),
        "root_position_w": (count, 3),
        "root_quaternion_w": (count, 4),
        "root_com_linear_velocity_b": (count, 3),
        "root_angular_velocity_b": (count, 3),
        "reward": (count,),
        "foot_position_w": (count, 4, 3),
        "foot_contact_forces_w": (count, 4, 3),
    }.items():
        value = np.asarray(arrays.get(name))
        if value.shape != (supplied, *shape[1:]):
            raise ValueError(f"Invalid supplied trace shape: {name}")
        trace[name] = _array({name: value[:count]}, name, shape)
    for name, value in flags.items():
        trace[name] = value[:count]
    for name in arrays:
        if name.startswith("ending_"):
            trace[name] = _array(arrays, name, (supplied,), boolean=True)[:count]
    for name, shape in {
        "initial_root_position_w": (3,),
        "initial_root_quaternion_w": (4,),
        "initial_foot_position_w": (4, 3),
    }.items():
        trace[name] = _array(arrays, name, shape)
    for name in ("root_quaternion_w", "initial_root_quaternion_w"):
        if not np.allclose(np.linalg.norm(trace[name], axis=-1), 1, atol=1e-4, rtol=0):
            raise ValueError("Trace root quaternions must be normalized wxyz")
    for stem, shape, source in (
        (
            "support",
            (count, 4),
            {
                name: np.asarray(arrays[name])[:count]
                for name in ("support_height_w", "support_valid")
                if name in arrays
            },
        ),
        ("initial_support", (4,), arrays),
    ):
        trace[f"{stem}_height_w"], trace[f"{stem}_valid"] = _support(
            source, f"{stem}_height_w", f"{stem}_valid", shape
        )
    if "env_origin_w" in arrays:
        trace["env_origin_w"] = _array(arrays, "env_origin_w", (3,))
    expected_time = np.arange(count) * PERIOD_S
    if not np.allclose(trace["decision_time_s"], expected_time, atol=1e-7, rtol=0):
        raise ValueError("Decision times must start at zero and advance by 0.02 s")
    commands = np.asarray(
        [case.command_at(float(time)) for time in expected_time]
    ).reshape(-1, 3)
    if not np.allclose(trace["command"], commands, atol=1e-7, rtol=0):
        raise ValueError("Applied commands differ from the fixed case")
    return trace
