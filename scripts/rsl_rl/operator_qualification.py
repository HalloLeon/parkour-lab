"""Strict, offline kinematic component for a future qualification protocol.

This module does not launch simulations or certify terrain/physical/operator
acceptance. The caller supplies an independently declared constant-command tape
for ONE first-attempt trial, not phases inferred from the measured commands.
Native arrays must be captured post-physics/pre-reset, with pre-action poses.
Historical development scorers are deliberately unchanged.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from numbers import Real

import numpy as np

try:
    from .operator_benchmark_core import DT, Phase, Thresholds, score_command_phase
except ImportError:
    from operator_benchmark_core import DT, Phase, Thresholds, score_command_phase


VERSION = "operator_qualification_kinematic_component_v1"


def _steps(seconds, dt, label):
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds <= 0
        or not math.isclose(seconds / dt, round(seconds / dt), abs_tol=1e-9, rel_tol=0)
    ):
        raise ValueError(f"{label} must be positive and control-step aligned")
    return round(seconds / dt)


def _plan(phases, dt, thresholds):
    if isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real) or dt != DT:
        raise ValueError("Kinematic component requires the native 0.02 s period")
    if not isinstance(thresholds, Thresholds):
        raise ValueError("Explicit Thresholds object required")
    for name, value in asdict(thresholds).items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"Invalid threshold: {name}")
    settling = _steps(thresholds.settling_s, dt, "Settling interval")
    block = _steps(thresholds.block_s, dt, "Tracking block")
    window = _steps(2.0, dt, "Stopping window")
    if block > settling:
        raise ValueError("Tracking block must end no later than acquisition")
    if not isinstance(phases, (list, tuple)) or not phases:
        raise ValueError("An independently declared nonempty phase tape is required")
    bounds, names, cursor = [], set(), 0
    for phase in phases:
        if (
            not isinstance(phase, Phase)
            or not isinstance(phase.name, str)
            or not phase.name
            or phase.name in names
        ):
            raise ValueError("Expected Phase declarations with unique nonempty names")
        command = np.asarray(phase.command)
        if (
            not isinstance(phase.command, tuple)
            or command.shape != (3,)
            or any(isinstance(value, (bool, np.bool_)) for value in phase.command)
            or not np.issubdtype(command.dtype, np.number)
            or np.issubdtype(command.dtype, np.complexfloating)
            or not np.isfinite(command).all()
        ):
            raise ValueError("Expected finite real body-twist commands")
        end = cursor + _steps(phase.duration_s, dt, "Phase duration")
        bounds.append((phase, cursor, end))
        names.add(phase.name)
        cursor = end
    return bounds, cursor, settling, window


def _trace_arrays(trace, expected_steps):
    index = trace.get("sample_index")
    if (
        not isinstance(index, np.ndarray)
        or index.ndim != 1
        or not np.issubdtype(index.dtype, np.integer)
        or len(index) > expected_steps
        or not np.array_equal(index, np.arange(len(index)))
    ):
        raise ValueError("Sample indices must be a contiguous prefix of the tape")
    length = len(index)
    widths = {
        "command": 3,
        "position": 3,
        "pre_position": 3,
        "quaternion": 4,
        "pre_quaternion": 4,
        "linear_velocity_b": 3,
        "root_link_lin_vel_b": 3,
        "angular_velocity_b": 3,
    }
    for name, width in widths.items():
        array = trace.get(name)
        if (
            not isinstance(array, np.ndarray)
            or array.shape != (length, width)
            or not np.issubdtype(array.dtype, np.floating)
        ):
            raise ValueError(f"Invalid physical array: {name}")
    for name in ("terminated", "time_out", "valid_first_attempt"):
        array = trace.get(name)
        if (
            not isinstance(array, np.ndarray)
            or array.shape != (length,)
            or array.dtype != np.bool_
        ):
            raise ValueError(f"Invalid boolean mask: {name}")
    done = trace["terminated"] | trace["time_out"]
    # Native validity includes the terminal action, but it earns no success credit.
    valid = np.cumsum(done) - done == 0
    if not np.array_equal(trace["valid_first_attempt"], valid):
        raise ValueError("First-attempt mask differs from native termination history")
    return length, widths, np.flatnonzero(done)


def _heading(quaternion):
    w, x, y, z = quaternion.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _phase_score(trace, start, end, phase, reference, plane, dt, th, settling, window):
    position = np.vstack((trace["pre_position"][start], trace["position"][start:end]))
    heading = np.unwrap(
        np.r_[
            _heading(trace["pre_quaternion"][start : start + 1]),
            _heading(trace["quaternion"][start:end]),
        ]
    )
    w, x, y, z = trace["quaternion"][start:end].T
    wx, wy, wz = trace["angular_velocity_b"][start:end].T
    world_yaw = (
        2 * (x * z - w * y) * wx
        + 2 * (y * z + w * x) * wy
        + (1 - 2 * (x * x + y * y)) * wz
    )
    field = "root_link_lin_vel_b" if reference == "root_link" else "linear_velocity_b"
    result = score_command_phase(
        phase.command,
        trace[field][start:end, :2],
        wz,
        world_yaw,
        position[:, :2],
        heading,
        dt=dt,
        thresholds=th,
    )
    if np.linalg.norm(phase.command[:2]) == 0:
        # Replace ONLY the legacy up-to-2s drift calculation in this new component.
        result["failures"] = [
            value
            for value in result["failures"]
            if value != "excessive stationary drift/braking distance"
        ]
        offsets = range(settling, len(position) - window)
        drifts = [
            float(
                np.linalg.norm(
                    position[offset : offset + window + 1, :2] - position[offset, :2],
                    axis=-1,
                ).max()
            )
            for offset in offsets
        ]
        result.update(
            full_settled_window_count=len(drifts),
            full_settled_window_duration_s=window * dt,
            first_settled_window_start_s=settling * dt if drifts else None,
            maximum_settled_drift_2s_m=max(drifts) if drifts else None,
        )
        if not drifts:
            result["failures"].append("missing complete settled 2 s window")
        if result["onset_excursion_m"] > th.stationary_onset_excursion_m or (
            drifts and max(drifts) > th.stationary_drift_2s_m
        ):
            result["failures"].append("excessive stationary drift/braking distance")
    if plane:
        # Include the pre-action onset attitude, not only subsequent recovery.
        q = np.vstack((trace["pre_quaternion"][start], trace["quaternion"][start:end]))
        w, x, y, z = q.T
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        result["maximum_abs_roll_rad"] = float(np.abs(roll).max())
        result["maximum_abs_pitch_rad"] = float(np.abs(pitch).max())
        if (
            max(result["maximum_abs_roll_rad"], result["maximum_abs_pitch_rad"])
            > th.flat_attitude_rad
        ):
            result["failures"].append("excessive plane attitude")
    result["kinematic_passed"] = not result["failures"]
    return result


def score_trial_kinematics(
    trace, phases, *, velocity_reference, plane, dt=DT, thresholds=None
):
    """Score a single declared trial without awarding physical/terrain acceptance.

    ``velocity_reference`` must explicitly select ``root_link`` or legacy ``com``.
    Both velocities are required in the trace. Missing tail data, terminal events
    (including final-step timeouts), nonfinite first-attempt states and hidden pose
    discontinuities fail the component. Post-terminal housekeeping cannot repair
    or poison earlier evidence. Structural/schema corruption raises ValueError.
    ``phases`` must come from a prospectively declared expected schedule, including
    actual resolved zero-command edges for source-loss tests.
    """
    if velocity_reference not in ("root_link", "com") or type(plane) is not bool:
        raise ValueError("Explicit velocity reference and boolean plane flag required")
    th = Thresholds() if thresholds is None else thresholds
    bounds, expected_steps, settling, window = _plan(phases, dt, th)
    length, widths, terminals = _trace_arrays(trace, expected_steps)
    terminal = int(terminals[0]) if len(terminals) else None
    eligible_end = length if terminal is None else terminal
    attempted_end = length if terminal is None else terminal + 1
    failures = []
    causes = []
    if terminal is not None:
        causes = [name for name in ("terminated", "time_out") if trace[name][terminal]]
        failures.append("first-attempt terminal event")
    if length < expected_steps:
        failures.append("incomplete recorded tape")
    bad = np.zeros(attempted_end, dtype=bool)
    for name in widths:
        invalid = ~np.isfinite(trace[name][:attempted_end]).all(axis=-1)
        if invalid.any():
            failures.append("nonfinite first-attempt field: " + name)
            bad |= invalid
    for name in ("pre_quaternion", "quaternion"):
        q = trace[name][:attempted_end]
        invalid = ~np.isclose(np.linalg.norm(q, axis=-1), 1, atol=1e-3, rtol=0)
        if invalid.any():
            failures.append("invalid first-attempt quaternion: " + name)
            bad |= invalid
    if bad.any():
        eligible_end = min(eligible_end, int(np.flatnonzero(bad)[0]))
    if eligible_end > 1:
        pre = trace["pre_position"][1:eligible_end]
        post = trace["position"][: eligible_end - 1]
        position_contiguous = np.isclose(pre, post, atol=1e-6, rtol=0).all(axis=-1)
        pre_q = trace["pre_quaternion"][1:eligible_end]
        post_q = trace["quaternion"][: eligible_end - 1]
        quaternion_contiguous = np.isclose(pre_q, post_q, atol=1e-6, rtol=0).all(
            axis=-1
        ) | np.isclose(pre_q, -post_q, atol=1e-6, rtol=0).all(axis=-1)
        discontinuous = ~(position_contiguous & quaternion_contiguous)
        if discontinuous.any():
            failures.append("unannounced first-attempt pose discontinuity")
            eligible_end = int(np.flatnonzero(discontinuous)[0]) + 1
    reports = []
    for phase, start, end in bounds:
        observed = max(0, min(eligible_end, end) - start)
        complete = observed == end - start
        result = {
            "name": phase.name,
            "expected_steps": end - start,
            "eligible_steps": observed,
            "complete": complete,
            "command": list(phase.command),
            "kinematic_passed": False,
            "failures": [],
        }
        command_end = max(start, min(attempted_end, end))
        actual = trace["command"][start:command_end]
        command_match = bool(np.isclose(actual, phase.command, atol=1e-6, rtol=0).all())
        if not complete:
            result["failures"].append("incomplete nonterminal first-attempt phase")
        elif end - start < settling:
            result["failures"].append("missing complete acquisition interval")
        else:
            result.update(
                _phase_score(
                    trace,
                    start,
                    end,
                    phase,
                    velocity_reference,
                    plane,
                    dt,
                    th,
                    settling,
                    window,
                )
            )
        result["command_match"] = command_match
        if not command_match:
            result["failures"].append("actual command differs from declared phase")
        result["kinematic_passed"] = not result["failures"]
        if not result["kinematic_passed"]:
            failures.append("phase failed: " + phase.name)
        reports.append(result)
    return {
        "version": VERSION,
        "status": "COMPONENT_FAIL" if failures else "COMPONENT_PASS",
        "kinematic_passed": not failures,
        "qualification_passed": False,
        "exit_allowed": False,
        "unscored_requirements": [
            "physical",
            "terrain_support",
            "transitions",
            "command_source_admission_and_lease",
            "streamed_input",
            "heldout_matrix",
            "process_exit",
        ],
        "velocity_reference": velocity_reference,
        "plane_attitude_gate": plane,
        "period_s": dt,
        "thresholds": asdict(th),
        "expected_steps": expected_steps,
        "recorded_steps": length,
        "first_terminal_step": terminal,
        "first_terminal_causes": causes,
        "eligible_nonterminal_steps": eligible_end,
        "excluded_housekeeping_steps": length - attempted_end,
        "failures": failures,
        "phases": reports,
    }
