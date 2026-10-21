"""Bounded cached teacher evidence; diagnostics never control promotion decisions."""

import math
from copy import deepcopy

VERSION = "go2_teacher_gate_diagnostics_v1"
GATES = (
    "physical_clear",
    "completion",
    "sample_alignment",
    "tracking_available",
    "planar_tracking",
    "yaw_tracking",
    "directional_progress",
    "nonflat_contact",
)
COUNTS = ("level", "revision", "steps", "episode_length_steps", "tracked_steps")
METRICS = (
    "duration_s",
    "contact_seconds",
    "command_speed_m_s",
    "planar_rmse_m_s",
    "planar_limit_m_s",
    "yaw_rmse_rad_s",
    "yaw_limit_rad_s",
    "directional_progress_m",
    "requested_travel_m",
    "minimum_progress_m",
)


def new_diagnostics():
    return dict(
        version=VERSION,
        admitted_outcomes=0,
        successful_outcomes=0,
        gate_failures=dict.fromkeys(GATES, 0),
        last_outcome=None,
    )


def record_outcome(state, outcome):
    """Retain counts plus one completed outcome, with no growing sample history."""
    diagnostic = state.setdefault("diagnostics", new_diagnostics())
    diagnostic["admitted_outcomes"] += 1
    diagnostic["successful_outcomes"] += int(outcome["success"])
    for name, passed in outcome["gates"].items():
        diagnostic["gate_failures"][name] += int(not passed)
    diagnostic["last_outcome"] = deepcopy(outcome)


def _integer(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_diagnostics(state, block_size):
    """Check finite values, cached verdicts and counts before any restore mutation."""
    value = state["diagnostics"]
    if (
        type(value) is not dict
        or set(value) != set(new_diagnostics())
        or value["version"] != VERSION
    ):
        raise ValueError("Invalid teacher diagnostics schema")
    admitted, successes, failures, last = (
        value[key]
        for key in (
            "admitted_outcomes",
            "successful_outcomes",
            "gate_failures",
            "last_outcome",
        )
    )
    expected_successes = sum(row["successes"] for row in state["history"]) + sum(
        state["block"]
    )
    if (
        not _integer(admitted)
        or admitted != state["completed_blocks"] * block_size + len(state["block"])
        or not _integer(successes)
        or successes != expected_successes
        or type(failures) is not dict
        or set(failures) != set(GATES)
        or any(not _integer(v) or v > admitted - successes for v in failures.values())
        or sum(failures.values()) < admitted - successes
    ):
        raise ValueError("Teacher diagnostics and outcome counts differ")
    if admitted == 0:
        if last is not None:
            raise ValueError("Unobserved teacher diagnostics have a last outcome")
        return
    expected = {
        *COUNTS,
        *METRICS,
        "physical_termination",
        "horizon",
        "boundary",
        "gates",
        "success",
    }
    if (
        type(last) is not dict
        or set(last) != expected
        or any(not _integer(last[key]) for key in COUNTS)
        or last["steps"] == 0
        or not 0 <= last["level"] < 16
        or last["tracked_steps"] > last["steps"]
        or any(last[key] is not None and not _number(last[key]) for key in METRICS)
        or any(
            type(last[key]) is not bool
            for key in ("physical_termination", "horizon", "boundary", "success")
        )
        or type(last["gates"]) is not dict
        or set(last["gates"]) != set(GATES)
        or any(type(v) is not bool for v in last["gates"].values())
    ):
        raise ValueError("Invalid last teacher outcome")
    speed, duration, contact = (
        last[key] for key in ("command_speed_m_s", "duration_s", "contact_seconds")
    )
    if (
        speed is None
        or speed <= 0
        or duration != last["steps"] * 0.02
        or contact is None
        or not 0 <= contact <= duration
        or last["planar_limit_m_s"] != 0.3 * speed
        or last["yaw_limit_rad_s"] != 0.15
        or last["requested_travel_m"] != speed * last["steps"] * 0.02
        or last["minimum_progress_m"] != 0.7 * last["requested_travel_m"]
    ):
        raise ValueError("Invalid teacher outcome units or thresholds")

    def below(key, limit):
        return last[key] is not None and 0 <= last[key] <= limit

    gates = dict(
        physical_clear=not last["physical_termination"],
        completion=last["horizon"] or last["boundary"],
        sample_alignment=last["steps"] == last["episode_length_steps"],
        tracking_available=last["tracked_steps"] > 0,
        planar_tracking=below("planar_rmse_m_s", last["planar_limit_m_s"]),
        yaw_tracking=below("yaw_rmse_rad_s", last["yaw_limit_rad_s"]),
        directional_progress=last["directional_progress_m"] is not None
        and last["directional_progress_m"] >= last["minimum_progress_m"],
        nonflat_contact=last["level"] == 0 or contact >= 1.0,
    )
    if gates != last["gates"] or last["success"] != all(gates.values()):
        raise ValueError("Teacher diagnostic verdict differs from recorded gates")
    if any(not passed and failures[name] == 0 for name, passed in gates.items()):
        raise ValueError("Last failed teacher gate is missing from cumulative counts")
    if state["block"] and last["success"] != state["block"][-1]:
        raise ValueError("Last teacher diagnostic differs from the pending block")


def snapshot(teacher):
    """Read cached administrative state only; never collect sensors or RNG draws."""
    if teacher is None:
        return None
    return dict(
        version=VERSION,
        families={
            name: dict(
                target_level=state["target_level"],
                revision=state["revision"],
                pending_outcomes=len(state["block"]),
                completed_blocks=state["completed_blocks"],
                **deepcopy(state["diagnostics"]),
            )
            for name, state in teacher["families"].items()
        },
    )
