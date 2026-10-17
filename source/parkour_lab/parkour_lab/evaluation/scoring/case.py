"""Authoritative first-attempt score from aligned native outgoing transitions."""

from ..cases import PERIOD_S, STEPS
from .exposure import surface_exposure
from .kinematics import motion_scores, phase_scores, stopping_scores
from .trace import validate_trace

VERSION = "go2_fixed_scores_development_v1"


def score_case(case, arrays):
    """Return original core reward sum and physical diagnostics, without pass gates."""
    trace = validate_trace(case, arrays)
    count = len(trace["reward"])
    terminated = bool(count and trace["terminated"][-1])
    truncated = bool(count and trace["truncated"][-1])
    first = None
    if terminated or truncated:
        causes = [
            name.removeprefix("ending_")
            for name, value in trace.items()
            if name.startswith("ending_") and value[-1]
        ]
        if terminated:
            causes.append("physical_failure")
        if truncated and not any(
            name in causes for name in ("terrain_boundary", "time_out")
        ):
            causes.append("unknown_truncation")
        first = dict(
            time_s=count * PERIOD_S,
            physical_failure=terminated,
            truncated=truncated,
            causes=sorted(set(causes)),
        )
    return dict(
        version=VERSION,
        case=case.as_dict(),
        case_id=case.case_id,
        seed=case.seed,
        status=(
            "complete"
            if count == STEPS
            else "ended" if first else "incomplete" if count else "unobserved"
        ),
        observed_steps=count,
        duration_s=count * PERIOD_S,
        coverage=count / STEPS,
        core_return=float(trace["reward"].sum()) if count == STEPS or first else None,
        recorded_return=float(trace["reward"].sum()),
        first_ending=first,
        physical_failure=terminated,
        survived_horizon=count == STEPS and not terminated,
        tracking_interpretation="Errors are conditional on recorded live samples; missing phases are unavailable, never zero error.",
        phases=phase_scores(trace),
        motion=motion_scores(case, trace),
        stopping=stopping_scores(trace),
        exposure=surface_exposure(case, trace),
    )
