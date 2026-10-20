"""Post-onset diagnostics and paired deltas with every prescribed outcome retained."""

from ..cases import DURATION_S, PERIOD_S
from ..robustness import matched_control
from .kinematics import tracking


def fault_scores(case, trace):
    """Report exposure and pre/post windows without counting early falls as recovery."""
    if case.fault is None:
        return None
    onset = case.fault.onset_s
    time = trace["decision_time_s"]
    duration = len(time) * PERIOD_S
    reached = duration > onset
    failed = bool(len(time) and trace["terminated"][-1])
    return dict(
        joint_name=case.fault.joint_name,
        drive_factor=case.fault.factor,
        onset_s=onset,
        onset_reached=reached,
        failure_before_onset=failed and not reached,
        post_onset_observed_s=max(0.0, duration - onset),
        post_onset_coverage=max(0.0, duration - onset) / (DURATION_S - onset),
        time_to_failure_after_onset_s=(
            duration - onset if failed and reached else None
        ),
        pre_onset=_window(trace, (time >= 2) & (time < onset)),
        post_onset_motion=_window(trace, (time >= onset) & (time < 14)),
        post_onset_total=_window(trace, time >= onset),
        interpretation="Window errors/returns describe observed samples only; pre-onset failures remain prescribed outcomes, not successful recovery.",
    )


def _window(trace, mask):
    count = int(mask.sum())
    return dict(
        observed_steps=count,
        recorded_return=float(trace["reward"][mask].sum()),
        **tracking(trace, mask),
    )


def paired_results(cases, indexed):
    """Match prescriptions, not observed survivor subsets or reruns."""
    prescribed = {case.case_id for case in cases}
    pairs = []
    for case in cases:
        control = matched_control(case)
        if control is None or control.case_id == case.case_id:
            continue
        result, healthy = indexed.get(case.case_id), indexed.get(control.case_id)
        known = (
            result is not None
            and healthy is not None
            and result["core_return"] is not None
            and healthy["core_return"] is not None
        )
        fault, baseline_fault = (
            row.get("fault") if row is not None else None for row in (result, healthy)
        )
        exposed = bool(
            known
            and fault
            and baseline_fault
            and fault["onset_reached"]
            and baseline_fault["onset_reached"]
        )
        pairs.append(
            dict(
                case_id=case.case_id,
                control_case_id=control.case_id,
                control_prescribed=control.case_id in prescribed,
                complete=known,
                core_return_delta=(
                    result["core_return"] - healthy["core_return"] if known else None
                ),
                duration_delta_s=(
                    result["duration_s"] - healthy["duration_s"] if known else None
                ),
                both_reached_onset=exposed if case.fault else None,
                post_onset_recorded_return_delta_conditional=(
                    fault["post_onset_total"]["recorded_return"]
                    - baseline_fault["post_onset_total"]["recorded_return"]
                    if exposed
                    else None
                ),
            )
        )
    return dict(
        version=f"go2_robustness_development_{cases[0].version}",
        interpretation="Paired deltas retain missing and pre-onset failures. Post-onset deltas require both attempts to reach onset and remain conditional on their unequal observed durations.",
        prescribed_pairs=len(pairs),
        complete_pairs=sum(pair["complete"] for pair in pairs),
        pairs=pairs,
    )
