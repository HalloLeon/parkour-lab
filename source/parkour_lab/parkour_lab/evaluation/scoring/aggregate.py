"""Prescribed-attempt denominators and independent-training-seed summaries."""

import math
from collections import defaultdict

from ..cases import EvaluationCase, case_bank
from ..robustness import block_name, robustness_bank
from .case import VERSION
from .robustness import paired_results


def _weights(cases):
    families = {case.family for case in cases}
    result = {}
    for case in cases:
        family = [item for item in cases if item.family == case.family]
        level = [item for item in family if item.physical_level == case.physical_level]
        cell = [item for item in level if item.subtype == case.subtype]
        profile = [item for item in cell if item.profile == case.profile]
        result[case.case_id] = 1 / (
            len(families)
            * len({item.physical_level for item in family})
            * len({item.subtype for item in level})
            * len({item.profile for item in cell})
            * len(profile)
        )
    return result


def _summary(cases, indexed):
    weights = _weights(cases)
    observed = [case for case in cases if case.case_id in indexed]
    resolved = [
        case for case in observed if indexed[case.case_id]["core_return"] is not None
    ]
    total_weight = sum(weights[case.case_id] for case in resolved)
    mean = (
        sum(
            weights[case.case_id] * indexed[case.case_id]["core_return"]
            for case in resolved
        )
        / total_weight
        if total_weight
        else None
    )
    failures = sum(indexed[case.case_id]["physical_failure"] for case in observed)
    survivors = sum(indexed[case.case_id]["survived_horizon"] for case in observed)
    return dict(
        prescribed_attempts=len(cases),
        observed_attempts=len(observed),
        missing_attempts=len(cases) - len(observed),
        unfinished_attempts=len(observed) - len(resolved),
        core_return_mean=mean if len(resolved) == len(cases) else None,
        core_return_mean_observed_conditional=mean,
        physical_failures=failures,
        survived_horizon=survivors,
        physical_failure_fraction_prescribed=failures / len(cases),
        survived_horizon_fraction_prescribed=survivors / len(cases),
        weighted_survived_horizon_fraction_prescribed=sum(
            weights[case.case_id] * indexed[case.case_id]["survived_horizon"]
            for case in observed
        ),
        weighted_coverage=sum(
            weights[case.case_id] * indexed[case.case_id]["coverage"]
            for case in observed
        ),
        diagnostics=_diagnostics(cases, indexed, weights),
        **(
            _fault_outcomes(cases, indexed, weights)
            if any(case.fault for case in cases)
            else {}
        ),
    )


def _fault_outcomes(cases, indexed, weights):
    outcomes = [
        (weights[case.case_id], indexed[case.case_id]["fault"])
        for case in cases
        if case.case_id in indexed
    ]
    return dict(
        fault_outcomes=dict(
            prescribed_attempts=len(cases),
            observed_attempts=len(outcomes),
            onset_reached=sum(row["onset_reached"] for _, row in outcomes),
            failures_before_onset=sum(
                row["failure_before_onset"] for _, row in outcomes
            ),
            weighted_post_onset_coverage_prescribed=sum(
                weight * row["post_onset_coverage"] for weight, row in outcomes
            ),
        )
    )


def _diagnostics(cases, indexed, weights):
    fields = {
        "settled_planar_rmse_m_s": ("phases", "settled_motion", "planar_rmse_m_s"),
        "settled_yaw_rmse_rad_s": ("phases", "settled_motion", "yaw_rmse_rad_s"),
        "signed_body_yaw_rad_s": ("motion", "mean_signed_body_yaw_rad_s"),
        "wrong_direction_fraction": ("motion", "wrong_direction_fraction"),
        "stopping_drift_m": ("stopping", "maximum_drift_from_stop_onset_m"),
        "nonflat_exposure_fraction": (
            "exposure",
            "demonstrated_nonflat_fraction_of_contact_samples",
        ),
        "post_onset_planar_rmse_m_s": ("fault", "post_onset_motion", "planar_rmse_m_s"),
        "post_onset_yaw_rmse_rad_s": ("fault", "post_onset_motion", "yaw_rmse_rad_s"),
    }
    result = {}
    for name, path in fields.items():
        values = []
        for case in cases:
            value = indexed.get(case.case_id)
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                values.append((weights[case.case_id], value))
        result[name] = dict(
            mean_conditional=(
                sum(weight * value for weight, value in values)
                / sum(weight for weight, _ in values)
                if values
                else None
            ),
            available_attempts=len(values),
            prescribed_attempts=len(cases),
        )
    return result


def _validate_result(case, result):
    if (
        result.get("version") != VERSION
        or result.get("case") != case.as_dict()
        or result.get("seed") != case.seed
    ):
        raise ValueError("Scored case identity differs from its prescription")
    if result.get("core_return") is not None and (
        isinstance(result.get("core_return"), bool)
        or not isinstance(result.get("core_return"), (int, float))
        or not math.isfinite(result["core_return"])
    ):
        raise ValueError("Require a finite observed core return")
    if any(
        type(result.get(name)) is not bool
        for name in ("physical_failure", "survived_horizon")
    ):
        raise ValueError("Require explicit failure and horizon outcomes")
    if (
        not isinstance(result.get("coverage"), (int, float))
        or isinstance(result["coverage"], bool)
        or not 0 <= result["coverage"] <= 1
    ):
        raise ValueError("Invalid observed first-attempt coverage")
    if (
        (result["physical_failure"] and result["survived_horizon"])
        or (result["survived_horizon"] and result["coverage"] != 1)
        or (
            result["core_return"] is None
            and (result["physical_failure"] or result["survived_horizon"])
        )
    ):
        raise ValueError("Failure, horizon and known-return outcomes conflict")


def _index_results(prescribed, results):
    indexed = {}
    if isinstance(results, dict):
        if any(
            not isinstance(value, dict) or key != value.get("case_id")
            for key, value in results.items()
        ):
            raise ValueError("Result mapping keys must equal their case IDs")
        results = results.values()
    for result in results:
        if not isinstance(result, dict) or "core_return" not in result:
            raise ValueError("Require a scored case with an explicit core return")
        identity = result.get("case_id")
        if identity not in prescribed or identity in indexed:
            raise ValueError("Unknown or duplicate scored case; do not select reruns")
        case = prescribed[identity]
        _validate_result(case, result)
        indexed[identity] = result
    return indexed


def _group_summaries(cases, core, indexed):
    groups = {}
    for name, key in (
        ("family", lambda case: case.family),
        ("profile", lambda case: case.profile),
        ("block", block_name),
        ("block_family", lambda case: f"{block_name(case)}/{case.family}"),
        (
            "family_level_subtype",
            lambda case: f"{case.family}/{case.physical_level}/{case.subtype}",
        ),
    ):
        cells = defaultdict(list)
        for case in (cases if name.startswith("block") else core):
            cells[key(case)].append(case)
        groups[name] = {
            label: _summary(group, indexed) for label, group in sorted(cells.items())
        }
    return groups


def aggregate_cases(cases, results):
    """One trained policy/seed: equal families, then levels, subtypes and profiles.

    Physical failures retain their original terminal return. Unknown missing
    returns remain None; only explicitly conditional observed means omit them.
    """
    cases = tuple(cases)
    if not cases or not all(isinstance(case, EvaluationCase) for case in cases):
        raise ValueError("Require prescribed evaluation cases")
    prescribed = {case.case_id: case for case in cases}
    if (
        len(prescribed) != len(cases)
        or len({case.partition for case in cases}) != 1
        or len({case.version for case in cases}) != 1
    ):
        raise ValueError("Cases require distinct IDs in one partition and version")
    indexed = _index_results(prescribed, results)
    missing = [case.case_id for case in cases if case.case_id not in indexed]
    core = tuple(case for case in cases if case.condition == "id")
    groups = _group_summaries(cases, core, indexed)
    attempts = max(case.attempt for case in cases) + 1
    canonical = case_bank(
        cases[0].partition, attempts=attempts, version=cases[0].version
    )
    has_robustness = len(core) != len(cases)
    full_core = {case.case_id for case in core} == {case.case_id for case in canonical}
    full_robustness = has_robustness and set(prescribed) == {
        case.case_id
        for case in robustness_bank(
            cases[0].partition, attempts=attempts, version=cases[0].version
        )
    }
    return dict(
        version=VERSION,
        partition=cases[0].partition,
        complete=not missing
        and all(result["core_return"] is not None for result in indexed.values()),
        full_partition_bank=full_core,
        full_robustness_bank=full_robustness,
        full_requested_bank=full_robustness if has_robustness else full_core,
        weights="Equal families; equal physical levels, subtypes, profiles and attempts within each family.",
        case_weights=_weights(core) if core else {},
        missing_case_ids=missing,
        overall=_summary(core, indexed) if core else None,
        overall_scope="Core ID cases only; robustness blocks have independent summaries and weights.",
        block_weights={
            label: _weights(tuple(case for case in cases if block_name(case) == label))
            for label in groups["block"]
        },
        paired=paired_results(cases, indexed),
        groups=groups,
        attempts=[
            indexed.get(
                case.case_id,
                dict(
                    case_id=case.case_id,
                    case=case.as_dict(),
                    seed=case.seed,
                    status="missing",
                    core_return=None,
                    coverage=0.0,
                ),
            )
            for case in cases
        ],
    )
