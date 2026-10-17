"""Prescribed-attempt denominators and independent-training-seed summaries."""

import math
from collections import defaultdict

import numpy as np

from ..cases import EvaluationCase, case_bank
from .case import VERSION


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


def aggregate_cases(cases, results):
    """One trained policy/seed: equal families, then levels, subtypes and profiles.

    Physical failures retain their original terminal return. Unknown missing
    returns remain None; only explicitly conditional observed means omit them.
    """
    cases = tuple(cases)
    if not cases or not all(isinstance(case, EvaluationCase) for case in cases):
        raise ValueError("Require prescribed evaluation cases")
    prescribed = {case.case_id: case for case in cases}
    if len(prescribed) != len(cases) or len({case.partition for case in cases}) != 1:
        raise ValueError("Cases require distinct IDs in one partition")
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
        indexed[identity] = result
    missing = [case.case_id for case in cases if case.case_id not in indexed]
    groups = {}
    for name, key in (
        ("family", lambda case: case.family),
        ("profile", lambda case: case.profile),
        (
            "family_level_subtype",
            lambda case: f"{case.family}/{case.physical_level}/{case.subtype}",
        ),
    ):
        cells = defaultdict(list)
        for case in cases:
            cells[key(case)].append(case)
        groups[name] = {
            label: _summary(group, indexed) for label, group in sorted(cells.items())
        }
    canonical = case_bank(
        cases[0].partition, attempts=max(case.attempt for case in cases) + 1
    )
    return dict(
        version=VERSION,
        partition=cases[0].partition,
        complete=not missing
        and all(result["core_return"] is not None for result in indexed.values()),
        full_partition_bank=set(prescribed) == {case.case_id for case in canonical},
        weights="Equal families; equal physical levels, subtypes, profiles and attempts within each family.",
        case_weights=_weights(cases),
        missing_case_ids=missing,
        overall=_summary(cases, indexed),
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


def aggregate_training_seeds(cases, results_by_seed):
    """Uncertainty is across independently trained seeds, never across test cases."""
    if (
        not isinstance(results_by_seed, dict)
        or not results_by_seed
        or any(type(seed) is not int or seed < 0 for seed in results_by_seed)
    ):
        raise ValueError("Require nonnegative independent training seed IDs")
    cases = tuple(cases)
    summaries = {
        seed: aggregate_cases(cases, results)
        for seed, results in sorted(results_by_seed.items())
    }
    values = [
        report["overall"]["core_return_mean"]
        for report in summaries.values()
        if report["complete"]
    ]
    complete = len(values) == len(summaries)
    mean = float(np.mean(values)) if values else None
    deviation = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return dict(
        version=VERSION,
        prescribed_training_seeds=len(summaries),
        complete_training_seeds=len(values),
        complete=complete,
        core_return_mean=mean if complete else None,
        core_return_mean_complete_seeds_conditional=mean,
        core_return_standard_deviation=deviation if complete else None,
        core_return_standard_error=(
            deviation / math.sqrt(len(values))
            if complete and deviation is not None
            else None
        ),
        uncertainty_unit="independently trained seed; no uncertainty from treating cases as independent learners",
        training_seeds={str(seed): report for seed, report in summaries.items()},
    )
