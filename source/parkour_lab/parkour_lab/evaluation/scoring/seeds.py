"""Across-training-seed uncertainty; cases and checkpoints are not extra learners."""

import math
from collections import defaultdict

import numpy as np

from ..robustness import block_name, matched_control
from .aggregate import _weights, aggregate_cases
from .case import VERSION


def _seed_statistics(values):
    known = [value for value in values if value is not None]
    complete = len(known) == len(values)
    mean = float(np.mean(known)) if known else None
    deviation = float(np.std(known, ddof=1)) if len(known) > 1 else None
    return dict(
        prescribed_training_seeds=len(values),
        available_training_seeds=len(known),
        mean=mean if complete else None,
        mean_available_conditional=mean,
        standard_deviation=deviation if complete else None,
        standard_error=(
            deviation / math.sqrt(len(known))
            if complete and deviation is not None
            else None
        ),
    )


def _group_seed_statistics(summaries):
    result = {}
    for kind, cells in next(iter(summaries.values()))["groups"].items():
        result[kind] = {}
        for label in cells:
            rows = [summary["groups"][kind][label] for summary in summaries.values()]
            complete = [
                not row["missing_attempts"] and not row["unfinished_attempts"]
                for row in rows
            ]
            fields = (
                "core_return_mean",
                "physical_failure_fraction_prescribed",
                "weighted_survived_horizon_fraction_prescribed",
                "weighted_coverage",
            )
            metrics = {
                name: _seed_statistics(
                    [
                        row[name] if known else None
                        for row, known in zip(rows, complete, strict=True)
                    ]
                )
                for name in fields
            }
            metrics["diagnostics"] = {
                name: dict(
                    **_seed_statistics(
                        [
                            (
                                row["diagnostics"][name]["mean_conditional"]
                                if known
                                else None
                            )
                            for row, known in zip(rows, complete, strict=True)
                        ]
                    ),
                    conditional_on_available_phases=True,
                )
                for name in rows[0]["diagnostics"]
            }
            if "fault_outcomes" in rows[0]:
                metrics["fault_outcomes"] = {
                    name: _seed_statistics(
                        [
                            row["fault_outcomes"][name] if known else None
                            for row, known in zip(rows, complete, strict=True)
                        ]
                    )
                    for name in rows[0]["fault_outcomes"]
                }
            result[kind][label] = metrics
    return result


def _paired_seed_statistics(cases, summaries):
    groups = defaultdict(list)
    for case in cases:
        control = matched_control(case)
        if control is not None and control.case_id != case.case_id:
            groups[f"{block_name(case)}"].append(case)
            groups[f"{block_name(case)}/{case.family}"].append(case)
    fields = (
        "core_return_delta",
        "duration_delta_s",
        "post_onset_recorded_return_delta_conditional",
    )
    result = {}
    for label, group in groups.items():
        weights, seeds = _weights(group), {}
        for seed, summary in summaries.items():
            indexed = {row["case_id"]: row for row in summary["paired"]["pairs"]}
            seeds[str(seed)] = {}
            for field in fields:
                values = [
                    (weights[c.case_id], indexed[c.case_id][field])
                    for c in group
                    if indexed[c.case_id][field] is not None
                ]
                mean = (
                    sum(w * v for w, v in values) / sum(w for w, _ in values)
                    if values
                    else None
                )
                seeds[str(seed)][field] = dict(
                    value=(
                        mean
                        if len(values) == len(group) or field.endswith("_conditional")
                        else None
                    ),
                    available_pairs=len(values),
                    prescribed_pairs=len(group),
                )
        result[label] = dict(
            statistics={
                field: _seed_statistics([row[field]["value"] for row in seeds.values()])
                for field in fields
            },
            training_seeds=seeds,
        )
    return result


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
    statistics = _seed_statistics(
        [
            (
                report["overall"]["core_return_mean"]
                if report["complete"] and report["overall"] is not None
                else None
            )
            for report in summaries.values()
        ]
    )
    complete = statistics["available_training_seeds"] == len(summaries)
    return dict(
        version=VERSION,
        prescribed_training_seeds=len(summaries),
        complete_training_seeds=statistics["available_training_seeds"],
        complete=complete,
        core_return_mean=statistics["mean"],
        core_return_mean_complete_seeds_conditional=statistics[
            "mean_available_conditional"
        ],
        core_return_standard_deviation=statistics["standard_deviation"],
        core_return_standard_error=statistics["standard_error"],
        group_statistics=_group_seed_statistics(summaries),
        paired_statistics=_paired_seed_statistics(cases, summaries),
        uncertainty_unit="independently trained seed; no uncertainty from treating cases as independent learners",
        training_seeds={str(seed): report for seed, report in summaries.items()},
    )
