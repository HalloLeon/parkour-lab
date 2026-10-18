"""Family/profile weights and all-prescribed denominators, including missing data."""

import unittest
from copy import deepcopy
from dataclasses import replace

from parkour_lab.evaluation.cases import EvaluationCase, case_bank
from parkour_lab.evaluation.scoring import (
    aggregate_cases,
    aggregate_training_seeds,
    score_case,
)
from tests.support.evaluation import analytic_trace


class AggregationTests(unittest.TestCase):
    def test_family_equal_weighting_and_both_directions_visible(self):
        cases = case_bank("validation")
        results = []
        values = dict(Flat=10, Uneven=20, Slopes=30, Stairs=40, Hills=50)
        for case in cases:
            result = score_case(case, analytic_trace(case))
            result["core_return"] = values[case.family]
            results.append(result)
        report = aggregate_cases(cases, reversed(results))
        self.assertTrue(report["complete"])
        self.assertTrue(report["full_partition_bank"])
        self.assertAlmostEqual(report["overall"]["core_return_mean"], 30)
        self.assertAlmostEqual(sum(report["case_weights"].values()), 1)
        for family in values:
            self.assertAlmostEqual(
                sum(
                    report["case_weights"][case.case_id]
                    for case in cases
                    if case.family == family
                ),
                0.2,
            )
        self.assertIn("pivot-left", report["groups"]["profile"])
        self.assertIn("pivot-right", report["groups"]["profile"])

    def test_failed_missing_and_unfinished_attempts_keep_prescribed_denominator(self):
        cases = case_bank("test")[:3]
        first = analytic_trace(cases[0])
        first["terminated"][9] = True
        failed = score_case(cases[0], first)
        unfinished = score_case(cases[1], analytic_trace(cases[1], steps=0))
        result = aggregate_cases(
            cases, {row["case_id"]: row for row in (failed, unfinished)}
        )
        self.assertFalse(result["complete"])
        overall = result["overall"]
        self.assertEqual(overall["prescribed_attempts"], 3)
        self.assertEqual(overall["observed_attempts"], 2)
        self.assertEqual(overall["missing_attempts"], 1)
        self.assertEqual(overall["unfinished_attempts"], 1)
        self.assertEqual(overall["physical_failures"], 1)
        self.assertAlmostEqual(overall["physical_failure_fraction_prescribed"], 1 / 3)
        self.assertIsNone(overall["core_return_mean"])
        self.assertEqual(overall["core_return_mean_observed_conditional"], 10)
        self.assertEqual(result["attempts"][-1]["status"], "missing")

    def test_duplicate_unknown_or_changed_case_identity_is_rejected(self):
        cases = case_bank("validation")[:2]
        result = score_case(cases[0], analytic_trace(cases[0]))
        variants = (
            [result, result],
            [dict(result, seed=1)],
            [dict(result, case_id="other")],
        )
        for rows in variants:
            with self.assertRaises(ValueError):
                aggregate_cases(cases, rows)

    def test_training_seed_uncertainty_uses_independent_learners(self):
        cases = case_bank("validation")[:1]
        first = score_case(cases[0], analytic_trace(cases[0]))
        second = deepcopy(first)
        first["core_return"], second["core_return"] = 10, 14
        report = aggregate_training_seeds(cases, {42: [first], 73: [second]})
        self.assertEqual(report["core_return_mean"], 12)
        self.assertAlmostEqual(report["core_return_standard_error"], 2)
        single = aggregate_training_seeds(cases, {42: [first]})
        self.assertIsNone(single["core_return_standard_error"])
        missing = aggregate_training_seeds(cases, {42: [first], 73: []})
        self.assertFalse(missing["complete"])
        self.assertIsNone(missing["core_return_mean"])

    def test_five_seed_group_statistics_and_unavailable_phases(self):
        cases = case_bank("validation")[:1]
        rows = {}
        for seed in range(5):
            result = score_case(cases[0], analytic_trace(cases[0]))
            result["core_return"] = seed + 1
            rows[seed] = [result]
        report = aggregate_training_seeds(cases, rows)
        self.assertAlmostEqual(report["core_return_mean"], 3)
        self.assertAlmostEqual(report["core_return_standard_deviation"], 2.5**0.5)
        self.assertAlmostEqual(report["core_return_standard_error"], 0.5**0.5)
        group = report["group_statistics"]["family"]["Flat"]
        self.assertEqual(group["core_return_mean"]["available_training_seeds"], 5)
        self.assertAlmostEqual(group["core_return_mean"]["standard_error"], 0.5**0.5)
        self.assertTrue(
            group["diagnostics"]["settled_planar_rmse_m_s"][
                "conditional_on_available_phases"
            ]
        )
        rows[4] = []
        missing = aggregate_training_seeds(cases, rows)["group_statistics"]["family"][
            "Flat"
        ]
        self.assertIsNone(missing["physical_failure_fraction_prescribed"]["mean"])
        self.assertEqual(
            missing["physical_failure_fraction_prescribed"]["available_training_seeds"],
            4,
        )

    def test_paired_block_deltas_are_averaged_within_training_seed(self):
        base = EvaluationCase("validation", "Slopes", 0, "pyramid", "forward-035")
        composed = replace(base, condition="composition")
        cases, rows = (base, composed), {}
        for seed, delta in enumerate((2.0, 4.0, 6.0)):
            pair = []
            for case, value in ((base, 100.0 + seed), (composed, 100.0 + seed + delta)):
                trace = analytic_trace(case, steps=1)
                trace["terminated"][:] = True
                trace["reward"][:] = value
                pair.append(score_case(case, trace))
            rows[seed] = pair
        summary = aggregate_training_seeds(cases, rows)
        statistics = summary["paired_statistics"]["composition"]["statistics"][
            "core_return_delta"
        ]
        self.assertAlmostEqual(statistics["mean"], 4)
        self.assertAlmostEqual(statistics["standard_deviation"], 2)
        self.assertAlmostEqual(statistics["standard_error"], 2 / 3**0.5)
        self.assertEqual(statistics["prescribed_training_seeds"], 3)


if __name__ == "__main__":
    unittest.main()
