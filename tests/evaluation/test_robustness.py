"""Excluded conditions, matched draws and post-onset accounting without simulator imports."""

import json
import unittest
from collections import Counter
from dataclasses import replace

from parkour_lab.environments.faults import FaultConfig
from parkour_lab.evaluation.cases import EvaluationCase, case_from_dict
from parkour_lab.evaluation.robustness import JOINTS, matched_control, robustness_bank
from parkour_lab.evaluation.scoring import aggregate_cases, score_case
from tests.support.evaluation import analytic_trace


class RobustnessTests(unittest.TestCase):
    def test_complete_bank_balances_joints_and_pairs_seeded_starts_and_onsets(self):
        cases = robustness_bank("validation")
        self.assertEqual(len(cases), 588)
        self.assertEqual(len({case.case_id for case in cases}), 588)
        self.assertEqual(
            Counter(case.condition for case in cases),
            dict(id=186, composition=150, terrain_ood=180, fault=72),
        )
        indexed = {case.case_id: case for case in cases}
        faults = [case for case in cases if case.fault]
        self.assertEqual(
            Counter(case.fault.joint_name for case in faults), dict.fromkeys(JOINTS, 6)
        )
        for case in cases:
            self.assertEqual(
                case, case_from_dict(json.loads(json.dumps(case.as_dict())))
            )
            control = matched_control(case)
            if control is not None:
                self.assertIn(control.case_id, indexed)
                self.assertEqual(case.seed, control.seed)
                if case.fault:
                    self.assertEqual(case.fault.onset_s, control.fault.onset_s)
        other = robustness_bank("test")
        self.assertTrue(
            {case.seed for case in cases}.isdisjoint(case.seed for case in other)
        )
        with self.assertRaises(ValueError):
            robustness_bank("train")

    def test_invalid_crossed_conditions_are_rejected_and_id_receipts_stay_unchanged(
        self,
    ):
        base = EvaluationCase("validation", "Flat", None, "plane", "stand")
        self.assertEqual(
            set(base.as_dict()),
            {
                "partition",
                "family",
                "physical_level",
                "subtype",
                "profile",
                "attempt",
                "version",
            },
        )
        for changes in (
            dict(condition="composition"),
            dict(condition="terrain_ood"),
            dict(condition="fault"),
            dict(fault=FaultConfig(JOINTS[0], 0, 7)),
        ):
            with self.assertRaises(ValueError):
                replace(base, **changes)

    def fault_pair(self):
        fault = next(
            case
            for case in robustness_bank("validation")
            if case.fault and case.fault.factor == 0
        )
        return fault, matched_control(fault)

    def test_pre_onset_failure_is_retained_without_inventing_post_fault_scores(self):
        fault, control = self.fault_pair()
        trace = analytic_trace(fault, steps=50)
        trace["terminated"][-1] = True
        failed = score_case(fault, trace)
        self.assertTrue(failed["fault"]["failure_before_onset"])
        self.assertFalse(failed["fault"]["onset_reached"])
        self.assertEqual(failed["fault"]["post_onset_coverage"], 0)
        self.assertIsNone(failed["fault"]["post_onset_motion"]["planar_rmse_m_s"])
        report = aggregate_cases((fault, control), [failed])
        self.assertIsNone(report["overall"])
        self.assertEqual(report["paired"]["prescribed_pairs"], 1)
        self.assertFalse(report["paired"]["pairs"][0]["complete"])
        self.assertIsNone(report["paired"]["pairs"][0]["core_return_delta"])
        self.assertEqual(
            report["groups"]["block"]["fault_drive_0"]["physical_failures"], 1
        )

    def test_onset_windows_and_paired_delta_do_not_mix_into_core_score(self):
        fault, control = self.fault_pair()
        core = replace(fault, condition="id", fault=None)
        results = [
            score_case(case, analytic_trace(case)) for case in (fault, control, core)
        ]
        results[0]["core_return"] = -100
        results[1]["core_return"] = 20
        report = aggregate_cases((fault, control, core), results)
        self.assertEqual(report["overall"]["core_return_mean"], 1000)
        self.assertEqual(report["paired"]["pairs"][0]["core_return_delta"], -120)
        self.assertTrue(report["paired"]["pairs"][0]["both_reached_onset"])
        self.assertAlmostEqual(results[0]["fault"]["post_onset_coverage"], 1)
        self.assertEqual(results[0]["fault"]["post_onset_motion"]["planar_rmse_m_s"], 0)
        self.assertEqual(report["case_weights"], {core.case_id: 1})

    def test_delivered_effort_wrong_joint_or_early_onset_cannot_be_scored(self):
        fault, _ = self.fault_pair()
        for change in ("applied_effort", "drive_factor", "fault_active"):
            trace = analytic_trace(fault)
            if change == "fault_active":
                trace[change][0] = True
            else:
                trace[change][0, 0] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                score_case(fault, trace)

    def test_full_bank_requires_every_robustness_attempt_and_each_block_has_own_weights(
        self,
    ):
        cases = robustness_bank("validation")
        report = aggregate_cases(cases, [])
        self.assertTrue(report["full_partition_bank"])
        self.assertTrue(report["full_robustness_bank"])
        self.assertTrue(report["full_requested_bank"])
        self.assertFalse(report["complete"])
        self.assertEqual(report["overall"]["prescribed_attempts"], 186)
        for weights in report["block_weights"].values():
            self.assertAlmostEqual(sum(weights.values()), 1)
        partial = aggregate_cases(cases[:-1], [])
        self.assertFalse(partial["full_requested_bank"])
        self.assertEqual(report["paired"]["prescribed_pairs"], 378)


if __name__ == "__main__":
    unittest.main()
