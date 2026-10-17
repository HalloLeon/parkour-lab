"""Stable disjoint partitions and paired fixed geometry/noise draws."""

import json
import unittest
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError

from parkour_lab.evaluation.cases import EvaluationCase, case_bank, case_from_dict


class FixedCaseTests(unittest.TestCase):
    def test_complete_bank_has_nineteen_cells_six_profiles_and_no_flat_ladder(self):
        cases = case_bank("validation")
        self.assertEqual(len(cases), 114)
        self.assertEqual(len({case.case_id for case in cases}), 114)
        self.assertEqual(
            Counter(case.family for case in cases),
            dict(Flat=6, Uneven=18, Slopes=36, Stairs=36, Hills=18),
        )
        self.assertEqual(
            {case.physical_level for case in cases if case.family == "Flat"}, {None}
        )
        self.assertTrue(all("/" not in case.case_id for case in cases))

    def test_profiles_pair_environment_seed_and_partitions_attempts_are_disjoint(self):
        identities, paired = set(), defaultdict(set)
        for partition in ("train", "validation", "test"):
            for case in reversed(case_bank(partition, attempts=2)):
                geometry = (
                    partition,
                    case.family,
                    case.physical_level,
                    case.subtype,
                    case.attempt,
                )
                paired[geometry].add(case.seed)
                self.assertTrue(0 <= case.seed < 2**32)
                self.assertEqual(
                    case, case_from_dict(json.loads(json.dumps(case.as_dict())))
                )
                identities.add(case.case_id)
        self.assertEqual(len(identities), 684)
        self.assertTrue(all(len(seeds) == 1 for seeds in paired.values()))
        self.assertEqual(len({next(iter(seeds)) for seeds in paired.values()}), 114)

    def test_twenty_second_windows_and_both_yaw_signs(self):
        for case in case_bank("validation"):
            for time in (0, 1.98, 14, 19.98, 20):
                self.assertEqual(case.command_at(time), (0, 0, 0))
            command = case.command_at(2)
            self.assertEqual(case.command_at(13.98), command)
            if case.profile == "pivot-left":
                self.assertEqual(command, (0, 0, 0.5))
            if case.profile == "arc-right":
                self.assertEqual(command, (0.35, 0, -0.3))

    def test_identity_fields_are_strict_and_case_is_frozen(self):
        case = EvaluationCase("test", "Flat", None, "plane", "stand")
        with self.assertRaises(FrozenInstanceError):
            case.attempt = 2
        for changes in (
            {"physical_level": 0},
            {"attempt": True},
            {"profile": "other"},
            {"partition": "final"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                EvaluationCase(**dict(case.as_dict(), **changes))
        for value in (dict(case.as_dict(), seed=case.seed), {}):
            with self.assertRaises(ValueError):
                case_from_dict(value)
        for count in (0, -1, True):
            with self.assertRaises(ValueError):
                case_bank("test", attempts=count)


if __name__ == "__main__":
    unittest.main()
