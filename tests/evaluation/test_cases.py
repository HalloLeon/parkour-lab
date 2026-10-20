"""Stable disjoint partitions and paired fixed geometry/noise draws."""

import hashlib
import json
import unittest
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError

from parkour_lab.evaluation.cases import EvaluationCase, case_bank, case_from_dict


class FixedCaseTests(unittest.TestCase):
    def test_legacy_bank_identities_and_missing_version_decode_are_unchanged(self):
        from parkour_lab.evaluation.robustness import robustness_bank

        for bank, expected in (
            (
                case_bank,
                "15146114196cba78d71ffb8da376f373108a6d629e3a6e365b7adbeb08846c0c",
            ),
            (
                robustness_bank,
                "d0569079c16067caa538219edc839f9b40aedd82a8196468d1552435ed4e765c",
            ),
        ):
            cases = bank("validation", attempts=2, version="v1")
            values = [
                dict(case=c.as_dict(), case_id=c.case_id, seed=c.seed) for c in cases
            ]
            self.assertEqual(
                hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
                expected,
            )
            for case in cases:
                self.assertNotIn("version", case.as_dict())
                self.assertEqual(case_from_dict(case.as_dict()), case)
        new = case_bank("validation")[0]
        self.assertEqual(new.as_dict()["version"], "v2")
        old = case_from_dict({k: v for k, v in new.as_dict().items() if k != "version"})
        self.assertEqual(old.version, "v1")
        self.assertNotEqual(new.seed, old.seed)
        self.assertNotEqual(new.case_id, old.case_id)

    def test_complete_bank_has_thirty_one_cells_six_profiles_and_no_flat_ladder(self):
        cases = case_bank("validation")
        self.assertEqual(len(cases), 186)
        self.assertEqual(len({case.case_id for case in cases}), 186)
        self.assertEqual(
            Counter(case.family for case in cases),
            dict(Flat=6, Uneven=30, Slopes=60, Stairs=60, Hills=30),
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
        self.assertEqual(len(identities), 1116)
        self.assertTrue(all(len(seeds) == 1 for seeds in paired.values()))
        self.assertEqual(len({next(iter(seeds)) for seeds in paired.values()}), 186)

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
