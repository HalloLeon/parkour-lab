"""Offline capture admission binds policy, recipe, counts and complete attempts."""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from parkour_lab.evaluation.capture import VERSION, reward_identity, score_capture
from parkour_lab.evaluation.cases import EvaluationCase
from parkour_lab.provenance import file_sha256, write_json
from tests.support.evaluation import analytic_trace


def write_capture(folder, *, steps=1000, case=None):
    case = case or EvaluationCase("validation", "Flat", None, "plane", "stand")
    path = folder / case.case_id
    path.mkdir(parents=True)
    np.savez_compressed(path / "trace.npz", **analytic_trace(case, steps=steps))
    recipe = {"control_dt_s": 0.02, "terms": {"tracking": {"weight": 1.0}}}
    identity = dict(
        policy_state_sha256="a" * 64,
        training_seed=42,
        checkpoint_id="caller-checkpoint",
        checkpoint_sha256="b" * 64,
    )
    manifest = dict(
        version=VERSION,
        status="COMPLETE",
        cases=[case.as_dict()],
        **identity,
        final_policy_state_sha256=identity["policy_state_sha256"],
        core_reward_recipe_sha256=reward_identity(recipe),
    )
    evidence = dict(
        version=VERSION,
        case=case.as_dict(),
        status="CAPTURED",
        **identity,
        cleanup={"environment": "complete", "application": "not_created"},
        observed_steps=steps,
        interactions=dict(session=steps, cumulative=steps, uncertain=0),
        realization={"task": {"reward_recipe": recipe}},
        core_reward_recipe_sha256=reward_identity(recipe),
        trace_sha256=file_sha256(path / "trace.npz"),
    )
    if case.condition != "id":
        from parkour_lab.evaluation.robustness import VERSION as robustness_version

        manifest["robustness_version"] = robustness_version
        evidence["realization"]["task"]["terrain"] = {
            "condition": case.condition if case.condition != "fault" else "id"
        }
        if case.fault:
            evidence["realization"]["fault"] = case.fault.as_dict()
    write_json(folder / "manifest.json", manifest)
    write_json(path / "capture.json", evidence)
    return case, manifest, evidence


class CaptureIntegrityTests(unittest.TestCase):
    def test_robustness_trace_requires_matching_realized_condition_and_protocol(self):
        from parkour_lab.evaluation.robustness import robustness_bank

        case = next(case for case in robustness_bank("validation") if case.fault)
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            _, manifest, evidence = write_capture(folder, case=case)
            self.assertTrue(score_capture(folder)["capture_valid"])
            for mutate in (
                lambda row: row["realization"].pop("fault"),
                lambda row: row["realization"]["fault"].update(factor=0),
                lambda row: row["realization"]["task"]["terrain"].update(
                    condition="composition"
                ),
            ):
                bad = deepcopy(evidence)
                mutate(bad)
                write_json(folder / case.case_id / "capture.json", bad)
                with self.assertRaises(ValueError):
                    score_capture(folder)
            write_json(folder / case.case_id / "capture.json", evidence)
            manifest.pop("robustness_version")
            write_json(folder / "manifest.json", manifest)
            with self.assertRaises(ValueError):
                score_capture(folder)

    def test_known_subset_is_valid_capture_and_explicitly_not_full_benchmark(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            write_capture(folder)
            report = score_capture(folder)
            self.assertTrue(report["capture_valid"])
            self.assertFalse(report["valid_for_comparison"])
            self.assertFalse(report["aggregate"]["full_partition_bank"])
            self.assertEqual(report["aggregate"]["overall"]["prescribed_attempts"], 1)

    def test_other_policy_checkpoint_seed_or_receipt_version_cannot_be_spliced(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            case, _, evidence = write_capture(folder)
            for name, value in (
                ("policy_state_sha256", "c" * 64),
                ("checkpoint_sha256", "d" * 64),
                ("checkpoint_id", "different-checkpoint"),
                ("training_seed", 73),
                ("version", "old-version"),
            ):
                bad = dict(evidence, **{name: value})
                write_json(folder / case.case_id / "capture.json", bad)
                with self.subTest(field=name), self.assertRaises(ValueError):
                    score_capture(folder)

    def test_recipe_body_and_its_digest_must_both_match_common_core(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            case, _, evidence = write_capture(folder)
            for consistent_digest in (False, True):
                bad = deepcopy(evidence)
                recipe = bad["realization"]["task"]["reward_recipe"]
                recipe["terms"]["tracking"]["weight"] = 3.0
                if consistent_digest:
                    bad["core_reward_recipe_sha256"] = reward_identity(recipe)
                write_json(folder / case.case_id / "capture.json", bad)
                with (
                    self.subTest(consistent_digest=consistent_digest),
                    self.assertRaises(ValueError),
                ):
                    score_capture(folder)

    def test_bad_cleanup_or_counts_retain_diagnostics_but_fail_admission(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            case, _, evidence = write_capture(folder)
            for mutate in (
                lambda row: row.update(cleanup="ERROR"),
                lambda row: row.update(observed_steps=999),
                lambda row: row["interactions"].update(session=999),
                lambda row: row["interactions"].update(cumulative=2000),
                lambda row: row["interactions"].update(uncertain=1),
            ):
                bad = deepcopy(evidence)
                mutate(bad)
                write_json(folder / case.case_id / "capture.json", bad)
                report = score_capture(folder)
                self.assertFalse(report["capture_valid"])
                self.assertFalse(report["cases"][0]["capture_valid"])
                self.assertEqual(report["cases"][0]["core_return"], 1000)

    def test_missing_recipe_cannot_be_admitted_using_only_a_copied_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            case, _, evidence = write_capture(folder)
            evidence.pop("realization")
            write_json(folder / case.case_id / "capture.json", evidence)
            report = score_capture(folder)
            self.assertFalse(report["capture_valid"])
            self.assertFalse(report["cases"][0]["capture_valid"])

    def test_empty_or_unfinished_capture_never_has_known_episode_return(self):
        for steps in (0, 15):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                write_capture(folder, steps=steps)
                report = score_capture(folder)
                self.assertFalse(report["capture_valid"])
                self.assertFalse(report["cases"][0]["capture_valid"])
                self.assertIsNone(report["cases"][0]["core_return"])
                self.assertEqual(report["cases"][0]["recorded_return"], steps)
                self.assertEqual(
                    report["aggregate"]["overall"]["unfinished_attempts"], 1
                )

    def test_trace_digest_protects_saved_measurements_and_missing_receipt_stays_missing(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            case, _, _ = write_capture(folder)
            trace = folder / case.case_id / "trace.npz"
            trace.write_bytes(trace.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Trace digest"):
                score_capture(folder)
            (folder / case.case_id / "capture.json").unlink()
            report = score_capture(folder)
            self.assertEqual(report["aggregate"]["overall"]["missing_attempts"], 1)
            self.assertFalse(report["capture_valid"])


if __name__ == "__main__":
    unittest.main()
