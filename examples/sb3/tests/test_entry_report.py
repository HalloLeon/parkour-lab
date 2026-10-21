"""Partial screening evidence must retain missing stages and original failures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from examples.sb3 import entry_report as entry


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class EntryReportTests(unittest.TestCase):
    def test_new_declaration_has_fixed_equal_budget_and_two_evaluation_points(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                entry,
                "dependency_identity",
                return_value={
                    "version": "inert",
                    "source_sha256": "inert",
                },
            ):
                plan = entry.declare_screen(root)
            self.assertEqual(plan["training_interactions_per_run"], 3200000)
            self.assertEqual(plan["evaluation_milestones"], [1600000, 3200000])
            self.assertTrue(
                all(point % 640 == 0 for point in plan["evaluation_milestones"])
            )
            self.assertEqual(len(plan["runs"]), 4)
            self.assertEqual({row["seed"] for row in plan["runs"]}, {1078, 2078})
            self.assertTrue(plan["diagnostic_only"])

    def test_declared_old_and_new_grids_use_their_own_budget_and_checkpoint_paths(self):
        for budget in (1024000, 3200000):
            with (
                self.subTest(budget=budget),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                plan = self.prepare(root, budget=budget)
                for run in plan["runs"]:
                    for point in plan["evaluation_milestones"]:
                        self.endpoint(root, plan, run, point)
                with patch.object(
                    entry, "summarize_flat_entry", return_value={"capture_valid": True}
                ) as summarize:
                    report = entry.report_screen(root)
                self.assertTrue(report["complete"])
                self.assertEqual(summarize.call_count, 8)
                self.assertEqual(
                    {row["interactions"] for row in report["endpoints"]},
                    {budget // 2, budget},
                )
                self.assertTrue(
                    all(
                        row["training_interactions"]["cumulative"] == budget
                        for row in report["endpoints"]
                    )
                )
                self.assertFalse(report["valid_for_comparison"])

    def test_mismatched_or_unaligned_declared_budget_is_rejected_before_artifacts(self):
        changes = (
            {"training_interactions_per_run": True},
            {"training_interactions_per_run": 0},
            {
                "training_interactions_per_run": 3200640,
                "evaluation_milestones": [1600320, 3200640],
            },
            {"training_interactions_per_run": 640, "evaluation_milestones": [320, 640]},
            {"evaluation_milestones": [512000, 1024000]},
            {"evaluation_milestones": [1600000.0, 3200000]},
        )
        for change in changes:
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                plan = self.prepare(root)
                plan.update(change)
                write(root / "diagnostic.json", plan)
                with patch.object(entry, "_check_training") as training:
                    with self.assertRaisesRegex(ValueError, "budget|grid"):
                        entry.report_screen(root)
                training.assert_not_called()

    def prepare(self, root, *, budget=3200000):
        points = (budget // 2, budget)
        plan = dict(
            version=entry.VERSION,
            runs=entry._runs(root),
            evaluation_milestones=list(points),
            diagnostic_only=True,
            command_lesson="forward-stand",
            task_id="Go2-Mixed-v1",
            num_envs=20,
            training_interactions_per_run=budget,
            evaluation_suite="flat-entry",
            package_sources={"core": "unchanged"},
            example_sources={"example": "unchanged"},
            learner_dependency=dict(version="2.8.0", source_sha256="library"),
        )
        write(root / "diagnostic.json", plan)
        for run in plan["runs"]:
            report = dict(
                status="LEARNER_CHECK_PASSED",
                cleanup=dict(environment="complete", application="complete"),
                interactions=dict(session=budget, cumulative=budget, uncertain=0),
                metadata=dict(
                    protocol="test",
                    configuration_id="configuration",
                    task_id="Go2-Mixed-v1",
                    command_lesson="forward-stand",
                    num_envs=20,
                    seed=run["seed"],
                ),
                package_sources=plan["package_sources"],
                example_sources=plan["example_sources"],
                dependencies=dict(stable_baselines3=plan["learner_dependency"]),
                algorithm=run["method"],
                training_seed=run["seed"],
                training_information_used=[],
            )
            write(root / run["prefix"] / "train/report.json", report)
            write(
                root / f"{run['prefix'].replace('/', '_')}_train_process_exit.json",
                dict(returncode=0),
            )
            for point in points:
                write(
                    root / run["prefix"] / f"evaluation_{point}/report.json",
                    dict(resources=dict(wall_time_s=1)),
                )
        return plan

    def endpoint(self, root, plan, run, point=None):
        """Inert bytes exercise all identity checks without loading learner data."""
        budget = plan["training_interactions_per_run"]
        point = budget if point is None else point
        relative = "checkpoint" if point == budget else f"checkpoints/step_{point}"
        checkpoint = root / run["prefix"] / "train" / relative
        checkpoint.mkdir(parents=True)
        for name in ("model.zip", "rng.pt") + (
            ("replay.pkl",) if run["method"] == "sac" else ()
        ):
            (checkpoint / name).write_bytes(b"inert; never deserialize")
        digest = entry.file_sha256(checkpoint / "model.zip")
        identity = dict(
            protocol="test",
            task_id=plan["task_id"],
            configuration_id="configuration",
            seed=run["seed"],
        )
        progress = dict(
            **identity,
            num_envs=20,
            command_lesson="forward-stand",
            checkpoint_id="sb3:" + digest,
            checkpoint_interactions=point,
            run_id=run["run_id"],
            interactions=dict(session=point, cumulative=point, uncertain=0),
        )
        write(checkpoint / "progress.json", progress)
        receipt = dict(
            checkpoint_id="sb3:" + digest,
            checkpoint_sha256=digest,
            run_id=run["run_id"],
            training_seed=run["seed"],
            training_interactions=point,
            algorithm=run["method"].upper(),
            task_identity=identity,
            files={path.name: entry.file_sha256(path) for path in checkpoint.iterdir()},
        )
        write(checkpoint / "checkpoint.json", receipt)
        link = {
            key: receipt[key]
            for key in ("checkpoint_id", "checkpoint_sha256", "training_seed")
        }
        evaluation = dict(
            status="LEARNER_EVALUATION_PASSED",
            cleanup=dict(environment="not_created", application="complete"),
            suite="flat-entry",
            full_bank=False,
            package_sources=plan["package_sources"],
            example_sources=plan["example_sources"],
            checkpoint=link,
        )
        write(root / run["prefix"] / f"evaluation_{point}/capture/manifest.json", link)
        write(root / run["prefix"] / f"evaluation_{point}/report.json", evaluation)
        write(
            root
            / f"{run['prefix'].replace('/', '_')}_evaluation_{point}_process_exit.json",
            dict(returncode=0),
        )
        training = entry._read(root / run["prefix"] / "train/report.json")
        return checkpoint, training, evaluation

    def test_complete_collection_with_physical_failure_remains_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root)
            summary = dict(cases={"forward-035": dict(physical_failure=True)})
            with patch.object(entry, "_check_endpoint", return_value=summary):
                report = entry.report_screen(root)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["endpoints"]), 8)
            self.assertFalse(report["valid_for_comparison"])
            self.assertFalse(report["calibration_established"])
            self.assertFalse(report["seed_count_sufficient_for_calibration"])
            self.assertTrue(
                report["endpoints"][0]["diagnostic_summary"]["cases"]["forward-035"][
                    "physical_failure"
                ]
            )

    def test_failed_original_training_and_missing_endpoint_are_never_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = self.prepare(root)
            first = plan["runs"][0]
            write(
                root / f"{first['prefix'].replace('/', '_')}_train_process_exit.json",
                dict(returncode=2),
            )
            (
                root / plan["runs"][-1]["prefix"] / "evaluation_3200000/report.json"
            ).unlink()
            with patch.object(entry, "_check_endpoint", return_value={}):
                report = entry.report_screen(root)
            self.assertFalse(report["complete"])
            self.assertEqual(len(report["endpoints"]), 8)
            failed = [row for row in report["endpoints"] if row["status"] != "complete"]
            self.assertEqual(len(failed), 3)
            self.assertIn("Training process failed", failed[0]["errors"])
            self.assertEqual(failed[0]["training_interactions"]["cumulative"], 3200000)

    def test_checkpoint_payload_corruption_is_rejected_without_loading_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = entry._runs(root)[0]
            checkpoint = root / run["prefix"] / "train/checkpoint"
            write(checkpoint / "checkpoint.json", dict(files={"model.zip": "wrong"}))
            (checkpoint / "model.zip").write_bytes(b"not a model; never deserialize")
            with self.assertRaisesRegex(ValueError, "payload hash"):
                entry._check_endpoint(
                    root,
                    {"training_interactions_per_run": 3200000},
                    run,
                    3200000,
                    {},
                    {},
                )

    def test_moved_packet_preserves_original_declared_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "original"
            self.prepare(root)
            moved = root.with_name("moved")
            root.rename(moved)
            with patch.object(entry, "_check_endpoint", return_value={}):
                self.assertTrue(entry.report_screen(moved)["complete"])

    def test_both_checkpoint_inventories_and_links_verify_without_deserialization(self):
        for method in entry.METHODS:
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                plan = self.prepare(root)
                run = next(run for run in plan["runs"] if run["method"] == method)
                _, training, evaluation = self.endpoint(root, plan, run)
                with patch.object(
                    entry, "summarize_flat_entry", return_value=dict(capture_valid=True)
                ) as summarize:
                    self.assertTrue(
                        entry._check_endpoint(
                            root, plan, run, 3200000, training, evaluation
                        )["capture_valid"]
                    )
                summarize.assert_called_once()

    def test_consistently_relabelled_or_miscounted_progress_is_rejected(self):
        for change in ("model", "progress", "accounting", "task", "inventory"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                plan = self.prepare(root)
                run = plan["runs"][0]
                checkpoint, training, evaluation = self.endpoint(root, plan, run)
                receipt = entry._read(checkpoint / "checkpoint.json")
                progress = entry._read(checkpoint / "progress.json")
                if change == "model":
                    receipt.update(checkpoint_id="sb3:other", checkpoint_sha256="other")
                    evaluation["checkpoint"].update(
                        checkpoint_id="sb3:other", checkpoint_sha256="other"
                    )
                    write(
                        root
                        / run["prefix"]
                        / "evaluation_3200000/capture/manifest.json",
                        evaluation["checkpoint"],
                    )
                elif change == "progress":
                    progress["checkpoint_id"] = "sb3:other"
                elif change == "accounting":
                    progress["interactions"]["uncertain"] = 1
                elif change == "task":
                    receipt["task_identity"]["seed"] += 1
                    progress["seed"] += 1
                else:
                    receipt["files"].pop("rng.pt")
                write(checkpoint / "progress.json", progress)
                receipt["files"]["progress.json"] = entry.file_sha256(
                    checkpoint / "progress.json"
                )
                write(checkpoint / "checkpoint.json", receipt)
                with patch.object(entry, "summarize_flat_entry") as summarize:
                    with self.assertRaises(ValueError):
                        entry._check_endpoint(
                            root, plan, run, 3200000, training, evaluation
                        )
                summarize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
