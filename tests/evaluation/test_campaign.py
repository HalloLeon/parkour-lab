"""Offline campaigns use real saved traces, independent seeds and declared endpoints."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from parkour_lab.evaluation import case_bank, report_campaign
from parkour_lab.evaluation.capture import VERSION, reward_identity
from parkour_lab.provenance import file_sha256, write_json
from tests.support.evaluation import analytic_trace


def campaign(
    directory, *, seeds=(7, 8), milestones=(256,), methods=("a",), case_version="v2"
):
    root = Path(directory)
    cases = case_bank("validation", version=case_version)
    recipe = {"terms": {"tracking": {"weight": 1.0}}, "control_dt_s": 0.02}
    dependencies = {
        name: dict(version="1", source_sha256="d" * 64, module_path="/first/install")
        for name in (
            "isaaclab",
            "isaaclab_assets",
            "isaaclab_tasks",
            "isaacsim",
            "torch",
            "gymnasium",
            "numpy",
        )
    }
    data = dict(
        version="go2_campaign_v1",
        purpose="development",
        partition="validation",
        suite="core",
        case_version=case_version,
        seeds=list(seeds),
        milestones=list(milestones),
        methods=[],
    )
    for method in methods:
        declaration = dict(
            method_id=method,
            source_id=f"source-{method}",
            recipe_id=f"recipe-{method}",
            ground_truth_usage=[],
            runs=[],
        )
        for seed in seeds:
            run = dict(
                training_seed=seed,
                run_id=f"{method}-{seed}",
                configuration_id=f"task-{seed}",
                training_report=f"{method}-{seed}/report.json",
                endpoints=[],
            )
            for point in milestones:
                folder = root / f"{method}-{seed}-{point}"
                folder.mkdir()
                identity = dict(
                    training_seed=seed,
                    checkpoint_id=f"{method}-{seed}-{point}",
                    checkpoint_sha256="b" * 64,
                    policy_state_sha256="a" * 64,
                )
                manifest = dict(
                    version=VERSION,
                    status="COMPLETE",
                    cases=[c.as_dict() for c in cases],
                    **identity,
                    final_policy_state_sha256=identity["policy_state_sha256"],
                    core_reward_recipe_sha256=reward_identity(recipe),
                    package_sources={"native.py": "c" * 64},
                    native_dependencies=deepcopy(dependencies),
                )
                for case in cases:
                    child = folder / case.case_id
                    child.mkdir()
                    trace = analytic_trace(case, steps=1)
                    trace["reward"][:] = seeds.index(seed) + 1
                    trace["terminated"][:] = True
                    np.savez_compressed(child / "trace.npz", **trace)
                    receipt = dict(
                        version=VERSION,
                        case=case.as_dict(),
                        status="CAPTURED",
                        **identity,
                        cleanup=dict(environment="complete", application="not_created"),
                        observed_steps=1,
                        interactions=dict(session=1, cumulative=1, uncertain=0),
                        realization={"task": {"reward_recipe": recipe}},
                        core_reward_recipe_sha256=reward_identity(recipe),
                        trace_sha256=file_sha256(child / "trace.npz"),
                    )
                    write_json(child / "capture.json", receipt)
                write_json(folder / "manifest.json", manifest)
                progress = dict(
                    version=f"go2_checkpoint_progress_{case_version}",
                    protocol="go2-blind-v1",
                    task_id="Go2-Mixed-v1",
                    num_envs=20,
                    run_id=run["run_id"],
                    configuration_id=run["configuration_id"],
                    seed=seed,
                    checkpoint_id=identity["checkpoint_id"],
                    checkpoint_interactions=point,
                    interactions=dict(
                        session=(
                            point
                            if point == milestones[0]
                            else point - milestones[milestones.index(point) - 1]
                        ),
                        cumulative=point,
                        uncertain=0,
                    ),
                    restart=dict(
                        count=milestones.index(point),
                        discarded_interactions=0,
                        reason=None,
                    ),
                )
                progress_path = root / f"{method}-{seed}-{point}.json"
                write_json(progress_path, progress)
                run["endpoints"].append(
                    dict(
                        interactions=point,
                        capture=folder.name,
                        progress=progress_path.name,
                        resources=dict(wall_time_s=float(point)),
                    )
                )
            declaration["runs"].append(run)
        data["methods"].append(declaration)
    path = root / "campaign.json"
    write_json(path, data)
    return path, data


class CampaignTests(unittest.TestCase):
    def test_historical_declaration_defaults_to_v1_without_relabelling_captures(self):
        with tempfile.TemporaryDirectory() as temp:
            path, declaration = campaign(temp, seeds=(7,), case_version="v1")
            declaration.pop("case_version")
            write_json(path, declaration)
            legacy = report_campaign(path)
            self.assertTrue(legacy["data_complete"])
            self.assertEqual(
                legacy["methods"]["a"]["endpoints"][0]["case_coverage"]["prescribed"],
                114,
            )
            declaration["case_version"] = "v2"
            write_json(path, declaration)
            mixed = report_campaign(path)
            self.assertFalse(mixed["data_complete"])
            self.assertEqual(mixed["methods"]["a"]["endpoints"][0]["status"], "invalid")

    def test_three_seed_complete_pilot_meets_data_design_without_calibration_claim(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = campaign(tmp, seeds=(7, 8, 9))
            result = report_campaign(path)
        self.assertTrue(result["ready_for_declared_design"])
        self.assertFalse(result["calibration_established"])
        self.assertEqual(result["minimum_training_seeds"], 3)

    def test_offline_report_does_not_import_native_or_learner_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign.json"
            write_json(
                path,
                dict(
                    version="go2_campaign_v1",
                    purpose="development",
                    partition="validation",
                    suite="core",
                    seeds=[7, 8, 9],
                    milestones=[256],
                    methods=[
                        dict(
                            method_id="external",
                            source_id="external-source",
                            recipe_id="external-recipe",
                            ground_truth_usage=[],
                            runs=[],
                        )
                    ],
                ),
            )
            program = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==n or fullname.startswith(n+'.') for n in
               ('torch','stable_baselines3','isaaclab','isaacsim','omni','pxr')):
            raise AssertionError('Unexpected runtime import: '+fullname)
sys.meta_path.insert(0,Guard())
from parkour_lab.__main__ import main
raise SystemExit(main(['report',sys.argv[1]]))
"""
            result = subprocess.run(
                [sys.executable, "-c", program, str(path)],
                capture_output=True,
                text=True,
                timeout=15,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)["data_complete"])

    def test_complete_two_seed_resumed_curve_is_not_extra_seeds_or_final_readiness(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = campaign(tmp, milestones=(256, 512))
            result = report_campaign(path)
        self.assertTrue(result["data_complete"])
        self.assertFalse(result["seed_count_sufficient"])
        self.assertFalse(result["ready_for_declared_design"])
        self.assertFalse(result["calibration_established"])
        self.assertEqual(result["prescribed_runs"], 2)
        self.assertEqual(result["prescribed_endpoints"], 4)
        method = result["methods"]["a"]
        for point in ("256", "512"):
            statistics = method["learning_curve"][point]
            self.assertEqual(statistics["prescribed_training_seeds"], 2)
            self.assertAlmostEqual(statistics["core_return_mean"], 1.5)
            self.assertAlmostEqual(statistics["core_return_standard_error"], 0.5)
        last = method["endpoints"][-1]
        self.assertEqual(last["observed_interactions"]["cumulative"], 512)
        self.assertEqual(last["restart"]["count"], 1)
        self.assertEqual(last["training_report"], "a-8/report.json")

    def test_missing_or_failed_planned_seed_is_not_omitted_or_zero_filled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, data = campaign(tmp)
            missing = data["methods"][0]["runs"].pop()
            write_json(path, data)
            result = report_campaign(path)
            self.assertFalse(result["complete"])
            stats = result["methods"]["a"]["learning_curve"]["256"]
            self.assertEqual(stats["prescribed_training_seeds"], 2)
            self.assertEqual(stats["complete_training_seeds"], 1)
            self.assertIsNone(stats["core_return_mean"])
            failure = stats["group_statistics"]["family"]["Flat"][
                "physical_failure_fraction_prescribed"
            ]
            self.assertIsNone(failure["mean"])
            self.assertEqual(failure["available_training_seeds"], 1)
            self.assertEqual(
                result["methods"]["a"]["endpoints"][-1]["status"], "missing"
            )
            missing["endpoints"][0].update(status="failed", reason="optimizer failed")
            data["methods"][0]["runs"].append(missing)
            write_json(path, data)
            failed = report_campaign(path)["methods"]["a"]["endpoints"][-1]
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["reason"], "optimizer failed")

    def test_counter_linkage_and_missing_identity_or_resources_block_with_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, data = campaign(tmp, seeds=(7,))
            endpoint = data["methods"][0]["runs"][0]["endpoints"][0]
            progress_path = Path(tmp) / endpoint["progress"]
            original = json.loads(progress_path.read_text())
            for mutate in (
                lambda r: r.update(seed=8),
                lambda r: r.update(checkpoint_id="other"),
                lambda r: r.update(checkpoint_interactions=True),
                lambda r: r["interactions"].update(cumulative=512),
                lambda r: r["interactions"].update(uncertain=1),
                lambda r: r["interactions"].update(session=True),
                lambda r: r["interactions"].update(session=257),
            ):
                changed = deepcopy(original)
                mutate(changed)
                write_json(progress_path, changed)
                result = report_campaign(path)
                self.assertFalse(result["data_complete"])
                self.assertEqual(
                    result["methods"]["a"]["endpoints"][0]["status"], "invalid"
                )
            write_json(progress_path, original)
            endpoint["resources"] = None
            write_json(path, data)
            self.assertFalse(report_campaign(path)["data_complete"])
            endpoint["resources"] = dict(wall_time_s=256.0)
            write_json(path, data)
            manifest_path = Path(tmp) / endpoint["capture"] / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest.pop("native_dependencies")
            write_json(manifest_path, manifest)
            row = report_campaign(path)["methods"]["a"]["endpoints"][0]
            self.assertIn("native dependency", row["reason"])

    def test_machine_paths_are_not_dependency_changes_but_source_changes_are(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, data = campaign(tmp)
            folder = (
                Path(tmp) / data["methods"][0]["runs"][1]["endpoints"][0]["capture"]
            )
            manifest_path = folder / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["native_dependencies"]["torch"]["module_path"] = "/another/machine"
            write_json(manifest_path, manifest)
            self.assertTrue(report_campaign(path)["data_complete"])
            manifest["native_dependencies"]["torch"]["source_sha256"] = "e" * 64
            write_json(manifest_path, manifest)
            self.assertFalse(report_campaign(path)["data_complete"])

    def test_different_method_sources_are_allowed_but_task_configs_cannot_differ(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, data = campaign(tmp, seeds=(7,), methods=("a", "b"))
            self.assertTrue(report_campaign(path)["data_complete"])
            run = data["methods"][1]["runs"][0]
            original = deepcopy(run)
            first = data["methods"][0]["runs"][0]
            run.update(run_id=first["run_id"], endpoints=deepcopy(first["endpoints"]))
            write_json(path, data)
            duplicated = report_campaign(path)
            self.assertFalse(duplicated["data_complete"])
            self.assertIn(
                "Duplicate capture",
                duplicated["methods"]["b"]["endpoints"][0]["reason"],
            )
            run.clear()
            run.update(original)
            run["configuration_id"] = "changed-task"
            progress_path = Path(tmp) / run["endpoints"][0]["progress"]
            progress = json.loads(progress_path.read_text())
            progress["configuration_id"] = run["configuration_id"]
            write_json(progress_path, progress)
            write_json(path, data)
            result = report_campaign(path)
            self.assertFalse(result["data_complete"])
            self.assertTrue(
                all(
                    row["status"] == "invalid"
                    for method in result["methods"].values()
                    for row in method["endpoints"]
                )
            )

    def test_duplicate_endpoints_unknown_gt_and_development_test_tuning_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, data = campaign(tmp, seeds=(7,))
            for mutate in (
                lambda d: d.update(partition="test"),
                lambda d: d["methods"][0].update(ground_truth_usage=["future_fault"]),
                lambda d: d["methods"][0]["runs"][0]["endpoints"].append(
                    deepcopy(d["methods"][0]["runs"][0]["endpoints"][0])
                ),
            ):
                invalid = deepcopy(data)
                mutate(invalid)
                write_json(path, invalid)
                with self.assertRaises(ValueError):
                    report_campaign(path)


if __name__ == "__main__":
    unittest.main()
