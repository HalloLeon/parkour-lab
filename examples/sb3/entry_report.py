"""Declare and summarize the finite two-seed forward-entry diagnostic."""

import argparse
import json
from pathlib import Path

from parkour_lab.evaluation import summarize_flat_entry
from parkour_lab.provenance import (
    dependency_identity,
    file_sha256,
    package_source_identity,
    write_json,
)

METHODS = ("ppo", "sac")
SEEDS = (1078, 2078)
POINTS = (1600000, 3200000)
VERSION = "sb3_forward_entry_screen_v1"


def _runs(root):
    return [
        dict(
            method=method,
            seed=seed,
            run_id=f"{root.name}:{method}:{seed}",
            prefix=f"{method}/seed_{seed}",
        )
        for method in METHODS
        for seed in SEEDS
    ]


def declare_screen(root):
    """Write the full diagnostic grid before collection; never replace it."""
    from .__main__ import _example_sources

    root = Path(root)
    destination = root / "diagnostic.json"
    if destination.exists():
        raise FileExistsError(destination)
    dependency = dependency_identity("stable_baselines3", "stable-baselines3")
    plan = dict(
        version=VERSION,
        diagnostic_only=True,
        command_lesson="forward-stand",
        task_id="Go2-Mixed-v1",
        num_envs=20,
        training_interactions_per_run=POINTS[-1],
        evaluation_milestones=list(POINTS),
        evaluation_suite="flat-entry",
        package_sources=package_source_identity(),
        example_sources=_example_sources(),
        learner_dependency={
            key: dependency[key] for key in ("version", "source_sha256")
        },
        runs=_runs(root),
    )
    write_json(destination, plan)
    return plan


def _read(path):
    value = json.loads(path.read_text())
    _require(type(value) is dict, f"Expected a JSON object: {path}")
    return value


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _declared_points(plan):
    budget = plan["training_interactions_per_run"]
    points = plan["evaluation_milestones"]
    _require(
        type(budget) is int and budget > 0 and budget % 1280 == 0,
        "Diagnostic budget and halfway point must be positive multiples of 640",
    )
    _require(
        type(points) is list
        and all(type(point) is int for point in points)
        and points == [budget // 2, budget],
        "Diagnostic evaluation grid must match the declared half/final budget",
    )
    return tuple(points)


def _check_training(root, plan, run, report):
    prefix = run["prefix"].replace("/", "_")
    _require(
        _read(root / f"{prefix}_train_process_exit.json")["returncode"] == 0,
        "Training process failed",
    )
    _require(report["status"] == "LEARNER_CHECK_PASSED", "Training did not complete")
    _require(
        report["cleanup"] == dict(environment="complete", application="complete"),
        "Training cleanup incomplete",
    )
    _require(
        report["interactions"]
        == dict(
            session=plan["training_interactions_per_run"],
            cumulative=plan["training_interactions_per_run"],
            uncertain=0,
        ),
        "Training accounting differs",
    )
    metadata = report["metadata"]
    for key, expected in dict(
        task_id=plan["task_id"],
        command_lesson="forward-stand",
        num_envs=20,
        seed=run["seed"],
    ).items():
        _require(metadata[key] == expected, f"Training {key} differs")
    for key in ("package_sources", "example_sources"):
        _require(report[key] == plan[key], f"Declared {key} differs")
    library = report["dependencies"]["stable_baselines3"]
    _require(
        {key: library[key] for key in plan["learner_dependency"]}
        == plan["learner_dependency"],
        "Learner dependency differs",
    )
    _require(
        report["algorithm"] == run["method"] and report["training_seed"] == run["seed"],
        "Training method/seed differs",
    )
    _require(
        report["training_information_used"] == [],
        "Undeclared learner training information",
    )


def _check_endpoint(root, plan, run, point, training, evaluation):
    prefix = run["prefix"]
    checkpoint = (
        root
        / prefix
        / "train"
        / (
            "checkpoint"
            if point == plan["training_interactions_per_run"]
            else f"checkpoints/step_{point}"
        )
    )
    receipt = _read(checkpoint / "checkpoint.json")
    expected_files = {"model.zip", "rng.pt", "progress.json"}
    if run["method"] == "sac":
        expected_files.add("replay.pkl")
    _require(
        set(receipt["files"]) == expected_files,
        "Checkpoint payload hash inventory differs",
    )
    for name, digest in receipt["files"].items():
        _require(
            Path(name).name == name and file_sha256(checkpoint / name) == digest,
            "Checkpoint payload hash differs",
        )
    digest = receipt["files"]["model.zip"]
    _require(
        receipt["checkpoint_sha256"] == digest
        and receipt["checkpoint_id"] == "sb3:" + digest,
        "Checkpoint model identity differs",
    )
    progress = _read(checkpoint / "progress.json")
    _require(
        progress["command_lesson"] == "forward-stand"
        and progress["checkpoint_interactions"] == point,
        "Checkpoint lesson/budget differs",
    )
    _require(
        progress["checkpoint_id"] == receipt["checkpoint_id"],
        "Checkpoint progress identity differs",
    )
    _require(
        progress["interactions"] == dict(session=point, cumulative=point, uncertain=0),
        "Checkpoint fresh-run accounting differs",
    )
    identity = {
        key: training["metadata"][key]
        for key in ("protocol", "task_id", "configuration_id", "seed")
    }
    _require(
        receipt["task_identity"] == identity
        and all(progress[key] == value for key, value in identity.items())
        and progress["num_envs"] == 20,
        "Checkpoint task identity differs",
    )
    _require(
        progress["run_id"] == run["run_id"] == receipt["run_id"],
        "Checkpoint run identity differs",
    )
    _require(
        receipt["training_interactions"] == point
        and receipt["training_seed"] == run["seed"]
        and receipt["algorithm"].lower() == run["method"],
        "Checkpoint method/seed/budget differs",
    )
    exit_path = (
        root / f"{prefix.replace('/', '_')}_evaluation_{point}_process_exit.json"
    )
    _require(_read(exit_path)["returncode"] == 0, "Evaluation process failed")
    _require(
        evaluation["status"] == "LEARNER_EVALUATION_PASSED"
        and evaluation["cleanup"]
        == dict(environment="not_created", application="complete"),
        "Evaluation status/cleanup differs",
    )
    _require(
        evaluation["suite"] == "flat-entry" and evaluation["full_bank"] is False,
        "Evaluation scope differs",
    )
    for key in ("package_sources", "example_sources"):
        _require(evaluation[key] == plan[key], f"Evaluation {key} differs")
    capture = root / prefix / f"evaluation_{point}" / "capture"
    manifest = _read(capture / "manifest.json")
    for key, expected in dict(
        checkpoint_id=receipt["checkpoint_id"],
        checkpoint_sha256=receipt["checkpoint_sha256"],
        training_seed=run["seed"],
    ).items():
        _require(
            manifest[key] == expected and evaluation["checkpoint"][key] == expected,
            f"Evaluation {key} link differs",
        )
    summary = summarize_flat_entry(capture)
    _require(summary["capture_valid"], "Flat diagnostic capture incomplete/invalid")
    return summary


def report_screen(root):
    """Keep every declared endpoint, including missing/failed original stages."""
    root = Path(root)
    plan = _read(root / "diagnostic.json")
    points = _declared_points(plan)
    keys = ("method", "seed", "prefix")
    declared = [{key: run[key] for key in keys} for run in plan["runs"]]
    expected = [{key: run[key] for key in keys} for run in _runs(root)]
    _require(
        plan["version"] == VERSION and declared == expected,
        "Diagnostic declaration differs",
    )
    for key, value in dict(
        diagnostic_only=True,
        command_lesson="forward-stand",
        task_id="Go2-Mixed-v1",
        num_envs=20,
        evaluation_suite="flat-entry",
    ).items():
        _require(plan[key] == value, f"Diagnostic {key} differs")
    rows = []
    for run in plan["runs"]:
        training, train_errors = {}, []
        try:
            training = _read(root / run["prefix"] / "train/report.json")
            _check_training(root, plan, run, training)
        except (OSError, ValueError, KeyError, TypeError) as error:
            train_errors.append(str(error))
        for point in points:
            row = dict(
                method=run["method"],
                seed=run["seed"],
                interactions=point,
                training_interactions=training.get("interactions"),
                training_resources=training.get("resources"),
                errors=list(train_errors),
            )
            try:
                evaluation = _read(
                    root / run["prefix"] / f"evaluation_{point}/report.json"
                )
                row["evaluation_resources"] = evaluation.get("resources")
                row["diagnostic_summary"] = _check_endpoint(
                    root, plan, run, point, training, evaluation
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                row["errors"].append(str(error))
            row["status"] = "complete" if not row["errors"] else "missing_or_failed"
            rows.append(row)
    return dict(
        version=VERSION,
        diagnostic_only=True,
        calibration_established=False,
        valid_for_comparison=False,
        full_bank=False,
        complete=all(row["status"] == "complete" for row in rows),
        prescribed_runs=4,
        prescribed_endpoints=8,
        prescribed_cases=16,
        minimum_calibration_seeds=3,
        seed_count_sufficient_for_calibration=False,
        endpoints=rows,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "report"))
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    if args.action == "declare":
        declare_screen(args.folder)
        return 0
    result = report_screen(args.folder)
    write_json(args.folder / "diagnostic_report.json", result)
    for row in result["endpoints"]:
        summary = row.get("diagnostic_summary", {})
        forward = summary.get("cases", {}).get("forward-035") or {}
        contrast = summary.get("command_contrast", {})
        print(
            f"{row['method']} seed={row['seed']} step={row['interactions']} {row['status']} travel_m={forward.get('actual_onset_forward_travel_m')} forward_minus_stand_m_s={contrast.get('forward_minus_stand_velocity_m_s')} errors={row['errors']}",
            flush=True,
        )
    print(f"Diagnostic report: {args.folder / 'diagnostic_report.json'}", flush=True)
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
