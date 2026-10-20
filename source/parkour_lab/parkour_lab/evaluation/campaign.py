"""Offline planned-run completeness and uncertainty; no learner/checkpoint loader."""

import json
import math
from pathlib import Path

from parkour_lab.benchmark.schema import FIELD_SPECS

from .capture import score_capture
from .cases import VERSIONS, case_bank
from .robustness import robustness_bank
from .scoring import aggregate_training_seeds

VERSION = "go2_campaign_v1"
NATIVE_COMPONENTS = (
    "isaaclab",
    "isaaclab_assets",
    "isaaclab_tasks",
    "isaacsim",
    "torch",
    "gymnasium",
    "numpy",
)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _read(path):
    value = json.loads(path.read_text())
    json.dumps(value, allow_nan=False)
    return value


def _declaration(data):
    if not isinstance(data, dict) or data.get("version") != VERSION:
        raise ValueError("Unsupported campaign declaration")
    seeds, milestones, methods = (
        data.get(k) for k in ("seeds", "milestones", "methods")
    )
    if (
        data.get("purpose") not in ("development", "comparison")
        or data.get("partition") not in ("validation", "test")
        or data.get("suite") not in ("core", "robustness")
        or type(data.get("case_version", "v1")) is not str
        or data.get("case_version", "v1") not in VERSIONS
        or (data.get("purpose") == "development" and data.get("partition") == "test")
        or not isinstance(seeds, list)
        or not seeds
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or len(set(seeds)) != len(seeds)
        or not isinstance(milestones, list)
        or not milestones
        or any(type(n) is not int or n <= 0 for n in milestones)
        or milestones != sorted(set(milestones))
        or not isinstance(methods, list)
        or not methods
    ):
        raise ValueError(
            "Require declared methods, unique seeds and ordered interaction milestones"
        )
    ids = set()
    for method in methods:
        if not isinstance(method, dict) or any(
            not _text(method.get(k)) for k in ("method_id", "source_id", "recipe_id")
        ):
            raise ValueError("Require method, source and recipe identities")
        if method["method_id"] in ids:
            raise ValueError("Duplicate campaign method")
        ids.add(method["method_id"])
        usage, runs = method.get("ground_truth_usage"), method.get("runs")
        if (
            not isinstance(usage, list)
            or any(not _text(v) or v not in FIELD_SPECS for v in usage)
            or len(set(usage)) != len(usage)
        ):
            raise ValueError(
                "Declare unique training-packet fields used by each method"
            )
        if not isinstance(runs, list):
            raise ValueError("Declare runs, including an empty list for missing runs")
        seen, run_ids = set(), set()
        for run in runs:
            if (
                not isinstance(run, dict)
                or type(run.get("training_seed")) is not int
                or run["training_seed"] not in seeds
                or run["training_seed"] in seen
            ):
                raise ValueError("Unknown or duplicate planned training seed")
            seen.add(run["training_seed"])
            if not _text(run.get("run_id")) or run["run_id"] in run_ids:
                raise ValueError("Each independent seed needs a distinct run identity")
            run_ids.add(run["run_id"])
            endpoints = run.get("endpoints")
            if not isinstance(endpoints, list):
                raise ValueError("Declare endpoint artifacts or an empty list")
            points = [
                e.get("interactions") if isinstance(e, dict) else None
                for e in endpoints
            ]
            if any(type(n) is not int or n not in milestones for n in points) or len(
                set(points)
            ) != len(points):
                raise ValueError("Unknown or duplicate endpoint; do not select reruns")
    return data


def _native_identity(value):
    if not isinstance(value, dict) or not set(NATIVE_COMPONENTS) <= set(value):
        raise ValueError("Capture lacks native dependency identities")
    fields = (
        "source_sha256",
        "version",
        "module_version",
        "commit",
        "diff_sha256",
        "dirty",
    )
    if any(
        not isinstance(value[n], dict) or not _text(value[n].get("source_sha256"))
        for n in NATIVE_COMPONENTS
    ):
        raise ValueError("Native dependency source identity is unavailable")
    return {
        name: {k: value[name][k] for k in fields if k in value[name]}
        for name in NATIVE_COMPONENTS
    }


def _endpoint(base, run, endpoint, cases):
    """Keep invalid or failed outcomes visible; only exact endpoints enter seed means."""
    row = dict(
        interactions=endpoint["interactions"],
        status="missing",
        resources=endpoint.get("resources"),
        capture=endpoint.get("capture"),
        progress=endpoint.get("progress"),
    )
    try:
        if endpoint.get("status", "completed") not in ("completed", "failed"):
            raise ValueError("Unknown endpoint status")
        if not all(_text(endpoint.get(k)) for k in ("capture", "progress")):
            raise ValueError("Endpoint capture/progress paths are unavailable")
        folder, progress_path = (base / endpoint[k] for k in ("capture", "progress"))
        progress = _read(progress_path)
        row["observed_interactions"] = progress.get("interactions")
        row["restart"] = progress.get("restart")
        counts = progress.get("interactions")
        if (
            not isinstance(counts, dict)
            or set(counts) != {"session", "cumulative", "uncertain"}
            or any(type(v) is not int or v < 0 for v in counts.values())
            or max(counts["session"], counts["uncertain"]) > counts["cumulative"]
            or type(progress.get("checkpoint_interactions")) is not int
        ):
            raise ValueError("Invalid exact checkpoint accounting")
        manifest = _read(folder / "manifest.json")
        score = score_capture(folder)
        row.update(
            checkpoint_id=score["checkpoint_id"],
            checkpoint_sha256=score["checkpoint_sha256"],
            capture_valid=score["capture_valid"],
            case_coverage=dict(
                observed=len([v for v in score["cases"] if v is not None]),
                prescribed=len(cases),
                missing=score["aggregate"]["missing_case_ids"],
            ),
        )
        resources = row["resources"]
        wall = resources.get("wall_time_s") if isinstance(resources, dict) else None
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
            raise ValueError(
                "Require cumulative declared training wall_time_s at this endpoint"
            )
        if (
            progress.get("version") != f"go2_checkpoint_progress_{cases[0].version}"
            or not _text(progress.get("protocol"))
            or not _text(progress.get("task_id"))
            or type(progress.get("num_envs")) is not int
            or progress["num_envs"] <= 0
            or progress.get("run_id") != run["run_id"]
            or progress.get("configuration_id") != run.get("configuration_id")
            or not _text(run.get("configuration_id"))
            or type(progress.get("seed")) is not int
            or progress["seed"] != run["training_seed"]
            or type(score["training_seed"]) is not int
            or score["training_seed"] != run["training_seed"]
            or progress.get("checkpoint_id") != score["checkpoint_id"]
            or progress.get("checkpoint_interactions") != endpoint["interactions"]
            or progress.get("interactions", {}).get("cumulative")
            != endpoint["interactions"]
            or progress["interactions"].get("uncertain") != 0
        ):
            raise ValueError(
                "Training run, checkpoint or exact cumulative budget linkage differs"
            )
        prescribed = {case.case_id: case.as_dict() for case in cases}
        observed = {v["case_id"]: v["case"] for v in score["cases"] if v is not None}
        if not score["valid_for_comparison"] or prescribed != observed:
            raise ValueError("Capture does not complete the prescribed partition/suite")
        identity = dict(
            package_sources=manifest.get("package_sources"),
            native_dependencies=_native_identity(manifest.get("native_dependencies")),
            reward=score.get("core_reward_recipe_sha256"),
            training={k: progress[k] for k in ("protocol", "task_id", "num_envs")},
        )
        if not identity["package_sources"] or not _text(identity["reward"]):
            raise ValueError("Capture source or common reward identity is unavailable")
        if endpoint.get("status") == "failed":
            row.update(
                status="failed", reason=endpoint.get("reason", "Declared failed run")
            )
            return row, None, None
        row["status"] = "complete"
        return row, score, identity
    except Exception as error:
        row.update(
            status="failed" if endpoint.get("status") == "failed" else "invalid",
            reason=endpoint.get("reason") or str(error),
        )
        if isinstance(error, FileNotFoundError) and endpoint.get("status") != "failed":
            row["status"] = "missing"
        return row, None, None


def report_campaign(path):
    """Recompute saved captures under an explicit method × seed × endpoint plan.

    Source/recipe/GT use and cumulative training resources are external declarations,
    not inferred from model files. Task progress is checkpoint-linked evidence;
    no learner archive is loaded. Completeness is not empirical calibration.
    """
    path = Path(path)
    declaration = _declaration(_read(path))
    seeds, milestones = declaration["seeds"], declaration["milestones"]
    bank = robustness_bank if declaration["suite"] == "robustness" else case_bank
    case_version = declaration.get("case_version", "v1")
    cases = bank(declaration["partition"], version=case_version)
    minimum = 3 if declaration["purpose"] == "development" else 5
    configurations = {seed: set() for seed in seeds}
    for method in declaration["methods"]:
        for run in method["runs"]:
            if _text(run.get("configuration_id")):
                configurations[run["training_seed"]].add(run["configuration_id"])
    reference, used_paths, methods = None, set(), {}
    for method in declaration["methods"]:
        runs = {run["training_seed"]: run for run in method["runs"]}
        outcomes, results = [], {point: {} for point in milestones}
        for seed in seeds:
            run = runs.get(seed)
            points = (
                {} if run is None else {e["interactions"]: e for e in run["endpoints"]}
            )
            previous_wall = -1
            for point in milestones:
                row, score, identity = (
                    dict(
                        interactions=point,
                        status="missing",
                        reason="Planned run/endpoint is absent",
                    ),
                    None,
                    None,
                )
                if point in points:
                    row, score, identity = _endpoint(
                        path.parent, run, points[point], cases
                    )
                    if score is not None:
                        capture = (path.parent / points[point]["capture"]).resolve()
                        if (
                            capture in used_paths
                            or len(configurations[seed]) != 1
                            or (reference is not None and identity != reference)
                            or row["resources"]["wall_time_s"] < previous_wall
                        ):
                            row.update(
                                status="invalid",
                                reason="Duplicate capture, incompatible configuration/source/recipe or decreasing cumulative resources",
                            )
                            score = None
                        else:
                            reference = identity
                            used_paths.add(capture)
                            previous_wall = row["resources"]["wall_time_s"]
                row.update(
                    training_seed=seed,
                    run_id=None if run is None else run["run_id"],
                    training_report=None if run is None else run.get("training_report"),
                )
                outcomes.append(row)
                results[point][seed] = [] if score is None else score["cases"]
        summaries = {
            str(point): aggregate_training_seeds(cases, values)
            for point, values in results.items()
        }
        methods[method["method_id"]] = dict(
            source_id=method["source_id"],
            recipe_id=method["recipe_id"],
            ground_truth_usage=method["ground_truth_usage"],
            endpoints=outcomes,
            learning_curve=summaries,
        )
    complete = all(
        row["status"] == "complete"
        for method in methods.values()
        for row in method["endpoints"]
    )
    return dict(
        version=VERSION,
        **({"case_version": case_version} if case_version != "v1" else {}),
        purpose=declaration["purpose"],
        partition=declaration["partition"],
        suite=declaration["suite"],
        seeds=seeds,
        milestones=milestones,
        prescribed_runs=len(seeds) * len(methods),
        prescribed_endpoints=len(seeds) * len(methods) * len(milestones),
        complete=complete,
        data_complete=complete,
        minimum_training_seeds=minimum,
        seed_count_sufficient=len(seeds) >= minimum,
        ready_for_declared_design=complete and len(seeds) >= minimum,
        calibration_established=False,
        common_evaluation_identity=reference,
        methods=methods,
        interpretation="Independent training seeds define uncertainty. Resume sessions and paired cases are not extra seeds. Missing/failed endpoints are retained; no best checkpoint or favorable case subset is selected. Source/recipe/GT use and cumulative resources are caller declarations. Data readiness does not establish empirical calibration or final ranking.",
    )
