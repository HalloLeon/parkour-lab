"""Bounded approach trials and optional paired raised-start controls; never promotion.

Run with the Isaac Lab Python from the repository root. Children are sequential,
without execution timeouts. Each owns a fresh directory and process receipt.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

from .operator_roa_checkpoint import (
    _validate_completion,
    export_roa_actor,
    verify_source_files,
)
from .operator_roa_pilot import ENVIRONMENT_STAGES, load_environment_source
from .operator_roa_evaluation import COMMAND_TAPE, command_tape_sha256
from . import operator_step_approach, operator_step_field, operator_step_support
from .operator_traversal_probe import COMMAND_TAPE as TRAVERSAL_TAPE
from .operator_train import file_sha256, recurrent_training_identity

SCREENS = {
    "higher_2076": (
        160,
        2076,
        [
            "--terrain-suite",
            "step_fields",
            "--step-field-version",
            "operator_step_field_v1",
        ],
    ),
    "ladder_1045": (
        80,
        1045,
        ["--terrain-suite", "traversal", "--traversal-layout", "step_ladder"],
    ),
    "standard": (
        80,
        1044,
        ["--terrain-suite", "traversal", "--traversal-layout", "standard"],
    ),
    "retention": (80, 1043, ["--terrain-suite", "procedural"]),
}
INITIAL_KEYS = (
    "initial_observation_sha256",
    "initial_dynamics_sha256",
    "initial_root_state_sha256",
    "initial_joint_pos_sha256",
    "initial_joint_vel_sha256",
    "terrain_assignment",
    "command_tape_sha256",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def read(path):
    return json.loads(path.read_text())


def identity_sha256(identity):
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def trace_check(path, expected_hash):
    require(file_sha256(path) == expected_hash, f"Trace changed: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        require(
            all(
                np.isfinite(arrays[key]).all()
                for key in arrays.files
                if arrays[key].dtype.kind in "biufc"
            ),
            f"Nonfinite trace: {path}",
        )


def seed_admission(seed, *, paired):
    """CPU XY admission only; the children still verify actual imported geometry.

    Cell levels and reset XY do not depend on difficulty. Generate each variant
    at a representative difficulty, then check every row's world-coordinate pool.
    """
    tiles = []
    for variant in range(12, 16):
        _, _, tile = operator_step_field.build_surface(0.35, seed=seed, variant=variant)
        tiles.extend(dict(tile, row=row) for row in range(3))
    preview = dict(seed=seed, tiles=tiles)
    table = operator_step_approach.candidates(preview)
    result = dict(seed=seed, directed_approaches=len(table))
    if paired:
        clearance = []
        for tile in operator_step_support._tiles(preview):
            origin = np.array([16 * (tile["row"] - 1), 16 * (tile["variant"] - 9.5)])
            positions = np.asarray(tile["positions_world_m"])[..., :2] - origin
            clearance.append(
                float(
                    (
                        operator_step_approach.WORKSPACE_HALF
                        - np.abs(positions)
                        - operator_step_approach.FOOTPRINT_RADIUS
                    ).min()
                )
            )
        require(
            min(clearance) > 0, f"Raised-start control outside workspace: seed {seed}"
        )
        result["raised_start_minimum_workspace_clearance_m"] = min(clearance)
    return result


def train_check(run, expected, *, seed=1063, layout="contact_approach", source=None):
    report, protocol = read(run / "report.json"), read(run / "training_protocol.json")
    stage = ENVIRONMENT_STAGES[layout]
    approach = layout == "contact_approach"
    require(
        layout in ("contact_approach", "contact_acquire")
        and protocol["environment_layout"] == layout
        and protocol["version"] == stage.version
        and protocol["seed"] == seed
        and protocol["num_envs"] == 160
        and protocol["history_interval"] == 5
        and protocol["cycles"] == 1000
        and report["native_step_field_geometry"]["seed"] == seed
        and all(
            report[key]["seed"] == seed + 1000
            for key in ("evaluation_before", "evaluation_after")
        ),
        "Training case seed/layout/budget changed",
    )
    require(
        source is None or protocol["learning_source"] == source,
        "Training did not start from the retained parent",
    )
    require(
        (
            (
                protocol.get("approach_recipe") == operator_step_approach.recipe()
                and "support_reset_recipe" not in protocol
            )
            if approach
            else (
                protocol.get("support_reset_recipe") == operator_step_support.recipe()
                and "approach_recipe" not in protocol
            )
        ),
        "Training occupancy recipe changed; approaches require workspace-admitted v2",
    )
    _validate_completion(report, stage.status, 38600, 160)
    require(
        protocol["source_identity"] == expected, "Training runtime/reference changed"
    )
    require(
        report["cleanup"]["environment"] == "complete",
        "Training environment cleanup failed",
    )
    require(
        report["ppo_updates_completed"] == 1000
        and report["adaptation_optimizer_steps"] == 3200,
        "Incomplete learning budget",
    )
    require(
        report["ppo_options"]["entropy_coef"] == 0.01
        and "entropy_ablation" not in protocol,
        "Unrequested noise ablation",
    )
    require(
        all(report["evaluation_initial_conditions_match"].values()),
        "Training before/after starts differ",
    )
    telemetry = report["training_telemetry"]
    require(
        telemetry["complete"]
        and not telemetry["pending_step"]
        and telemetry["incomplete_block_decisions"] == 0,
        "Incomplete training trace",
    )
    require(
        [item["end_cycle"] for item in telemetry["files"]]
        == [5, *range(100, 1001, 100)],
        "Missing training windows",
    )
    holds = 0
    for item in telemetry["files"]:
        path = run / "training_telemetry" / f"block_{item['end_cycle']:06d}.npz"
        require(
            item["path"] == str(path.relative_to(run)), "Unexpected training trace path"
        )
        trace_check(path, item["sha256"])
        with np.load(path, allow_pickle=False) as arrays:
            require(
                arrays["reward_total"].shape == (184, 160)
                and arrays["ppo_return"].shape == (5, 24, 160, 1),
                "Wrong training trace dimensions",
            )
            if not approach:
                continue
            holding = arrays["approach_hold_pre"]
            command = arrays["command_b_pre"][holding]
            require(
                holding.dtype == np.bool_ and holding.shape == (184, 160),
                "Invalid approach hold mask",
            )
            require(
                (arrays["approach_edge_id"][holding] >= 0).all(),
                "Hold without an approach episode",
            )
            require(
                (command[:, 1:] == 0).all()
                and ((command[:, 0] >= 0.3) & (command[:, 0] <= 0.5)).all(),
                "Approach command changed",
            )
            holds += int(holding.sum())
    require(
        not approach or holds > 0,
        "The approach mechanism never delivered a sampled training command",
    )
    checkpoint = run / "learning_1000.pt"
    require(
        file_sha256(checkpoint) == report["checkpoint_sha256"],
        "Training checkpoint changed",
    )
    return checkpoint


def screen_check(run, exported, expected, name, route, baseline=None):
    report, protocol = read(run / "report.json"), read(run / "evaluation_protocol.json")
    diagnostic = None if route == "causal" else "privileged_latent"
    status = (
        "ROA_EXPORTED_SCREEN_COMPLETED_NOT_QUALIFIED"
        if diagnostic is None
        else "ROA_INPUT_DIAGNOSTIC_COMPLETED_NOT_DEPLOYABLE"
    )
    require(
        report["status"] == status
        or (
            report["status"] == "SESSION_COMPLETED_CLEANUP_PENDING"
            and report.get("session_status") == status
        ),
        "Incomplete frozen screen",
    )
    require(
        report["learning_updates"] == 0
        and report["exit_allowed"] is False
        and report["cleanup"]["environment"] == "complete",
        "Invalid screen scope/cleanup",
    )
    require(
        protocol["source_identity"] == expected
        and protocol.get("diagnostic_input") == diagnostic,
        "Wrong screen runtime/route",
    )
    require(
        protocol["checkpoint_source"] == exported["source"]
        and protocol["controller"]["artifact_sha256"] == exported["sha256"],
        "Wrong screen weights",
    )
    n, seed, _ = SCREENS[name]
    ev, motor = report["evaluation"], report["motor_delivery"]
    field, retention = name == "higher_2076", name == "retention"
    suite = "step_fields" if field else "procedural" if retention else "traversal"
    layout = (
        None
        if field or retention
        else "step_ladder" if name == "ladder_1045" else "standard"
    )
    difficulty = (
        list(operator_step_field.DIFFICULTY)
        if field
        else [0.05, 0.15] if retention else None
    )
    require(
        protocol["terrain_suite"] == suite
        and protocol.get("traversal_layout") == layout
        and protocol["difficulty_range"] == difficulty
        and protocol.get("step_field_geometry")
        == (operator_step_field.envelope() if field else None)
        and protocol["traversal_geometry"] == report.get("native_geometry")
        and protocol["seed"] == protocol["terrain_generator_seed"] == seed
        and protocol["num_envs"] == n,
        "Screen geometry or allocation differs from the requested fixture",
    )
    tape = (
        operator_step_field.SCREEN_COMMAND_TAPE
        if field
        else COMMAND_TAPE if retention else TRAVERSAL_TAPE
    )
    tape_hash = command_tape_sha256(tape)
    require(
        all(
            value["command_tape_sha256"]
            == command_tape_sha256(value["command_tape"])
            == tape_hash
            for value in (protocol, ev)
        )
        and report.get("diagnostic_input") == ev.get("diagnostic_input") == diagnostic,
        "Screen tape or executed input route changed",
    )
    require(
        ev["complete_tape"]
        and ev["control_steps"] == 900
        and ev["num_envs"] == n
        and ev["seed"] == seed
        and ev["environment_transitions"] == 900 * n,
        "Screen tape or allocation changed",
    )
    require(
        ev["policy_state_sha256_before"]
        == ev["policy_state_sha256_after"]
        == exported["source"]["policy_state_sha256"],
        "Frozen policy changed",
    )
    parity = ev["controller_parity"]
    parity_status = (
        "EXACT_ON_EXECUTED_DECISIONS"
        if diagnostic is None
        else "EXACT_SHADOW_NOT_APPLIED"
    )
    require(
        parity["status"] == parity_status
        and parity["decision_count"] == 900
        and parity["state_sha256_before"] == parity["state_sha256_after"],
        "Controller parity failed",
    )
    require(
        all(
            motor[key] == 900
            for key in (
                "encoded_steps",
                "native_hint_steps",
                "verified_delivery_steps",
                "native_step_returns",
            )
        ),
        "Incomplete motor delivery",
    )
    require(
        not motor["faulted"]
        and not motor["pending_delivery"]
        and motor["target_only_steps"] == 0
        and motor["native_verified_rows"] + motor["excluded_terminal_rows"] == 900 * n,
        "Invalid motor verification",
    )
    if baseline is not None:
        require(
            all(ev[key] == baseline["evaluation"][key] for key in INITIAL_KEYS),
            "Unmatched frozen initial conditions",
        )
        require(
            all(
                report.get(key) == baseline.get(key)
                for key in ("native_geometry", "native_step_field_geometry")
            ),
            "Unmatched frozen geometry",
        )
    if name != "retention":
        traces = [("input_diagnostic_trace.npz", ev["input_diagnostic"]["trace"])]
        spatial = "field_exposure" if name == "higher_2076" else "traversal"
        traces.append(
            (
                (
                    "field_trace.npz"
                    if spatial == "field_exposure"
                    else "traversal_trace.npz"
                ),
                report[spatial]["trace"],
            )
        )
        for filename, receipt in traces:
            trace_check(run / filename, receipt["sha256"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument(
        "--identity-sha256",
        required=True,
        help="Reviewed physical-reference/runtime identity, excluding docs/tests",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("logs/rsl_rl/go2_operator_refinement"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[1063])
    parser.add_argument(
        "--paired-control",
        action="store_true",
        help="Also train the existing raised-start/free-command recipe for each seed",
    )
    args = parser.parse_args(argv)
    if any(seed < 0 for seed in args.training_seeds) or len(
        set(args.training_seeds)
    ) != len(args.training_seeds):
        parser.error("training-seeds must be distinct nonnegative integers")
    args.reference, args.parent = args.reference.resolve(
        strict=True
    ), args.parent.resolve(strict=True)
    expected = recurrent_training_identity(args.reference)

    def preflight():
        require(args.cpu_threads > 0, "cpu-threads must be positive")
        require(
            identity_sha256(recurrent_training_identity(args.reference))
            == args.identity_sha256,
            "Runtime/reference changed; sync all reviewed scripts/source files",
        )
        require(
            file_sha256(args.parent) == args.parent_sha256, "Retained parent changed"
        )

    preflight()
    _, source = load_environment_source(
        args.parent,
        expected["physical_reference"],
        args.training_seeds[0],
        layout="contact_approach",
    )
    require(
        source["training_seed"] not in args.training_seeds,
        "Training seeds must differ from the retained parent's seed",
    )
    admissions = [
        seed_admission(seed, paired=args.paired_control) for seed in args.training_seeds
    ]
    layouts = (
        ("contact_approach", "contact_acquire")
        if args.paired_control
        else ("contact_approach",)
    )
    sweep = len(args.training_seeds) > 1 or args.paired_control
    trials = [
        dict(
            seed=seed,
            layout=layout,
            label=f"seed{seed}/{layout}" if sweep else "candidate",
        )
        for seed in args.training_seeds
        for layout in layouts
    ]
    require(
        not any(
            args.output_parent.resolve().is_relative_to(Path(path).parent)
            for path in source["files"]
        ),
        "Output must be outside immutable source runs",
    )
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(prefix="roa_step_approach_", dir=args.output_parent)
    ).resolve()
    print(f"Output: {output}", flush=True)
    summary = dict(
        status="STARTED_NOT_QUALIFIED",
        exit_allowed=False,
        source_identity=expected,
        parent=source,
        trials=trials,
        screens={},
    )
    write(
        output / "batch_plan.json",
        dict(
            summary,
            seed_admission=admissions,
            approach_recipe=operator_step_approach.recipe(),
            control_recipe=(
                operator_step_support.recipe() if args.paired_control else None
            ),
            updates=1000,
            screens=SCREENS,
            routes=["causal", "privileged_latent"],
            scope="Prespecified development seeds, each from the same immutable parent. Optional paired bundled occupancy recipes, not a single-factor intervention or independent parent replications. Parent screens once; no ranking, chaining, retries or promotion. Analytic seed admission is not native contact certification.",
        ),
    )
    started = time.monotonic()

    def child(folder, module, arguments):
        preflight()
        verify_source_files(source)
        folder.mkdir(parents=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            module,
            str(args.reference),
            *map(str, arguments),
            "--device",
            args.device,
            "--cpu-threads",
            str(args.cpu_threads),
            "--output-parent",
            str(folder),
        ]
        print(
            f"Running {folder.relative_to(output)}; log: {folder / 'console.log'}",
            flush=True,
        )
        beginning, code, error = time.monotonic(), None, None
        try:
            with (folder / "console.log").open("x") as log:
                code = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=False
                ).returncode
        except BaseException as exc:
            error = repr(exc)
            raise
        finally:
            write(
                folder / "process_exit.json",
                dict(
                    command=command,
                    returncode=code,
                    interrupted=code in (-2, -15, 130, 143) or error is not None,
                    error=error,
                    wall_seconds=time.monotonic() - beginning,
                ),
            )
        require(code == 0, f"Child failed; inspect {folder / 'console.log'}")
        preflight()
        reports = list(folder.glob("operator_roa_*/report.json"))
        require(len(reports) == 1, f"Missing or ambiguous child report: {folder}")
        return reports[0].parent

    controls, paired_starts = {}, {}

    def screens(label, weights):
        case = output / label
        case.mkdir(parents=True, exist_ok=True)
        exported = export_roa_actor(weights, case / "actor.pt")
        write(case / "export.json", exported)
        summary["screens"][label] = {}
        for name, (n, seed, options) in SCREENS.items():
            for route in (
                ["causal"] if name == "retention" else ["causal", "privileged_latent"]
            ):
                extra = [] if route == "causal" else ["--diagnostic-input", route]
                run = child(
                    case / name / route,
                    "scripts.rsl_rl.operator_roa_screen",
                    [
                        "--checkpoint",
                        weights,
                        "--controller-artifact",
                        case / "actor.pt",
                        "--num-envs",
                        n,
                        "--seed",
                        seed,
                        *options,
                        *extra,
                    ],
                )
                report = screen_check(
                    run, exported, expected, name, route, controls.get(name)
                )
                if label == "parent":
                    controls.setdefault(name, report)
                summary["screens"][label][f"{name}/{route}"] = dict(
                    report=str(run / "report.json"),
                    sha256=file_sha256(run / "report.json"),
                    first_episode=report["evaluation"]["first_episode"],
                    tracking=report["evaluation"]["tracking"]["first_episode"],
                    behavior_accepted=False,
                )

    try:
        for trial in trials:
            seed, layout, label = trial["seed"], trial["layout"], trial["label"]
            case = output / label if sweep else output
            run = child(
                case / "training",
                "scripts.rsl_rl.operator_roa_pilot",
                [
                    "--environment-checkpoint",
                    args.parent,
                    "--environment-layout",
                    layout,
                    "--regularization",
                    "off",
                    "--orientation-weight",
                    "-2.5",
                    "--stumble-weight",
                    "0",
                    "--learning-updates",
                    "1000",
                    "--history-interval",
                    "5",
                    "--num-envs",
                    "160",
                    "--seed",
                    seed,
                    "--training-telemetry",
                ],
            )
            checkpoint = train_check(
                run, expected, seed=seed, layout=layout, source=source
            )
            report = read(run / "report.json")
            start = {
                key: report["evaluation_before"][key]
                for key in (*INITIAL_KEYS, "policy_state_sha256_before")
            }
            start["native_step_field_geometry"] = report["native_step_field_geometry"]
            require(
                start == paired_starts.setdefault(seed, start),
                f"Paired training initial conditions differ: seed {seed}",
            )
            trial.update(
                checkpoint=str(checkpoint),
                training_report=str(run / "report.json"),
                training_report_sha256=file_sha256(run / "report.json"),
            )
            if not controls:
                screens("parent", args.parent)
            screens(label, checkpoint)
            trial["status"] = "COMPLETE_NOT_PROMOTED"
        summary["status"] = "COMPLETE_NOT_PROMOTED"
    except BaseException as exc:
        summary.update(status="FAILED_OR_INTERRUPTED_NOT_QUALIFIED", error=repr(exc))
        raise
    finally:
        summary["wall_seconds"] = time.monotonic() - started
        write(output / "batch_check.json", summary)
        print(
            f"Return the complete folder, including partial outputs on failure: {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
