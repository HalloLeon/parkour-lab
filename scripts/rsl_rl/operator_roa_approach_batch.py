"""Configurable ROA development experiments; never qualification or promotion.

Run with the Isaac Lab Python from the repository root. Children are sequential,
without execution timeouts. Each owns a fresh directory and process receipt.
Use --config FILE (or - for JSON on stdin) for explicit training cases and screen
selection. The original CLI remains available for historical run recovery.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

from .operator_roa_checkpoint import (
    _validate_completion,
    export_roa_actor,
    load_completed_checkpoint,
    load_screen_checkpoint,
    verify_source_files,
)
from .operator_roa_pilot import (
    ENVIRONMENT_STAGES,
    LEARNING_CURVE_UPDATES,
    learning_curve_recipe,
    learning_coefficients,
    load_environment_source,
    parse_args as training_args,
)
from .operator_roa_evaluation import (
    COMMAND_TAPE,
    _hash_tensors,
    command_tape_sha256,
    frozen_start_comparison,
)
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


def experiment_config(value):
    """Validate data, not executable commands; native child admission stays strict."""
    require(
        type(value) is dict
        and set(value) == {"version", "trials", "screens", "routes"}
        and type(value["version"]) is int
        and value["version"] == 1,
        "Config requires version=1, trials, screens and routes",
    )
    require(type(value["trials"]) is list and value["trials"], "Empty trial list")
    names, trials = set(), []
    for raw in value["trials"]:
        require(
            type(raw) is dict
            and set(raw)
            == {
                "name",
                "layout",
                "seed",
                "updates",
                "history_interval",
                "regularization",
                "num_envs",
            },
            "Each trial needs name/layout/seed/updates/history_interval/regularization/num_envs",
        )
        name = raw["name"]
        require(
            isinstance(name, str)
            and re.fullmatch(r"[a-z][a-z0-9_-]*", name)
            and name != "parent"
            and name not in names,
            "Trial names must be unique safe directory names, excluding parent",
        )
        require(
            raw["layout"] in ("contact_approach", "contact_acquire", "contact_continue")
            and raw["regularization"] in ("off", "ramp")
            and all(
                type(raw[k]) is int and raw[k] > 0
                for k in ("updates", "history_interval", "num_envs")
            )
            and type(raw["seed"]) is int
            and raw["seed"] >= 0,
            "Invalid trial settings",
        )
        trial = {**raw, "label": name}
        # Parse the actual pilot CLI before starting any children. Do not silently
        # broaden the checkpoint/training recipes supported by that implementation.
        training_args(
            [
                "reference.pt",
                "--environment-checkpoint",
                "parent.pt",
                *trial_arguments(trial),
            ]
        )
        names.add(name)
        trials.append(trial)
    for key, choices in (
        ("screens", SCREENS),
        ("routes", ("causal", "privileged_latent")),
    ):
        entries = value[key]
        require(
            type(entries) is list
            and entries
            and all(isinstance(x, str) and x in choices for x in entries)
            and len(set(entries)) == len(entries),
            f"Invalid or duplicate {key}",
        )
    require(value["routes"][0] == "causal", "Causal screens must run first")
    return {**value, "trials": trials}


def trial_arguments(trial):
    """One path from declared settings to the training child command."""
    return [
        "--environment-layout",
        trial["layout"],
        "--regularization",
        trial["regularization"],
        "--orientation-weight",
        "-2.5",
        "--stumble-weight",
        "0",
        "--learning-updates",
        str(trial["updates"]),
        "--history-interval",
        str(trial["history_interval"]),
        "--num-envs",
        str(trial["num_envs"]),
        "--seed",
        str(trial["seed"]),
        "--training-telemetry",
    ]


def gpu_inventory():
    """Fail closed if GPU/process accounting is unavailable; never kill processes."""

    def query(arguments):
        result = subprocess.run(
            ["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))

    processes = query(["--query-compute-apps=gpu_uuid"])
    require(
        all(
            len(row) == 1 and re.fullmatch(r"GPU-[a-fA-F0-9-]+", row[0])
            for row in processes
        ),
        "GPU process accounting is unavailable or malformed",
    )
    busy = {row[0] for row in processes}
    return [
        dict(
            index=index,
            uuid=uuid,
            name=name,
            memory_mib=int(memory),
            utilization=int(utilization),
            compute_busy=uuid in busy,
        )
        for index, uuid, name, memory, utilization in query(
            ["--query-gpu=index,uuid,name,memory.used,utilization.gpu"]
        )
    ]


def idle_gpu(model):
    """Select within existing CUDA visibility and acquire a cooperative host lock.

    This does not reserve a GPU against noncooperating jobs. Use the cluster
    scheduler for exclusive allocation when available. File stays; flock is
    released on close/process exit, so a crashed run cannot leave a stale lease.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = None if visible is None else {x.strip() for x in visible.split(",")}
    require(
        visible is None or all(x.startswith("GPU-") for x in visible),
        "--idle-gpu requires UUID-based CUDA visibility or an unset CUDA_VISIBLE_DEVICES; for a scheduler-assigned numeric device, omit --idle-gpu",
    )
    for gpu in gpu_inventory():
        if (
            model not in gpu["name"]
            or gpu["compute_busy"]
            or gpu["memory_mib"] > 256
            or gpu["utilization"] > 5
            or (visible is not None and not (gpu["uuid"] in visible))
        ):
            continue
        require(re.fullmatch(r"GPU-[a-fA-F0-9-]+", gpu["uuid"]), "Invalid GPU UUID")
        try:
            lock = str(Path("/tmp") / f"parkour-{gpu['uuid']}.lock")
            flags = os.O_RDONLY | os.O_NOFOLLOW
            try:
                descriptor = os.open(lock, flags)
            except FileNotFoundError:
                descriptor = os.open(lock, flags | os.O_CREAT | os.O_EXCL, 0o644)
            lease = os.fdopen(descriptor, "r")
        except (PermissionError, FileExistsError):
            continue
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.close()
            continue
        try:
            current = next(g for g in gpu_inventory() if g["uuid"] == gpu["uuid"])
            if (
                current["compute_busy"]
                or current["memory_mib"] > 256
                or current["utilization"] > 5
            ):
                lease.close()
                continue
        except BaseException:
            lease.close()
            raise
        return lease, current
    raise ValueError(f"No idle visible {model!r} GPU; no job started")


def require_idle_gpu(uuid):
    # Utilization is sampled over a driver interval and may lag an exited child.
    # Retry that case briefly; an actual compute process is never ignored.
    for attempt in range(4):
        current = next(g for g in gpu_inventory() if g["uuid"] == uuid)
        require(
            not current["compute_busy"],
            "Selected GPU has another compute process; batch stopped",
        )
        if current["memory_mib"] <= 256 and current["utilization"] <= 5:
            return
        if attempt < 3:
            time.sleep(1)
    raise ValueError(
        "Selected GPU is no longer idle; batch stopped without killing any process"
    )


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


def train_check(
    run,
    expected,
    *,
    seed=1063,
    layout="contact_approach",
    source=None,
    learning_curve=False,
    updates=None,
    history_interval=5,
    regularization="off",
    num_envs=160,
):
    report, protocol = read(run / "report.json"), read(run / "training_protocol.json")
    stage = ENVIRONMENT_STAGES[layout]
    approach = layout == "contact_approach"
    updates = updates if updates is not None else 6000 if learning_curve else 1000
    require(
        protocol.get("learning_curve")
        == (learning_curve_recipe() if learning_curve else None),
        "Training learning-curve recipe changed",
    )
    require(
        layout in ("contact_approach", "contact_acquire", "contact_continue")
        and protocol["environment_layout"] == layout
        and protocol["version"] == stage.version
        and protocol["seed"] == seed
        and protocol["num_envs"] == num_envs
        and protocol["history_interval"] == history_interval
        and protocol["cycles"] == updates
        and protocol["regularization_coefficients"]
        == list(learning_coefficients(regularization, updates))
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
    _validate_completion(
        report,
        stage.status,
        updates * 24 + updates // history_interval * 64 + 1800,
        num_envs,
    )
    require(
        protocol["source_identity"] == expected, "Training runtime/reference changed"
    )
    require(
        report["cleanup"]["environment"] == "complete",
        "Training environment cleanup failed",
    )
    require(
        report["ppo_updates_completed"] == updates
        and report["adaptation_optimizer_steps"] == updates // history_interval * 16,
        "Incomplete learning budget",
    )
    require(
        report["ppo_options"]["entropy_coef"] == 0.01
        and "entropy_ablation" not in protocol,
        "Unrequested noise ablation",
    )
    frozen_start_comparison(
        report["evaluation_before"],
        report["evaluation_after"],
        contact_conditioned=stage.contact_conditioned,
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
        == sorted({history_interval, updates, *range(100, updates + 1, 100)}),
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
                arrays["reward_total"].shape == (history_interval * 24 + 64, num_envs)
                and arrays["ppo_return"].shape == (history_interval, 24, num_envs, 1),
                "Wrong training trace dimensions",
            )
            if not approach:
                continue
            holding = arrays["approach_hold_pre"]
            command = arrays["command_b_pre"][holding]
            require(
                holding.dtype == np.bool_
                and holding.shape == (history_interval * 24 + 64, num_envs),
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
    checkpoint = run / f"learning_{updates}.pt"
    require(
        file_sha256(checkpoint) == report["checkpoint_sha256"],
        "Training checkpoint changed",
    )
    if learning_curve:
        for n in LEARNING_CURVE_UPDATES:
            load_screen_checkpoint(run / f"learning_{n}.pt")
    return checkpoint


def recovery_files(run):
    """Immutable evidence of one endpoint in a seed/layout batch, not a resume."""
    batch = run.parents[3]
    paths = (
        batch / "batch_plan.json",
        batch / "batch_check.json",
        run.parent / "process_exit.json",
        *(
            run / name
            for name in ("report.json", "training_protocol.json", "learning_1000.pt")
        ),
    )
    return {str(path.relative_to(batch)): file_sha256(path) for path in paths}


def recovery_check(run, evidence_sha256, expected, source):
    files = recovery_files(run)
    require(
        identity_sha256(files) == evidence_sha256, "Archived training evidence changed"
    )
    batch = run.parents[3]
    plan, failed = read(batch / "batch_plan.json"), read(batch / "batch_check.json")
    protocol = read(run / "training_protocol.json")
    seed, layout = protocol["seed"], protocol["environment_layout"]
    label = f"seed{seed}/{layout}"
    require(
        layout in ("contact_approach", "contact_acquire")
        and run.parent.relative_to(batch).as_posix() == f"{label}/training"
        and dict(seed=seed, layout=layout, label=label) in plan["trials"]
        and plan["updates"] == 1000
        and plan["parent"] == source
        and failed["status"] == "FAILED_OR_INTERRUPTED_NOT_QUALIFIED"
        and failed["error"] == "ValueError('Training before/after starts differ')",
        "Recovery requires the reviewed failed-start-comparison batch endpoint",
    )
    original = protocol["source_identity"]
    require(
        original == plan["source_identity"] == failed["source_identity"]
        and original["physical_reference"] == expected["physical_reference"]
        and original["runtime"].keys() == expected["runtime"].keys(),
        "Archived runtime/reference is not the planned training identity",
    )
    changed = {
        name: dict(training=old, screening=expected["runtime"][name])
        for name, old in original["runtime"].items()
        if old != expected["runtime"][name]
    }
    require(
        changed.keys()
        <= {
            "scripts/rsl_rl/operator_roa_approach_batch.py",
            "scripts/rsl_rl/operator_roa_evaluation.py",
            "scripts/rsl_rl/operator_roa_screen.py",
            "scripts/rsl_rl/operator_roa_pilot.py",
            "scripts/rsl_rl/operator_roa_checkpoint.py",
        },
        "Recovery runtime changed beyond the reviewed frozen-screen and snapshot support",
    )
    process = read(run.parent / "process_exit.json")
    require(
        process["returncode"] == 0
        and process["interrupted"] is False
        and process["error"] is None,
        "Archived training process did not exit cleanly",
    )
    checkpoint = train_check(run, original, seed=seed, layout=layout, source=source)
    # Preserve all native geometry, checkpoint, motor and ancestry validators.
    policy, _, _, receipt = load_completed_checkpoint(checkpoint)
    require(
        policy.actor.contact_conditioned,
        "Recovery requires the contact-conditioned ROA",
    )
    report = read(run / "report.json")
    return checkpoint, dict(
        version="operator_roa_frozen_recovery_v1",
        batch=str(batch),
        training_run=str(run),
        evidence_sha256=evidence_sha256,
        files={str(batch / name): digest for name, digest in files.items()},
        source=receipt,
        training_identity_sha256=identity_sha256(original),
        reviewed_runtime_changes=changed,
        original_initial_conditions_match=report["evaluation_initial_conditions_match"],
        frozen_start_comparison=frozen_start_comparison(
            report["evaluation_before"],
            report["evaluation_after"],
            contact_conditioned=True,
        ),
        new_learning_updates=0,
        exit_allowed=False,
    )


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
        else "step_ladder"
        if name == "ladder_1045"
        else "standard"
    )
    difficulty = (
        list(operator_step_field.DIFFICULTY)
        if field
        else [0.05, 0.15]
        if retention
        else None
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
        else COMMAND_TAPE
        if retention
        else TRAVERSAL_TAPE
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
        frozen_start_comparison(baseline["evaluation"], ev, contact_conditioned=True)
        require(
            all(
                report.get(key) == baseline.get(key)
                for key in ("native_geometry", "native_step_field_geometry")
            ),
            "Unmatched frozen geometry",
        )
    initial = ev["initial_terrain"]
    require(
        initial["version"] == "operator_roa_initial_terrain_v1"
        and initial["path"] == "initial_terrain.npz"
        and initial["shape"] == [n, 264]
        and initial["dtype"] == "float32"
        and initial["observation_group_sha256"]
        == ev["initial_observation_group_sha256"]["terrain"],
        "Invalid initial terrain capture",
    )
    trace_check(run / initial["path"], initial["sha256"])
    with np.load(run / initial["path"], allow_pickle=False) as arrays:
        import torch

        scan = arrays["terrain"]
        require(
            arrays.files == ["terrain"]
            and scan.shape == (n, 264)
            and scan.dtype == np.float32
            and _hash_tensors({"terrain": torch.from_numpy(scan)})
            == initial["observation_group_sha256"],
            "Initial terrain capture differs from the frozen observation",
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
    parser.add_argument("--training-seeds", nargs="+", type=int)
    parser.add_argument(
        "--config",
        help="Experiment JSON file, or - to read stdin; replaces legacy trial selectors",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate source/configuration and print the resolved workload without starting simulation",
    )
    parser.add_argument(
        "--idle-gpu",
        metavar="MODEL",
        help="Select an idle visible model (e.g. 'RTX 5080'), cooperative lock and UUID isolation; does not replace scheduler reservation",
    )
    parser.add_argument(
        "--learning-curve",
        action="store_true",
        help="Two paired seeds, four uninterrupted 6000-update runs with separate 1000/3000/6000 snapshot screens",
    )
    parser.add_argument(
        "--paired-control",
        action="store_true",
        help="Also train the existing raised-start/free-command recipe for each seed",
    )
    parser.add_argument(
        "--screen-training-run",
        type=Path,
        help="Screen one reviewed archived seed/layout endpoint and its parent; no training or batch resume",
    )
    parser.add_argument(
        "--training-evidence-sha256",
        help="Reviewed digest of recovery_files for --screen-training-run",
    )
    args = parser.parse_args(argv)
    configured = None
    if args.config:
        if (
            args.training_seeds is not None
            or args.paired_control
            or args.learning_curve
            or args.screen_training_run
        ):
            parser.error(
                "--config cannot be combined with legacy trial/recovery selectors"
            )
        configured = experiment_config(
            json.load(sys.stdin) if args.config == "-" else read(Path(args.config))
        )
        args.training_seeds = sorted({t["seed"] for t in configured["trials"]})
    if bool(args.screen_training_run) != bool(args.training_evidence_sha256):
        parser.error(
            "screen-training-run and training-evidence-sha256 require each other"
        )
    if args.screen_training_run and (
        args.training_seeds is not None or args.paired_control or args.learning_curve
    ):
        parser.error(
            "screens-only recovery cannot request training seeds or paired training"
        )
    if args.screen_training_run:
        args.screen_training_run = args.screen_training_run.resolve(strict=True)
        args.training_seeds = [
            read(args.screen_training_run / "training_protocol.json")["seed"]
        ]
    else:
        args.training_seeds = args.training_seeds or [1063]
    if args.learning_curve and not (
        args.paired_control and len(args.training_seeds) == 2
    ):
        parser.error(
            "learning-curve requires --paired-control and exactly two training seeds"
        )
    if any(seed < 0 for seed in args.training_seeds) or len(
        set(args.training_seeds)
    ) != len(args.training_seeds):
        parser.error("training-seeds must be distinct nonnegative integers")
    args.reference, args.parent = (
        args.reference.resolve(strict=True),
        args.parent.resolve(strict=True),
    )
    expected = recurrent_training_identity(args.reference)
    recovery = None

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
        if recovery is not None:
            verify_source_files(recovery)
            verify_source_files(recovery["source"])

    preflight()
    _, source = load_environment_source(
        args.parent,
        expected["physical_reference"],
        args.training_seeds[0],
        layout=configured["trials"][0]["layout"] if configured else "contact_approach",
    )
    require(
        source["training_seed"] not in args.training_seeds,
        "Training seeds must differ from the retained parent's seed",
    )
    layouts = (
        ("contact_approach", "contact_acquire")
        if args.paired_control
        else ("contact_approach",)
    )
    sweep = (
        configured is not None or len(args.training_seeds) > 1 or args.paired_control
    )
    trials = (
        [dict(trial) for trial in configured["trials"]]
        if configured
        else [
            dict(
                seed=seed,
                layout=layout,
                label=f"seed{seed}/{layout}" if sweep else "candidate",
                updates=6000 if args.learning_curve else 1000,
                history_interval=5,
                regularization="off",
                num_envs=160,
            )
            for seed in args.training_seeds
            for layout in layouts
        ]
    )
    admissions = []
    if not args.screen_training_run:
        for trial in trials:
            _, trial_source = load_environment_source(
                args.parent,
                expected["physical_reference"],
                trial["seed"],
                layout=trial["layout"],
            )
            require(
                trial_source == source, "Trials must share the same immutable parent"
            )
            if trial["layout"] in ("contact_approach", "contact_acquire"):
                admissions.append(
                    seed_admission(
                        trial["seed"], paired=trial["layout"] == "contact_acquire"
                    )
                )
    if args.screen_training_run:
        checkpoint, recovery = recovery_check(
            args.screen_training_run, args.training_evidence_sha256, expected, source
        )
        trials = [
            dict(
                seed=args.training_seeds[0],
                layout=read(args.screen_training_run / "training_protocol.json")[
                    "environment_layout"
                ],
                label="candidate",
                updates=1000,
                history_interval=5,
                regularization="off",
                num_envs=160,
            )
        ]
        require(
            not args.output_parent.resolve().is_relative_to(Path(recovery["batch"])),
            "Recovery output must be outside the immutable failed batch",
        )
    require(
        not any(
            args.output_parent.resolve().is_relative_to(Path(path).parent)
            for path in source["files"]
        ),
        "Output must be outside immutable source runs",
    )
    selected_screens = configured["screens"] if configured else list(SCREENS)
    routes = configured["routes"] if configured else ["causal", "privileged_latent"]
    workload = dict(
        training_children=0 if recovery else len(trials),
        training_environment_transitions=0
        if recovery
        else sum(
            t["num_envs"]
            * (t["updates"] * 24 + t["updates"] // t["history_interval"] * 64)
            for t in trials
        ),
        frozen_screen_children=(
            1
            + len(trials) * (len(LEARNING_CURVE_UPDATES) if args.learning_curve else 1)
        )
        * sum(1 if name == "retention" else len(routes) for name in selected_screens),
    )
    if args.plan_only:
        print(
            json.dumps(
                dict(
                    trials=trials,
                    screens=selected_screens,
                    routes=routes,
                    workload=workload,
                    source_identity=expected,
                    exit_allowed=False,
                ),
                indent=2,
            )
        )
        return
    lease, gpu = idle_gpu(args.idle_gpu) if args.idle_gpu else (None, None)
    child_environment = None
    if gpu:
        child_environment = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu["uuid"]}
        args.device = "cuda:0"
        print(
            f"Selected {gpu['name']} ({gpu['uuid']}); cooperative lock held", flush=True
        )
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(
            prefix="operator_experiment_" if configured else "roa_step_approach_",
            dir=args.output_parent,
        )
    ).resolve()
    print(f"Output: {output}", flush=True)
    summary = dict(
        status="STARTED_NOT_QUALIFIED",
        exit_allowed=False,
        source_identity=expected,
        parent=source,
        trials=trials,
        screens={},
        training_recovery=recovery,
        workload=workload,
        experiment_config=configured,
        experiment_config_sha256=identity_sha256(configured) if configured else None,
        gpu=gpu,
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
            updates=0
            if recovery
            else None
            if configured
            else 6000
            if args.learning_curve
            else 1000,
            learning_curve=learning_curve_recipe() if args.learning_curve else None,
            screens={name: SCREENS[name] for name in selected_screens},
            routes=routes,
            scope=(
                "One archived endpoint and its parent, fourteen fresh frozen screens; no training, resume or promotion. Historical full-observation mismatch remains unresolved."
                if recovery
                else "Prespecified development seeds, each from the same immutable parent. Optional paired bundled occupancy recipes, not a single-factor intervention or independent parent replications. Parent screens once; no ranking, chaining, retries or promotion. Analytic seed admission is not native contact certification."
            ),
        ),
    )
    started = time.monotonic()

    def child(folder, module, arguments):
        preflight()
        verify_source_files(source)
        if gpu:
            require_idle_gpu(gpu["uuid"])
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
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    **({"env": child_environment} if child_environment else {}),
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
        for name in selected_screens:
            n, seed, options = SCREENS[name]
            for route in ["causal"] if name == "retention" else routes:
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
                    frozen_start_comparison=frozen_start_comparison(
                        controls[name]["evaluation"],
                        report["evaluation"],
                        contact_conditioned=True,
                    ),
                    behavior_accepted=False,
                )

    try:
        for trial in trials:
            seed, layout, label = trial["seed"], trial["layout"], trial["label"]
            case = output / label if sweep else output
            run = (
                args.screen_training_run
                if recovery
                else child(
                    case / "training",
                    "scripts.rsl_rl.operator_roa_pilot",
                    [
                        "--environment-checkpoint",
                        args.parent,
                        *trial_arguments(trial),
                        *(["--learning-curve"] if args.learning_curve else []),
                    ],
                )
            )
            checkpoint = train_check(
                run,
                (
                    read(run / "training_protocol.json")["source_identity"]
                    if recovery
                    else expected
                ),
                seed=seed,
                layout=layout,
                source=source,
                learning_curve=args.learning_curve,
                updates=trial["updates"],
                history_interval=trial["history_interval"],
                regularization=trial["regularization"],
                num_envs=trial["num_envs"],
            )
            report = read(run / "report.json")
            # These are frozen pre-training evaluations, not initial PPO states:
            # the two training occupancy recipes intentionally differ.
            start = report["evaluation_before"]
            geometry = report["native_step_field_geometry"]
            paired_start, paired_geometry = paired_starts.setdefault(
                (seed, ENVIRONMENT_STAGES[layout].geometry_version, trial["num_envs"]),
                (start, geometry),
            )
            require(
                start["policy_state_sha256_before"]
                == paired_start["policy_state_sha256_before"]
                == source["policy_state_sha256"]
                and geometry == paired_geometry,
                f"Paired frozen parent policy or geometry differs: seed {seed}",
            )
            trial.update(
                checkpoint=str(checkpoint),
                training_report=str(run / "report.json"),
                training_report_sha256=file_sha256(run / "report.json"),
                paired_frozen_start_comparison=frozen_start_comparison(
                    paired_start, start, contact_conditioned=True
                ),
                frozen_start_comparison=frozen_start_comparison(
                    report["evaluation_before"],
                    report["evaluation_after"],
                    contact_conditioned=True,
                ),
            )
            if not controls:
                screens("parent", args.parent)
            if args.learning_curve:
                trial["milestone_checkpoints"] = {
                    str(n): str(run / f"learning_{n}.pt")
                    for n in LEARNING_CURVE_UPDATES
                }
                for n in LEARNING_CURVE_UPDATES:
                    screens(f"{label}/updates{n}", run / f"learning_{n}.pt")
            else:
                screens(label, checkpoint)
            trial["status"] = "COMPLETE_NOT_PROMOTED"
        summary["status"] = "COMPLETE_NOT_PROMOTED"
    except BaseException as exc:
        summary.update(status="FAILED_OR_INTERRUPTED_NOT_QUALIFIED", error=repr(exc))
        raise
    finally:
        if lease is not None:
            lease.close()
        summary["wall_seconds"] = time.monotonic() - started
        write(output / "batch_check.json", summary)
        print(
            f"Return the complete folder, including partial outputs on failure: {output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
