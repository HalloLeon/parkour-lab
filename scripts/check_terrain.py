"""Bounded native terrain, curriculum and continuation checks; no learning."""

import argparse
import json
import sys
import traceback
from pathlib import Path

from _terrain_checks.contacts import check_contact
from _terrain_checks.curriculum import check_boundary, check_curriculum
from _terrain_checks.geometry import assignments, check_geometry

from parkour_lab.environments.terrains.specification import LEVEL_COUNT
from parkour_lab.provenance import (
    dependency_identity,
    package_source_identity,
    write_json,
    write_run_provenance,
)
from parkour_lab.runtime.console import log_event, restore_console, setup_console
from parkour_lab.runtime.session import exit_native_process, finish_session


def _dependencies(output):
    import isaaclab

    result = {
        name: dependency_identity(name, distribution)
        for name, distribution in (
            ("isaaclab", "isaaclab"),
            ("isaaclab_assets", "isaaclab_assets"),
            ("isaaclab_tasks", "isaaclab_tasks"),
            ("isaacsim", "isaacsim"),
            ("torch", "torch"),
            ("gymnasium", "gymnasium"),
            ("numpy", "numpy"),
            ("scipy", "scipy"),
        )
    }
    if "commit" in result["isaaclab"]:
        write_run_provenance(output / "sdk", isaaclab.__file__)
    return result


def _run(task, app, args, output, report):
    from parkour_lab.environments.terrains.curriculum import curriculum_state

    native = task._env
    if args.resume:
        record = json.loads(args.resume.read_text())
        restored = task.restore(record, checkpoint_id="terrain-contract-fixture")
        assert curriculum_state(native) == record["curriculum"]
        assert native.parkour_randomization.state_dict() == record["sampler"]
        assert (
            native.command_manager.get_term("base_velocity").sampling_report
            == record["sampling_report"]
        )
        report["restored_interactions"] = restored["interactions"]
    observation, _ = task.reset(seed=args.seed)
    log_event(
        "terrain_ready",
        role=args.role,
        fixed_level=args.level,
        interactions=task.interactions,
    )
    assert set(observation) == ({"actor", "training"} if task.training else {"actor"})
    assert set(native.observation_manager.active_terms) == {"proprio"}
    report["terrain_definition"] = native.cfg.benchmark_terrain
    report["initial_assignments"] = assignments(native)
    if args.resume:
        current = curriculum_state(native)
        saved = record["curriculum"]
        for field in ("version", "definition", "types", "families"):
            assert current[field] == saved[field]
        # Fresh physical episodes take current targets, including rows that were
        # still running at an earlier family target when the save was taken.
        assert all(
            current["levels"][row["env_id"]]
            == (
                0
                if row["family"] == "Flat"
                else current["families"][row["family"]]["target_level"]
            )
            for row in report["initial_assignments"]
        )
    elif task.training:
        assert all(row["row"] == 0 for row in report["initial_assignments"])
    else:
        assert "training_information" not in task.metadata
        assert all(row["row"] == 0 for row in report["initial_assignments"])
    report["geometry"] = check_geometry(task, output)
    log_event("terrain_geometry_verified", role=args.role, fixed_level=args.level)
    report["contact"] = check_contact(task, app, output)
    log_event("terrain_contact_verified", **report["contact"])
    report["boundary"] = check_boundary(task)
    log_event("terrain_boundary_verified", **report["boundary"])
    if task.training:
        report["curriculum_cases"] = check_curriculum(task)
        log_event("curriculum_fixtures_verified", **report["curriculum_cases"])
        record = task.progress(
            run_id="terrain-contract", checkpoint_id="terrain-contract-fixture"
        )
        write_json(output / "progress.json", record)
        assert json.loads((output / "progress.json").read_text()) == record
    report.update(status="TERRAIN_CHECK_PASSED", interactions=task.interactions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("train", "eval"), required=True)
    parser.add_argument("--level", type=int, choices=range(LEVEL_COUNT))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=1077)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.role == "eval") != (args.level is not None) or (
        args.resume and args.role != "train"
    ):
        parser.error("Evaluation requires --level; training accepts optional --resume")
    if not 0 <= args.seed < 2**32:
        parser.error("Require a uint32 seed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_console = setup_console()
    log_event("terrain_check_started", role=args.role, output=str(output))
    report = dict(
        status="RUNNING",
        role=args.role,
        fixed_level=args.level,
        calibration_validated=False,
        behavior_validated=False,
        learner_restored=False,
        package_sources=package_source_identity(),
        interpreter=dict(executable=sys.executable, version=sys.version),
    )

    def publish():
        write_json(output / "report.json", report)

    app = task = None
    code = 2
    try:
        write_run_provenance(output, __file__)
        publish()
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device=args.device, fast_shutdown=False).app
        from parkour_lab.benchmark import Go2Config, TerrainConfig, build_config, make

        report["dependencies"] = _dependencies(output)
        options = Go2Config(
            task_id="Go2-Mixed-v1",
            num_envs=20,
            seed=args.seed,
            device=args.device,
            terrain=TerrainConfig(fixed_level=args.level),
        )
        task = make(build_config(options, training=args.role == "train"))
        _run(task, app, args, output, report)
        code = 0
    except BaseException as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
        log_event("terrain_check_failed", error=str(error))
        traceback.print_exc(file=sys.stderr)
    finally:
        code = finish_session(task, app, report, publish, code)
        log_event(
            "terrain_check_finished",
            status=report["status"],
            report=str(output / "report.json"),
        )
        restore_console(previous_console)
        if app is not None:
            exit_native_process(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
