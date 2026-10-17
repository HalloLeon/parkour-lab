"""Bounded native composed-terrain and post-limit drive-loss engineering checks."""

import argparse
import sys
import traceback
from pathlib import Path

from parkour_lab.provenance import (
    dependency_identity,
    file_sha256,
    package_source_identity,
    write_json,
    write_run_provenance,
)
from parkour_lab.runtime.session import exit_native_process, finish_session


def _task(args, folder, *, condition=None):
    import yaml

    from parkour_lab.benchmark import Go2Config, TerrainConfig, build_config, make
    from parkour_lab.environments.faults import FaultConfig
    from parkour_lab.evaluation.environment import _prepare_case_stage

    _prepare_case_stage()
    options = dict(num_envs=2, seed=args.seed, device=args.device)
    if condition is None:
        options.update(
            task_id="Go2-Flat-v1",
            fault=FaultConfig(args.joint, args.factor, onset_s=0.04),
        )
    else:
        options.update(
            task_id=f"Go2-{args.family}-v1",
            terrain=TerrainConfig(fixed_level=args.level, condition=condition),
        )
    cfg = build_config(Go2Config(**options), training=False)
    (folder / "resolved_env.yaml").write_text(yaml.dump(cfg.to_dict(), sort_keys=False))
    task = make(cfg)
    try:
        # The controller host owns sensing in the motor check. Avoid preparing
        # additional public actor frames or resetting learner RNGs here.
        task._env.reset(seed=None)
        return task
    except BaseException:
        task.close()
        raise


def _geometry(args, output, report, publish, app):
    from scripts._robustness_checks.contact import check_outer_contact
    from scripts._robustness_checks.geometry import compare_surfaces, sample_surface

    samples, definitions = {}, {}
    for condition in ("id", "composition", "terrain_ood"):
        folder = output / condition
        folder.mkdir()
        receipt = dict(status="RUNNING", condition=condition)
        report["contexts"][condition] = receipt

        def save():
            write_json(folder / "report.json", receipt)
            publish()

        task, code = None, 2
        try:
            save()
            task = _task(args, folder, condition=condition)
            receipt["metadata"] = task.metadata
            samples[condition], receipt["surface"] = sample_surface(task, folder)
            definitions[condition] = task._env.cfg.benchmark_terrain
            assert task.interactions == dict(session=0, cumulative=0, uncertain=0)
            if condition == "composition":
                receipt["contact"] = check_outer_contact(task, app, folder)
            receipt.update(status="SURFACE_CAPTURED", interactions=task.interactions)
            code = 0
        except BaseException as error:
            receipt.update(
                status="ERROR", error=str(error), traceback=traceback.format_exc()
            )
            raise
        finally:
            if task is not None:
                receipt["interactions"] = task.interactions
            code = finish_session(task, None, receipt, save, code)
        if code:
            raise RuntimeError(f"Could not close {condition} environment cleanly")
    report["geometry"] = compare_surfaces(samples, definitions)
    report["collected_interactions"] = sum(
        context["interactions"]["session"] for context in report["contexts"].values()
    )


def _motor(args, output, report, publish, app):
    from scripts._robustness_checks.motor import direct_motor_checks
    from scripts._robustness_checks.physics import check_physics

    receipt = dict(status="RUNNING")
    report["contexts"]["motor"] = receipt
    task, code = None, 2
    try:
        task = _task(args, output)
        receipt["metadata"] = task.metadata
        direct = direct_motor_checks(task)
        write_json(output / "direct_motor.json", direct)
        receipt["direct_motor"] = dict(passed=True, witness="direct_motor.json")
        publish()
        physical = check_physics(task, app)
        write_json(output / "physics.json", physical)
        receipt.update(
            status="MOTOR_CHECK_PASSED",
            physics=dict(passed=True, witness="physics.json"),
        )
        code = 0
    except BaseException as error:
        receipt.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        if task is not None:
            receipt["interactions"] = report["interactions"] = task.interactions
        code = finish_session(task, None, receipt, publish, code)
    if code:
        raise RuntimeError("Could not close motor environment cleanly")


def _arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", required=True, choices=("geometry", "motor"))
    parser.add_argument(
        "--family", choices=("Slopes", "Stairs", "Hills"), default="Slopes"
    )
    parser.add_argument("--level", type=int, choices=(0, 3, 7), default=3)
    parser.add_argument("--factor", type=float, choices=(0, 0.5, 1), default=0.5)
    joints = tuple(
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    )
    parser.add_argument("--joint", choices=joints, default="FR_thigh_joint")
    parser.add_argument("--seed", type=int, default=1077)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**32:
        parser.error("Require a uint32 seed")
    return args


def main(argv=None):
    args = _arguments(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).parent
    sources = [
        Path(__file__),
        *(
            scripts / "_terrain_checks" / name
            for name in ("geometry.py", "contacts.py")
        ),
        *sorted((scripts / "_robustness_checks").glob("*.py")),
    ]
    report = dict(
        status="RUNNING",
        behavior_validated=False,
        calibration_validated=False,
        arguments={**vars(args), "output": str(output)},
        package_sources=package_source_identity(),
        host_sources={
            str(path.relative_to(scripts)): file_sha256(path) for path in sources
        },
        interpreter=dict(executable=sys.executable, version=sys.version),
        contexts={},
    )

    def publish():
        write_json(output / "report.json", report)

    app, code = None, 2
    try:
        write_run_provenance(output, __file__)
        publish()
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device=args.device, fast_shutdown=False).app
        report["dependencies"] = {
            name: dependency_identity(name, name)
            for name in (
                "isaaclab",
                "isaaclab_assets",
                "isaaclab_tasks",
                "isaacsim",
                "torch",
                "numpy",
                "gymnasium",
            )
        }
        if args.check == "geometry":
            _geometry(args, output, report, publish, app)
        else:
            _motor(args, output, report, publish, app)
        assert all(
            context["cleanup"]["environment"] == "complete"
            for context in report["contexts"].values()
        )
        report["status"] = "ROBUSTNESS_CHECK_PASSED"
        code = 0
    except BaseException as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        code = finish_session(None, app, report, publish, code)
        if app is not None:
            exit_native_process(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
