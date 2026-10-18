"""External Stable-Baselines3 training, continuation and frozen evaluation."""

import argparse
import platform
import sys
import time
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


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser(
        "train", help="Collect, learn and save a full checkpoint"
    )
    train.add_argument("--algorithm", choices=("ppo", "sac"), required=True)
    train.add_argument("--total-interactions", type=int, default=2560)
    train.add_argument("--resume", type=Path, help="Existing full checkpoint folder")
    train.add_argument("--task", default="Go2-Mixed-v1")
    train.add_argument("--num-envs", type=int, default=20)
    train.add_argument("--seed", type=int, default=1077)
    train.add_argument("--run-id")
    train.add_argument(
        "--save-every-updates",
        type=int,
        default=5000,
        help="Full checkpoint interval in declared learner update units",
    )
    train.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=(),
        help="Absolute interaction totals for full checkpoint saves",
    )
    evaluate = commands.add_parser("evaluate", help="Evaluate an already-saved policy")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--full-bank", action="store_true")
    evaluate.add_argument("--suite", choices=("core", "robustness"), default="core")
    for command in (train, evaluate):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--learner-device", default="cpu")
    return parser


def _validate(parser, args):
    if args.command == "train":
        if args.total_interactions <= 0 or args.num_envs <= 0:
            parser.error("Interaction target and replica count must be positive")
        if not 0 <= args.seed < 2**32:
            parser.error("Require a uint32 seed")
        if args.save_every_updates <= 0 or any(
            point <= 0 or point > args.total_interactions for point in args.milestones
        ):
            parser.error(
                "Require positive save interval and milestones within the target"
            )
        checkpoint = args.resume
    else:
        checkpoint = args.checkpoint
    if checkpoint is not None and not checkpoint.is_dir():
        parser.error("Checkpoint must be an existing folder")


def _dependencies():
    return {
        name: dependency_identity(name, distribution)
        for name, distribution in (
            ("isaaclab", "isaaclab"),
            ("isaaclab_assets", "isaaclab_assets"),
            ("isaaclab_tasks", "isaaclab_tasks"),
            ("isaacsim", "isaacsim"),
            ("stable_baselines3", "stable-baselines3"),
            ("torch", "torch"),
            ("gymnasium", "gymnasium"),
            ("numpy", "numpy"),
        )
    }


def _example_sources():
    root = Path(__file__).parent
    paths = [path for path in root.glob("*.py")]
    requirements = root / "requirements.txt"
    if requirements.exists():
        paths.append(requirements)
    return {path.name: file_sha256(path) for path in sorted(paths)}


def _cases(full_bank, suite="core"):
    from parkour_lab.evaluation import EvaluationCase, case_bank

    if suite == "robustness":
        from parkour_lab.evaluation import robustness_bank

        return robustness_bank("validation")
    if full_bank:
        return case_bank("validation")
    return tuple(
        EvaluationCase("validation", *fields)
        for fields in (
            ("Flat", None, "plane", "stand"),
            ("Uneven", 0, "noise", "forward-035"),
            ("Slopes", 3, "pyramid", "pivot-left"),
            ("Stairs", 7, "inverted_pyramid", "pivot-right"),
            ("Hills", 3, "wave", "arc-left"),
            ("Flat", None, "plane", "arc-right"),
        )
    )


def _train(args, app, output, report, task):
    from .training import run_training

    report.update(
        run_training(
            task,
            algorithm=args.algorithm,
            total_interactions=args.total_interactions,
            output=output,
            learner_device=args.learner_device,
            resume=args.resume,
            app=app,
            run_id=args.run_id,
            save_every_updates=args.save_every_updates,
            milestones=args.milestones,
        )
    )
    if task.interactions["cumulative"] != args.total_interactions:
        raise RuntimeError("Training did not finish at the declared interaction target")
    report.update(status="LEARNER_CHECK_PASSED", interactions=task.interactions)


def _evaluate(args, app, output, report):
    from parkour_lab.evaluation import evaluate, score_capture

    from .policy import load_policy

    policy, checkpoint = load_policy(args.checkpoint, device=args.learner_device)
    cases = _cases(args.full_bank, args.suite)
    result = evaluate(
        policy,
        cases,
        output / "capture",
        device=args.device,
        training_seed=checkpoint["training_seed"],
        checkpoint_id=checkpoint["checkpoint_id"],
        checkpoint_sha256=checkpoint["checkpoint_sha256"],
        app=app,
    )
    if not result["capture_valid"]:
        raise RuntimeError("Policy evaluation did not produce a valid complete capture")
    full_bank = args.full_bank or args.suite == "robustness"
    if result["valid_for_comparison"] is not full_bank:
        raise RuntimeError("Evaluation bank coverage differs from the requested scope")
    if score_capture(output / "capture") != result:
        raise RuntimeError("Saved capture scoring differs from native evaluation")
    report.update(
        status="LEARNER_EVALUATION_PASSED",
        checkpoint=checkpoint,
        full_bank=full_bank,
        suite=args.suite,
        evaluation=result,
        recomputed_score_parity=True,
    )


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(
        status="RUNNING",
        command=args.command,
        package_sources=package_source_identity(),
        example_sources=_example_sources(),
        interpreter=dict(executable=sys.executable, version=sys.version),
        behavior_validated=False,
        calibration_validated=False,
        learner_device=args.learner_device,
        resources=dict(
            physics_device=args.device,
            learner_device=args.learner_device,
            system=platform.platform(),
            processor=platform.processor(),
        ),
    )

    def publish():
        report["resources"]["wall_time_s"] = time.perf_counter() - started
        write_json(output / "report.json", report)

    app = task = None
    code = 2
    try:
        write_run_provenance(output, __file__)
        publish()
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device=args.device, fast_shutdown=False).app
        import torch

        torch.set_num_threads(2)
        report["resources"].update(
            torch_threads=torch.get_num_threads(),
            available_cuda_devices=[
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        )
        report["dependencies"] = _dependencies()
        if args.command == "train":
            import yaml

            from parkour_lab.benchmark import Go2Config, build_config, make

            cfg = build_config(
                Go2Config(
                    task_id=args.task,
                    num_envs=args.num_envs,
                    seed=args.seed,
                    device=args.device,
                ),
                training=True,
            )
            (output / "resolved_env.yaml").write_text(
                yaml.dump(cfg.to_dict(), sort_keys=False)
            )
            task = make(cfg)
            report["metadata"] = task.metadata
            _train(args, app, output, report, task)
        else:
            _evaluate(args, app, output, report)
        code = 0
    except BaseException as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        if task is not None:
            report["interactions"] = task.interactions
        code = finish_session(task, app, report, publish, code)
        if app is not None:
            exit_native_process(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
