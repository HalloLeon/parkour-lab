"""Small command surface for current training, inference and result inspection."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import tempfile
import time

from parkour_lab.config import ExperimentConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    for name in ("train", "evaluate", "play"):
        command = commands.add_parser(name)
        command.add_argument(
            "--config",
            type=Path,
            help="Current experiment JSON; omitted fields use explicit defaults",
        )
        command.add_argument(
            "--device", help="Explicit simulation/learning device; no GPU discovery"
        )
        command.add_argument("--num-envs", type=int)
        command.add_argument("--seed", type=int)
        command.add_argument("--cpu-threads", type=int, default=2)
        command.add_argument(
            "--output-parent", type=Path, default=Path("logs/parkour_lab")
        )
        if name == "train":
            command.add_argument(
                "--updates",
                type=int,
                help="Additional updates; not a wall-clock guarantee",
            )
            command.add_argument(
                "--checkpoint",
                type=Path,
                help="Continue a CURRENT learning snapshot with fresh simulator state",
            )
        else:
            command.add_argument("actor", type=Path)
            source = command.add_mutually_exclusive_group()
            source.add_argument(
                "--tape", type=Path, help="Replay completed portable body-twist tape"
            )
            source.add_argument(
                "--command",
                nargs=3,
                type=float,
                default=(0.3, 0.0, 0.0),
                metavar=("VX", "VY", "WZ"),
                help="Body twist followed by a final 1 s stop; steps <= 50 means all stop",
            )
            command.add_argument("--steps", type=int, default=900)
    export = commands.add_parser("export")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("destination", type=Path)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    if hasattr(args, "cpu_threads") and args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if hasattr(args, "command") and not all(math.isfinite(x) for x in args.command):
        parser.error("Commands must be finite")
    return args


def resolve_config(args):
    if args.config is not None:
        config = ExperimentConfig.load(args.config)
    else:
        artifact = getattr(args, "checkpoint", None) or getattr(args, "actor", None)
        if artifact is None:
            config = ExperimentConfig()
        else:
            from parkour_lab.artifacts import load_artifact

            data = load_artifact(
                artifact, kind="training" if args.operation == "train" else "actor"
            )
            config = ExperimentConfig.from_dict(data["config"])
    overrides = {
        key: getattr(args, key)
        for key in ("device", "seed", "num_envs")
        if getattr(args, key) is not None
    }
    if args.operation == "play" and args.num_envs is None:
        overrides["num_envs"] = 1
    config = replace(config, task=replace(config.task, **overrides))
    if getattr(args, "updates", None) is not None:
        config = replace(config, updates=args.updates)
    return config


def main(argv=None):
    args = parse_args(argv)
    if args.operation == "analyze":
        report = args.run / "report.json" if args.run.is_dir() else args.run
        print(json.dumps(json.loads(report.read_text()), indent=2, allow_nan=False))
        return 0
    if args.operation == "export":
        from parkour_lab.artifacts import export_actor

        print(json.dumps(export_actor(args.checkpoint, args.destination), indent=2))
        return 0
    config = resolve_config(args)
    if args.operation == "train":
        config.validate_training()
    import numpy as np
    import torch

    torch.set_num_threads(args.cpu_threads)
    random.seed(config.task.seed)
    np.random.seed(config.task.seed)
    torch.manual_seed(config.task.seed)
    commands = None
    if args.operation != "train":
        from parkour_lab.control.command_tape import load_tape
        from parkour_lab.evaluation.runner import command_sequence

        commands = command_sequence(
            tape=load_tape(args.tape) if args.tape else None,
            command=args.command,
            steps=args.steps,
        )
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(prefix=f"{args.operation}_", dir=args.output_parent)
    ).resolve()
    print(f"Output: {output}", flush=True)
    from parkour_lab.provenance import (
        package_source_identity,
        write_json,
        write_run_provenance,
    )
    from parkour_lab.runtime.session import finish_session

    report = {
        "status": "RUNNING",
        "qualified": False,
        "exit_allowed": False,
        "operation": args.operation,
        "config": config.to_dict(),
        "package_sources": package_source_identity(),
    }

    def publish():
        write_json(output / "report.json", report)

    app = env = None
    code = 2
    started = time.monotonic()
    try:
        write_json(output / "config.json", config.to_dict())
        try:
            write_run_provenance(output, __file__)
        except RuntimeError as error:
            # Installed wheels need no Git checkout. Package hashes still record
            # the executed source; inability to collect Git is reported honestly.
            report["git_provenance_unavailable"] = str(error)
        publish()
        from isaaclab.app import AppLauncher

        # close() must return so we can publish cleanup and preserve the exit status.
        app = AppLauncher(
            headless=args.operation != "play",
            device=config.task.device,
            fast_shutdown=False,
        ).app
        from isaaclab.envs import ManagerBasedRLEnv
        from parkour_lab.environments.configuration import build_environment_config

        cfg = build_environment_config(
            config.task, evaluation=args.operation != "train"
        )
        cfg.validate()
        import yaml

        (output / "resolved_env.yaml").write_text(
            yaml.dump(cfg.to_dict(), sort_keys=False)
        )
        env = ManagerBasedRLEnv(cfg=cfg)
        if args.operation == "train":
            from parkour_lab.experiment import train

            train(env, app, config, output, report, checkpoint=args.checkpoint)
        else:
            from parkour_lab.artifacts import load_actor, file_sha256
            from parkour_lab.evaluation.runner import evaluate

            loaded = load_actor(args.actor, device=config.task.device)
            report["actor_sha256"] = file_sha256(args.actor)
            evaluate(env, app, loaded, commands, output, report)
        code = 0
    except BaseException as error:
        report.update(status="ERROR", error=repr(error))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        code = finish_session(env, app, report, publish, code)
        print(f"Results: {output}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
