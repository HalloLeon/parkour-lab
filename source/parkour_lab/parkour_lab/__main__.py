"""Small command surface for current training, inference and result inspection."""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import tempfile
import time
import traceback

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
                help="Additional method-owned collection/update cycles; not equal compute or a time guarantee",
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
            if name == "evaluate":
                source.add_argument(
                    "--profile",
                    help="Fixed 11 s flat command diagnostic; not the acceptance bank",
                )
                source.add_argument(
                    "--bank-profile",
                    help="Frozen flat development group: 100 independent attempts, balanced dynamics",
                )
            command.add_argument(
                "--steps", type=int, help="Command playback length (default: 900)"
            )
    export = commands.add_parser("export")
    export.add_argument("checkpoint", type=Path)
    export.add_argument("destination", type=Path)
    export.add_argument(
        "--physics-hz",
        type=int,
        choices=(200, 400),
        help="Explicit new actor physics binding; unchanged weights need native revalidation (default: retain source)",
    )
    analyze = commands.add_parser("analyze")
    analyze.add_argument("run", type=Path)
    analyze.add_argument(
        "--flat-bank",
        action="store_true",
        help="Verify and aggregate flat development profile reports",
    )
    args = parser.parse_args(argv)
    if hasattr(args, "cpu_threads") and args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    if hasattr(args, "command") and not all(math.isfinite(x) for x in args.command):
        parser.error("Commands must be finite")
    profile = getattr(args, "profile", None)
    if profile is None:
        profile = getattr(args, "bank_profile", None)
    if profile is not None:
        from parkour_lab.evaluation.flat import PROFILES

        if profile not in PROFILES:
            parser.error(
                f"Unknown flat profile {profile!r}; choose from {', '.join(PROFILES)}"
            )
    if (
        getattr(args, "profile", None) or getattr(args, "bank_profile", None)
    ) and args.steps is not None:
        parser.error("A profile fixes the tape length; omit --steps")
    if getattr(args, "bank_profile", None) and (
        args.num_envs is not None or args.seed is not None
    ):
        parser.error(
            "A development bank fixes attempt IDs/count/seeds; omit --num-envs and --seed"
        )
    return args


def resolve_config(args):
    artifact = getattr(args, "checkpoint", None) or getattr(args, "actor", None)
    values = {}
    artifact_physics_hz = None
    if artifact is not None:
        from parkour_lab.artifacts import load_artifact

        values = load_artifact(
            artifact, kind="training" if args.operation == "train" else "actor"
        )["config"]
        artifact_physics_hz = ExperimentConfig.from_dict(values).task.physics_hz
    if args.config is not None:
        supplied = json.loads(args.config.read_text())
        explicit = ExperimentConfig.from_dict(supplied)
        if values:
            frozen = ExperimentConfig.from_dict(values)
            if "method" in supplied and explicit.method != frozen.method:
                raise ValueError("Artifact method settings cannot be overridden")
        values = {
            **values,
            **supplied,
            "task": {**values.get("task", {}), **supplied.get("task", {})},
        }
    config = ExperimentConfig.from_dict(values)
    if (
        artifact_physics_hz is not None
        and config.task.physics_hz != artifact_physics_hz
    ):
        raise ValueError(
            "Artifact physics rate cannot be overridden; export a new actor with --physics-hz"
        )
    overrides = {
        key: getattr(args, key)
        for key in ("device", "seed", "num_envs")
        if getattr(args, key) is not None
    }
    if args.operation == "play" and args.num_envs is None:
        overrides["num_envs"] = 1
    if getattr(args, "bank_profile", None):
        overrides.update(
            bank_profile=args.bank_profile, num_envs=100, terrain="flat", seed=0
        )
    config = replace(config, task=replace(config.task, **overrides))
    if getattr(args, "updates", None) is not None:
        config = replace(config, updates=args.updates)
    if getattr(args, "profile", None) is not None:
        if config.task.terrain != "flat" or config.task.episode_length_s <= 11:
            raise ValueError(
                "Flat profiles require flat terrain and an episode longer than 11 s"
            )
    if config.task.bank_profile is not None and (
        args.operation != "evaluate"
        or getattr(args, "bank_profile", None) != config.task.bank_profile
    ):
        raise ValueError(
            "Use --bank-profile explicitly to execute a development manifest"
        )
    return config


def main(argv=None, *, standalone=False):
    """Run a command; only the standalone native CLI owns process termination."""
    args = parse_args(argv)
    if args.operation == "analyze":
        if args.flat_bank:
            from parkour_lab.evaluation.flat import analyze_development_bank

            print(
                json.dumps(
                    analyze_development_bank(args.run), indent=2, allow_nan=False
                )
            )
            return 0
        report = args.run / "report.json" if args.run.is_dir() else args.run
        print(json.dumps(json.loads(report.read_text()), indent=2, allow_nan=False))
        return 0
    if args.operation == "export":
        from parkour_lab.artifacts import export_actor

        print(
            json.dumps(
                export_actor(
                    args.checkpoint, args.destination, physics_hz=args.physics_hz
                ),
                indent=2,
            )
        )
        return 0
    config = resolve_config(args)
    if args.operation == "train":
        config.validate_training()
    from parkour_lab.methods import get_backend

    dependencies = get_backend(config.method.name).dependencies()
    config_data = config.to_dict()
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

        if getattr(args, "profile", None) or getattr(args, "bank_profile", None):
            from parkour_lab.evaluation.flat import profile_commands

            commands = profile_commands(args.profile or args.bank_profile)
        else:
            commands = command_sequence(
                tape=load_tape(args.tape) if args.tape else None,
                command=args.command,
                steps=args.steps if args.steps is not None else 900,
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
    from parkour_lab.runtime.session import exit_native_process, finish_session

    report = {
        "status": "RUNNING",
        "qualified": False,
        "exit_allowed": False,
        "operation": args.operation,
        "config": config_data,
        "dependencies": dependencies,
        "package_sources": package_source_identity(),
    }

    def publish():
        write_json(output / "report.json", report)

    app = env = None
    code = 2
    started = time.monotonic()
    try:
        write_json(output / "config.json", config_data)
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
        from parkour_lab.environments.configuration import build_environment_config
        from parkour_lab.environments.runtime import LocomotionEnv

        cfg = build_environment_config(
            config.task, evaluation=args.operation != "train"
        )
        cfg.validate()
        import yaml

        (output / "resolved_env.yaml").write_text(
            yaml.dump(cfg.to_dict(), sort_keys=False)
        )
        env = LocomotionEnv(cfg=cfg)
        from parkour_lab.environments.rewards import reward_recipe

        report["reward_recipe"] = reward_recipe(env)
        if args.operation == "train":
            from parkour_lab.experiment import train

            train(env, app, config, output, report, checkpoint=args.checkpoint)
        else:
            from parkour_lab.artifacts import load_actor, file_sha256
            from parkour_lab.evaluation.runner import evaluate

            loaded = load_actor(args.actor, device=config.task.device)
            report["actor_sha256"] = file_sha256(args.actor)
            evaluate(
                env,
                app,
                loaded,
                commands,
                output,
                report,
                profile=getattr(args, "profile", None)
                or getattr(args, "bank_profile", None),
            )
        report["transition_observations"] = dict(env.transition_counts)
        code = 0
    except BaseException as error:
        report.update(status="ERROR", error=repr(error))
        if not standalone or app is None:
            raise
        # Print before controlled exit: it intentionally skips exception unwinding.
        traceback.print_exc()
        if isinstance(error, KeyboardInterrupt):
            code = 130
        elif (
            isinstance(error, SystemExit)
            and type(error.code) is int
            and 1 <= error.code <= 255
        ):
            code = error.code
    finally:
        report["wall_seconds"] = time.monotonic() - started
        code = finish_session(env, app, report, publish, code)
        try:
            print(f"Results: {output}", flush=True)
        except (OSError, ValueError):
            if not standalone:
                raise
            code = code or 2
        if standalone and app is not None:
            # Stay inside this frame until exit: even unwinding its native locals
            # can touch plugins already released by app.close().
            exit_native_process(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main(standalone=True))
