"""Attended, simulation-only play of a verified actor artifact (no PPO).

The physical reference and learned archive are validated before launch. After
the complete training configuration matches its archive, explicit live-scene
overrides select one tile and direct keyboard twist. No legacy RMA teleoperator,
privileged steering, hardware driver or per-frame disk capture is used.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import tempfile
import traceback


TERRAINS = ("plane", "rough_flat", "hills", "step_hills", "tilted_ramps")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Physical reference checkpoint")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Learned recurrent checkpoint"
    )
    parser.add_argument(
        "--actor-bundle", type=Path, required=True, help="Motor-bound actor V2"
    )
    parser.add_argument("--terrain", choices=TERRAINS, default="plane")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--livestream",
        type=int,
        choices=(0, 2),
        default=2,
        help="2: server stream; 0: local window",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="CPU source checks only; no simulator/window",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("logs/rsl_rl/go2_operator_refinement"),
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    return args


def apply_live_overrides(cfg, args, *, command_class, recorders):
    """Call only after archive reconstruction; keep motors/actions/physics intact."""
    if cfg.decimation != 4 or cfg.sim.render_interval != 4 or cfg.sim.dt != 0.005:
        raise ValueError("Live input requires stock physics and end-of-frame rendering")
    generator = cfg.scene.terrain.terrain_generator
    selected = [
        sub for sub in generator.sub_terrains.values() if sub.profile == args.terrain
    ]
    if not selected or args.terrain not in TERRAINS:
        raise ValueError(
            "Requested supported live terrain is absent from the source layout"
        )
    tile = copy.deepcopy(selected[0])
    tile.proportion = 1.0
    generator.sub_terrains = {args.terrain: tile}
    generator.seed = cfg.seed = args.seed
    generator.num_rows = generator.num_cols = 1
    generator.difficulty_range = (0.15, 0.35)
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.observations.proprio.enable_corruption = False
    cfg.episode_length_s = 300.0
    # A single tile cannot use the procedural sampler's fixed profile layout.
    # This stock term samples only zero; NativeActorSession owns every command
    # consumed by inference and prevents resampling during active delivery.
    command = cfg.commands.base_velocity
    command.class_type = command_class
    command.heading_command = False
    command.rel_heading_envs = command.rel_standing_envs = 0.0
    command.ranges.heading = None
    command.ranges.lin_vel_x = command.ranges.lin_vel_y = command.ranges.ang_vel_z = (
        0.0,
        0.0,
    )
    command.resampling_time_range = (1.0e9, 1.0e9)
    command.debug_vis = False
    cfg.recorders = recorders
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (3.0, -3.0, 2.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)


def main(argv=None):
    args = parse_args(argv)
    # Source readers import RSL-RL for strict archive verification, but the live
    # NativeActorSession uses only the extracted actor, never a training runner.
    from . import operator_train as training
    from .operator_live import MOTION_KEYS, run_keyboard_actor
    from .operator_runtime import NativeActorSession, actor_bundle_source
    from .teleoperation import validate_operator_display

    try:
        args.reference = args.reference.resolve(strict=True)
        args.checkpoint = args.checkpoint.resolve(strict=True)
        args.actor_bundle = args.actor_bundle.resolve(strict=True)
        for source in (args.reference, args.checkpoint):
            if args.output_parent.resolve().is_relative_to(source.parent):
                raise ValueError(
                    "Live output must be outside immutable source run directories"
                )
        validate_operator_display(
            headless=args.livestream != 0, livestream=args.livestream
        )
        if importlib.metadata.version("rsl-rl-lib") != "3.1.2":
            raise ValueError("Archive preflight requires RSL-RL 3.1.2")
        agent = training.read_yaml_data(args.reference.parent / "params/agent.yaml")
        saved = training.read_yaml_data(args.reference.parent / "params/env.yaml")
        training.load_reference_checkpoint(args.reference, agent)
        training.select_profile(saved, agent, "source")
        identity = training.recurrent_training_identity(args.reference)
        policy, metadata, archived_protocol, sources = (
            training.recurrent_evaluation_source(
                args.checkpoint, identity["physical_reference"]
            )
        )
        del policy
        bundle = actor_bundle_source(args.actor_bundle, sources["checkpoint"], metadata)
    except Exception as error:
        print(f"Live source preflight failed: {error}", flush=True)
        return 2
    protocol = {
        "version": "operator_streamed_actor_v1",
        "source_identity": identity,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sources": sources,
        "checkpoint_learning_updates": metadata["learning_updates"],
        "actor_bundle": bundle,
        "seed": args.seed,
        "num_envs": 1,
        "terrain": args.terrain,
        "difficulty_range": [0.15, 0.35],
        "episode_length_s": 300.0,
        "livestream": args.livestream,
        "device": args.device,
        "motion_keys_body_twist": MOTION_KEYS,
        "lease_s": 0.25,
        "command_clock": "local monotonic receipt time; real key repeats only",
        "policy_clock": "completed native control steps times 0.02 seconds",
        "configuration_check": "full archived recipe before one-tile live overrides",
        "learning_updates": 0,
        "exit_allowed": False,
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "SOURCE_VALIDATED_NOT_SIMULATED",
                    "native_configuration_comparison": "UNRUN",
                    "protocol": protocol,
                },
                indent=2,
                allow_nan=False,
            )
        )
        return 0
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(prefix="operator_live_", dir=args.output_parent)
    ).resolve()
    print(f"Live session: {output}", flush=True)
    app = env = None
    report = {"status": "ERROR", "learning_updates": 0, "exit_allowed": False}
    code = 2
    try:
        training.write_run_provenance(output, __file__)
        training.write_json(output / "live_protocol.json", protocol)
        detected = importlib.metadata.version("isaaclab")
        if detected not in training.PROCEDURAL_ISAACLAB_DISTRIBUTIONS:
            raise ValueError(f"Unsupported Isaac Lab version: {detected}")
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            headless=args.livestream != 0,
            livestream=args.livestream,
            device=args.device,
        ).app
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.envs.mdp import UniformVelocityCommand
        from isaaclab.managers import RecorderManagerBaseCfg
        import yaml

        cfg, _ = training.recurrent_source_configs(
            saved, agent, args, archived_protocol, metadata, args.checkpoint
        )
        training._configure_recurrent_terrain(cfg, (0.15, 0.35))
        apply_live_overrides(
            cfg,
            args,
            command_class=UniformVelocityCommand,
            recorders=RecorderManagerBaseCfg(),
        )
        cfg.validate()
        (output / "resolved_env.yaml").write_text(
            yaml.dump(cfg.to_dict(), sort_keys=False)
        )
        env = ManagerBasedRLEnv(cfg=cfg)
        env.reset(seed=args.seed)
        host = NativeActorSession(
            env,
            args.actor_bundle,
            checkpoint_sha256=sources["checkpoint"],
            source_manifest=metadata["controller_manifest"],
            learning_updates=metadata["learning_updates"],
            artifact_sha256=bundle["sha256"],
        )
        report["motor_verification"] = host.motor_verification
        report.update(run_keyboard_actor(env, host, app))
        if (
            training.recurrent_evaluation_files(args.checkpoint) != sources
            or training.recurrent_training_identity(args.reference) != identity
            or training.file_sha256(args.actor_bundle) != bundle["sha256"]
        ):
            raise ValueError("Source files or runtime changed during the live session")
        report["status"] = "INTERACTIVE_SESSION_FINISHED_NOT_ACCEPTED"
        code = 0
    except KeyboardInterrupt:
        report["status"] = "INTERRUPTED_NOT_ACCEPTED"
        code = 130
    except Exception as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
        print(report["traceback"], flush=True)
    finally:
        for label, resource in (("environment", env), ("application", app)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    report[f"{label}_cleanup_error"] = str(error)
                    report["status"] = "ERROR"
                    code = 2
        training.write_json(output / "report.json", report)
    print(f"{report['status']}: {output / 'report.json'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
