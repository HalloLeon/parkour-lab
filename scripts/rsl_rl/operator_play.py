"""Simulation-only play or bounded headless smoke of a verified actor (no PPO).

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
import os
from pathlib import Path
import tempfile
import traceback


TERRAINS = ("plane", "rough_flat", "hills", "step_hills", "tilted_ramps")
KIT_THREAD_SETTINGS = (
    "/plugins/carb.tasking.plugin/threadCount",
    "/plugins/omni.tbb.globalcontrol/maxThreadCount",
    "/persistent/physics/numThreads",
)


def configure_live_execution(cpu_threads):
    """Bound host worker pools before source readers or Kit initialize them.

    This standalone one-robot process is latency-sensitive, unlike batched PPO.
    These are scheduling limits, not changed physics or input-lease parameters.
    """
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("Live CPU thread budget must be a positive integer")
    logical_count = os.cpu_count()
    try:
        affinity_count = (
            len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        )
    except OSError:
        affinity_count = None
    limit = min(cpu_threads, affinity_count or logical_count or 1)
    environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PXR_WORK_THREAD_LIMIT": str(limit),
    }
    os.environ.update(environment)
    import torch

    torch.set_num_threads(1)
    # Interop can only be configured before its first work. Do not silently
    # continue with an unconstrained pool if this entry point was called late.
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Cannot establish single-threaded live Torch execution")
    return {
        "version": "operator_single_robot_execution_v1",
        "requested_cpu_threads": cpu_threads,
        "cpu_threads": limit,
        "logical_cpu_count": logical_count,
        "affinity_cpu_count": affinity_count,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "environment": environment,
        "kit_args": " ".join(f"--{key}={limit}" for key in KIT_THREAD_SETTINGS),
        "scope": "process-local scheduling limits; no real-time guarantee; USD/BLAS environment requests, not worker-count measurements",
    }


def verify_live_execution(execution):
    """Read back Kit/Torch settings after launcher initialization, before motion."""
    import carb
    import torch

    settings = carb.settings.get_settings()
    actual = {key: settings.get(key) for key in KIT_THREAD_SETTINGS}
    if any(
        type(value) is not int or not 1 <= value <= execution["cpu_threads"]
        for value in actual.values()
    ):
        raise RuntimeError(f"Kit did not retain the live CPU thread budget: {actual}")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Live Torch thread budget changed during simulator startup")
    return {
        "kit_settings": actual,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        # SimulationApp versions may overwrite USD/BLAS environment variables.
        # Report that honestly; env values are not observed native pool sizes.
        "environment_after_launcher": {
            key: os.environ.get(key) for key in execution["environment"]
        },
    }


def _write_live_report(path, report):
    """Replace only a fully written snapshot; preserve the last one on failure."""
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".report-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
        "--cpu-threads",
        type=int,
        default=4,
        help="Per-pool Kit/TBB/PhysX thread cap for one-robot play (default 4, capped by CPU affinity); Torch uses one",
    )
    smoke_modes = parser.add_mutually_exclusive_group()
    smoke_modes.add_argument(
        "--headless-smoke",
        action="store_true",
        help="Fixed 600-step one-robot plane integration with synthetic input and wall-clock watchdog; no GUI or training",
    )
    smoke_modes.add_argument(
        "--headless-functional-smoke",
        action="store_true",
        help="Fixed 600-step plane functional integration with controlled synthetic command time; permits slow hosts, does not validate live timing",
    )
    parser.add_argument(
        "--livestream",
        type=int,
        choices=(0, 2),
        default=None,
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
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.headless_smoke or args.headless_functional_smoke:
        if args.terrain != "plane" or args.livestream not in (None, 0):
            parser.error("Headless smoke modes require plane terrain and no livestream")
        args.livestream = 0
    elif args.livestream is None:
        args.livestream = 2
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
    # The actor has no teacher/value/terrain input. Do not compute these unused
    # groups again on every native step. Keep sensors and physical failure terms
    # intact (notably the single center ray used for terrain-relative clearance).
    cfg.observations.policy = None
    cfg.observations.terrain = None
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
    headless_smoke = args.headless_smoke or args.headless_functional_smoke
    try:
        execution = configure_live_execution(args.cpu_threads)
    except Exception as error:
        print(f"Live execution setup failed: {error}", flush=True)
        return 2
    # Source readers import RSL-RL for strict archive verification, but the live
    # NativeActorSession uses only the extracted actor, never a training runner.
    from . import operator_train as training
    from .operator_live import MOTION_KEYS, run_keyboard_actor
    from .operator_runtime import NativeActorSession, actor_bundle_source
    from .operator_live_probe import HeadlessSmoke, smoke_protocol
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
        if not headless_smoke:
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
        "execution": execution,
        "mode": (
            "headless_functional_smoke"
            if args.headless_functional_smoke
            else "headless_smoke" if args.headless_smoke else "interactive"
        ),
        "headless": headless_smoke or args.livestream != 0,
        "motion_keys_body_twist": MOTION_KEYS,
        "lease_s": 0.25,
        "command_clock": "local monotonic receipt time; real key repeats only",
        "policy_clock": "completed native control steps times 0.02 seconds",
        "configuration_check": "full archived recipe before one-tile live overrides",
        "observation_groups": ["proprio"],
        "learning_updates": 0,
        "exit_allowed": False,
    }
    if args.headless_functional_smoke:
        protocol["smoke"] = smoke_protocol(functional=True)
        protocol["command_clock"] = (
            "controlled synthetic command time advanced by native control steps; "
            "not wall-clock input freshness"
        )
        protocol["wall_clock"] = (
            "local monotonic time for measured host durations and administrative timeout"
        )
        protocol["live_timing_validation"] = "UNRUN"
    elif args.headless_smoke:
        protocol["smoke"] = smoke_protocol()
        protocol["command_clock"] = (
            "local monotonic receipt time; explicitly synthetic scripted key events"
        )
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
    print(
        f"Live host budget: per-pool Kit/TBB/PhysX thread cap {execution['cpu_threads']}; "
        "Torch intra/inter-op = 1/1; simulation device unchanged: "
        f"{args.device}",
        flush=True,
    )
    app = env = probe = None
    report = {"status": "ERROR", "learning_updates": 0, "exit_allowed": False}
    if args.headless_functional_smoke:
        report["live_timing_validation"] = "UNRUN"
    code = 2
    try:
        training.write_run_provenance(output, __file__)
        training.write_json(output / "live_protocol.json", protocol)
        detected = importlib.metadata.version("isaaclab")
        if detected not in training.PROCEDURAL_ISAACLAB_DISTRIBUTIONS:
            raise ValueError(f"Unsupported Isaac Lab version: {detected}")
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            headless=protocol["headless"],
            livestream=args.livestream,
            device=args.device,
            kit_args=execution["kit_args"],
        ).app
        report["execution"] = verify_live_execution(execution)
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
        if headless_smoke:
            probe = (
                HeadlessSmoke(env, functional=True)
                if args.headless_functional_smoke
                else HeadlessSmoke(env)
            )
            report.update(probe.run(host, app))
        else:
            report.update(run_keyboard_actor(env, host, app))
        if (
            training.recurrent_evaluation_files(args.checkpoint) != sources
            or training.recurrent_training_identity(args.reference) != identity
            or training.file_sha256(args.actor_bundle) != bundle["sha256"]
        ):
            raise ValueError("Source files or runtime changed during the live session")
        report["status"] = (
            "HEADLESS_FUNCTIONAL_SMOKE_PASSED_NOT_ACCEPTED"
            if args.headless_functional_smoke
            else (
                "HEADLESS_SMOKE_PASSED_NOT_ACCEPTED"
                if args.headless_smoke
                else "INTERACTIVE_SESSION_FINISHED_NOT_ACCEPTED"
            )
        )
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
        if probe is not None:
            try:
                report["smoke_progress"] = probe.progress()
            except Exception as error:
                report["progress_error"] = str(error)
                report["status"] = "ERROR"
                code = 2
        report["session_status"] = report["status"]
        resources = (("environment", env), ("application", app))
        report["cleanup"] = {
            label: "pending" if resource is not None else "not_created"
            for label, resource in resources
        }
        if code == 0:
            report["status"] = "SESSION_COMPLETED_CLEANUP_PENDING"
        report_path = output / "report.json"

        def persist_report():
            nonlocal code
            try:
                _write_live_report(report_path, report)
            except (OSError, ValueError, TypeError) as error:
                report["report_write_error"] = str(error)
                report["status"] = "ERROR"
                code = 2
                print(f"Cannot save live report {report_path}: {error}", flush=True)

        # Native cleanup can terminate the process without returning to Python.
        # Publish diagnostics first, and never publish a pass while close is pending.
        print(f"Live report (before cleanup): {report_path}", flush=True)
        if probe is not None:
            print(f"Smoke progress: {report.get('smoke_progress', {})}", flush=True)
        persist_report()
        for label, resource in resources:
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    report["cleanup"][label] = "error"
                    report[f"{label}_cleanup_error"] = str(error)
                    report["status"] = "ERROR"
                    code = 2
                else:
                    report["cleanup"][label] = "complete"
                persist_report()
        if code == 0:
            report["status"] = report["session_status"]
        persist_report()
    print(f"{report['status']}: {output / 'report.json'}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
