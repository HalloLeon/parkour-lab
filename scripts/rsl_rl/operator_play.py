"""Simulation-only play, scripted demo or headless smoke of an actor (no PPO).

The physical reference and learned archive are validated before launch. After
the complete training configuration matches its archive, explicit live-scene
overrides select one tile and direct body-twist commands. No legacy RMA teleoperator,
privileged steering, hardware driver or per-frame disk capture is used.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
import traceback


TERRAINS = (
    "plane",
    "rough_flat",
    "hills",
    "step_hills",
    "tilted_ramps",
    "rough_stress",
)
DEFAULT_DIFFICULTY = (0.15, 0.35)
NONPLANE_DEMO_POSE_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "yaw": (math.pi / 2, math.pi / 2),
}
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
        "--checkpoint",
        type=Path,
        required=True,
        help="Archived recurrent scene recipe; also the policy source with --actor-bundle",
    )
    controllers = parser.add_mutually_exclusive_group(required=True)
    controllers.add_argument(
        "--actor-bundle",
        type=Path,
        help="Legacy motor-bound actor V2 of the exact --checkpoint",
    )
    controllers.add_argument(
        "--controller-artifact",
        type=Path,
        help="Explicit inference artifact independent of scene weights; must match the scene motor contract",
    )
    from .operator_backend import BACKENDS

    parser.add_argument(
        "--controller-backend",
        choices=tuple(BACKENDS),
        help="Trusted in-code artifact adapter; required with --controller-artifact",
    )
    parser.add_argument("--terrain", choices=TERRAINS, default="plane")
    parser.add_argument(
        "--difficulty",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=DEFAULT_DIFFICULTY,
        help="Ordered terrain difficulty bounds in [0, 1] (default 0.15 0.35); headless smokes retain the default",
    )
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
    smoke_modes.add_argument(
        "--scripted-demo",
        action="store_true",
        help="Bounded terrain-specific body-twist sequence for streamed or local viewing; no keyboard input or live-timing validation",
    )
    smoke_modes.add_argument(
        "--replay-commands",
        type=Path,
        help="Replay a complete validated simulation command tape by control-step index; no keyboard or training",
    )
    parser.add_argument(
        "--record-commands",
        action="store_true",
        help="Record completed applied body-twist commands into this run's command_tape.json; manual reset and native termination invalidate the tape",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run scripted demo or command replay without a window or stream; not interactive keyboard control",
    )
    parser.add_argument(
        "--livestream",
        type=int,
        choices=(0, 2),
        default=None,
        help="2: server stream; 0: local window",
    )
    parser.add_argument(
        "--keyboard-controls",
        choices=("single-key", "legacy"),
        default="single-key",
        help="Interactive controller (default single-key); ignored for scripted demo, headless smokes use legacy synthetic input",
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
    if bool(args.controller_artifact) != bool(args.controller_backend):
        parser.error(
            "--controller-artifact and --controller-backend require each other"
        )
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if not all(math.isfinite(value) for value in args.difficulty) or not (
        0.0 <= args.difficulty[0] <= args.difficulty[1] <= 1.0
    ):
        parser.error(
            "--difficulty requires finite ordered bounds: 0 <= LOW <= HIGH <= 1"
        )
    args.difficulty = tuple(args.difficulty)
    if args.headless and not (args.scripted_demo or args.replay_commands):
        parser.error("--headless requires --scripted-demo or --replay-commands")
    if args.headless and args.livestream == 2:
        parser.error("--headless cannot be combined with --livestream 2")
    if args.record_commands and (args.headless_smoke or args.headless_functional_smoke):
        parser.error("--record-commands does not support headless smoke reset tapes")
    if args.terrain == "rough_stress":
        if not (args.scripted_demo or args.replay_commands):
            parser.error(
                "rough_stress is evaluation-only and requires --scripted-demo or --replay-commands"
            )
        if args.difficulty[0] <= 0.0 or args.difficulty[0] != args.difficulty[1]:
            parser.error(
                "rough_stress requires fixed positive --difficulty LOW HIGH with LOW == HIGH"
            )
    if args.headless_smoke or args.headless_functional_smoke:
        if args.terrain != "plane" or args.livestream not in (None, 0):
            parser.error("Headless smoke modes require plane terrain and no livestream")
        if args.difficulty != DEFAULT_DIFFICULTY:
            parser.error(
                "Headless smoke modes require the default --difficulty 0.15 0.35"
            )
        args.livestream = 0
    elif args.headless:
        args.livestream = 0
    elif args.livestream is None:
        args.livestream = 2
    return args


def _same_controller_receipt(recorded, current):
    """Exact JSON identity, without Python's True == 1 or 1 == 1.0 coercion."""
    return json.dumps(recorded, sort_keys=True, allow_nan=False) == json.dumps(
        current, sort_keys=True, allow_nan=False
    )


def _validate_replay_scene(args, tape):
    """Keep declared recording context fixed; only one known spawn override exists."""
    metadata = tape["metadata"]
    required = {
        "terrain",
        "seed",
        "difficulty_range",
        "spawn_override",
        "resolved_scene_configuration_sha256",
        "runtime_source_sha256",
        "portable_motor_sha256",
        "controller_interface_sha256",
        "source_mode",
    }
    generic = args.controller_artifact is not None
    if generic != ("controller" in metadata):
        raise ValueError("Replay controller selection mode differs from the recording")
    checkpoint_key = "scene_checkpoint_sha256" if generic else "checkpoint_sha256"
    artifact_key = "controller_artifact" if generic else "actor_bundle"
    required |= {checkpoint_key, artifact_key}
    if not required <= metadata.keys():
        raise ValueError(
            "Replay requires the recorded scene and runtime identity metadata"
        )
    bundle = metadata[artifact_key]
    if (
        type(bundle) is not dict
        or set(bundle) != {"path", "sha256"}
        or type(bundle["path"]) is not str
        or not bundle["path"]
        or type(metadata["source_mode"]) is not str
        or metadata["source_mode"]
        not in ("interactive", "scripted_demo", "command_replay")
    ):
        raise ValueError(
            "Replay requires a recorded actor bundle and simulation source mode"
        )
    identities = {
        key: metadata[key]
        for key in (
            "resolved_scene_configuration_sha256",
            "runtime_source_sha256",
            "portable_motor_sha256",
            checkpoint_key,
            "controller_interface_sha256",
        )
    }
    identities[f"{artifact_key}.sha256"] = bundle["sha256"]
    for key, value in identities.items():
        if (
            type(value) is not str
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError(f"Replay requires a recorded SHA256 identity: {key}")
    if metadata["terrain"] != args.terrain:
        raise ValueError("Replay terrain differs from the recorded command tape")
    if type(metadata["seed"]) is not int or metadata["seed"] != args.seed:
        raise ValueError("Replay seed differs from the recorded command tape")
    bounds = metadata["difficulty_range"]
    if (
        type(bounds) is not list
        or len(bounds) != 2
        or any(type(value) not in (int, float) for value in bounds)
        or tuple(bounds) != args.difficulty
    ):
        raise ValueError("Replay difficulty differs from the recorded command tape")
    override = metadata["spawn_override"]
    if override is not None:
        if (
            args.terrain == "plane"
            or type(override) is not dict
            or set(override) != {"reset_base_pose_range", "scope"}
            or type(override["scope"]) is not str
            or json.dumps(override["reset_base_pose_range"], sort_keys=True)
            != json.dumps(NONPLANE_DEMO_POSE_RANGE, sort_keys=True)
        ):
            raise ValueError("Unknown or incompatible command replay spawn override")
        args._command_replay_spawn = copy.deepcopy(NONPLANE_DEMO_POSE_RANGE)
    if args.terrain == "rough_stress" and override is None:
        raise ValueError(
            "rough_stress replay requires the recorded supported demo spawn"
        )


def _scene_configuration_sha256(resolved):
    """Bind resolved scene/physics while excluding only administrative/view settings."""
    import yaml

    normalized = copy.deepcopy(resolved)
    for key in (
        "episode_length_s",
        "log_dir",
        "recorders",
        "viewer",
        "ui_window_class_type",
    ):
        normalized.pop(key, None)
    if "sim" in normalized:
        for key in ("log_dir", "logging_level", "save_logs_to_file", "render"):
            normalized["sim"].pop(key, None)
    return hashlib.sha256(yaml.dump(normalized, sort_keys=True).encode()).hexdigest()


def _runtime_source_sha256(identity):
    return hashlib.sha256(
        json.dumps(identity["runtime"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _rough_stress_geometry(args, *, size=(16.0, 16.0)):
    """Build deterministic evaluation geometry without simulator initialization."""
    from .operator_stress_terrain import (
        FULL_HEIGHT_BOUND,
        build_stress_surface,
        stress_terrain_envelope,
    )

    difficulty = tuple(args.difficulty)
    if (
        not (
            getattr(args, "scripted_demo", False)
            or getattr(args, "_command_replay_spawn", None)
        )
        or len(difficulty) != 2
        or not all(math.isfinite(value) for value in difficulty)
        or not 0.0 < difficulty[0] == difficulty[1] <= 1.0
    ):
        raise ValueError(
            "rough_stress requires scripted demo or its recorded replay spawn and fixed positive difficulty"
        )
    surface = build_stress_surface(difficulty[0], seed=args.seed, size=size)
    return {
        "envelope": stress_terrain_envelope(),
        "realized": surface["metadata"],
        "fallback_floor_bound": {
            "minimum_surface_z_m": -FULL_HEIGHT_BOUND * difficulty[0],
            "scope": "Geometry-derived fallback floor only; minimum clearance, fall margin and tilt limit unchanged",
        },
    }


def apply_live_overrides(cfg, args, *, command_class, recorders):
    """Call only after archive reconstruction; keep motors/actions/physics intact."""
    if cfg.decimation != 4 or cfg.sim.render_interval != 4 or cfg.sim.dt != 0.005:
        raise ValueError("Live input requires stock physics and end-of-frame rendering")
    generator = cfg.scene.terrain.terrain_generator
    template_profile = "rough_flat" if args.terrain == "rough_stress" else args.terrain
    selected = [
        sub
        for sub in generator.sub_terrains.values()
        if sub.profile == template_profile
    ]
    if not selected or args.terrain not in TERRAINS:
        raise ValueError(
            "Requested supported live terrain is absent from the source layout"
        )
    tile = copy.deepcopy(selected[0])
    tile.proportion = 1.0
    if args.terrain == "rough_stress":
        from dataclasses import fields

        from isaaclab.utils import configclass
        from .operator_stress_terrain import RESOLUTION, stress_terrain

        geometry = _rough_stress_geometry(args, size=tuple(generator.size))
        preflight_geometry = getattr(args, "_rough_stress_geometry", None)
        if preflight_geometry is not None and geometry != preflight_geometry:
            raise ValueError(
                "Native rough_stress geometry differs from the CPU preflight manifest"
            )

        @configclass
        class StressTerrainCfg(type(tile)):
            # Native cfg.copy() uses dataclasses.replace, so this must be a
            # declared field rather than a dynamically attached attribute.
            expected_height_sha256: str = ""

        tile = StressTerrainCfg(
            **{
                field.name: copy.deepcopy(getattr(tile, field.name))
                for field in fields(tile)
                if field.init
            },
            expected_height_sha256=geometry["realized"]["height_float64_sha256"],
        )
        tile.function = stress_terrain
        tile.profile = "rough_stress"
        tile.variant = 0
        tile.seed = args.seed
        generator.horizontal_scale = RESOLUTION
        generator.use_cache = False
        cfg.terminations.procedural_physical_failure.params["minimum_surface_z_m"] = (
            geometry["fallback_floor_bound"]["minimum_surface_z_m"]
        )
    generator.sub_terrains = {args.terrain: tile}
    generator.seed = cfg.seed = args.seed
    generator.num_rows = generator.num_cols = 1
    generator.difficulty_range = tuple(getattr(args, "difficulty", DEFAULT_DIFFICULTY))
    if getattr(args, "scripted_demo", False) and args.terrain != "plane":
        cfg.events.reset_base.params["pose_range"].update(NONPLANE_DEMO_POSE_RANGE)
    if getattr(args, "_command_replay_spawn", None) is not None:
        cfg.events.reset_base.params["pose_range"].update(args._command_replay_spawn)
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
    cfg.episode_length_s = max(
        300.0, getattr(args, "_command_replay_steps", 0) * 0.02 + 1.0
    )
    # NativeControllerSession owns every command, independent of terrain layout.
    from .operator_runtime import configure_external_command

    configure_external_command(cfg.commands.base_velocity, command_class=command_class)
    cfg.recorders = recorders
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.asset_name = "robot"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (3.0, -3.0, 2.0)
    cfg.viewer.lookat = (0.0, 0.0, 0.0)


def main(argv=None):
    args = parse_args(argv)
    headless_smoke = args.headless_smoke or args.headless_functional_smoke
    replay_tape = replay_file_sha256 = None
    controller_receipt = archived_controller_motor_check = None
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

    if args.controller_artifact:
        from .operator_backend import load_controller_artifact
        from .operator_runtime import NativeControllerSession, validate_controller_scene
    from .operator_live_probe import HeadlessSmoke, smoke_protocol
    from .operator_simple_input import SINGLE_KEY_MOTION_KEYS, simple_keyboard_protocol
    from .teleoperation import validate_operator_display

    if args.scripted_demo:
        from .operator_demo import ScriptedDemo, demo_protocol
    if args.record_commands or args.replay_commands:
        from .operator_command_tape import CommandRecording, CommandReplay

    try:
        args.reference = args.reference.resolve(strict=True)
        args.checkpoint = args.checkpoint.resolve(strict=True)
        artifact_path = (args.controller_artifact or args.actor_bundle).resolve(
            strict=True
        )
        immutable_sources = [args.reference, args.checkpoint, artifact_path]
        if args.replay_commands:
            from parkour_lab.learning.command_tape import load_tape

            args.replay_commands = args.replay_commands.resolve(strict=True)
            replay_file_sha256 = training.file_sha256(args.replay_commands)
            replay_tape = load_tape(args.replay_commands)
            if training.file_sha256(args.replay_commands) != replay_file_sha256:
                raise ValueError("Command tape changed during source preflight")
            args._command_replay_steps = replay_tape["steps"]
            _validate_replay_scene(args, replay_tape)
            immutable_sources.append(args.replay_commands)
        for source in immutable_sources:
            if args.output_parent.resolve().is_relative_to(source.parent):
                raise ValueError(
                    "Live output must be outside immutable source run directories"
                )
        if not headless_smoke and not args.headless:
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
        runtime_source_sha256 = _runtime_source_sha256(identity)
        if (
            replay_tape
            and replay_tape["metadata"]["runtime_source_sha256"]
            != runtime_source_sha256
        ):
            raise ValueError(
                "Replay runtime source differs from the recorded command tape"
            )
        policy, metadata, archived_protocol, sources = (
            training.recurrent_evaluation_source(
                args.checkpoint, identity["physical_reference"]
            )
        )
        del policy
        if args.controller_artifact:
            loaded = load_controller_artifact(
                artifact_path, backend=args.controller_backend, device="cpu"
            )
            archived_controller_motor_check = validate_controller_scene(
                loaded, metadata
            )
            controller_receipt = loaded.receipt()
            bundle = {"path": str(artifact_path), "sha256": loaded.artifact_sha256}
            if replay_tape:
                recorded = replay_tape["metadata"]
                if (
                    recorded["scene_checkpoint_sha256"] != sources["checkpoint"]
                    or recorded["controller_artifact"]["sha256"] != bundle["sha256"]
                    or not _same_controller_receipt(
                        recorded["controller"], controller_receipt
                    )
                    or recorded["controller_interface_sha256"]
                    != controller_receipt["controller_interface_sha256"]
                ):
                    raise ValueError(
                        "Replay selected controller or scene source differs from recording"
                    )
            del loaded
        else:
            bundle = actor_bundle_source(artifact_path, sources["checkpoint"], metadata)
        if args.terrain == "rough_stress":
            args._rough_stress_geometry = _rough_stress_geometry(args)
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
        "difficulty_range": list(args.difficulty),
        "episode_length_s": max(
            300.0, getattr(args, "_command_replay_steps", 0) * 0.02 + 1.0
        ),
        "livestream": args.livestream,
        "device": args.device,
        "execution": execution,
        "mode": (
            "headless_functional_smoke"
            if args.headless_functional_smoke
            else "headless_smoke" if args.headless_smoke else "interactive"
        ),
        "headless": headless_smoke or args.headless or args.livestream != 0,
        "keyboard_controls": "legacy" if headless_smoke else args.keyboard_controls,
        "motion_keys_body_twist": MOTION_KEYS,
        "lease_s": 0.25,
        "command_clock": "local monotonic receipt time; real key repeats only",
        "policy_clock": "completed native control steps times 0.02 seconds",
        "configuration_check": "full archived recipe before one-tile live overrides",
        "observation_groups": ["proprio"],
        "learning_updates": 0,
        "exit_allowed": False,
    }
    if controller_receipt is not None:
        for key in ("checkpoint", "checkpoint_sources", "checkpoint_learning_updates"):
            protocol[f"scene_{key}"] = protocol.pop(key)
        protocol["controller_artifact"] = protocol.pop("actor_bundle")
        protocol["controller"] = controller_receipt
        protocol["archived_controller_motor_check"] = archived_controller_motor_check
    if args.terrain == "rough_stress":
        protocol["rough_stress_geometry"] = args._rough_stress_geometry
    if args.scripted_demo:
        protocol.update(
            mode="scripted_demo",
            command_clock="completed native control steps times 0.02 seconds",
            command_source="predefined body-twist sequence; no keyboard or network input",
            keyboard_controls="unused",
            live_timing_validation="UNRUN",
            demo=demo_protocol(terrain=args.terrain),
        )
        if args.terrain != "plane":
            protocol["spawn_override"] = {
                "reset_base_pose_range": copy.deepcopy(NONPLANE_DEMO_POSE_RANGE),
                "scope": "scripted non-plane demo only; fixed initial pose, not feedback steering",
            }
        protocol.pop("motion_keys_body_twist")
        protocol.pop("lease_s")
    elif args.replay_commands:
        protocol.update(
            mode="command_replay",
            command_clock="completed native control steps times 0.02 seconds",
            command_source="recorded applied body-twist sequence; no keyboard or network input",
            keyboard_controls="unused",
            live_timing_validation="UNRUN",
            command_replay={
                "path": str(args.replay_commands),
                "file_sha256": replay_file_sha256,
                "tape_sha256": replay_tape["sha256"],
                "steps": replay_tape["steps"],
                "source_metadata": replay_tape["metadata"],
                "scope": "Command-only simulation replay; current actor and scene are recorded separately, not a physical trajectory reproduction or acceptance test.",
            },
        )
        protocol.pop("motion_keys_body_twist")
        protocol.pop("lease_s")
        if getattr(args, "_command_replay_spawn", None) is not None:
            protocol["spawn_override"] = {
                "reset_base_pose_range": copy.deepcopy(args._command_replay_spawn),
                "scope": "Supported fixed spawn inherited from the recorded scripted demo; not feedback steering or physical trajectory reproduction",
            }
    elif not headless_smoke and args.keyboard_controls == "single-key":
        protocol["keyboard"] = simple_keyboard_protocol()
        protocol["motion_keys_body_twist"] = SINGLE_KEY_MOTION_KEYS
        protocol["motion_keys_body_twist_scale"] = (
            "Base twists multiplied by selected speed; scale captured on fresh press"
        )
        protocol.pop("lease_s")
        protocol["command_clock"] = (
            "local monotonic receipt time; fresh press starts bounded initial grace, "
            "real key repeats renew the shorter lease"
        )
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
    if headless_smoke:
        protocol["smoke"].update(
            keyboard_controls="legacy",
            input_coverage="legacy synthetic input only; not the interactive single-key default",
            single_key_controls_validation="UNRUN",
        )
    if args.record_commands:
        protocol["command_recording"] = {
            "filename": "command_tape.json",
            "maximum_steps": 30000,
            "scope": "Completed applied body twists only; no raw key/network events, observations, actions, recurrent state or PPO. Manual reset/native termination invalidates the tape.",
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
    print(
        f"Live host budget: per-pool Kit/TBB/PhysX thread cap {execution['cpu_threads']}; "
        "Torch intra/inter-op = 1/1; simulation device unchanged: "
        f"{args.device}",
        flush=True,
    )
    app = env = host = probe = demo = replay = recording = None
    report = {"status": "ERROR", "learning_updates": 0, "exit_allowed": False}
    if args.headless_functional_smoke or args.scripted_demo or args.replay_commands:
        report["live_timing_validation"] = "UNRUN"
    if args.scripted_demo:
        report["demo_progress"] = {"status": "NOT_STARTED"}
    if args.replay_commands:
        report["replay_progress"] = {"status": "NOT_STARTED"}
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
        training._configure_recurrent_terrain(cfg, args.difficulty)
        apply_live_overrides(
            cfg,
            args,
            command_class=UniformVelocityCommand,
            recorders=RecorderManagerBaseCfg(),
        )
        cfg.validate()
        resolved_configuration = cfg.to_dict()
        scene_configuration_sha256 = _scene_configuration_sha256(resolved_configuration)
        if (
            replay_tape
            and replay_tape["metadata"]["resolved_scene_configuration_sha256"]
            != scene_configuration_sha256
        ):
            raise ValueError(
                "Replay resolved scene configuration differs from the recorded command tape"
            )
        report["resolved_scene_configuration_sha256"] = scene_configuration_sha256
        (output / "resolved_env.yaml").write_text(
            yaml.dump(resolved_configuration, sort_keys=False)
        )
        if args.terrain == "rough_stress":
            from .operator_stress_terrain import (
                clear_stress_terrain_receipt,
                stress_terrain_receipt,
            )

            clear_stress_terrain_receipt()
        env = ManagerBasedRLEnv(cfg=cfg)
        if args.terrain == "rough_stress":
            receipt = stress_terrain_receipt()
            report["rough_stress_native_geometry"] = receipt
            if (
                not isinstance(receipt, dict)
                or type(receipt.get("calls")) is not int
                or receipt["calls"] != 1
                or receipt.get("seed") != args.seed
                or receipt.get("difficulty") != args.difficulty[0]
                or receipt.get("height_float64_sha256")
                != protocol["rough_stress_geometry"]["realized"][
                    "height_float64_sha256"
                ]
            ):
                raise ValueError(
                    "Native rough_stress geometry callback receipt is missing or mismatched"
                )
        env.reset(seed=args.seed)
        if args.controller_artifact:
            loaded = load_controller_artifact(
                artifact_path,
                backend=args.controller_backend,
                device=args.device,
                expected_sha256=bundle["sha256"],
            )
            if not _same_controller_receipt(loaded.receipt(), controller_receipt):
                raise ValueError("Controller identity changed after source preflight")
            host = NativeControllerSession(
                env,
                loaded.controller,
                loaded.motor_contract,
                preserve_native_raw=loaded.preserve_native_raw,
            )
            if (
                host.session.interface_sha256
                != controller_receipt["controller_interface_sha256"]
            ):
                raise ValueError(
                    "Native controller interface differs from the preflight receipt"
                )
        else:
            host = NativeActorSession(
                env,
                artifact_path,
                checkpoint_sha256=sources["checkpoint"],
                source_manifest=metadata["controller_manifest"],
                learning_updates=metadata["learning_updates"],
                artifact_sha256=bundle["sha256"],
            )
        report["motor_verification"] = host.motor_verification
        if args.record_commands:
            controller_source = (
                {
                    "controller_artifact": copy.deepcopy(bundle),
                    "controller": copy.deepcopy(controller_receipt),
                    "scene_checkpoint_sha256": sources["checkpoint"],
                }
                if controller_receipt is not None
                else {
                    "actor_bundle": copy.deepcopy(bundle),
                    "checkpoint_sha256": sources["checkpoint"],
                }
            )
            recording = CommandRecording(
                {
                    **controller_source,
                    "controller_interface_sha256": host.session.interface_sha256,
                    "portable_motor_sha256": host.motor_verification[
                        "portable_motor_sha256"
                    ],
                    "source_mode": protocol["mode"],
                    "terrain": args.terrain,
                    "difficulty_range": list(args.difficulty),
                    "seed": args.seed,
                    "spawn_override": copy.deepcopy(protocol.get("spawn_override")),
                    "resolved_scene_configuration_sha256": scene_configuration_sha256,
                    "runtime_source_sha256": runtime_source_sha256,
                }
            )
        recording_options = {"recording": recording} if recording is not None else {}
        if args.scripted_demo:
            demo = ScriptedDemo(env, terrain=args.terrain)
            report.update(demo.run(host, app, **recording_options))
        elif args.replay_commands:
            replay = CommandReplay(env, replay_tape)
            report["command_replay"] = replay.protocol
            report.update(replay.run(host, app, **recording_options))
        elif headless_smoke:
            probe = (
                HeadlessSmoke(env, functional=True)
                if args.headless_functional_smoke
                else HeadlessSmoke(env)
            )
            report.update(probe.run(host, app))
        else:
            report.update(
                run_keyboard_actor(
                    env, host, app, controls=args.keyboard_controls, **recording_options
                )
            )
        if (
            training.recurrent_evaluation_files(args.checkpoint) != sources
            or training.recurrent_training_identity(args.reference) != identity
            or training.file_sha256(artifact_path) != bundle["sha256"]
            or (
                args.replay_commands
                and training.file_sha256(args.replay_commands) != replay_file_sha256
            )
        ):
            raise ValueError("Source files or runtime changed during the live session")
        if args.scripted_demo:
            report["status"] = "SCRIPTED_DEMO_COMPLETED_NOT_ACCEPTED"
        elif args.replay_commands:
            report["status"] = "COMMAND_REPLAY_COMPLETED_NOT_ACCEPTED"
        elif args.headless_functional_smoke:
            report["status"] = "HEADLESS_FUNCTIONAL_SMOKE_PASSED_NOT_ACCEPTED"
        elif args.headless_smoke:
            report["status"] = "HEADLESS_SMOKE_PASSED_NOT_ACCEPTED"
        else:
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
        if host is not None:
            try:
                report["motor_delivery"] = host.motor.progress()
            except Exception as error:
                report["motor_delivery_error"] = str(error)
                report["status"] = "ERROR"
                code = 2
        if replay is not None:
            try:
                report["replay_progress"] = replay.progress()
            except Exception as error:
                report["replay_progress_error"] = str(error)
                report["status"] = "ERROR"
                code = 2
        if demo is not None:
            try:
                report["demo_progress"] = demo.progress()
            except Exception as error:
                report["demo_progress_error"] = str(error)
                report["status"] = "ERROR"
                code = 2
        if probe is not None:
            try:
                report["smoke_progress"] = probe.progress()
            except Exception as error:
                report["progress_error"] = str(error)
                report["status"] = "ERROR"
                code = 2
        if recording is not None:
            try:
                report["command_recording"] = recording.finish(
                    output / "command_tape.json",
                    completed=code == 0,
                    error=(
                        None if code == 0 else report.get("error", report["status"])
                    ),
                )
                if code == 0 and not report["command_recording"]["complete"]:
                    raise RuntimeError(
                        "Command recording did not produce a complete tape"
                    )
            except Exception as error:
                report["command_recording_error"] = str(error)
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
        if args.scripted_demo:
            print(f"Demo progress: {report['demo_progress']}", flush=True)
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
