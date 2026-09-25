"""Frozen exported ROA controller on newly seeded native procedural terrain.

This development screen uses the training sensor-noise model. A separate frozen
training actor is a parity oracle, never an action fallback. Explicit input
diagnostics instead drive the frozen motor with one privileged input replacement;
they are not deployable-controller evaluations. No learning or qualification pass.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import math
from pathlib import Path
import tempfile
import time
import traceback

VERSION = "operator_roa_export_screen_v1"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference", type=Path, help="Original stock48 physical checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Completed ROA learning or estimator-refinement checkpoint",
    )
    parser.add_argument("--controller-artifact", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1043)
    parser.add_argument(
        "--terrain-suite",
        choices=("procedural", "traversal"),
        default="procedural",
        help="Traversal uses four fixed rough ramp/vertical-step fixtures and a straight approach tape",
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        nargs=2,
        default=None,
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument("--num-envs", type=int, choices=(80, 160, 320), default=80)
    parser.add_argument(
        "--diagnostic-input",
        choices=("true_velocity", "privileged_latent"),
        help="Traversal only: frozen privileged input intervention, NOT deployable or qualifying",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("logs/rsl_rl/go2_operator_refinement"),
    )
    args = parser.parse_args(argv)
    if args.diagnostic_input and args.terrain_suite != "traversal":
        parser.error("Input diagnostics require --terrain-suite traversal")
    if args.terrain_suite == "traversal" and args.difficulty is not None:
        parser.error("Traversal has fixed SI geometry; do not supply --difficulty")
    if args.terrain_suite == "procedural" and args.difficulty is None:
        args.difficulty = (0.05, 0.15)
    if (
        args.seed < 0
        or args.cpu_threads < 1
        or (
            args.difficulty is not None
            and not (
                all(math.isfinite(value) for value in args.difficulty)
                and 0 <= args.difficulty[0] <= args.difficulty[1] <= 1
            )
        )
    ):
        parser.error(
            "Require nonnegative seed, positive threads and 0 <= LOW <= HIGH <= 1"
        )
    return args


def main(argv=None):
    args = parse_args(argv)
    from .operator_play import configure_live_execution, verify_live_execution

    execution = configure_live_execution(args.cpu_threads)
    from . import operator_train as training
    from .operator_roa_checkpoint import load_completed_checkpoint, verify_source_files
    from .operator_roa_pilot import PilotEnvironment, validate_events, finish_session
    from .operator_backend import load_controller_artifact
    from .operator_roa_evaluation import (
        evaluate_history,
        COMMAND_TAPE,
        command_tape_sha256,
    )
    from parkour_lab.learning.motor_contract import binding_sha256, verify_runtime_motor
    from parkour_lab.learning.operator_roa import state_sha256

    command_tape = COMMAND_TAPE
    fixtures = None
    if args.terrain_suite == "traversal":
        from . import operator_traversal_terrain as traversal
        from .operator_traversal_probe import (
            TraversalProbe,
            COMMAND_TAPE as TRAVERSAL_TAPE,
        )

        command_tape = TRAVERSAL_TAPE
        fixtures = traversal.preflight(args.seed)
    args.reference = args.reference.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.controller_artifact = args.controller_artifact.resolve(strict=True)
    if any(
        args.output_parent.resolve().is_relative_to(path.parent)
        for path in (args.reference, args.checkpoint)
    ):
        raise ValueError("Screen output must be outside immutable source runs")
    identity = training.recurrent_training_identity(args.reference)
    agent = training.read_yaml_data(args.reference.parent / "params/agent.yaml")
    saved = training.read_yaml_data(args.reference.parent / "params/env.yaml")
    training.load_reference_checkpoint(args.reference, agent)
    policy, contract, _, source = load_completed_checkpoint(args.checkpoint)
    loaded = load_controller_artifact(
        args.controller_artifact, backend="roa_history_v1"
    )
    if (
        source["physical_reference"] != identity["physical_reference"]
        or loaded.source["physical_reference"] != source["physical_reference"]
        or loaded.source["checkpoint_sha256"] != source["checkpoint_sha256"]
        or loaded.source["learning_updates"] != source["learning_updates"]
        or binding_sha256(loaded.motor_contract) != binding_sha256(contract)
        or args.seed == source["training_seed"]
    ):
        raise ValueError(
            "Require matching ROA artifact, motor, physical source and a new terrain seed"
        )
    verify_source_files(source)
    if (
        training.recurrent_training_identity(args.reference) != identity
        or training.file_sha256(args.controller_artifact) != loaded.artifact_sha256
    ):
        raise ValueError("Screen sources changed during preflight")
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(prefix="operator_roa_screen_", dir=args.output_parent)
    ).resolve()
    protocol = {
        "version": VERSION,
        "source_identity": identity,
        "checkpoint_source": source,
        "controller": loaded.receipt(),
        "controller_artifact": str(args.controller_artifact),
        "seed": args.seed,
        "terrain_generator_seed": args.seed,
        "num_envs": args.num_envs,
        "terrain_suite": args.terrain_suite,
        "difficulty_range": list(args.difficulty) if args.difficulty else None,
        "traversal_geometry": fixtures,
        "command_tape": command_tape,
        "command_tape_sha256": command_tape_sha256(command_tape),
        "sensing": "Original noisy45 proprioceptive frame, unchanged scales; controller receives no true velocity, dynamics or scan",
        "scope": (
            "Four fixed full-width rough ramp/vertical-step layouts; yaw-zero fixed approach, no feedback steering. Root/foot/contact diagnostics, not supported completion, sim-to-real or exit qualification"
            if fixtures
            else "Newly generated five-profile terrain, frozen causal development screen; not high-step/stair, sim-to-real or exit qualification"
        ),
        "learning_updates": 0,
        "exit_allowed": False,
    }
    if args.diagnostic_input:
        protocol.update(
            diagnostic_input=args.diagnostic_input,
            action_source="PRIVILEGED_SIMULATION_DIAGNOSTIC_NOT_DEPLOYABLE",
            sensing="Unchanged noisy45 frames/history; replace only motor velocity with true native body-COM velocity OR history latent with privileged dynamics latent, as selected",
            scope=protocol["scope"]
            + "; simulator-input intervention, NOT a causal controller test or an oracle upper bound",
        )
    training.write_json(output / "evaluation_protocol.json", protocol)
    report = {
        "status": "SOURCE_VALIDATED_NOT_SIMULATED",
        "learning_updates": 0,
        "exit_allowed": False,
    }
    if args.diagnostic_input:
        report.update(
            diagnostic_input=args.diagnostic_input,
            action_source=protocol["action_source"],
        )

    def publish():
        training.write_json(output / "report.json", report)

    publish()
    print(f"ROA screen: {output}", flush=True)
    if args.validate_only:
        return 0
    app = env = host = None
    code = 2
    started = time.monotonic()
    try:
        training.write_run_provenance(output, __file__)
        if (
            importlib.metadata.version("isaaclab")
            not in training.PROCEDURAL_ISAACLAB_DISTRIBUTIONS
        ):
            raise ValueError("Unsupported Isaac Lab version")
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            headless=True, device=args.device, kit_args=execution["kit_args"]
        ).app
        report["execution"] = verify_live_execution(execution)
        from isaaclab.envs import ManagerBasedRLEnv
        import yaml

        args.iterations = 0
        cfg, _ = training.proprioceptive_procedural_configs(saved, agent, args)
        if fixtures:
            traversal.configure(cfg, args.seed, fixtures)
        else:
            training._configure_recurrent_terrain(cfg, args.difficulty)
        cfg.seed = cfg.scene.terrain.terrain_generator.seed = args.seed
        validate_events(cfg)
        cfg.validate()
        (output / "resolved_env.yaml").write_text(
            yaml.dump(cfg.to_dict(), sort_keys=False)
        )
        env = ManagerBasedRLEnv(cfg=cfg)
        if fixtures:
            report["native_geometry"] = traversal.native_receipts()
            if report["native_geometry"] != fixtures:
                raise ValueError("Require all four constructed traversal mesh receipts")
        host = PilotEnvironment(env, app)
        if binding_sha256(contract["binding"]) != binding_sha256(
            host.motor_contract["binding"]
        ):
            raise ValueError("ROA artifact motor differs from the native screen motor")
        report["exported_motor_verification"] = verify_runtime_motor(
            contract, loaded.controller.spec.manifest(), host.bridge.binding
        )
        loaded = load_controller_artifact(
            args.controller_artifact,
            backend="roa_history_v1",
            device=env.device,
            expected_sha256=protocol["controller"]["artifact_sha256"],
        )
        policy.to(env.device)
        if state_sha256(policy) != source["policy_state_sha256"]:
            raise ValueError("Frozen reference weights changed")
        report["status"] = "RUNNING_NOT_QUALIFIED"
        publish()
        probe = (
            TraversalProbe(env, output / "traversal_trace.npz") if fixtures else None
        )
        report["evaluation"] = evaluate_history(
            host,
            policy,
            seed=args.seed,
            controller=loaded.controller,
            command_tape=command_tape,
            observer=probe,
            diagnostic_input=args.diagnostic_input,
            diagnostic_output=(
                output / "input_diagnostic_trace.npz" if args.diagnostic_input else None
            ),
        )
        if probe is not None:
            report["traversal"] = probe.report()
        report["evaluation"]["scope"] = protocol["scope"]
        verify_source_files(source)
        if training.recurrent_training_identity(args.reference) != identity:
            raise RuntimeError("Physical source or runtime changed during screen")
        if training.file_sha256(args.controller_artifact) != loaded.artifact_sha256:
            raise RuntimeError("Controller artifact changed during screen")
        report["status"] = (
            "ROA_INPUT_DIAGNOSTIC_COMPLETED_NOT_DEPLOYABLE"
            if args.diagnostic_input
            else "ROA_EXPORTED_SCREEN_COMPLETED_NOT_QUALIFIED"
        )
        code = 0
    except Exception as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
        traceback.print_exc()
    finally:
        report["wall_seconds"] = time.monotonic() - started
        if host is not None:
            report["motor_delivery"] = host.bridge.progress()
            report["motor_verification"] = host.bridge.motor_verification
        try:
            print(f"ROA screen report: {output / 'report.json'}", flush=True)
        finally:
            code = finish_session(env, app, report, publish, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
