"""Training orchestration: configuration, native lifecycle and current snapshots.

The host manages run lifecycle; each method owns learning and persistence.
"""

import json
import time
from contextlib import ExitStack
from pathlib import Path

from parkour_lab.provenance import write_json


def train(env, app, config, output, report, *, checkpoint=None, diagnose_training=False):
    from parkour_lab.artifacts import file_sha256, load_artifact, save_artifact
    from parkour_lab.config import ExperimentConfig
    from parkour_lab.methods import get_backend
    from parkour_lab.runtime.training import TrainingHost

    config.validate_training()
    host = TrainingHost(env, app)
    backend = get_backend(config.method.name)
    method = backend.create(host, config.method.options, config.task.seed)
    if getattr(env, "parkour_randomization", None) is not None:
        from parkour_lab.environments.dynamics import dynamics_report, start_report
        import numpy as np

        write_json(Path(output) / "manifest.json", env.parkour_randomization.manifest())
        np.savez_compressed(
            Path(output) / "initial_inputs.npz", **host.first_causal_sample
        )
        report["task_realization"] = dynamics_report(env)
        report["initial_state"] = start_report(env)
        report["motor_verification"] = host.bridge.motor_verification
        report["learner_sampling"] = {"backend": "torch", "seed": config.task.seed}
        report["evidence_sha256"] = {
            name: file_sha256(Path(output) / name)
            for name in ("manifest.json", "initial_inputs.npz")
        }
    if checkpoint is not None:
        previous = load_artifact(checkpoint, kind="training")
        if ExperimentConfig.from_dict(previous["config"]).method != config.method:
            raise ValueError("Continuing learning requires unchanged method settings")
        from parkour_lab.runtime.motor import NativeJointTargetBridge

        NativeJointTargetBridge(
            env,
            previous["motor_contract"],
            previous["motor_manifest"],
            preserve_native_raw=True,
        )
        method.load_state_dict(previous["state"])
        if method.updates != previous["updates"]:
            raise ValueError("Checkpoint update counters disagree")
        report["continuation"] = (
            "learning state restored; fresh simulator, history and RNG, not exact resume"
        )
        report["source_checkpoint"] = {
            "path": str(Path(checkpoint).resolve()),
            "sha256": file_sha256(checkpoint),
        }
    report["initial_updates"] = method.updates
    report["dependencies"] = backend.dependencies()
    # Preserve backend-owned model/optimizer state, not simulator or RNG state.
    # Payload hashes identify bytes; differing serializers may encode equal values.
    initial = Path(output) / "initial_learning_state.payload"
    payload = backend.dump(
        method.state_dict(), "training", config.method.options, method.updates
    )
    with initial.open("xb") as stream:
        stream.write(payload)
    report["initial_learning_state"] = {
        "path": initial.name,
        "updates": method.updates,
        "sha256": file_sha256(initial),
    }
    report.setdefault("evidence_sha256", {})[initial.name] = report[
        "initial_learning_state"
    ]["sha256"]
    started = time.monotonic()
    write_json(Path(output) / "report.json", report)
    with ExitStack() as stack, (Path(output) / "metrics.jsonl").open("x") as stream:
        if diagnose_training:
            stack.enter_context(method.diagnostics(output, report))
        for index in range(config.updates):
            metrics = dict(method.advance())
            metrics.update(
                task_metrics=host.metrics.drain(),
                environment_transitions=host.steps * env.num_envs,
                training_wall_seconds=time.monotonic() - started,
            )
            if config.task.terrain == "rough":
                metrics["terrain_level_counts"] = (
                    env.scene.terrain.terrain_levels.bincount(
                        minlength=config.task.num_rows
                    )
                    .cpu()
                    .tolist()
                )
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            stream.flush()
            report.update(
                updates=method.updates,
                environment_transitions=host.steps * env.num_envs,
                method_metrics=metrics,
            )
            if (
                method.updates % config.save_interval == 0
                or index + 1 == config.updates
            ):
                path = Path(output) / f"checkpoint_{method.updates:06d}.plab"
                save_artifact(
                    path,
                    kind="training",
                    config=config,
                    state=method.state_dict(),
                    motor_contract=host.motor_contract,
                    motor_manifest=host.manifest,
                    updates=method.updates,
                )
                report["checkpoint"] = str(path)
                write_json(Path(output) / "report.json", report)
            print(f"Update {method.updates}: {metrics}", flush=True)
    report["motor_delivery"] = host.bridge.progress()
    from parkour_lab.runtime.native import sensor_noise_report

    report["sensor_noise"] = sensor_noise_report(env)
    if config.task.terrain in ("flat", "rough"):
        report["command_sampling"] = getattr(
            env.command_manager.get_term("base_velocity"), "sampling_report", None
        )
        report["task_realization"] = dynamics_report(env)
    report["status"] = "TRAINING_COMPLETE_NOT_QUALIFIED"
    # Evaluation/export are explicit operations, never inferred from training loss.
