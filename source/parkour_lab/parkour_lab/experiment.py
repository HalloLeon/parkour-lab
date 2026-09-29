"""Training orchestration: configuration, native lifecycle and current snapshots.

The host manages run lifecycle; each method owns learning and persistence.
"""

import json
from pathlib import Path
import time

from parkour_lab.provenance import write_json


def train(env, app, config, output, report, *, checkpoint=None):
    from parkour_lab.artifacts import file_sha256, load_artifact, save_artifact
    from parkour_lab.config import ExperimentConfig
    from parkour_lab.methods import get_backend
    from parkour_lab.runtime.training import TrainingHost

    config.validate_training()
    host = TrainingHost(env, app)
    backend = get_backend(config.method.name)
    method = backend.create(host, config.method.options, config.task.seed)
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
    started = time.monotonic()
    write_json(Path(output) / "report.json", report)
    with (Path(output) / "metrics.jsonl").open("x") as stream:
        for index in range(config.updates):
            metrics = dict(method.advance())
            metrics.update(
                environment_transitions=host.steps * env.num_envs,
                training_wall_seconds=time.monotonic() - started,
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
    report["status"] = "TRAINING_COMPLETE_NOT_QUALIFIED"
    # Evaluation/export are explicit operations, never inferred from training loss.
