"""Training orchestration: configuration, native lifecycle and current snapshots.

The host manages run lifecycle; ROA owns its learning schedule and optimizer state.
"""

from dataclasses import asdict
from pathlib import Path

from parkour_lab.provenance import write_json


def train(env, app, config, output, report, *, checkpoint=None):
    from parkour_lab.artifacts import file_sha256, load_artifact, save_artifact
    from parkour_lab.config import ExperimentConfig
    from parkour_lab.methods.roa.environment import ROAEnvironment
    from parkour_lab.methods.roa.model import build_policy
    from parkour_lab.methods.roa.training import ROATrainingMethod

    config.validate_training()
    host = ROAEnvironment(env, app, contact_conditioned=config.roa.contact_conditioned)
    observations, _ = host.reset(seed=config.task.seed)
    policy, _ = build_policy(
        observations,
        contact_conditioned=config.roa.contact_conditioned,
        initial_action_std=config.roa.initial_action_std,
    )
    options = asdict(config.roa)
    for key in ("contact_conditioned", "initial_action_std"):
        options.pop(key)
    method = ROATrainingMethod(
        policy,
        observations,
        num_envs=env.num_envs,
        device=env.device,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        max_grad_norm=1.0,
        **options,
    )
    if checkpoint is not None:
        previous = load_artifact(checkpoint, kind="training")
        if ExperimentConfig.from_dict(previous["config"]).roa != config.roa:
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
    write_json(Path(output) / "report.json", report)
    with (Path(output) / "metrics.jsonl").open("x") as stream:
        for index in range(config.updates):
            observations, metrics = method.advance(host, observations)
            import json

            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            stream.flush()
            report.update(
                updates=method.updates,
                environment_transitions=host.steps * env.num_envs,
                adaptation_optimizer_steps=method.adaptation_optimizer_steps,
            )
            if (
                method.updates % config.save_interval == 0
                or index + 1 == config.updates
            ):
                path = Path(output) / f"checkpoint_{method.updates:06d}.pt"
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
            print(f"Update {method.updates}: {metrics['ppo_losses']}", flush=True)
    report["motor_delivery"] = host.bridge.progress()
    report["status"] = "TRAINING_COMPLETE_NOT_QUALIFIED"
    # Evaluation/export are explicit operations, never inferred from training loss.
