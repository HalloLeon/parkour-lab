"""Read completed ROA experiments and export causal simulator-development actors.

Receipts bind local files, not an untrusted producer. Optimizers and privileged
weights remain in the training archive; they are never part of the actor bundle.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import torch
from torch import nn

from parkour_lab.learning.operator_roa import ROAActor, state_sha256
from parkour_lab.learning.motor_contract import validate_motor_contract, binding_sha256


def _policy_template():
    """Fixed checkpoint schema without importing a simulator or PPO implementation."""
    from parkour_lab.learning.distillation.teacher.model import StockTerrainInput

    def mlp(output):
        return nn.Sequential(
            nn.Linear(48, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, output),
        )

    with torch.random.fork_rng(devices=[]):
        policy = nn.Module()
        policy.register_parameter("std", nn.Parameter(torch.ones(12)))
        policy.actor = ROAActor(mlp(12))
        policy.critic = mlp(1)
        policy.critic[0] = StockTerrainInput(policy.critic[0])
    return policy


def verify_source_files(receipt):
    for name, digest in receipt["files"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"ROA source changed: {name}")


def load_completed_checkpoint(path):
    """Only completed final v3 checkpoints; no resume or intermediate selection."""
    from .operator_roa_pilot import learning_coefficients

    path = Path(path).resolve(strict=True)
    paths = (path, path.parent / "training_protocol.json", path.parent / "report.json")
    encoded = {str(item): item.read_bytes() for item in paths}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in encoded.items()}
    try:
        saved = torch.load(
            io.BytesIO(encoded[str(path)]), map_location="cpu", weights_only=True
        )
        protocol, report = (json.loads(encoded[str(item)]) for item in paths[1:])
        updates, count = saved["completed_cycles"], protocol["num_envs"]
        status = "ROA_INCREMENTAL_EXPERIMENT_COMPLETED_NOT_QUALIFIED"
        expected_steps = updates * 24 + updates // 20 * 64 + 1800
        motor = report["motor_delivery"]
        verification = report["motor_verification"]
        physical = protocol["source_identity"]["physical_reference"]
        if (
            set(saved)
            != {
                "version",
                "readiness_only",
                "deployment_allowed",
                "policy_state",
                "ppo_optimizer",
                "adaptation_optimizer",
                "motor_contract",
                "motor_manifest",
                "completed_cycles",
                "regularization_coefficients",
                "history_interval",
                "learning_source",
            }
            or saved["version"] != "operator_roa_learning_pilot_v3"
            or protocol["version"] != saved["version"]
            or type(updates) is not int
            or updates not in (100, 500, 1000)
            or type(count) is not int
            or count not in (80, 160, 320)
            or protocol["cycles"] != updates
            or report["ppo_updates_completed"] != updates
            or report.get("session_status", report["status"]) != status
            or report["status"] not in (status, "SESSION_COMPLETED_CLEANUP_PENDING")
            or report["checkpoint_sha256"] != hashes[str(path)]
            or {"path": path.name, "sha256": hashes[str(path)], "ppo_updates": updates}
            not in report["checkpoints"]
            or saved["readiness_only"] is not True
            or saved["deployment_allowed"] is not False
            or any(
                report[key] is not False
                for key in ("exit_allowed", "behavior_validated")
            )
            or protocol["exit_allowed"] is not False
            or saved["history_interval"] != 20
            or protocol["history_interval"] != 20
            or saved["learning_source"] != protocol["learning_source"]
            or protocol["rollout_steps_per_update"] != 24
            or protocol["history_steps_per_cycle"] != 64
            or protocol["normalization"] != "none; retain stock observation scales"
            or protocol["actor_scan"] is not False
            or any(
                protocol["model"][key] != value
                for key, value in {
                    "version": "operator_roa_pilot_v1",
                    "frame_dim": 45,
                    "history_frames": 25,
                    "dynamics_dim": 7,
                    "latent_dim": 8,
                    "velocity_dim": 3,
                }.items()
            )
            or list(saved["regularization_coefficients"])
            != protocol["regularization_coefficients"]
            or tuple(saved["regularization_coefficients"])
            not in (
                learning_coefficients("off", updates),
                learning_coefficients("ramp", updates),
            )
            or report["adaptation_optimizer_steps"] != updates // 20 * 16
            or motor["faulted"] is not False
            or motor["pending_delivery"] is not False
            or any(
                motor[key] != expected_steps
                for key in (
                    "encoded_steps",
                    "verified_delivery_steps",
                    "native_step_returns",
                )
            )
            or report["environment_transitions"] != expected_steps * count
            or set(physical) != {"checkpoint", "agent.yaml", "env.yaml"}
            or any(
                type(digest) is not str
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                for digest in physical.values()
            )
        ):
            raise ValueError("Require a source-bound completed final v3 ROA experiment")
        validate_motor_contract(saved["motor_contract"], saved["motor_manifest"])
        motor_digest = saved["motor_manifest"]["actuator_profile"].removeprefix(
            "native_motor_sha256:"
        )
        if (
            verification["source_motor_binding_sha256"] != motor_digest
            or verification["runtime_motor_binding_sha256"] != motor_digest
            or verification["portable_motor_sha256"]
            != binding_sha256(
                {
                    "version": saved["motor_contract"]["version"],
                    "binding": saved["motor_contract"]["binding"],
                }
            )
            or any(
                value != count
                for value in (
                    verification["source_num_envs"],
                    verification["runtime_num_envs"],
                    saved["motor_contract"]["source_num_envs"],
                )
            )
        ):
            raise ValueError(
                "ROA checkpoint motor differs from native delivery receipts"
            )
        policy = _policy_template()
        expected, state = policy.state_dict(), saved["policy_state"]
        if (
            set(state) != set(expected)
            or any(
                not isinstance(value, torch.Tensor)
                or value.shape != expected[key].shape
                or value.dtype != torch.float32
                or not torch.isfinite(value).all()
                for key, value in state.items()
            )
            or (state["std"] <= 0).any()
        ):
            raise ValueError("Invalid ROA policy tensor schema")
        policy.load_state_dict(state, strict=True)
        policy.eval().requires_grad_(False)
        if state_sha256(policy) != report["policy_state_sha256"]:
            raise ValueError("ROA policy weights differ from the completed report")
    except (KeyError, TypeError, AttributeError, RuntimeError, OverflowError) as error:
        raise ValueError("Malformed or incomplete ROA experiment") from error
    receipt = {
        "files": hashes,
        "checkpoint_sha256": hashes[str(path)],
        "policy_state_sha256": report["policy_state_sha256"],
        "learning_updates": updates,
        "physical_reference": physical,
        "training_seed": protocol["seed"],
        "training_evaluation_reset_seed": protocol["evaluation"]["seed"],
        "source_cleanup": report.get("cleanup"),
        "scope": "Completed final v3 experiment; simulator-development only, not hardware qualification",
    }
    verify_source_files(receipt)
    return policy, saved["motor_contract"], saved["motor_manifest"], receipt


def export_roa_actor(checkpoint, output):
    from parkour_lab.learning.operator_roa_runtime import (
        ROAHistoryController,
        roa_tensor_sha256,
        actor_bundle,
        _load_actor_bytes,
    )

    policy, contract, manifest, receipt = load_completed_checkpoint(checkpoint)
    output = Path(output).resolve()
    if output.is_relative_to(Path(checkpoint).resolve().parent):
        raise ValueError("Export outside the immutable training run")
    controller = ROAHistoryController(
        policy.actor.motor,
        policy.actor.estimator,
        joint_names=manifest["joint_names"],
        default_position_rad=torch.tensor(
            manifest["configuration"]["default_position_rad"]
        ),
        artifact_sha256=roa_tensor_sha256(policy.actor.motor, policy.actor.estimator),
        actuator_profile=manifest["actuator_profile"],
    )
    bundle = actor_bundle(
        controller,
        source_checkpoint_sha256=receipt["checkpoint_sha256"],
        learning_updates=receipt["learning_updates"],
        physical_reference=receipt["physical_reference"],
        motor_contract=contract,
    )
    buffer = io.BytesIO()
    torch.save(bundle, buffer)
    encoded = buffer.getvalue()
    _, metadata, digest = _load_actor_bytes(encoded)
    verify_source_files(receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(encoded)
    return {
        "status": "ROA_CAUSAL_ACTOR_EXPORTED_NOT_QUALIFIED",
        "path": str(output),
        "sha256": digest,
        "source": receipt,
        "metadata": metadata,
        "exit_allowed": False,
    }
