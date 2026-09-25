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


def _validated_policy(saved, report):
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
    return policy


def _validate_native_motor(saved, report, count):
    validate_motor_contract(saved["motor_contract"], saved["motor_manifest"])
    verification = report["motor_verification"]
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
        raise ValueError("ROA checkpoint motor differs from native delivery receipts")


def _load_refinement(path, hashes, saved, protocol, report):
    """One estimator-only stage over a completed v3 source; never recursive training."""
    from .operator_roa_adapt import DIFFICULTY, NATIVE_STEPS, SCHEDULE, VERSION

    status = "ROA_ESTIMATOR_REFINEMENT_COMPLETED_NOT_QUALIFIED"
    count, seed = protocol["num_envs"], protocol["seed"]
    motor = report["motor_delivery"]
    source_path = saved["source_checkpoint"]
    if (
        set(saved)
        != {
            "version",
            "readiness_only",
            "deployment_allowed",
            "policy_state",
            "adaptation_optimizer",
            "motor_contract",
            "motor_manifest",
            "source_checkpoint",
            "source",
            "completed_blocks",
        }
        or path.name != "adapted.pt"
        or protocol["version"] != VERSION
        or saved["readiness_only"] is not True
        or saved["deployment_allowed"] is not False
        or type(saved["adaptation_optimizer"]) is not dict
        or type(count) is not int
        or count not in (80, 160, 320)
        or type(seed) is not int
        or seed < 0
        or protocol["evaluation_reset_seed"] != seed + 1000
        or protocol["terrain_rows"] != 3
        or protocol["difficulty_range"] != list(DIFFICULTY)
        or protocol["planned_environment_transitions"] != NATIVE_STEPS * count
        or type(source_path) is not str
        or not Path(source_path).is_absolute()
        or protocol["source_checkpoint"] != source_path
        or saved["source"] != protocol["source"]
        or saved["completed_blocks"] != SCHEDULE["blocks"]
        or protocol["schedule"] != SCHEDULE
        or protocol["exit_allowed"] is not False
        or report.get("session_status", report["status"]) != status
        or report["status"] not in (status, "SESSION_COMPLETED_CLEANUP_PENDING")
        or report["checkpoint_sha256"] != hashes[str(path)]
        or report["ppo_updates_completed"] != 0
        or report["adaptation_optimizer_steps"] != SCHEDULE["estimator_optimizer_steps"]
        or report["completed_blocks"] != SCHEDULE["blocks"]
        or report["environment_transitions"] != NATIVE_STEPS * count
        or any(
            report[key] is not False for key in ("exit_allowed", "behavior_validated")
        )
        or report["fixed_modules_unchanged"] is not True
        or report["estimator_changed"] is not True
        or any(
            not isinstance(report[key], dict)
            for key in ("evaluations_before", "evaluations_after")
        )
        or motor["faulted"] is not False
        or motor["pending_delivery"] is not False
        or any(
            motor[key] != NATIVE_STEPS
            for key in (
                "encoded_steps",
                "verified_delivery_steps",
                "native_step_returns",
            )
        )
    ):
        raise ValueError(
            "Require a source-bound completed estimator-only ROA refinement"
        )
    original, source_contract, _, source = load_completed_checkpoint(
        source_path, allow_refinement=False
    )
    if (
        saved["source"] != source
        or seed == source["training_seed"]
        or protocol["source_identity"]["physical_reference"]
        != source["physical_reference"]
    ):
        raise ValueError("ROA refinement differs from its completed v3 source receipt")
    _validate_refinement_exposure(report["training_exposure"], count)
    for key, digest in (
        ("evaluations_before", source["policy_state_sha256"]),
        ("evaluations_after", report["policy_state_sha256"]),
    ):
        evaluation = report[key]
        if (
            evaluation["version"] != "operator_roa_history_evaluation_v1"
            or evaluation["seed"] != seed + 1000
            or evaluation["control_steps"] != 900
            or evaluation["num_envs"] != count
            or evaluation["environment_transitions"] != 900 * count
            or evaluation["complete_tape"] is not True
            or evaluation["policy_state_sha256_before"] != digest
            or evaluation["policy_state_sha256_after"] != digest
            or evaluation.get("diagnostic_input") is not None
        ):
            raise ValueError(
                "ROA refinement requires complete frozen causal evaluations"
            )
    _validate_native_motor(saved, report, count)
    if binding_sha256(saved["motor_contract"]["binding"]) != binding_sha256(
        source_contract["binding"]
    ):
        raise ValueError("ROA refinement changed the source motor binding")
    policy = _validated_policy(saved, report)
    before, after = original.state_dict(), policy.state_dict()
    if any(
        not torch.equal(value, after[key])
        for key, value in before.items()
        if not key.startswith("actor.estimator.")
    ):
        raise ValueError("ROA refinement changed a frozen non-estimator tensor")
    if all(
        torch.equal(value, after[key])
        for key, value in before.items()
        if key.startswith("actor.estimator.")
    ):
        raise ValueError("ROA refinement did not change the estimator")
    receipt = {
        "files": {**source["files"], **hashes},
        "checkpoint_sha256": hashes[str(path)],
        "policy_state_sha256": report["policy_state_sha256"],
        "learning_updates": source["learning_updates"],
        "additional_adaptation_optimizer_steps": SCHEDULE["estimator_optimizer_steps"],
        "stage": "estimator_refinement",
        "physical_reference": source["physical_reference"],
        "training_seed": seed,
        "training_evaluation_reset_seed": seed + 1000,
        "source_cleanup": report.get("cleanup"),
        "scope": "Estimator-only diagnostic refinement; simulator-development only, not terrain or hardware qualification",
    }
    verify_source_files(receipt)
    return policy, saved["motor_contract"], saved["motor_manifest"], receipt


def _validate_refinement_exposure(exposure, count):
    """Accounting only: no minimum motion, support, or success is inferred."""
    from .operator_roa_adapt import SCHEDULE

    steps = SCHEDULE["blocks"] * SCHEDULE["history_block_steps"]
    profiles = ("plane", "rough_flat", "hills", "step_hills", "tilted_ramps")
    columns, levels = exposure["column_ids"], exposure["level_ids"]
    if (
        exposure["control_steps"] != steps
        or len(columns) != count
        or len(levels) != count
        or any(type(value) is not int or not 0 <= value < 20 for value in columns)
        or any(type(value) is not int or not 0 <= value < 3 for value in levels)
    ):
        raise ValueError("Invalid ROA refinement exposure rows or collection budget")
    expected = {(profile, level): 0 for profile in profiles for level in range(3)}
    for column, level in zip(columns, levels, strict=True):
        expected[(profiles[column // 4], level)] += steps
    observed = set()
    names = (
        "translation_command_samples",
        "moving_samples",
        "moving_outside_flat_regions_samples",
        "moving_outside_flat_regions_nonzero_surface_samples",
        "invalid_ray_or_root_samples",
    )
    for group in exposure["groups"]:
        key = (group["profile"], group["level"])
        samples = group["samples"]
        if (
            key not in expected
            or key in observed
            or type(group["level"]) is not int
            or type(samples) is not int
            or samples != expected[key]
            or any(
                type(group[name]) is not int or not 0 <= group[name] <= samples
                for name in names
            )
            or not group[names[3]] <= group[names[2]] <= group[names[1]]
        ):
            raise ValueError("Invalid ROA refinement exposure sample accounting")
        observed.add(key)
    if observed != set(expected):
        raise ValueError("Incomplete ROA refinement exposure groups")


def load_completed_checkpoint(path, *, allow_refinement=True):
    """Completed final v3 or its single estimator-only stage; no partial selection."""
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
        if saved["version"] == "operator_roa_estimator_refinement_v1":
            if not allow_refinement:
                raise ValueError(
                    "ROA refinement requires a completed v3 source, not another refinement"
                )
            return _load_refinement(path, hashes, saved, protocol, report)
        updates, count = saved["completed_cycles"], protocol["num_envs"]
        status = "ROA_INCREMENTAL_EXPERIMENT_COMPLETED_NOT_QUALIFIED"
        expected_steps = updates * 24 + updates // 20 * 64 + 1800
        motor = report["motor_delivery"]
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
        _validate_native_motor(saved, report, count)
        policy = _validated_policy(saved, report)
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
    if any(output.is_relative_to(Path(name).parent) for name in receipt["files"]):
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
