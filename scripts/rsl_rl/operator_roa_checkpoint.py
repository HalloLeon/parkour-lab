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


def _policy_template(*, contact_conditioned=False):
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
        policy.actor = ROAActor(mlp(12), contact_conditioned=contact_conditioned)
        policy.critic = mlp(1)
        policy.critic[0] = StockTerrainInput(policy.critic[0])
    return policy


def verify_source_files(receipt):
    for name, digest in receipt["files"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"ROA source changed: {name}")


def _validate_completion(report, status, steps, count):
    """Completed work can retain pending cleanup, but never an error or lost delivery."""
    motor = report["motor_delivery"]
    counts = [
        motor[key]
        for key in ("encoded_steps", "verified_delivery_steps", "native_step_returns")
    ]
    if (
        report.get("session_status", report["status"]) != status
        or report["status"] not in (status, "SESSION_COMPLETED_CLEANUP_PENDING")
        or any(
            report[key] is not False for key in ("exit_allowed", "behavior_validated")
        )
        or motor["faulted"] is not False
        or motor["pending_delivery"] is not False
        or any(type(value) is not int or value != steps for value in counts)
        or type(report["environment_transitions"]) is not int
        or report["environment_transitions"] != steps * count
    ):
        raise ValueError("Incomplete ROA session or native delivery receipt")


def _validated_policy(saved, report, *, contact_conditioned=False):
    policy = _policy_template(contact_conditioned=contact_conditioned)
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
        or report["checkpoint_sha256"] != hashes[str(path)]
        or report["ppo_updates_completed"] != 0
        or report["adaptation_optimizer_steps"] != SCHEDULE["estimator_optimizer_steps"]
        or report["completed_blocks"] != SCHEDULE["blocks"]
        or report["fixed_modules_unchanged"] is not True
        or report["estimator_changed"] is not True
        or any(
            not isinstance(report[key], dict)
            for key in ("evaluations_before", "evaluations_after")
        )
    ):
        raise ValueError(
            "Require a source-bound completed estimator-only ROA refinement"
        )
    _validate_completion(report, status, NATIVE_STEPS, count)
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
    _validate_training_exposure(
        report["training_exposure"],
        count,
        steps=SCHEDULE["blocks"] * SCHEDULE["history_block_steps"],
    )
    for key, digest in (
        ("evaluations_before", source["policy_state_sha256"]),
        ("evaluations_after", report["policy_state_sha256"]),
    ):
        _validate_frozen_evaluation(report[key], count, seed + 1000, digest)
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


def _validate_frozen_evaluation(evaluation, count, seed, digest):
    if (
        evaluation["version"] != "operator_roa_history_evaluation_v1"
        or evaluation["seed"] != seed
        or evaluation["control_steps"] != 900
        or evaluation["num_envs"] != count
        or evaluation["environment_transitions"] != 900 * count
        or evaluation["complete_tape"] is not True
        or evaluation["policy_state_sha256_before"] != digest
        or evaluation["policy_state_sha256_after"] != digest
        or evaluation.get("diagnostic_input") is not None
    ):
        raise ValueError("ROA stage requires complete frozen causal evaluations")


def _validate_training_exposure(exposure, count, *, steps):
    """Accounting only: no minimum motion, support, or success is inferred."""
    profiles = ("plane", "rough_flat", "hills", "step_hills", "tilted_ramps")
    columns, levels = exposure["column_ids"], exposure["level_ids"]
    if (
        exposure["control_steps"] != steps
        or len(columns) != count
        or len(levels) != count
        or any(type(value) is not int or not 0 <= value < 20 for value in columns)
        or any(type(value) is not int or not 0 <= value < 3 for value in levels)
    ):
        raise ValueError("Invalid ROA training exposure rows or collection budget")
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
            raise ValueError("Invalid ROA training exposure sample accounting")
        observed.add(key)
    if observed != set(expected):
        raise ValueError("Incomplete ROA training exposure groups")


def _environment_source(saved, protocol, report, layout, stage):
    """Validate a bounded joint stage and its strictly earlier source stage."""
    source_path = protocol["environment_checkpoint"]
    seed, count = protocol["seed"], protocol["num_envs"]
    if (
        type(source_path) is not str
        or not Path(source_path).is_absolute()
        or type(seed) is not int
        or seed < 0
        or saved["completed_cycles"] != stage.updates
        or any(saved["regularization_coefficients"])
        or protocol["terrain_rows"] != 3
        or protocol["difficulty_range"] != [0.15, 0.55]
        or protocol["evaluation"]["seed"] != seed + 1000
        or protocol["planned_environment_transitions"]
        != (stage.updates * 24 + stage.updates // 20 * 64 + 1800) * count
        or protocol["adaptation_optimizer"]["learning_rate"] != 1e-4
        or report["ppo_options"]["learning_rate"] != 1e-4
        or report["ppo_options"]["schedule"] != "fixed"
    ):
        raise ValueError(
            f"Require the bounded {stage.updates}-update free-environment ROA stage"
        )
    original, contract, _, source = load_completed_checkpoint(
        source_path,
        allow_environment=bool(stage.geometry_version),
        allow_step_fields=stage.support_resets,
        allow_step_support=stage.source_contact_conditioned,
        expected_stage=stage.source_stage,
    )
    if (
        source.get("stage") != stage.source_stage
        or source["learning_updates"] != stage.source_updates
        or saved["learning_source"] != source
        or source["physical_reference"]
        != protocol["source_identity"]["physical_reference"]
        or source["training_seed"] == seed
        or binding_sha256(contract["binding"])
        != binding_sha256(saved["motor_contract"]["binding"])
    ):
        source_label = stage.source_stage.replace("_", "-")
        raise ValueError(f"Environment stage differs from its {source_label} source")
    _validate_training_exposure(
        report["training_exposure"],
        count,
        steps=stage.updates * 24 + stage.updates // 20 * 64,
    )
    if stage.geometry_version:
        from . import operator_step_field

        if (
            protocol["environment_layout"] != layout
            or protocol["step_field_geometry"]
            != operator_step_field.envelope(stage.geometry_version)
            or report["native_step_field_geometry"]["version"] != stage.geometry_version
            or report["training_exposure"]["geometry_overrides"]
            != {"step_hills": stage.geometry_version}
        ):
            raise ValueError("Step-field geometry or measured exposure recipe changed")
        operator_step_field.validate_geometry_report(
            report["native_step_field_geometry"], seed=seed
        )
    if stage.support_resets:
        from . import operator_step_support

        if protocol["support_reset_recipe"] != operator_step_support.recipe():
            raise ValueError("Support-reset recipe changed")
        operator_step_support.validate_receipt(
            report["native_support_patches"], report["native_step_field_geometry"]
        )
        operator_step_support.validate_training_report(
            report["training_support_resets"],
            num_envs=count,
            steps=stage.updates * 24 + stage.updates // 20 * 64,
        )
    if stage.step_clearance:
        from . import operator_step_clearance

        if (
            protocol["step_clearance_recipe"] != operator_step_clearance.recipe()
            or report["training_step_clearance"]["binding"]["columns"]
            != report["training_exposure"]["column_ids"]
        ):
            raise ValueError("Step-clearance reward recipe changed")
        operator_step_clearance.validate_training_report(
            report["training_step_clearance"],
            num_envs=count,
            steps=stage.updates * 24 + stage.updates // 20 * 64,
        )
    initial_digest = source["policy_state_sha256"]
    if stage.contact_conditioned:
        from . import operator_roa_contacts

        if protocol["teacher_contacts"] != operator_roa_contacts.recipe():
            raise ValueError("Teacher contact recipe changed")
        if not stage.source_contact_conditioned:
            original.actor.enable_contact_conditioning()
            initial_digest = state_sha256(original)
            if report["contact_initialization"] != {
                "source_policy_state_sha256": source["policy_state_sha256"],
                "initialized_policy_state_sha256": initial_digest,
                "teacher_latent_exact": True,
                "causal_action_exact": True,
                "privileged_action_exact": True,
                "new_projection_zero": True,
            }:
                raise ValueError(
                    "Teacher contact initialization differs from its source"
                )
        operator_roa_contacts.validate_report(
            report["teacher_contact_observations"],
            num_envs=count,
            steps=stage.updates * 24 + stage.updates // 20 * 64 + 1800,
        )
    if stage.source_contact_conditioned:
        from .operator_roa_pilot import orientation_objective

        if (
            json.dumps(report["orientation_objective"], sort_keys=True)
            != json.dumps(
                orientation_objective(protocol["orientation_weight"]), sort_keys=True
            )
            or "contact_initialization" in report
        ):
            raise ValueError("Contact follow-up objective or initialization changed")
    for key, digest in (
        ("evaluation_before", initial_digest),
        ("evaluation_after", report["policy_state_sha256"]),
    ):
        _validate_frozen_evaluation(report[key], count, seed + 1000, digest)
    return source


def load_completed_checkpoint(
    path,
    *,
    allow_refinement=True,
    allow_environment=True,
    allow_step_fields=True,
    allow_step_support=True,
    expected_stage=None,
):
    """Completed bounded stages with finite ancestry; no partial/resume selection."""
    from .operator_roa_pilot import ENVIRONMENT_STAGES, learning_coefficients

    path = Path(path).resolve(strict=True)
    paths = (path, path.parent / "training_protocol.json", path.parent / "report.json")
    encoded = {str(item): item.read_bytes() for item in paths}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in encoded.items()}
    try:
        saved = torch.load(
            io.BytesIO(encoded[str(path)]), map_location="cpu", weights_only=True
        )
        protocol, report = (json.loads(encoded[str(item)]) for item in paths[1:])
        layout = next(
            (
                key
                for key, stage in ENVIRONMENT_STAGES.items()
                if saved["version"] == stage.version
            ),
            None,
        )
        environment = layout is not None
        stage = ENVIRONMENT_STAGES[layout] if environment else None
        declared_stage = (
            stage.result_stage
            if stage
            else (
                "estimator_refinement"
                if saved["version"] == "operator_roa_estimator_refinement_v1"
                else None
            )
        )
        if expected_stage is not None and declared_stage != expected_stage:
            raise ValueError(
                f"Require the declared {expected_stage.replace('_', '-')} source"
            )
        if environment and (not allow_refinement or not allow_environment):
            raise ValueError(
                "Require original v3/refinement ancestry, not an environment stage"
            )
        if stage and stage.geometry_version and not allow_step_fields:
            raise ValueError("Require earlier ancestry, not another step-field stage")
        if stage and stage.support_resets and not allow_step_support:
            raise ValueError(
                "Require earlier ancestry, not another support-reset stage"
            )
        if saved["version"] == "operator_roa_estimator_refinement_v1":
            if not allow_refinement:
                raise ValueError(
                    "ROA refinement requires a completed v3 source, not another refinement"
                )
            return _load_refinement(path, hashes, saved, protocol, report)
        updates, count = saved["completed_cycles"], protocol["num_envs"]
        status = (
            stage.status
            if environment
            else "ROA_INCREMENTAL_EXPERIMENT_COMPLETED_NOT_QUALIFIED"
        )
        expected_steps = updates * 24 + updates // 20 * 64 + 1800
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
            or saved["version"]
            not in (
                "operator_roa_learning_pilot_v3",
                *(item.version for item in ENVIRONMENT_STAGES.values()),
            )
            or protocol["version"] != saved["version"]
            or type(updates) is not int
            or updates not in (100, 500, 1000)
            or type(count) is not int
            or count not in (80, 160, 320)
            or protocol["cycles"] != updates
            or report["ppo_updates_completed"] != updates
            or report["checkpoint_sha256"] != hashes[str(path)]
            or {"path": path.name, "sha256": hashes[str(path)], "ppo_updates": updates}
            not in report["checkpoints"]
            or saved["readiness_only"] is not True
            or saved["deployment_allowed"] is not False
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
                    "version": (
                        "operator_roa_contact_pilot_v1"
                        if stage and stage.contact_conditioned
                        else "operator_roa_pilot_v1"
                    ),
                    "frame_dim": 45,
                    "history_frames": 25,
                    "dynamics_dim": 7,
                    "latent_dim": 8,
                    "velocity_dim": 3,
                }.items()
            )
            or (
                stage
                and stage.contact_conditioned
                and protocol["model"].get("contact_dim") != 12
            )
            or list(saved["regularization_coefficients"])
            != protocol["regularization_coefficients"]
            or tuple(saved["regularization_coefficients"])
            not in (
                learning_coefficients("off", updates),
                learning_coefficients("ramp", updates),
            )
            or report["adaptation_optimizer_steps"] != updates // 20 * 16
            or set(physical) != {"checkpoint", "agent.yaml", "env.yaml"}
            or any(
                type(digest) is not str
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                for digest in physical.values()
            )
        ):
            raise ValueError(
                "Require a source-bound completed final environment ROA experiment"
                if environment
                else "Require a source-bound completed final v3 ROA experiment"
            )
        _validate_completion(report, status, expected_steps, count)
        _validate_native_motor(saved, report, count)
        policy = _validated_policy(
            saved, report, contact_conditioned=bool(stage and stage.contact_conditioned)
        )
        source = (
            _environment_source(saved, protocol, report, layout, stage)
            if environment
            else None
        )
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
    if source is not None:
        inherited_adaptation = (
            source["learning_updates"] // 20 * 16
            + source["additional_adaptation_optimizer_steps"]
            if layout == "hills"
            else source["total_adaptation_optimizer_steps"]
        )
        receipt.update(
            files={**source["files"], **hashes},
            learning_updates=source["learning_updates"] + updates,
            stage_learning_updates=updates,
            stage_adaptation_optimizer_steps=report["adaptation_optimizer_steps"],
            inherited_adaptation_optimizer_steps=inherited_adaptation,
            total_adaptation_optimizer_steps=inherited_adaptation
            + report["adaptation_optimizer_steps"],
            counting_scope="Selected v3 experiment and descendant stages only; excludes the v2 initialization pilot and stock pretraining",
            stage=stage.result_stage,
            scope="Bounded free-environment joint ROA stage; simulator-development only, not terrain or hardware qualification",
        )
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
