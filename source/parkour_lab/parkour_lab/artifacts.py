"""Current method-neutral artifacts: JSON metadata and an opaque numerical payload.

The selected installed backend owns payload encoding and validation. No Python
class paths or legacy readers are accepted. Learning continuation starts a fresh
simulator/RNG; export does not imply behavioral qualification.
"""

import copy
from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

from parkour_lab.config import ExperimentConfig
from parkour_lab.control.motor_contract import (
    BOUNDED_VERSION,
    nominal_binding_sha256,
    validate_motor_contract,
)
from parkour_lab.methods import get_backend
from parkour_lab.provenance import file_sha256

FORMAT = "parkour_lab_method_v3"


def _validate(metadata, kind):
    if (
        not isinstance(metadata, dict)
        or set(metadata)
        != {
            "format",
            "kind",
            "config",
            "motor_contract",
            "motor_manifest",
            "updates",
            "payload_sha256",
        }
        or metadata["format"] != FORMAT
        or kind not in ("training", "actor")
        or metadata["kind"] != kind
    ):
        raise ValueError("Unsupported artifact; only the current format is supported")
    config = ExperimentConfig.from_dict(metadata["config"])
    if type(metadata["updates"]) is not int or metadata["updates"] < 1:
        raise ValueError("Invalid artifact update count")
    binding = validate_motor_contract(
        metadata["motor_contract"], metadata["motor_manifest"]
    )
    if (metadata["motor_contract"]["version"] == BOUNDED_VERSION) != (
        config.action_mode == "joint_limits_v1"
    ):
        raise ValueError("Artifact action mode disagrees with motor contract")
    if (binding["physics_dt_s"], binding["decimation"]) != (
        1 / config.task.physics_hz,
        config.task.physics_hz // 50,
    ):
        raise ValueError("Artifact task physics rate disagrees with motor binding")
    return config, get_backend(config.method.name)


def save_artifact(
    path, *, kind, config, state, motor_contract, motor_manifest, updates
):
    """Validate/serialize before creating a destination; never overwrite."""
    backend = get_backend(config.method.name)
    payload = backend.dump(state, kind, config.method.options, updates)
    metadata = dict(
        format=FORMAT,
        kind=kind,
        config=config.to_dict(),
        motor_contract=motor_contract,
        motor_manifest=motor_manifest,
        updates=updates,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
    )
    _validate(metadata, kind)
    if kind == "actor":
        _controller({**metadata, "state": state})
    _write_artifact(path, metadata, payload)


def _write_artifact(path, metadata, payload):
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata, allow_nan=False))
        archive.writestr("payload", payload)
    with Path(path).open("xb") as stream:
        stream.write(buffer.getvalue())


def bound_actions(checkpoint, destination, *, report):
    """Copy a learner with explicit bounded semantics and byte-identical payload.

    The report provides actual native joint limits; the new runtime must match
    their 90% soft interval. No network, optimizer, reward or counter is changed.
    """
    import torch
    from parkour_lab.control.action import BOUNDED_MODE, BOUNDED_RAW_ACTION_MEANING

    data = load_artifact(checkpoint, kind="training")
    config = ExperimentConfig.from_dict(data["config"])
    if config.action_mode != "unbounded":
        raise ValueError("bound-actions requires an original unbounded checkpoint")
    realization = json.loads(Path(report).read_text())["task_realization"]
    binding = data["motor_contract"]["binding"]
    if realization["joint_names"] != binding["joint_names"]:
        raise ValueError("Native limit joint order differs from checkpoint")
    limits = torch.tensor(
        realization["physical_readback"]["joint_limits"], dtype=torch.float32
    )
    if (
        limits.ndim != 3
        or limits.shape[1:] != (12, 2)
        or len(limits) < 1
        or not torch.isfinite(limits).all()
        or not torch.equal(limits, limits[0].expand_as(limits))
        or not (limits[:, :, 0] < limits[:, :, 1]).all()
    ):
        raise ValueError("Require shared finite native mechanical joint limits")
    # Same float32 expression as Isaac Lab's Articulation soft limits, factor 0.9.
    center = (limits[0, :, 0] + limits[0, :, 1]) / 2
    half_range = 0.5 * (limits[0, :, 1] - limits[0, :, 0]) * 0.9
    targets = torch.stack((center - half_range, center + half_range), dim=-1).tolist()
    metadata = {
        key: copy.deepcopy(value) for key, value in data.items() if key != "state"
    }
    metadata["config"] = replace(config, action_mode=BOUNDED_MODE).to_dict()
    metadata["motor_contract"].update(
        version=BOUNDED_VERSION, target_limits_rad=targets
    )
    metadata["motor_manifest"]["configuration"]["action_clip"] = {
        "version": BOUNDED_MODE,
        "target_limits_rad": targets,
    }
    metadata["motor_manifest"]["raw_action_meaning"] = BOUNDED_RAW_ACTION_MEANING
    if "preprocessing_version" in metadata["motor_manifest"]:
        metadata["motor_manifest"]["preprocessing_version"] += "_bounded_targets_v1"
    _validate(metadata, "training")
    with ZipFile(checkpoint) as archive:
        payload = archive.read("payload")
    _write_artifact(destination, metadata, payload)
    return {
        "artifact": str(destination),
        "sha256": file_sha256(destination),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "source_limits_report_sha256": file_sha256(report),
        "payload_sha256": metadata["payload_sha256"],
        "learning_payload_unchanged": True,
        "updates": data["updates"],
        "action_mode": BOUNDED_MODE,
        "target_limits_rad": targets,
        "soft_joint_pos_limit_factor": 0.9,
        "requires_native_validation": True,
        "qualified": False,
    }


def load_artifact(path, *, kind):
    with ZipFile(path) as archive:
        if sorted(archive.namelist()) != ["metadata.json", "payload"]:
            raise ValueError("Invalid artifact members")
        metadata = json.loads(archive.read("metadata.json"))
        config, backend = _validate(metadata, kind)
        payload = archive.read("payload")
    if hashlib.sha256(payload).hexdigest() != metadata["payload_sha256"]:
        raise ValueError("Artifact payload checksum mismatch")
    state = backend.load(payload, kind, config.method.options, metadata["updates"])
    return {**metadata, "state": state}


def export_actor(checkpoint, destination, *, physics_hz=None):
    """Export frozen inference; an explicit timing change needs new native validation."""
    data = load_artifact(checkpoint, kind="training")
    config = ExperimentConfig.from_dict(data["config"])
    source_hz = config.task.physics_hz
    contract = copy.deepcopy(data["motor_contract"])
    manifest = copy.deepcopy(data["motor_manifest"])
    if physics_hz is not None:
        config = replace(config, task=replace(config.task, physics_hz=physics_hz))
        contract["binding"].update(
            physics_dt_s=1 / physics_hz, decimation=physics_hz // 50
        )
        manifest["actuator_profile"] = "native_motor_sha256:" + nominal_binding_sha256(
            contract["binding"]
        )
    state = get_backend(config.method.name).export(data["state"], config.method.options)
    # Generate current inference metadata rather than copying training-only or
    # stale contact-timing prose into a newly exported actor.
    manifest = _controller(
        {
            **data,
            "config": config.to_dict(),
            "state": state,
            "motor_contract": contract,
            "motor_manifest": manifest,
        }
    ).spec.manifest()
    save_artifact(
        destination,
        kind="actor",
        config=config,
        state=state,
        motor_contract=contract,
        motor_manifest=manifest,
        updates=data["updates"],
    )
    return {
        "artifact": str(destination),
        "sha256": file_sha256(destination),
        "source_checkpoint_sha256": file_sha256(checkpoint),
        "source_physics_hz": source_hz,
        "physics_hz": config.task.physics_hz,
        "source_actuator_profile": data["motor_manifest"]["actuator_profile"],
        "actuator_profile": manifest["actuator_profile"],
        "timing_changed": source_hz != config.task.physics_hz,
        "timing_change_requires_validation": source_hz != config.task.physics_hz,
        "qualified": False,
    }


def _controller(data, device="cpu"):
    config = ExperimentConfig.from_dict(data["config"])
    controller = get_backend(config.method.name).controller(
        data["state"], config.method.options, data["motor_manifest"], device
    )
    validate_motor_contract(data["motor_contract"], controller.spec.manifest())
    return controller


@dataclass(frozen=True)
class LoadedActor:
    controller: object
    motor_contract: dict
    config: ExperimentConfig
    updates: int


def load_actor(path, *, device="cpu"):
    data = load_artifact(path, kind="actor")
    return LoadedActor(
        _controller(data, device),
        data["motor_contract"],
        ExperimentConfig.from_dict(data["config"]),
        data["updates"],
    )
