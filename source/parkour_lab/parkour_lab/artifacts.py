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
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("metadata.json", json.dumps(metadata, allow_nan=False))
        archive.writestr("payload", payload)
    with Path(path).open("xb") as stream:
        stream.write(buffer.getvalue())


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
