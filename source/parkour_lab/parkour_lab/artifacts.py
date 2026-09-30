"""Current method-neutral artifacts: JSON metadata and an opaque numerical payload.

The selected installed backend owns payload encoding and validation. No Python
class paths or legacy readers are accepted. Learning continuation starts a fresh
simulator/RNG; export does not imply behavioral qualification.
"""

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

from parkour_lab.config import ExperimentConfig
from parkour_lab.control.motor_contract import validate_motor_contract
from parkour_lab.methods import get_backend

FORMAT = "parkour_lab_method_v2"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    validate_motor_contract(metadata["motor_contract"], metadata["motor_manifest"])
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


def export_actor(checkpoint, destination):
    data = load_artifact(checkpoint, kind="training")
    config = ExperimentConfig.from_dict(data["config"])
    state = get_backend(config.method.name).export(data["state"], config.method.options)
    save_artifact(
        destination,
        kind="actor",
        config=config,
        state=state,
        motor_contract=data["motor_contract"],
        motor_manifest=data["motor_manifest"],
        updates=data["updates"],
    )
    return {
        "artifact": str(destination),
        "sha256": file_sha256(destination),
        "source_checkpoint_sha256": file_sha256(checkpoint),
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
