"""Current ROA learning snapshots and causal inference artifacts.

No legacy readers, executable class names, reference checkpoints or run-directory
ancestry are accepted. Loading a snapshot restores learning state, not the
simulator or an interrupted rollout. Export is not behavioral qualification.
"""

from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path

import torch

from parkour_lab.config import ExperimentConfig
from parkour_lab.control.motor_contract import validate_motor_contract
from parkour_lab.methods.roa.runtime import (
    ROAHistoryController,
    _fixed_modules,
    roa_tensor_sha256,
)

FORMAT = "parkour_lab_roa_45d_v1"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_artifact(
    path, *, kind, config, state, motor_contract, motor_manifest, updates
):
    """Create once; never overwrite an existing checkpoint or exported actor."""
    data = dict(
        format=FORMAT,
        kind=kind,
        config=config.to_dict(),
        state=state,
        motor_contract=motor_contract,
        motor_manifest=motor_manifest,
        updates=updates,
    )
    _validate(data, kind)
    buffer = io.BytesIO()
    torch.save(data, buffer)
    # Serialization happens before opening the destination, so validation or
    # serialization failures cannot leave a misleading empty checkpoint.
    with Path(path).open("xb") as stream:
        stream.write(buffer.getvalue())


def _validate(data, kind):
    if (
        not isinstance(data, dict)
        or set(data)
        != {
            "format",
            "kind",
            "config",
            "state",
            "motor_contract",
            "motor_manifest",
            "updates",
        }
        or data["format"] != FORMAT
        or kind not in ("training", "actor")
        or data["kind"] != kind
    ):
        raise ValueError("Unsupported artifact; only the current format is supported")
    ExperimentConfig.from_dict(data["config"])
    if (
        type(data["updates"]) is not int
        or data["updates"] < 1
        or not isinstance(data["state"], dict)
    ):
        raise ValueError("Invalid artifact learning state")
    validate_motor_contract(data["motor_contract"], data["motor_manifest"])
    state = data["state"]
    if kind == "training":
        if (
            set(state)
            != {
                "policy_state",
                "ppo_optimizer",
                "adaptation_optimizer",
                "updates",
                "adaptation_optimizer_steps",
            }
            or type(state["updates"]) is not int
            or state["updates"] != data["updates"]
            or type(state["adaptation_optimizer_steps"]) is not int
            or state["adaptation_optimizer_steps"] < 0
        ):
            raise ValueError("Invalid learning snapshot schema or counters")
        weights = state["policy_state"]
        if (
            not isinstance(weights, dict)
            or not weights
            or any(
                not isinstance(v, torch.Tensor)
                or v.dtype != torch.float32
                or not torch.isfinite(v).all()
                for v in weights.values()
            )
        ):
            raise ValueError("Learning weights must be finite float32 tensors")
        std = weights.get("std")
        if std is None or std.shape != (12,) or (std <= 0).any():
            raise ValueError("Learning snapshot requires positive action std")
        for name in ("ppo_optimizer", "adaptation_optimizer"):
            _validate_optimizer(state[name])
    else:
        _controller(data)


def _validate_optimizer(optimizer):
    """Reject corrupt optimizer containers/moments before native initialization."""
    if (
        not isinstance(optimizer, dict)
        or set(optimizer) != {"state", "param_groups"}
        or not isinstance(optimizer["state"], dict)
        or not isinstance(optimizer["param_groups"], list)
        or not optimizer["param_groups"]
    ):
        raise ValueError("Invalid optimizer state")
    parameters = []
    for group in optimizer["param_groups"]:
        if (
            not isinstance(group, dict)
            or not isinstance(group.get("params"), list)
            or not group["params"]
            or any(type(p) is not int or p < 0 for p in group["params"])
        ):
            raise ValueError("Invalid optimizer parameter groups")
        parameters.extend(group["params"])
    if (
        len(set(parameters)) != len(parameters)
        or set(optimizer["state"]) - set(parameters)
        or any(not isinstance(state, dict) for state in optimizer["state"].values())
    ):
        raise ValueError("Optimizer state does not match its parameter groups")

    def finite(value):
        if isinstance(value, torch.Tensor):
            return bool(torch.isfinite(value).all())
        if isinstance(value, dict):
            return all(finite(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(finite(item) for item in value)
        return not isinstance(value, float) or math.isfinite(value)

    if not finite(optimizer):
        raise ValueError("Optimizer values must be finite")


def load_artifact(path, *, kind):
    data = torch.load(Path(path), map_location="cpu", weights_only=True)
    _validate(data, kind)
    return data


def export_actor(checkpoint, destination):
    data = load_artifact(checkpoint, kind="training")
    parameters = data["state"]["policy_state"]
    state = {
        name: {
            key.removeprefix(f"actor.{name}."): value
            for key, value in parameters.items()
            if key.startswith(f"actor.{name}.")
        }
        for name in ("motor", "estimator")
    }
    # Check the complete causal architecture and motor binding before publishing.
    _controller({**data, "state": state})
    save_artifact(
        destination,
        kind="actor",
        config=ExperimentConfig.from_dict(data["config"]),
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
    state = data["state"]
    if set(state) != {"motor", "estimator"}:
        raise ValueError("An actor contains only causal motor and estimator weights")
    with torch.random.fork_rng(devices=[]):
        motor, estimator = _fixed_modules()
    for name, module in (("motor", motor), ("estimator", estimator)):
        if not isinstance(state[name], dict) or any(
            not isinstance(v, torch.Tensor)
            or v.dtype != torch.float32
            or not torch.isfinite(v).all()
            for v in state[name].values()
        ):
            raise ValueError("Actor weights must be finite float32 tensors")
        module.load_state_dict(state[name], strict=True)
        module.to(device)
    manifest = data["motor_manifest"]
    controller = ROAHistoryController(
        motor,
        estimator,
        joint_names=manifest["joint_names"],
        default_position_rad=torch.tensor(
            manifest["configuration"]["default_position_rad"],
            dtype=torch.float32,
            device=device,
        ),
        artifact_sha256=roa_tensor_sha256(motor, estimator),
        actuator_profile=manifest["actuator_profile"],
    )
    validate_motor_contract(data["motor_contract"], controller.spec.manifest())
    return controller


@dataclass(frozen=True)
class LoadedActor:
    controller: ROAHistoryController
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
