"""Small batched inference boundary; not a plugin loader or hardware executor.

Adapters own feature packing, normalization and memory. The host owns commands,
sensor provenance, actuator configuration and safety. Times use one monotonic
clock shared by a synchronous batch; asynchronous hardware scheduling is not
implemented here. Privilege filtering is an API boundary, not a Python sandbox.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import copy
from types import MappingProxyType
from typing import Mapping, Protocol

import torch

VERSION = "operator_controller_v1"


@dataclass(frozen=True)
class SensorSpec:
    shape: tuple[int, ...]
    units: str
    frame: str
    max_age_s: float
    required: bool = True
    privileged: bool = False


@dataclass(frozen=True)
class Sample:
    value: torch.Tensor
    acquired_at_s: float
    valid: torch.Tensor  # One bool per environment; optional missing input = None.
    units: str
    frame: str
    privileged: bool = False  # Set by the trusted sensor provider, not the adapter.


@dataclass(frozen=True)
class ControllerSpec:
    name: str
    artifact_sha256: str
    preprocessing_version: str
    joint_names: tuple[str, ...]
    period_s: float
    actuator_profile: str
    sensors: Mapping[str, SensorSpec]
    raw_action_meaning: str
    configuration: Mapping[str, object] = field(default_factory=dict)

    def manifest(self):
        return {
            "version": VERSION,
            "name": self.name,
            "artifact_sha256": self.artifact_sha256,
            "preprocessing_version": self.preprocessing_version,
            "joint_names": list(self.joint_names),
            "period_s": self.period_s,
            "actuator_profile": self.actuator_profile,
            "sensors": {k: asdict(v) for k, v in self.sensors.items()},
            "raw_action_meaning": self.raw_action_meaning,
            "configuration": dict(self.configuration),
            "command": "body vx,vy,wz; m/s,m/s,rad/s; unchanged applied command",
            "output": "absolute joint-position targets in rad; no PD gain override",
            "clock": "shared monotonic seconds; synchronous batch",
        }


@dataclass(frozen=True)
class ControllerInput:
    time_s: float
    command: torch.Tensor
    command_time_s: float
    sensors: Mapping[str, Sample | None]


@dataclass(frozen=True)
class JointTargets:
    joint_names: tuple[str, ...]
    position_rad: torch.Tensor
    raw_action: torch.Tensor | None = None


class Controller(Protocol):
    spec: ControllerSpec

    def reset(self, mask: torch.Tensor) -> None: ...

    def act(self, inputs: ControllerInput) -> JointTargets: ...


def finite_tensor(value, shape, reference=None):
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != tuple(shape)
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
        or (
            reference is not None
            and (value.device != reference.device or value.dtype != reference.dtype)
        )
    ):
        raise ValueError(
            f"Expected finite floating tensor {tuple(shape)} with matching device/dtype"
        )


class ControllerSession:
    """Validate/project inputs and validate outputs; never deliver motor commands.

    Any failed step latches this session. Construct a new session/controller after
    handling the fault externally. No zero-action fallback or hot policy switch.
    A reset mask identifies the first frame of each new episode, before act().
    """

    def __init__(
        self,
        controller: Controller,
        *,
        joint_names: tuple[str, ...],
        actuator_profile: str,
        allow_privileged=False,
        capture=False,
    ):
        self.controller = controller
        spec = controller.spec
        if (
            not spec.name
            or not spec.preprocessing_version
            or not spec.raw_action_meaning
            or len(spec.artifact_sha256) != 64
            or any(c not in "0123456789abcdef" for c in spec.artifact_sha256)
            or not spec.joint_names
            or len(set(spec.joint_names)) != len(spec.joint_names)
            or any(not isinstance(n, str) or not n for n in spec.joint_names)
            or spec.joint_names != joint_names
            or not math.isfinite(spec.period_s)
            or spec.period_s <= 0
            or not actuator_profile
            or spec.actuator_profile != actuator_profile
        ):
            raise ValueError(
                "Invalid controller identity, joints, period or actuator profile"
            )
        for name, sensor in spec.sensors.items():
            if (
                not name
                or not sensor.units
                or not sensor.frame
                or not sensor.shape
                or any(type(n) is not int or n < 1 for n in sensor.shape)
                or not math.isfinite(sensor.max_age_s)
                or sensor.max_age_s < 0
            ):
                raise ValueError("Invalid sensor specification")
            if sensor.privileged and not allow_privileged:
                raise ValueError(f"Privileged sensor forbidden: {name}")
        self.manifest = copy.deepcopy(spec.manifest())
        encoded = json.dumps(self.manifest, sort_keys=True, allow_nan=False).encode()
        self.interface_sha256 = hashlib.sha256(encoded).hexdigest()
        self.time_s = None
        self.batch_signature = None
        self.faulted = False
        self.allow_privileged = allow_privileged
        self.reset_at_s = None
        self.record = None
        self.capture = capture

    def step(self, *, time_s, command, command_time_s, sensors, reset_mask):
        if self.faulted:
            raise RuntimeError("Controller session faulted; external recovery required")
        self.record = None
        try:
            return self._step(time_s, command, command_time_s, sensors, reset_mask)
        except Exception as error:
            self.faulted = True
            if self.record is not None:
                self.record.update(status="FAULT", error=str(error))
            raise

    def _step(self, time_s, command, command_time_s, sensors, reset_mask):
        spec = self.controller.spec
        if spec.manifest() != self.manifest:
            raise ValueError("Controller specification changed during session")
        if not math.isfinite(time_s) or time_s < 0:
            raise ValueError("Invalid decision time")
        if self.time_s is not None and not math.isclose(
            time_s - self.time_s, spec.period_s, rel_tol=0, abs_tol=1e-7
        ):
            raise ValueError(
                "Decision cadence changed; scheduler must handle this explicitly"
            )
        finite_tensor(command, (len(command), 3))
        if (
            not math.isfinite(command_time_s)
            or not 0 <= time_s - command_time_s <= spec.period_s + 1e-9
        ):
            raise ValueError(
                "Stale/future applied command; host must resolve the command lease"
            )
        batch = len(command)
        signature = (batch, command.device, command.dtype)
        if batch < 1 or (
            self.batch_signature is not None and signature != self.batch_signature
        ):
            raise ValueError("Batch/device/dtype changed")
        if (
            reset_mask.shape != (batch,)
            or reset_mask.dtype != torch.bool
            or reset_mask.device != command.device
            or (self.time_s is None and not reset_mask.all())
        ):
            raise ValueError(
                "Reset mask must initialize every environment at cold start"
            )
        reset_at = (
            torch.full((batch,), time_s, dtype=torch.float64, device=command.device)
            if self.reset_at_s is None
            else self.reset_at_s.clone()
        )
        reset_at[reset_mask] = time_s
        delivered = {}
        for name, requirement in spec.sensors.items():
            sample = sensors.get(name)
            if sample is None:
                if requirement.required:
                    raise ValueError(f"Missing required sensor: {name}")
                delivered[name] = None
                continue
            finite_tensor(sample.value, (batch, *requirement.shape), command)
            if sample.units != requirement.units or sample.frame != requirement.frame:
                raise ValueError(f"Sensor units/frame differ: {name}")
            if sample.privileged and (
                not self.allow_privileged or not requirement.privileged
            ):
                raise ValueError(f"Privileged provider sample forbidden: {name}")
            age = time_s - sample.acquired_at_s
            if not math.isfinite(age) or age < 0 or age > requirement.max_age_s + 1e-9:
                raise ValueError(f"Stale/future sensor: {name}")
            if (
                sample.valid.shape != (batch,)
                or sample.valid.dtype != torch.bool
                or sample.valid.device != command.device
                or (requirement.required and not sample.valid.all())
            ):
                raise ValueError(f"Invalid sensor validity: {name}")
            if torch.any(sample.valid & (sample.acquired_at_s < reset_at)):
                raise ValueError(f"Sensor predates episode reset: {name}")
            value = sample.value.detach().clone()
            value[~sample.valid] = (
                0  # Invalid optional rows must not leak old episode data.
            )
            delivered[name] = Sample(
                value,
                sample.acquired_at_s,
                sample.valid.clone(),
                sample.units,
                sample.frame,
                sample.privileged,
            )
        # Snapshot before inference: even a badly behaved adapter cannot rewrite
        # what the recorder says it received. Store one step, not another run log.
        self.record = (
            None
            if not self.capture
            else {
                "version": VERSION,
                "interface_sha256": self.interface_sha256,
                "time_s": time_s,
                "applied_command": command.detach().cpu().tolist(),
                "applied_at_s": command_time_s,
                "reset_mask": reset_mask.cpu().tolist(),
                "sensors": {
                    name: (
                        None
                        if sample is None
                        else {
                            "value": sample.value.cpu().tolist(),
                            "acquired_at_s": sample.acquired_at_s,
                            "valid": sample.valid.cpu().tolist(),
                            "units": sample.units,
                            "frame": sample.frame,
                            "privileged": sample.privileged,
                        }
                    )
                    for name, sample in delivered.items()
                },
                "joint_names": list(spec.joint_names),
                "requested_position_rad": None,
                "raw_action": None,
                "status": "INFERENCE_PENDING",
            }
        )
        # Only declared inputs reach the adapter. No environment or full sensor dict.
        self.controller.reset(reset_mask.clone())
        with torch.inference_mode():
            output = self.controller.act(
                ControllerInput(
                    time_s,
                    command.detach().clone(),
                    command_time_s,
                    MappingProxyType(delivered),
                )
            )
        if output.joint_names != spec.joint_names:
            raise ValueError("Output joint order differs from contract")
        finite_tensor(output.position_rad, (batch, len(spec.joint_names)), command)
        if output.raw_action is not None:
            if output.raw_action.ndim != 2 or output.raw_action.shape[1] < 1:
                raise ValueError("Raw action must be a batched private vector")
            finite_tensor(
                output.raw_action, (batch, output.raw_action.shape[1]), command
            )
        self.time_s, self.batch_signature = time_s, signature
        self.reset_at_s = reset_at
        if self.record is not None:
            self.record.update(
                status="TARGETS_COMPUTED_NOT_DELIVERED",
                requested_position_rad=output.position_rad.detach().cpu().tolist(),
                raw_action=(
                    None
                    if output.raw_action is None
                    else output.raw_action.detach().cpu().tolist()
                ),
            )
        return JointTargets(
            output.joint_names,
            output.position_rad.detach().clone(),
            None if output.raw_action is None else output.raw_action.detach().clone(),
        )
