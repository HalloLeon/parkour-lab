"""Causal ROA inference only: fixed history, estimator and shared motor.

No privileged encoder, critic, optimizer or simulator is retained. The versioned
tensor bundle records provenance and motor semantics, not deployment acceptance.
Hashes detect mismatches; they do not authenticate an artifact producer.
Use ControllerSession for sensor provenance and timing checks, and a trusted host
for physical actuation; this controller does not certify its input provider.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json

import torch
from torch import nn

from .controller import ControllerSpec, JointTargets, SensorSpec, finite_tensor
from .motor_contract import validate_motor_contract
from .operator_roa import (
    CODE_DIM,
    FRAME_DIM,
    HISTORY_LENGTH,
    CausalHistory,
    LatentMotor,
)

BUNDLE_VERSION = "go2_operator_roa_history_bundle_v1"
PREPROCESSING_VERSION = "go2_operator_roa_causal_history_v1"
RAW_ACTION_MEANING = (
    "stock unscaled action; q_target = default_q + 0.25 * raw_action; no clip"
)


def _sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def roa_tensor_sha256(motor, estimator):
    digest = hashlib.sha256()
    for prefix, module in (("motor", motor), ("estimator", estimator)):
        for name, value in module.state_dict().items():
            digest.update(
                f"{prefix}.{name}:{value.dtype}:{tuple(value.shape)}".encode()
            )
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _linear(layer, inputs, outputs, *, bias=True):
    if (
        type(layer) is not nn.Linear
        or (layer.in_features, layer.out_features) != (inputs, outputs)
        or set(layer.state_dict()) != ({"weight", "bias"} if bias else {"weight"})
        or (layer.bias is not None) != bias
    ):
        raise ValueError("Unexpected ROA linear layer")
    finite_tensor(layer.weight, (outputs, inputs))
    if bias:
        finite_tensor(layer.bias, (outputs,), layer.weight)


def _elu(layer):
    if (
        type(layer) is not nn.ELU
        or layer.alpha != 1
        or layer.inplace
        or layer.state_dict()
    ):
        raise ValueError("Require unchanged ROA ELU activations")


def _validate_modules(motor, estimator):
    if (
        type(motor) is not LatentMotor
        or set(motor._modules) != {"stock_first", "tail", "latent_projection"}
        or motor._parameters
        or motor._buffers
        or type(motor.tail) is not nn.Sequential
        or len(motor.tail) != 6
        or type(estimator) is not nn.Sequential
        or len(estimator) != 5
    ):
        raise ValueError("Require only the fixed ROA motor and causal estimator")
    _linear(motor.stock_first, 48, 128)
    _linear(motor.latent_projection, 8, 128, bias=False)
    for index, shape in ((1, (128, 128)), (3, (128, 128)), (5, (128, 12))):
        _linear(motor.tail[index], *shape)
        _elu(motor.tail[index - 1])
    for index, shape in ((0, (1125, 128)), (2, (128, 64)), (4, (64, 11))):
        _linear(estimator[index], *shape)
        if index:
            _elu(estimator[index - 1])
    for module in (motor.tail, estimator):
        if module._parameters or module._buffers:
            raise ValueError("Unexpected ROA container state")


class ROAHistoryController:
    """Private frozen causal weights and exactly one history push per action."""

    def __init__(
        self,
        motor,
        estimator,
        *,
        joint_names,
        default_position_rad,
        artifact_sha256,
        actuator_profile,
    ):
        _validate_modules(motor, estimator)
        if (
            not isinstance(joint_names, (tuple, list))
            or len(joint_names) != 12
            or any(not isinstance(name, str) or not name for name in joint_names)
            or len(set(joint_names)) != 12
            or not _sha256(artifact_sha256)
            or artifact_sha256 != roa_tensor_sha256(motor, estimator)
            or not isinstance(actuator_profile, str)
            or not actuator_profile.startswith("native_motor_sha256:")
            or not _sha256(actuator_profile.removeprefix("native_motor_sha256:"))
        ):
            raise ValueError(
                "Require named joints and exact ROA tensor/motor identities"
            )
        reference = motor.stock_first.weight
        finite_tensor(default_position_rad, (12,), reference)
        for module in (motor, estimator):
            for parameter in module.parameters():
                finite_tensor(parameter, parameter.shape, reference)
        if reference.dtype != torch.float32:
            raise ValueError("ROA runtime requires finite float32 weights and inputs")
        self.motor = copy.deepcopy(motor).eval().requires_grad_(False)
        self.estimator = copy.deepcopy(estimator).eval().requires_grad_(False)
        self.default_position_rad = default_position_rad.detach().clone()
        self._history = CausalHistory()
        self._reset_mask = None
        self._last_estimate = None
        self.spec = ControllerSpec(
            name="roa_operator_causal_history",
            artifact_sha256=artifact_sha256,
            preprocessing_version=PREPROCESSING_VERSION,
            joint_names=tuple(joint_names),
            period_s=0.02,
            actuator_profile=actuator_profile,
            sensors={
                "base_ang_vel": SensorSpec((3,), "rad/s", "body", 0.02),
                "projected_gravity": SensorSpec((3,), "unitless", "body", 0.02),
                "joint_position_relative_default": SensorSpec(
                    (12,), "rad", "joint", 0.02
                ),
                "joint_velocity": SensorSpec((12,), "rad/s", "joint", 0.02),
                "stock_previous_raw_action": SensorSpec(
                    (12,), "unitless", "joint", 0.02
                ),
            },
            raw_action_meaning=RAW_ACTION_MEANING,
            configuration={
                "default_position_rad": default_position_rad.tolist(),
                "action_scale": 0.25,
                "action_clip": None,
                "frame_dim": FRAME_DIM,
                "history_length": HISTORY_LENGTH,
                "history_order": "oldest to newest; includes current delivered frame",
                "input_order": [
                    "base_ang_vel",
                    "projected_gravity",
                    "velocity_commands",
                    "joint_pos",
                    "joint_vel",
                    "actions",
                ],
                "estimator": "1125 -> ELU MLP(128,64) -> velocity3, tanh(latent8)",
                "motor": "[velocity3,frame45] -> ELU MLP(128,128,128) -> 12; latent8 -> first preactivation",
                "reset": "repeat current reset frame 25 times; previous delivered raw action must be zero",
                "privileged_encoder_exported": False,
                "privileged_critic_exported": False,
            },
        )

    @property
    def history_frames(self):
        return None if self._history.frames is None else self._history.frames.clone()

    @property
    def last_estimate(self):
        return None if self._last_estimate is None else self._last_estimate.clone()

    @torch.inference_mode()
    def infer(self, frame, history):
        """Explicit-state kernel returning raw action and code, without a history push."""
        if not isinstance(frame, torch.Tensor) or frame.ndim != 2 or len(frame) < 1:
            raise ValueError("Require a nonempty causal frame batch")
        finite_tensor(frame, (len(frame), FRAME_DIM), self.default_position_rad)
        finite_tensor(history, (len(frame), HISTORY_LENGTH, FRAME_DIM), frame)
        if not torch.equal(frame, history[:, -1]):
            raise ValueError(
                "Newest history frame must equal the delivered current frame"
            )
        raw = self.estimator(history.detach().clone().flatten(1))
        code = torch.cat((raw[:, :3], raw[:, 3:].tanh()), dim=-1)
        finite_tensor(code, (len(frame), CODE_DIM), frame)
        action = self.motor(
            frame.detach(), torch.cat((code[:, :3].detach(), code[:, 3:]), dim=-1)
        )
        finite_tensor(action, (len(frame), 12), frame)
        return action, code

    @torch.inference_mode()
    def reset(self, mask):
        if (
            not isinstance(mask, torch.Tensor)
            or mask.ndim != 1
            or not len(mask)
            or mask.dtype != torch.bool
            or mask.device != self.default_position_rad.device
            or self._reset_mask is not None
        ):
            raise ValueError(
                "Require one matching boolean reset mask before each action"
            )
        if self._history.frames is None:
            if not mask.all():
                raise ValueError("First history frame must reset every environment")
        elif len(mask) != len(self._history.frames):
            raise ValueError("History batch size cannot change")
        self._reset_mask = mask.clone()

    @torch.inference_mode()
    def act(self, inputs):
        if self._reset_mask is None:
            raise ValueError("Call reset before each causal action")
        batch = len(self._reset_mask)
        if set(inputs.sensors) != set(self.spec.sensors):
            raise ValueError("Require exactly the five causal ROA sensor fields")
        for name, spec in self.spec.sensors.items():
            finite_tensor(
                inputs.sensors[name].value,
                (batch, *spec.shape),
                self.default_position_rad,
            )
        finite_tensor(inputs.command, (batch, 3), self.default_position_rad)
        previous = inputs.sensors["stock_previous_raw_action"].value
        if torch.any(previous[self._reset_mask] != 0):
            raise ValueError("Previous delivered raw action must be zero after reset")
        frame = torch.cat(
            (
                inputs.sensors["base_ang_vel"].value,
                inputs.sensors["projected_gravity"].value,
                inputs.command,
                inputs.sensors["joint_position_relative_default"].value,
                inputs.sensors["joint_velocity"].value,
                previous,
            ),
            dim=-1,
        )
        history = self._history.push(frame, self._reset_mask)
        action, self._last_estimate = self.infer(frame, history)
        position = self.default_position_rad + 0.25 * action
        finite_tensor(position, (batch, 12), self.default_position_rad)
        self._reset_mask = None
        return JointTargets(self.spec.joint_names, position, action)


def _source(checkpoint, updates, physical_reference):
    if (
        not _sha256(checkpoint)
        or type(updates) is not int
        or updates < 1
        or type(physical_reference) is not dict
        or set(physical_reference) != {"checkpoint", "agent.yaml", "env.yaml"}
        or any(not _sha256(value) for value in physical_reference.values())
    ):
        raise ValueError(
            "Require checkpoint/physical reference hashes and completed update count"
        )


def actor_bundle(
    controller,
    *,
    source_checkpoint_sha256,
    learning_updates,
    physical_reference,
    motor_contract,
):
    """Export frozen causal tensors and verified metadata, never live history."""
    _source(source_checkpoint_sha256, learning_updates, physical_reference)
    if (
        roa_tensor_sha256(controller.motor, controller.estimator)
        != controller.spec.artifact_sha256
    ):
        raise ValueError("ROA controller identity differs from its tensor weights")
    validate_motor_contract(motor_contract, controller.spec.manifest())
    return {
        "format": BUNDLE_VERSION,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_learning_updates": learning_updates,
        "source_physical_reference": copy.deepcopy(physical_reference),
        "controller_manifest": copy.deepcopy(controller.spec.manifest()),
        "state_dicts": {
            name: {
                key: value.detach().cpu().clone()
                for key, value in module.state_dict().items()
            }
            for name, module in (
                ("motor", controller.motor),
                ("estimator", controller.estimator),
            )
        },
        "motor_contract": copy.deepcopy(motor_contract),
        "exit_allowed": False,
        "deployment_allowed": False,
    }


def _fixed_modules():
    factory = {"device": "cpu", "dtype": torch.float32}
    stock = nn.Sequential(
        nn.Linear(48, 128, **factory),
        nn.ELU(),
        nn.Linear(128, 128, **factory),
        nn.ELU(),
        nn.Linear(128, 128, **factory),
        nn.ELU(),
        nn.Linear(128, 12, **factory),
    )
    estimator = nn.Sequential(
        nn.Linear(1125, 128, **factory),
        nn.ELU(),
        nn.Linear(128, 64, **factory),
        nn.ELU(),
        nn.Linear(64, 11, **factory),
    )
    return LatentMotor(stock), estimator


def _load_actor_bytes(encoded, device="cpu"):
    """Decode only the fixed tensor bundle; external file hash checks belong to the loader."""
    saved = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    try:
        if (
            type(saved) is not dict
            or set(saved)
            != {
                "format",
                "source_checkpoint_sha256",
                "source_learning_updates",
                "source_physical_reference",
                "controller_manifest",
                "state_dicts",
                "motor_contract",
                "exit_allowed",
                "deployment_allowed",
            }
            or saved["format"] != BUNDLE_VERSION
            or saved["exit_allowed"] is not False
            or saved["deployment_allowed"] is not False
            or type(saved["state_dicts"]) is not dict
            or set(saved["state_dicts"]) != {"motor", "estimator"}
        ):
            raise ValueError("Invalid causal ROA bundle schema")
        _source(
            saved["source_checkpoint_sha256"],
            saved["source_learning_updates"],
            saved["source_physical_reference"],
        )
        metadata = copy.deepcopy(
            {key: value for key, value in saved.items() if key != "state_dicts"}
        )
        # JSON-only finite metadata; no arbitrary layer configurations or RNG changes.
        json.dumps(metadata, sort_keys=True, allow_nan=False)
        with torch.random.fork_rng(devices=[]):
            motor, estimator = _fixed_modules()
        for name, module in (("motor", motor), ("estimator", estimator)):
            state = saved["state_dicts"][name]
            expected = module.state_dict()
            if type(state) is not dict or set(state) != set(expected):
                raise ValueError("Unexpected causal ROA tensor names")
            for key, value in state.items():
                finite_tensor(value, expected[key].shape, expected[key])
            module.load_state_dict(state, strict=True)
        manifest = saved["controller_manifest"]
        default = torch.tensor(
            manifest["configuration"]["default_position_rad"],
            device=device,
            dtype=torch.float32,
        )
        controller = ROAHistoryController(
            motor.to(device),
            estimator.to(device),
            joint_names=manifest["joint_names"],
            default_position_rad=default,
            artifact_sha256=manifest["artifact_sha256"],
            actuator_profile=manifest["actuator_profile"],
        )
        if json.dumps(
            controller.spec.manifest(), sort_keys=True, allow_nan=False
        ) != json.dumps(manifest, sort_keys=True, allow_nan=False):
            raise ValueError(
                "ROA bundle manifest differs from the fixed causal interface"
            )
        validate_motor_contract(saved["motor_contract"], manifest)
        return controller, metadata, hashlib.sha256(encoded).hexdigest()
    except (KeyError, TypeError, AttributeError, RuntimeError) as error:
        raise ValueError("Invalid causal ROA artifact") from error
