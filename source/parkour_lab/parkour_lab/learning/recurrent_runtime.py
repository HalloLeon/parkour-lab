# Copyright (c) 2026, Leon Yi Bai
# SPDX-License-Identifier: BSD-3-Clause
"""Torch-only operator inference: no simulator, RSL-RL, critic or optimizer.

The artifact is a versioned tensor/state-dict bundle, not executable model code,
a training-resume checkpoint or a hardware controller. Use ControllerSession for
input/output validation and a trusted host for commands, sensing and actuation.
Hashes detect mismatches; they do not authenticate an untrusted artifact producer.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import torch
from torch import nn

from .controller import ControllerSpec, JointTargets, SensorSpec
from .motor_contract import make_motor_contract, validate_motor_contract

RECURRENT_OPERATOR_VERSION = "go2_operator_proprio_gru_v1"
ACTOR_BUNDLE_VERSION = "go2_operator_actor_bundle_v1"
BOUND_ACTOR_BUNDLE_VERSION = "go2_operator_actor_bundle_v2"
FRAME_DIM = 45
FRAME_TERMS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
)


def _check_tensor(value, shape, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != shape
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite floating tensor of shape {shape}")


def _sha256_string(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def actor_tensor_sha256(memory, actor):
    """Same named-tensor identity as the original native training checkpoints."""
    digest = hashlib.sha256()
    for prefix, module in (("memory", memory), ("actor", actor)):
        for name, value in module.state_dict().items():
            digest.update(
                f"{prefix}.{name}:{value.dtype}:{tuple(value.shape)}".encode()
            )
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class ActorOnlyController:
    """Shared native/exported inference implementation with private causal state."""

    def __init__(
        self,
        memory,
        actor,
        *,
        joint_names,
        default_position_rad,
        artifact_sha256,
        actuator_profile,
    ):
        if (
            type(memory) is not nn.GRU
            or memory.input_size != FRAME_DIM
            or memory.hidden_size != 128
            or memory.num_layers != 1
            or not memory.bias
            or memory.batch_first
            or memory.bidirectional
            or memory.dropout != 0
            or type(actor) is not nn.Sequential
            or len(actor) != 7
            or any(
                type(actor[i]) is not nn.Linear
                or (actor[i].in_features, actor[i].out_features) != shape
                or actor[i].bias is None
                for i, shape in (
                    (0, (128, 128)),
                    (2, (128, 128)),
                    (4, (128, 128)),
                    (6, (128, 12)),
                )
            )
            or any(
                type(actor[i]) is not nn.ELU
                or actor[i].alpha != 1.0
                or actor[i].inplace
                for i in (1, 3, 5)
            )
            or len(joint_names) != 12
            or len(set(joint_names)) != 12
            or any(not isinstance(name, str) or not name for name in joint_names)
            or not _sha256_string(artifact_sha256)
            or not isinstance(actuator_profile, str)
            or not actuator_profile
        ):
            raise ValueError("Require the fixed 45-D GRU actor and twelve named joints")
        _check_tensor(default_position_rad, (12,), "default joint position")
        parameter = next(actor.parameters())
        if any(
            p.dtype != parameter.dtype
            or p.device != parameter.device
            or not torch.isfinite(p).all()
            for module in (memory, actor)
            for p in module.parameters()
        ) or (
            default_position_rad.device != parameter.device
            or default_position_rad.dtype != parameter.dtype
        ):
            raise ValueError(
                "Actor, memory and default pose must match device/dtype and be finite"
            )
        self.memory = copy.deepcopy(memory).eval().requires_grad_(False)
        self.actor = copy.deepcopy(actor).eval().requires_grad_(False)
        # Deepcopy splits cuDNN's packed GRU storage; repack the private copy once.
        self.memory.flatten_parameters()
        self.default_position_rad = default_position_rad.detach().clone()
        self.hidden_state = None
        self.reset_mask = None
        self.spec = ControllerSpec(
            name="recurrent_operator_proprio",
            artifact_sha256=artifact_sha256,
            preprocessing_version=RECURRENT_OPERATOR_VERSION,
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
            raw_action_meaning="stock unscaled action; q_target = default_q + 0.25 * raw_action; no clip",
            configuration={
                "default_position_rad": default_position_rad.tolist(),
                "action_scale": 0.25,
                "action_clip": None,
                "actor": "45 -> GRU(128,1) -> ELU MLP(128,128,128) -> 12",
                "input_order": [name for name, _ in FRAME_TERMS],
                "reset": "zero GRU state; previous delivered raw action must be zero",
                "privileged_critic_exported": False,
            },
        )

    def initial_state(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError("Require a positive integer batch size")
        return self.default_position_rad.new_zeros((1, batch, 128))

    @torch.inference_mode()
    def infer(self, frame, hidden):
        """Explicit-state kernel; caller owns resets. Does not mutate private memory."""
        if not isinstance(frame, torch.Tensor) or frame.ndim != 2 or len(frame) < 1:
            raise ValueError("Require a nonempty batched proprioceptive frame")
        _check_tensor(frame, (len(frame), FRAME_DIM), "proprioceptive frame")
        _check_tensor(hidden, (1, len(frame), 128), "GRU state")
        reference = self.default_position_rad
        if any(
            v.dtype != reference.dtype or v.device != reference.device
            for v in (frame, hidden)
        ):
            raise ValueError("Frame/state must match actor device/dtype")
        memory, next_hidden = self.memory(frame.unsqueeze(0), hidden)
        action = self.actor(memory.squeeze(0))
        _check_tensor(action, (len(frame), 12), "recurrent raw action")
        _check_tensor(next_hidden, (1, len(frame), 128), "GRU state")
        return action, next_hidden

    @torch.inference_mode()
    def reset(self, mask):
        if (
            mask.ndim != 1
            or mask.dtype != torch.bool
            or mask.device != self.default_position_rad.device
        ):
            raise ValueError("Require a matching boolean reset mask")
        if self.hidden_state is None:
            if not len(mask) or not mask.all():
                raise ValueError("First recurrent frame must reset every environment")
        else:
            _check_tensor(self.hidden_state, (1, len(mask), 128), "GRU state")
            self.hidden_state[:, mask] = 0
        self.reset_mask = mask.clone()

    @torch.inference_mode()
    def act(self, inputs):
        sensors = inputs.sensors
        previous = sensors["stock_previous_raw_action"].value
        if self.reset_mask is None or torch.any(previous[self.reset_mask] != 0):
            raise ValueError("Previous delivered action must be zero after reset")
        frame = torch.cat(
            (
                sensors["base_ang_vel"].value,
                sensors["projected_gravity"].value,
                inputs.command,
                sensors["joint_position_relative_default"].value,
                sensors["joint_velocity"].value,
                previous,
            ),
            dim=-1,
        )
        _check_tensor(frame, (len(self.reset_mask), FRAME_DIM), "proprioceptive frame")
        hidden = (
            self.initial_state(len(frame))
            if self.hidden_state is None
            else self.hidden_state
        )
        action, self.hidden_state = self.infer(frame, hidden)
        return JointTargets(
            self.spec.joint_names, self.default_position_rad + 0.25 * action, action
        )


def actor_bundle(
    controller, *, source_checkpoint_sha256, learning_updates, motor_binding=None
):
    """Take only actor weights and the exact inference interface, never live state."""
    if (
        not _sha256_string(source_checkpoint_sha256)
        or type(learning_updates) is not int
        or learning_updates < 1
    ):
        raise ValueError(
            "Require the source checkpoint hash and completed update count"
        )
    if (
        actor_tensor_sha256(controller.memory, controller.actor)
        != controller.spec.artifact_sha256
    ):
        raise ValueError("Controller actor identity differs from its tensor weights")
    bound = motor_binding is not None
    return {
        "format": BOUND_ACTOR_BUNDLE_VERSION if bound else ACTOR_BUNDLE_VERSION,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "source_learning_updates": learning_updates,
        "controller_manifest": copy.deepcopy(controller.spec.manifest()),
        "state_dicts": {
            name: {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
            for name, module in (
                ("memory", controller.memory),
                ("actor", controller.actor),
            )
        },
        "exit_allowed": False,
        **(
            {
                "motor_contract": make_motor_contract(
                    motor_binding, controller.spec.actuator_profile
                )
            }
            if bound
            else {}
        ),
    }


def _load_actor_bytes(encoded, device="cpu"):
    saved = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    try:
        bound = (
            type(saved) is dict and saved.get("format") == BOUND_ACTOR_BUNDLE_VERSION
        )
        if (
            type(saved) is not dict
            or set(saved)
            != {
                "format",
                "source_checkpoint_sha256",
                "source_learning_updates",
                "controller_manifest",
                "state_dicts",
                "exit_allowed",
            }
            | ({"motor_contract"} if bound else set())
            or saved["format"] not in (ACTOR_BUNDLE_VERSION, BOUND_ACTOR_BUNDLE_VERSION)
            or not _sha256_string(saved["source_checkpoint_sha256"])
            or type(saved["source_learning_updates"]) is not int
            or saved["source_learning_updates"] < 1
            or saved["exit_allowed"] is not False
            or type(saved["state_dicts"]) is not dict
            or set(saved["state_dicts"]) != {"memory", "actor"}
        ):
            raise ValueError("Invalid actor-only artifact schema")
        manifest = saved["controller_manifest"]
        # Fixed architecture only; no import paths, arbitrary layer configs or RNG side effects.
        with torch.random.fork_rng(devices=[]):
            factory = {"device": "cpu", "dtype": torch.float32}
            memory = nn.GRU(FRAME_DIM, 128, 1, **factory)
            actor = nn.Sequential(
                nn.Linear(128, 128, **factory),
                nn.ELU(),
                nn.Linear(128, 128, **factory),
                nn.ELU(),
                nn.Linear(128, 128, **factory),
                nn.ELU(),
                nn.Linear(128, 12, **factory),
            )
        for name, module in (("memory", memory), ("actor", actor)):
            state = saved["state_dicts"][name]
            expected = module.state_dict()
            if type(state) is not dict or set(state) != set(expected):
                raise ValueError(f"Unexpected actor-only state keys: {name}")
            for key, value in state.items():
                _check_tensor(value, expected[key].shape, f"{name}.{key}")
                if value.dtype != torch.float32:
                    raise ValueError("Actor artifacts require finite float32 weights")
            module.load_state_dict(state, strict=True)
        if actor_tensor_sha256(memory, actor) != manifest["artifact_sha256"]:
            raise ValueError("Actor tensor hash mismatch")
        controller = ActorOnlyController(
            memory.to(device),
            actor.to(device),
            joint_names=tuple(manifest["joint_names"]),
            default_position_rad=torch.tensor(
                manifest["configuration"]["default_position_rad"],
                dtype=torch.float32,
                device=device,
            ),
            artifact_sha256=manifest["artifact_sha256"],
            actuator_profile=manifest["actuator_profile"],
        )
        if (
            not manifest["actuator_profile"].startswith("native_motor_sha256:")
            or not _sha256_string(
                manifest["actuator_profile"].removeprefix("native_motor_sha256:")
            )
            or json.dumps(controller.spec.manifest(), sort_keys=True, allow_nan=False)
            != json.dumps(manifest, sort_keys=True, allow_nan=False)
        ):
            raise ValueError(
                "Actor-only controller manifest differs from the fixed interface"
            )
        if bound:
            validate_motor_contract(saved["motor_contract"], manifest)
    except (KeyError, TypeError, AttributeError, RuntimeError) as error:
        raise ValueError("Incomplete or malformed actor-only artifact") from error
    metadata = {k: copy.deepcopy(v) for k, v in saved.items() if k != "state_dicts"}
    return controller, metadata, hashlib.sha256(encoded).hexdigest()


def load_actor_bundle(path, device="cpu", *, expected_sha256=None):
    """Load trusted tensor-only data, optionally bound to an external file receipt."""
    encoded = Path(path).read_bytes()
    if expected_sha256 is not None and (
        not _sha256_string(expected_sha256)
        or hashlib.sha256(encoded).hexdigest() != expected_sha256
    ):
        raise ValueError("Actor artifact file hash mismatch")
    return _load_actor_bytes(encoded, device)
