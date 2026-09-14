# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native recurrent operator policy, strict checkpoint loading and causal adapter.

This module is independent of simulator tasks, training scripts and evaluation
protocols. RSL-RL is imported only when constructing or loading its native policy;
checkpoints retain the native state-dict keys and versioned controller manifest.
"""

from __future__ import annotations

import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path

import torch
from torch import nn


RECURRENT_OPERATOR_VERSION = "go2_operator_proprio_gru_v1"
FRAME_DIM = 45
FRAME_TERMS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
)


def _check_tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if (
        value.shape != shape
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite floating tensor of shape {shape}")


def recurrent_policy_config():
    """Native RSL-RL policy; fresh learning, not a stock-checkpoint conversion."""
    return {
        "class_name": "ActorCriticRecurrent",
        "rnn_type": "gru",
        "rnn_hidden_dim": 128,
        "rnn_num_layers": 1,
        "actor_hidden_dims": [128, 128, 128],
        "critic_hidden_dims": [128, 128, 128],
        "activation": "elu",
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "state_dependent_std": False,
        "noise_std_type": "log",
        "init_noise_std": 0.5,
    }


def build_recurrent_operator_policy(observations):
    """Construct an unmodified causal actor/asymmetric critic on its input device."""
    from rsl_rl.modules import ActorCriticRecurrent

    proprio = observations["proprio"]
    if proprio.ndim != 2 or len(proprio) < 1:
        raise ValueError("Require batched 45-D proprioceptive observations")
    for name, width in (("proprio", 45), ("policy", 48), ("terrain", 264)):
        value = observations[name]
        _check_tensor(value, (len(proprio), width), name)
        if value.device != proprio.device or value.dtype != proprio.dtype:
            raise ValueError("Recurrent observation device/dtype must agree")
    config = recurrent_policy_config()
    config.pop("class_name")
    return ActorCriticRecurrent(
        observations,
        {"policy": ["proprio"], "critic": ["policy", "terrain"]},
        12,
        **config,
    ).to(proprio)


def _recurrent_actor_sha256(policy):
    """Preserve the named actor-tensor identity used in acquisition checkpoints."""
    digest = hashlib.sha256()
    for prefix, module in (("memory", policy.memory_a.rnn), ("actor", policy.actor)):
        for name, value in module.state_dict().items():
            digest.update(
                f"{prefix}.{name}:{value.dtype}:{tuple(value.shape)}".encode()
            )
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_recurrent_checkpoint(path, device="cpu"):
    """Strict native state loading; no pickled callables, resume or oracle adapter."""
    from tensordict import TensorDict

    def file_sha256(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    path = Path(path).resolve(strict=True)
    digest = file_sha256(path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    try:
        metadata = saved["infos"]["recurrent_training"]
        updates = metadata["learning_updates"]
        recipe = metadata["recipe"]
        manifest = metadata["controller_manifest"]
        if (
            type(updates) is not int
            or updates < 1
            or type(saved["iter"]) is not int
            or saved["iter"] != updates - 1
            or type(saved["infos"]["learning_updates"]) is not int
            or saved["infos"]["learning_updates"] != updates
            or metadata["policy_version"] != RECURRENT_OPERATOR_VERSION
            or recipe["policy"] != recurrent_policy_config()
            or recipe["obs_groups"]
            != {"policy": ["proprio"], "critic": ["policy", "terrain"]}
            or recipe["algorithm"]["class_name"] != "PPO"
            or recipe.get("resume")
            or type(recipe["num_steps_per_env"]) is not int
            or recipe["num_steps_per_env"] < 1
            or type(metadata["control_steps"]) is not int
            or metadata["control_steps"] != updates * recipe["num_steps_per_env"]
            or type(metadata["environment_transitions"]) is not int
            or metadata["environment_transitions"] < metadata["control_steps"]
            or metadata["environment_transitions"] % metadata["control_steps"]
            or metadata["exit_allowed"] is not False
            or metadata["resume_supported"] is not False
            or metadata["actor_artifact_identity"] != "named_actor_state_tensor_sha256"
            or manifest["actuator_profile"]
            != "native_motor_sha256:" + metadata["motor_binding_sha256"]
        ):
            raise ValueError(
                "Recurrent checkpoint metadata differs from the native recipe"
            )
        json.dumps(metadata, allow_nan=False)
        observations = TensorDict(
            {
                name: torch.zeros(1, width)
                for name, width in (("proprio", 45), ("policy", 48), ("terrain", 264))
            },
            batch_size=[1],
        )
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(
            io.StringIO()
        ):
            policy = build_recurrent_operator_policy(observations)
        expected = policy.state_dict()
        state = saved["model_state_dict"]
        if set(state) != set(expected):
            raise ValueError("Recurrent checkpoint state keys differ")
        for name, value in state.items():
            _check_tensor(value, expected[name].shape, name)
            if value.dtype != expected[name].dtype:
                raise ValueError("Recurrent checkpoint parameter dtype differs")
        policy.load_state_dict(state, strict=True)
        policy.eval().requires_grad_(False)
        adapter = RecurrentOperatorAdapter(
            policy,
            joint_names=tuple(manifest["joint_names"]),
            default_position_rad=torch.tensor(
                manifest["configuration"]["default_position_rad"]
            ),
            artifact_sha256=_recurrent_actor_sha256(policy),
            actuator_profile=manifest["actuator_profile"],
        )
        # JSON canonicalization accounts only for tuple/list serialization.
        if json.dumps(adapter.spec.manifest(), sort_keys=True) != json.dumps(
            manifest, sort_keys=True
        ):
            raise ValueError("Recurrent checkpoint actor identity/interface differs")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Incomplete native recurrent checkpoint metadata") from error
    if file_sha256(path) != digest:
        raise ValueError("Recurrent checkpoint changed while loading")
    return policy.to(device), copy.deepcopy(metadata), digest


class RecurrentOperatorAdapter:
    """Extract only a causal actor; private zero-start memory never copies a rollout."""

    def __init__(
        self,
        policy,
        *,
        joint_names,
        default_position_rad,
        artifact_sha256,
        actuator_profile,
    ):
        from rsl_rl.modules import ActorCriticRecurrent
        from parkour_lab.learning.controller import ControllerSpec, SensorSpec

        if (
            type(policy) is not ActorCriticRecurrent
            or policy.obs_groups
            != {"policy": ["proprio"], "critic": ["policy", "terrain"]}
            or policy.actor_obs_normalization
            or type(policy.actor_obs_normalizer) is not nn.Identity
            or policy.state_dependent_std
            or type(policy.memory_a.rnn) is not nn.GRU
            or type(policy.memory_c.rnn) is not nn.GRU
            or policy.memory_a.rnn.input_size != 45
            or policy.memory_c.rnn.input_size != 312
            or policy.memory_a.rnn.hidden_size != 128
            or policy.memory_a.rnn.num_layers != 1
            or policy.memory_a.rnn.batch_first
            or policy.memory_a.rnn.bidirectional
            or policy.memory_a.rnn.dropout != 0
            or len(policy.actor) != 7
            or any(
                type(policy.actor[i]) is not nn.Linear
                or (policy.actor[i].in_features, policy.actor[i].out_features) != shape
                for i, shape in (
                    (0, (128, 128)),
                    (2, (128, 128)),
                    (4, (128, 128)),
                    (6, (128, 12)),
                )
            )
            or any(
                type(policy.actor[i]) is not nn.ELU or policy.actor[i].alpha != 1.0
                for i in (1, 3, 5)
            )
            or any(not torch.isfinite(p).all() for p in policy.parameters())
            or len(joint_names) != 12
            or len(set(joint_names)) != 12
        ):
            raise ValueError(
                "Require the native 45-D GRU operator policy and twelve named joints"
            )
        _check_tensor(default_position_rad, (12,), "default joint position")
        parameter = next(policy.actor.parameters())
        if (
            default_position_rad.device != parameter.device
            or default_position_rad.dtype != parameter.dtype
        ):
            raise ValueError("Default joint position must match policy device/dtype")
        self.memory = copy.deepcopy(policy.memory_a.rnn).eval().requires_grad_(False)
        self.actor = copy.deepcopy(policy.actor).eval().requires_grad_(False)
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
        from parkour_lab.learning.controller import JointTargets

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
        memory, hidden = self.memory(frame.unsqueeze(0), self.hidden_state)
        action = self.actor(memory.squeeze(0))
        _check_tensor(action, (len(frame), 12), "recurrent raw action")
        _check_tensor(hidden, (1, len(frame), 128), "GRU state")
        self.hidden_state = hidden.detach()
        return JointTargets(
            self.spec.joint_names, self.default_position_rad + 0.25 * action, action
        )
