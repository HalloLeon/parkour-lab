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
import math
import os
from pathlib import Path
import tempfile

import torch
from torch import nn


from .recurrent_runtime import (
    FRAME_DIM as FRAME_DIM,
    FRAME_TERMS as FRAME_TERMS,
    RECURRENT_OPERATOR_VERSION as RECURRENT_OPERATOR_VERSION,
    ActorOnlyController,
    _check_tensor,
    _load_actor_bytes,
    actor_bundle,
    actor_tensor_sha256,
)

RECURRENT_RESUME_MODE = "model_adam_fresh_environment_v1"


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
    return actor_tensor_sha256(policy.memory_a.rnn, policy.actor)


def load_recurrent_checkpoint(path, device="cpu"):
    """Strict policy loading; training continuation restores Adam separately."""
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
            or type(metadata["resume_supported"]) is not bool
            or metadata["actor_artifact_identity"] != "named_actor_state_tensor_sha256"
            or manifest["actuator_profile"]
            != "native_motor_sha256:" + metadata["motor_binding_sha256"]
        ):
            raise ValueError(
                "Recurrent checkpoint metadata differs from the native recipe"
            )
        resumed = metadata.get("resume_from")
        if resumed is not None:
            start = resumed["learning_updates"]
            if (
                metadata["resume_supported"] is not True
                or resumed["mode"] != RECURRENT_RESUME_MODE
                or type(start) is not int
                or not 0 < start < updates
                or type(metadata["session_learning_updates"]) is not int
                or metadata["session_learning_updates"] != updates - start
                or type(metadata["session_control_steps"]) is not int
                or metadata["session_control_steps"]
                != (updates - start) * recipe["num_steps_per_env"]
                or type(metadata["session_environment_transitions"]) is not int
                or metadata["session_environment_transitions"]
                != metadata["environment_transitions"] // updates * (updates - start)
            ):
                raise ValueError("Invalid resumed checkpoint/session accounting")
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


def validate_recurrent_optimizer(policy, state, expected_steps):
    """Validate native Adam's order, options, moments and completed step count."""
    groups = state["param_groups"]
    if len(groups) != 1 or set(state) != {"state", "param_groups"}:
        raise ValueError("Require one complete native Adam parameter group")
    rate = groups[0]["lr"]
    if type(rate) is not float or not math.isfinite(rate) or not 0 < rate <= 0.01:
        raise ValueError("Invalid saved Adam learning rate")
    parameters = list(policy.parameters())
    template = torch.optim.Adam(parameters, lr=rate).state_dict()["param_groups"][0]
    actual = dict(groups[0])
    # Newer PyTorch writes this explicit false default; older Adam omits it.
    template.setdefault("decoupled_weight_decay", False)
    actual.setdefault("decoupled_weight_decay", False)
    if (
        actual != template
        or type(expected_steps) is not int
        or expected_steps < 1
        or set(state["state"]) != set(range(len(parameters)))
    ):
        raise ValueError("Saved Adam options, parameter order or coverage differs")
    for index, parameter in enumerate(parameters):
        values = state["state"][index]
        if set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Incomplete Adam moments")
        step = values["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.shape != ()
            or step.dtype != torch.float32
            or not torch.isfinite(step)
            or step.item() != expected_steps
        ):
            raise ValueError("Saved Adam step count differs from completed updates")
        for name in ("exp_avg", "exp_avg_sq"):
            value = values[name]
            _check_tensor(value, parameter.shape, f"Adam {index} {name}")
            if value.dtype != parameter.dtype or (
                name == "exp_avg_sq" and torch.any(value < 0)
            ):
                raise ValueError("Invalid Adam moment dtype or variance")
    return rate


def restore_recurrent_optimizer(algorithm, state, expected_steps):
    """Restore Adam and PPO's adaptive LR scalar without a silent fresh restart."""
    if type(algorithm.optimizer) is not torch.optim.Adam or algorithm.optimizer.state:
        raise ValueError("Restore Adam once into a fresh native optimizer")
    parameters = list(algorithm.policy.parameters())
    groups = algorithm.optimizer.param_groups
    if len(groups) != 1 or [id(p) for p in groups[0]["params"]] != [
        id(p) for p in parameters
    ]:
        raise ValueError("Native Adam parameters differ from policy order")
    rate = validate_recurrent_optimizer(algorithm.policy, state, expected_steps)
    algorithm.optimizer.load_state_dict(copy.deepcopy(state))
    algorithm.learning_rate = rate
    restored = algorithm.optimizer.state_dict()
    expected_groups = copy.deepcopy(state["param_groups"])
    for group in (*restored["param_groups"], *expected_groups):
        group.setdefault("decoupled_weight_decay", False)
    if restored["param_groups"] != expected_groups or any(
        not torch.equal(value.cpu(), restored["state"][index][key].cpu())
        for index, values in state["state"].items()
        for key, value in values.items()
    ):
        raise RuntimeError("Adam state was not restored exactly")


class RecurrentOperatorAdapter(ActorOnlyController):
    """Validate the native training policy, then use the shared actor-only runtime."""

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

        if (
            type(policy) is not ActorCriticRecurrent
            or policy.obs_groups
            != {"policy": ["proprio"], "critic": ["policy", "terrain"]}
            or policy.actor_obs_normalization
            or type(policy.actor_obs_normalizer) is not nn.Identity
            or policy.state_dependent_std
            or type(policy.memory_c.rnn) is not nn.GRU
            or policy.memory_c.rnn.input_size != 312
            or any(not torch.isfinite(p).all() for p in policy.parameters())
        ):
            raise ValueError(
                "Require the native 45-D GRU operator policy and twelve named joints"
            )
        super().__init__(
            policy.memory_a.rnn,
            # RSL's MLP is a Sequential subclass; retain its fixed layers without
            # carrying the training-library container into the inference runtime.
            nn.Sequential(*policy.actor),
            joint_names=joint_names,
            default_position_rad=default_position_rad,
            artifact_sha256=artifact_sha256,
            actuator_profile=actuator_profile,
        )


@torch.inference_mode()
def export_recurrent_actor(checkpoint, output, *, motor_report=None):
    """CPU-only, checked extraction; atomically publish a new file without clobbering."""
    policy, metadata, checkpoint_sha = load_recurrent_checkpoint(checkpoint, "cpu")
    manifest = metadata["controller_manifest"]
    motor_report = (
        Path(motor_report)
        if motor_report is not None
        else Path(checkpoint).parent / "report.json"
    )
    try:
        binding = json.loads(motor_report.read_text())["motor_binding"]
        if type(binding) is not dict:
            raise ValueError("Missing source motor binding")
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(
            f"Require a source motor-binding report: {motor_report}; use --motor-report if stored elsewhere"
        ) from error
    original = RecurrentOperatorAdapter(
        policy,
        joint_names=tuple(manifest["joint_names"]),
        default_position_rad=torch.tensor(
            manifest["configuration"]["default_position_rad"]
        ),
        artifact_sha256=manifest["artifact_sha256"],
        actuator_profile=manifest["actuator_profile"],
    )
    payload = actor_bundle(
        original,
        source_checkpoint_sha256=checkpoint_sha,
        learning_updates=metadata["learning_updates"],
        motor_binding=binding,
    )
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    encoded = buffer.getvalue()
    restored, _, bundle_sha = _load_actor_bytes(encoded)
    generator = torch.Generator().manual_seed(47)
    hidden = restored.initial_state(3)
    previous = torch.zeros(3, 12)
    # Compare against native RSL, not merely two instances of the shared wrapper.
    for step in range(64):
        reset = torch.full((3,), step == 0, dtype=torch.bool)
        if step in (21, 42):
            reset[step // 21 - 1] = True
        hidden[:, reset] = 0
        previous[reset] = 0
        frame = torch.randn(3, FRAME_DIM, generator=generator)
        frame[:, -12:] = previous
        policy.reset(reset)
        expected = policy.act_inference({"proprio": frame})
        actual, hidden = restored.infer(frame, hidden)
        if not torch.equal(actual, expected) or not torch.equal(
            hidden, policy.memory_a.hidden_state
        ):
            raise RuntimeError(
                "Actor export differs from native CPU recurrent inference"
            )
        previous = actual
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=".operator_actor_", suffix=".tmp"
    ) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
        # Hard-link publication is exclusive even if another writer wins a race.
        os.link(stream.name, output)
    return {
        "status": "ACTOR_EXPORTED_CPU_PARITY_VERIFIED_NOT_DEPLOYED",
        "path": str(output.resolve()),
        "sha256": bundle_sha,
        "source_checkpoint_sha256": checkpoint_sha,
        "actor_tensor_sha256": manifest["artifact_sha256"],
        "source_learning_updates": metadata["learning_updates"],
        "bytes": len(encoded),
        "format": payload["format"],
        "motor_contract": payload["motor_contract"],
        "parity_steps": 64,
        "parity_batch": 3,
        "interface": restored.spec.manifest(),
        "exit_allowed": False,
    }
