"""SB3 smoke-check state fingerprints and external learner RNG ownership."""

import hashlib
import json
import random
from copy import deepcopy
from enum import Enum

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.save_util import recursive_getattr


def _digest(value, digest):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous().numpy()
    if isinstance(value, Enum):
        value = f"{type(value).__name__}:{value.value}"
    if isinstance(value, np.ndarray):
        digest.update(f"array:{value.dtype}:{value.shape}:".encode())
        digest.update(value.tobytes(order="C"))
    elif isinstance(value, dict):
        digest.update(b"dict{")
        for key in sorted(value, key=lambda item: (type(item).__name__, str(item))):
            _digest(key, digest)
            _digest(value[key], digest)
        digest.update(b"}")
    elif isinstance(value, (tuple, list)):
        digest.update(f"{type(value).__name__}[".encode())
        for item in value:
            _digest(item, digest)
        digest.update(b"]")
    elif isinstance(value, (str, int, float, bool, type(None), np.generic)):
        if isinstance(value, np.generic):
            value = value.item()
        digest.update(f"{type(value).__name__}:".encode())
        digest.update(json.dumps(value, allow_nan=False).encode())
        digest.update(b";")
    else:
        raise TypeError(f"Unsupported SB3 checkpoint state: {type(value).__name__}")


def state_sha256(value):
    digest = hashlib.sha256()
    _digest(value, digest)
    return digest.hexdigest()


def validate_model(model):
    """Keep this short example's resumable state deliberately bounded."""
    if type(model) not in (PPO, SAC):
        raise ValueError("This example supports genuine SB3 PPO and SAC only")
    if callable(model.learning_rate) or model.use_sde or model.action_noise is not None:
        raise ValueError("Use constant learning rate without gSDE or action noise")
    if model.get_vec_normalize_env() is not None:
        raise ValueError("This example does not use VecNormalize")
    if isinstance(model, SAC) and (model.n_steps != 1 or model.optimize_memory_usage):
        raise ValueError("SAC continuation requires one-step, ordinary replay")


def capture_model_state(model):
    """Capture saved networks, every optimizer/entropy tensor and learning counters.

    Rollout/physical episode state is deliberately excluded: save only after a
    completed learn call, then resume at a fresh physical episode boundary.
    """
    validate_model(model)
    settings = (
        "seed",
        "n_envs",
        "gamma",
        "learning_rate",
        "num_timesteps",
        "_n_updates",
        "_total_timesteps",
        "_current_progress_remaining",
        "_episode_num",
        "batch_size",
        "n_steps",
        "n_epochs",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "normalize_advantage",
        "max_grad_norm",
        "target_kl",
        "tau",
        "train_freq",
        "gradient_steps",
        "learning_starts",
        "target_update_interval",
        "target_entropy",
    )
    _, variables = model._get_torch_save_params()
    schedules = {
        name: [getattr(model, name)(fraction) for fraction in (0.0, 0.5, 1.0)]
        for name in ("lr_schedule", "clip_range", "clip_range_vf")
        if callable(getattr(model, name, None))
    }
    return deepcopy(
        dict(
            algorithm=type(model).__name__,
            spaces=dict(
                observation_shape=model.observation_space.shape,
                observation_dtype=str(model.observation_space.dtype),
                action_low=model.action_space.low,
                action_high=model.action_space.high,
            ),
            parameters=model.get_parameters(),
            variables={name: recursive_getattr(model, name) for name in variables},
            settings={
                name: getattr(model, name) for name in settings if hasattr(model, name)
            },
            schedules=schedules,
        )
    )


def model_state_sha256(model):
    return state_sha256(capture_model_state(model))


def replay_state_sha256(model):
    replay = model.replay_buffer
    occupied = replay.buffer_size if replay.full else replay.pos
    return state_sha256(
        dict(
            fields={
                name: getattr(replay, name)[:occupied]
                for name in (
                    "observations",
                    "next_observations",
                    "actions",
                    "rewards",
                    "dones",
                    "timeouts",
                )
            },
            cursor={
                name: getattr(replay, name)
                for name in (
                    "pos",
                    "full",
                    "buffer_size",
                    "n_envs",
                    "optimize_memory_usage",
                    "handle_timeout_termination",
                )
            },
        )
    )


def capture_rng(action_space=None):
    return dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
        action_space=(
            deepcopy(action_space.np_random.bit_generator.state)
            if action_space is not None
            else None
        ),
    )


def restore_rng(state, action_space=None):
    if state["cuda"] is not None:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Saved CUDA RNG device count differs")
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if action_space is not None:
        action_space.np_random.bit_generator.state = deepcopy(state["action_space"])
