"""Physical action bounds, observation spaces and native configuration identity."""

import hashlib
import json
from collections import Counter
from copy import deepcopy

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from parkour_lab.control.action import JointActionTransform
from parkour_lab.control.proprioception import FRAME_DIM
from parkour_lab.runtime.motor import _runtime_motor_binding

from .schema import information_schema


def action_transform(env):
    """Use the same affine native action and inward-rounded joint bounds."""
    robot = env.scene["robot"]
    default = robot.data.default_joint_pos
    limits = robot.data.soft_joint_pos_limits
    _, binding_id = _runtime_motor_binding(env)
    if (
        limits.shape != (env.num_envs, 12, 2)
        or not torch.isfinite(limits).all()
        or not torch.equal(limits, limits[:1].expand_as(limits))
        or not torch.all(limits[..., 0] < limits[..., 1])
    ):
        raise ValueError(
            "Task requires the shared 200/50 Hz bounded Go2 action interface"
        )
    return JointActionTransform(default, limits[0].tolist()), binding_id


def task_spaces(env, transform, *, training):
    """Expose one-robot and GPU-batch shapes without selecting a learning library."""
    actor = gym.spaces.Box(-np.inf, np.inf, shape=(FRAME_DIM,), dtype=np.float32)
    fields = {"actor": actor}
    if training:
        specification = information_schema(env.scene["robot"].joint_names)
        fields["training"] = gym.spaces.Dict(
            [
                (
                    name,
                    (
                        gym.spaces.MultiBinary(tuple(field["shape"]))
                        if field["dtype"] == "bool"
                        else gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=tuple(field["shape"]),
                            dtype=np.float32,
                        )
                    ),
                )
                for name, field in specification["fields"].items()
            ]
        )
    observation = gym.spaces.Dict(list(fields.items()))
    action = gym.spaces.Box(
        transform.lower[0].cpu().numpy(),
        transform.upper[0].cpu().numpy(),
        dtype=np.float32,
    )
    return (
        observation,
        action,
        batch_space(observation, env.num_envs),
        batch_space(action, env.num_envs),
    )


def configuration_id(cfg):
    """Hash the requested native configuration before managers resolve selectors."""
    return _identity(cfg.to_dict())


def reset_indices(env_ids, *, num_envs, device):
    """Normalize partial resets before native callbacks can mutate episode state."""
    if env_ids is None:
        return None
    try:
        ids = torch.as_tensor(env_ids, device=device)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(
            "Reset IDs require a one-dimensional integer sequence"
        ) from error
    if ids.ndim != 1 or (ids.numel() and ids.dtype not in (torch.int32, torch.int64)):
        raise ValueError("Reset IDs require a one-dimensional integer sequence")
    ids = ids.to(dtype=torch.long)
    if ((ids < 0) | (ids >= num_envs)).any() or ids.unique().numel() != ids.numel():
        raise ValueError("Reset IDs must be unique and within the replica batch")
    return ids


def terrain_metadata(env):
    """Record static family allocation without exposing live levels to actors."""
    specification = deepcopy(env.cfg.benchmark_terrain)
    if env.cfg.benchmark_task_id == "Go2-Flat-v1":
        counts = {"Flat": env.num_envs}
    else:
        columns = specification["columns"]
        types = env.scene.terrain.terrain_types.detach().cpu().tolist()
        if len(types) != env.num_envs or any(
            type(index) is not int or not 0 <= index < len(columns) for index in types
        ):
            raise ValueError(
                "Realized terrain families differ from the declared columns"
            )
        counts = dict(Counter(columns[index]["family"] for index in types))
    specification["realized_family_counts"] = counts
    return specification


def validate_task_config(cfg):
    """Allow native reward customization while enforcing the published task preset."""
    from .config import Go2Config, TerrainConfig, build_config

    reference = build_config(
        Go2Config(
            task_id=cfg.benchmark_task_id,
            num_envs=cfg.scene.num_envs,
            seed=cfg.seed,
            device=cfg.sim.device,
            terrain=TerrainConfig(**cfg.benchmark_terrain_options),
        ),
        training=cfg.benchmark_training,
    )
    actual, expected = cfg.to_dict(), reference.to_dict()
    actual.pop("rewards")
    expected.pop("rewards")
    if _identity(actual) != _identity(expected):
        raise ValueError("Only reward terms may differ from the fixed Go2 task preset")


def _identity(configuration):

    def encode(value):
        if isinstance(value, slice):
            return {"slice": [value.start, value.stop, value.step]}
        if isinstance(value, torch.Tensor):
            return {"dtype": str(value.dtype), "values": value.detach().cpu().tolist()}
        if isinstance(value, np.ndarray):
            return {"dtype": str(value.dtype), "values": value.tolist()}
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(
            f"Unsupported configuration identity value: {type(value).__name__}"
        )

    encoded = json.dumps(configuration, sort_keys=True, allow_nan=False, default=encode)
    return hashlib.sha256(encoded.encode()).hexdigest()
