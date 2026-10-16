"""Tensor inputs and a deterministic learner host; not simulator evidence."""

from types import SimpleNamespace

import torch
from tensordict import TensorDict
from torch import nn

from parkour_lab.control.controller import Sample
from parkour_lab.control.proprioception import CONTACT_SLICE, FRAME_DIM


def stock_motor():
    return nn.Sequential(
        nn.Linear(FRAME_DIM + 3, 128),
        nn.ELU(),
        nn.Linear(128, 128),
        nn.ELU(),
        nn.Linear(128, 128),
        nn.ELU(),
        nn.Linear(128, 12),
    )


def random_frames(*batch_shape):
    result = torch.randn(*batch_shape, FRAME_DIM)
    result[..., CONTACT_SLICE] = torch.randint(2, (*batch_shape, 4)).float()
    return result


def observations(count=8):
    history = random_frames(count, 25)
    return TensorDict(
        {
            "policy": history[:, -1].clone(),
            "history": history.flatten(1),
            "dynamics": torch.randn(count, 7),
            "critic_state": torch.randn(count, FRAME_DIM + 3),
            "terrain": torch.randn(count, 264),
        },
        batch_size=[count],
    )


def actor_input(obs, actor=None):
    groups = actor.privileged_obs_groups if actor else ["policy", "dynamics", "history"]
    return torch.cat([obs[name] for name in groups], dim=-1)


def packet(controller, frame, step, reset):
    offsets = ((0, 3), (3, 6), (9, 21), (21, 33), (33, 45), (45, 49))
    return {
        "time_s": step * 0.02,
        "command_time_s": step * 0.02,
        "command": frame[:, 6:9],
        "reset_mask": reset,
        "sensors": {
            name: Sample(
                frame[:, start:end],
                step * 0.02,
                torch.ones(len(frame), dtype=torch.bool),
                spec.units,
                spec.frame,
            )
            for (name, spec), (start, end) in zip(
                controller.spec.sensors.items(), offsets
            )
        },
    }


class TensorHost:
    """Deterministic tensor fixture, not a simulator or physical acceptance test."""

    def __init__(self, obs):
        self.env = SimpleNamespace(device="cpu")
        self.obs = obs.clone()
        self.phases = []

    def step(self, action, phase):
        self.phases.append(phase)
        count = len(action)
        reward = 1.0 - action.square().mean(-1)
        return self.obs.clone(), reward, torch.zeros(count, dtype=torch.bool), {}
