"""Deterministic inference from an already loaded Stable-Baselines3 model."""

import hashlib
import json

import numpy as np
import torch

from .checkpoint import load_model


class SB3FramePolicy:
    """Expose blind frames and native encoded actions without environment access."""

    def __init__(self, model):
        if model.observation_space.shape != (49,) or model.action_space.shape != (12,):
            raise ValueError("Require the common 49D observation and 12D action")
        if model.get_vec_normalize_env() is not None:
            raise ValueError("This example uses unnormalized observations and rewards")
        self.model = model
        model.policy.set_training_mode(False)

    def reset(self, mask):
        # These feed-forward PPO/SAC policies have no episode memory.
        if mask.dtype != torch.bool or mask.ndim != 1:
            raise ValueError("Require a per-row boolean reset mask")

    def state_sha256(self):
        digest = hashlib.sha256(type(self.model).__name__.encode())
        values = {
            **dict(self.model.policy.state_dict()),
            "action_space.low": self.model.action_space.low,
            "action_space.high": self.model.action_space.high,
        }
        for name, value in sorted(values.items()):
            array = (
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor)
                else np.asarray(value)
            )
            digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    @torch.inference_mode()
    def act(self, observation):
        # predict performs PPO clipping or SAC unscaling in the saved action space.
        action, _ = self.model.predict(
            observation.detach().cpu().numpy(), deterministic=True
        )
        return torch.as_tensor(
            action, device=observation.device, dtype=observation.dtype
        ).clone()


def load_policy(checkpoint, *, device="cpu"):
    model, receipt = load_model(checkpoint, device=device)
    return SB3FramePolicy(model), receipt
