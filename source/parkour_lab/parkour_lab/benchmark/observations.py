"""Owned causal observations; terminal previews do not advance sensor streams."""

import torch

from parkour_lab.control.proprioception import CAUSAL_NOISE_AMPLITUDES, FRAME_DIM
from parkour_lab.runtime.native import causal_sensor_noise


def actor_frame(env, clean, *, final_env_ids=None):
    """Apply shared sensor noise once, retaining commands/actions/contact bits.

    Compact outgoing finals preview the next draw for their own rows. The
    ordinary reset/survivor observation stream therefore does not depend on
    whether a trainer uses terminal observations.
    """
    count = env.num_envs if final_env_ids is None else len(final_env_ids)
    if clean.shape != (count, FRAME_DIM) or not torch.isfinite(clean).all():
        raise ValueError("Require finite causal frames with 49 values per robot")
    frame = clean.detach().clone()
    if final_env_ids is None:
        noise = causal_sensor_noise(env)
    else:
        noise = env.parkour_randomization.uniform(
            final_env_ids.cpu().numpy(),
            30,
            "observation-noise",
            low=-1.0,
            high=1.0,
            advance=False,
        )
        noise *= CAUSAL_NOISE_AMPLITUDES
        noise = torch.as_tensor(noise, dtype=frame.dtype, device=frame.device)
    if noise is not None:
        frame[:, :6] += noise[:, :6]
        frame[:, 9:33] += noise[:, 6:]
    return frame


def task_observation(env, native, *, training, final_env_ids=None, information=None):
    """Select the permitted role-specific fields; never forward native groups."""
    if training and final_env_ids is not None and information is None:
        raise RuntimeError("Missing owned outgoing training information")
    observation = {
        "actor": actor_frame(env, native["proprio"], final_env_ids=final_env_ids)
    }
    if training:
        if information is None:
            from .information import capture_information

            information = capture_information(env)
        observation["training"] = information
    return observation
