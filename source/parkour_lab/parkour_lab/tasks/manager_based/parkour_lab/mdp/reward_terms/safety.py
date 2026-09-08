# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Safety and clearance penalties."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from ..navigation import route
from ..terrain import queries


def physical_failure(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Charge one impulse for physical failure, never a sum of failure reasons.

    Success and time limits are not physical failures. Read the already
    evaluated termination bits; do not repeat contact or route computations.
    Use instead of (not in addition to) the chassis/off-route reward terms.
    """
    manager = env.termination_manager
    failed = (
        manager.get_term("chassis_contact")
        | manager.get_term("off_route")
        | manager.get_term("fell_below_course")
    )
    return failed.float() / float(env.step_dt)


def base_clearance_below_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Penalty signal for the robot base/root being too close to the surface
    directly underneath it.

    The surface may be:
      - the ground
      - the top of an obstacle
      - later, another support surface

    This is a normalized L2 penalty:

        error = clamp((min_clearance - clearance) / min_clearance, 0, 1)
        penalty = error^2

    where:

        clearance = base_height - support_surface_height_under_base

    Normalizing by the commanded minimum gives the term a useful and stable
    ``[0, 1]`` scale instead of squaring a small distance measured in metres.
    A missed ray is neutral: unsupported gap flight is not equivalent to a
    valid surface being too close to the base. Use with a negative reward
    weight.

    Returns:
        [num_envs]
    """

    clearance, clearance_valid = queries._base_clearance_components(env, asset_cfg)

    min_clearance = route.current_min_clearances(env, default=0.27).to(
        device=clearance.device, dtype=clearance.dtype
    )

    normalization = min_clearance.clamp_min(torch.finfo(clearance.dtype).eps)
    clearance_error = torch.clamp(
        (min_clearance - clearance) / normalization,
        min=0.0,
        max=1.0,
    )

    return torch.where(
        clearance_valid,
        clearance_error.square(),
        torch.zeros_like(clearance_error),
    )


def chassis_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("chassis_contact"),
    timestep_independent: bool = False,
) -> torch.Tensor:
    """Penalty signal for fatal base or head contact.

    Set ``timestep_independent`` when contact also terminates the episode. The
    one-step signal is then divided by the control timestep before Isaac Lab's
    reward integration, so its configured weight is the exact crash penalty.

    Returns:
        Tensor of shape [num_envs].
    """

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # [num_envs, history_length, num_bodies, 3]
    net_forces = contact_sensor.data.net_forces_w_history

    if sensor_cfg.body_ids is not None:
        net_forces = net_forces[:, :, sensor_cfg.body_ids, :]

    # [num_envs, history_length, selected_bodies]
    force_norm = torch.linalg.norm(net_forces, dim=-1)

    # [num_envs]
    has_chassis_contact = torch.any(force_norm > threshold, dim=(1, 2))

    penalty = has_chassis_contact.float()
    if timestep_independent:
        return penalty / float(env.step_dt)
    return penalty
