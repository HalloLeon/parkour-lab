# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Safety and clearance penalties."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .._shared import terrain
from ..commands import get_min_clearance


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

    This is an L2 penalty:

        penalty = max(min_clearance - clearance, 0)^2

    where:

        clearance = base_height - support_surface_height_under_base

    Use with a negative reward weight.

    Returns:
        [num_envs]
    """

    clearance = terrain._base_clearance(env, asset_cfg)

    min_clearance = get_min_clearance(env).to(
        device=clearance.device, dtype=clearance.dtype
    )

    clearance_error = torch.clamp(min_clearance - clearance, min=0.0)

    return clearance_error.square()


def base_contact(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_contact", body_names="trunk"),
) -> torch.Tensor:
    """
    Penalty signal for illegal trunk/base contact.

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
    has_illegal_contact = torch.any(force_norm > threshold, dim=(1, 2))

    return has_illegal_contact.float()
