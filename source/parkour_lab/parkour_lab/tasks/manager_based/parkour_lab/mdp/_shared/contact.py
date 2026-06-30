from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import torch


def _contact_mask(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float
) -> torch.Tensor:
    """
    Boolean contact mask for selected contact-sensor bodies.

    Returns:
        [num_envs, num_bodies]
    """

    _require_body_ids(sensor_cfg, role="contact detection")

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    # [num_envs, history_length, num_sensor_bodies, 3]
    net_forces_w = contact_sensor.data.net_forces_w_history

    # [num_envs, history_length, selected_bodies, 3]
    net_forces_w = net_forces_w[:, :, sensor_cfg.body_ids, :]

    # [num_envs, history_length, selected_bodies]
    force_norm = torch.linalg.norm(net_forces_w, dim=-1)

    # [num_envs, selected_bodies]
    return torch.any(force_norm > threshold, dim=1)


def _require_body_ids(
    entity_cfg: SceneEntityCfg,
    *,
    role: str
) -> None:
    """
    Ensure that a SceneEntityCfg has resolved body_ids.

    Raises:
        ValueError: If body_ids are missing.
    """

    if entity_cfg.body_ids is None:
        raise ValueError(
            f"SceneEntityCfg for '{entity_cfg.name}' must resolve body_ids "
            f"when used for {role}. Pass body_names, for example "
            "body_names='.*_foot'."
        )


def _selected_contact_forces_w_history(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Contact forces for selected contact-sensor bodies.

    Returns:
        [num_envs, history_length, num_bodies, 3]
    """

    _require_body_ids(sensor_cfg, role="contact force selection")

    contact_sensor: ContactSensor = env.scene[sensor_cfg.name]

    return contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
