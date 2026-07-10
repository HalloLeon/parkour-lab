import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from .contact import _require_body_ids
from .runtime import _get_or_init_env_buffer, _set_env_buffer


def _root_forward_xy_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Robot root forward direction in world XY.

    Returns:
        [num_envs, 2]
    """

    asset: Articulation = env.scene[asset_cfg.name]

    # In the robot's own body frame, we define "forward" as +X.
    #
    # So before considering the robot's current orientation, the local forward
    # vector is:
    #
    #     forward_b = [1, 0, 0]
    #
    # The suffix "_b" means body frame.
    forward_b = torch.zeros(
        (asset.data.root_quat_w.shape[0], 3), device=asset.data.root_quat_w.device, dtype=asset.data.root_quat_w.dtype
    )

    forward_b[:, 0] = 1.0

    # Rotate the body-frame forward vector into the world frame.
    #
    # If q is the robot root orientation, this applies:
    #
    #     forward_w = q * forward_b * q^-1
    #
    # The suffix "_w" means world frame.
    forward_w = quat_apply(asset.data.root_quat_w, forward_b)

    # Keep only the horizontal part of the world-frame forward vector.
    #
    # We discard z because goal navigation uses the ground-plane heading:
    #
    #     forward_w  = [x, y, z]
    #     forward_xy = [x, y]
    #
    # This tells us where the robot is facing in the world XY plane.
    forward_xy = forward_w[:, :2]

    # Normalize the XY vector to unit length.
    #
    #     forward_xy_unit = forward_xy / sqrt(x^2 + y^2)
    #
    # This makes the result a direction only, independent of vector magnitude.
    #
    # clamp_min(1.0e-6) avoids division by zero if the horizontal projection is
    # extremely small, for example if the robot is nearly vertical.
    return forward_xy / torch.linalg.norm(forward_xy, dim=-1, keepdim=True).clamp_min(1.0e-6)


def _root_height_env(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Robot root/base height in each environment's local frame.

    Returns:
        [num_envs]
    """

    return _root_pos_env(env, asset_cfg)[:, 2]


def _root_lin_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Robot root linear velocity in the world XY plane.

    Returns:
        [num_envs, 2]
    """

    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.root_lin_vel_w[:, :2]


def _root_lin_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Robot root linear velocity in the world Z direction.


    Returns:
        [num_envs]
    """

    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.root_lin_vel_w[:, 2]


def _root_pos_env(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Robot root position in each environment's local frame.

    Returns:
        [num_envs, 3]
    """

    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def _root_projected_gravity_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Projected gravity XY components.

    This is a compact roll/pitch orientation signal.

    Returns:
        [num_envs, 2]
    """

    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.projected_gravity_b[:, :2]


def _root_roll_pitch_rate(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Robot root roll/pitch angular velocity.

    Returns:
        [num_envs, 2]
    """

    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.root_ang_vel_b[:, :2]


def _root_xy_delta_from_previous(
    env: ManagerBasedRLEnv, *, buffer_name: str, reset_mask: torch.Tensor, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Root XY displacement since the previous control step.

    The previous-position buffer is always updated.

    Returns:
        [num_envs, 2]
    """

    current_root_xy = _root_pos_env(env, asset_cfg)[:, :2]

    previous_root_xy = _get_or_init_env_buffer(env, name=buffer_name, value=current_root_xy)

    root_delta_xy = current_root_xy - previous_root_xy

    root_delta_xy = torch.where(reset_mask[:, None], torch.zeros_like(root_delta_xy), root_delta_xy)

    _set_env_buffer(env, name=buffer_name, value=current_root_xy)

    return root_delta_xy


def _selected_body_lin_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Linear velocity of selected articulation bodies in world frame.

    Returns:
        [num_envs, num_bodies, 3]
    """

    _require_body_ids(asset_cfg, role="body velocity selection")

    asset: Articulation = env.scene[asset_cfg.name]

    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]


def _selected_body_speed_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Speed magnitude of selected articulation bodies in world frame.

    Returns:
        [num_envs, num_bodies]
    """

    body_lin_vel_w = _selected_body_lin_vel_w(env, asset_cfg)

    return torch.linalg.norm(body_lin_vel_w, dim=-1)


def _selected_joint_pos_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Position error of selected joints relative to their default joint positions.

    Returns:
        [num_envs, num_joints]
    """

    if asset_cfg.joint_ids is None:
        raise ValueError(
            f"SceneEntityCfg for '{asset_cfg.name}' must resolve joint_ids. "
            "Pass joint_names, for example joint_names='.*_hip_joint'."
        )

    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]

    return joint_pos - default_joint_pos
