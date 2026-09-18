"""Command-gated tracking and soft stance objectives for operator training.

These rewards use native commands and robot state, not inference-time feedback.
Privileged reward inputs do not enter the causal actor's observations. There
is no world-position/heading anchor, timer, phase label, action override or
persistent state. A soft joint-posture prior is not a rigid stance, and zero
velocity is NOT absolute position/heading restoration.
"""

import math

import torch


def joint_posture_stopped(env, command_name: str):
    """Soft default-pose cost for exact zero twist, not a joint-target override.

    Use the L2 norm (not its square) in native radians. Moving commands,
    including pure pivots and arbitrarily small commands, pay no new cost.
    No speed gate lets the policy escape the cost by moving during a stop.
    The caller supplies the negative weight; RewardManager integrates dt.
    """
    command = env.command_manager.get_command(command_name)
    data = env.scene["robot"].data
    deviation = torch.linalg.vector_norm(
        data.joint_pos - data.default_joint_pos, dim=-1
    )
    return torch.where(torch.all(command == 0, dim=-1), deviation, 0.0)


def teacher_physical_failure(
    env,
    termination_names=(
        "base_contact",
        "course_chassis",
        "course_fall",
        "course_off_route",
        "persistent_tilt",
    ),
):
    """One training impulse per physical failure, even if several gates fire."""
    failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in termination_names:
        failure |= env.termination_manager.get_term(name)
    return failure.float() / env.step_dt


def _kernels(error_squared, std, stationary_std, precision_fraction):
    if not (
        math.isfinite(std)
        and math.isfinite(stationary_std)
        and 0 < stationary_std < std
        and math.isfinite(precision_fraction)
        and 0 < precision_fraction < 1
    ):
        raise ValueError("Use 0 < stationary_std < std and 0 < precision_fraction < 1")
    broad = torch.exp(-error_squared / std**2)
    fine = torch.exp(-error_squared / stationary_std**2)
    # Keep the original peak and a broad acquisition component. Do not multiply
    # the two kernels: that would suppress all large-error learning signal.
    return broad, (1 - precision_fraction) * broad + precision_fraction * fine


def track_lin_vel_xy_stationary(
    env,
    command_name: str,
    std: float,
    stationary_std: float,
    precision_fraction: float,
    full_stop_only: bool = False,
    pivot_only: bool = False,
):
    """Blend fine planar tracking only when commanded planar velocity is zero.

    Both yaw signs use identical gating. Nonzero planar commands, even small
    ones, retain the exact broad reward. With full_stop_only, pure pivots also
    retain the broad reward. With pivot_only, exact zero twist retains it
    instead. These restrictions are mutually exclusive; defaults preserve
    historical recipes. Training command modes emit exact zeros.
    """
    if full_stop_only and pivot_only:
        raise ValueError("full_stop_only and pivot_only are mutually exclusive")
    command = env.command_manager.get_command(command_name)
    velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]
    error_squared = torch.sum(torch.square(command[:, :2] - velocity), dim=1)
    broad, fine = _kernels(error_squared, std, stationary_std, precision_fraction)
    stopped = torch.all((command if full_stop_only else command[:, :2]) == 0, dim=1)
    if pivot_only:
        stopped &= command[:, 2] != 0
    return torch.where(stopped, fine, broad)


def track_ang_vel_z_stopped(
    env,
    command_name: str,
    std: float,
    stationary_std: float,
    precision_fraction: float,
    include_pivots: bool = False,
    full_stop_std: float | None = None,
):
    """Blend fine yaw tracking at zero twist, optionally including pure pivots.

    The default preserves historical stop-only recipes. With include_pivots,
    both yaw signs track their commanded rate, not zero. Every nonzero planar
    command retains the exact broad reward, including tiny commands and arcs.
    full_stop_std optionally changes only the fine width at exact zero twist;
    the mixture fraction and pure-pivot kernel stay unchanged.
    This stateless rate objective is not absolute heading restoration.
    """
    command = env.command_manager.get_command(command_name)
    yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2]
    error_squared = torch.square(command[:, 2] - yaw_rate)
    broad, fine = _kernels(error_squared, std, stationary_std, precision_fraction)
    if full_stop_std is not None:
        _, stop_fine = _kernels(error_squared, std, full_stop_std, precision_fraction)
        fine = torch.where(torch.all(command == 0, dim=1), stop_fine, fine)
    stationary = torch.all((command[:, :2] if include_pivots else command) == 0, dim=1)
    return torch.where(stationary, fine, broad)
