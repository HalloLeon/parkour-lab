"""Command-gated fine tracking for operator training.

These are training rewards, not inference-time feedback. They use the same
body-frame velocities and command. Velocity may be privileged for a causal
actor; this reward does not add it to the actor's observations. They contain no
pose anchor, timer, phase label, action override or persistent state. In
particular, zero velocity is NOT absolute position/heading restoration.
"""

import math

import torch


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
):
    """Blend fine planar tracking only when commanded planar velocity is zero.

    Both yaw signs use identical gating. Nonzero planar commands, even small
    ones, retain the exact broad reward. With full_stop_only, pure pivots also
    retain the broad reward. Training command modes emit exact zeros.
    """
    command = env.command_manager.get_command(command_name)
    velocity = env.scene["robot"].data.root_lin_vel_b[:, :2]
    error_squared = torch.sum(torch.square(command[:, :2] - velocity), dim=1)
    broad, fine = _kernels(error_squared, std, stationary_std, precision_fraction)
    stopped = torch.all((command if full_stop_only else command[:, :2]) == 0, dim=1)
    return torch.where(stopped, fine, broad)


def track_ang_vel_z_stopped(
    env, command_name: str, std: float, stationary_std: float, precision_fraction: float
):
    """Blend fine yaw stopping only for a zero-twist command, never a pivot/arc."""
    command = env.command_manager.get_command(command_name)
    yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2]
    error_squared = torch.square(command[:, 2] - yaw_rate)
    broad, fine = _kernels(error_squared, std, stationary_std, precision_fraction)
    return torch.where(torch.all(command == 0, dim=1), fine, broad)
