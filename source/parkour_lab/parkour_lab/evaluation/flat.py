"""Fixed flat-command diagnostics and first-attempt kinematics, independent of RL.

These tapes use the required command profiles, but do not implement the independently
seeded starts, noise and dynamics strata of the acceptance bank. A diagnostic pass
cannot qualify a policy. Positions/headings include the pre-command onset sample;
velocities and termination flags are post-physics, before automatic reset.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np

from parkour_lab.control.command_tape import PERIOD_S

PROFILES = {
    "stand": (0.0, 0.0, 0.0),
    "forward-02": (0.2, 0.0, 0.0),
    "forward-05": (0.5, 0.0, 0.0),
    "reverse-02": (-0.2, 0.0, 0.0),
    "left-02": (0.0, 0.2, 0.0),
    "right-02": (0.0, -0.2, 0.0),
    "pivot-left": (0.0, 0.0, 0.5),
    "pivot-right": (0.0, 0.0, -0.5),
    "arc-left": (0.35, 0.0, 0.3),
    "arc-right": (0.35, 0.0, -0.3),
}
STEPS = 550


@dataclass(frozen=True)
class Thresholds:
    settling_s: float = 1.0
    block_s: float = 0.4
    planar_error_m_s: float = 0.10
    pivot_yaw_error_rad_s: float = 0.10
    moving_yaw_error_rad_s: float = 0.15
    wrong_sign_fraction: float = 0.05
    stationary_speed_m_s: float = 0.08
    stationary_onset_excursion_m: float = 0.15
    stationary_drift_2s_m: float = 0.10
    zero_twist_heading_excursion_rad: float = 0.15
    flat_attitude_rad: float = math.radians(15)


def profile_commands(profile):
    """Eleven seconds: stop 2 s, named command 6 s, stop 3 s."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown flat profile: {profile}")
    return [(0.0, 0.0, 0.0)] * 100 + [PROFILES[profile]] * 300 + [(0.0, 0.0, 0.0)] * 150


def score_command_phase(command, velocity, body_yaw, world_yaw, position, heading):
    """Retain the reviewed acquisition/block/drift arithmetic at 50 Hz."""
    th, dt = Thresholds(), PERIOD_S
    cmd = np.asarray(command)
    length = len(velocity)
    block, settling = round(th.block_s / dt), round(th.settling_s / dt)
    if (
        length < settling
        or velocity.shape != (length, 2)
        or body_yaw.shape != (length,)
        or world_yaw.shape != (length,)
        or position.shape != (length + 1, 2)
        or heading.shape != (length + 1,)
        or cmd.shape != (3,)
        or not all(
            np.isfinite(a).all()
            for a in (cmd, velocity, body_yaw, world_yaw, position, heading)
        )
    ):
        raise ValueError(
            "Command scoring requires a complete finite phase and onset pose"
        )
    planar_error = np.linalg.norm(velocity - cmd[:2], axis=-1)
    yaw_error = np.maximum(np.abs(world_yaw - cmd[2]), np.abs(body_yaw - cmd[2]))
    stationary = np.linalg.norm(cmd[:2]) == 0
    planar_limit = th.stationary_speed_m_s if stationary else th.planar_error_m_s
    yaw_limit = th.pivot_yaw_error_rad_s if stationary else th.moving_yaw_error_rad_s
    blocks = [
        {
            "start_s": offset * dt,
            "duration_s": (min(offset + block, length) - offset) * dt,
            "planar_error_m_s": float(planar_error[offset : offset + block].mean()),
            "yaw_error_rad_s": float(yaw_error[offset : offset + block].mean()),
        }
        for offset in range(settling - block, length, block)
    ]
    failed = [
        i
        for i, row in enumerate(blocks)
        if row["planar_error_m_s"] > planar_limit or row["yaw_error_rad_s"] > yaw_limit
    ]
    failures = ["tracking not sustained after 1 s"] if failed else []
    result = dict(
        command=cmd.tolist(),
        blocks=blocks,
        failed_tracking_blocks=failed,
        acquisition_failed=0 in failed,
        later_tracking_failed=any(i > 0 for i in failed),
    )
    if cmd[2] != 0:
        wrong_sign = (
            max(
                float(np.mean(rate[settling:] * cmd[2] < 0))
                for rate in (world_yaw, body_yaw)
            )
            if length > settling
            else None
        )
        result["wrong_sign_fraction"] = wrong_sign
        if wrong_sign is None or wrong_sign > th.wrong_sign_fraction:
            failures.append("missing or excessive wrong-sign yaw")
    if stationary:
        excursion = float(np.linalg.norm(position[1:] - position[0], axis=-1).max())
        drift = (
            max(
                float(
                    np.linalg.norm(
                        position[offset : offset + round(2 / dt) + 1]
                        - position[offset],
                        axis=-1,
                    ).max()
                )
                for offset in range(settling + 1, len(position))
            )
            if length > settling
            else None
        )
        result.update(onset_excursion_m=excursion, maximum_settled_drift_2s_m=drift)
        if excursion > th.stationary_onset_excursion_m or (
            drift is not None and drift > th.stationary_drift_2s_m
        ):
            failures.append("excessive stationary drift/braking distance")
        if cmd[2] == 0:
            excursion = float(np.abs(heading[1:] - heading[0]).max())
            result["heading_onset_excursion_rad"] = excursion
            if excursion > th.zero_twist_heading_excursion_rad:
                failures.append("excessive zero-twist heading drift")
    result.update(failures=failures, kinematic_passed=not failures)
    return result


def _angles(quaternion):
    w, x, y, z = quaternion.T
    return (
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
        np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))),
    )


def score_flat_profile(trace, profile):
    """Score each original row once; all later auto-reset episodes are excluded.

    Missing/malformed arrays reject the evidence. Nonfinite first-attempt state
    fails that row. No per-group acceptance rates or dynamics strata are inferred
    from this diagnostic's existing startup distribution.
    """
    schedule = np.asarray(profile_commands(profile))
    origins = np.asarray(trace["env_origins"])
    if (
        origins.ndim != 2
        or origins.shape[1] != 3
        or not len(origins)
        or not np.isfinite(origins).all()
    ):
        raise ValueError("Require finite environment origins")
    count = len(origins)
    shapes = {
        key: (STEPS, count, width)
        for key, width in {
            "command": 3,
            "position_w": 3,
            "quaternion_w": 4,
            "linear_velocity_b": 3,
            "angular_velocity_b": 3,
            "angular_velocity_w": 3,
            "base_contact_force_w": 3,
        }.items()
    }
    shapes.update(
        initial_position_w=(count, 3),
        initial_quaternion_w=(count, 4),
        terminated=(STEPS, count),
        truncated=(STEPS, count),
    )
    for key, shape in shapes.items():
        if key not in trace or trace[key].shape != shape:
            raise ValueError(f"Incomplete motion trace: {key} must have shape {shape}")
    body = trace.get("body_position_w")
    if (
        body is None
        or body.ndim != 4
        or body.shape[:2] != (STEPS, count)
        or not body.shape[2]
        or body.shape[3] != 3
    ):
        raise ValueError("Require all native body-link positions")
    if any(trace[key].dtype != np.bool_ for key in ("terminated", "truncated")):
        raise ValueError("Termination flags must be boolean")
    if not np.allclose(trace["command"], schedule[:, None], rtol=0, atol=1e-6):
        raise ValueError("Executed commands differ from the flat profile")
    # Merge adjacent identical commands, particularly the single 11 s stand.
    boundaries = [
        0,
        *(np.flatnonzero(np.any(np.diff(schedule, axis=0), axis=1)) + 1),
        STEPS,
    ]
    results = []
    for row in range(count):
        ends = np.flatnonzero(trace["terminated"][:, row] | trace["truncated"][:, row])
        length = int(ends[0] + 1) if len(ends) else STEPS
        failures, phases = [], []
        if trace["terminated"][:length, row].any():
            failures.append("native termination")
        if trace["truncated"][:length, row].any():
            failures.append("unexpected timeout")
        position = np.vstack(
            (trace["initial_position_w"][row], trace["position_w"][:length, row])
        ).astype(np.float64)
        quaternion = np.vstack(
            (trace["initial_quaternion_w"][row], trace["quaternion_w"][:length, row])
        ).astype(np.float64)
        finite = all(
            np.isfinite(trace[key][:length, row]).all()
            for key in shapes
            if key not in ("initial_position_w", "initial_quaternion_w")
        )
        finite = (
            finite
            and np.isfinite(position).all()
            and np.isfinite(quaternion).all()
            and np.isfinite(body[:length, row]).all()
        )
        if not finite:
            failures.append("nonfinite first-attempt state")
        elif not np.allclose(
            np.linalg.norm(quaternion, axis=-1), 1.0, atol=1e-3, rtol=0
        ):
            failures.append("invalid quaternion")
        else:
            roll, pitch, heading = _angles(quaternion)
            if (
                max(np.abs(roll).max(), np.abs(pitch).max())
                > Thresholds().flat_attitude_rad
            ):
                failures.append("flat attitude exceeds 15 degrees")
            if (np.abs(position[:, :2] - origins[row, :2]) > 6).any() or (
                np.abs(body[:length, row, :, :2] - origins[row, :2]) > 6
            ).any():
                failures.append("body link leaves flat domain")
            forces = np.linalg.norm(
                trace["base_contact_force_w"][:length, row], axis=-1
            )
            if (
                length >= 5
                and (
                    np.convolve(
                        (forces > 20).astype(int), np.ones(5, dtype=int), mode="valid"
                    )
                    >= 5
                ).any()
            ):
                failures.append("sustained base contact")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end > length:
                    phases.append(dict(start_s=start * PERIOD_S, complete=False))
                    failures.append("incomplete command phase")
                    continue
                command = schedule[start]
                phase = score_command_phase(
                    command,
                    trace["linear_velocity_b"][start:end, row, :2],
                    trace["angular_velocity_b"][start:end, row, 2],
                    trace["angular_velocity_w"][start:end, row, 2],
                    position[start : end + 1, :2],
                    heading[start : end + 1],
                )
                # Full 3 s displacement windows after 1 s; never along a fixed axis.
                stalled = np.linalg.norm(command[:2]) >= 0.2 and any(
                    np.linalg.norm(position[k + 150, :2] - position[k, :2]) < 0.10
                    for k in range(start + 50, end - 150 + 1)
                )
                if stalled:
                    phase["failures"].append("stall")
                    phase["kinematic_passed"] = False
                phase.update(start_s=start * PERIOD_S, complete=True)
                failures.extend(
                    f"phase {start * PERIOD_S:g}s: {reason}"
                    for reason in phase["failures"]
                )
                phases.append(phase)
        results.append(
            dict(
                env_id=row,
                first_attempt_steps=length,
                diagnostic_passed=not failures,
                failures=failures,
                phases=phases,
            )
        )
    return dict(
        version="flat_command_diagnostic_v1",
        group_id=f"flat/{profile}",
        namespace="parkour-lab/flat-diagnostic/v1",
        qualified=False,
        qualification_eligible=False,
        thresholds=asdict(Thresholds()),
        scope="First-attempt kinematics under the recorded task distribution, not the independent noisy nominal/randomized acceptance bank. Map checks use root/body-link centres, not collider extents.",
        passed=sum(row["diagnostic_passed"] for row in results),
        total=count,
        trials=results,
    )
