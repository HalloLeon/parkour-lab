"""Body-frame tracking and onset-anchored path diagnostics from live samples."""

import numpy as np

from ..cases import PERIOD_S, PHASES, PROFILES


def heading(quaternion):
    """World heading is diagnostic; it is not substituted for body-Z yaw rate."""
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def tracking(trace, mask):
    command = trace["command"][mask]
    velocity = trace["root_com_linear_velocity_b"][mask, :2]
    yaw = trace["root_angular_velocity_b"][mask, 2]
    turning = command[:, 2] != 0
    return dict(
        planar_rmse_m_s=(
            float(np.sqrt(np.mean(np.sum((velocity - command[:, :2]) ** 2, axis=-1))))
            if len(yaw)
            else None
        ),
        yaw_rmse_rad_s=(
            float(np.sqrt(np.mean((yaw - command[:, 2]) ** 2))) if len(yaw) else None
        ),
        mean_signed_body_yaw_rad_s=float(yaw.mean()) if len(yaw) else None,
        wrong_direction_fraction=(
            float(np.mean(yaw[turning] * command[turning, 2] < 0))
            if turning.any()
            else None
        ),
    )


def phase_scores(trace):
    time = trace["decision_time_s"]
    result = {}
    for name, (start, end) in PHASES.items():
        mask = (time >= start) & (time < end)
        samples = int(mask.sum())
        prescribed = round((end - start) / PERIOD_S)
        result[name] = dict(
            start_s=start,
            end_s=end,
            samples=samples,
            prescribed_samples=prescribed,
            coverage=samples / prescribed,
            **tracking(trace, mask),
        )
    return result


def _anchor(trace, time_s):
    index = round(time_s / PERIOD_S) - 1
    if index < 0:
        return trace["initial_root_position_w"][:2]
    return (
        trace["root_position_w"][index, :2] if len(trace["command"]) > index else None
    )


def motion_scores(case, trace):
    time = trace["decision_time_s"]
    mask = (time >= 2) & (time < 14)
    anchor = _anchor(trace, 2)
    result = dict(
        samples=int(mask.sum()),
        **tracking(trace, mask),
        unwanted_planar_speed_m_s=None,
        maximum_onset_xy_excursion_m=None,
        body_yaw_integral_rad=None,
        world_heading_change_rad=None,
        flat_reference=None,
    )
    if not mask.any() or anchor is None:
        return result
    positions = trace["root_position_w"][mask, :2]
    angles = heading(
        np.vstack((trace["initial_root_quaternion_w"], trace["root_quaternion_w"]))
    )
    first, last = np.flatnonzero(mask)[[0, -1]]
    result.update(
        body_yaw_integral_rad=float(
            trace["root_angular_velocity_b"][mask, 2].sum() * PERIOD_S
        ),
        world_heading_change_rad=float(angles[last + 1] - angles[first]),
    )
    if case.profile.startswith("pivot"):
        result.update(
            unwanted_planar_speed_m_s=float(
                np.linalg.norm(
                    trace["root_com_linear_velocity_b"][mask, :2], axis=-1
                ).mean()
            ),
            maximum_onset_xy_excursion_m=float(
                np.linalg.norm(positions - anchor, axis=-1).max()
            ),
        )
    if case.family == "Flat":
        velocity, _, yaw = PROFILES[case.profile]
        elapsed, angle = time[mask] + PERIOD_S - 2, angles[first]
        reference = np.tile(anchor, (len(elapsed), 1))
        if yaw:
            reference += (
                velocity
                / yaw
                * np.column_stack(
                    (
                        np.sin(angle + yaw * elapsed) - np.sin(angle),
                        np.cos(angle) - np.cos(angle + yaw * elapsed),
                    )
                )
            )
        else:
            reference += (
                velocity * elapsed[:, None] * np.array([np.cos(angle), np.sin(angle)])
            )
        error = np.linalg.norm(positions - reference, axis=-1)
        result["flat_reference"] = dict(
            onset_position_xy_m=anchor.tolist(),
            onset_heading_rad=float(angle),
            path_rmse_m=float(np.sqrt(np.mean(error**2))),
            maximum_path_error_m=float(error.max()),
            requested_motion_duration_s=12.0,
            requested_heading_change_rad=yaw * 12,
            requested_forward_travel_m=velocity * 12,
            requested_radius_m=abs(velocity / yaw) if velocity and yaw else None,
        )
    return result


def stopping_scores(trace):
    time = trace["decision_time_s"]
    stop, hold = time >= 14, time >= 15
    anchor = _anchor(trace, 14)
    speed = np.linalg.norm(trace["root_com_linear_velocity_b"][hold, :2], axis=-1)
    yaw = np.abs(trace["root_angular_velocity_b"][hold, 2])
    return dict(
        hold_samples=int(hold.sum()),
        residual_planar_speed_m_s=float(speed.mean()) if len(speed) else None,
        residual_absolute_yaw_rad_s=float(yaw.mean()) if len(yaw) else None,
        maximum_drift_from_stop_onset_m=(
            float(
                np.linalg.norm(
                    trace["root_position_w"][stop, :2] - anchor, axis=-1
                ).max()
            )
            if stop.any() and anchor is not None
            else None
        ),
    )
