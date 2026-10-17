"""Analytic post-physics traces with exact body-twist integration, no simulator."""

import numpy as np

from parkour_lab.evaluation.cases import PERIOD_S, STEPS


def analytic_trace(case, *, steps=STEPS, heading_rad=0.0):
    """Integrate each prescribed twist from an arbitrary measured onset pose."""
    time = np.arange(steps) * PERIOD_S
    command = np.asarray([case.command_at(float(t)) for t in time]).reshape(-1, 3)
    position = np.zeros((steps, 3))
    quaternion = np.zeros((steps, 4))
    current, angle = np.array([2.0, -3.0, 0.35]), heading_rad
    for index, (forward, lateral, yaw) in enumerate(command):
        next_angle = angle + yaw * PERIOD_S
        if yaw:
            current[:2] += (
                np.array(
                    [
                        forward * (np.sin(next_angle) - np.sin(angle))
                        + lateral * (np.cos(next_angle) - np.cos(angle)),
                        forward * (np.cos(angle) - np.cos(next_angle))
                        + lateral * (np.sin(next_angle) - np.sin(angle)),
                    ]
                )
                / yaw
            )
        else:
            current[:2] += PERIOD_S * np.array(
                [
                    forward * np.cos(angle) - lateral * np.sin(angle),
                    forward * np.sin(angle) + lateral * np.cos(angle),
                ]
            )
        angle = next_angle
        position[index] = current
        quaternion[index] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    feet = np.zeros((steps, 4, 3))
    feet[..., :2] = position[:, None, :2] + np.array(
        [[0.2, 0.15], [0.2, -0.15], [-0.2, 0.15], [-0.2, -0.15]]
    )
    feet[..., 2] = 0.02
    forces = np.zeros_like(feet)
    forces[..., 2] = 35
    initial_feet = (
        np.column_stack((feet[0, :, :2], np.full(4, 0.02)))
        if steps
        else np.array(
            [
                [2.2, -2.85, 0.02],
                [2.2, -3.15, 0.02],
                [1.8, -2.85, 0.02],
                [1.8, -3.15, 0.02],
            ]
        )
    )
    trace = dict(
        decision_time_s=time,
        command=command,
        root_position_w=position,
        root_quaternion_w=quaternion,
        root_com_linear_velocity_b=np.column_stack((command[:, :2], np.zeros(steps))),
        root_angular_velocity_b=np.column_stack((np.zeros((steps, 2)), command[:, 2])),
        reward=np.ones(steps),
        terminated=np.zeros(steps, dtype=bool),
        truncated=np.zeros(steps, dtype=bool),
        foot_position_w=feet,
        foot_contact_forces_w=forces,
        support_height_w=np.zeros((steps, 4)),
        support_valid=np.ones((steps, 4), dtype=bool),
        initial_root_position_w=np.array([2.0, -3.0, 0.35]),
        initial_root_quaternion_w=np.array(
            [np.cos(heading_rad / 2), 0, 0, np.sin(heading_rad / 2)]
        ),
        initial_foot_position_w=initial_feet,
        initial_support_height_w=np.zeros(4),
        initial_support_valid=np.ones(4, dtype=bool),
        env_origin_w=np.array([2.0, -3.0, 0]),
    )
    if case.fault is not None:
        from parkour_lab.evaluation.robustness import JOINTS

        active = time + PERIOD_S - 0.005 >= case.fault.onset_s - 1e-9
        factors = np.ones((steps, 12))
        factors[active, JOINTS.index(case.fault.joint_name)] = case.fault.factor
        trace.update(
            fault_active=active,
            drive_factor=factors,
            limited_effort=np.full((steps, 12), 10.0),
            applied_effort=10.0 * factors,
        )
    return trace
