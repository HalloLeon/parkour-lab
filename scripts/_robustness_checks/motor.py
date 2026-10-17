"""Native actuator arithmetic and readbacks; direct calls do not advance physics."""

from contextlib import contextmanager


def direct_motor_checks(task):
    import torch
    from isaaclab.utils.types import ArticulationActions

    native = task._env
    fault = native.parkour_fault
    counter = native._sim_step_counter
    records = []
    try:
        for name, motor in native.scene["robot"].actuators.items():
            strength = motor.motor_strength.clone()
            for regime, scale, speed in (
                ("unsaturated_positive", 0.2, 0.0),
                ("saturated_positive", 10.0, 0.0),
                ("saturated_negative", -10.0, 0.0),
                ("positive_overspeed", 10.0, 3.0),
                ("negative_overspeed", -10.0, -3.0),
            ):
                for after in (False, True):
                    # Reset the intervention clock/state only for this synthetic
                    # function check. Restore the clock before real collection.
                    native._sim_step_counter = counter
                    fault.reset(torch.arange(task.num_envs, device=task.device))
                    native._sim_step_counter = counter + (
                        fault.onset_step + 1 if after else 1
                    )
                    velocity = speed * motor.velocity_limit
                    feedforward = scale * motor.effort_limit / strength
                    position = torch.zeros_like(feedforward)
                    result = motor.compute(
                        ArticulationActions(
                            joint_positions=position.clone(),
                            joint_velocities=velocity.clone(),
                            joint_efforts=feedforward.clone(),
                        ),
                        joint_pos=position,
                        joint_vel=velocity,
                    )
                    request = feedforward * strength
                    clipped_speed = velocity.clamp(
                        -motor._vel_at_effort_lim, motor._vel_at_effort_lim
                    )
                    upper = (
                        motor._saturation_effort
                        * (1 - clipped_speed / motor.velocity_limit)
                    ).clamp(max=motor.effort_limit)
                    lower = (
                        motor._saturation_effort
                        * (-1 - clipped_speed / motor.velocity_limit)
                    ).clamp(min=-motor.effort_limit)
                    expected = request.clamp(lower, upper)
                    factor = torch.ones_like(expected)
                    if after and fault.config.joint_name in motor.joint_names:
                        factor[:, motor.joint_names.index(fault.config.joint_name)] = (
                            fault.config.factor
                        )
                    torch.testing.assert_close(motor.computed_effort, request)
                    torch.testing.assert_close(motor.limited_effort, expected)
                    torch.testing.assert_close(result.joint_efforts, expected * factor)
                    torch.testing.assert_close(motor.applied_effort, expected * factor)
                    assert torch.equal(motor.motor_strength, strength)
                    records.append(
                        dict(
                            actuator=name,
                            joint_names=list(motor.joint_names),
                            regime=regime,
                            after_onset=after,
                            velocity_rad_s=velocity.cpu().tolist(),
                            strength_scaled_request_Nm=request.cpu().tolist(),
                            native_limited_Nm=motor.limited_effort.cpu().tolist(),
                            delivered_Nm=motor.applied_effort.cpu().tolist(),
                            factor=factor.cpu().tolist(),
                        )
                    )
    finally:
        native._sim_step_counter = counter
        native.reset(seed=None)
    assert task.interactions["session"] == 0
    return dict(
        control="synthetic actuator inputs and clock ages; native compute, no physics integration",
        physics_interactions=0,
        clock_restored=True,
        cases=records,
    )


@contextmanager
def observe_substeps(native):
    """Copy actual computed/delivered efforts without replacing motor arithmetic."""
    import torch

    originals, records = [], []
    fault = native.parkour_fault
    for name, motor in native.scene["robot"].actuators.items():
        original = motor.compute

        def observe(*args, _motor=motor, _original=original, _name=name, **kwargs):
            output = _original(*args, **kwargs)
            expected = (
                native._sim_step_counter - fault.reset_steps - 1
            ) >= fault.onset_step
            assert torch.equal(fault.fault_active, expected)
            factor = fault.drive_factor[:, _motor._drive_fault_columns]
            torch.testing.assert_close(
                _motor.applied_effort, _motor.limited_effort * factor
            )
            records.append(
                dict(
                    physics_counter=native._sim_step_counter,
                    actuator=_name,
                    joint_names=list(_motor.joint_names),
                    reset_steps=fault.reset_steps.cpu().tolist(),
                    active=fault.fault_active.cpu().tolist(),
                    factor=factor.cpu().tolist(),
                    limited_Nm=_motor.limited_effort.detach().cpu().tolist(),
                    applied_Nm=_motor.applied_effort.detach().cpu().tolist(),
                )
            )
            return output

        motor.compute = observe
        originals.append((motor, original))
    try:
        yield records
    finally:
        for motor, original in originals:
            motor.compute = original


def physical_readback(native, reward):
    import torch

    from parkour_lab.environments.faults import JOINT_NAMES, fault_state

    robot = native.scene["robot"]
    state = fault_state(native)
    order = [robot.joint_names.index(name) for name in JOINT_NAMES]
    applied = robot.data.applied_torque[:, order]
    target = robot._joint_effort_target_sim[:, order]
    torch.testing.assert_close(applied, state["applied_effort"])
    torch.testing.assert_close(target, applied)
    torch.testing.assert_close(
        state["applied_effort"], state["limited_effort"] * state["drive_factor"]
    )
    terms = [
        dict(native.reward_manager.get_active_iterable_terms(i))
        for i in range(native.num_envs)
    ]
    weight = native.reward_manager.get_term_cfg("dof_torques_l2").weight
    effort_term = applied.square().sum(-1) * weight
    recorded = torch.tensor(
        [row["dof_torques_l2"][0] for row in terms], device=applied.device
    )
    torch.testing.assert_close(recorded, effort_term, rtol=1e-5, atol=1e-6)
    total = (
        torch.tensor(
            [sum(values[0] for values in row.values()) for row in terms],
            device=reward.device,
        )
        * native.step_dt
    )
    torch.testing.assert_close(total, reward, rtol=1e-5, atol=1e-6)
    return dict(
        joint_names=list(JOINT_NAMES),
        **{name: value.cpu().tolist() for name, value in state.items()},
        physx_effort_target_Nm=target.cpu().tolist(),
        torque_reward_without_dt=recorded.cpu().tolist(),
        sum_weighted_terms_times_dt=total.cpu().tolist(),
        reward=reward.cpu().tolist(),
    )
