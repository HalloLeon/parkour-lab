"""Short real physics onset/persistence/reset check with a blind fixed controller."""


class FixedPolicy:
    def __init__(self):
        self.action_value = 0.125
        self.memory = None
        self.reset_rows = 0

    def state_sha256(self):
        import hashlib

        return hashlib.sha256(repr(self.action_value).encode()).hexdigest()

    def reset(self, mask):
        import torch

        if self.memory is None:
            self.memory = torch.zeros(len(mask), device=mask.device)
        self.memory[mask] = 0
        self.reset_rows += int(mask.sum())

    def act(self, observation):
        import torch

        assert observation.shape == (len(observation), 49)
        self.memory += 1
        return torch.full(
            (len(observation), 12), self.action_value, device=observation.device
        )


def check_physics(task, app):
    import torch

    from parkour_lab.environments.faults import fault_state
    from parkour_lab.evaluation.environment import bind_policy
    from parkour_lab.runtime.motor import _runtime_motor_binding
    from scripts._robustness_checks.motor import observe_substeps, physical_readback

    native = task._env
    policy = FixedPolicy()
    fingerprint = policy.state_sha256()
    host = bind_policy(native, policy)
    binding = _runtime_motor_binding(native)
    native.reset(seed=None)
    reset = torch.ones(task.num_envs, dtype=torch.bool, device=task.device)
    command = torch.tensor([[0.35, 0.0, 0.0]] * task.num_envs, device=task.device)
    steps, repeats = [], None
    baseline = native._sim_step_counter
    with observe_substeps(native) as substeps:
        for index in range(6):
            assert app.is_running()
            targets = host.act(command, time_s=index * native.step_dt, reset_mask=reset)
            _, reward, terminated, truncated, _ = native.step(
                host.motor.encode(targets)
            )
            host.motor.verify_delivery(terminated, truncated)
            assert not (
                terminated | truncated
            ).any(), "Fixture ended before bounded onset/reset checks"
            assert native._sim_step_counter == baseline + (index + 1) * 4
            witness = physical_readback(native, reward)
            witness.update(
                control_step=index + 1,
                interactions=dict(task.interactions),
                policy_memory=policy.memory.cpu().tolist(),
            )
            steps.append(witness)
            reset = terminated | truncated
            if index == 2:
                assert native.parkour_fault.fault_active.all()
                other_before = native.parkour_fault.drive_factor[1].clone()
                native.reset(seed=None, env_ids=torch.tensor([0], device=task.device))
                reset[0] = True
                assert not native.parkour_fault.fault_active[0]
                assert native.parkour_fault.fault_active[1]
                assert torch.equal(native.parkour_fault.drive_factor[1], other_before)
                before = fault_state(native)
                count = task.interactions
                for _ in range(2):
                    native.scene.write_data_to_sim()
                after = fault_state(native)
                assert task.interactions == count
                assert torch.equal(before["fault_active"], after["fault_active"])
                assert torch.equal(before["drive_factor"], after["drive_factor"])
                repeats = dict(
                    physics_counter=native._sim_step_counter,
                    same_counter_writes=2,
                    factors=after["drive_factor"].cpu().tolist(),
                )
    assert [step["fault_active"] for step in steps] == [
        [False, False],
        [False, False],
        [True, True],
        [False, True],
        [False, True],
        [True, True],
    ]
    assert policy.reset_rows == 3 and policy.memory.tolist() == [3, 6]
    assert policy.state_sha256() == fingerprint
    assert _runtime_motor_binding(native) == binding
    assert task.interactions == dict(session=12, cumulative=12, uncertain=0)
    return dict(
        control="fixed blind untrained policy; deliberately short 0.04s fault onset; no recovery gait claim",
        policy_state_unchanged=policy.state_sha256() == fingerprint,
        policy_reset_rows=policy.reset_rows,
        no_memory_reset_at_onset=True,
        actual_vector_physics_intervals=24,
        control_steps=6,
        repeated_write_witness=repeats,
        steps=steps,
        motor_compute_calls=substeps,
        interactions=task.interactions,
    )
