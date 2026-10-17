"""Controlled outer-surface placement and actual PhysX foot-contact witness."""


def check_outer_contact(task, app, output):
    import torch

    from parkour_lab.control.proprioception import FOOT_NAMES
    from parkour_lab.provenance import write_json
    from scripts._terrain_checks.contacts import _contact_sample
    from scripts._terrain_checks.geometry import _cast, _mesh

    native, robot = task._env, task._env.scene["robot"]
    mesh, _ = _mesh(native)
    offsets = torch.tensor([[2.4, 0.0, 0.0]], device=task.device)
    support = _cast(native, mesh, offsets)[:, 0]
    pose = robot.data.root_link_pose_w.clone()
    pose[:, :3] = support
    pose[:, 2] += 0.45
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(task.num_envs, 6, device=task.device))
    feet = [robot.body_names.index(name) for name in FOOT_NAMES]
    origin, initial = native.scene.env_origins, task.interactions
    witnessed = {}
    original = native._reset_idx
    outgoing = {}

    def sample(ids):
        sample = _contact_sample(native, ids)
        position = robot.data.body_link_pos_w[ids][:, feet].clone()
        # The ray helper broadcasts per-row offsets over the N native origins.
        offsets = torch.zeros(task.num_envs, 4, 3, device=task.device)
        offsets[ids] = position - origin[ids, None]
        height = _cast(native, mesh, offsets)[ids, :, 2]
        sample.update(foot_position_w=position, support_height_w=height)
        return sample

    def before_reset(ids):
        if native._capturing:
            values = sample(ids)
            outgoing.update(
                {
                    int(row): {key: value[i].clone() for key, value in values.items()}
                    for i, row in enumerate(ids)
                }
            )
        original(ids)

    native._reset_idx = before_reset
    records = []
    result = dict(
        status="RUNNING",
        control="Controlled root placement above x=2.4m surface, nominal-pose actions; no learned traversal",
        requested_root_pose_w=pose.cpu().tolist(),
        upward_force_threshold_N=1.5,
        maximum_foot_support_distance_m=0.12,
        witnesses={},
        steps=records,
    )
    write_json(output / "outer_contact.json", result)
    try:
        for step in range(32):
            assert app.is_running()
            outgoing.clear()
            _, _, terminated, truncated, _ = task.step(
                torch.zeros(task.num_envs, 12, device=task.device)
            )
            current = sample(torch.arange(task.num_envs, device=task.device))
            rows = []
            for row in range(task.num_envs):
                values = outgoing.get(
                    row, {key: value[row] for key, value in current.items()}
                )
                force, position = values["force_w"], values["foot_position_w"]
                rows.append(
                    {key: value.cpu().tolist() for key, value in values.items()}
                )
                outside = ((position[:, :2] - origin[row, :2]).abs() > 1.3).any(-1)
                near = abs(position[:, 2] - values["support_height_w"]) < 0.12
                contact = (force[:, 2] > 1.5) & outside & near
                if row not in witnessed and contact.any():
                    witnessed[row] = dict(
                        control_step=step + 1,
                        captured_before_reset=row in outgoing,
                        **{key: value.cpu().tolist() for key, value in values.items()},
                    )
            records.append(
                dict(
                    step=step + 1,
                    terminated=terminated.cpu().tolist(),
                    truncated=truncated.cpu().tolist(),
                    rows=rows,
                )
            )
            result.update(
                witnesses={str(row): value for row, value in witnessed.items()},
                interactions=task.interactions,
            )
            write_json(output / "outer_contact.json", result)
            assert all(
                row in witnessed or not (terminated[row] | truncated[row])
                for row in range(task.num_envs)
            ), "Replica ended without an outer foot-support witness"
            if len(witnessed) == task.num_envs:
                assert task.interactions == dict(
                    session=initial["session"] + (step + 1) * task.num_envs,
                    cumulative=initial["cumulative"] + (step + 1) * task.num_envs,
                    uncertain=initial["uncertain"],
                )
                result["status"] = "CONTACT_CAPTURED"
                write_json(output / "outer_contact.json", result)
                return dict(
                    passed=True,
                    witness="outer_contact.json",
                    control_steps=step + 1,
                    interactions=task.interactions,
                )
        raise AssertionError("No outer composed foot contact within 32 control calls")
    finally:
        native._reset_idx = original
        if result["status"] == "RUNNING":
            result["status"] = "INCOMPLETE"
        result["interactions"] = task.interactions
        write_json(output / "outer_contact.json", result)
