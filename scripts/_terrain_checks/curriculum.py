"""Controlled native distance-progression fixtures; no learned policy."""

from .geometry import assignments


def check_boundary(task):
    """Cross a tile edge with a host-written pose and retain the outgoing witness."""
    import torch

    native, terrain = task._env, task._env.scene.terrain
    row = task.num_envs - 1
    level = int(terrain.terrain_levels[row])
    origin = terrain.env_origins[row].clone()
    pose = native.scene["robot"].data.root_link_pose_w.clone()
    pose[row, :3] = origin + pose.new_tensor((6.1, 0, 2.0))
    native.scene["robot"].write_root_pose_to_sim(pose)
    if task.training:
        native.command_manager.get_term("base_velocity").curriculum_eligible[
            row
        ] = False
    _, _, _, truncated, info = task.step(
        torch.zeros(task.num_envs, 12, device=task.device)
    )
    assert truncated[row] and (info["final_env_ids"] == row).any()
    offset = (
        native.scene["robot"].data.root_link_pos_w[row, :2]
        - terrain.env_origins[row, :2]
    )
    assert (offset.abs() < 6.0).all()
    assert int(terrain.terrain_levels[row]) == level
    levels, types = terrain.terrain_levels.clone(), terrain.terrain_types.clone()
    counts, survivors = task.interactions, native.episode_length_buf[:row].clone()
    task.reset(env_ids=[row])
    assert task.interactions == counts
    assert torch.equal(terrain.terrain_levels, levels)
    assert torch.equal(terrain.terrain_types, types)
    assert torch.equal(native.episode_length_buf[:row], survivors)
    offset = (
        native.scene["robot"].data.root_link_pos_w[row, :2]
        - terrain.env_origins[row, :2]
    )
    assert native.episode_length_buf[row] == 0 and (offset.abs() < 6.0).all()
    return dict(
        env_id=row,
        requested_offset_xy_m=[6.1, 0.0],
        truncated=True,
        reset_offset_xy_m=offset.cpu().tolist(),
        explicit_partial_reset_ids=[row],
    )


def check_curriculum(task):
    import torch

    from parkour_lab.environments.commands.episode import CATEGORIES

    native, terrain = task._env, task._env.scene.terrain
    command = native.command_manager.get_term("base_velocity")
    rows = [row["env_id"] for row in assignments(native) if row["family"] != "Flat"][
        :12
    ] + [0]
    cases = (
        ("promote", 3, 6.1, (0.4, 0, 0), True, 4),
        ("demote", 3, 0, (0.4, 0, 0), True, 2),
        ("translation_hold", 3, 4.5, (0.4, 0, 0), True, 3),
        ("stand", 3, 6.1, (0, 0, 0), False, 3),
        ("pivot_left", 3, 6.1, (0, 0, 0.5), False, 3),
        ("pivot_right", 3, 6.1, (0, 0, -0.5), False, 3),
        ("arc_left", 3, 6.1, (0.35, 0, 0.3), False, 3),
        ("arc_right", 3, 6.1, (0.35, 0, -0.3), False, 3),
        ("bottom", 0, 0, (0.4, 0, 0), True, 0),
        ("top", 7, 6.1, (0.4, 0, 0), True, 7),
        ("changed_packet", 3, 6.1, (0.4, 0, 0), True, 3),
        ("no_episode", 3, 6.1, (0.4, 0, 0), True, 3),
        ("flat_hold", 0, 6.1, (0.4, 0, 0), True, 0),
    )
    types = terrain.terrain_types.clone()
    pose = native.scene["robot"].data.root_link_pose_w.clone()
    for row, (name, level, distance, packet, eligible, _) in zip(rows, cases):
        terrain.terrain_levels[row] = level
        terrain.env_origins[row] = terrain.terrain_origins[level, types[row]]
        pose[row, :2] = terrain.env_origins[row, :2] + pose.new_tensor((distance, 0))
        command.vel_command_b[row] = pose.new_tensor(packet)
        command.episode_command[row] = pose.new_tensor(packet)
        command.curriculum_eligible[row] = eligible
        command.category[row] = CATEGORIES.index(name) if name in CATEGORIES else 0
        native.episode_length_buf[row] = 0 if name == "no_episode" else 500
    command.vel_command_b[rows[10], 0] += 0.1
    native.scene["robot"].write_root_pose_to_sim(pose)
    before = terrain.terrain_levels.clone()
    counts = task.interactions
    cpu_rng = torch.get_rng_state()
    gpu_rng = (
        torch.cuda.get_rng_state(task.device)
        if torch.device(task.device).type == "cuda"
        else None
    )
    native.curriculum_manager.compute(env_ids=torch.tensor(rows, device=task.device))
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if gpu_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(task.device), gpu_rng)
    expected = before.clone()
    for row, case in zip(rows, cases):
        expected[row] = case[-1]
    assert torch.equal(terrain.terrain_levels, expected)
    assert torch.equal(terrain.terrain_types, types)
    torch.testing.assert_close(
        terrain.env_origins, terrain.terrain_origins[expected, types]
    )
    native.episode_length_buf.zero_()
    task.reset()
    assert torch.equal(terrain.terrain_levels, expected)
    assert task.interactions == counts
    return [
        dict(
            name=case[0],
            env_id=row,
            category=CATEGORIES.index(case[0]) if case[0] in CATEGORIES else 0,
            before=int(before[row]),
            after=int(expected[row]),
        )
        for row, case in zip(rows, cases)
    ]
