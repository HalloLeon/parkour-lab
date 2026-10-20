"""Real boundary monitoring and explicitly injected teacher-rule fixtures."""

from copy import deepcopy

from .geometry import assignments


def check_boundary(task):
    """Cross a tile edge and witness one outgoing monitor sample before reset."""
    import torch

    from parkour_lab.environments.terrains import curriculum

    native, terrain = task._env, task._env.scene.terrain
    row = task.num_envs - 1
    teacher = getattr(native, "_benchmark_teacher", None)
    family = assignments(native)[row]["family"]
    expected = int(terrain.terrain_levels[row])
    target_before = expected
    if teacher is not None:
        target_before = (
            0 if family == "Flat" else teacher["families"][family]["target_level"]
        )
    origin = terrain.env_origins[row].clone()
    pose = native.scene["robot"].data.root_link_pose_w.clone()
    pose[row, :3] = origin + pose.new_tensor((6.1, 0, 2.0))
    native.scene["robot"].write_root_pose_to_sim(pose)
    if teacher is not None:
        native.command_manager.get_term("base_velocity").curriculum_eligible[
            row
        ] = False
    witness, original = {}, curriculum.observe_step

    def observe(env):
        before = teacher["episodes"]["steps"].clone()
        original(env)
        after = teacher["episodes"]["steps"]
        assert torch.equal(after, before + 1)
        witness.update(before=int(before[row]), outgoing=int(after[row]))

    if teacher is not None:
        curriculum.observe_step = observe
    counts = task.interactions
    try:
        _, _, _, truncated, info = task.step(
            torch.zeros(task.num_envs, 12, device=task.device)
        )
    finally:
        curriculum.observe_step = original
    assert task.interactions == dict(
        session=counts["session"] + task.num_envs,
        cumulative=counts["cumulative"] + task.num_envs,
        uncertain=counts["uncertain"],
    )
    assert truncated[row] and (info["final_env_ids"] == row).any()
    if teacher is not None:
        expected = (
            0 if family == "Flat" else teacher["families"][family]["target_level"]
        )
    offset = (
        native.scene["robot"].data.root_link_pos_w[row, :2]
        - terrain.env_origins[row, :2]
    )
    assert (offset.abs() < 6.0).all()
    assert int(terrain.terrain_levels[row]) == expected
    if teacher is not None:
        assert witness and int(teacher["episodes"]["steps"][row]) == 0
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
        family_target_before=target_before,
        reset_level=expected,
        reset_offset_xy_m=offset.cpu().tolist(),
        explicit_partial_reset_ids=[row],
        outgoing_monitor=witness if teacher is not None else None,
    )


def _target(native, family, level):
    """Supply a valid administrative history reaching the requested fixture target."""
    native._benchmark_teacher["families"][family] = dict(
        target_level=level,
        revision=level,
        block=[],
        completed_blocks=level,
        history=[
            dict(block=i + 1, level=i, successes=20, next_level=i + 1)
            for i in range(level)
        ],
    )


def _episode(native, row, level, *, success=True):
    """Inject summaries and native buffers; these are not measured robot motion."""
    episode = native._benchmark_teacher["episodes"]
    for value in episode.values():
        value.zero_()
    command = native.command_manager.get_term("base_velocity")
    command.command[row] = command.command.new_tensor((0.4, 0, 0))
    command.episode_command[row] = command.command[row]
    command.category[row] = 0
    command.curriculum_eligible[row] = True
    for key, value in dict(
        eligible=True,
        steps=1000,
        tracked=950,
        contact_steps=50,
        revision=level,
        level=level,
        speed=0.4,
    ).items():
        episode[key][row] = value
    episode["command"][row] = command.command[row]
    episode["direction"][row, 0] = 1
    origin = native.scene.terrain.env_origins[row, :2]
    episode["anchor"][row] = origin
    pose = native.scene["robot"].data.root_link_pose_w.clone()
    pose[row, :2] = origin + pose.new_tensor((8.0, 0))
    native.scene["robot"].write_root_pose_to_sim(pose)
    native.episode_length_buf[row] = 1000
    native.reset_terminated.zero_()
    native.reset_time_outs.zero_()
    native.reset_terminated[row] = not success
    native.reset_time_outs[row] = success


def _block(native, row, family, name, level, successes, expected, terms):
    import torch

    _target(native, family, level)
    state = native._benchmark_teacher["families"][family]
    for index in range(20):
        success = index < successes
        _episode(native, row, level, success=success)
        for values in terms.values():
            values.zero_()
        terms["time_out"][row] = success
        native.curriculum_manager.compute(
            env_ids=torch.tensor([row], device=native.device)
        )
        if index < 19:
            assert state["target_level"] == level
            assert len(state["block"]) == index + 1
    assert state["target_level"] == expected and state["block"] == []
    assert state["completed_blocks"] == level + 1
    assert state["history"][-1] == dict(
        block=level + 1, level=level, successes=successes, next_level=expected
    )
    return dict(
        name=name, before=level, after=expected, outcomes=20, successes=successes
    )


def _holds(native, rows, terms):
    import torch

    from parkour_lab.environments.terrains.curriculum import observe_step

    shaped, flat = rows[0], rows[1]
    family = assignments(native)[shaped]["family"]
    _target(native, family, 0)
    before = deepcopy(native._benchmark_teacher["families"])
    for name, row in (("flat_hold", flat), ("changed_packet", shaped)):
        _episode(native, row, 0)
        if name == "changed_packet":
            native.command_manager.get_term("base_velocity").command[row, 0] += 0.1
            observe_step(native)
            assert not native._benchmark_teacher["episodes"]["eligible"][row]
        for values in terms.values():
            values.zero_()
        terms["time_out"][row] = True
        native.curriculum_manager.compute(
            env_ids=torch.tensor([row], device=native.device)
        )
        assert native._benchmark_teacher["families"] == before
    return [
        dict(name=name, before=0, after=0) for name in ("flat_hold", "changed_packet")
    ]


def check_curriculum(task):
    """Exercise the native manager with injected summaries, then restore/reset.

    No physics is advanced. Fixture poses, flags, commands and teacher decisions
    are restored before fresh native episodes; reset sampling advances normally.
    """
    import torch

    native, terrain = task._env, task._env.scene.terrain
    command, manager = (
        native.command_manager.get_term("base_velocity"),
        native.termination_manager,
    )
    teacher = deepcopy(native._benchmark_teacher)
    original_families = deepcopy(teacher["families"])
    pose = native.scene["robot"].data.root_link_pose_w.clone()
    buffers = [
        (value, value.clone())
        for value in (
            terrain.terrain_levels,
            terrain.terrain_types,
            terrain.env_origins,
            native.scene.env_origins,
            native.episode_length_buf,
            native.reset_terminated,
            native.reset_time_outs,
            command.command,
            command.episode_command,
            command.category,
            command.curriculum_eligible,
        )
    ]
    getter = manager.get_term
    terms = {name: getter(name).clone() for name in manager.active_terms}
    rows = assignments(native)
    shaped = next(row for row in rows if row["family"] != "Flat")
    flat = next(row["env_id"] for row in rows if row["family"] == "Flat")
    counts, results = task.interactions, []
    rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state(native.device)
        if torch.device(native.device).type == "cuda"
        else None
    )
    try:
        manager.get_term = terms.__getitem__
        for name, level, successes, expected in (
            ("promote_after_full_block", 3, 16, 4),
            ("demote", 3, 8, 2),
            ("hold", 3, 12, 3),
            ("bottom", 0, 0, 0),
            ("top", 15, 20, 15),
        ):
            results.append(
                _block(
                    native,
                    shaped["env_id"],
                    shaped["family"],
                    name,
                    level,
                    successes,
                    expected,
                    terms,
                )
            )
        results.extend(_holds(native, (shaped["env_id"], flat), terms))
        assert task.interactions == counts
        assert torch.equal(torch.get_rng_state(), rng)
        if cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(native.device), cuda_rng)
    finally:
        manager.get_term = getter
        for target, value in buffers:
            target.copy_(value)
        native.scene["robot"].write_root_pose_to_sim(pose)
        native._benchmark_teacher = teacher
        native.episode_length_buf.zero_()
        native.reset_terminated.zero_()
        native.reset_time_outs.zero_()
        native.reset(seed=None)
    assert task.interactions == counts
    assert native._benchmark_teacher["families"] == original_families
    return dict(
        fixture="host-injected summaries/poses/flags; no measured locomotion",
        physics_steps=0,
        charged_interactions=0,
        restored_teacher=True,
        fresh_native_reset=True,
        cases=results,
    )
