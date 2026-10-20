"""Shared family targets from complete, measured translation-episode blocks."""

from copy import deepcopy

import torch

VERSION = "go2_family_block_curriculum_v2"
BLOCK_SIZE = 20
PROMOTE_SUCCESSES = 16
DEMOTE_SUCCESSES = 8


def _terrain(env):
    terrain = getattr(env.scene, "terrain", None)
    if getattr(terrain, "terrain_origins", None) is None:
        return None
    definition = getattr(env.cfg, "benchmark_terrain", None)
    if not isinstance(definition, dict):
        raise ValueError("Terrain progression requires its benchmark definition")
    origins, levels, types = (
        terrain.terrain_origins,
        terrain.terrain_levels,
        terrain.terrain_types,
    )
    if (
        origins.shape
        != (
            definition.get("num_rows", definition["level_count"]),
            len(definition["columns"]),
            3,
        )
        or not torch.isfinite(origins).all()
        or levels.shape != (env.num_envs,)
        or types.shape != levels.shape
        or levels.dtype != torch.long
        or types.dtype != torch.long
        or ((levels < 0) | (levels >= origins.shape[0])).any()
        or ((types < 0) | (types >= len(definition["columns"]))).any()
    ):
        raise ValueError("Invalid realized terrain assignments or origin table")
    return terrain


def _families(env):
    terrain = env.scene.terrain
    columns = env.cfg.benchmark_terrain["columns"]
    return [columns[index]["family"] for index in terrain.terrain_types.tolist()]


def _new_family():
    return dict(target_level=0, revision=0, block=[], completed_blocks=0, history=[])


def _new_episodes(env):
    count, device = env.num_envs, env.device
    reference = env.scene["robot"].data.root_pos_w

    def scalar(dtype):
        return torch.zeros(count, dtype=dtype, device=device)

    return dict(
        eligible=scalar(torch.bool),
        steps=scalar(torch.long),
        tracked=scalar(torch.long),
        planar_squared=scalar(torch.float64),
        yaw_squared=scalar(torch.float64),
        contact_steps=scalar(torch.long),
        revision=scalar(torch.long),
        level=scalar(torch.long),
        speed=scalar(torch.float64),
        baseline_set=scalar(torch.bool),
        command=reference.new_zeros(count, 3),
        anchor=reference.new_zeros(count, 2),
        direction=reference.new_zeros(count, 2),
        support=reference.new_zeros(count, 4),
        support_valid=torch.zeros(count, 4, dtype=torch.bool, device=device),
    )


def initialize_levels(env):
    """All families start on their true-plane entry row; draw no learner RNG."""
    if not env.cfg.benchmark_training or _terrain(env) is None:
        return
    families = {
        name: _new_family() for name in dict.fromkeys(_families(env)) if name != "Flat"
    }
    env._benchmark_teacher = dict(families=families, episodes=_new_episodes(env))
    terrain = env.scene.terrain
    terrain.terrain_levels.zero_()
    terrain.env_origins.copy_(
        terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
    )


@torch.no_grad()
def begin_episodes(env, env_ids):
    """Anchor fresh physical episodes after native reset and command resampling."""
    teacher = getattr(env, "_benchmark_teacher", None)
    if teacher is None:
        return
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if not len(ids):
        return
    episode = teacher["episodes"]
    for value in episode.values():
        value[ids] = 0
    command = env.command_manager.get_term("base_velocity")
    robot = env.scene["robot"].data
    episode["anchor"][ids] = robot.root_pos_w[ids, :2]
    episode["command"][ids] = command.command[ids]
    speed = command.command[ids, :2].norm(dim=-1)
    episode["speed"][ids] = speed.to(torch.float64)
    quaternion = robot.root_quat_w[ids]
    w, x, y, z = quaternion.unbind(-1)
    heading = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    forward = command.command[ids, :2] / speed.clamp_min(1e-12)[:, None]
    episode["direction"][ids] = torch.stack(
        (
            forward[:, 0] * heading.cos() - forward[:, 1] * heading.sin(),
            forward[:, 0] * heading.sin() + forward[:, 1] * heading.cos(),
        ),
        -1,
    )
    episode["level"][ids] = env.scene.terrain.terrain_levels[ids]
    families = _families(env)
    for row in ids.tolist():
        family = families[row]
        if family != "Flat":
            episode["revision"][row] = teacher["families"][family]["revision"]
            episode["eligible"][row] = bool(
                command.category[row] == 0
                and command.curriculum_eligible[row]
                and episode["speed"][row] > 0
            )


def _contact_samples(env, ids, episode):
    """Contact evidence is host-only and reads the actual collider without RNG."""
    from parkour_lab.control.proprioception import FOOT_NAMES
    from parkour_lab.runtime.native import foot_contact_forces, foot_support

    robot = env.scene["robot"]
    feet = robot.data.body_pos_w[ids][
        :, [robot.body_names.index(name) for name in FOOT_NAMES]
    ]
    support, valid = foot_support(env, feet)
    fresh = ~episode["baseline_set"][ids]
    episode["support"][ids[fresh]] = support[fresh]
    episode["support_valid"][ids[fresh]] = valid[fresh]
    episode["baseline_set"][ids] = True
    forces = foot_contact_forces(env)[ids]
    supported = (
        (forces.norm(dim=-1) > 1.5) & valid & ((feet[..., 2] - support).abs() <= 0.08)
    )
    changed = (
        supported
        & episode["support_valid"][ids]
        & ((support - episode["support"][ids]).abs() > 0.0005)
    )
    high = torch.where(supported, support, -torch.inf).max(-1).values
    low = torch.where(supported, support, torch.inf).min(-1).values
    exposed = changed.any(-1) | ((supported.sum(-1) > 1) & (high - low > 0.0005))
    outside = (feet[..., :2] - env.scene.terrain.env_origins[ids, None, :2]).abs().max(
        -1
    ).values > 1.3
    families = _families(env)
    shaped = torch.tensor(
        [families[row] in ("Slopes", "Stairs") for row in ids.tolist()],
        device=env.device,
    )
    return torch.where(shaped, (supported & outside).any(-1), exposed)


@torch.no_grad()
def observe_step(env):
    """Read each outgoing step once, before native automatic reset can erase it."""
    teacher = getattr(env, "_benchmark_teacher", None)
    if teacher is None:
        return
    episode = teacher["episodes"]
    command = env.command_manager.get_term("base_velocity")
    episode["eligible"] &= (command.command == episode["command"]).all(
        -1
    ) & command.curriculum_eligible
    episode["steps"] += 1
    ids = (episode["eligible"] & (episode["steps"] > 50)).nonzero().flatten()
    data = env.scene["robot"].data
    episode["tracked"][ids] += 1
    episode["planar_squared"][ids] += (
        (data.root_com_lin_vel_b[ids, :2] - episode["command"][ids, :2])
        .square()
        .sum(-1)
    )
    episode["yaw_squared"][ids] += (
        data.root_ang_vel_b[ids, 2] - episode["command"][ids, 2]
    ).square()
    contact_ids = (episode["eligible"] & (episode["level"] > 0)).nonzero().flatten()
    if len(contact_ids):
        episode["contact_steps"][contact_ids] += _contact_samples(
            env, contact_ids, episode
        ).long()


def _success(env, row, episode):
    steps, tracked = int(episode["steps"][row]), int(episode["tracked"][row])
    manager = env.termination_manager
    terms = {name: bool(manager.get_term(name)[row]) for name in manager.active_terms}
    horizon = (
        terms.get("time_out", False)
        and steps >= round(env.max_episode_length_s / env.step_dt) - 1
    )
    boundary = terms.get("terrain_boundary", False) and steps * env.step_dt >= 8.0
    complete = (horizon or boundary) and not bool(env.reset_terminated[row])
    if not complete or tracked == 0 or steps != int(env.episode_length_buf[row]):
        return False
    planar = (episode["planar_squared"][row] / tracked).sqrt()
    yaw = (episode["yaw_squared"][row] / tracked).sqrt()
    progress = (
        (env.scene["robot"].data.root_pos_w[row, :2] - episode["anchor"][row])
        * episode["direction"][row]
    ).sum()
    requested = episode["speed"][row] * steps * env.step_dt
    exposed = (
        episode["level"][row] == 0 or episode["contact_steps"][row] * env.step_dt >= 1.0
    )
    return bool(
        planar <= 0.3 * episode["speed"][row]
        and yaw <= 0.15
        and progress >= 0.7 * requested
        and exposed
    )


def terrain_levels(env, env_ids):
    """Consume complete outcomes at reset; assign shared targets only to reset rows."""
    terrain = _terrain(env)
    teacher = getattr(env, "_benchmark_teacher", None)
    if terrain is None or teacher is None:
        return torch.tensor(0.0, device=env.device)
    episode, families = teacher["episodes"], _families(env)
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    for row in sorted(ids.tolist()):
        family = families[row]
        if family == "Flat":
            continue
        state = teacher["families"][family]
        ended = bool(env.reset_terminated[row] or env.reset_time_outs[row])
        current = (
            episode["level"][row] == state["target_level"]
            and episode["revision"][row] == state["revision"]
        )
        if (
            ended
            and episode["eligible"][row]
            and episode["steps"][row] > 0
            and env.episode_length_buf[row] > 0
            and current
        ):
            state["block"].append(_success(env, row, episode))
            if len(state["block"]) == BLOCK_SIZE:
                successes, old = sum(state["block"]), state["target_level"]
                delta = (
                    1
                    if successes >= PROMOTE_SUCCESSES
                    else -1 if successes <= DEMOTE_SUCCESSES else 0
                )
                new = max(0, min(terrain.terrain_origins.shape[0] - 1, old + delta))
                state["completed_blocks"] += 1
                state["history"].append(
                    dict(
                        block=state["completed_blocks"],
                        level=old,
                        successes=successes,
                        next_level=new,
                    )
                )
                state["block"].clear()
                state["target_level"] = new
                state["revision"] += int(new != old)
    for row in ids.tolist():
        terrain.terrain_levels[row] = (
            0
            if families[row] == "Flat"
            else teacher["families"][families[row]]["target_level"]
        )
    terrain.env_origins[ids] = terrain.terrain_origins[
        terrain.terrain_levels[ids], terrain.terrain_types[ids]
    ]
    return terrain.terrain_levels.float().mean()


def curriculum_state(env):
    """Persist complete family decisions and pending blocks, not live trajectories."""
    terrain = _terrain(env)
    if terrain is None:
        return None
    teacher = getattr(env, "_benchmark_teacher", None)
    return dict(
        version=VERSION,
        definition=deepcopy(env.cfg.benchmark_terrain),
        levels=terrain.terrain_levels.tolist(),
        types=terrain.terrain_types.tolist(),
        families=None if teacher is None else deepcopy(teacher["families"]),
    )


def validate_curriculum_state(env, state):
    """Prepare all teacher/assignment state without mutating any live component."""
    terrain = _terrain(env)
    expected = curriculum_state(env)
    if terrain is None and state is None:
        return None
    if (
        terrain is None
        or type(state) is not dict
        or set(state) != set(expected)
        or state.get("version") != VERSION
    ):
        raise ValueError(
            "Curriculum continuation requires the v2 family teacher schema"
        )
    if (
        state["definition"] != expected["definition"]
        or type(state["types"]) is not list
        or state["types"] != expected["types"]
        or any(type(v) is not int for v in state["types"])
    ):
        raise ValueError("Curriculum definition or family assignments differ")
    levels = state["levels"]
    if (
        type(levels) is not list
        or len(levels) != env.num_envs
        or any(
            type(v) is not int or not 0 <= v < terrain.terrain_origins.shape[0]
            for v in levels
        )
    ):
        raise ValueError("Curriculum levels are outside the declared ladder")
    names = _families(env)
    if any(levels[row] != 0 for row, name in enumerate(names) if name == "Flat"):
        raise ValueError("Flat assignments must remain at zero")
    families = state["families"]
    if (
        type(families) is not dict
        or type(expected["families"]) is not dict
        or set(families) != set(expected["families"])
    ):
        raise ValueError("Curriculum family teacher identity differs")
    for record in families.values():
        if type(record) is not dict or set(record) != set(_new_family()):
            raise ValueError("Invalid curriculum family schema")
        target, revision, block, completed, history = (
            record[key]
            for key in (
                "target_level",
                "revision",
                "block",
                "completed_blocks",
                "history",
            )
        )
        if (
            type(target) is not int
            or not 0 <= target < terrain.terrain_origins.shape[0]
            or type(revision) is not int
            or revision < 0
            or type(completed) is not int
            or completed < 0
            or type(block) is not list
            or len(block) >= BLOCK_SIZE
            or any(type(v) is not bool for v in block)
            or type(history) is not list
            or len(history) != completed
        ):
            raise ValueError("Invalid curriculum family counters or pending block")
        level = changes = 0
        for index, item in enumerate(history, 1):
            if (
                type(item) is not dict
                or set(item) != {"block", "level", "successes", "next_level"}
                or any(type(v) is not int for v in item.values())
                or item["block"] != index
                or item["level"] != level
                or not 0 <= item["successes"] <= BLOCK_SIZE
            ):
                raise ValueError("Invalid curriculum decision history")
            delta = (
                1
                if item["successes"] >= PROMOTE_SUCCESSES
                else -1 if item["successes"] <= DEMOTE_SUCCESSES else 0
            )
            following = max(0, min(terrain.terrain_origins.shape[0] - 1, level + delta))
            if item["next_level"] != following:
                raise ValueError("Curriculum decision differs from shared thresholds")
            changes += int(following != level)
            level = following
        if level != target or changes != revision:
            raise ValueError("Curriculum history and current target differ")
    return dict(
        levels=torch.tensor(levels, device=env.device, dtype=torch.long),
        families=deepcopy(families),
    )


def restore_curriculum_state(env, prepared):
    """Restore administrative decisions; the caller starts fresh physical episodes."""
    if prepared is not None:
        terrain = env.scene.terrain
        terrain.terrain_levels.copy_(prepared["levels"])
        terrain.env_origins.copy_(
            terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        )
        env._benchmark_teacher = dict(
            families=deepcopy(prepared["families"]), episodes=_new_episodes(env)
        )
