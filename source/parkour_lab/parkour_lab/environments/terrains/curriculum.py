"""Native distance progression with prescribed episode eligibility and held caps."""

from copy import deepcopy

import torch

VERSION = "go2_eligible_distance_curriculum_v1"


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


def _flat(env, types):
    columns = env.cfg.benchmark_terrain["columns"]
    return torch.tensor(
        [column["family"] == "Flat" for column in columns],
        device=types.device,
        dtype=torch.bool,
    )[types]


def initialize_levels(env):
    """Choose low initial rows privately; Flat has no artificial level ladder."""
    if not env.cfg.benchmark_training:
        return
    terrain = _terrain(env)
    if terrain is None:
        return
    definition = env.cfg.benchmark_terrain
    low, high = definition["initial_min_level"], definition["initial_max_level"]
    if not 0 <= low <= high < definition["level_count"]:
        raise ValueError("Initial terrain range is outside the physical rows")
    draws = env.parkour_randomization.uniform(None, 1, "geometry")[:, 0]
    levels = torch.as_tensor(
        low + (draws * (high - low + 1)).astype("int64"),
        dtype=torch.long,
        device=terrain.terrain_levels.device,
    )
    levels[_flat(env, terrain.terrain_types)] = 0
    terrain.terrain_levels.copy_(levels)
    terrain.env_origins.copy_(terrain.terrain_origins[levels, terrain.terrain_types])


def terrain_levels(env, env_ids):
    """Apply the stock net-distance heuristic only to unchanged translation episodes.

    Bottom clips to zero and the highest physical row holds. The native importer
    computes origins and preserves types; its unconditional unused recycling
    draw is isolated from the learner's global random stream.
    """
    terrain = _terrain(env)
    if terrain is None:
        return torch.tensor(0.0, device=env.device)
    ids = torch.as_tensor(
        env_ids, device=terrain.terrain_levels.device, dtype=torch.long
    )
    command = env.command_manager.get_term("base_velocity")
    current = command.command[ids]
    eligible = (
        command.curriculum_eligible[ids]
        & (command.category[ids] == 0)
        & (current == command.episode_command[ids]).all(-1)
        & (env.episode_length_buf[ids] > 0)
        & ~_flat(env, terrain.terrain_types[ids])
    )
    ids, current = ids[eligible], current[eligible]
    if len(ids):
        root = env.scene["robot"].data.root_pos_w
        distance = (root[ids, :2] - terrain.env_origins[ids, :2]).norm(dim=-1)
        move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < current[:, :2].norm(dim=-1) * env.max_episode_length_s * 0.5
        ) & ~move_up
        move_up &= terrain.terrain_levels[ids] < terrain.terrain_origins.shape[0] - 1
        device = torch.device(env.device)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            terrain.update_env_origins(ids, move_up, move_down)
    return terrain.terrain_levels.float().mean()


def curriculum_state(env):
    """Capture physical assignments and their versioned family definition."""
    terrain = _terrain(env)
    if terrain is None:
        return None
    return dict(
        version=VERSION,
        definition=deepcopy(env.cfg.benchmark_terrain),
        levels=terrain.terrain_levels.cpu().tolist(),
        types=terrain.terrain_types.cpu().tolist(),
    )


def validate_curriculum_state(env, state):
    """Validate without mutation; return prepared levels for the commit boundary."""
    terrain = _terrain(env)
    if state is None and terrain is None:
        return None
    if (
        terrain is None
        or type(state) is not dict
        or set(state) != {"version", "definition", "levels", "types"}
        or state["version"] != VERSION
        or state["definition"] != env.cfg.benchmark_terrain
        or type(state["levels"]) is not list
        or len(state["levels"]) != env.num_envs
        or any(type(value) is not int for value in state["levels"])
        or type(state["types"]) is not list
        or any(type(value) is not int for value in state["types"])
        or state["types"] != terrain.terrain_types.cpu().tolist()
    ):
        raise ValueError("Curriculum continuation identity or assignments differ")
    levels = torch.tensor(state["levels"], device=env.device, dtype=torch.long)
    if ((levels < 0) | (levels >= terrain.terrain_origins.shape[0])).any() or (
        levels[_flat(env, terrain.terrain_types)] != 0
    ).any():
        raise ValueError("Curriculum levels are outside the declared family ladder")
    return levels


def restore_curriculum_state(env, levels):
    """Commit prepared assignments before fresh reset; no live episode is restored."""
    if levels is not None:
        terrain = env.scene.terrain
        terrain.terrain_levels.copy_(levels)
        terrain.env_origins.copy_(
            terrain.terrain_origins[levels, terrain.terrain_types]
        )
