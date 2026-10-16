"""Native origin-update fixture with independent private randomization streams."""

import ast
from types import MethodType
from types import SimpleNamespace as NS

import torch

from parkour_lab.environments.randomization import TaskRandomization
from parkour_lab.runtime.accounting import InteractionCounter
from tests.support.isaaclab import pinned_source_root


def curriculum_environment():
    """Execute the SDK's real update_env_origins, with no simulator or meshes."""
    path = pinned_source_root() / "terrains/terrain_importer.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TerrainImporter"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "update_env_origins"
    )
    namespace = {"torch": torch}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    types = torch.tensor([0, 1, 2, 3, 4, 1, 1, 1])
    levels = torch.tensor([0, 2, 3, 4, 5, 0, 7, 4])
    origins = torch.zeros(8, 5, 3)
    origins[:, :, 0] = torch.arange(8)[:, None] * 12
    origins[:, :, 1] = torch.arange(5)[None] * 12
    terrain = NS(
        terrain_origins=origins,
        terrain_levels=levels,
        terrain_types=types,
        max_terrain_level=8,
        env_origins=origins[levels, types].clone(),
        cfg=NS(terrain_generator=NS(size=(12.0, 12.0))),
    )
    terrain.update_env_origins = MethodType(namespace["update_env_origins"], terrain)
    command = NS(
        command=torch.tensor([[0.4, 0.0, 0.0]]).repeat(8, 1),
        episode_command=torch.tensor([[0.4, 0.0, 0.0]]).repeat(8, 1),
        curriculum_eligible=torch.ones(8, dtype=torch.bool),
        category=torch.zeros(8, dtype=torch.long),
        sampling_report=dict(
            version="go2_episode_twist_v1",
            period_steps=1000,
            samples=0,
            translations=0,
            stops=0,
            pivots=0,
            arcs=0,
        ),
    )

    class Scene(dict):
        pass

    scene = Scene(robot=NS(data=NS(root_pos_w=terrain.env_origins.clone())))
    scene.terrain = terrain
    scene.env_origins = terrain.env_origins
    sampler = TaskRandomization(dict(num_envs=8, seed=42, dynamics="randomized"))
    sampler.dynamics()
    return NS(
        cfg=NS(
            benchmark_version="go2-blind-v1",
            benchmark_task_id="Go2-Mixed-v1",
            benchmark_training=True,
            seed=42,
            benchmark_terrain=dict(
                version="test-physical-levels-v1",
                level_count=8,
                num_rows=8,
                initial_min_level=0,
                initial_max_level=1,
                columns=[
                    dict(family=name, subtype="test")
                    for name in ("Flat", "Uneven", "Slopes", "Stairs", "Hills")
                ],
            ),
        ),
        num_envs=8,
        device="cpu",
        scene=scene,
        max_episode_length_s=20.0,
        episode_length_buf=torch.ones(8, dtype=torch.long),
        command_manager=NS(get_term=lambda name: command),
        parkour_randomization=sampler,
        interactions=InteractionCounter(8),
    )
