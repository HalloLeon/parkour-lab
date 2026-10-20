"""Measured family-teacher fixture; no native simulation or fake learner state."""

from types import SimpleNamespace as NS

import torch

from parkour_lab.control.proprioception import FOOT_NAMES
from parkour_lab.environments.randomization import TaskRandomization
from parkour_lab.environments.terrains.curriculum import (
    begin_episodes,
    initialize_levels,
)
from parkour_lab.runtime.accounting import InteractionCounter


def curriculum_environment():
    """Eight asynchronous replicas with stable five-family column assignments."""
    types = torch.tensor([0, 1, 2, 3, 4, 1, 1, 1])
    levels = torch.zeros(8, dtype=torch.long)
    origins = torch.zeros(16, 5, 3)
    origins[:, :, 0] = torch.arange(16)[:, None] * 12
    origins[:, :, 1] = torch.arange(5)[None] * 12
    terrain = NS(
        terrain_origins=origins,
        terrain_levels=levels,
        terrain_types=types,
        env_origins=origins[levels, types].clone(),
    )
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
    data = NS(
        root_pos_w=terrain.env_origins.clone(),
        root_quat_w=torch.tensor([[1.0, 0, 0, 0]]).repeat(8, 1),
        root_com_lin_vel_b=torch.tensor([[0.4, 0, 0]]).repeat(8, 1),
        root_ang_vel_b=torch.zeros(8, 3),
        body_pos_w=terrain.env_origins[:, None, :].repeat(1, 4, 1),
    )
    data.root_pos_w[:, 2] = 0.4
    data.body_pos_w[:, :, 2] = 0.02

    class Scene(dict):
        pass

    scene = Scene(
        robot=NS(data=data, body_names=list(FOOT_NAMES)),
        contact_forces=NS(
            body_names=list(FOOT_NAMES), data=NS(net_forces_w=torch.zeros(8, 4, 3))
        ),
    )
    scene.terrain = terrain
    scene["contact_forces"].data.net_forces_w[:, :, 2] = 20.0
    terms = {
        name: torch.zeros(8, dtype=torch.bool)
        for name in ("time_out", "terrain_boundary", "base_contact")
    }
    sampler = TaskRandomization(dict(num_envs=8, seed=42, dynamics="randomized"))
    sampler.dynamics()
    env = NS(
        cfg=NS(
            benchmark_version="go2-blind-v1",
            benchmark_task_id="Go2-Mixed-v1",
            benchmark_training=True,
            seed=42,
            benchmark_terrain=dict(
                version="test-physical-levels-v2",
                level_count=16,
                num_rows=16,
                initial_min_level=0,
                initial_max_level=0,
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
        step_dt=0.02,
        episode_length_buf=torch.zeros(8, dtype=torch.long),
        command_manager=NS(get_term=lambda name: command),
        termination_manager=NS(active_terms=list(terms), get_term=terms.__getitem__),
        reset_terminated=torch.zeros(8, dtype=torch.bool),
        reset_time_outs=torch.zeros(8, dtype=torch.bool),
        parkour_randomization=sampler,
        interactions=InteractionCounter(8),
    )
    initialize_levels(env)
    begin_episodes(env, torch.arange(8))
    return env
