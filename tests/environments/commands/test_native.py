"""Native command sampling and physical-reset ownership contracts."""

import sys
import unittest
from dataclasses import asdict
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from parkour_lab.environments.terrains.procedural import PROFILE_BY_COLUMN
from tests.support.commands import native_commands


class NativeCommandResetTests(unittest.TestCase):
    def config(self):
        return NS(heading_command=False, rel_heading_envs=0.0, rel_standing_envs=0.0)

    def test_rough_seeded_packets_follow_current_level_without_heading_assistance(self):
        import numpy as np

        from parkour_lab.config import TaskConfig
        from parkour_lab.environments.randomization import TaskRandomization
        from tests.support.isaaclab import (
            NativeConfig,
            pinned_generator,
            stock_rough_config,
        )

        generator = stock_rough_config().scene.terrain.terrain_generator
        generator.num_cols = 25
        generator.sub_terrains["flat"] = NativeConfig(proportion=0.25)
        # Execute the real SDK column dispatch without generating any mesh.
        columns = {}
        with pinned_generator() as (_, _, native_generator):
            subject = NS(
                cfg=generator,
                np_rng=np.random.default_rng(73),
                _get_terrain_mesh=lambda difficulty, sub: (None, None),
                _add_sub_terrain=lambda mesh, origin, row, col, sub: columns.setdefault(
                    col,
                    next(
                        name
                        for name, value in generator.sub_terrains.items()
                        if value is sub
                    ),
                ),
            )
            native_generator._generate_curriculum_terrains(subject)
        self.assertEqual(
            list(columns.values()),
            ["pyramid_stairs"] * 4
            + ["pyramid_stairs_inv"] * 4
            + ["boxes"] * 4
            + ["random_rough"] * 4
            + ["hf_pyramid_slope"] * 2
            + ["hf_pyramid_slope_inv"] * 2
            + ["flat"] * 5,
        )
        global_rng = torch.get_rng_state().clone()
        for seed, fraction in ((0, 0.0), (42, 0.0), (0, 0.2), (42, 0.2), (42, 0.9)):
            task = TaskConfig(
                terrain="rough",
                num_envs=100,
                seed=seed,
                rough_pivot_fraction=fraction,
            )
            randomization, expected = TaskRandomization(task), TaskRandomization(task)
            terrain = NS(
                terrain_types=torch.arange(100) % 25,
                terrain_levels=torch.zeros(100, dtype=torch.long),
            )
            env = NS(
                num_envs=100,
                device="cpu",
                episode_length_buf=torch.ones(100, dtype=torch.long),
                parkour_randomization=randomization,
                cfg=NS(
                    scene=NS(terrain=NS(terrain_generator=generator)),
                    parkour_task=asdict(task),
                ),
                scene=NS(terrain=terrain),
            )
            with (
                native_commands() as commands,
                patch.dict(
                    sys.modules,
                    {
                        "parkour_lab.environments.dynamics": NS(
                            get_randomization=lambda env: env.parkour_randomization
                        )
                    },
                ),
            ):
                command = commands.RoughVelocityCommand(self.config(), env)
            pivot_count = 0
            for level in (0, 1, 4):
                terrain.terrain_levels.fill_(level)
                command._resample(torch.arange(100))
                draws = expected.uniform(None, 4, "commands").astype(np.float32)
                values = draws[:, :3] * np.array(
                    [0.7, 0.4, 1.0], dtype=np.float32
                ) + np.array([-0.2, -0.2, -0.5], dtype=np.float32)
                restricted = np.zeros(100, dtype=bool)
                # Stairs, two-sided boxes and rounded pyramid faces narrow from row1.
                if level >= 1:
                    restricted = (
                        (terrain.terrain_types < 12)
                        | ((terrain.terrain_types >= 16) & (terrain.terrain_types < 20))
                    ).numpy()
                values[restricted, 0] = 0.2 + 0.3 * draws[restricted, 0]
                values[restricted, 1:] = 0
                stops = (draws[:, 3] < 0.1) & ~restricted
                values[stops] = 0
                pivots = (
                    (draws[:, 3] >= 0.1) & (draws[:, 3] < 0.1 + fraction) & ~restricted
                )
                yaw = draws[pivots, 2]
                values[pivots, :2] = 0
                values[pivots, 2] = np.where(
                    yaw < 0.5, -0.3 - 0.4 * yaw, 0.3 + 0.4 * (yaw - 0.5)
                )
                pivot_count += int(pivots.sum())
                self.assertTrue(
                    torch.allclose(
                        command.vel_command_b,
                        torch.tensor(values, dtype=torch.float32),
                        rtol=0,
                        atol=1e-7,
                    )
                )
                if fraction == 0:
                    self.assertTrue(
                        torch.equal(command.vel_command_b, torch.from_numpy(values))
                    )
                self.assertEqual(command.is_standing_env.tolist(), stops.tolist())
                self.assertFalse(command.is_heading_env.any())
                self.assertEqual(command.sampling_report["pivots"], pivot_count)
            before = command.vel_command_b.clone()
            command._resample([0])
            expected.uniform([0], 4, "commands")
            env.episode_length_buf[0] = 0
            command.compute(0.02)
            self.assertEqual(command.remaining_ticks[:2].tolist(), [200, 199])
            self.assertTrue(torch.equal(before[1:], command.vel_command_b[1:]))
            self.assertEqual(randomization.manifest(), expected.manifest())
            command.time_left.fill_(float("inf"))
            command.vel_command_b[:] = torch.tensor([-0.2, 0.1, -0.4])
            before = command.vel_command_b.clone()
            report = dict(command.sampling_report)
            manifest = randomization.manifest()
            for _ in range(201):
                command.compute(0.02)
            self.assertTrue(torch.equal(command.vel_command_b, before))
            self.assertTrue(torch.isinf(command.time_left).all())
            self.assertEqual(command.sampling_report, report)
            self.assertEqual(randomization.manifest(), manifest)
        self.assertTrue(torch.equal(global_rng, torch.get_rng_state()))

    def test_rough_pivot_partition_yaw_boundaries_and_restricted_subset(self):
        from parkour_lab.config import TaskConfig
        from tests.support.isaaclab import stock_rough_config

        task = TaskConfig(terrain="rough", num_envs=10, rough_pivot_fraction=0.2)
        ids = torch.tensor([7, 2, 5, 1, 4, 0, 9, 8])
        # Stop upper bound, both pivot signs, pivot upper bound, and two restricted rows.
        draws = torch.tensor(
            [
                [0.5, 0.5, 0.0, 0.099],
                [0.5, 0.5, 0.0, 0.1],
                [0.5, 0.5, 0.5, 0.299],
                [0.5, 0.5, 0.25, 0.3],
                [0.0, 0.5, 0.499, 0.1],
                [1.0, 0.5, 0.999, 0.299],
                [0.5, 0.5, 0.499, 0.1],
                [0.5, 0.5, 0.999, 0.299],
            ]
        )
        generator = stock_rough_config().scene.terrain.terrain_generator
        terrain = NS(
            terrain_types=torch.tensor([0, 12, 12, 12, 0, 12, 12, 12, 12, 12]),
            terrain_levels=torch.ones(10, dtype=torch.long),
        )
        calls = []

        def uniform(selected, width, stream):
            calls.append((selected.tolist(), width, stream))
            return draws

        env = NS(
            num_envs=10,
            device="cpu",
            episode_length_buf=torch.ones(10),
            cfg=NS(
                parkour_task=asdict(task),
                scene=NS(terrain=NS(terrain_generator=generator)),
            ),
            scene=NS(terrain=terrain),
            parkour_randomization=NS(uniform=uniform),
        )
        with (
            native_commands() as commands,
            patch.dict(
                sys.modules,
                {
                    "parkour_lab.environments.dynamics": NS(
                        get_randomization=lambda env: env.parkour_randomization
                    ),
                },
            ),
        ):
            command = commands.RoughVelocityCommand(self.config(), env)
        command.vel_command_b.fill_(9)
        command._resample(ids)
        torch.testing.assert_close(
            command.vel_command_b[ids],
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, -0.3],
                    [0.0, 0.0, 0.3],
                    [0.15, 0.0, -0.25],
                    [0.2, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.0, 0.0, -0.4996],
                    [0.0, 0.0, 0.4996],
                ]
            ),
        )
        self.assertEqual(command.is_standing_env[ids].tolist(), [True] + [False] * 7)
        self.assertTrue((command.vel_command_b[[3, 6]] == 9).all())
        self.assertEqual(calls, [(ids.tolist(), 4, "commands")])
        self.assertEqual(command.sampling_report["pivots"], 4)
        self.assertEqual(command.sampling_report["stops"], 1)
        self.assertEqual(command.sampling_report["restricted_samples"], 2)
        self.assertTrue((command.remaining_ticks[ids] == 200).all())

    def test_rough_clearance_allows_grid_descent_but_rejects_plunge_and_tilt(self):
        data = NS(
            root_pos_w=torch.tensor(
                [
                    [0.0, 0.0, -1.26],
                    [0.0, 0.0, -2.5],
                    [0.0, 0.0, -3.0],
                    [0.0, 0.0, 1.61],
                ]
            ),
            projected_gravity_b=torch.tensor(
                [[0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.0, -1.0], [0.0, 0.8, -0.6]]
            ),
        )
        scene = {
            "robot": NS(data=data),
            "base_height_scanner": NS(
                data=NS(
                    ray_hits_w=torch.tensor(
                        [
                            [[0.0, 0.0, -1.61]],
                            [[float("inf")] * 3],
                            [[0.0, 0.0, -3.05]],
                            [[0.0, 0.0, 1.26]],
                        ]
                    )
                )
            ),
        }

        class Scene(dict):
            env_origins = torch.tensor([[0.0, 0.0, 1.61]] * 4)

        env = NS(num_envs=4, scene=Scene(scene))
        with native_commands() as commands:
            failed = commands.procedural_physical_failure(
                env, 0.12, -4.0, 0.5, max_tilt_rad=torch.pi / 4
            )
        self.assertEqual(failed.tolist(), [False, False, True, True])
        data.root_pos_w[1, 2] = -3.0
        with native_commands() as commands:
            failed = commands.procedural_physical_failure(
                env, 0.12, -4.0, 0.5, max_tilt_rad=torch.pi / 4
            )
        self.assertTrue(failed[1])

    def test_flat_uniform_sampler_exact_period_and_partial_reset(self):
        from parkour_lab.config import TaskConfig
        from parkour_lab.environments.randomization import TaskRandomization

        env = NS(
            num_envs=100,
            device="cpu",
            episode_length_buf=torch.ones(100, dtype=torch.long),
        )
        env.parkour_randomization = TaskRandomization(
            TaskConfig(terrain="flat", num_envs=100)
        )
        with (
            native_commands() as commands,
            patch.dict(
                sys.modules,
                {
                    "parkour_lab.environments.dynamics": NS(
                        get_randomization=lambda env: env.parkour_randomization
                    )
                },
            ),
        ):
            command = commands.FlatVelocityCommand(self.config(), env)
        state = torch.get_rng_state().clone()
        command._resample(torch.arange(100))
        first = command.vel_command_b.clone()
        self.assertTrue((first >= torch.tensor([-0.2, -0.2, -0.5])).all())
        self.assertTrue((first <= torch.tensor([0.5, 0.2, 0.5])).all())
        for _ in range(199):
            command.compute(0.02)
        self.assertTrue(torch.equal(first, command.vel_command_b))
        command.compute(0.02)
        self.assertTrue((command.command_counter == 2).all())
        self.assertFalse(torch.equal(first, command.vel_command_b))
        env.episode_length_buf[0] = 0
        command._resample(torch.tensor([0]))
        command.compute(0.02)
        self.assertEqual(command.remaining_ticks[:2].tolist(), [200, 199])
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(command.sampling_report["samples"], 201)
        command.time_left.fill_(float("inf"))
        external = command.vel_command_b.clone()
        for _ in range(250):
            command.compute(0.02)
        self.assertTrue(torch.equal(external, command.vel_command_b))
        self.assertTrue(torch.isinf(command.time_left).all())
        self.assertEqual(command.sampling_report["samples"], 201)

    def test_native_resample_uses_counter_reset_not_previous_episode_category(self):
        with native_commands() as commands:
            command = commands.OperatorVelocityCommand(
                self.config(), NS(num_envs=3, device="cpu")
            )
        command.command_counter[:] = torch.tensor([0, 4, 1])
        command.category[:] = torch.tensor([7, 8, 1])
        calls = []

        def sample(count, **kwargs):
            calls.append((count, kwargs))
            return (
                torch.tensor([3, 0]),
                torch.tensor([[0.4, 0, 0], [0, 0, 0]]),
                torch.tensor([4.0, 7.0]),
            )

        command.sampler = NS(sample=sample)
        command._resample_command(torch.tensor([2, 0]))
        self.assertEqual(calls[0][0], 2)
        self.assertEqual(calls[0][1]["previous_category"].tolist(), [1, 7])
        self.assertEqual(calls[0][1]["new_episode"].tolist(), [False, True])
        self.assertEqual(command.category.tolist(), [0, 8, 3])
        self.assertEqual(command.time_left.tolist(), [7.0, 0.0, 4.0])
        self.assertEqual(command.is_standing_env.tolist(), [True, False, False])
        command._resample_command(torch.tensor([], dtype=torch.long))
        self.assertEqual(len(calls), 1)

    def test_sequence_hook_receives_same_physical_reset_mask_and_owns_selected_rows(
        self,
    ):
        generator = NS(
            curriculum=True,
            num_cols=20,
            num_rows=3,
            sub_terrains={
                str(i): NS(profile=profile, proportion=0.05)
                for i, profile in enumerate(PROFILE_BY_COLUMN)
            },
        )
        env = NS(
            num_envs=3,
            device="cpu",
            cfg=NS(
                scene=NS(terrain=NS(terrain_generator=generator)),
                curriculum=NS(terrain_levels=None),
            ),
            scene=NS(terrain=NS(terrain_types=torch.tensor([0, 12, 4]))),
        )
        with native_commands() as commands:
            command = commands.ProceduralTerrainCommand(self.config(), env)
        command.command_counter[:] = torch.tensor([0, 4, 1])
        command.category[1] = 7
        command.vel_command_b[1] = torch.tensor([0.4, 0.0, 0.2])
        command.time_left[1] = 8.0
        command.sampler = NS(
            sample=lambda count, **kwargs: (
                torch.zeros(count, dtype=torch.int64),
                torch.zeros(count, 3),
                torch.ones(count),
            )
        )
        calls = []

        def resample(ids, new_episode):
            calls.append((ids.clone(), new_episode.clone()))
            return (
                torch.tensor([0]),
                torch.tensor([[0.0, 0.0, -0.5]]),
                torch.tensor([2]),
                torch.tensor([4.0]),
            )

        command.sequence_plan = NS(resample=resample)
        command._resample_command(torch.tensor([2, 0]))
        self.assertEqual(calls[0][0].tolist(), [2, 0])
        self.assertEqual(calls[0][1].tolist(), [False, True])
        self.assertEqual(command.category.tolist(), [2, 7, 0])
        self.assertEqual(command.vel_command_b[0].tolist(), [0.0, 0.0, -0.5])
        self.assertFalse(command.is_standing_env[0])
        self.assertEqual(command.time_left[0], 4.0)
        torch.testing.assert_close(
            command.vel_command_b[1], torch.tensor([0.4, 0.0, 0.2])
        )
        self.assertEqual(command.time_left[1], 8.0)

    def test_procedural_sampler_rough_distribution_never_draws_reverse_or_lateral(self):
        generator = NS(
            curriculum=True,
            num_cols=20,
            num_rows=3,
            sub_terrains={
                str(i): NS(profile=profile, proportion=0.05)
                for i, profile in enumerate(PROFILE_BY_COLUMN)
            },
        )
        env = NS(
            num_envs=3,
            device="cpu",
            cfg=NS(
                scene=NS(terrain=NS(terrain_generator=generator)),
                curriculum=NS(terrain_levels=None),
            ),
            scene=NS(terrain=NS(terrain_types=torch.tensor([0, 12, 4]))),
        )
        with native_commands() as commands:
            command = commands.ProceduralTerrainCommand(self.config(), env)
        self.assertEqual(command.flat_command_rows.tolist(), [True, False, True])
        categories, values, _ = command.rough_sampler.sample(
            4096, generator=torch.Generator().manual_seed(23)
        )
        self.assertTrue((categories < 6).all())
        self.assertTrue((values[:, 0] >= 0).all())
        self.assertTrue((values[:, 1] == 0).all())
        self.assertTrue((values[:, 2] > 0).any())
        self.assertTrue((values[:, 2] < 0).any())


if __name__ == "__main__":
    unittest.main()
