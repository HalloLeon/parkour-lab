"""One private command draw per episode, with permanent eligibility invalidation."""

import importlib.util
import sys
import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from tests.support import SOURCE_ROOT
from tests.support.commands import native_commands
from tests.support.curriculum import curriculum_environment


@contextmanager
def episode_command(env):
    path = SOURCE_ROOT / "parkour_lab/environments/commands/episode.py"
    with (
        native_commands() as native,
        patch.dict(
            sys.modules,
            {
                "parkour_lab.environments.commands.native": native,
                "parkour_lab.environments.dynamics": NS(
                    get_randomization=lambda env: env.parkour_randomization
                ),
            },
        ),
    ):
        spec = importlib.util.spec_from_file_location(
            "parkour_lab.environments.commands._episode_test", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module.EpisodeVelocityCommand(
            NS(heading_command=False, rel_heading_envs=0.0, rel_standing_envs=0.0), env
        )


class EpisodeCommandTests(unittest.TestCase):
    def test_command_draws_do_not_depend_on_terrain_family_or_level(self):
        first, second = curriculum_environment(), curriculum_environment()
        second.scene.terrain.terrain_types = second.scene.terrain.terrain_types.flip(0)
        with episode_command(first) as left, episode_command(second) as right:
            left._resample(torch.arange(8))
            right._resample(torch.arange(8))
            torch.testing.assert_close(
                left.vel_command_b, right.vel_command_b, rtol=0, atol=0
            )
            self.assertEqual(left.sampling_report, right.sampling_report)

    def test_six_prescribed_categories_ranges_and_no_global_draws(self):
        env = curriculum_environment()
        with episode_command(env) as command:
            packets = np.array(
                [
                    [p, 0.5, 0.75, 0.25]
                    for p in (0.1, 0.55, 0.65, 0.75, 0.85, 0.95, 0.1, 0.55)
                ]
            )
            command.randomization = NS(uniform=lambda ids, width, stream: packets[ids])
            rng = torch.get_rng_state().clone()
            command._resample(torch.arange(8))
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            self.assertEqual(command.category.tolist(), [0, 1, 2, 3, 4, 5, 0, 1])
            self.assertEqual(
                command.curriculum_eligible.tolist(),
                [True, False, False, False, False, False, True, False],
            )
            torch.testing.assert_close(
                command.vel_command_b[:6],
                torch.tensor(
                    [
                        [0.35, 0.1, 0],
                        [0, 0, 0],
                        [0, 0, 0.35],
                        [0, 0, -0.35],
                        [0.35, 0, 0.275],
                        [0.35, 0, -0.275],
                    ]
                ),
            )
            self.assertEqual(command.remaining_ticks.tolist(), [1000] * 8)
            self.assertEqual(
                command.sampling_report,
                dict(
                    version="go2_episode_twist_v1",
                    period_steps=1000,
                    samples=8,
                    translations=2,
                    stops=2,
                    pivots=2,
                    arcs=2,
                ),
            )

    def test_packet_is_held_for_whole_horizon_and_partial_reset_owns_only_its_row(self):
        env = curriculum_environment()
        with episode_command(env) as command:
            command._resample(torch.arange(8))
            original = command.vel_command_b.clone()
            for _ in range(999):
                command.compute(0.02)
            torch.testing.assert_close(command.vel_command_b, original, rtol=0, atol=0)
            self.assertEqual(command.sampling_report["samples"], 8)
            command.command_counter[0] = 0
            command._resample([0])
            self.assertEqual(command.remaining_ticks.tolist(), [1000] + [1] * 7)
            torch.testing.assert_close(
                command.vel_command_b[1:], original[1:], rtol=0, atol=0
            )
            self.assertEqual(command.sampling_report["samples"], 9)

    def test_empty_and_ordered_subset_sequences_preserve_other_rows_and_streams(self):
        env = curriculum_environment()
        with episode_command(env) as command:
            before = command.randomization.state_dict()
            command._resample([])
            self.assertEqual(command.randomization.state_dict(), before)
            self.assertEqual(command.sampling_report["samples"], 0)
            command._resample([3, 0])
            self.assertEqual(command.command_counter.tolist(), [1, 0, 0, 1, 0, 0, 0, 0])
            self.assertTrue((command.category[[0, 3]] >= 0).all())
            self.assertTrue((command.category[[1, 2, 4, 5, 6, 7]] == -1).all())
            after = command.randomization.state_dict()
            for row in (1, 2, 4, 5, 6, 7):
                self.assertEqual(after["generators"][row], before["generators"][row])
            self.assertEqual(command.sampling_report["samples"], 2)

    def test_changed_then_restored_packet_never_regains_eligibility_before_reset(self):
        env = curriculum_environment()
        with episode_command(env) as command:
            command.randomization = NS(
                uniform=lambda ids, width, stream: np.tile(
                    [0.1, 0.5, 0.5, 0.5], (len(ids), 1)
                )
            )
            command._resample(torch.arange(8))
            original = command.vel_command_b.clone()
            command.vel_command_b[2, 2] = 0.2
            command.compute(0.02)
            command.vel_command_b.copy_(original)
            command.compute(0.02)
            self.assertFalse(command.curriculum_eligible[2])
            self.assertTrue(command.curriculum_eligible[[0, 1, 3, 4, 5, 6, 7]].all())
            command.command_counter[2] = 0
            command._resample(torch.tensor([2]))
            self.assertTrue(command.curriculum_eligible[2])
