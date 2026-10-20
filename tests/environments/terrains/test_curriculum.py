"""Measured translation outcomes, asynchronous family blocks and atomic continuation."""

import json
import unittest
from copy import deepcopy
from unittest.mock import patch

import torch

from parkour_lab.benchmark.progress import capture_progress, restore_progress
from parkour_lab.environments.terrains import curriculum as teacher
from tests.support.curriculum import curriculum_environment


def checkpoint(env):
    return capture_progress(
        env,
        run_id="seed42",
        checkpoint_id="learner-A",
        configuration_id="mixed-v2",
        milestones=[],
    )


def restore(env, record):
    return restore_progress(
        env, record, checkpoint_id="learner-A", configuration_id="mixed-v2"
    )


def episode(
    env,
    row=1,
    *,
    steps=1000,
    velocity=0.4,
    yaw=0.0,
    progress_velocity=None,
    ending="time_out",
    contact=True,
):
    """Advance the actual accumulator over every controlled outgoing sample."""
    robot = env.scene["robot"].data
    ids = torch.tensor([row])
    env.episode_length_buf.zero_()
    env.termination_manager.terminated.zero_()
    env.termination_manager.time_outs.zero_()
    for name in env.termination_manager.active_terms:
        env.termination_manager.get_term(name).zero_()
    robot.root_pos_w[row] = env.scene.terrain.env_origins[row]
    teacher.begin_episodes(env, ids)
    robot.root_com_lin_vel_b[row] = torch.tensor([velocity, 0, 0])
    robot.root_ang_vel_b[row, 2] = yaw

    def contact_samples(e, ids, state):
        if type(contact) is bool:
            return torch.full((len(ids),), contact, dtype=torch.bool)
        return state["steps"][ids] > steps - contact

    with patch.object(teacher, "_contact_samples", side_effect=contact_samples):
        for _ in range(steps):
            env.episode_length_buf += 1
            # Like the SDK, install step aliases only after physics begins.
            env.reset_terminated = env.termination_manager.terminated
            env.reset_time_outs = env.termination_manager.time_outs
            robot.root_pos_w[row, 0] += (
                velocity if progress_velocity is None else progress_velocity
            ) * env.step_dt
            teacher.observe_step(env)
    env.reset_terminated[row] = ending == "base_contact"
    env.reset_time_outs[row] = ending in ("time_out", "terrain_boundary")
    if ending:
        env.termination_manager.get_term(ending)[row] = True
    teacher.terrain_levels(env, ids)


def at_level(env, family, level):
    state = env._benchmark_teacher["families"][family]
    state["history"] = [
        dict(block=i + 1, level=i, successes=20, next_level=i + 1) for i in range(level)
    ]
    state.update(target_level=level, revision=level, completed_blocks=level, block=[])
    for row, name in enumerate(teacher._families(env)):
        if name == family:
            env.scene.terrain.terrain_levels[row] = level
            env.scene.terrain.env_origins[row] = env.scene.terrain.terrain_origins[
                level, env.scene.terrain.terrain_types[row]
            ]
    teacher.begin_episodes(env, torch.arange(env.num_envs))


class CurriculumTests(unittest.TestCase):
    def test_first_reset_without_step_aliases_assigns_targets_then_records_first_end(
        self,
    ):
        env = curriculum_environment()
        del env.reset_terminated, env.reset_time_outs
        before = teacher.curriculum_state(env)
        env._benchmark_teacher["episodes"]["eligible"].zero_()
        teacher.terrain_levels(env, torch.arange(env.num_envs))
        self.assertEqual(teacher.curriculum_state(env), before)
        self.assertFalse(hasattr(env, "reset_terminated"))
        self.assertFalse(hasattr(env, "reset_time_outs"))
        episode(env)
        self.assertEqual(env._benchmark_teacher["families"]["Uneven"]["block"], [True])

    def test_observed_episode_requires_real_manager_ending_masks(self):
        env = curriculum_environment()
        env._benchmark_teacher["episodes"]["steps"][1] = 1
        env.episode_length_buf[1] = 1
        del env.termination_manager.terminated
        with self.assertRaises(AttributeError):
            teacher.terrain_levels(env, torch.tensor([1]))

    def test_physical_failure_uses_manager_not_a_stale_step_alias(self):
        env = curriculum_environment()
        episode(env)
        state = env._benchmark_teacher["families"]["Uneven"]
        state["block"].clear()
        env.termination_manager.terminated[1] = True
        env.reset_terminated = torch.zeros_like(env.termination_manager.terminated)
        teacher.terrain_levels(env, torch.tensor([1]))
        self.assertEqual(state["block"], [False])

    def test_all_start_zero_without_global_or_private_rng_draws(self):
        env = curriculum_environment()
        before = deepcopy(env.parkour_randomization.state_dict())
        rng = torch.get_rng_state().clone()
        teacher.initialize_levels(env)
        self.assertEqual(env.scene.terrain.terrain_levels.tolist(), [0] * 8)
        self.assertEqual(before, env.parkour_randomization.state_dict())
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_native_buffer_dtype_is_independent_of_external_torch_default(self):
        env = curriculum_environment()
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            teacher.initialize_levels(env)
            teacher.begin_episodes(env, torch.arange(8))
            env.episode_length_buf += 1
            teacher.observe_step(env)
        finally:
            torch.set_default_dtype(previous)
        episode = env._benchmark_teacher["episodes"]
        for name in ("command", "anchor", "direction", "support"):
            self.assertEqual(
                episode[name].dtype, env.scene["robot"].data.root_pos_w.dtype
            )
        self.assertEqual(episode["planar_squared"].dtype, torch.float64)

    def test_measured_correct_stationary_wrong_direction_yaw_fall_and_short_boundary(
        self,
    ):
        cases = [
            ({}, True),
            ({"velocity": 0}, False),
            ({"velocity": -0.4}, False),
            ({"yaw": 0.3}, False),
            ({"progress_velocity": 0}, False),
            ({"ending": "base_contact", "steps": 10}, False),
            ({"ending": "terrain_boundary", "steps": 399}, False),
            ({"ending": "terrain_boundary", "steps": 400}, True),
            ({"ending": "time_out", "steps": 400}, False),
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                env = curriculum_environment()
                episode(env, **kwargs)
                self.assertEqual(
                    env._benchmark_teacher["families"]["Uneven"]["block"], [expected]
                )

    def test_level_zero_contact_exemption_and_nonflat_minimum_time(self):
        env = curriculum_environment()
        episode(env, contact=False)
        self.assertEqual(env._benchmark_teacher["families"]["Uneven"]["block"], [True])
        at_level(env, "Uneven", 1)
        episode(env, contact=False)
        episode(env, contact=49)
        episode(env, contact=50)
        self.assertEqual(
            env._benchmark_teacher["families"]["Uneven"]["block"], [False, False, True]
        )

    def test_twenty_outcome_blocks_promote_demote_hold_and_clear(self):
        for successes, new in ((16, 1), (15, 0), (8, 0)):
            env = curriculum_environment()
            state = env._benchmark_teacher["families"]["Uneven"]
            state["block"] = [True] * (successes - 1) + [False] * (20 - successes)
            episode(env)
            self.assertEqual(state["target_level"], new)
            self.assertEqual(state["block"], [])
            self.assertEqual(
                state["history"],
                [dict(block=1, level=0, successes=successes, next_level=new)],
            )
        env = curriculum_environment()
        at_level(env, "Uneven", 1)
        state = env._benchmark_teacher["families"]["Uneven"]
        state["block"] = [True] * 8 + [False] * 11
        episode(env, velocity=0)
        self.assertEqual(state["target_level"], 0)
        self.assertEqual(state["revision"], 2)

    def test_family_target_reset_only_and_obsolete_inflight_outcome_does_not_count(
        self,
    ):
        env = curriculum_environment()
        state = env._benchmark_teacher["families"]["Uneven"]
        state["block"] = [True] * 19
        episode(env, row=1)
        self.assertEqual(state["target_level"], 1)
        self.assertEqual(env.scene.terrain.terrain_levels[1], 1)
        self.assertEqual(env.scene.terrain.terrain_levels[5], 0)
        self.assertEqual(
            env.scene.terrain.terrain_types.tolist(), [0, 1, 2, 3, 4, 1, 1, 1]
        )
        # Row5's old-level episode started before the other replica's promotion.
        env.reset_time_outs[5] = True
        env.termination_manager.get_term("time_out")[5] = True
        teacher.terrain_levels(env, torch.tensor([5]))
        self.assertEqual(state["block"], [])
        self.assertEqual(env.scene.terrain.terrain_levels[5], 1)
        self.assertEqual(
            env._benchmark_teacher["families"]["Slopes"]["target_level"], 0
        )

    def test_pivot_changed_command_and_manual_reset_are_not_translation_outcomes(self):
        env = curriculum_environment()
        term = env.command_manager.get_term("base_velocity")
        term.category[1] = 2
        episode(env)
        term.category[1] = 0
        episode(env, ending=None)
        term.command[1, 0] = 0.3
        teacher.observe_step(env)
        env.reset_time_outs[1] = True
        teacher.terrain_levels(env, torch.tensor([1]))
        self.assertEqual(env._benchmark_teacher["families"]["Uneven"]["block"], [])

    def test_actual_support_contact_filters_and_platform_gate(self):
        env = curriculum_environment()
        at_level(env, "Slopes", 1)
        ep = env._benchmark_teacher["episodes"]
        ids = torch.tensor([2])
        feet = env.scene["robot"].data.body_pos_w
        feet[2, :, 0] = env.scene.terrain.env_origins[2, 0] + 1.5

        def ray(e, xyz):
            return torch.zeros(xyz.shape[:2]), torch.ones(
                xyz.shape[:2], dtype=torch.bool
            )

        with patch("parkour_lab.runtime.native.foot_support", side_effect=ray):
            self.assertTrue(teacher._contact_samples(env, ids, ep).item())
            feet[2, :, 0] = env.scene.terrain.env_origins[2, 0] + 0.5
            self.assertFalse(teacher._contact_samples(env, ids, ep).item())

            feet[2, :, 0] = env.scene.terrain.env_origins[2, 0] + 1.5
            feet[2, :, 2] = 0.09
            self.assertFalse(teacher._contact_samples(env, ids, ep).item())
            feet[2, :, 2] = 0.02
            env.scene["contact_forces"].data.net_forces_w.zero_()
            self.assertFalse(teacher._contact_samples(env, ids, ep).item())

    def test_no_step_reset_ignores_stale_teacher_and_termination_buffers(self):
        env = curriculum_environment()
        env._benchmark_teacher["episodes"]["steps"][1] = 1000
        env.reset_time_outs[1] = True
        env.termination_manager.get_term("time_out")[1] = True
        teacher.terrain_levels(env, torch.tensor([1]))
        self.assertEqual(env._benchmark_teacher["families"]["Uneven"]["block"], [])

    def test_changed_support_and_missing_ray_are_conservative(self):
        env = curriculum_environment()
        ep = env._benchmark_teacher["episodes"]
        ids = torch.tensor([1])
        height = torch.zeros(1, 4)
        valid = torch.ones(1, 4, dtype=torch.bool)
        with patch(
            "parkour_lab.runtime.native.foot_support",
            side_effect=lambda e, f: (height.clone(), valid.clone()),
        ):
            self.assertFalse(teacher._contact_samples(env, ids, ep).item())
            height[0, 0] = 0.001
            self.assertTrue(teacher._contact_samples(env, ids, ep).item())
            valid.zero_()
            self.assertFalse(teacher._contact_samples(env, ids, ep).item())

    def test_json_restore_keeps_blocks_history_and_sampler_but_discards_partial_episode(
        self,
    ):
        source = curriculum_environment()
        source.interactions.sync(6, 24)
        state = source._benchmark_teacher["families"]["Uneven"]
        state["block"] = [True] * 19
        episode(source)
        source._benchmark_teacher["episodes"]["steps"][2] = 30
        record = json.loads(json.dumps(checkpoint(source)))
        target = curriculum_environment()
        # A fresh SDK process has manager buffers but no step-created aliases.
        del target.reset_terminated, target.reset_time_outs
        restore(target, record)
        self.assertEqual(teacher.curriculum_state(target), record["curriculum"])
        self.assertFalse(target._benchmark_teacher["episodes"]["steps"].any())
        self.assertEqual(
            target.parkour_randomization.state_dict(),
            source.parkour_randomization.state_dict(),
        )
        self.assertEqual(target.interactions.cumulative, 48)
        env_state = teacher.curriculum_state(target)
        teacher.terrain_levels(target, torch.arange(8))
        self.assertEqual(
            target.scene.terrain.terrain_levels.tolist(), [0, 1, 0, 0, 0, 1, 1, 1]
        )
        self.assertEqual(
            teacher.curriculum_state(target)["families"], env_state["families"]
        )

    def test_old_schema_and_late_corruptions_reject_without_partial_restore(self):
        source = curriculum_environment()
        record = checkpoint(source)
        variants = []
        for mutate in (
            lambda r: r.update(version="go2_checkpoint_progress_v1"),
            lambda r: r["curriculum"].update(
                version="go2_eligible_distance_curriculum_v1"
            ),
            lambda r: r["curriculum"]["levels"].__setitem__(1, 16),
            lambda r: r["curriculum"]["types"].__setitem__(1, True),
            lambda r: r["curriculum"]["families"]["Uneven"].update(target_level=1),
            lambda r: r["curriculum"]["families"]["Uneven"].update(block=[True] * 20),
            lambda r: r["sampling_report"].update(period_steps=4),
        ):
            bad = deepcopy(record)
            mutate(bad)
            variants.append(bad)
        for bad in variants:
            env = curriculum_environment()
            before = teacher.curriculum_state(env)
            sampler = deepcopy(env.parkour_randomization.state_dict())
            with self.subTest(record=bad), self.assertRaises(ValueError):
                restore(env, bad)
            self.assertEqual(teacher.curriculum_state(env), before)
            self.assertEqual(env.parkour_randomization.state_dict(), sampler)
            self.assertEqual(env.interactions.cumulative, 0)
