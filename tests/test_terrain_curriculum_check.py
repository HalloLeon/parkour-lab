"""Portable execution of native-host teacher fixtures; no simulator claims."""

import unittest
from copy import deepcopy
from types import SimpleNamespace as NS

import torch

from parkour_lab.environments.terrains import curriculum
from scripts._terrain_checks.curriculum import check_boundary, check_curriculum
from tests.support.curriculum import curriculum_environment


class TaskFixture:
    def __init__(self):
        self._env = env = curriculum_environment()
        self.device, self.num_envs, self.training = "cpu", env.num_envs, True
        self.reset_calls = self.manager_calls = 0
        env.scene.env_origins = env.scene.terrain.env_origins.clone()
        robot = env.scene["robot"]
        robot.data.root_link_pos_w = robot.data.root_pos_w
        robot.data.root_link_pose_w = torch.cat(
            (robot.data.root_pos_w, robot.data.root_quat_w), -1
        )

        def write(pose):
            robot.data.root_pos_w.copy_(pose[:, :3])
            robot.data.root_quat_w.copy_(pose[:, 3:])
            robot.data.root_link_pose_w.copy_(pose)

        robot.write_root_pose_to_sim = write

        def compute(env_ids):
            self.manager_calls += 1
            return curriculum.terrain_levels(env, env_ids)

        env.curriculum_manager = NS(compute=compute)
        env.reset = self._reset

    @property
    def interactions(self):
        return self._env.interactions.state_dict()

    def _reset(self, *, seed=None, env_ids=None):
        assert seed is None
        env = self._env
        ids = torch.arange(env.num_envs) if env_ids is None else torch.tensor(env_ids)
        curriculum.terrain_levels(env, ids)
        env.episode_length_buf[ids] = 0
        env.reset_terminated[ids] = env.reset_time_outs[ids] = False
        pose = env.scene["robot"].data.root_link_pose_w.clone()
        pose[ids, :3] = env.scene.terrain.env_origins[ids]
        pose[ids, 2] += 0.4
        env.scene["robot"].write_root_pose_to_sim(pose)
        env.scene.env_origins.copy_(env.scene.terrain.env_origins)
        curriculum.begin_episodes(env, ids)
        self.reset_calls += 1

    def reset(self, *, env_ids=None):
        self._reset(env_ids=env_ids)

    def step(self, action):
        env = self._env
        env.episode_length_buf += 1
        curriculum.observe_step(env)
        delta = (
            env.scene["robot"].data.root_pos_w[:, :2]
            - env.scene.terrain.env_origins[:, :2]
        )
        truncated = (delta.abs() > 6).any(-1)
        ids = truncated.nonzero().flatten()
        env.reset_time_outs.copy_(truncated)
        env.termination_manager.get_term("terrain_boundary").copy_(truncated)
        self._reset(env_ids=ids.tolist())
        env.interactions.sync(1, 4)
        return (
            None,
            None,
            torch.zeros(env.num_envs, dtype=torch.bool),
            truncated,
            {"final_env_ids": ids},
        )


class NativeTeacherHostTests(unittest.TestCase):
    def test_complete_rule_fixtures_restore_state_without_collecting(self):
        task = TaskFixture()
        env = task._env
        original = deepcopy(env._benchmark_teacher["families"])
        getter = env.termination_manager.get_term
        command = env.command_manager.get_term("base_velocity").command.clone()
        rng = torch.get_rng_state()
        result = check_curriculum(task)
        self.assertEqual(result["physics_steps"], 0)
        self.assertEqual(result["charged_interactions"], 0)
        self.assertEqual(task.interactions, dict(session=0, cumulative=0, uncertain=0))
        self.assertEqual(task.reset_calls, 1)
        self.assertEqual(task.manager_calls, 102)
        self.assertEqual(env._benchmark_teacher["families"], original)
        self.assertIs(env.termination_manager.get_term, getter)
        torch.testing.assert_close(
            env.command_manager.get_term("base_velocity").command, command
        )
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(
            {case["name"]: case["after"] for case in result["cases"]},
            dict(
                promote_after_full_block=4,
                demote=2,
                hold=3,
                bottom=0,
                top=15,
                flat_hold=0,
                changed_packet=0,
            ),
        )

    def test_injected_manager_failure_still_restores_and_starts_fresh(self):
        task = TaskFixture()
        env = task._env
        original = deepcopy(env._benchmark_teacher["families"])
        getter = env.termination_manager.get_term

        def fail(env_ids):
            raise RuntimeError("injected manager failure")

        env.curriculum_manager.compute = fail
        with self.assertRaisesRegex(RuntimeError, "injected manager failure"):
            check_curriculum(task)
        self.assertEqual(env._benchmark_teacher["families"], original)
        self.assertIs(env.termination_manager.get_term, getter)
        self.assertEqual(task.reset_calls, 1)
        self.assertEqual(task.interactions["cumulative"], 0)
        self.assertFalse(env.reset_terminated.any() or env.reset_time_outs.any())

    def test_boundary_monitors_once_and_applies_pending_family_target(self):
        task = TaskFixture()
        state = task._env._benchmark_teacher["families"]["Uneven"]
        state["target_level"] = 3
        state["revision"] = 3
        before = curriculum.observe_step
        result = check_boundary(task)
        self.assertIs(curriculum.observe_step, before)
        self.assertEqual(result["reset_level"], 3)
        self.assertEqual(result["outgoing_monitor"], dict(before=0, outgoing=1))
        self.assertEqual(task.interactions["session"], task.num_envs)
        self.assertEqual(task.reset_calls, 2)


if __name__ == "__main__":
    unittest.main()
