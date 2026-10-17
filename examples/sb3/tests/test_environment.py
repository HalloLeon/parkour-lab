"""Real SB3 vector/replay contracts over a synthetic bounded continuous task."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3 import PPO, SAC

from examples.sb3.environment import Go2VecEnv
from examples.sb3.tests.fixtures import FakeTask


class EnvironmentTests(unittest.TestCase):
    def test_compact_outgoing_finals_and_physical_timeout_precedence(self):
        task = FakeTask()
        env = Go2VecEnv(task)
        initial = env.reset()
        self.assertEqual(initial.shape, (4, 49))
        self.assertFalse((initial == 123456).any())
        task.next_terminated = torch.tensor([False, False, True, True])
        task.next_truncated = torch.tensor([True, False, True, False])
        actions = np.full((4, 12), 0.25, dtype=np.float32)
        env.step_async(actions)
        actions.fill(99)
        observation, reward, done, infos = env.step_wait()
        self.assertEqual(done.tolist(), [True, False, True, True])
        self.assertEqual(
            [x["TimeLimit.truncated"] for x in infos], [True, False, False, False]
        )
        self.assertNotIn("terminal_observation", infos[1])
        for index in (0, 2, 3):
            self.assertEqual(infos[index]["terminal_observation"][12], 1)
            self.assertEqual(observation[index, 12], 0)
            self.assertEqual(infos[index]["episode"]["l"], 1)
        np.testing.assert_array_equal(task.last_actions.numpy(), 0.25)
        self.assertEqual(task.reset_calls, 1)
        self.assertEqual(env.interactions["cumulative"], 4)
        owned = (
            observation.copy(),
            reward.copy(),
            infos[0]["terminal_observation"].copy(),
        )
        env.step(np.zeros((4, 12), dtype=np.float32))
        np.testing.assert_array_equal(observation, owned[0])
        np.testing.assert_array_equal(reward, owned[1])
        np.testing.assert_array_equal(infos[0]["terminal_observation"], owned[2])
        env.close()

    def test_real_sac_collection_stores_terminal_frames_and_masks_only_timeouts(self):
        task = FakeTask()
        env = Go2VecEnv(task, total_interactions=4)
        model = SAC(
            "MlpPolicy",
            env,
            seed=42,
            device="cpu",
            learning_starts=100,
            buffer_size=32,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs={"net_arch": [8]},
        )
        task.next_terminated = torch.tensor([False, False, True, True])
        task.next_truncated = torch.tensor([True, False, True, False])
        model.learn(total_timesteps=4)
        replay = model.replay_buffer
        self.assertEqual(replay.pos, 1)
        np.testing.assert_array_equal(replay.dones[0], [1, 0, 1, 1])
        np.testing.assert_array_equal(replay.timeouts[0], [1, 0, 0, 0])
        np.testing.assert_array_equal(replay.next_observations[0, :, 12], [1, 1, 1, 1])
        # Sampling the actual SB3 buffer proves the bootstrap mask it delivers.
        for _ in range(5):
            sample = replay.sample(64)
            expected = (sample.next_observations[:, 48] >= 3).float()[:, None]
            torch.testing.assert_close(sample.dones, expected)
            self.assertTrue((sample.next_observations[:, 12] == 1).all())
        self.assertEqual(task.reset_calls, 1)
        self.assertEqual(model.num_timesteps, task.interactions["cumulative"])
        env.close()

    def test_budget_and_application_guards_precede_physics(self):
        task = FakeTask()
        env = Go2VecEnv(task, total_interactions=4)
        env.reset()
        env.step(np.zeros((4, 12), dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "interaction budget"):
            env.step(np.zeros((4, 12), dtype=np.float32))
        self.assertEqual(task.interactions["cumulative"], 4)
        stopped = FakeTask()
        other = Go2VecEnv(stopped, app=SimpleNamespace(is_running=lambda: False))
        other.reset()
        with self.assertRaisesRegex(RuntimeError, "application stopped"):
            other.step(np.zeros((4, 12), dtype=np.float32))
        self.assertEqual(stopped.interactions["cumulative"], 0)
        env.close()
        other.close()

    def test_real_ppo_bootstraps_timeout_from_outgoing_value_only(self):
        task = FakeTask()
        env = Go2VecEnv(task, total_interactions=8)
        model = PPO(
            "MlpPolicy",
            env,
            seed=42,
            device="cpu",
            n_steps=2,
            batch_size=8,
            n_epochs=1,
            gamma=0.9,
            policy_kwargs={"net_arch": [8]},
        )
        task.next_terminated = torch.tensor([False, False, True, True])
        task.next_truncated = torch.tensor([True, False, True, False])
        step, physical_rewards = task.step, []

        def record_reward(actions):
            transition = step(actions)
            physical_rewards.append(transition[1].numpy().copy())
            return transition

        task.step = record_reward
        # Terminal age=1 gives V=18; the replacement episode age=0 gives V=5.
        with patch.object(
            model.policy,
            "predict_values",
            side_effect=lambda obs: 13 * obs[:, 12:13] + 5,
        ) as values:
            model.learn(total_timesteps=8)
        expected = physical_rewards[0].copy()
        expected[0] += 0.9 * 18
        np.testing.assert_allclose(model.rollout_buffer.rewards[0], expected, atol=1e-6)
        self.assertEqual(values.call_args_list[0].args[0][0, 12], 1)
        self.assertEqual(task.reset_calls, 1)
        self.assertEqual(task.interactions["cumulative"], 8)
        self.assertGreater(model._n_updates, 0)
        env.close()

    def test_invalid_actions_do_not_collect_and_bad_finals_latch_after_collection(self):
        task = FakeTask()
        env = Go2VecEnv(task)
        env.reset()
        for actions in (np.zeros((3, 12)), np.full((4, 12), np.nan)):
            with self.assertRaises(ValueError):
                env.step_async(actions)
        self.assertEqual(task.interactions["cumulative"], 0)
        step = task.step

        def malformed(actions):
            result = step(actions)
            result[-1]["final_env_ids"] = torch.tensor([0])
            return result

        task.step = malformed
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            env.step(np.zeros((4, 12), dtype=np.float32))
        self.assertEqual(task.interactions["cumulative"], 4)
        with self.assertRaisesRegex(RuntimeError, "faulted"):
            env.step(np.zeros((4, 12), dtype=np.float32))
        env.close()

    def test_seed_identity_and_no_generic_native_escape(self):
        task = FakeTask()
        env = Go2VecEnv(task)
        before = torch.random.get_rng_state()
        self.assertEqual(env.seed(42), [42] * 4)
        self.assertEqual(env.seed(), [42] * 4)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for seed in (43, 42.0, True):
            with self.assertRaises(ValueError):
                env.seed(seed)
        self.assertEqual(env.get_attr("render_mode", indices=[0, 2]), [None, None])
        with self.assertRaises(AttributeError):
            env.get_attr("training")
        with self.assertRaises(NotImplementedError):
            env.env_method("reset")
        with self.assertRaises(NotImplementedError):
            env.set_attr("seed", 1)
        with self.assertRaises(NotImplementedError):
            env.set_options({"terrain": "different"})
        env.close()

    def test_reset_cancels_uncollected_actions_and_resume_budget_keeps_prior_spend(
        self,
    ):
        task = FakeTask()
        task.interactions["cumulative"] = 8
        env = Go2VecEnv(task, total_interactions=12)
        env.step_async(np.ones((4, 12), dtype=np.float32))
        env.reset()
        with self.assertRaisesRegex(RuntimeError, "No action"):
            env.step_wait()
        self.assertEqual(task.interactions["cumulative"], 8)
        env.step(np.zeros((4, 12), dtype=np.float32))
        self.assertEqual(task.interactions["cumulative"], 12)
        with self.assertRaisesRegex(RuntimeError, "interaction budget"):
            env.step(np.zeros((4, 12), dtype=np.float32))
        env.close()


if __name__ == "__main__":
    unittest.main()
