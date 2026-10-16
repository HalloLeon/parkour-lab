"""Causal noise ownership and role-filtered public observation contracts."""

import unittest
from copy import deepcopy
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from parkour_lab.benchmark.observations import actor_frame, task_observation
from parkour_lab.control.proprioception import CAUSAL_NOISE_AMPLITUDES, FRAME_DIM
from parkour_lab.environments.randomization import TaskRandomization


def environment():
    task = dict(num_envs=3, seed=42, dynamics="nominal", terrain="flat")
    return NS(
        num_envs=3,
        device="cpu",
        cfg=NS(parkour_task=task),
        parkour_randomization=TaskRandomization(task),
    )


def clean_frame():
    frame = torch.arange(3 * FRAME_DIM, dtype=torch.float32).reshape(3, FRAME_DIM)
    frame[:, 45:49] = torch.tensor([1.0, 0.0, 1.0, 0.0])
    return frame.requires_grad_()


class PublicObservationTests(unittest.TestCase):
    def test_terminal_preview_preserves_stream_and_matches_next_ordinary_rows(self):
        env, clean = environment(), clean_frame()
        ids = torch.tensor([2, 0])
        sampler = env.parkour_randomization
        initial = deepcopy(sampler.state_dict())
        outgoing = clean[ids]
        saved = outgoing.detach().clone()
        final = actor_frame(env, outgoing, final_env_ids=ids)
        self.assertEqual(sampler.state_dict(), initial)
        self.assertFalse(hasattr(env, "_causal_noise_decisions"))
        self.assertFalse(final.requires_grad)
        self.assertIsNone(final.grad_fn)
        self.assertNotEqual(final.data_ptr(), outgoing.data_ptr())

        # The ordinary frame must consume exactly the draw that was previewed,
        # for every row, rather than advancing the selected rows twice.
        expected_sampler = environment().parkour_randomization
        expected_sampler.uniform(None, 30, "observation-noise", low=-1.0, high=1.0)
        ordinary = actor_frame(env, clean)
        torch.testing.assert_close(final, ordinary[ids], rtol=0, atol=0)
        self.assertEqual(sampler.state_dict(), expected_sampler.state_dict())
        self.assertEqual(env._causal_noise_decisions, 1)
        final.fill_(-999)
        torch.testing.assert_close(outgoing, saved, rtol=0, atol=0)

    def test_only_declared_sensor_slices_are_noised_and_storage_is_owned(self):
        env, clean = environment(), clean_frame()
        saved = clean.detach().clone()
        expected_sampler = environment().parkour_randomization
        noise = expected_sampler.uniform(
            None, 30, "observation-noise", low=-1.0, high=1.0
        )
        noise *= CAUSAL_NOISE_AMPLITUDES
        noise = torch.as_tensor(noise, dtype=torch.float32)
        expected = saved.clone()
        expected[:, :6] += noise[:, :6]
        expected[:, 9:33] += noise[:, 6:]

        frame = actor_frame(env, clean)
        torch.testing.assert_close(frame, expected, rtol=0, atol=0)
        for interval in (slice(6, 9), slice(33, 45), slice(45, 49)):
            torch.testing.assert_close(
                frame[:, interval], saved[:, interval], rtol=0, atol=0
            )
        self.assertFalse(frame.requires_grad)
        self.assertIsNone(frame.grad_fn)
        self.assertNotEqual(frame.data_ptr(), clean.data_ptr())
        previous = frame.clone()
        actor_frame(env, clean)
        torch.testing.assert_close(frame, previous, rtol=0, atol=0)
        frame.fill_(-999)
        torch.testing.assert_close(clean, saved, rtol=0, atol=0)

    def test_training_final_requires_outgoing_truth_before_reading_or_drawing(self):
        env, clean = environment(), clean_frame()
        before = deepcopy(env.parkour_randomization.state_dict())
        with self.assertRaisesRegex(RuntimeError, "outgoing training information"):
            task_observation(
                env,
                {"proprio": clean[:1]},
                training=True,
                final_env_ids=torch.tensor([0]),
            )
        self.assertEqual(env.parkour_randomization.state_dict(), before)
        self.assertFalse(hasattr(env, "_causal_noise_decisions"))

    def test_evaluation_never_extracts_or_forwards_privileged_groups(self):
        env, clean = environment(), clean_frame()
        native = {
            "proprio": clean,
            "policy": torch.full((3, 52), 123.0),
            "terrain": torch.full((3, 264), 456.0),
        }
        with patch(
            "parkour_lab.benchmark.information.capture_information",
            side_effect=AssertionError("Evaluation must not extract training truth"),
        ) as capture:
            observation = task_observation(
                env, native, training=False, information={"secret": object()}
            )
            self.assertEqual(set(observation), {"actor"})
            self.assertEqual(observation["actor"].shape, (3, FRAME_DIM))
            self.assertFalse(observation["actor"].requires_grad)

            ids = torch.tensor([1])
            outgoing = {name: value[ids] for name, value in native.items()}
            first = task_observation(
                env,
                outgoing,
                training=False,
                final_env_ids=ids,
                information={"secret": torch.tensor([999.0])},
            )
            second = task_observation(env, outgoing, training=False, final_env_ids=ids)
            self.assertEqual(set(first), {"actor"})
            torch.testing.assert_close(first["actor"], second["actor"], rtol=0, atol=0)
            self.assertFalse(first["actor"].requires_grad)
            capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
