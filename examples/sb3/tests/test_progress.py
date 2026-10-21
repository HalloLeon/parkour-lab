"""Console visibility observes cached state without perturbing real learners."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from examples.sb3 import progress, snapshots, training
from examples.sb3.state import capture_rng, state_sha256
from examples.sb3.tests.fixtures import FakeTask


class ProgressTests(unittest.TestCase):
    def test_callback_throttles_and_reads_only_existing_caches(self):
        task = FakeTask()
        sampling = dict(samples=20, translations=18, stops=2, pivots=0, arcs=0)
        task._env = NS(
            command_manager=NS(get_term=lambda name: NS(sampling_report=sampling))
        )
        model = NS(
            num_timesteps=80,
            _n_updates=3,
            ep_info_buffer=[dict(r=4.0, l=8), dict(r=6.0, l=10)],
            logger=NS(name_to_value={"train/loss": 0.25, "train/bad": float("nan")}),
            get_env=lambda: task,
        )
        teacher = dict(
            families={"Uneven": dict(target_level=0, block_size=2, successes=0)}
        )
        before = state_sha256(capture_rng())
        with (
            patch.object(progress, "perf_counter", side_effect=[0, 5, 10, 10, 15]),
            patch.object(progress, "is_enabled", return_value=True),
            patch.object(progress, "log_event") as emit,
            patch(
                "parkour_lab.environments.terrains.curriculum.curriculum_diagnostics",
                return_value=teacher,
            ) as diagnostics,
        ):
            callback = progress.TrainingProgress(task, algorithm="ppo", target=512)
            callback.init_callback(model)
            task.interactions.update(session=80, cumulative=80)
            self.assertTrue(callback.on_step())
            self.assertTrue(callback.on_step())
            self.assertTrue(callback.on_step())
        self.assertEqual(emit.call_count, 1)
        event = emit.call_args
        self.assertEqual(event.args, ("training_progress",))
        self.assertEqual(event.kwargs["interactions_per_s"], 8)
        self.assertEqual(event.kwargs["recent_episode_return"], 5)
        self.assertEqual(event.kwargs["recent_episode_length_steps"], 9)
        self.assertEqual(event.kwargs["curriculum"], teacher)
        self.assertEqual(event.kwargs["command_samples"], sampling)
        self.assertIsNone(event.kwargs["learner"]["bad"])
        diagnostics.assert_called_once_with(task._env)
        self.assertEqual(task.reset_calls, 0)
        self.assertEqual(state_sha256(capture_rng()), before)

    def test_disabled_callback_does_not_read_diagnostics(self):
        task = FakeTask()
        model = NS(num_timesteps=0, get_env=lambda: task)
        callback = progress.TrainingProgress(task, algorithm="sac", target=512)
        callback.init_callback(model)
        with (
            patch.object(progress, "is_enabled", return_value=False),
            patch.object(
                progress,
                "_cached_diagnostics",
                side_effect=AssertionError("unexpected cache read"),
            ),
            patch.object(progress, "log_event") as emit,
        ):
            callback.emit("training_started", model)
            self.assertTrue(callback.on_step())
        emit.assert_not_called()

    def test_enabled_progress_preserves_optimizer_replay_rng_and_live_episodes(self):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for algorithm in ("ppo", "sac"):
                with (
                    self.subTest(algorithm=algorithm),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    root = Path(tmp)
                    reference_task = FakeTask(horizon=10000)
                    reference = training.run_training(
                        reference_task,
                        algorithm=algorithm,
                        total_interactions=512,
                        output=root / "reference",
                        milestones=(256,),
                    )
                    observed_task = FakeTask(horizon=10000)
                    clock = iter(range(0, 100000, 5))
                    with (
                        patch.object(training, "is_enabled", return_value=True),
                        patch.object(progress, "is_enabled", return_value=True),
                        patch.object(
                            progress, "perf_counter", side_effect=lambda: next(clock)
                        ),
                        patch.object(progress, "log_event") as events,
                        patch.object(snapshots, "log_event") as saves,
                        redirect_stdout(io.StringIO()) as console,
                    ):
                        observed = training.run_training(
                            observed_task,
                            algorithm=algorithm,
                            total_interactions=512,
                            output=root / "observed",
                            milestones=(256,),
                        )
                    self.assertIn("n_updates", console.getvalue())
                    names = [call.args[0] for call in events.call_args_list]
                    self.assertEqual(names[0], "training_started")
                    self.assertEqual(names[-1], "training_completed")
                    self.assertIn("training_progress", names)
                    self.assertEqual(saves.call_count, 2)
                    for key in (
                        "model_state_sha256",
                        "rng_state_sha256",
                        "replay_state_sha256",
                    ):
                        self.assertEqual(
                            observed["checkpoint"][key], reference["checkpoint"][key]
                        )
                    torch.testing.assert_close(
                        observed_task.state, reference_task.state, rtol=0, atol=0
                    )
                    self.assertEqual(
                        observed_task.interactions, reference_task.interactions
                    )
                    self.assertEqual(observed_task.reset_calls, 1)
        finally:
            torch.set_num_threads(threads)


if __name__ == "__main__":
    unittest.main()
