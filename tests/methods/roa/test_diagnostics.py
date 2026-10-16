"""Opt-in numerical evidence without changing ROA acquisition or learning."""

import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from parkour_lab import experiment
from parkour_lab.config import ExperimentConfig, MethodConfig, TaskConfig
from parkour_lab.methods.roa import Learner
from parkour_lab.methods.roa import model as roa
from parkour_lab.methods.roa.diagnostics import NumericalCapture
from parkour_lab.methods.roa.training import ROATrainingMethod
from tests.support.native import TrainingEnvironmentFixture
from tests.support.roa import TensorHost, observations


class RecordingHost(TensorHost):
    def __init__(self, obs):
        super().__init__(obs)
        self.actions = []
        self.host = self
        self.steps = 0
        self.previous = torch.zeros(len(obs), 12)
        self.bridge = SimpleNamespace(default=torch.ones(len(obs), 12))
        self.env.step_dt = 0.02
        self.env.reward_manager = SimpleNamespace(
            active_terms=["action_rate_l2"],
            _step_reward=torch.zeros(len(obs), 1),
        )

    def step(self, action, phase):
        self.actions.append(action.detach().clone())
        result = super().step(action, phase)
        self.env.reward_manager._step_reward[:, 0] = -0.01 * (
            action - self.previous
        ).square().sum(-1)
        self.previous = action.detach().clone()
        self.steps += 1
        return result


class NumericalCaptureTests(unittest.TestCase):
    def setUp(self):
        fork = torch.random.fork_rng(devices=[])
        fork.__enter__()
        self.addCleanup(fork.__exit__, None, None, None)
        torch.manual_seed(843)
        self.obs = observations(8)
        self.obs["history"].reshape(8, 25, 49)[:, :, 33:45] = 0
        self.obs["policy"][:, 33:45] = 0
        with contextlib.redirect_stdout(io.StringIO()):
            self.policy, _ = roa.build_policy(self.obs)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)

    def method(self):
        return ROATrainingMethod(
            copy.deepcopy(self.policy),
            self.obs.clone(),
            num_envs=8,
            rollout_steps=2,
            history_steps=2,
            history_interval=1,
            adaptation_epochs=1,
            adaptation_batches=2,
            adaptation_learning_rate=0.001,
            num_learning_epochs=1,
            num_mini_batches=2,
            learning_rate=0.0002,
            device="cpu",
        )

    def capture(self, method):
        capture = NumericalCapture(self.output, method, {})
        capture.__enter__()
        self.addCleanup(capture.close)
        return capture

    def assert_learning_state_equal(self, expected, actual):
        torch.testing.assert_close(
            expected.state_dict(), actual.state_dict(), rtol=0, atol=0
        )

    def test_normal_step_returns_native_result_without_writes_or_mutation(self):
        method = self.method()
        host = RecordingHost(self.obs)
        before = copy.deepcopy(method)
        frame = self.obs.clone()
        action = torch.full((8, 12), 0.25)
        result = (self.obs.clone(), torch.ones(8), torch.zeros(8).bool(), {})
        rng = torch.get_rng_state().clone()
        capture = self.capture(method)
        with patch.object(host, "step", return_value=result) as step:
            self.assertIs(capture.step(host, frame, action, "privileged_ppo"), result)
        step.assert_called_once()
        self.assertIs(step.call_args.args[0], action)
        self.assertEqual(step.call_args.args[1], "privileged_ppo")
        self.assertTrue(torch.equal(action, torch.full((8, 12), 0.25)))
        self.assertTrue(torch.equal(frame["policy"], self.obs["policy"]))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assert_learning_state_equal(before, method)
        self.assertFalse(list(capture.output.iterdir()))

    def test_real_ppo_and_history_updates_are_identical_with_capture_enabled(self):
        baseline, diagnostic = self.method(), self.method()
        plain_host, captured_host = RecordingHost(self.obs), RecordingHost(self.obs)
        before = torch.get_rng_state().clone()
        plain_metrics = Learner(baseline, plain_host, self.obs.clone()).advance()
        after = torch.get_rng_state().clone()
        torch.set_rng_state(before)
        capture = self.capture(diagnostic)
        captured_metrics = Learner(
            diagnostic, captured_host, self.obs.clone()
        ).advance()
        self.assertTrue(torch.equal(torch.get_rng_state(), after))
        self.assertEqual(plain_metrics, captured_metrics)
        self.assertEqual(plain_host.phases, captured_host.phases)
        for plain, captured in zip(
            plain_host.actions, captured_host.actions, strict=True
        ):
            self.assertTrue(torch.equal(plain, captured))
        self.assert_learning_state_equal(baseline, diagnostic)
        self.assertFalse(list(capture.output.iterdir()))
        capture.close()
        self.assertIsNone(diagnostic.diagnostic)
        for module in (
            diagnostic.policy.actor.motor,
            diagnostic.policy.actor.estimator,
        ):
            self.assertFalse(module._forward_hooks)

    def test_first_excursion_captures_detached_transition_and_actual_forwards(self):
        for phase in ("privileged_ppo", "history_adaptation"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                method, host = self.method(), RecordingHost(self.obs)
                obs = self.obs.clone()
                with NumericalCapture(directory, method, {}) as capture:
                    with torch.no_grad():
                        normal = method.policy.actor.history_action(
                            obs["policy"], obs["history"].reshape(8, 25, 49)
                        )
                    capture.step(host, obs, normal, phase)
                    with torch.no_grad():
                        method.policy.actor.motor.tail[-1].weight.zero_()
                        method.policy.actor.motor.tail[-1].bias.fill_(201)
                        action = (
                            method.algorithm.act(obs)
                            if phase == "privileged_ppo"
                            else method.policy.actor.history_action(
                                obs["policy"], obs["history"].reshape(8, 25, 49)
                            )
                        )
                    expected_forward = {
                        name: value["output"].clone()
                        for name, value in capture.latest_forward.items()
                    }
                    expected_frame, expected_action = (
                        obs["policy"].clone(),
                        action.clone(),
                    )
                    expected_history = obs["history"].clone()
                    capture.step(host, obs, action, phase)
                    self.assertTrue(capture.active)
                    obs["policy"].fill_(-999)
                    obs["history"].fill_(-999)
                    action.zero_()
                    self.assertTrue(
                        torch.equal(
                            capture.previous_transition["observations"]["history"],
                            expected_history,
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            capture.previous_transition["raw_action"], expected_action
                        )
                    )
                    event = torch.load(
                        capture.output / "first_transition.pt", weights_only=True
                    )
                    after = torch.load(
                        capture.output / "first_transition_after.pt", weights_only=True
                    )
                    self.assertTrue(torch.equal(event["trigger_rows"], torch.arange(8)))
                    current = event["current"]
                    self.assertEqual(current["phase"], phase)
                    self.assertEqual(current["control_step"], 1)
                    self.assertTrue(torch.equal(current["raw_action"], expected_action))
                    self.assertTrue(
                        torch.equal(current["observations"]["policy"], expected_frame)
                    )
                    self.assertTrue(
                        torch.equal(event["previous"]["raw_action"], normal)
                    )
                    self.assertTrue(
                        torch.equal(
                            current["requested_position_rad"],
                            host.bridge.default + 0.25 * expected_action,
                        )
                    )
                    for name, expected in expected_forward.items():
                        self.assertTrue(
                            torch.equal(current["forward"][name]["output"], expected)
                        )
                    self.assertTrue(
                        torch.equal(after["encoded_raw_action"], expected_action)
                    )
                    self.assertTrue(
                        torch.equal(
                            after["reward"], 1 - expected_action.square().mean(-1)
                        )
                    )
                    capture.step(host, self.obs, torch.full((8, 12), 500.0), phase)
                    self.assertEqual(
                        len(list(capture.output.glob("first_transition*.pt"))), 2
                    )

    def test_delta_alarm_detects_extreme_change_below_raw_action_threshold(self):
        method, host = self.method(), RecordingHost(self.obs)
        capture = self.capture(method)
        action = torch.zeros(8, 12)
        action[3] = 50
        capture.step(host, self.obs, action, "history_adaptation")
        event = torch.load(capture.output / "first_transition.pt", weights_only=True)
        self.assertTrue(torch.equal(event["trigger_rows"], torch.tensor([3])))
        self.assertEqual(float(event["current"]["action_delta_l2"][3]), 30000)

    def test_pre_transition_survives_native_step_exception(self):
        method, host = self.method(), RecordingHost(self.obs)
        error = RuntimeError("native step failed")
        with self.assertRaises(RuntimeError) as caught:
            with NumericalCapture(self.output, method, {}) as capture:
                with patch.object(host, "step", side_effect=error):
                    capture.step(
                        host, self.obs, torch.full((8, 12), 101.0), "privileged_ppo"
                    )
        self.assertIs(caught.exception, error)
        event = torch.load(capture.output / "first_transition.pt", weights_only=True)
        self.assertEqual(event["current"]["control_step"], 0)
        self.assertFalse((capture.output / "first_transition_after.pt").exists())
        self.assertNotIn("capture_error", capture.report)
        saved = torch.load(capture.output / "exception.pt", weights_only=True)
        self.assertEqual(saved["error"], repr(error))

    def test_ppo_exception_keeps_exact_batch_and_nonfinite_gradients(self):
        for gradient_value in (float("nan"), float("inf")):
            with (
                self.subTest(gradient=gradient_value),
                tempfile.TemporaryDirectory() as directory,
            ):
                method, host = self.method(), RecordingHost(self.obs)
                name, parameter = next(
                    (name, parameter)
                    for name, parameter in method.policy.named_parameters()
                    if name.startswith("critic.")
                )
                hook = parameter.register_hook(
                    lambda grad: torch.full_like(grad, gradient_value)
                )
                self.addCleanup(hook.remove)
                original = None
                with self.assertRaisesRegex(RuntimeError, "non-finite") as caught:
                    with NumericalCapture(directory, method, {}) as capture:
                        obs = method.collect(host, self.obs.clone())
                        try:
                            method.update(obs)
                        except RuntimeError as error:
                            original = error
                            raise
                self.assertIs(caught.exception, original)
                saved = torch.load(capture.output / "exception.pt", weights_only=True)
                self.assertEqual(saved["error"], repr(original))
                self.assertEqual(saved["rollout"]["filled_steps"], 2)
                self.assertTrue(
                    torch.equal(
                        saved["rollout"]["actions"], method.algorithm.storage.actions
                    )
                )
                torch.testing.assert_close(
                    saved["gradients"][name],
                    parameter.grad,
                    rtol=0,
                    atol=0,
                    equal_nan=True,
                )
                self.assertTrue(torch.equal(saved["policy_state"][name], parameter))
                norm = saved["gradient_norms_float64"][name]
                self.assertEqual(norm.dtype, torch.float64)
                if gradient_value == float("inf"):
                    self.assertTrue(torch.isinf(saved["gradients"][name]).all())
                    self.assertTrue(torch.isinf(norm))
                else:
                    self.assertTrue(torch.isnan(saved["gradients"][name]).all())
                    self.assertTrue(torch.isnan(norm))
                failed = next(
                    frame["tensors"]
                    for frame in saved["frames"]
                    if frame["file"].endswith("/training.py")
                    and "ratio" in frame["tensors"]
                )
                self.assertTrue(torch.isfinite(failed["loss"]))
                self.assertEqual(failed["actions"].shape, (8, 12))
                self.assertEqual(failed["obs"]["policy"].shape, (8, 49))
                self.assertEqual(failed["returns"].shape, (8, 1))

    def test_capture_write_error_does_not_replace_learning_exception(self):
        method = self.method()
        original = RuntimeError("original learner failure")
        report = {}
        capture = NumericalCapture(self.output, method, report)
        with self.assertRaises(RuntimeError) as caught:
            with patch.object(capture, "save", side_effect=OSError("disk full")):
                with capture:
                    raise original
        self.assertIs(caught.exception, original)
        self.assertIn("disk full", report["numerical_diagnostic"]["capture_error"])
        self.assertIsNone(method.diagnostic)

    def test_history_event_captures_next_ppo_once_before_storage_is_cleared(self):
        method, host = self.method(), RecordingHost(self.obs)
        learner = Learner(method, host, self.obs.clone())
        original = method.policy.actor.history_action
        capture = self.capture(method)
        with patch.object(
            method.policy.actor,
            "history_action",
            side_effect=lambda frame, history: original(frame, history) + 50,
        ):
            learner.advance()
        self.assertEqual(capture.report["event_phase"], "history_adaptation")
        self.assertTrue(capture.active)
        self.assertFalse((capture.output / "first_rollout.pt").exists())
        learner.advance()
        self.assertFalse(capture.active)
        saved = torch.load(capture.output / "first_rollout.pt", weights_only=True)
        self.assertEqual(saved["method_updates"], 1)
        self.assertEqual(saved["filled_steps"], 2)
        self.assertEqual(saved["actions"].shape, (2, 8, 12))
        self.assertEqual(method.algorithm.storage.step, 0)
        self.assertFalse(
            torch.equal(saved["actions"], torch.zeros_like(saved["actions"]))
        )
        self.assertEqual(
            capture.report["files"].count("numerical_diagnostic/first_rollout.pt"), 1
        )

    def test_experiment_flag_reaches_real_hooks_and_preserves_checkpoint_cadence(self):
        config = ExperimentConfig(
            task=TaskConfig(num_envs=8, device="cpu", seed=97),
            method=MethodConfig(
                options=dict(
                    rollout_steps=2,
                    history_steps=2,
                    history_interval=2,
                    adaptation_epochs=1,
                    adaptation_batches=2,
                    num_learning_epochs=1,
                    num_mini_batches=2,
                    contact_conditioned=False,
                )
            ),
            updates=2,
            save_interval=2,
        )
        native = TrainingEnvironmentFixture()
        native.env.cfg.seed = config.task.seed
        report, captures = {}, []
        original = Learner.diagnostics

        def record(learner, output, report):
            capture = original(learner, output, report)
            captures.append(capture)
            return capture

        with patch.object(Learner, "diagnostics", autospec=True, side_effect=record):
            with contextlib.redirect_stdout(io.StringIO()):
                experiment.train(
                    native.env,
                    native.app,
                    config,
                    self.output,
                    report,
                    diagnose_training=True,
                )
        self.assertEqual(len(captures), 1)
        capture = captures[0]
        self.assertEqual(report["status"], "TRAINING_COMPLETE_NOT_QUALIFIED")
        self.assertFalse(report["numerical_diagnostic"]["event_captured"])
        self.assertFalse(report["numerical_diagnostic"]["files"])
        self.assertEqual(set(capture.latest_forward), {"estimator", "motor"})
        self.assertEqual(capture.previous_transition["phase"], "history_adaptation")
        self.assertIsNone(capture.method.diagnostic)
        self.assertFalse(list(capture.output.iterdir()))
        self.assertEqual(
            [path.name for path in self.output.glob("checkpoint_*.plab")],
            ["checkpoint_000002.plab"],
        )


if __name__ == "__main__":
    unittest.main()
