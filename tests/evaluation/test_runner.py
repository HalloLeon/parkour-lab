"""Loaded-policy playback, outgoing samples and independent offline reports."""

import json
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

from parkour_lab.evaluation import runner, score_capture
from parkour_lab.evaluation.cases import EvaluationCase


def _state(position=0.0):
    return {
        "position_w": torch.tensor([[position, 0.0, 0.4]]),
        "quaternion_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "linear_velocity_b": torch.zeros(1, 3),
        "angular_velocity_b": torch.zeros(1, 3),
        "foot_position_w": torch.tensor([[[0.0, 0.0, 0.04]] * 4]),
        "foot_net_forces_w": torch.tensor([[[0.0, 0.0, 10.0]] * 4]),
        "support_height_w": torch.zeros(1, 4),
        "support_valid": torch.ones(1, 4, dtype=torch.bool),
    }


@contextmanager
def _native_fixture(*, ending="terminated", fail=None, changed=False):
    env = NS(device="cpu", step_dt=0.02, scene=NS(env_origins=torch.zeros(1, 3)))
    env.index = 0
    live = _state()
    policy = NS(state_sha256=lambda: ("b" if changed and env.index else "a") * 64)
    host = Mock(last_sensor_noise=None)
    host.motor_verification = {"runtime": True}
    host.controller.spec.manifest.return_value = {"external": True}
    host.motor.encode.return_value = torch.zeros(1, 12)
    host.motor.progress.return_value = {"verified": True}
    if fail == "inference":
        host.act.side_effect = RuntimeError("inference failed")
    if fail == "verification":
        host.motor.verify_delivery.side_effect = RuntimeError("verification failed")

    def step(action):
        env.index += 1
        task.interactions.update(session=env.index, cumulative=env.index)
        done = env.index == 2
        terminated = torch.tensor([done and ending in ("terminated", "both")])
        truncated = torch.tensor([done and ending in ("truncated", "both")])
        outgoing = _state(float(env.index))
        if ending == "truncated":
            outgoing["ending_terrain_boundary"] = truncated.clone()
        # Mimic automatic reset overwriting the live row after the owned final.
        live["position_w"].fill_(-99 if done else env.index)
        return (
            {},
            torch.tensor([float(env.index)]),
            terminated,
            truncated,
            {"motion_state": outgoing},
        )

    env.step = step
    task = NS(
        _env=env,
        device="cpu",
        interactions={"session": 0, "cumulative": 0, "uncertain": 0},
        close=Mock(),
    )
    with ExitStack() as stack:
        for name, replacement in {
            "make_case_task": Mock(return_value=task),
            "reset_native": Mock(return_value=torch.ones(1, dtype=torch.bool)),
            "realization": Mock(
                return_value={"task": {"reward_recipe": {"core": True}}}
            ),
            "bind_policy": Mock(return_value=host),
            "check_fixed_terrain": Mock(),
        }.items():
            stack.enter_context(
                patch("parkour_lab.evaluation.environment." + name, replacement)
            )
        stack.enter_context(patch.object(runner, "motion_state", lambda env: live))
        stack.enter_context(
            patch.object(runner, "_native_dependencies", return_value={"fixture": True})
        )
        yield env, task, host, policy


class LoadedPolicyPlaybackTests(unittest.TestCase):
    def setUp(self):
        self.case = EvaluationCase("validation", "Flat", None, "plane", "stand")

    def test_progress_events_follow_complete_case_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            with (
                _native_fixture() as (_, task, _, policy),
                patch.object(runner, "log_event") as log,
            ):
                report = runner.evaluate(policy, [self.case], Path(temp) / "capture")
                self.assertEqual(
                    [call.args[0] for call in log.call_args_list],
                    [
                        "evaluation_start",
                        "evaluation_case_start",
                        "evaluation_case_finish",
                        "evaluation_finish",
                    ],
                )
                fields = log.call_args_list[1].kwargs
                self.assertEqual((fields["index"], fields["total"]), (1, 1))
                self.assertEqual(fields["case_id"], self.case.case_id)
                self.assertTrue(log.call_args.kwargs["capture_valid"])
                self.assertEqual(report["cases"][0]["core_return"], 3.0)
                task.close.assert_called_once()

    def test_progress_error_keeps_failure_and_final_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            with (
                _native_fixture(fail="inference") as (_, task, _, policy),
                patch.object(runner, "log_event") as log,
            ):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    runner.evaluate(policy, [self.case], Path(temp) / "capture")
                self.assertEqual(log.call_args_list[-2].args, ("evaluation_error",))
                self.assertEqual(
                    log.call_args_list[-2].kwargs["error"], "inference failed"
                )
                self.assertEqual(log.call_args.args, ("evaluation_finish",))
                self.assertEqual(log.call_args.kwargs["status"], "ERROR")
                self.assertFalse(log.call_args.kwargs["capture_valid"])
                task.close.assert_called_once()

    def test_original_attempt_keeps_terminal_reward_and_owned_outgoing_state(self):
        for ending in ("terminated", "truncated", "both"):
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp) / "capture"
                with _native_fixture(ending=ending) as (env, task, host, policy):
                    report = runner.evaluate(policy, [self.case], folder, device="cpu")
                    self.assertEqual(env.index, 2)
                    task.close.assert_called_once()
                    self.assertEqual(host.act.call_count, 2)
                    self.assertTrue(
                        host.act.call_args_list[0].kwargs["reset_mask"].all()
                    )
                    self.assertFalse(
                        host.act.call_args_list[1].kwargs["reset_mask"].any()
                    )
                result = report["cases"][0]
                self.assertEqual(result["observed_steps"], 2)
                self.assertEqual(result["core_return"], 3.0)
                self.assertEqual(result["physical_failure"], ending != "truncated")
                self.assertIsNone(result["phases"]["settled_motion"]["yaw_rmse_rad_s"])
                self.assertEqual(report, score_capture(folder))
                self.assertEqual(
                    report, json.loads((folder / "report.json").read_text())
                )
                self.assertTrue(report["capture_valid"])
                self.assertEqual(report["native_dependencies"], {"fixture": True})
                self.assertFalse(report["valid_for_comparison"])
                with np.load(folder / self.case.case_id / "trace.npz") as trace:
                    np.testing.assert_array_equal(
                        trace["root_position_w"][:, 0], [1, 2]
                    )
                    self.assertEqual(trace["reward"].sum(), 3)

    def test_inference_failure_preserves_unobserved_case_and_closes_context(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "capture"
            with _native_fixture(fail="inference") as (_, task, _, policy):
                with self.assertRaisesRegex(RuntimeError, "inference failed"):
                    runner.evaluate(policy, [self.case], folder, device="cpu")
                task.close.assert_called_once()
            report = score_capture(folder)
            self.assertEqual(report["capture_status"], "ERROR")
            self.assertFalse(report["valid_for_comparison"])
            self.assertEqual(report["aggregate"]["overall"]["prescribed_attempts"], 1)
            self.assertIsNone(report["aggregate"]["overall"]["core_return_mean"])

    def test_delivery_failure_retains_physical_transition_and_partial_return(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "capture"
            with _native_fixture(fail="verification") as (env, task, _, policy):
                with self.assertRaisesRegex(RuntimeError, "verification failed"):
                    runner.evaluate(policy, [self.case], folder, device="cpu")
                self.assertEqual(env.index, 1)
                task.close.assert_called_once()
            report = score_capture(folder)
            self.assertFalse(report["valid_for_comparison"])
            self.assertEqual(report["cases"][0]["observed_steps"], 1)
            self.assertIsNone(report["cases"][0]["core_return"])

    def test_changed_normalization_or_weights_fails_and_keeps_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "capture"
            with _native_fixture(changed=True) as (_, task, _, policy):
                with self.assertRaisesRegex(RuntimeError, "Policy parameters"):
                    runner.evaluate(policy, [self.case], folder, device="cpu")
                task.close.assert_called_once()
            report = score_capture(folder)
            self.assertFalse(report["valid_for_comparison"])
            self.assertEqual(report["final_policy_state_sha256"], "b" * 64)

    def test_interruption_keeps_unattempted_cases_in_prescribed_denominator(self):
        other = EvaluationCase("validation", "Flat", None, "plane", "arc-right")
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "capture"
            policy = NS(state_sha256=lambda: "a" * 64)
            with (
                patch.object(runner, "_run_case") as run,
                patch.object(runner, "_native_dependencies", return_value={}),
            ):
                report = runner.evaluate(
                    policy,
                    [self.case, other],
                    folder,
                    app=NS(is_running=lambda: False),
                )
                run.assert_not_called()
            overall = report["aggregate"]["overall"]
            self.assertEqual(overall["prescribed_attempts"], 2)
            self.assertEqual(overall["missing_attempts"], 2)
            self.assertEqual(report["capture_status"], "INTERRUPTED")

    def test_trace_digest_detects_later_rewrite(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "capture"
            with _native_fixture() as (_, _, _, policy):
                runner.evaluate(policy, [self.case], folder, device="cpu")
            path = folder / self.case.case_id / "trace.npz"
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "Trace digest"):
                score_capture(folder)

    def test_bad_case_selection_rejected_before_native_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "capture"
            with patch.object(runner, "_run_case") as run:
                with self.assertRaisesRegex(ValueError, "distinct"):
                    runner.evaluate(None, [self.case, self.case], folder)
                run.assert_not_called()
            self.assertFalse(folder.exists())

    def test_legacy_prescription_is_offline_only_before_output_or_policy_read(self):
        from dataclasses import replace

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "capture"
            with patch.object(runner, "_run_case") as run:
                with self.assertRaisesRegex(ValueError, "offline-only"):
                    runner.evaluate(None, [replace(self.case, version="v1")], folder)
                run.assert_not_called()
            self.assertFalse(folder.exists())
