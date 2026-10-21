"""CLI role boundaries and ordered native cleanup without launching a simulator."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch  # noqa: F401 - load before patch.dict removes newly imported modules

from examples.sb3 import __main__ as cli


def module(name, **values):
    result = types.ModuleType(name)
    result.__dict__.update(values)
    return result


class CliTests(unittest.TestCase):
    def test_flat_entry_uses_exact_existing_controls_and_is_diagnostic_only(self):
        from parkour_lab.evaluation import case_bank

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            summary = {
                "command_contrast": {"forward_minus_stand_travel_m": 0.0},
                "cases": {},
            }
            with (
                self.native(root) as native,
                patch(
                    "parkour_lab.evaluation.summarize_flat_entry", return_value=summary
                ) as summarize,
            ):
                self.assertEqual(
                    cli.main(
                        [
                            "evaluate",
                            "--suite",
                            "flat-entry",
                            "--checkpoint",
                            str(checkpoint),
                            "--output",
                            str(root / "flat"),
                        ]
                    ),
                    0,
                )
                selected = native.evaluate.call_args.args[1]
                self.assertEqual(
                    selected,
                    tuple(
                        case
                        for case in case_bank("validation")
                        if case.family == "Flat"
                        and case.profile in ("stand", "forward-035")
                    ),
                )
                self.assertEqual(len(selected), 2)
                self.assertEqual(selected[0].seed, selected[1].seed)
                summarize.assert_called_once_with((root / "flat" / "capture").resolve())
                native.make.assert_not_called()
                native.train.assert_not_called()
            report = json.loads((root / "flat" / "report.json").read_text())
            self.assertFalse(report["full_bank"])
            self.assertFalse(report["evaluation"]["valid_for_comparison"])
            self.assertTrue(report["diagnostic_only"])
            self.assertFalse(report["calibration_validated"])
            self.assertEqual(report["diagnostic_summary"], summary)

    def test_flat_entry_full_bank_conflict_fails_before_output_or_native(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            with (
                self.native(root) as native,
                contextlib.redirect_stderr(io.StringIO()) as stderr,
                self.assertRaises(SystemExit) as error,
            ):
                cli.main(
                    [
                        "evaluate",
                        "--suite",
                        "flat-entry",
                        "--full-bank",
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(root / "invalid"),
                    ]
                )
            self.assertEqual(error.exception.code, 2)
            self.assertIn("diagnostic only", stderr.getvalue())
            self.assertFalse((root / "invalid").exists())
            native.launcher.assert_not_called()

    def test_robustness_suite_collects_full_bank_from_one_frozen_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            with self.native(root) as native:
                native.evaluated["valid_for_comparison"] = True
                code = cli.main(
                    [
                        "evaluate",
                        "--suite",
                        "robustness",
                        "--checkpoint",
                        str(checkpoint),
                        "--output",
                        str(root / "robustness"),
                    ]
                )
                self.assertEqual(code, 0)
                self.assertEqual(len(native.evaluate.call_args.args[1]), 588)
                native.train.assert_not_called()
                native.load.assert_called_once_with(checkpoint, device="cpu")
            report = json.loads((root / "robustness" / "report.json").read_text())
            self.assertTrue(report["full_bank"])
            self.assertEqual(report["suite"], "robustness")

    @contextlib.contextmanager
    def native(self, root, *, training_result=None, training_error=None):
        closed = []
        app = types.SimpleNamespace(close=lambda: closed.append("application"))
        task = types.SimpleNamespace(
            metadata={"training": True, "task_id": "Go2-Mixed-v1"},
            interactions={"session": 2560, "cumulative": 2560, "uncertain": 0},
            close=lambda: closed.append("environment"),
            reset=Mock(side_effect=AssertionError("CLI reset before continuation")),
        )
        launcher = Mock(return_value=types.SimpleNamespace(app=app))
        cfg = types.SimpleNamespace(to_dict=lambda: {"seed": 1077})
        build = Mock(return_value=cfg)
        make = Mock(return_value=task)
        train = Mock(
            return_value=training_result or {"learner_updated": True},
            side_effect=training_error,
        )
        receipt = dict(
            algorithm="ppo",
            training_seed=1077,
            checkpoint_id="full-checkpoint-identity",
            checkpoint_sha256="a" * 64,
            training_interactions=5120,
        )
        policy = object()
        load = Mock(return_value=(policy, receipt))
        evaluated = {"capture_valid": True, "valid_for_comparison": False}
        evaluate = Mock(return_value=evaluated)
        score = Mock(return_value=evaluated)
        from parkour_lab.benchmark import Go2Config

        with (
            patch.dict(
                sys.modules,
                {
                    "isaaclab": module("isaaclab"),
                    "isaaclab.app": module("isaaclab.app", AppLauncher=launcher),
                    "parkour_lab.benchmark": module(
                        "parkour_lab.benchmark",
                        Go2Config=Go2Config,
                        build_config=build,
                        make=make,
                    ),
                    "examples.sb3.training": module(
                        "examples.sb3.training", run_training=train
                    ),
                    "examples.sb3.policy": module(
                        "examples.sb3.policy", load_policy=load
                    ),
                },
            ),
            patch.object(cli, "_dependencies", return_value={"checked": True}),
            patch.object(cli, "write_run_provenance"),
            patch.object(cli, "exit_native_process") as exit_process,
            patch("parkour_lab.evaluation.evaluate", evaluate),
            patch("parkour_lab.evaluation.score_capture", score),
        ):
            yield types.SimpleNamespace(**locals())

    def test_training_forwards_absolute_budget_and_resume_without_resetting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "previous"
            checkpoint.mkdir()
            output = root / "training"
            with self.native(root) as native:
                code = cli.main(
                    [
                        "train",
                        "--algorithm",
                        "sac",
                        "--output",
                        str(output),
                        "--resume",
                        str(checkpoint),
                        "--run-id",
                        "run-1077",
                        "--command-lesson",
                        "forward-stand",
                        "--save-every-updates",
                        "10",
                        "--milestones",
                        "1280",
                        "2560",
                    ]
                )
                self.assertEqual(code, 0)
                self.assertEqual(native.closed, ["environment", "application"])
                native.task.reset.assert_not_called()
                arguments = native.train.call_args.kwargs
                self.assertEqual(arguments["algorithm"], "sac")
                self.assertEqual(arguments["total_interactions"], 2560)
                self.assertEqual(arguments["resume"], checkpoint)
                self.assertEqual(arguments["learner_device"], "cpu")
                self.assertEqual(arguments["run_id"], "run-1077")
                self.assertEqual(arguments["save_every_updates"], 10)
                self.assertEqual(arguments["milestones"], [1280, 2560])
                self.assertIs(arguments["app"], native.app)
                self.assertEqual(native.build.call_args.kwargs, {"training": True})
                self.assertEqual(
                    native.build.call_args.args[0].command_lesson, "forward-stand"
                )
                native.exit_process.assert_called_once_with(0)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "LEARNER_CHECK_PASSED")
            self.assertTrue(report["learner_updated"])
            self.assertEqual(
                report["cleanup"], dict(environment="complete", application="complete")
            )
            self.assertTrue((output / "resolved_env.yaml").exists())

    def test_training_failure_remains_durable_and_closes_environment_before_app(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "failed"
            with (
                self.native(
                    root, training_error=RuntimeError("checkpoint write failed")
                ) as native,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
                contextlib.redirect_stderr(io.StringIO()) as stderr,
            ):
                self.assertEqual(
                    cli.main(["train", "--algorithm", "ppo", "--output", str(output)]),
                    2,
                )
                self.assertEqual(native.closed, ["environment", "application"])
                native.exit_process.assert_called_once_with(2)
                self.assertIn("[parkour_lab] start", stdout.getvalue())
                self.assertIn("[parkour_lab] ready", stdout.getvalue())
                self.assertIn("[parkour_lab] error", stdout.getvalue())
                self.assertIn("[parkour_lab] finish", stdout.getvalue())
                self.assertIn("Traceback (most recent call last)", stderr.getvalue())
                self.assertIn("checkpoint write failed", stderr.getvalue())
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "ERROR")
            self.assertEqual(report["error"], "checkpoint write failed")
            self.assertIn("RuntimeError", report["traceback"])
            self.assertEqual(
                report["interactions"],
                {"session": 2560, "cumulative": 2560, "uncertain": 0},
            )

    def test_evaluation_loads_policy_without_creating_a_training_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            output = root / "evaluation"
            with self.native(root) as native:
                self.assertEqual(
                    cli.main(
                        [
                            "evaluate",
                            "--checkpoint",
                            str(checkpoint),
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
                native.load.assert_called_once_with(checkpoint, device="cpu")
                native.make.assert_not_called()
                native.train.assert_not_called()
                args, kwargs = native.evaluate.call_args
                self.assertIs(args[0], native.policy)
                self.assertEqual(len(args[1]), 6)
                self.assertEqual(
                    {case.family for case in args[1]},
                    {"Flat", "Uneven", "Slopes", "Stairs", "Hills"},
                )
                self.assertEqual(kwargs["training_seed"], 1077)
                self.assertEqual(
                    kwargs["checkpoint_id"], native.receipt["checkpoint_id"]
                )
                self.assertEqual(
                    kwargs["checkpoint_sha256"], native.receipt["checkpoint_sha256"]
                )
                self.assertEqual(native.closed, ["application"])
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "LEARNER_EVALUATION_PASSED")
            self.assertEqual(report["cleanup"]["environment"], "not_created")
            self.assertTrue(report["recomputed_score_parity"])

    def test_existing_output_is_not_overwritten_or_opened_in_native_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "report.json"
            marker.write_text("existing evidence")
            with self.native(root) as native, self.assertRaises(FileExistsError):
                cli.main(["train", "--algorithm", "ppo", "--output", str(root)])
            native.launcher.assert_not_called()
            self.assertEqual(marker.read_text(), "existing evidence")

    def test_invalid_budget_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                cli.main(
                    [
                        "train",
                        "--algorithm",
                        "ppo",
                        "--total-interactions",
                        "0",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_help_imports_neither_learning_library_nor_simulator(self):
        program = """
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.')
               for name in ('torch', 'stable_baselines3', 'isaaclab', 'isaacsim', 'omni', 'pxr')):
            raise AssertionError('Help imported ' + fullname)
sys.meta_path.insert(0, Guard())
sys.argv = ['examples.sb3', *sys.argv[1:]]
runpy.run_module('examples.sb3', run_name='__main__')
"""
        for command in ([], ["train"], ["evaluate"]):
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-c", program, *command, "--help"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--help", result.stdout)


if __name__ == "__main__":
    unittest.main()
