"""Offline CLI dispatch, output ownership and retirement of learner workflows."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from parkour_lab.__main__ import main, parse_args


class CLITests(unittest.TestCase):
    def test_robustness_prescriptions_are_available_without_native_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cases.json"
            main(["cases", "--suite", "robustness", "--output", str(output)])
            cases = json.loads(output.read_text())
            self.assertEqual(len(cases), 384)
            self.assertEqual(
                sum(case.get("condition") == "fault" for case in cases), 72
            )

    def test_help_only_exposes_offline_commands(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as stopped:
                main(["--help"])
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("{score,report,cases}", output.getvalue())
        for command in (
            "train",
            "evaluate",
            "play",
            "export",
            "bound-actions",
            "analyze",
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args([command])

    def test_score_dispatches_to_capture_scorer_without_loading_policy(self):
        module = ModuleType("parkour_lab.evaluation")
        module.score_capture = Mock(return_value={"capture": "checked", "score": 0.0})
        with patch.dict("sys.modules", {module.__name__: module}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["score", "capture-folder"]), 0)
        module.score_capture.assert_called_once_with(Path("capture-folder"))
        self.assertEqual(
            json.loads(output.getvalue()), {"capture": "checked", "score": 0.0}
        )

    def test_report_dispatches_offline_and_retains_incomplete_design(self):
        module = ModuleType("parkour_lab.evaluation")
        module.report_campaign = Mock(return_value={"data_complete": False})
        with patch.dict("sys.modules", {module.__name__: module}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(["report", "campaign.json"]), 0)
        module.report_campaign.assert_called_once_with(Path("campaign.json"))
        self.assertFalse(json.loads(output.getvalue())["data_complete"])

    def test_score_output_file_is_complete_json_and_scorer_errors_propagate(self):
        module = ModuleType("parkour_lab.evaluation")
        module.score_capture = Mock(return_value={"attempts": [1, 2]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "score.json"
            with patch.dict("sys.modules", {module.__name__: module}):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    main(["score", "capture", "--output", str(path)])
                self.assertEqual(output.getvalue(), "")
                self.assertEqual(json.loads(path.read_text()), {"attempts": [1, 2]})
                module.score_capture.side_effect = ValueError("invalid capture")
                with self.assertRaisesRegex(ValueError, "invalid capture"):
                    main(["score", "capture"])

    def test_cases_preserve_case_definitions_and_requested_partition(self):
        module = ModuleType("parkour_lab.evaluation.cases")
        definition = {"id": "turn_left", "command": [0, 0, 0.5]}
        module.case_bank = Mock(
            return_value=(SimpleNamespace(as_dict=lambda: definition),)
        )
        with patch.dict("sys.modules", {module.__name__: module}):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(["cases", "--partition", "validation", "--attempts", "2"])
        module.case_bank.assert_called_once_with(partition="validation", attempts=2)
        self.assertEqual(json.loads(output.getvalue()), [definition])
