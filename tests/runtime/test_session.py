"""Cleanup order and durable receipts on success, failure and interrupted exit."""

import contextlib
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from parkour_lab.runtime import session


class NativeProcessExitTests(unittest.TestCase):
    def test_exit_preserves_status_flushes_logs_and_skips_late_finalizers(self):
        script = """
import atexit, os, sys
from parkour_lab.runtime.session import exit_native_process
atexit.register(lambda: os._exit(99))
print('final receipt', end='')
print('error details', end='', file=sys.stderr)
exit_native_process(int(sys.argv[1]))
"""
        for code in (0, 2, 7, 130):
            with self.subTest(code=code):
                result = subprocess.run(
                    [sys.executable, "-c", script, str(code)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                self.assertEqual(result.returncode, code)
                self.assertEqual(result.stdout, "final receipt")
                self.assertEqual(result.stderr, "error details")

    def test_failed_flush_is_nonzero_without_skipping_the_other_stream(self):
        for code in (0, 7):
            for error in (BrokenPipeError("closed pipe"), ValueError("closed stream")):
                with self.subTest(code=code, error=error):
                    stdout, stderr = Mock(), Mock()
                    stdout.flush.side_effect = error
                    with (
                        patch.object(session.sys, "stdout", stdout),
                        patch.object(session.sys, "stderr", stderr),
                        patch.object(session.os, "_exit") as terminate,
                    ):
                        session.exit_native_process(code)
                    terminate.assert_called_once_with(code or 2)
                    stdout.flush.assert_called_once_with()
                    stderr.flush.assert_called_once_with()


class SessionCleanupTests(unittest.TestCase):
    def report(self):
        return {
            "status": "TRAINING_COMPLETE_NOT_QUALIFIED",
            "exit_allowed": False,
            "behavior_validated": False,
        }

    def resources(self, order, fail=()):
        def resource(name):
            def close():
                order.append(name)
                if name in fail:
                    raise RuntimeError(name + " close failed")

            return NS(close=close)

        return resource("environment"), resource("application")

    def test_success_closes_in_order_and_publishes_pending_then_complete_receipts(self):
        report, order, saved = self.report(), [], []
        env, app = self.resources(order)
        result = session.finish_session(
            env, app, report, lambda: saved.append(copy.deepcopy(report)), 0
        )
        self.assertEqual(result, 0)
        self.assertEqual(order, ["environment", "application"])
        self.assertEqual(
            saved[0]["cleanup"], {"environment": "pending", "application": "pending"}
        )
        self.assertEqual(saved[0]["status"], "SESSION_COMPLETED_CLEANUP_PENDING")
        self.assertEqual(
            saved[1]["cleanup"], {"environment": "complete", "application": "pending"}
        )
        self.assertEqual(
            saved[-1]["cleanup"],
            {"environment": "complete", "application": "complete"},
        )
        self.assertEqual(saved[-1], report)
        self.assertEqual(report["status"], report["session_status"])
        self.assertFalse(report["exit_allowed"])
        self.assertFalse(report["behavior_validated"])

    def test_close_exceptions_are_isolated_and_never_skip_the_other_resource(self):
        for fail in (
            ("environment",),
            ("application",),
            ("environment", "application"),
        ):
            with self.subTest(fail=fail):
                report, order, saved = self.report(), [], []
                env, app = self.resources(order, fail)
                result = session.finish_session(
                    env, app, report, lambda: saved.append(copy.deepcopy(report)), 0
                )
                self.assertEqual(result, 2)
                self.assertEqual(order, ["environment", "application"])
                self.assertEqual(report["status"], "ERROR")
                self.assertEqual(report["session_status"], self.report()["status"])
                self.assertEqual(saved[-1], report)
                for name in ("environment", "application"):
                    expected = (
                        "ERROR: " + name + " close failed"
                        if name in fail
                        else "complete"
                    )
                    self.assertEqual(report["cleanup"][name], expected)

    def test_publish_failure_at_any_boundary_keeps_both_closes_and_error_status(self):
        for failure in (0, 1, 2, 3, "all"):
            with self.subTest(failure=failure):
                report, order, attempted = self.report(), [], []
                env, app = self.resources(order)

                def publish():
                    attempt = len(attempted)
                    attempted.append(copy.deepcopy(report))
                    if failure == "all" or attempt == failure:
                        raise OSError("report sink unavailable")

                with contextlib.redirect_stdout(io.StringIO()):
                    result = session.finish_session(env, app, report, publish, 0)
                self.assertEqual(result, 2)
                self.assertEqual(len(attempted), 4)
                self.assertEqual(order, ["environment", "application"])
                self.assertEqual(report["status"], "ERROR")
                self.assertEqual(
                    report["report_write_error"], "report sink unavailable"
                )
                self.assertEqual(
                    report["cleanup"],
                    {"environment": "complete", "application": "complete"},
                )

    def test_broken_logging_during_publish_failure_still_closes_both_resources(self):
        report, order = self.report(), []
        env, app = self.resources(order)

        def publish():
            raise OSError("report sink unavailable")

        with patch("builtins.print", side_effect=BrokenPipeError("log pipe closed")):
            result = session.finish_session(env, app, report, publish, 0)
        self.assertEqual(result, 2)
        self.assertEqual(order, ["environment", "application"])
        self.assertEqual(report["status"], "ERROR")
        self.assertEqual(report["report_write_error"], "report sink unavailable")

    def test_application_hard_exit_leaves_durable_pending_receipt_not_false_success(
        self,
    ):
        report, order = self.report(), []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"

            def publish():
                path.write_text(json.dumps(report, allow_nan=False))

            env = NS(close=lambda: order.append("environment"))

            def close_application():
                order.append("application")
                durable = json.loads(path.read_text())
                self.assertEqual(
                    durable["cleanup"],
                    {"environment": "complete", "application": "pending"},
                )
                self.assertEqual(durable["status"], "SESSION_COMPLETED_CLEANUP_PENDING")
                raise SystemExit(23)

            with self.assertRaises(SystemExit) as raised:
                session.finish_session(
                    env, NS(close=close_application), report, publish, 0
                )
            self.assertEqual(raised.exception.code, 23)
            self.assertEqual(order, ["environment", "application"])
            self.assertEqual(json.loads(path.read_text()), report)
            self.assertEqual(report["cleanup"]["application"], "pending")
            self.assertEqual(report["session_status"], self.report()["status"])

    def test_prelaunch_failure_records_uncreated_resources_without_fabricated_cleanup(
        self,
    ):
        report = self.report()
        report.update(status="ERROR", error="prelaunch failure")
        saved = []
        result = session.finish_session(
            None, None, report, lambda: saved.append(copy.deepcopy(report)), 2
        )
        self.assertEqual(result, 2)
        self.assertEqual(report["status"], "ERROR")
        self.assertEqual(report["session_status"], "ERROR")
        self.assertEqual(report["error"], "prelaunch failure")
        self.assertEqual(
            report["cleanup"],
            {"environment": "not_created", "application": "not_created"},
        )
        self.assertEqual(saved[-1], report)


if __name__ == "__main__":
    unittest.main()
