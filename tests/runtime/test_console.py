"""Library silence and progress output failures must preserve run behavior."""

import io
import unittest
from unittest.mock import Mock

from parkour_lab.runtime.console import (
    is_enabled,
    log_event,
    restore_console,
    setup_console,
)


class ConsoleTests(unittest.TestCase):
    def test_library_is_quiet_and_nested_cli_restores_its_stream(self):
        outer, inner = io.StringIO(), io.StringIO()
        previous = setup_console(outer)
        try:
            self.assertTrue(is_enabled())
            nested = setup_console(inner)
            log_event("checkpoint_saved", interactions=512000)
            restore_console(nested)
            log_event("training_complete", interactions=1024000)
            self.assertNotIn("checkpoint_saved", outer.getvalue())
            self.assertIn("training_complete", outer.getvalue())
            self.assertIn('"interactions": 512000', inner.getvalue())
            self.assertRegex(inner.getvalue(), r"^\d{4}-\d\d-\d\dT.*\+00:00")
            restore_console(None)
            self.assertFalse(is_enabled())
            log_event("not_emitted")
            self.assertNotIn("not_emitted", outer.getvalue() + inner.getvalue())
        finally:
            restore_console(previous)

    def test_progress_flushes_and_a_broken_stream_does_not_abort_the_run(self):
        stream = Mock()
        previous = setup_console(stream)
        try:
            log_event("ready")
            stream.flush.assert_called_once()
            for error in (BrokenPipeError("closed"), ValueError("closed stream")):
                stream.write.side_effect = error
                log_event("cleanup", environment="complete")
        finally:
            restore_console(previous)


if __name__ == "__main__":
    unittest.main()
