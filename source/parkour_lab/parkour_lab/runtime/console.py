"""CLI-enabled UTC progress messages; importing the library stays quiet."""

import json
import sys
from datetime import datetime, timezone

_stream = None


def setup_console(stream=None):
    """Enable flushed progress on the CLI stream and return its previous owner."""
    global _stream
    previous = _stream
    _stream = sys.stdout if stream is None else stream
    return previous


def restore_console(previous):
    """Restore the caller's console after an embedded CLI invocation."""
    global _stream
    _stream = previous


def is_enabled():
    return _stream is not None


def log_event(event, **fields):
    """Report existing state without allowing a broken log pipe to stop cleanup."""
    if _stream is None:
        return
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    detail = json.dumps(fields, sort_keys=True, default=str)
    try:
        print(f"{stamp} [parkour_lab] {event} {detail}", file=_stream, flush=True)
    except (OSError, ValueError):
        pass
