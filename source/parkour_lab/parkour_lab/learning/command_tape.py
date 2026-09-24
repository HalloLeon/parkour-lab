"""Portable, finite body-twist sequences; no actions, sensing or backend code.

Ticks count completed control steps, not wall time. The digest detects accidental
changes; it is not a signature or evidence of physical tracking or safe stopping.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

VERSION = "operator_applied_command_tape_v1"
PERIOD_S = 0.02
MAX_STEPS = 30_000
MAX_BYTES = 16 * 1024 * 1024


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def seal(payload):
    """Copy a JSON payload and attach its canonical content digest."""
    payload = json.loads(_json(payload))
    payload.pop("sha256", None)
    return {**payload, "sha256": hashlib.sha256(_json(payload).encode()).hexdigest()}


def _command(value):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
    ):
        raise ValueError("Command must be finite body (vx m/s, vy m/s, yaw rad/s)")
    return list(value)


def validate_tape(tape, *, require_complete=True):
    """Validate a bounded, contiguous sequence before any simulator launch."""
    if not isinstance(tape, dict) or set(tape) != {
        "version",
        "period_s",
        "steps",
        "complete",
        "metadata",
        "segments",
        "error",
        "sha256",
    }:
        raise ValueError("Invalid command tape fields")
    if tape["version"] != VERSION or tape["period_s"] != PERIOD_S:
        raise ValueError("Unsupported command tape version or control period")
    if type(tape["steps"]) is not int or not 0 <= tape["steps"] <= MAX_STEPS:
        raise ValueError("Command tape exceeds the bounded control-step budget")
    if type(tape["complete"]) is not bool or not isinstance(tape["metadata"], dict):
        raise ValueError("Invalid command tape completion or metadata")
    if tape["error"] is not None and not isinstance(tape["error"], str):
        raise ValueError("Invalid command tape error")
    if not isinstance(tape["segments"], list) or len(tape["segments"]) > MAX_STEPS:
        raise ValueError("Invalid command tape segments")
    end = 0
    for segment in tape["segments"]:
        if not isinstance(segment, dict) or set(segment) != {
            "start_step",
            "end_step_exclusive",
            "command",
        }:
            raise ValueError("Invalid command segment fields")
        start, stop = segment["start_step"], segment["end_step_exclusive"]
        if (
            type(start) is not int
            or type(stop) is not int
            or start != end
            or not start < stop <= tape["steps"]
        ):
            raise ValueError("Command tape segments must be contiguous and nonempty")
        _command(segment["command"])
        end = stop
    if end != tape["steps"] or seal(tape)["sha256"] != tape["sha256"]:
        raise ValueError("Command tape length or digest mismatch")
    if tape["complete"] and (
        not end or tape["error"] is not None or any(tape["segments"][-1]["command"])
    ):
        raise ValueError(
            "Complete command tapes must end with an explicit zero command"
        )
    if require_complete and not tape["complete"]:
        raise ValueError("Incomplete command tapes cannot be replayed")
    return tape


def load_tape(path):
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate command tape field: {key}")
            result[key] = value
        return result

    with Path(path).open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError("Command tape exceeds size limit")
    return validate_tape(json.loads(data, object_pairs_hook=unique_fields))


def write_tape(path, tape):
    """Publish once, atomically; never overwrite an existing recording."""
    validate_tape(tape, require_complete=False)
    data = _json(tape).encode() + b"\n"
    if len(data) > MAX_BYTES:
        raise ValueError("Command tape exceeds size limit")
    path = Path(path)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".command_tape_", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.link(temporary, path)  # Fails if the destination already exists.
        finally:
            temporary.unlink(missing_ok=True)


class TapeBuilder:
    """Run-length encode only commands known to have completed native delivery."""

    def __init__(self, metadata):
        self.metadata = json.loads(_json(metadata))
        self.segments = []
        self.steps = 0

    def append(self, step, command):
        if type(step) is not int or step != self.steps or step >= MAX_STEPS:
            raise ValueError("Command recording steps must be consecutive and bounded")
        command = _command(command)
        if self.segments and self.segments[-1]["command"] == command:
            self.segments[-1]["end_step_exclusive"] = step + 1
        else:
            self.segments.append(
                dict(start_step=step, end_step_exclusive=step + 1, command=command)
            )
        self.steps += 1

    def finish(self, *, completed, error=None):
        if (
            completed
            and not error
            and (not self.steps or any(self.segments[-1]["command"]))
        ):
            error = "Recording must end with a delivered zero command; release motion before exit"
        return seal(
            dict(
                version=VERSION,
                period_s=PERIOD_S,
                steps=self.steps,
                complete=bool(completed and not error),
                metadata=self.metadata,
                segments=self.segments,
                error=error,
            )
        )
