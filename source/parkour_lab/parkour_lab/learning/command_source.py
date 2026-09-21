"""Serialized simulator-host command authority with a local receipt-time lease.

The trusted host supplies one monotonic clock to every method. ``time_s`` is a
local receipt/decision time, never a remote sender's timestamp. This bounds how
long a received command survives silence; it does not authenticate a producer,
detect freshly numbered cached packets, or certify transport/hardware safety.
The lease neither observes robot state nor changes policy memory or motor actions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class LeaseDecision:
    """Applied body twist and last accepted source evidence in one local clock."""

    command: tuple[float, float, float]
    decision_time_s: float
    source_time_s: float | None
    sequence: int | None
    generation: int
    status: str


def _number(value):
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError("Expected a finite real number, not a boolean")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("Expected a finite real number") from error
    if not math.isfinite(value):
        raise ValueError("Expected a finite real number")
    return value


class BodyTwistLease:
    """Fail-zero direct ``(vx, vy, wz)`` ingress; no clamping or motion interlock.

    Only an explicit ``arm`` starts a new host-issued generation. Arm itself
    applies zero, and release, disconnect, expiry or malformed fresh input latch
    zero until another arm. All calls must be serialized by the host. Sequence
    numbers are nonnegative Python integers, strictly increasing within an epoch.

    Expiry is strict: a packet remains live at receipt + timeout and expires on
    the first later observation. Decisions, replay and foreign generations never
    renew the deadline. Invalid clocks/metadata raise after latching zero; invalid
    fresh command content returns False after latching zero. A valid stale packet
    returns False without erasing a still-live command.
    """

    def __init__(self, timeout_s=0.25):
        self._timeout_s = _number(timeout_s)
        if self._timeout_s <= 0:
            raise ValueError("Command timeout must be positive")
        self._generation = 0
        self._last_time_s = None
        self._source_time_s = None
        self._sequence = None
        self._command = (0.0, 0.0, 0.0)
        self._armed = False
        self._status = "disarmed"

    @property
    def timeout_s(self):
        return self._timeout_s

    def _disarm(self, status):
        self._armed = False
        self._command = (0.0, 0.0, 0.0)
        self._status = status

    def _observe_time(self, time_s):
        try:
            now = _number(time_s)
            if now < 0 or (self._last_time_s is not None and now < self._last_time_s):
                raise ValueError(
                    "Command source clock must be nonnegative and monotonic"
                )
        except ValueError:
            self._disarm("invalid_time")
            raise
        self._last_time_s = now
        if (
            self._armed
            and self._source_time_s is not None
            and now > self._source_time_s + self._timeout_s
        ):
            self._disarm("expired")
        return now

    def arm(self, time_s):
        """Explicit host authorization, not an episode or recurrent-state reset."""
        self._observe_time(time_s)
        self._generation += 1
        self._source_time_s = self._sequence = None
        self._command = (0.0, 0.0, 0.0)
        self._armed = True
        self._status = "armed_waiting"
        return self._generation

    def receive(self, command, *, time_s, sequence, generation):
        """Admit a tuple/list triple; cached/replayed packets are not heartbeats."""
        now = self._observe_time(time_s)
        if type(generation) is not int or generation < 1:
            self._disarm("invalid_packet")
            raise ValueError("Generation must be a positive host-issued Python integer")
        if generation != self._generation or not self._armed:
            return False
        if type(sequence) is not int or sequence < 0:
            self._disarm("invalid_packet")
            raise ValueError("Sequence must be a nonnegative Python integer")
        if self._sequence is not None and sequence <= self._sequence:
            return False
        try:
            if not isinstance(command, (tuple, list)) or len(command) != 3:
                raise ValueError("Body twist must contain exactly three finite numbers")
            copied = tuple(_number(value) for value in command)
            if len(copied) != 3:
                raise ValueError("Body twist changed length while being copied")
        except (TypeError, ValueError, OverflowError):
            self._disarm("invalid_packet")
            return False
        self._command = copied
        self._source_time_s = now
        self._sequence = sequence
        self._status = "active"
        return True

    def release(self, time_s):
        self._observe_time(time_s)
        self._disarm("released")

    def disconnect(self, time_s):
        self._observe_time(time_s)
        self._disarm("disconnected")

    def resolve(self, time_s):
        now = self._observe_time(time_s)
        return LeaseDecision(
            self._command,
            now,
            self._source_time_s,
            self._sequence,
            self._generation,
            self._status,
        )
