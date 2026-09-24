"""Simulation-only single-key body-twist input, independent of legacy ingress.

Every fresh direction press starts immediately, without Shift or a separate arm
key. A press permits a short OS-repeat grace period; only actual repeat events
renew it. Polling never acts as an input heartbeat. This is not a hardware
controller, a certified disconnect detector, or evidence of streamed GUI testing.
"""

from __future__ import annotations

import math
from numbers import Real

from parkour_lab.learning.command_source import LeaseDecision


SINGLE_KEY_MOTION_KEYS = {
    "UP": (0.4, 0.0, 0.0),
    "DOWN": (-0.3, 0.0, 0.0),
    "LEFT": (0.0, 0.2, 0.0),
    "RIGHT": (0.0, -0.2, 0.0),
    "J": (0.0, 0.0, 0.5),
    "L": (0.0, 0.0, -0.5),
    "U": (0.4, 0.0, 0.4),
    "O": (0.4, 0.0, -0.4),
}
SPEED_LEVELS = {"KEY_1": 0.25, "KEY_2": 0.5, "KEY_3": 1.0}
DEFAULT_SPEED_SCALE = 0.5
INITIAL_PRESS_GRACE_S = 0.75
REPEAT_TIMEOUT_S = 0.25
HOST_STALL_TIMEOUT_S = 0.25
_LOCK_MODIFIERS = 16 | 32  # CapsLock, NumLock; reject other/unknown modifier bits.
_SHORTCUT_KEYS = frozenset(
    f"{side}_{modifier}"
    for side in ("LEFT", "RIGHT")
    for modifier in ("SHIFT", "CONTROL", "ALT", "SUPER")
)
_ADMIN_KEYS = frozenset(("X", "N", "ESCAPE"))
_ZERO = (0.0, 0.0, 0.0)


def simple_keyboard_protocol():
    """Return JSON-friendly metadata for the default interactive simulator UI."""
    return {
        "mode": "single-key",
        "simulation_only": True,
        "streamed_gui_certified": False,
        "motion_keys": {
            key: list(command) for key, command in SINGLE_KEY_MOTION_KEYS.items()
        },
        "motion_keys_scale": "Full-scale maxima; multiply by selected speed scale",
        "speed_levels": dict(SPEED_LEVELS),
        "default_speed_scale": DEFAULT_SPEED_SCALE,
        "speed_selection": "Applies to the next fresh direction press only",
        "motion_start": "Fresh unmodified direction press; no arm key or Shift",
        "motion_stop": "Release the direction key, or press X",
        "reset_key": "N",
        "quit_key": "ESCAPE",
        "initial_press_grace_s": INITIAL_PRESS_GRACE_S,
        "repeat_timeout_s": REPEAT_TIMEOUT_S,
        "host_stall_timeout_s": HOST_STALL_TIMEOUT_S,
        "deadline_rule": "Live at the exact deadline; expires strictly after",
        "renewal": "Only actual repeats of the active direction key",
        "recovery": "Release all direction keys, then make one fresh press",
        "mouse_navigation_guard": (
            "Kit adapter disallows motion while left, middle, or right mouse button is held"
        ),
        "limits": (
            "Synthetic CPU tests do not certify streamed GUI delivery, OS repeat "
            "timing, remote disconnect detection, braking, or hardware safety"
        ),
    }


class SingleKeyTwist:
    """Serialized host-local, fail-zero input; ordinary stops do not reset GRUs.

    At most one direction may be held. Stopped keys remain blocked until their
    release is observed, so a repeat or duplicate press cannot revive authority
    after focus loss, expiry, a conflict, or a loop stall. Speed keys select the
    next command's scale and never change or renew an already active command.
    """

    host_stall_timeout_s = HOST_STALL_TIMEOUT_S

    def __init__(self):
        self.available = False
        self.reset_requested = False
        self.quit_requested = False
        self.speed_scale = DEFAULT_SPEED_SCALE
        self.last_poll_gap_s = 0.0
        self.max_poll_gap_s = 0.0
        self._last_poll = None
        self._last_time = None
        self._generation = 0
        self._source_time = None
        self._sequence = None
        self._command = _ZERO
        self._status = "disarmed"
        self._deadline = None
        self._key = None
        # Only the eight recognized motion keys enter this set.
        self._held = set()

    def _disarm(self, status):
        self._key = None
        self._command = _ZERO
        self._deadline = None
        self._status = status
        self.reset_requested = False

    def _observe_time(self, now):
        try:
            if not isinstance(now, Real) or isinstance(now, bool):
                raise ValueError("Input clock must be a finite real number")
            now = float(now)
            if (
                not math.isfinite(now)
                or now < 0
                or (self._last_time is not None and now < self._last_time)
            ):
                raise ValueError(
                    "Input clock must be finite, nonnegative and monotonic"
                )
        except (ValueError, OverflowError) as error:
            self._disarm("invalid_time")
            self.available = False
            raise ValueError(
                "Input clock must be finite, nonnegative and monotonic"
            ) from error
        self._last_time = now
        if self._deadline is not None and now > self._deadline:
            self._disarm("expired")
        return now

    def _decision(self, now):
        return LeaseDecision(
            self._command,
            now,
            self._source_time,
            self._sequence,
            self._generation,
            self._status,
        )

    def resolve(self, now):
        """Inspect authority without renewing its local receipt-time deadline."""
        return self._decision(self._observe_time(now))

    def stop(self, now, *, disconnected=False):
        """Latch zero, retaining held-key evidence; never request a robot reset."""
        now = self._observe_time(now)
        self._disarm("disconnected" if disconnected else "released")
        if disconnected:
            self.available = False
        return self._decision(now)

    def poll(self, now, *, available):
        """Check focus/play and the independent 250 ms host-loop watchdog."""
        now = self._observe_time(now)
        self.last_poll_gap_s = 0.0 if self._last_poll is None else now - self._last_poll
        self.max_poll_gap_s = max(self.max_poll_gap_s, self.last_poll_gap_s)
        if not available:
            self._disarm("disconnected")
        elif self.last_poll_gap_s > self.host_stall_timeout_s:
            self._disarm("host_stall")
        self.available = bool(available)
        self._last_poll = now
        return self._decision(now)

    def key_event(self, key, event_type, now, *, modifiers=0):
        """Consume real ``press``/``repeat``/``release`` callbacks, not key polls.

        Shift/Control/Alt/Super combinations are rejected. Lock-state modifier
        bits are harmless. Releases still clear held-key evidence regardless of
        modifiers, focus, or whether their preceding press was accepted.
        """
        now = self._observe_time(now)
        if event_type not in ("press", "repeat", "release"):
            return
        motion_key = key in SINGLE_KEY_MOTION_KEYS
        if motion_key and event_type == "release":
            self._held.discard(key)
            if key == self._key:
                self._disarm("released")
            return
        if key in _SHORTCUT_KEYS and event_type in ("press", "repeat"):
            self._disarm("modified_input")
            return
        unmodified = (
            type(modifiers) is int
            and modifiers >= 0
            and not (modifiers & ~_LOCK_MODIFIERS)
        )
        if not unmodified:
            if motion_key:
                self._held.add(key)
            if motion_key or key in _ADMIN_KEYS:
                self._disarm("modified_input")
            return
        if key == "ESCAPE" and event_type == "press":
            self._disarm("released")
            self.quit_requested = True
        elif key == "X" and event_type in ("press", "repeat"):
            self._disarm("released")
        elif not self.available:
            if motion_key:
                self._held.add(key)
            return
        elif key == "N" and event_type == "press":
            self._disarm("released")
            self.reset_requested = True
        elif key in SPEED_LEVELS and event_type == "press":
            self.speed_scale = SPEED_LEVELS[key]
        elif motion_key:
            if event_type == "press":
                blocked = bool(self._held) or self._key is not None
                self._held.add(key)
                if blocked:
                    self._disarm("conflicting_input")
                    return
                self._generation += 1
                self._sequence = 0
                self._key = key
                self._command = tuple(
                    value * self.speed_scale for value in SINGLE_KEY_MOTION_KEYS[key]
                )
                self._source_time = now
                self._deadline = now + INITIAL_PRESS_GRACE_S
                self._status = "active"
                self.reset_requested = False
            elif event_type == "repeat":
                self._held.add(key)
                if key == self._key and self._status == "active":
                    self._sequence += 1
                    self._source_time = now
                    self._deadline = now + REPEAT_TIMEOUT_S
                elif self._key is not None:
                    self._disarm("conflicting_input")
