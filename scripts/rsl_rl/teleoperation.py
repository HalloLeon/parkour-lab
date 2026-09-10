# Copyright (c) 2026, Leon Yi Bai
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-only, fail-zero keyboard ingress for the Go2 RMA controller."""

from __future__ import annotations

import math
import os
import time


def validate_operator_display(*, headless: bool, livestream: int) -> None:
    """Allow a local window or WebRTC, but reject a truly windowless session.

    Match AppLauncher's CLI/environment precedence. Streaming intentionally
    runs headless on the host while exposing the GUI and keyboard remotely.
    """
    stream_env = int(os.environ.get("LIVESTREAM", "0"))
    headless_env = int(os.environ.get("HEADLESS", "0"))
    if stream_env not in (0, 1, 2) or livestream not in (-1, 0, 1, 2):
        raise ValueError("LIVESTREAM/--livestream must select mode 0, 1 or 2.")
    if headless_env not in (0, 1):
        raise ValueError("HEADLESS must be 0 or 1.")
    stream = stream_env if livestream == -1 else livestream
    if (headless or headless_env) and stream == 0:
        raise ValueError(
            "--teleop requires a local GUI or --livestream=2 for the Streaming Client; "
            "headless operation without streaming has no operator input."
        )


class OperatorCommand:
    """Testable forward/stop/pivot interlock, independent of Kit and PyTorch.

    R arms a new episode. Shift is a hold-to-run deadman. Loss of focus,
    a stalled producer, stale motion-key events, X, or an episode end requires
    a new R/reset. Only actual input callbacks renew the input lease, never
    polling the cached key state. A new direction press has bounded grace for
    the initial repeat delay; real repeats then use the shorter input lease.
    This best-effort keyboard liveness is not a transport-level heartbeat: a
    client that does not deliver repeats fails closed, and a provider that
    synthesizes them after disconnect cannot certify client presence.
    This is a simulation command safeguard, not a guarantee of physical stopping.
    """

    def __init__(
        self,
        speed: float = 0.55,
        yaw_rate: float = 0.5,
        timeout_s: float = 0.25,
        input_timeout_s: float = 0.25,
        initial_input_grace_s: float = 0.75,
        control_dt: float = 0.02,
    ):
        if not all(
            math.isfinite(v)
            for v in (
                speed,
                yaw_rate,
                timeout_s,
                input_timeout_s,
                initial_input_grace_s,
                control_dt,
            )
        ):
            raise ValueError("Operator limits must be finite.")
        if (
            not 0.45 <= speed <= 0.7
            or not 0.25 <= yaw_rate <= 0.8
            or timeout_s <= 0
            or control_dt <= 0
        ):
            raise ValueError(
                "Operator commands require speed in [0.45, 0.7] m/s and yaw rate "
                "in [0.25, 0.8] rad/s, with positive timeout and control timestep."
            )
        if not 0 < input_timeout_s <= initial_input_grace_s <= 1.0:
            raise ValueError(
                "Keyboard leases require 0 < input timeout <= initial grace <= 1 second."
            )
        self.speed = speed
        self.yaw_rate = yaw_rate
        self.timeout_s = timeout_s
        self.input_timeout_s = input_timeout_s
        self.initial_input_grace_s = initial_input_grace_s
        self.control_dt = control_dt
        self.keys: set[str] = set()
        self.latched = True
        self._last_time: float | None = None
        self._last_motion_input_time: float | None = None
        self._motion_input_deadline: float | None = None
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._pivot_direction: str | None = None
        self._pivot_braking_elapsed_s: float | None = None
        self._pivot_active = False

    def _clear_pivot(self) -> None:
        """Return the braking/pivot state machine to its idle state."""
        self._pivot_direction = None
        self._pivot_braking_elapsed_s = None
        self._pivot_active = False
        self._yaw_rate = 0.0

    def stop(self) -> None:
        self.latched = True
        self.keys.clear()
        self._last_motion_input_time = None
        self._motion_input_deadline = None
        self._speed = self._yaw_rate = 0.0
        self._clear_pivot()

    def reset(self, now: float) -> None:
        self.stop()
        self.latched = not math.isfinite(now)
        self._last_time = now

    def key_event(
        self,
        key: str,
        now: float,
        *,
        pressed: bool,
        repeat: bool = False,
        shift_held: bool = False,
    ) -> None:
        """Consume a real press/release/repeat; repeats cannot arm stale keys.

        The event's modifier state confirms the deadman on every direction
        refresh. A lost Shift release therefore cannot be hidden by subsequent
        arrow repeats. Unrelated key traffic never renews the motion lease.
        """
        if not math.isfinite(now) or (
            self._last_motion_input_time is not None
            and (
                now < self._last_motion_input_time or now > self._motion_input_deadline
            )
        ):
            self.stop()
            return
        if not pressed:
            self.keys.discard(key)
            if key in {"UP", "LEFT", "RIGHT", "LEFT_SHIFT"}:
                self._clear_pivot()
            if not self.keys & {"UP", "LEFT", "RIGHT"}:
                self._last_motion_input_time = None
                self._motion_input_deadline = None
            return
        if self.latched or (repeat and key not in self.keys):
            return
        new_press = key not in self.keys
        self.keys.add(key)
        if new_press and key in {"UP", "LEFT", "RIGHT"}:
            self._clear_pivot()
        if key in {"UP", "LEFT", "RIGHT"}:
            if not shift_held:
                self.keys.discard("LEFT_SHIFT")
                self._speed = self._yaw_rate = 0.0
                self._clear_pivot()
                return
            if new_press or repeat:
                self._last_motion_input_time = now
                self._motion_input_deadline = now + (
                    self.initial_input_grace_s if new_press else self.input_timeout_s
                )

    def command(
        self, now: float, *, focused: bool, planar_speed: float
    ) -> tuple[float, float]:
        dt = 0.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        if (
            not focused
            or not math.isfinite(now)
            or not math.isfinite(planar_speed)
            or planar_speed < 0.0
            or dt < 0
            or dt > self.timeout_s
        ):
            self.stop()
        directions = self.keys & {"UP", "LEFT", "RIGHT"}
        if len(directions) > 1:
            self.stop()
        if directions and (
            self._last_motion_input_time is None
            or now - self._last_motion_input_time < 0.0
            or now > self._motion_input_deadline
        ):
            self.stop()
        if (
            self.latched
            or "LEFT_SHIFT" not in self.keys
            or "DOWN" in self.keys
            or len(directions) != 1
        ):
            self._speed = self._yaw_rate = 0.0
            self._clear_pivot()
            return 0.0, 0.0
        if "UP" in directions:
            self._clear_pivot()
            # Match the trained command steps. A command ramp through positive
            # sub-minimum speeds is not an actuator rate limiter and introduces
            # a deployment-only command distribution.
            self._speed = self.speed
        else:
            self._speed = 0.0
            self._yaw_rate = self._pivot_command(
                next(iter(directions)), planar_speed, dt
            )
        return self._speed, self._yaw_rate

    def _pivot_command(self, direction: str, planar_speed: float, dt: float) -> float:
        """Brake, establish a quiet interval, then hold one nonchattering pivot.

        Entry requires speed <=0.15 m/s for 0.10 s of observed control steps.
        Wall-clock delays cannot fill this dwell faster than physics advances.
        Once active, ordinary stepping between 0.15 and 0.25 m/s does not toggle
        yaw off/on. Exceeding 0.25 m/s during the pivot is an abort that latches
        stop, requiring R.
        Any release, new direction, translation, or reset clears this state.
        These speed checks do not certify stable terrain or body attitude.
        """
        if direction != self._pivot_direction:
            self._clear_pivot()
            self._pivot_direction = direction
        if self._pivot_active:
            if planar_speed > 0.25:
                self.stop()
                return 0.0
            return self.yaw_rate if direction == "LEFT" else -self.yaw_rate
        if planar_speed > 0.15:
            self._pivot_braking_elapsed_s = None
            return 0.0
        if self._pivot_braking_elapsed_s is None:
            self._pivot_braking_elapsed_s = 0.0
            return 0.0
        self._pivot_braking_elapsed_s += min(dt, self.control_dt)
        if self._pivot_braking_elapsed_s < 0.10 - 1.0e-9:
            return 0.0
        self._pivot_active = True
        return self.yaw_rate if direction == "LEFT" else -self.yaw_rate


def run_keyboard_control(
    env, policy, simulation_app, *, speed: float, yaw_rate: float
) -> None:
    """Run a local or streamed, real-time, single-environment operator session."""
    import carb
    import omni.appwindow
    import torch

    window = omni.appwindow.get_default_app_window()
    if window is None or window.get_keyboard() is None:
        raise RuntimeError(
            "Keyboard operation requires an Isaac Sim GUI (local or --livestream=2)."
        )
    input_interface = carb.input.acquire_input_interface()
    keyboard = window.get_keyboard()
    control = OperatorCommand(speed, yaw_rate, control_dt=env.unwrapped.step_dt)
    reset_requested = False
    quit_requested = False

    def on_key(event, *_):
        nonlocal reset_requested, quit_requested
        key = event.input.name
        pressed = event.type == carb.input.KeyboardEventType.KEY_PRESS
        repeated = event.type == carb.input.KeyboardEventType.KEY_REPEAT
        if pressed:
            if key == "R":
                reset_requested = True
            elif key == "X":
                reset_requested = False
                control.stop()
            elif key == "ESCAPE":
                reset_requested = False
                control.stop()
                quit_requested = True
        if key not in {"R", "X", "ESCAPE"} and (
            pressed
            or repeated
            or event.type == carb.input.KeyboardEventType.KEY_RELEASE
        ):
            # Carbonite's documented KeyboardModifierFlags defines Shift as
            # bit zero. Missing modifier metadata cannot certify the deadman.
            control.key_event(
                key,
                time.monotonic(),
                pressed=pressed or repeated,
                repeat=repeated,
                shift_held=bool(getattr(event, "modifiers", 0) & 1),
            )
        return True

    def on_focus_change(_event):
        nonlocal reset_requested
        reset_requested = False
        control.stop()

    subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_key)
    # Latch even if focus was lost and regained between two control steps.
    focus_subscription = (
        window.get_window_focus_event_stream().create_subscription_to_pop(
            on_focus_change
        )
    )
    base_env = env.unwrapped
    intent = base_env.command_manager.get_term("intent")
    intent.invalidate()
    direction = torch.tensor([[1.0, 0.0]], device=base_env.device)
    print(
        "[TELEOP] R: reset/arm; hold LEFT SHIFT + UP: forward; SHIFT + LEFT/RIGHT: pivot; DOWN: stop; X: latch stop; ESC: exit."
    )
    print(
        "[TELEOP] Release Shift to stop. Conflicting arrows, focus loss, expired input, server stalls >0.25s or episode end latch stop and require R, then fresh keys. Brake-before-pivot; operator must choose stable ground."
    )
    print(
        "[TELEOP] Fresh direction presses allow 0.75s for initial repeat, then real repeats renew a 0.25s input lease with Shift held. Non-repeating streams fail closed. This is best-effort keyboard liveness, not certified network connectivity; client repeat delivery must be verified. Commands use trained fixed amplitudes."
    )
    print(
        "[TELEOP] Pivot entry waits for planar speed <=0.15 m/s for 0.10s of observed control steps; an active pivot aborts above 0.25 m/s and requires R. The operator must choose stable ground."
    )
    try:
        with torch.inference_mode():
            while simulation_app.is_running() and not quit_requested:
                started = time.monotonic()
                if not base_env.sim.is_playing():
                    control.stop()
                    intent.invalidate()
                    simulation_app.update()
                    # Discard R delivered while paused, including the update
                    # that resumes playback. Require an explicit post-resume arm.
                    reset_requested = False
                    time.sleep(0.02)
                    continue
                focused = window.is_focused() and not window.get_input_blocking_state(
                    carb.input.DeviceType.KEYBOARD
                )
                if reset_requested:
                    env.reset()
                    control.reset(time.monotonic())
                    reset_requested = False
                # Match the gravity-aligned planar speed used by evaluation;
                # body-frame XY would change this interlock when the base tilts.
                velocity = base_env.scene["robot"].data.root_lin_vel_w[0, :2]
                forward, yaw = control.command(
                    time.monotonic(),
                    focused=focused,
                    planar_speed=float(velocity.norm()),
                )
                intent.set_external_intent(
                    None,
                    direction,
                    torch.tensor([forward], device=base_env.device),
                    torch.tensor([yaw], device=base_env.device),
                )
                observations = env.refresh_command_observations()
                actions = policy(observations)
                if not torch.isfinite(actions).all():
                    intent.invalidate()
                    raise FloatingPointError(
                        "Non-finite policy action: operator simulation stopped."
                    )
                _, _, dones, _ = env.step(actions)
                if dones.any():
                    control.stop()
                    intent.invalidate()
                    print(
                        "[TELEOP] Episode ended. Motion is disarmed; press R, then hold Shift to continue."
                    )
                time.sleep(max(0.0, base_env.step_dt - (time.monotonic() - started)))
    finally:
        intent.invalidate()
        input_interface.unsubscribe_to_keyboard_events(keyboard, subscription)
        del focus_subscription
