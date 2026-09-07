# Copyright (c) 2026, Leon Yi Bai
# SPDX-License-Identifier: BSD-3-Clause

"""Simulation-only, fail-zero keyboard ingress for the Go2 RMA controller."""

from __future__ import annotations

import math
import time


class OperatorCommand:
    """Testable forward/stop/pivot interlock, independent of Kit and PyTorch.

    R arms a new episode. Shift is a hold-to-run deadman. Loss of focus,
    a stalled producer, X, or an episode end requires a new R/reset.
    This is a command safeguard, not a guarantee of physical stopping.
    """

    def __init__(
        self,
        speed: float = 0.55,
        yaw_rate: float = 0.5,
        timeout_s: float = 0.25,
        control_dt: float = 0.02,
    ):
        if not all(math.isfinite(v) for v in (speed, yaw_rate, timeout_s, control_dt)):
            raise ValueError("Operator limits must be finite.")
        if (
            not 0.05 < speed <= 0.7
            or not 0.05 < yaw_rate <= 0.8
            or min(timeout_s, control_dt) <= 0
        ):
            raise ValueError(
                "Operator limits must remain inside the trained command range."
            )
        self.speed = speed
        self.yaw_rate = yaw_rate
        self.timeout_s = timeout_s
        self.control_dt = control_dt
        self.keys: set[str] = set()
        self.latched = True
        self._last_time: float | None = None
        self._speed = 0.0
        self._yaw_rate = 0.0

    def stop(self) -> None:
        self.latched = True
        self.keys.clear()
        self._speed = self._yaw_rate = 0.0

    def reset(self, now: float) -> None:
        self.stop()
        self.latched = False
        self._last_time = now

    def command(
        self, now: float, *, focused: bool, planar_speed: float
    ) -> tuple[float, float]:
        dt = 0.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        if (
            not focused
            or not math.isfinite(now)
            or not math.isfinite(planar_speed)
            or dt < 0
            or dt > self.timeout_s
        ):
            self.stop()
        directions = self.keys & {"UP", "LEFT", "RIGHT"}
        # A slow GUI must not accelerate the requested motion faster per
        # physics step. Wall time above is only the producer watchdog.
        ramp_dt = min(dt, self.control_dt)
        if (
            self.latched
            or "LEFT_SHIFT" not in self.keys
            or "DOWN" in self.keys
            or len(directions) != 1
        ):
            self._speed = self._yaw_rate = 0.0
            return 0.0, 0.0
        if "UP" in directions:
            self._yaw_rate = 0.0
            self._speed = min(self.speed, self._speed + 0.8 * ramp_dt)
        else:
            self._speed = 0.0
            # Brake before rotating; do not turn a translation request into
            # an untrained combined command while the body is still moving.
            target = (
                (self.yaw_rate if "LEFT" in directions else -self.yaw_rate)
                if planar_speed <= 0.15
                else 0.0
            )
            if target == 0.0 or target * self._yaw_rate < 0.0:
                self._yaw_rate = 0.0
            else:
                self._yaw_rate += max(
                    -1.5 * ramp_dt, min(1.5 * ramp_dt, target - self._yaw_rate)
                )
        return self._speed, self._yaw_rate


def run_keyboard_control(
    env, policy, simulation_app, *, speed: float, yaw_rate: float
) -> None:
    """Run a visible, real-time, single-environment operator session."""
    import carb
    import omni.appwindow
    import torch

    window = omni.appwindow.get_default_app_window()
    if window is None or window.get_keyboard() is None:
        raise RuntimeError("Keyboard operation requires a visible Isaac Sim window.")
    input_interface = carb.input.acquire_input_interface()
    keyboard = window.get_keyboard()
    control = OperatorCommand(speed, yaw_rate, control_dt=env.unwrapped.step_dt)
    reset_requested = False
    quit_requested = False

    def on_key(event, *_):
        nonlocal reset_requested, quit_requested
        key = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key == "R":
                reset_requested = True
            elif key == "X":
                reset_requested = False
                control.stop()
            elif key == "ESCAPE":
                reset_requested = False
                control.stop()
                quit_requested = True
            else:
                control.keys.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            control.keys.discard(key)
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
        "[TELEOP] Release Shift to stop. Conflicting arrows stop. Focus loss/stall/episode end requires R. Pivots only on stable ground."
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
                velocity = base_env.scene["robot"].data.root_lin_vel_b[0, :2]
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
