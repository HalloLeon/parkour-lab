"""Simulation-only keyboard ingress for the frozen recurrent operator.

Kit callbacks and the policy loop must run serially on the simulator host. Only
real motion-key repeats renew the wall-clock lease; polling held keys never does.
This is not a certified remote disconnect detector or a hardware controller.
"""

from __future__ import annotations

import math
import time

from parkour_lab.learning.command_source import BodyTwistLease


MOTION_KEYS = {
    "W": (0.4, 0.0, 0.0),
    "S": (-0.3, 0.0, 0.0),
    "A": (0.0, 0.2, 0.0),
    "D": (0.0, -0.2, 0.0),
    "Q": (0.0, 0.0, 0.5),
    "E": (0.0, 0.0, -0.5),
    "Z": (0.4, 0.0, 0.4),
    "C": (0.4, 0.0, -0.4),
}


class KeyboardTwist:
    """One key owns the whole command. Stop events never reset policy memory.

    A fresh press after R selects a key but applies zero until its first real
    repeat. This avoids granting a long initial motion lease for OS repeat delay.
    A second motion key disarms instead of renewing a possibly stale chord.
    Shift must be present in each motion callback's modifier metadata.
    """

    def __init__(self):
        self.lease = BodyTwistLease(timeout_s=0.25)
        self.available = False
        self.reset_requested = False
        self.quit_requested = False
        self._key = None
        self._last_poll = None
        self._sequence = 0

    def stop(self, now, *, disconnected=False):
        self._key = None
        self.reset_requested = False
        if disconnected:
            self.available = False
            self.lease.disconnect(now)
        else:
            self.lease.release(now)

    def resolve(self, now):
        decision = self.lease.resolve(now)
        if decision.status not in ("armed_waiting", "active"):
            self._key = None
        return decision

    def poll(self, now, *, available):
        decision = self.resolve(now)  # Validate clock before touching cached state.
        if not available or (
            self._last_poll is not None and now - self._last_poll > self.lease.timeout_s
        ):
            self.stop(now, disconnected=True)
        self.available = bool(available)
        self._last_poll = decision.decision_time_s
        return self.resolve(now)

    def key_event(self, key, event_type, now, *, shift_held=False):
        decision = self.resolve(now)
        if event_type not in ("press", "repeat", "release"):
            return
        if key == "ESCAPE" and event_type == "press":
            self.stop(now)
            self.quit_requested = True
        elif key == "X" and event_type in ("press", "repeat"):
            self.stop(now)
        elif not self.available:
            return  # Discard even arm/reset events delivered while unavailable.
        elif key == "R" and event_type == "press":
            self._key = None
            self.reset_requested = False
            self._sequence = 0
            self.lease.arm(now)
        elif key == "N" and event_type == "press":
            self.stop(now)
            self.reset_requested = True  # The main loop owns the physical reset.
        elif key in ("LEFT_SHIFT", "RIGHT_SHIFT") and event_type == "release":
            self.stop(now)
        elif key in MOTION_KEYS:
            if event_type == "release":
                if key == self._key:
                    self.stop(now)
            elif shift_held is not True:
                self.stop(now)
            elif decision.status in ("armed_waiting", "active"):
                if event_type == "press":
                    if self._key is not None:
                        self.stop(now)
                    else:
                        self._key = key
                elif key == self._key:
                    self.lease.receive(
                        MOTION_KEYS[key],
                        time_s=now,
                        sequence=self._sequence,
                        generation=decision.generation,
                    )
                    self._sequence += 1
                else:
                    # An unrelated repeat cannot refresh a lost-release command.
                    self.stop(now)


def run_live_loop(
    env,
    host,
    app,
    control,
    *,
    is_available,
    clock=time.monotonic,
    sleep=time.sleep,
):
    """Serialize input decisions, physical resets and fixed-clock actor delivery.

    Paused app updates do not advance actor time. An episode auto-reset or an N
    reset is the only reason to reset GRU rows; the simulation clock never rewinds.
    No per-frame files or unbounded trace buffers are created.
    """
    import torch

    if (
        env.num_envs != 1
        or not math.isclose(env.step_dt, 0.02, rel_tol=0.0, abs_tol=1e-10)
        or env.cfg.decimation != 4
        or env.cfg.sim.render_interval != 4
    ):
        raise ValueError(
            "Interactive actor requires one 50 Hz environment with end-of-frame rendering"
        )
    started = clock()
    step_index = episode_ends = manual_resets = 0
    last_status = None
    reset_mask = torch.ones(1, dtype=torch.bool, device=env.device)
    with torch.inference_mode():
        while app.is_running() and not control.quit_requested:
            tick = clock()
            if not env.sim.is_playing():
                control.poll(tick, available=False)
                # Pinned SimulationContext.render suppresses physics during UI
                # updates, including the one that resumes play. app.update does
                # not, and could advance physics outside our policy clock.
                env.sim.render()
                # Includes the update that resumes play: R/N during it are lost.
                control.stop(clock(), disconnected=True)
                sleep(env.step_dt)
                continue
            decision = control.poll(tick, available=is_available())
            if control.reset_requested:
                control.stop(clock(), disconnected=True)
                env.reset()
                reset_mask.fill_(True)
                manual_resets += 1
                # env.reset may render and dispatch callbacks; discard them too.
                control.stop(clock(), disconnected=True)
                # Recheck quit/play/focus and callback faults before inference.
                # A reset is not a policy step and never advances its clock.
                continue
            if decision.status != last_status:
                print(f"[OPERATOR] {decision.status}", flush=True)
                last_status = decision.status
            command = env.scene["robot"].data.joint_pos.new_tensor([decision.command])
            result = host.act(
                command, time_s=step_index * env.step_dt, reset_mask=reset_mask
            )
            # Inference itself can stall. Never deliver its now-stale moving
            # action, or step the GRU twice to try to repair the same frame.
            latest = control.resolve(clock())
            if latest.command != decision.command:
                raise RuntimeError("Input expired during inference; delivery aborted")
            _, _, terminated, timed_out, _ = env.step(result.raw_action)
            step_index += 1
            reset_mask = (terminated | timed_out).clone()
            if reset_mask.any():
                episode_ends += 1
                control.stop(clock(), disconnected=True)
                print(
                    "[OPERATOR] Episode reset; press R, then fresh motion keys.",
                    flush=True,
                )
            sleep(max(0.0, env.step_dt - (clock() - tick)))
    return {
        "control_steps": step_index,
        "simulated_seconds": step_index * env.step_dt,
        "wall_seconds": clock() - started,
        "episode_resets": episode_ends,
        "manual_resets": manual_resets,
        "learning_updates": 0,
    }


def run_keyboard_actor(env, host, app):
    """Attach the local/streamed Kit window; never fall back to synthetic repeats."""
    import carb
    import omni.appwindow

    window = omni.appwindow.get_default_app_window()
    if window is None or window.get_keyboard() is None:
        raise RuntimeError("A local or streamed Isaac Sim keyboard window is required")
    keyboard = window.get_keyboard()
    inputs = carb.input.acquire_input_interface()
    control = KeyboardTwist()
    callback_errors = []

    def is_available():
        if callback_errors:
            raise RuntimeError("Keyboard callback failed") from callback_errors[0]
        return bool(
            env.sim.is_playing()
            and window.is_focused()
            and not window.get_input_blocking_state(carb.input.DeviceType.KEYBOARD)
        )

    def on_key(event, *_):
        try:
            now = time.monotonic()
            if not is_available():
                control.stop(now, disconnected=True)
            event_type = {
                carb.input.KeyboardEventType.KEY_PRESS: "press",
                carb.input.KeyboardEventType.KEY_REPEAT: "repeat",
                carb.input.KeyboardEventType.KEY_RELEASE: "release",
            }.get(event.type)
            control.key_event(
                event.input.name,
                event_type,
                now,
                shift_held=bool(getattr(event, "modifiers", 0) & 1),
            )
        except Exception as error:
            if not callback_errors:
                callback_errors.append(error)
            control.stop(time.monotonic(), disconnected=True)
        return True

    def on_focus_change(_event):
        control.stop(time.monotonic(), disconnected=True)

    subscription = inputs.subscribe_to_keyboard_events(keyboard, on_key)
    focus_subscription = None
    try:
        focus_subscription = (
            window.get_window_focus_event_stream().create_subscription_to_pop(
                on_focus_change
            )
        )
        print(
            "[OPERATOR] SIMULATION ONLY. R: arm (no reset); Shift + ONE key: "
            "W/S forward/back, A/D lateral, Q/E pivot, Z/C forward arcs. "
            "Release stops and requires R. N: physical reset, X: stop, Esc: exit.",
            flush=True,
        )
        print(
            "[OPERATOR] Motion starts on the first real key repeat. Missing repeats, "
            "Shift release, conflicting keys, focus loss, pause, >0.25s host stalls "
            "or episode ends latch zero body twist. Re-arm with R and fresh keys. "
            "A client that fabricates repeats after disconnect cannot certify presence; "
            "test the stream while attended. Zero twist is NOT zero motor action.",
            flush=True,
        )
        result = run_live_loop(env, host, app, control, is_available=is_available)
        if callback_errors:
            raise RuntimeError("Keyboard callback failed") from callback_errors[0]
        return result
    finally:
        control.stop(time.monotonic(), disconnected=True)
        inputs.unsubscribe_to_keyboard_events(keyboard, subscription)
        del focus_subscription
