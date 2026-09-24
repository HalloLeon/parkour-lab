"""Simulation-only keyboard ingress for the frozen recurrent operator.

Kit callbacks and the policy loop must run serially on the simulator host. A
fresh press can start single-key motion; only real repeats extend its deadline.
Polling held keys never renews input authority.
This is not a certified remote disconnect detector or a hardware controller.
"""

from __future__ import annotations

import math
import sys
import time

from parkour_lab.learning.command_source import BodyTwistLease
from scripts.rsl_rl.operator_simple_input import (
    SINGLE_KEY_MOTION_KEYS,
    SPEED_LEVELS,
    SingleKeyTwist,
)


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
        self.last_poll_gap_s = 0.0
        self.max_poll_gap_s = 0.0
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
        self.last_poll_gap_s = 0.0 if self._last_poll is None else now - self._last_poll
        self.max_poll_gap_s = max(self.max_poll_gap_s, self.last_poll_gap_s)
        if not available or self.last_poll_gap_s > self.lease.timeout_s:
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
    command_clock=None,
    sleep=time.sleep,
    pace=True,
    max_steps=None,
    max_wall_seconds=None,
    before_poll=None,
    after_step=None,
    timings=None,
):
    """Serialize input decisions, physical resets and fixed-clock actor delivery.

    Paused app updates do not advance actor time. An episode auto-reset or an N
    reset is the only reason to reset GRU rows; the simulation clock never rewinds.
    No per-frame files or unbounded trace buffers are created.
    Optional bounded-probe hooks inject synthetic events BEFORE the poll clock
    is sampled, and inspect each completed native step with its original reset
    mask. Ordinary keyboard operation supplies neither hook nor a step limit.
    Optional timings hold bounded host-call durations, including failed calls;
    no extra CUDA synchronization is introduced. ``clock`` always measures host
    time (durations, administrative timeout and pacing). Explicit functional
    probes and scripted demonstrations can supply a separate ``command_clock``;
    a streamed demo retains pacing, while offline functional probes disable it.
    Live keyboard input always defaults to the host clock for every receipt,
    poll and delivery check.
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
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("Control-step limit must be a positive integer")
    if max_wall_seconds is not None and (
        isinstance(max_wall_seconds, bool)
        or not math.isfinite(max_wall_seconds)
        or max_wall_seconds <= 0
    ):
        raise ValueError("Wall-time limit must be finite and positive")
    if type(pace) is not bool:
        raise ValueError("Pacing must be a boolean")
    if command_clock is None:
        command_clock = clock
    started = clock()
    step_index = episode_ends = manual_resets = 0
    if timings is not None:
        timings.clear()

    def measured(name, operation, *args, **kwargs):
        if timings is None:
            return operation(*args, **kwargs)
        before = clock()
        try:
            return operation(*args, **kwargs)
        finally:
            elapsed = clock() - before
            entry = timings.setdefault(name, {"calls": 0, "max_s": 0.0})
            entry.update(
                calls=entry["calls"] + 1,
                previous_step=entry.get("last_step"),
                previous_s=entry.get("last_s"),
                last_step=step_index,
                last_s=elapsed,
                max_step=step_index if elapsed >= entry["max_s"] else entry["max_step"],
                max_s=max(entry["max_s"], elapsed),
            )

    last_status = None
    reset_mask = torch.ones(1, dtype=torch.bool, device=env.device)
    with torch.inference_mode():
        while (
            app.is_running()
            and not control.quit_requested
            and (max_steps is None or step_index < max_steps)
        ):
            if max_wall_seconds is not None and clock() - started > max_wall_seconds:
                raise TimeoutError("Live loop exceeded its wall-time budget")
            if before_poll is not None:
                measured("source_events", before_poll, step_index)
            tick = clock()
            command_tick = command_clock()
            if not env.sim.is_playing():
                control.poll(command_tick, available=False)
                # Pinned SimulationContext.render suppresses physics during UI
                # updates, including the one that resumes play. app.update does
                # not, and could advance physics outside our policy clock.
                env.sim.render()
                # Includes the update that resumes play: R/N during it are lost.
                control.stop(command_clock(), disconnected=True)
                sleep(env.step_dt)
                continue
            decision = measured(
                "input_poll", control.poll, command_tick, available=is_available()
            )
            if control.reset_requested:
                control.stop(command_clock(), disconnected=True)
                measured("physical_reset", env.reset)
                reset_mask.fill_(True)
                manual_resets += 1
                # env.reset may render and dispatch callbacks; discard them too.
                control.stop(command_clock(), disconnected=True)
                # Recheck quit/play/focus and callback faults before inference.
                # A reset is not a policy step and never advances its clock.
                continue
            if decision.status != last_status:
                print(f"[OPERATOR] {decision.status}", flush=True)
                last_status = decision.status
            command = env.scene["robot"].data.joint_pos.new_tensor([decision.command])
            result = measured(
                "actor",
                host.act,
                command,
                time_s=step_index * env.step_dt,
                reset_mask=reset_mask,
            )
            # Inference itself can stall. Never deliver its now-stale moving
            # action, or step the GRU twice to try to repair the same frame.
            latest = control.resolve(command_clock())
            host_limit = getattr(control, "host_stall_timeout_s", None)
            if (
                host_limit is not None
                and any(decision.command)
                and clock() - tick > host_limit
            ):
                # The initial key-repeat grace must not also permit slow/stale
                # inference delivery. Offline legacy probes do not use this guard.
                control.stop(command_clock(), disconnected=True)
                raise RuntimeError("Host stalled during inference; delivery aborted")
            if latest.command != decision.command:
                raise RuntimeError("Input expired during inference; delivery aborted")
            _, _, terminated, timed_out, _ = measured(
                "native_step", env.step, result.raw_action
            )
            if after_step is not None:
                measured(
                    "observer",
                    after_step,
                    step_index,
                    decision,
                    reset_mask,
                    result,
                    terminated,
                    timed_out,
                )
            step_index += 1
            reset_mask = (terminated | timed_out).clone()
            if reset_mask.any():
                episode_ends += 1
                control.stop(command_clock(), disconnected=True)
                print(
                    "[OPERATOR] Episode reset; "
                    + (
                        "release all motion keys, then press a fresh direction key."
                        if isinstance(control, SingleKeyTwist)
                        else "press R, then fresh motion keys."
                    ),
                    flush=True,
                )
            if pace:
                sleep(max(0.0, env.step_dt - (clock() - tick)))
    if max_wall_seconds is not None and clock() - started > max_wall_seconds:
        raise TimeoutError("Live loop exceeded its wall-time budget")
    return {
        "control_steps": step_index,
        "simulated_seconds": step_index * env.step_dt,
        "wall_seconds": clock() - started,
        "episode_resets": episode_ends,
        "manual_resets": manual_resets,
        "learning_updates": 0,
    }


def run_keyboard_actor(env, host, app, *, controls="single-key"):
    """Attach the local/streamed Kit window; never fall back to synthetic repeats."""
    import carb
    import omni.appwindow

    if controls not in ("single-key", "legacy"):
        raise ValueError("Unknown keyboard controls profile")
    simple = controls == "single-key"
    window = omni.appwindow.get_default_app_window()
    if window is None or window.get_keyboard() is None:
        raise RuntimeError("A local or streamed Isaac Sim keyboard window is required")
    keyboard = window.get_keyboard()
    mouse = window.get_mouse() if simple else None
    if simple and mouse is None:
        raise RuntimeError(
            "Single-key controls require a mouse for the navigation guard"
        )
    inputs = carb.input.acquire_input_interface()
    control = SingleKeyTwist() if simple else KeyboardTwist()
    callback_errors = []

    def note_failure(error, detail):
        if hasattr(error, "add_note"):
            error.add_note(detail)
        else:  # Python 3.10 has no exception notes.
            print(f"[OPERATOR] {detail}", file=sys.stderr, flush=True)

    def callback_failed(error):
        first_failure = not callback_errors
        if first_failure:
            callback_errors.append(error)
        try:
            control.stop(time.monotonic(), disconnected=True)
        except Exception as stop_error:
            # A secondary disarm failure must not escape the Kit callback or
            # replace the first fault. The latch prevents subsequent inference.
            if first_failure:
                note_failure(error, f"Callback disarm also failed: {stop_error!r}")

    def is_available():
        if callback_errors:
            raise RuntimeError("Keyboard callback failed") from callback_errors[0]
        return bool(
            env.sim.is_playing()
            and window.is_focused()
            and not window.get_input_blocking_state(carb.input.DeviceType.KEYBOARD)
            and (
                not simple
                or not any(
                    inputs.get_mouse_value(mouse, button)
                    for button in (
                        carb.input.MouseInput.LEFT_BUTTON,
                        carb.input.MouseInput.MIDDLE_BUTTON,
                        carb.input.MouseInput.RIGHT_BUTTON,
                    )
                )
            )
        )

    def on_key(event, *_):
        try:
            now = time.monotonic()
            available = is_available()
            if not available:
                control.stop(now, disconnected=True)
            event_type = {
                carb.input.KeyboardEventType.KEY_PRESS: "press",
                carb.input.KeyboardEventType.KEY_REPEAT: "repeat",
                carb.input.KeyboardEventType.KEY_RELEASE: "release",
            }.get(event.type)
            if event_type is None:
                # CHAR carries text, not a KeyboardInput enum. Do not inspect
                # its payload or treat text entry as a motion/arm/lease event.
                return True
            key_input = event.input
            key = (
                key_input
                if isinstance(key_input, str)
                else getattr(key_input, "name", None)
            )
            if not isinstance(key, str) or not key:
                raise ValueError("Keyboard key event requires a nonempty key name")
            modifiers = getattr(event, "modifiers", 0)
            if simple:
                previous_speed = control.speed_scale
                control.key_event(key, event_type, now, modifiers=modifiers)
                if control.speed_scale != previous_speed:
                    print(
                        f"[OPERATOR] Speed {control.speed_scale:.0%}; "
                        "applies on next fresh direction press.",
                        flush=True,
                    )
                # False prevents subsequent subscribers handling owned plain
                # keys. It cannot undo a shortcut handled by an earlier one.
                if (
                    available
                    and type(modifiers) is int
                    and modifiers >= 0
                    and modifiers & ~48 == 0  # Caps/NumLock are not navigation.
                    and key
                    in {*SINGLE_KEY_MOTION_KEYS, *SPEED_LEVELS, "X", "N", "ESCAPE"}
                ):
                    return False
            else:
                control.key_event(key, event_type, now, shift_held=bool(modifiers & 1))
        except Exception as error:
            callback_failed(error)
        return True

    def on_focus_change(_event):
        try:
            control.stop(time.monotonic(), disconnected=True)
        except Exception as error:
            callback_failed(error)

    subscription = inputs.subscribe_to_keyboard_events(keyboard, on_key)
    focus_subscription = None
    primary_error = None
    try:
        focus_subscription = (
            window.get_window_focus_event_stream().create_subscription_to_pop(
                on_focus_change
            )
        )
        if simple:
            print(
                "[OPERATOR] SIMULATION ONLY. Hold ONE key: arrows move forward/back/"
                "left/right; J/L pivot; U/O forward arcs. No R or Shift. "
                "Release requests stop. 1/2/3: 25/50/100% speed (default 50%, "
                "next fresh direction press). X: stop, N: physical reset, Esc: exit.",
                flush=True,
            )
            print(
                "[OPERATOR] Click viewport, then release mouse buttons. Fresh press "
                "starts immediately; 0.75s first-repeat grace, then 0.25s repeat lease. "
                "Focus loss, pause, mouse navigation, modifiers, conflicting keys, "
                "expiry or >0.25s host stalls stop commands. Release all motion keys "
                "before a fresh press. Zero twist is NOT zero motor action; "
                "client-fabricated repeats cannot certify remote presence.",
                flush=True,
            )
        else:
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
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_errors = []
        try:
            control.stop(time.monotonic(), disconnected=True)
        except Exception as error:
            cleanup_errors.append(("disarm", error))
        try:
            inputs.unsubscribe_to_keyboard_events(keyboard, subscription)
        except Exception as error:
            cleanup_errors.append(("keyboard unsubscribe", error))
        finally:
            # Kit's focus event subscription is retained/released by its handle.
            del focus_subscription
        if cleanup_errors:
            details = "; ".join(f"{name}: {error!r}" for name, error in cleanup_errors)
            if primary_error is not None:
                note_failure(primary_error, f"Keyboard cleanup failed: {details}")
            else:
                raise RuntimeError(f"Keyboard cleanup failed: {details}") from (
                    cleanup_errors[0][1]
                )
