"""One-shot simulated-time actor demonstration, not live-input validation.

This source owns its commands directly: no keyboard callbacks, focus checks,
network receipts, synthetic key events or input leases. Slow rendering extends
wall duration without changing the command/physics timeline. Simulation only.
"""

from __future__ import annotations

import time

from parkour_lab.learning.command_source import LeaseDecision

from .operator_live import run_live_loop


STEP_SECONDS = 0.02
WALL_SECONDS = 1800.0
DEMO_VERSION = "operator_scripted_demo_v2"
ZERO = (0.0, 0.0, 0.0)
# Native steps, phase name, body-relative (vx m/s, vy m/s, yaw rad/s).
# Ten simulated seconds initially allow time to connect the viewer; this does
# not detect a connected viewer and is not a network-readiness handshake.
SEGMENTS = (
    (500, "initial_stand", ZERO),
    # Full existing command limits, not increased joint/motor action scales.
    (300, "forward", (0.4, 0.0, 0.0)),  # 6 s: nominal 2.4 m outward.
    (50, "stop_after_forward", ZERO),
    (400, "backward", (-0.3, 0.0, 0.0)),  # 8 s: nominal 2.4 m return.
    (50, "stop_after_backward", ZERO),
    (250, "left", (0.0, 0.2, 0.0)),  # 5 s: nominal 1 m sideways.
    (50, "stop_after_left", ZERO),
    (250, "right", (0.0, -0.2, 0.0)),
    (50, "stop_after_right", ZERO),
    (314, "pivot_left", (0.0, 0.0, 0.5)),  # 6.28 s: about 180 degrees.
    (50, "stop_after_pivot_left", ZERO),
    (314, "pivot_right", (0.0, 0.0, -0.5)),
    (50, "stop_after_pivot_right", ZERO),
    # Opposite ~360-degree loops, separated by a stop, trace a figure-eight
    # only under ideal tracking. Radius v/w = 1 m; no position-feedback steering.
    (786, "arc_left", (0.4, 0.0, 0.4)),
    (50, "stop_after_arc_left", ZERO),
    (786, "arc_right", (0.4, 0.0, -0.4)),
    (150, "final_stand", ZERO),
)
STEPS = sum(length for length, _, _ in SEGMENTS)


def demo_protocol():
    start = 0
    phases = []
    for length, name, command in SEGMENTS:
        phases.append(
            dict(
                name=name,
                start_step=start,
                end_step_exclusive=start + length,
                command_body_twist=list(command),
            )
        )
        start += length
    return {
        "version": DEMO_VERSION,
        "presentation": "full-speed long legs, half-turns and two opposing circles",
        "geometry_scope": "nominal command integrals only; no measured path closure or tracking guarantee",
        "control_steps": STEPS,
        "simulated_seconds": STEPS * STEP_SECONDS,
        "phases": phases,
        "command_units": ["m/s", "m/s", "rad/s"],
        "command_clock": "completed native control steps times 0.02 seconds",
        "loop_wall_timeout_s": WALL_SECONDS,
        "timeout_scope": "cooperative loop budget; cannot preempt a blocked native call",
        "wall_pacing": "at most 50 control steps per wall second; slow hosts are allowed",
        "source": "fixed one-shot sequence; no keyboard, focus or network input",
        "viewer_connection_check": "NONE; initial stand is not a connection handshake",
        "live_input_watchdog": "NOT_APPLICABLE_TO_SCRIPTED_SOURCE",
        "live_timing_validation": "UNRUN",
        "streamed_view_validation": "UNRUN; automatic execution cannot confirm client video",
        "terminal_policy": "abort on first terminal return; native env may already auto-reset inside that step",
        "completion": "full command sequence delivered, not behavioral acceptance",
        "behavioral_acceptance": False,
        "learning_updates": 0,
    }


class _SequenceControl:
    """Minimal loop command source; intentionally contains no lease or heartbeat."""

    reset_requested = False
    quit_requested = False

    def __init__(self):
        self.command = ZERO
        self.status = "scripted:initial_stand"

    def resolve(self, now):
        return LeaseDecision(self.command, now, None, None, 0, self.status)

    def poll(self, now, *, available):
        if not available:
            self.stop(now, disconnected=True)
        return self.resolve(now)

    def stop(self, now, *, disconnected=False):
        self.command = ZERO
        self.status = "scripted:paused" if disconnected else "scripted:finished"


class ScriptedDemo:
    def __init__(self, env, *, clock=time.monotonic, sleep=time.sleep):
        self.env = env
        self.clock, self.sleep = clock, sleep
        self.control = _SequenceControl()
        self.current_step = 0
        self.executed_steps = 0
        self.last_attempted_step = None
        self.phase_counts = {name: 0 for _, name, _ in SEGMENTS}
        self.reset_mask_steps = []
        self.terminal_event = None
        self.timings = {}
        self.completed = False

    def command_time(self):
        return self.current_step * STEP_SECONDS

    def prepare(self, step):
        if step != self.executed_steps or not 0 <= step < STEPS:
            raise RuntimeError("Scripted demo skipped or duplicated a control step")
        self.current_step = step
        self.last_attempted_step = step
        start = 0
        for length, name, command in SEGMENTS:
            if step < start + length:
                self.control.command = command
                self.control.status = f"scripted:{name}"
                return
            start += length

    def observe(self, step, decision, reset_mask, result, terminated, timed_out):
        import torch

        if step != self.executed_steps:
            raise RuntimeError("Scripted demo observer skipped or duplicated a step")
        # env.step has returned; count this completed step even if it terminated.
        self.executed_steps += 1
        self.phase_counts[decision.status.removeprefix("scripted:")] += 1
        ended, timeout = bool(terminated.any()), bool(timed_out.any())
        if ended or timeout:
            self.terminal_event = {
                "step": step,
                "terminated": ended,
                "timed_out": timeout,
                "post_reset_state_inspected": False,
            }
            raise RuntimeError(
                f"Scripted demo ended on episode termination/timeout at step {step}"
            )
        if reset_mask.shape != (1,) or reset_mask.dtype != torch.bool:
            raise RuntimeError("Invalid scripted demo actor reset mask")
        reset = bool(reset_mask.item())
        if reset != (step == 0):
            raise RuntimeError("Unexpected actor-memory reset during scripted demo")
        if reset:
            self.reset_mask_steps.append(step)
        # The shared host still binds/validates the stock motor contract. Verify
        # command and raw-action delivery, without collecting full traces/video.
        applied = self.env.command_manager.get_term("base_velocity").command
        if not torch.equal(applied, applied.new_tensor([decision.command])):
            raise RuntimeError("Scripted command differs from native command buffer")
        if not torch.equal(self.env.action_manager.action, result.raw_action):
            raise RuntimeError("Scripted action delivery differs from actor output")

    def progress(self):
        return {
            "version": DEMO_VERSION,
            "completed": self.completed,
            "executed_control_steps": self.executed_steps,
            "simulated_seconds": self.executed_steps * STEP_SECONDS,
            "last_attempted_step": self.last_attempted_step,
            "phase_counts": dict(self.phase_counts),
            "actor_reset_mask_steps": list(self.reset_mask_steps),
            "terminal_event": self.terminal_event,
            "host_call_timings": {
                name: dict(entry) for name, entry in self.timings.items()
            },
            "timing_scope": "host elapsed durations; GPU work may be charged at later synchronization",
            "live_timing_validation": "UNRUN",
            "streamed_view_validation": "UNRUN",
            "behavioral_acceptance": False,
        }

    def run(self, host, app):
        print(
            f"[DEMO] SIMULATION ONLY: automatic one-shot {STEPS * STEP_SECONDS:g}s "
            "simulation-time showcase at existing full command speeds. "
            "No keyboard input; does not wait for viewer connection. "
            "Slow hosts take longer. Ctrl+C in the launch terminal stops the process.",
            flush=True,
        )
        try:
            result = run_live_loop(
                self.env,
                host,
                app,
                self.control,
                is_available=lambda: True,
                clock=self.clock,
                command_clock=self.command_time,
                sleep=self.sleep,
                pace=True,
                max_steps=STEPS,
                max_wall_seconds=WALL_SECONDS,
                before_poll=self.prepare,
                after_step=self.observe,
                timings=self.timings,
            )
            if (
                result["control_steps"] != STEPS
                or self.executed_steps != STEPS
                or result["episode_resets"] != 0
                or result["manual_resets"] != 0
                or self.reset_mask_steps != [0]
                or self.phase_counts != {name: length for length, name, _ in SEGMENTS}
                or self.terminal_event is not None
            ):
                raise RuntimeError(
                    "Scripted demo interrupted before full sequence completion"
                )
            self.completed = True
            result["demo_progress"] = self.progress()
            return result
        finally:
            self.control.stop(self.command_time())
