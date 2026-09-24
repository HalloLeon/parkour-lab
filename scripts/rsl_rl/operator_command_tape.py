"""Simulator adapters for portable applied-command tapes, never motor playback."""

from __future__ import annotations

from bisect import bisect_right
import time

from parkour_lab.learning.command_tape import (
    MAX_STEPS,
    PERIOD_S,
    TapeBuilder,
    validate_tape,
    write_tape,
)

from .operator_demo import _SequenceControl, WALL_SECONDS
from .operator_live import run_live_loop


def _check_boundary(step, reset_mask, terminated, timed_out):
    import torch

    if bool(terminated.any()) or bool(timed_out.any()):
        raise RuntimeError(
            f"Command sequence ended on native termination/timeout at step {step}"
        )
    if (
        reset_mask.shape != (1,)
        or reset_mask.dtype != torch.bool
        or bool(reset_mask.item()) != (step == 0)
    ):
        raise RuntimeError("Command sequences permit only the initial actor reset")


class CommandRecording:
    max_steps = MAX_STEPS

    def __init__(self, metadata):
        self.builder = TapeBuilder(metadata)
        self.error = None

    def before_reset(self):
        self.error = (
            "Manual reset aborted before execution; start a new recording after reset"
        )
        raise RuntimeError(self.error)

    def after_step(self, step, decision, reset_mask, result, terminated, timed_out):
        try:
            # A terminal return is still a completed native call; retain it in
            # the incomplete prefix, without inspecting its replacement scene.
            self.builder.append(step, decision.command)
            _check_boundary(step, reset_mask, terminated, timed_out)
        except Exception as error:
            self.error = str(error)
            raise

    def finish(self, path, *, completed, error=None):
        tape = self.builder.finish(completed=completed, error=self.error or error)
        write_tape(path, tape)
        return {key: tape[key] for key in ("complete", "steps", "sha256", "error")} | {
            "path": str(path)
        }


class CommandReplay:
    """Finite simulation-time source; current controller reads current sensing.

    This is not a keyboard/network lease and cannot attest operator presence.
    Scene identity is checked by the CLI before simulator launch; native motor
    identity is checked here after environment construction, before any action.
    """

    def __init__(self, env, tape, *, clock=time.monotonic, sleep=time.sleep):
        # Copy/validate to avoid later caller mutation changing a running tape.
        import copy

        self.tape = validate_tape(copy.deepcopy(tape))
        self.env = env
        self.steps = tape["steps"]
        self.starts = [segment["start_step"] for segment in tape["segments"]]
        self.clock, self.sleep = clock, sleep
        self.control = _SequenceControl()
        self.executed_steps = self.current_step = 0
        self.completed = False
        self.timings = {}
        self.protocol = {
            "tape_sha256": tape["sha256"],
            "steps": self.steps,
            "source_metadata": self.tape["metadata"],
            "clock": "completed native control steps times 0.02 seconds",
            "scope": "body-twist commands against fresh sensing, not recorded actions or physical trajectories",
            "live_watchdog_validation": "UNRUN",
            "behavioral_acceptance": False,
        }

    def prepare(self, step):
        if step != self.executed_steps or not 0 <= step < self.steps:
            raise RuntimeError("Replay skipped or duplicated a control step")
        self.current_step = step
        segment = self.tape["segments"][bisect_right(self.starts, step) - 1]
        self.control.command = tuple(segment["command"])
        self.control.status = (
            "replay:active" if any(self.control.command) else "replay:zero"
        )

    def observe(self, step, decision, reset_mask, result, terminated, timed_out):
        import torch

        self.executed_steps += 1
        _check_boundary(step, reset_mask, terminated, timed_out)
        applied = self.env.command_manager.get_term("base_velocity").command
        if not torch.equal(applied, applied.new_tensor([decision.command])):
            raise RuntimeError("Replay command differs from native command buffer")
        if not torch.equal(self.env.action_manager.action, result.raw_action):
            raise RuntimeError(
                "Replay motor delivery differs from current actor output"
            )

    def progress(self):
        return {
            "completed": self.completed,
            "executed_control_steps": self.executed_steps,
            "tape_sha256": self.tape["sha256"],
            "host_call_timings": self.timings,
            "behavioral_acceptance": False,
        }

    def run(self, host, app, *, recording=None):
        motor = self.tape["metadata"].get("portable_motor_sha256")
        if not motor or motor != host.motor_verification["portable_motor_sha256"]:
            raise ValueError("Replay requires the recorded native motor contract")
        print(
            f"[REPLAY] SIMULATION ONLY: {self.steps * PERIOD_S:g}s of recorded body commands; no keyboard input.",
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
                command_clock=lambda: self.current_step * PERIOD_S,
                sleep=self.sleep,
                max_steps=self.steps,
                max_wall_seconds=WALL_SECONDS,
                before_poll=self.prepare,
                after_step=self.observe,
                timings=self.timings,
                recording=recording,
            )
            if (
                result["control_steps"] != self.steps
                or self.executed_steps != self.steps
            ):
                raise RuntimeError("Command replay interrupted before completion")
            self.completed = True
            return {**result, "replay_progress": self.progress()}
        finally:
            self.control.stop(self.current_step * PERIOD_S)
