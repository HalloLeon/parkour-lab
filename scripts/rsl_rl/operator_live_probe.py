"""Fixed headless native-loop integration, not keyboard or behavioral acceptance.

Synthetic key events exercise the SAME lease, policy loop, physical reset and
motor delivery used by live play. By default the receipt clock is real monotonic
time. Explicit offline functional mode uses controlled command-test time instead
and cannot validate live responsiveness. Host timings and administrative timeout
always use real elapsed time. Only bounded aggregates are retained.
"""

from __future__ import annotations

import time

from .operator_live import KeyboardTwist, MOTION_KEYS, run_live_loop


STEPS = 600
WALL_SECONDS = 120.0
FUNCTIONAL_WALL_SECONDS = 600.0
RESET_STEP = 450
PHASES = (
    (0, 100, "initial_zero"),
    (100, 200, "forward"),
    (200, 250, "release_zero"),
    (250, 350, "pivot"),
    (350, 400, "silence"),
    (400, 450, "late_repeat_zero"),
    (450, 500, "physical_reset_zero"),
    (500, 550, "arc"),
    (550, 575, "unavailable_zero"),
    (575, 600, "recovered_without_arm_zero"),
)


def smoke_protocol(*, functional=False):
    return {
        "version": (
            "operator_native_functional_smoke_v1"
            if functional
            else "operator_native_live_smoke_v2"
        ),
        "control_steps": STEPS,
        "simulated_seconds": STEPS * 0.02,
        "loop_wall_timeout_s": FUNCTIONAL_WALL_SECONDS if functional else WALL_SECONDS,
        "manual_reset_step": RESET_STEP,
        "phases": [
            {"start_step": start, "end_step_exclusive": end, "name": name}
            for start, end, name in PHASES
        ],
        "source": "synthetic key events; no Kit keyboard, focus or network transport",
        "repeat_schedule": "one explicit synthetic repeat per intended-active poll; none during silence",
        "command_clock": (
            "controlled synthetic time: control-step index times 0.02 seconds; unchanged during host work or physical reset"
            if functional
            else "local monotonic receipt time"
        ),
        "command_lease_s": 0.25,
        "host_stall_timeout_s": None if functional else 0.25,
        "wall_clock_watchdog_validation": (
            "UNRUN" if functional else "IN_SCOPE_NOT_CERTIFIED"
        ),
        "real_time_validation": "UNRUN",
        "wall_pacing": not functional,
        "silence_oracle": "strict command decision time > last scripted repeat receipt + 0.25s",
        "scope": "one-plane native scene, causal actor, input lease, reset masks and motor delivery only",
        "gui_validation": "UNRUN",
        "behavioral_acceptance": False,
        "learning_updates": 0,
    }


class HeadlessSmoke:
    """Consume one-shot events once, even when a physical reset repeats a tick."""

    def __init__(
        self, env, *, functional=False, clock=time.monotonic, sleep=time.sleep
    ):
        if type(functional) is not bool:
            raise ValueError("Functional smoke selection must be a boolean")
        self.env = env
        self.functional = functional
        self.clock, self.sleep = clock, sleep
        self.control = KeyboardTwist()
        self.last_prepared_step = -1
        self.current_step = 0
        self.last_pivot_repeat_receipt = None
        self.verified_steps = 0
        self.last_attempted_step = None
        self.reset_mask_steps = []
        self.silence_expired_steps = 0
        self.phase_counts = {name: 0 for _, _, name in PHASES}
        self.max_abs_raw_action = 0.0
        self.max_abs_joint_speed_rad_s = 0.0
        self.max_root_com_planar_speed_m_s = 0.0
        self.timings = {}
        self.last_input_check = None

    def command_time(self):
        return self.current_step * 0.02 if self.functional else self.clock()

    def _event(self, key, kind):
        now = self.command_time()
        self.control.key_event(key, kind, now, shift_held=True)
        if kind == "repeat" and key == "Q" and 250 <= self.current_step < 350:
            self.last_pivot_repeat_receipt = now

    def prepare(self, step):
        if not self.env.sim.is_playing():
            raise RuntimeError("Headless smoke cannot wait for an interactive resume")
        if step == self.last_prepared_step:
            return  # N resets without advancing the policy clock; never replay N.
        if step != self.last_prepared_step + 1:
            raise RuntimeError("Smoke source skipped a control step")
        self.current_step = step
        self.last_prepared_step = step
        if step in (100, 250, 500):
            key = {100: "W", 250: "Q", 500: "Z"}[step]
            self._event("R", "press")
            self._event(key, "press")
        # Simulation ticks need not take 20ms of wall time. These are explicit
        # test-source receipts, not a keyboard heartbeat or automatic rearming.
        # Strict mode rejects real >250ms stalls. Functional mode advances only
        # explicit test time; it is never a fallback for live keyboard input.
        if 100 <= step < 200:
            self._event("W", "repeat")
        elif 250 <= step < 350 or 400 <= step < 450:
            self._event("Q", "repeat")
        elif 500 <= step < 550 or 575 <= step < STEPS:
            self._event("Z", "repeat")
        if step == 200:
            self._event("W", "release")
        if step == RESET_STEP:
            self._event("N", "press")
        if step == 560:
            # These must be discarded while unavailable, not queued for recovery.
            self._event("R", "press")
            self._event("N", "press")

    def available(self):
        return not 550 <= self.current_step < 575

    def _expected_command(self, step, decision_time):
        if 100 <= step < 200:
            return MOTION_KEYS["W"]
        if 250 <= step < 350:
            return MOTION_KEYS["Q"]
        if 350 <= step < 400:
            if self.last_pivot_repeat_receipt is None:
                raise RuntimeError("Smoke did not emit its required pivot repeats")
            if decision_time <= self.last_pivot_repeat_receipt + 0.25:
                return MOTION_KEYS["Q"]
        if 500 <= step < 550:
            return MOTION_KEYS["Z"]
        return (0.0, 0.0, 0.0)

    def observe(self, step, decision, reset_mask, result, terminated, timed_out):
        import torch

        self.last_attempted_step = step
        # Native step auto-resets before returning. Never treat its replacement
        # state as a successful observation of the failed physical attempt.
        if terminated.any() or timed_out.any():
            raise RuntimeError(
                f"Unexpected episode termination/timeout at smoke step {step}"
            )
        if step != self.verified_steps:
            raise RuntimeError("Smoke observer skipped or duplicated a physical step")
        expected = self._expected_command(step, decision.decision_time_s)
        self.last_input_check = {
            "step": step,
            "status": decision.status,
            "expected_command": list(expected),
            "actual_command": list(decision.command),
            "generation": decision.generation,
            "poll_gap_s": self.control.last_poll_gap_s,
            "source_receipt_age_s": (
                None
                if decision.source_time_s is None
                else decision.decision_time_s - decision.source_time_s
            ),
        }
        if decision.command != expected:
            raise RuntimeError(
                f"Synthetic input command mismatch at smoke step {step}: "
                f"status={decision.status}, poll_gap_s={self.control.last_poll_gap_s:.6f}, "
                f"expected={expected}, actual={decision.command}. "
                "The 0.25s lease remains enforced in the declared command clock; "
                "see smoke_progress timing scopes."
            )
        expected_generation = (
            0 if step < 100 else 1 if step < 250 else 2 if step < 500 else 3
        )
        if decision.generation != expected_generation:
            raise RuntimeError(f"Unexpected source arm generation at smoke step {step}")
        if reset_mask.shape != (1,) or reset_mask.dtype != torch.bool:
            raise RuntimeError("Invalid smoke actor reset mask")
        reset = bool(reset_mask.item())
        if reset != (step in (0, RESET_STEP)):
            raise RuntimeError(
                f"Unexpected actor-memory reset mask at smoke step {step}"
            )
        robot = self.env.scene["robot"].data
        term = self.env.action_manager.get_term("joint_pos")
        applied = self.env.command_manager.get_term("base_velocity").command
        if not torch.equal(applied, applied.new_tensor([expected])):
            raise RuntimeError("Native command buffer differs from the lease decision")
        if not all(
            torch.equal(value, result.position_rad)
            for value in (term.processed_actions, robot.joint_pos_target)
        ):
            raise RuntimeError(
                "Native motor target differs from the verified actor target"
            )
        for name, value, shape in (
            ("position", robot.root_pos_w, (1, 3)),
            ("orientation", robot.root_quat_w, (1, 4)),
            ("linear velocity", robot.root_lin_vel_b, (1, 3)),
            ("angular velocity", robot.root_ang_vel_b, (1, 3)),
            ("joint position", robot.joint_pos, (1, 12)),
            ("joint velocity", robot.joint_vel, (1, 12)),
            ("motor target", robot.joint_pos_target, (1, 12)),
            ("raw action", self.env.action_manager.action, (1, 12)),
        ):
            if value.shape != shape or not torch.isfinite(value).all():
                raise RuntimeError(f"Invalid post-step {name} at smoke step {step}")
        if 350 <= step < 400 and expected == (0.0, 0.0, 0.0):
            if decision.status != "expired":
                raise RuntimeError("Silent source did not expire through its lease")
            self.silence_expired_steps += 1
        for start, end, name in PHASES:
            if start <= step < end:
                self.phase_counts[name] += 1
                break
        if reset:
            self.reset_mask_steps.append(step)
        self.max_abs_raw_action = max(
            self.max_abs_raw_action, float(self.env.action_manager.action.abs().max())
        )
        self.max_abs_joint_speed_rad_s = max(
            self.max_abs_joint_speed_rad_s, float(robot.joint_vel.abs().max())
        )
        self.max_root_com_planar_speed_m_s = max(
            self.max_root_com_planar_speed_m_s,
            float(robot.root_lin_vel_b[:, :2].norm(dim=-1).max()),
        )
        self.verified_steps += 1

    def progress(self):
        protocol = smoke_protocol(functional=self.functional)
        return {
            "version": protocol["version"],
            "command_clock": protocol["command_clock"],
            "input_timing_scope": "poll gaps and receipt ages use the declared command clock, not necessarily wall time",
            "wall_clock_watchdog_validation": protocol[
                "wall_clock_watchdog_validation"
            ],
            "real_time_validation": "UNRUN",
            "verified_control_steps": self.verified_steps,
            "verification_scope": "observer-local input/reset/state/target checks; full native delivery equality is motor_delivery.verified_delivery_steps",
            "last_attempted_step": self.last_attempted_step,
            "phase_counts": dict(self.phase_counts),
            "actor_reset_mask_steps": list(self.reset_mask_steps),
            "silence_expired_steps": self.silence_expired_steps,
            "max_abs_raw_action": self.max_abs_raw_action,
            "max_abs_joint_speed_rad_s": self.max_abs_joint_speed_rad_s,
            "max_root_com_body_xy_speed_m_s": self.max_root_com_planar_speed_m_s,
            "last_input_check": self.last_input_check,
            "max_poll_gap_s": self.control.max_poll_gap_s,
            "host_call_timings": {
                name: dict(entry) for name, entry in self.timings.items()
            },
            "timing_scope": "host elapsed durations; GPU work may be charged at its next synchronization, not kernel profiling",
            "metrics_scope": "descriptive post-control-step maxima, not substep extrema or behavioral thresholds",
            "gui_validation": "UNRUN",
        }

    def run(self, host, app):
        try:
            result = run_live_loop(
                self.env,
                host,
                app,
                self.control,
                is_available=self.available,
                clock=self.clock,
                command_clock=self.command_time,
                sleep=self.sleep,
                pace=not self.functional,
                max_steps=STEPS,
                max_wall_seconds=(
                    FUNCTIONAL_WALL_SECONDS if self.functional else WALL_SECONDS
                ),
                before_poll=self.prepare,
                after_step=self.observe,
                timings=self.timings,
            )
            if (
                result["control_steps"] != STEPS
                or self.verified_steps != STEPS
                or result["manual_resets"] != 1
                or result["episode_resets"] != 0
                or self.reset_mask_steps != [0, RESET_STEP]
                or self.phase_counts
                != {name: end - start for start, end, name in PHASES}
                or self.silence_expired_steps == 0
            ):
                raise RuntimeError(
                    "Headless smoke did not complete its full declared protocol"
                )
            return result
        finally:
            self.control.stop(self.command_time(), disconnected=True)
