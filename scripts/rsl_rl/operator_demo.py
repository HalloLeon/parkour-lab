"""One-shot simulated-time actor demonstration, not live-input validation.

This source owns its commands directly: no keyboard callbacks, focus checks,
network receipts, synthetic key events or input leases. Slow rendering extends
wall duration without changing the command/physics timeline. Simulation only.
"""

from __future__ import annotations

import math
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
TERRAIN_VERSION = "operator_scripted_terrain_demo_v2"
MOTION_SAMPLE_STEPS = 2
NONPLANE_TERRAINS = (
    "rough_flat",
    "hills",
    "step_hills",
    "tilted_ramps",
    "rough_stress",
)
# Initial yaw pi/2 is set by the demo-only scene override, not feedback steering.
# Forward commands cross the flat central strip to both sides. No reverse or
# lateral motion on slopes; these were not trained there. Magnitudes are inside
# the archived acquisition range (vx<=0.7, |yaw|<=0.8), not validated performance.
TERRAIN_SEGMENTS = (
    (500, "initial_stand", ZERO),
    (300, "outward", (0.7, 0.0, 0.0)),  # Nominal +4.2 m world Y.
    (100, "hold_outward", ZERO),
    (196, "turn_outward", (0.0, 0.0, 0.8)),  # Approximately pi radians.
    (50, "hold_after_turn_outward", ZERO),
    (600, "cross_terrain", (0.7, 0.0, 0.0)),  # Across center to Y=-4.2 m.
    (100, "hold_far_side", ZERO),
    (196, "turn_far_side", (0.0, 0.0, -0.8)),
    (50, "hold_after_turn_far_side", ZERO),
    (300, "return_to_center", (0.7, 0.0, 0.0)),
    (150, "final_stand", ZERO),
)


def _segments(terrain):
    if terrain == "plane":
        return SEGMENTS
    if terrain in NONPLANE_TERRAINS:
        return TERRAIN_SEGMENTS
    raise ValueError("Unsupported scripted demo terrain")


def demo_protocol(*, terrain="plane"):
    segments = _segments(terrain)
    start = 0
    phases = []
    for length, name, command in segments:
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
        "version": DEMO_VERSION if terrain == "plane" else TERRAIN_VERSION,
        "terrain": terrain,
        "presentation": (
            "full-speed long legs, half-turns and two opposing circles"
            if terrain == "plane"
            else "0.7 m/s forward terrain crossings with stops and +/-0.8 rad/s half-turns"
        ),
        "geometry_scope": "nominal command integrals only; no measured path closure or tracking guarantee",
        "control_steps": start,
        "simulated_seconds": start * STEP_SECONDS,
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
        "motion_sample_interval_steps": (
            None if terrain == "plane" else MOTION_SAMPLE_STEPS
        ),
        "motion_measurement": (
            "terrain demo only: every second nonterminal native return, identically for control and stress; root-link body velocity, sampled XY path and center-ray height; sampled observations, not complete extrema or under-foot support; diagnostic only, never actor inputs"
        ),
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
    def __init__(self, env, *, terrain="plane", clock=time.monotonic, sleep=time.sleep):
        self.env = env
        self.terrain = terrain
        self.segments = _segments(terrain)
        self.steps = sum(length for length, _, _ in self.segments)
        self.clock, self.sleep = clock, sleep
        self.control = _SequenceControl()
        self.current_step = 0
        self.executed_steps = 0
        self.last_attempted_step = None
        self.phase_counts = {name: 0 for _, name, _ in self.segments}
        self.reset_mask_steps = []
        self.terminal_event = None
        self.timings = {}
        self.completed = False
        self.started = self.wall_seconds = None
        self.initial_xy = self.previous_xy = None
        self.motion = {}

    def command_time(self):
        return self.current_step * STEP_SECONDS

    def prepare(self, step):
        if step != self.executed_steps or not 0 <= step < self.steps:
            raise RuntimeError("Scripted demo skipped or duplicated a control step")
        if self.terrain != "plane" and self.initial_xy is None:
            self.initial_xy = (
                self.env.scene["robot"].data.root_pos_w[0, :2].detach().cpu().tolist()
            )
            if not all(math.isfinite(v) for v in self.initial_xy):
                raise RuntimeError("Nonfinite initial demo root position")
            self.previous_xy = self.initial_xy
        self.current_step = step
        self.last_attempted_step = step
        start = 0
        for length, name, command in self.segments:
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
        # The shared motor bridge verifies native actions/targets independently
        # of adapter-private raw diagnostics. Check command ownership here.
        applied = self.env.command_manager.get_term("base_velocity").command
        if not torch.equal(applied, applied.new_tensor([decision.command])):
            raise RuntimeError("Scripted command differs from native command buffer")
        if self.terrain != "plane" and step % MOTION_SAMPLE_STEPS == 0:
            self.measure_motion(decision)

    def measure_motion(self, decision):
        """Sparse diagnostics only; never steer or inspect an auto-reset frame."""
        import torch

        robot = self.env.scene["robot"].data
        origin = self.env.scene.env_origins[0]
        values = (
            torch.cat(
                (
                    robot.root_pos_w[0, :2],
                    robot.root_pos_w[0, :2] - origin[:2],
                    robot.root_link_lin_vel_b[0, :2],
                    robot.root_ang_vel_b[0, 2:3],
                    self.env.scene["base_height_scanner"].data.ray_hits_w[0, :1, 2]
                    - origin[2],
                )
            )
            .detach()
            .cpu()
            .tolist()
        )
        x, y, local_x, local_y, vx, vy, wz, height = values
        if not all(math.isfinite(v) for v in values[:-1]):
            raise RuntimeError("Nonfinite demo motion diagnostics")
        phase = decision.status.removeprefix("scripted:")
        stats = self.motion.setdefault(
            phase,
            dict(
                samples=0,
                vx_sum=0.0,
                yaw_sum=0.0,
                vx_error_sum=0.0,
                yaw_error_sum=0.0,
                sampled_xy_path_m=0.0,
                maximum_xy_distance_from_start_m=0.0,
                untapered_region_moving_samples=0,
                finite_height_samples=0,
                missing_height_samples=0,
                minimum_surface_height_m=None,
                maximum_surface_height_m=None,
            ),
        )
        stats["samples"] += 1
        stats["vx_sum"] += vx
        stats["yaw_sum"] += wz
        stats["vx_error_sum"] += abs(vx - decision.command[0])
        stats["yaw_error_sum"] += abs(wz - decision.command[2])
        stats["sampled_xy_path_m"] += math.dist(self.previous_xy, (x, y))
        stats["maximum_xy_distance_from_start_m"] = max(
            stats["maximum_xy_distance_from_start_m"],
            math.dist(self.initial_xy, (x, y)),
        )
        self.previous_xy = (x, y)
        # Region excludes spawn/strip/edge tapers of the unchanged 16m surface.
        # Location and speed are not proof of terrain relief or foot support.
        if (
            2.0 <= max(abs(local_x), abs(local_y)) <= 6.0
            and abs(local_y) >= 1.6
            and decision.command[0] > 0.1
            and math.hypot(vx, vy) > 0.1
        ):
            stats["untapered_region_moving_samples"] += 1
        if math.isfinite(height):
            stats["finite_height_samples"] += 1
            for key, combine in (
                ("minimum_surface_height_m", min),
                ("maximum_surface_height_m", max),
            ):
                stats[key] = (
                    height if stats[key] is None else combine(stats[key], height)
                )
        else:
            stats["missing_height_samples"] += 1

    def progress(self):
        wall = self.wall_seconds
        if wall is None and self.started is not None:
            wall = self.clock() - self.started
        motion = {}
        for phase, values in self.motion.items():
            stats = dict(values)
            count = stats["samples"]
            for source, target in (
                ("vx_sum", "mean_root_link_forward_m_s"),
                ("yaw_sum", "mean_body_yaw_rad_s"),
                ("vx_error_sum", "mean_abs_forward_error_m_s"),
                ("yaw_error_sum", "mean_abs_yaw_error_rad_s"),
            ):
                stats[target] = stats.pop(source) / count
            motion[phase] = stats
        return {
            "version": DEMO_VERSION if self.terrain == "plane" else TERRAIN_VERSION,
            "terrain": self.terrain,
            "completed": self.completed,
            "executed_control_steps": self.executed_steps,
            "nonterminal_control_returns": self.executed_steps
            - int(self.terminal_event is not None),
            "excluded_terminal_returns": int(self.terminal_event is not None),
            "simulated_seconds": self.executed_steps * STEP_SECONDS,
            "last_attempted_step": self.last_attempted_step,
            "wall_seconds": wall,
            "real_time_factor": (
                self.executed_steps * STEP_SECONDS / wall if wall else None
            ),
            "motion_by_phase": motion,
            "motion_sample_interval_steps": (
                None if self.terrain == "plane" else MOTION_SAMPLE_STEPS
            ),
            "motion_scope": "every second nonterminal return, not extrema or acceptance; sampled path chords include previous phase boundary; terminal replacement scene and unsampled tail excluded; center-ray height is not foot contact or slope",
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

    def run(self, host, app, *, recording=None):
        print(
            f"[DEMO] SIMULATION ONLY: automatic one-shot {self.steps * STEP_SECONDS:g}s "
            f"simulation-time {self.terrain} showcase. "
            "No keyboard input; does not wait for viewer connection. "
            "Slow hosts take longer. Ctrl+C in the launch terminal stops the process.",
            flush=True,
        )
        self.started = self.clock()
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
                max_steps=self.steps,
                max_wall_seconds=WALL_SECONDS,
                before_poll=self.prepare,
                after_step=self.observe,
                timings=self.timings,
                **({"recording": recording} if recording is not None else {}),
            )
            if (
                result["control_steps"] != self.steps
                or self.executed_steps != self.steps
                or result["episode_resets"] != 0
                or result["manual_resets"] != 0
                or self.reset_mask_steps != [0]
                or self.phase_counts
                != {name: length for length, name, _ in self.segments}
                or self.terminal_event is not None
            ):
                raise RuntimeError(
                    "Scripted demo interrupted before full sequence completion"
                )
            self.completed = True
            result["demo_progress"] = self.progress()
            print(
                f"[DEMO] Delivered {self.steps * STEP_SECONDS:g} simulated seconds in "
                f"{result['wall_seconds']:.1f} wall seconds. "
                f"Real-time factor: {self.progress()['real_time_factor']:.3f}. "
                "Motion measurements are in report.json; completion is not traversal acceptance.",
                flush=True,
            )
            return result
        finally:
            self.wall_seconds = self.clock() - self.started
            self.control.stop(self.command_time())
