"""Deterministic command-source loss probe; no simulator or learned-policy imports."""

from __future__ import annotations

import copy
import math

import numpy as np

ARMS = ("release", "disconnect", "silence", "replay")
STATUS_CODES = {
    name: i
    for i, name in enumerate(
        (
            "disarmed",
            "armed_waiting",
            "active",
            "expired",
            "released",
            "disconnected",
            "invalid_packet",
            "invalid_time",
        )
    )
}
EVENT_CODES = {"none": 0, "packet": 1, "release": 2, "disconnect": 3}
PREFIX = "command_source_"
STEPS, COUNT, PERIOD_S, LOSS_STEP = 800, 80, 0.02, 500


def command_source_protocol():
    """Fields layered onto the host's frozen recurrent evaluation protocol."""
    return {
        "version": "go2_operator_proprio_command_source_v1",
        "steps": STEPS,
        "phases": [
            {
                "name": name,
                "duration_s": duration,
                "flat_command": list(command),
                "rough_command": list(command),
            }
            for name, duration, command in (
                ("cold_stand", 2, (0, 0, 0)),
                ("forward", 8, (0.4, 0, 0)),
                ("source_loss", 6, (0, 0, 0)),
            )
        ],
        "initial_condition": {
            "assignment": "ascending env id within each of 20 columns; arm ranks 0..3",
            "arms": list(ARMS),
            "heading_rad_by_column_parity": [math.pi / 2, -math.pi / 2],
            "override": "yaw only once after seeded reset; no event/phase reset",
        },
        "source_lease": {
            "version": "operator_command_source_probe_v1",
            "lease_s": 0.25,
            "period_s": PERIOD_S,
            "loss_step": LOSS_STEP,
            "last_accepted_source_step": 499,
            "late_fresh_attempt_step": 525,
            "expected_zero_steps": [500, 500, 512, 512],
            "status_codes": copy.deepcopy(STATUS_CODES),
            "event_codes": copy.deepcopy(EVENT_CODES),
            "ordering": "source event, lease expiry/resolve, command sample, env.step",
            "clock": "deterministic pre-action time step*0.02; post-action state at (step+1)*0.02",
            "packet_policy": "fresh seq=step through 499; replay arm repeats seq499 after loss except fresh seq525; generation never changes",
            "phase_command_scope": "source_loss phase command is nominal zero; actual lease commands remain forward until per-arm zero step",
            "telemetry": "event admission and resolved status are separate; requested values/sequence/generation valid only when packet_present; source receipt valid only when source_present; zero_time valid only when zero_started; absent scalar sentinel -1, absent packet zero",
            "housekeeping": "resolve every arm, then zero commands for rows outside valid_first_attempt; no replacement credit",
            "physical_measurement": "post-physics/pre-reset, first attempt excluding terminal frame; planar speed is ROOT COM velocity expressed in body axes; displacement is ROOT LINK world-XY drift from its pre-action onset pose; heading is ROOT LINK world yaw; these are descriptive different reference points; rough-profile placement is not proven nonflat stopping",
            "settled_after_zero_s": 1.0,
        },
        "scope": (
            "Frozen healthy-controller command-source loss diagnostic on a deterministic "
            "simulation clock. Release, disconnect, silence and replay are injected source "
            "events, not real network, scheduler, sensor, watchdog or hardware fault "
            "certification. No learning, steering, phase reset, automatic recovery or acceptance."
        ),
    }


def assignment(columns):
    columns = np.asarray(columns)
    if (
        columns.shape != (COUNT,)
        or not np.issubdtype(columns.dtype, np.integer)
        or np.any((columns < 0) | (columns >= 20))
        or not np.array_equal(np.bincount(columns, minlength=20), np.full(20, 4))
    ):
        raise ValueError("Command-source probe requires four trials per terrain column")
    arms = np.empty(COUNT, dtype=np.int64)
    for column in range(20):
        arms[np.flatnonzero(columns == column)] = np.arange(4)
    headings = np.where(columns % 2 == 0, math.pi / 2, -math.pi / 2).astype(np.float32)
    return headings, arms


class CommandSourceProbe:
    """Four explicitly armed leases; events never mutate controller/GRU state."""

    def __init__(self, columns, period_s=PERIOD_S):
        from parkour_lab.learning.command_source import BodyTwistLease

        if isinstance(period_s, bool) or period_s != PERIOD_S:
            raise ValueError("Command-source probe requires 50 Hz")
        self.heading_rad, self.arm_ids = assignment(columns)
        self.leases = [BodyTwistLease(0.25) for _ in ARMS]
        self.generations = [lease.arm(0.0) for lease in self.leases]
        self.zero_times = [None] * 4
        self.next_step = 0

    def step(self, step_index):
        if (
            type(step_index) is not int
            or step_index != self.next_step
            or step_index >= STEPS
        ):
            raise ValueError(
                "Probe steps must be consecutive integers from 0 through 799"
            )
        now = step_index * PERIOD_S
        desired, rows = [], []
        for arm, lease in enumerate(self.leases):
            event, packet, accepted = "none", False, False
            requested, sequence, generation = (0.0, 0.0, 0.0), -1, -1
            if (
                step_index < LOSS_STEP
                or step_index == 525
                or (arm == 3 and step_index >= LOSS_STEP)
            ):
                event, packet = "packet", True
                sequence = (
                    step_index if step_index < LOSS_STEP or step_index == 525 else 499
                )
                generation = self.generations[arm]
                requested = (0.0 if step_index < 100 else 0.4, 0.0, 0.0)
                accepted = lease.receive(
                    requested, time_s=now, sequence=sequence, generation=generation
                )
            elif step_index == LOSS_STEP and arm < 2:
                event, accepted = ARMS[arm], True
                getattr(lease, event)(now)
            decision = lease.resolve(now)
            if (
                step_index >= LOSS_STEP
                and not any(decision.command)
                and self.zero_times[arm] is None
            ):
                self.zero_times[arm] = now
            source_present = decision.source_time_s is not None
            rows.append(
                {
                    "event": EVENT_CODES[event],
                    "event_present": event != "none",
                    "packet_present": packet,
                    "requested_command": requested,
                    "requested_sequence": sequence,
                    "requested_generation": generation,
                    "event_accepted": bool(accepted),
                    "source_present": source_present,
                    "source_time_s": decision.source_time_s if source_present else -1.0,
                    "current_sequence": (
                        decision.sequence if decision.sequence is not None else -1
                    ),
                    "generation": decision.generation,
                    "status": STATUS_CODES[decision.status],
                    "decision_time_s": decision.decision_time_s,
                    "zero_started": self.zero_times[arm] is not None,
                    "zero_time_s": (
                        self.zero_times[arm]
                        if self.zero_times[arm] is not None
                        else -1.0
                    ),
                }
            )
            desired.append(decision.command)
        self.next_step += 1
        telemetry = {
            PREFIX + key: np.asarray([row[key] for row in rows])[self.arm_ids]
            for key in rows[0]
        }
        telemetry[PREFIX + "requested_command"] = telemetry[
            PREFIX + "requested_command"
        ].astype(np.float32)
        return np.asarray(desired, dtype=np.float32)[self.arm_ids], telemetry


def _heading(quaternion):
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def validate_command_source_trace(trace, protocol):
    """Replay all admissions/decisions plus an independent hard-coded time oracle."""
    expected_protocol = command_source_protocol()
    if (
        any(protocol.get(k) != v for k, v in expected_protocol.items())
        or protocol.get("num_envs") != COUNT
        or protocol.get("period_s") != PERIOD_S
        or "command_limits" in protocol
        or "foot_telemetry" in protocol
        or "reward_telemetry" in protocol
    ):
        raise ValueError("Unreviewed command-source protocol")
    columns = np.asarray(trace["terrain_column_id"])
    headings, arms = assignment(columns)
    for key, expected in (("initial_heading_rad", headings), (PREFIX + "arm_id", arms)):
        if not np.array_equal(trace.get(key), expected):
            raise ValueError(f"Command-source assignment differs: {key}")
    shapes = {
        "command": (STEPS, COUNT, 3),
        "position": (STEPS, COUNT, 3),
        "pre_position": (STEPS, COUNT, 3),
        "linear_velocity_b": (STEPS, COUNT, 3),
        "quaternion": (STEPS, COUNT, 4),
        "pre_quaternion": (STEPS, COUNT, 4),
    }
    for key, shape in shapes.items():
        value = np.asarray(trace[key])
        if (
            value.shape != shape
            or not np.issubdtype(value.dtype, np.floating)
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"Invalid command-source physical field: {key}")
    for key in (
        "terminated",
        "time_out",
        "procedural_workspace",
        "valid_first_attempt",
        PREFIX + "reset_mask",
    ):
        if (
            np.asarray(trace[key]).shape != (STEPS, COUNT)
            or np.asarray(trace[key]).dtype != np.bool_
        ):
            raise ValueError(f"Invalid command-source mask: {key}")
    done = trace["terminated"] | trace["time_out"]
    valid = np.concatenate(
        (np.ones_like(done[:1]), ~np.maximum.accumulate(done[:-1], axis=0))
    )
    reset = np.concatenate((np.ones_like(done[:1]), done[:-1]))
    if not np.array_equal(trace["valid_first_attempt"], valid) or not np.array_equal(
        trace[PREFIX + "reset_mask"], reset
    ):
        raise ValueError("Source event/phase reset or replacement-trial credit")
    if np.any(trace["procedural_workspace"] & ~trace["time_out"]):
        raise ValueError("Workspace censoring requires a timeout")
    quat = np.zeros((COUNT, 4), dtype=np.float32)
    quat[:, 0], quat[:, 3] = np.cos(headings / 2), np.sin(headings / 2)
    if not np.allclose(trace["pre_quaternion"][0], quat, atol=1e-6, rtol=0):
        raise ValueError("Command-source initial heading differs")
    probe, packets, telemetry = CommandSourceProbe(columns), [], {}
    for step in range(STEPS):
        desired, fields = probe.step(step)
        packets.append(desired)
        for key, value in fields.items():
            telemetry.setdefault(key, []).append(value)
    for key, values in telemetry.items():
        expected = np.stack(values)
        actual = np.asarray(trace.get(key))
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError(f"Command-source telemetry replay differs: {key}")
    # Never let a matching producer/replayer bug certify a delayed stop or renewal.
    zero_steps = np.asarray([500, 500, 512, 512])[arms]
    ticks = np.arange(STEPS)[:, None]
    independent = np.zeros((STEPS, COUNT, 3), dtype=np.float32)
    independent[..., 0] = np.where(
        (ticks >= 100) & (ticks < zero_steps), np.float32(0.4), 0
    )
    if not np.array_equal(np.stack(packets), independent):
        raise ValueError("Lease resolution violates independent source-loss deadlines")
    expected_status = np.full((STEPS, COUNT), STATUS_CODES["active"], dtype=np.int64)
    for arm, status in enumerate(("released", "disconnected", "expired", "expired")):
        expected_status[(ticks >= zero_steps) & (arms[None] == arm)] = STATUS_CODES[
            status
        ]
    source_seq = np.broadcast_to(np.minimum(ticks, 499), (STEPS, COUNT))
    accepted = np.broadcast_to(ticks < LOSS_STEP, (STEPS, COUNT)).copy()
    accepted[LOSS_STEP, arms < 2] = True  # Explicit release/disconnect events.
    if (
        not np.array_equal(trace[PREFIX + "current_sequence"], source_seq)
        or not np.array_equal(trace[PREFIX + "source_time_s"], source_seq * PERIOD_S)
        or not np.array_equal(trace[PREFIX + "status"], expected_status)
        or not np.array_equal(trace[PREFIX + "event_accepted"], accepted)
        or not np.all(trace[PREFIX + "source_present"])
        or not np.array_equal(trace[PREFIX + "zero_started"], ticks >= zero_steps)
        or not np.array_equal(
            trace[PREFIX + "zero_time_s"],
            np.where(ticks >= zero_steps, zero_steps * PERIOD_S, -1.0),
        )
        or not np.all(trace[PREFIX + "generation"] == 1)
    ):
        raise ValueError("Source receipt, generation or original latch cause changed")
    independent[~valid] = 0
    if not np.array_equal(trace["command"], independent):
        raise ValueError(
            "Applied command differs from independent source-loss schedule"
        )
    speed = np.linalg.norm(trace["linear_velocity_b"][..., :2], axis=-1)
    heading, pre_heading = _heading(trace["quaternion"]), _heading(
        trace["pre_quaternion"]
    )
    trials = []
    for row, arm in enumerate(arms):
        terminal_indices = np.flatnonzero(done[:, row])
        terminal = int(terminal_indices[0]) if len(terminal_indices) else None
        end = (
            STEPS if terminal is None else terminal
        )  # Terminal frame earns no physical credit.
        physical = terminal is not None and bool(trace["terminated"][terminal, row])
        workspace = terminal is not None and bool(
            trace["procedural_workspace"][terminal, row]
        )
        complete = terminal in (None, STEPS - 1) and not physical and not workspace
        zero_step = int(zero_steps[row])

        def window(start):
            indices = np.arange(start, max(start, end))
            if not len(indices):
                return {
                    "samples": 0,
                    "duration_s": 0.0,
                    "mean_planar_speed_m_s": None,
                    "settled_mean_planar_speed_m_s": None,
                    "max_displacement_m": None,
                    "max_abs_heading_change_rad": None,
                }
            drift = (
                trace["position"][indices, row, :2]
                - trace["pre_position"][start, row, :2]
            )
            yaw = (
                np.unwrap(np.r_[pre_heading[start, row], heading[indices, row]])[1:]
                - pre_heading[start, row]
            )
            settled = indices[indices >= start + 50]
            return {
                "samples": len(indices),
                "duration_s": len(indices) * PERIOD_S,
                "mean_planar_speed_m_s": float(speed[indices, row].mean()),
                "settled_mean_planar_speed_m_s": (
                    float(speed[settled, row].mean()) if len(settled) else None
                ),
                "max_displacement_m": float(np.linalg.norm(drift, axis=-1).max()),
                "max_abs_heading_change_rad": float(np.abs(yaw).max()),
            }

        rejected = (
            trace[PREFIX + "packet_present"][:, row]
            & ~trace[PREFIX + "event_accepted"][:, row]
        )
        preloss = np.arange(450, min(500, end))
        trials.append(
            {
                "env_id": row,
                "arm": ARMS[int(arm)],
                "terrain_column_id": int(columns[row]),
                "first_terminal_step": terminal,
                "physical_failure": physical,
                "workspace_timeout": workspace,
                "horizon_completed": complete,
                "reached_loss": end > LOSS_STEP,
                "delivered_zero_observed": end > zero_step,
                "decision_zero_step": zero_step,
                "loss_to_decision_zero_s": (zero_step - LOSS_STEP) * PERIOD_S,
                "loss_to_delivered_zero_s": (
                    (zero_step - LOSS_STEP) * PERIOD_S if end > zero_step else None
                ),
                "last_source_to_decision_zero_s": (zero_step - 499) * PERIOD_S,
                "timeout_deadline_s": 499 * PERIOD_S + 0.25 if arm >= 2 else None,
                "timeout_deadline_lateness_s": (
                    zero_step * PERIOD_S - (499 * PERIOD_S + 0.25) if arm >= 2 else None
                ),
                "rejected_packet_events": int(rejected.sum()),
                "rejected_packet_events_before_terminal": int(rejected[:end].sum()),
                "preloss_motion_samples": len(preloss),
                "preloss_mean_planar_speed_m_s": (
                    float(speed[preloss, row].mean()) if len(preloss) else None
                ),
                "from_loss": window(LOSS_STEP),
                "from_applied_zero": window(zero_step),
            }
        )
    return {
        "version": expected_protocol["source_lease"]["version"],
        "policy_acceptance": False,
        "scope": expected_protocol["scope"],
        "physical_scope": expected_protocol["source_lease"]["physical_measurement"],
        "arms": {
            name: {
                "scheduled": 20,
                **{
                    key: sum(bool(t[key]) for t in trials if t["arm"] == name)
                    for key in (
                        "reached_loss",
                        "delivered_zero_observed",
                        "horizon_completed",
                        "physical_failure",
                        "workspace_timeout",
                    )
                },
                "loss_to_decision_zero_s": (int([500, 500, 512, 512][arm]) - LOSS_STEP)
                * PERIOD_S,
                "observed_delivered_zero_latencies_s": sorted(
                    {
                        t["loss_to_delivered_zero_s"]
                        for t in trials
                        if t["arm"] == name
                        and t["loss_to_delivered_zero_s"] is not None
                    }
                ),
                "full_postzero_window_observed": sum(
                    t["horizon_completed"] and t["delivered_zero_observed"]
                    for t in trials
                    if t["arm"] == name
                ),
            }
            for arm, name in enumerate(ARMS)
        },
        "trials": trials,
    }
