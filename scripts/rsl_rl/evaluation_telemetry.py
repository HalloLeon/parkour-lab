# SPDX-License-Identifier: BSD-3-Clause

"""Simulator-independent, streaming telemetry for single-robot flat-ground trials."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path


def _euler(root: Sequence[float]) -> tuple[float, float, float]:
    """Return roll, pitch and heading from an Isaac-style (w, x, y, z) quaternion."""
    w, x, y, z = root[3:7]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("Telemetry requires a nonzero root quaternion.")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return (
        math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
        math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))),
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
    )


def _angle_delta(end: float, start: float) -> float:
    return math.atan2(math.sin(end - start), math.cos(end - start))


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _tracking(rows: list[dict]) -> dict:
    errors = [row["world_angular_velocity_z_rad_s"] - row["command_yaw_rate_rad_s"] for row in rows]
    signed = [row for row in rows if abs(row["command_yaw_rate_rad_s"]) > 1e-6]
    return {
        "sample_count": len(rows),
        "mean_world_angular_velocity_z_rad_s": _mean([row["world_angular_velocity_z_rad_s"] for row in rows]),
        "mean_abs_error_rad_s": _mean([abs(error) for error in errors]),
        "rms_error_rad_s": math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None,
        "wrong_sign_sample_fraction": _mean(
            [float(row["world_angular_velocity_z_rad_s"] * row["command_yaw_rate_rad_s"] < 0.0) for row in signed]
        ),
    }


class EvaluationTelemetry:
    """Write raw post-physics samples and phase summaries, never reset poses.

    ``root_start`` and ``root_end`` must bracket the same physics step, before
    any automatic reset. Foot heights refer to rigid-body origins above the
    analytic level-zero support surface, not sole clearance. World angular
    velocity z is reported separately from unwrapped Euler heading changes.
    """

    ACQUISITION_S = 0.5
    TIME_AVERAGE_S = 0.4

    def __init__(
        self,
        directory: str,
        step_dt: float,
        foot_names: Sequence[str],
        *,
        phase_names: Sequence[str] = (),
        phase_signal_names: Sequence[str] = (),
    ):
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("Telemetry step_dt must be positive and finite.")
        if len(set(foot_names)) != len(foot_names):
            raise ValueError("Telemetry foot names must be unique.")
        if len(set(phase_names)) != len(phase_names) or len(set(phase_signal_names)) != len(phase_signal_names):
            raise ValueError("Telemetry diagnostic names must be unique.")
        if phase_signal_names and not phase_names:
            raise ValueError("Telemetry diagnostic signals require phase names.")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.step_dt = step_dt
        self.foot_names = tuple(foot_names)
        self.phase_names = tuple(phase_names)
        self._phase_signal_columns = {
            name: name if name.startswith("reward_") else f"diagnostic_{name}" for name in phase_signal_names
        }
        self._file = (self.directory / "telemetry.csv").open("w", newline="", encoding="utf-8")
        columns = [
            "step",
            "episode",
            "window",
            "phase",
            "episode_time_s",
            "phase_time_s",
            "command_speed_m_s",
            "command_yaw_rate_rad_s",
            "root_x_m",
            "root_y_m",
            "root_z_m",
            "roll_rad",
            "pitch_rad",
            "heading_rad",
            "unwrapped_heading_rad",
            "world_angular_velocity_z_rad_s",
            "done",
            "success",
        ]
        for name in self.foot_names:
            columns.extend((f"{name}_contact", f"{name}_body_origin_height_above_support_m"))
        if self.phase_names:
            columns.extend(("diagnostic_phase_id", "diagnostic_phase", "diagnostic_phase_time_s"))
            columns.extend(self._phase_signal_columns.values())
        self._writer = csv.DictWriter(self._file, fieldnames=columns)
        self._writer.writeheader()
        self._rows: list[dict] = []
        self._windows: list[dict] = []
        self._step = self._episode = self._episode_step = 0
        self._previous_yaw: float | None = None
        self._heading = 0.0
        self._key: tuple | None = None
        self._baseline: tuple[float, float, float] | None = None
        self._metadata: dict | None = None

    def record(
        self,
        *,
        command_speed_m_s: float,
        command_yaw_rate_rad_s: float,
        root_start: Sequence[float],
        root_end: Sequence[float],
        yaw_rate_rad_s: float,
        foot_contact: Sequence[bool],
        foot_height_above_support_m: Sequence[float],
        done: bool,
        success: bool,
        phase_diagnostics: Mapping[str, float | int] | None = None,
    ) -> None:
        if self._metadata is not None:
            raise RuntimeError("Cannot record telemetry after close().")
        if len(root_start) != 7 or len(root_end) != 7:
            raise ValueError("Telemetry root poses must contain x, y, z, qw, qx, qy, qz.")
        if len(foot_contact) != len(self.foot_names) or len(foot_height_above_support_m) != len(self.foot_names):
            raise ValueError("Telemetry foot arrays must match foot_names.")
        if not all(
            math.isfinite(value)
            for value in (
                command_speed_m_s,
                command_yaw_rate_rad_s,
                yaw_rate_rad_s,
                *root_start,
                *root_end,
                *foot_height_above_support_m,
            )
        ):
            raise ValueError("Telemetry samples must be finite.")
        diagnostic_row = self._diagnostic_row(phase_diagnostics)
        start_yaw = _euler(root_start)[2]
        roll, pitch, end_yaw = _euler(root_end)
        start_heading = (
            start_yaw if self._previous_yaw is None else self._heading + _angle_delta(start_yaw, self._previous_yaw)
        )
        phase = (
            "translating"
            if abs(command_speed_m_s) > 1e-6
            else "pivot" if abs(command_yaw_rate_rad_s) > 1e-6 else "stop"
        )
        key = (phase, command_yaw_rate_rad_s if phase == "pivot" else None)
        if key != self._key and self._rows:
            self._finish_window("command_change")
        if not self._rows:
            self._key = key
            self._baseline = (root_start[0], root_start[1], start_heading)
        self._heading = start_heading + _angle_delta(end_yaw, start_yaw)
        self._previous_yaw = end_yaw
        row = {
            "step": self._step,
            "episode": self._episode,
            "window": len(self._windows),
            "phase": phase,
            "episode_time_s": (self._episode_step + 1) * self.step_dt,
            "phase_time_s": (len(self._rows) + 1) * self.step_dt,
            "command_speed_m_s": command_speed_m_s,
            "command_yaw_rate_rad_s": command_yaw_rate_rad_s,
            "root_x_m": root_end[0],
            "root_y_m": root_end[1],
            "root_z_m": root_end[2],
            "roll_rad": roll,
            "pitch_rad": pitch,
            "heading_rad": end_yaw,
            "unwrapped_heading_rad": self._heading,
            "world_angular_velocity_z_rad_s": yaw_rate_rad_s,
            "done": bool(done),
            "success": bool(success),
        }
        for name, contact, height in zip(self.foot_names, foot_contact, foot_height_above_support_m):
            row[f"{name}_contact"] = bool(contact)
            row[f"{name}_body_origin_height_above_support_m"] = height
        row.update(diagnostic_row)
        self._writer.writerow(row)
        self._rows.append(row)
        self._step += 1
        self._episode_step += 1
        if done:
            self._finish_window("episode_success" if success else "episode_failure")
            self._episode += 1
            self._episode_step = 0
            self._previous_yaw = None

    def _diagnostic_row(self, sample: Mapping[str, float | int] | None) -> dict:
        """Validate the optional fixed schema before changing recorder state."""
        if not self.phase_names:
            if sample is not None:
                raise ValueError("Configure telemetry phase names before recording diagnostics.")
            return {}
        expected = {"phase_id", "phase_time_s", *self._phase_signal_columns}
        if sample is None or set(sample) != expected:
            raise ValueError("Telemetry phase diagnostics must match the configured signal schema.")
        if not all(math.isfinite(value) for value in sample.values()):
            raise ValueError("Telemetry phase diagnostics must be finite.")
        phase_id = int(sample["phase_id"])
        if phase_id != sample["phase_id"] or not 0 <= phase_id < len(self.phase_names):
            raise ValueError("Telemetry diagnostic phase_id is outside the configured phases.")
        if sample["phase_time_s"] < 0.0:
            raise ValueError("Telemetry diagnostic phase time must be non-negative.")
        return {
            "diagnostic_phase_id": phase_id,
            "diagnostic_phase": self.phase_names[phase_id],
            "diagnostic_phase_time_s": sample["phase_time_s"],
            **{column: sample[name] for name, column in self._phase_signal_columns.items()},
        }

    def _phase_summaries(self, rows: list[dict]) -> dict:
        """Summarize observed subphases without splitting legacy command windows."""
        summaries = {}
        for phase_id, phase in enumerate(self.phase_names):
            samples = [row for row in rows if row["diagnostic_phase_id"] == phase_id]
            if not samples:
                continue
            summaries[phase] = {
                "sample_count": len(samples),
                "duration_s": len(samples) * self.step_dt,
                "signal_means": {
                    name: _mean([row[column] for row in samples]) for name, column in self._phase_signal_columns.items()
                },
            }
        return summaries

    def _time_averages(self, rows: list[dict]) -> dict:
        """Average complete, nonoverlapping 0.4 s blocks (not detected strides)."""
        count = int((len(rows) * self.step_dt + 1e-9) / self.TIME_AVERAGE_S)
        blocks = []
        for block in range(count):
            start, end = block * self.TIME_AVERAGE_S, (block + 1) * self.TIME_AVERAGE_S
            yaw_integral = command_integral = 0.0
            for index in range(int(start / self.step_dt), min(len(rows), math.ceil(end / self.step_dt))):
                overlap = max(0.0, min(end, (index + 1) * self.step_dt) - max(start, index * self.step_dt))
                yaw_integral += overlap * rows[index]["world_angular_velocity_z_rad_s"]
                command_integral += overlap * rows[index]["command_yaw_rate_rad_s"]
            blocks.append(
                {
                    "start_s": start,
                    "end_s": end,
                    "mean_world_angular_velocity_z_rad_s": yaw_integral / self.TIME_AVERAGE_S,
                    "mean_command_yaw_rate_rad_s": command_integral / self.TIME_AVERAGE_S,
                    "abs_error_rad_s": abs(yaw_integral - command_integral) / self.TIME_AVERAGE_S,
                }
            )
        return {
            "window_s": self.TIME_AVERAGE_S,
            "full_window_count": count,
            "discarded_tail_s": max(0.0, len(rows) * self.step_dt - count * self.TIME_AVERAGE_S),
            "mean_abs_error_rad_s": _mean([block["abs_error_rad_s"] for block in blocks]),
            "windows": blocks,
        }

    def _finish_window(self, end_reason: str) -> None:
        rows, baseline = self._rows, self._baseline
        assert rows and baseline is not None
        target_angle = sum(row["command_yaw_rate_rad_s"] for row in rows) * self.step_dt
        achieved_angle = rows[-1]["unwrapped_heading_rad"] - baseline[2]
        tail = [row for index, row in enumerate(rows) if index * self.step_dt >= self.ACQUISITION_S - 1e-9]
        self._windows.append(
            {
                "window": len(self._windows),
                "episode": rows[0]["episode"],
                "phase": rows[0]["phase"],
                "complete": end_reason != "recording_stopped",
                "end_reason": end_reason,
                "sample_count": len(rows),
                "duration_s": len(rows) * self.step_dt,
                "start_unwrapped_heading_rad": baseline[2],
                "end_unwrapped_heading_rad": rows[-1]["unwrapped_heading_rad"],
                "commanded_angle_rad": target_angle,
                "achieved_heading_change_rad": achieved_angle,
                "heading_error_rad": achieved_angle - target_angle,
                "world_angular_velocity_z_integral_rad": sum(row["world_angular_velocity_z_rad_s"] for row in rows)
                * self.step_dt,
                "instantaneous_world_yaw_rate_tracking": _tracking(rows),
                "post_acquisition_world_yaw_rate_tracking": {
                    "excluded_initial_s": self.ACQUISITION_S,
                    **_tracking(tail),
                },
                "time_averaged_world_yaw_rate_tracking": self._time_averages(rows),
                "max_planar_excursion_m": max(
                    math.hypot(row["root_x_m"] - baseline[0], row["root_y_m"] - baseline[1]) for row in rows
                ),
                "max_abs_roll_rad": max(abs(row["roll_rad"]) for row in rows),
                "max_abs_pitch_rad": max(abs(row["pitch_rad"]) for row in rows),
                "feet": {
                    name: {
                        "contact_fraction": _mean([float(row[f"{name}_contact"]) for row in rows]),
                        "max_swing_body_origin_height_above_support_m": max(
                            (
                                row[f"{name}_body_origin_height_above_support_m"]
                                for row in rows
                                if not row[f"{name}_contact"]
                            ),
                            default=None,
                        ),
                    }
                    for name in self.foot_names
                },
            }
        )
        if self.phase_names:
            self._windows[-1]["phase_diagnostics"] = self._phase_summaries(rows)
        self._rows = []

    def close(self) -> dict:
        """Flush artifacts; mark a still-active phase partial, and return metadata."""
        if self._metadata is not None:
            return self._metadata
        if self._rows:
            self._finish_window("recording_stopped")
        self._file.close()
        pivots = [window for window in self._windows if window["phase"] == "pivot" and window["complete"]]
        self._metadata = {
            "schema_version": 2 if self.phase_names else 1,
            "samples_file": "telemetry.csv",
            "command_windows_file": "command_windows.json",
            "step_dt_s": self.step_dt,
            "foot_names": list(self.foot_names),
            "sample_count": self._step,
            "completed_episode_count": self._episode,
            "window_count": len(self._windows),
            "complete_window_count": sum(window["complete"] for window in self._windows),
            "partial_window_count": sum(not window["complete"] for window in self._windows),
            "sample_timing": "Commands are pre-physics; state and contacts are post-physics, before automatic reset.",
            "complete_window_definition": "Observed until command change or episode termination; not a command-success verdict.",
            "heading_definition": "Unwrapped Euler yaw, reset per episode; distinct from the integral of world angular velocity z.",
            "foot_height_definition": "Rigid-body origin above analytic level-zero support; not sole clearance.",
            "time_average_definition": "Nonoverlapping complete 0.4 s time windows, not detected or phase-aligned strides.",
            "complete_pivot_windows": {
                "count": len(pivots),
                "command_change_count": sum(window["end_reason"] == "command_change" for window in pivots),
                "episode_terminated_count": sum(window["end_reason"].startswith("episode_") for window in pivots),
                "episode_failure_count": sum(window["end_reason"] == "episode_failure" for window in pivots),
                "mean_abs_heading_error_rad": _mean([abs(window["heading_error_rad"]) for window in pivots]),
                "mean_instantaneous_abs_world_yaw_rate_error_rad_s": _mean(
                    [window["instantaneous_world_yaw_rate_tracking"]["mean_abs_error_rad_s"] for window in pivots]
                ),
            },
        }
        if self.phase_names:
            self._metadata["phase_diagnostics"] = {
                "phase_names": list(self.phase_names),
                "signal_columns": self._phase_signal_columns,
                "phase_definition": (
                    "Diagnostic subphases are independent of legacy command windows. Restart and pivot acquisition "
                    "cover the first second; the command-mode clock does not reset at their end."
                ),
                "reward_definition": (
                    "Signed weighted reward rates under the evaluated runtime config, before multiplying by step_dt. "
                    "reward_total_rate is the full objective, excluding the zero-valued logging term; selected terms "
                    "are not an exhaustive decomposition. "
                    "Pivot yaw and stability components decompose stationary_velocity_tracking; do not add them twice."
                ),
                "summary_definition": (
                    "Means use only observed samples in each diagnostic phase; missing phases are omitted. "
                    "Sample sums equal signal_means times sample_count; rate integrals also multiply by step_dt_s. "
                    "Command-window completeness still applies, including recording-stopped partial windows."
                ),
            }
        with (self.directory / "command_windows.json").open("w", encoding="utf-8") as handle:
            json.dump({"metadata": self._metadata, "windows": self._windows}, handle, indent=2, allow_nan=False)
            handle.write("\n")
        return self._metadata
