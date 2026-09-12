"""Read-only action/reference and terminal-safe actuator/contact diagnostics.

This captures the unchanged stock benchmark, not a new controller or training
recipe. Reference outputs are counterfactual actions on learner-visited states;
they are never executed and do not demonstrate counterfactual recovery.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

try:
    from .operator_benchmark_core import (
        DT,
        file_sha256,
        load_reference_actor,
        read_yaml_data,
    )
except ImportError:
    from operator_benchmark_core import (
        DT,
        file_sha256,
        load_reference_actor,
        read_yaml_data,
    )


VERSION = "operator_control_trace_v1"


def load_diagnostic_reference(checkpoint: Path, reference: Path):
    """Require matching saved physical/observation contracts; preserve CPU RNG."""
    checkpoint, reference = (
        checkpoint.resolve(strict=True),
        reference.resolve(strict=True),
    )
    environment_hash = file_sha256(checkpoint.parent / "params/env.yaml")
    if file_sha256(reference.parent / "params/env.yaml") != environment_hash:
        raise ValueError("Diagnostic reference must have the same saved env.yaml")
    agent_path = reference.parent / "params/agent.yaml"
    # Actor construction initializes layers before loading weights. Diagnostics
    # must not consume the RNG sequence used by environment initialization.
    with torch.random.fork_rng(devices=[]):
        actor, iteration = load_reference_actor(reference, read_yaml_data(agent_path))
    return actor, {
        "checkpoint": str(reference),
        "iteration": iteration,
        "sha256": {
            "checkpoint": file_sha256(reference),
            "env.yaml": environment_hash,
            "agent.yaml": file_sha256(agent_path),
        },
    }


def _snapshot(values):
    return {
        name: value.detach().to("cpu", copy=True).numpy()
        for name, value in values.items()
    }


class OperatorControlTrace:
    """Pair the actual pre-action frame with the post-physics/pre-reset state."""

    def __init__(self, env, reference):
        if reference.training or any(p.requires_grad for p in reference.parameters()):
            raise ValueError("Diagnostic reference must be frozen and in eval mode")
        self.env, self.reference = env, reference
        self.pending = None
        self.samples = []
        robot, sensor = env.scene["robot"], env.scene["contact_forces"]
        joints, bodies, contacts = (
            list(robot.joint_names),
            list(robot.body_names),
            list(sensor.body_names),
        )
        feet = [name for name in bodies if name.endswith("_foot")]
        if (
            len(joints) != 12
            or len(set(joints)) != 12
            or len(set(bodies)) != len(bodies)
            or len(set(contacts)) != len(contacts)
            or len(feet) != 4
            or not set(feet).issubset(contacts)
        ):
            raise ValueError(
                "Require twelve joints and four uniquely resolved Go2 feet"
            )
        self.foot_ids = [bodies.index(name) for name in feet]
        self.metadata = {
            "schema_version": VERSION,
            "step_dt_s": DT,
            "joint_names": joints,
            "foot_names": feet,
            "contact_body_names": contacts,
            "observation_pre": "exact delivered 48-D frame; command at 9:12, previous raw action at 36:48",
            "action": "original actor's raw mean action, never replaced",
            "reference_action": "frozen reference on that same frame; shadow only",
            "pre_fields": "before env.step and action processing",
            "post_fields": "record_post_step: after physics, before any auto-reset",
            "joint_units": "position/target rad; velocity rad/s; actuator torque N m",
            "contact_normal_force_w": "N; net NORMAL forces only, not tangential friction or full contact wrench",
            "foot_velocity_w": "m/s; body linear velocity in world frame, not a contact-point slip measurement",
            "sampling": "50 Hz; torque/contact fields are last-substep snapshots, not peaks over all physics substeps",
            "scope": "diagnostic only; no action blending, reference rollout, training, or acceptance relaxation",
        }

    def before_step(self, observation, action):
        if self.pending is not None:
            raise RuntimeError("Previous diagnostic action lacks a pre-reset capture")
        with torch.inference_mode():
            reference_action = self.reference(observation)
        if (
            observation.ndim != 2
            or observation.shape[1] != 48
            or action.shape != (len(observation), 12)
            or reference_action.shape != action.shape
            or not all(
                torch.isfinite(x).all() for x in (observation, action, reference_action)
            )
        ):
            raise ValueError("Invalid diagnostic observation or action")
        robot = self.env.scene["robot"].data
        self.pending = _snapshot(
            {
                "observation_pre": observation,
                "action": action,
                "reference_action": reference_action,
                "joint_position_pre": robot.joint_pos,
                "joint_velocity_pre": robot.joint_vel,
            }
        )

    def after_step(self):
        if self.pending is None:
            raise RuntimeError("Post-physics diagnostics lack the delivered action")
        robot = self.env.scene["robot"].data
        sample = {
            **self.pending,
            **_snapshot(
                {
                    "joint_position_post": robot.joint_pos,
                    "joint_velocity_post": robot.joint_vel,
                    "joint_position_target": robot.joint_pos_target,
                    "computed_torque": robot.computed_torque,
                    "applied_torque": robot.applied_torque,
                    "contact_normal_force_w": self.env.scene[
                        "contact_forces"
                    ].data.net_forces_w,
                    "foot_velocity_w": robot.body_lin_vel_w[:, self.foot_ids],
                }
            ),
        }
        self.samples.append(sample)
        self.pending = None

    def finish(self):
        if self.pending is not None or not self.samples:
            raise RuntimeError("Incomplete control diagnostic capture")
        return {
            key: np.stack([s[key] for s in self.samples]) for key in self.samples[0]
        }


def summarize_control_trace(control, trace, labels, metadata):
    """Validate pairing and summarize zero-command windows without bridging resets.

    Keep per-second bins to distinguish initial braking from later drift. Full
    arrays remain the source of truth; no causal diagnosis is assigned here.
    """
    steps, count, width = trace["command"].shape
    if width != 3 or count != len(labels) or steps == 0:
        raise ValueError("Invalid physical trace layout")
    shapes = {
        name: (steps, count, 12)
        for name in (
            "action",
            "reference_action",
            "joint_position_pre",
            "joint_velocity_pre",
            "joint_position_post",
            "joint_velocity_post",
            "joint_position_target",
            "computed_torque",
            "applied_torque",
        )
    }
    shapes.update(
        observation_pre=(steps, count, 48),
        contact_normal_force_w=(steps, count, len(metadata["contact_body_names"]), 3),
        foot_velocity_w=(steps, count, 4, 3),
    )
    if set(control) != set(shapes) or any(
        control[key].shape != shape or not np.isfinite(control[key]).all()
        for key, shape in shapes.items()
    ):
        raise ValueError("Missing, malformed or nonfinite control diagnostic arrays")
    if not np.array_equal(control["observation_pre"][:, :, 9:12], trace["command"]):
        raise ValueError("Diagnostic frame/physical command mismatch")
    done = trace["terminated"] | trace["time_out"]
    if done.shape != (steps, count) or done.dtype != np.bool_:
        raise ValueError("Invalid physical reset masks")
    continuing = ~done[:-1]
    if not np.array_equal(
        control["observation_pre"][1:, :, 36:48][continuing],
        control["action"][:-1][continuing],
    ):
        raise ValueError("Diagnostic previous-action timing mismatch")
    if not np.array_equal(
        control["joint_position_pre"][1:][continuing],
        control["joint_position_post"][:-1][continuing],
    ):
        raise ValueError("Diagnostic pre/post joint-state timing mismatch")
    foot_contacts = [
        metadata["contact_body_names"].index(name) for name in metadata["foot_names"]
    ]
    windows = []
    for env_id, label in enumerate(labels):
        terminal = np.flatnonzero(done[:, env_id])
        end = int(terminal[0]) + 1 if len(terminal) else steps
        zero = (trace["command"][:end, env_id] == 0).all(axis=1)
        boundaries = np.diff(np.r_[False, zero, False].astype(int))
        for start, stop in zip(
            np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
        ):
            bins = []
            for left in range(int(start), int(stop), round(1 / DT)):
                right = min(left + round(1 / DT), int(stop))
                part = slice(left, right)
                delta = (
                    control["action"][part, env_id]
                    - control["reference_action"][part, env_id]
                )
                tracking = (
                    control["joint_position_target"][part, env_id]
                    - control["joint_position_post"][part, env_id]
                )
                clipping = (
                    control["computed_torque"][part, env_id]
                    - control["applied_torque"][part, env_id]
                )
                bins.append(
                    {
                        "start_s": (left - int(start)) * DT,
                        "duration_s": (right - left) * DT,
                        "mean_body_yaw_rad_s": float(
                            trace["angular_velocity_b"][part, env_id, 2].mean()
                        ),
                        "mean_world_up_yaw_rad_s": float(
                            trace["angular_velocity_w"][part, env_id, 2].mean()
                        ),
                        "reference_action_rms_raw_per_joint": np.sqrt(
                            np.mean(delta**2, axis=0)
                        ).tolist(),
                        "joint_target_error_rms_rad_per_joint": np.sqrt(
                            np.mean(tracking**2, axis=0)
                        ).tolist(),
                        "torque_clipping_gap_max_nm_per_joint": np.abs(clipping)
                        .max(axis=0)
                        .tolist(),
                        "foot_mean_normal_force_z_n": control["contact_normal_force_w"][
                            part, env_id
                        ][:, foot_contacts, 2]
                        .mean(axis=0)
                        .tolist(),
                        "foot_max_speed_xy_m_s": np.linalg.norm(
                            control["foot_velocity_w"][part, env_id, :, :2], axis=-1
                        )
                        .max(axis=0)
                        .tolist(),
                    }
                )
            windows.append(
                {
                    "env_id": env_id,
                    "profile": label,
                    "start_s": int(start) * DT,
                    "duration_s": int(stop - start) * DT,
                    "ends_in_physical_reset": bool(done[int(stop) - 1, env_id]),
                    "bins": bins,
                }
            )
    return {
        "schema_version": VERSION,
        "status": "CAPTURE_VALID",
        "control_steps": steps,
        "metadata": metadata,
        "zero_command_windows": windows,
        "scope": "paired diagnostics, NOT a behavioral pass or proof of root cause; exclude all frames after the first physical reset per trial",
    }


def validate_control_artifacts(output, report, reference, learner_sha256):
    """Fail closed if a nominally successful worker omitted the requested capture."""
    diagnostic = report.get("control_diagnostics", {})
    summary = json.loads((output / "control_report.json").read_text())
    interface = json.loads((output / "control_interface.json").read_text())
    if not all(isinstance(value, dict) for value in (diagnostic, summary, interface)):
        raise ValueError("Diagnostic metadata must be JSON objects")
    expected_hashes = {
        "learner_checkpoint": learner_sha256,
        "physical_trace": file_sha256(output / "trace.npz"),
        "control_trace": file_sha256(output / "control_trace.npz"),
        "interface": file_sha256(output / "control_interface.json"),
    }
    if (
        diagnostic.get("status") != "CAPTURE_VALID"
        or diagnostic.get("sha256") != file_sha256(output / "control_report.json")
        or diagnostic.get("reference") != reference
        or summary.get("status") != "CAPTURE_VALID"
        or summary.get("schema_version") != VERSION
        or summary.get("reference") != reference
        or summary.get("sha256") != expected_hashes
        or interface.get("reference") != reference
        or interface.get("learner_sha256") != learner_sha256
    ):
        raise ValueError("Diagnostic artifact/reference identity mismatch")
