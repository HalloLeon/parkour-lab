"""Read-only action/reference and terminal-safe actuator/contact diagnostics.

This captures the unchanged stock benchmark, not a new controller or training
recipe. Reference outputs are counterfactual actions on learner-visited states;
they are never executed and do not demonstrate counterfactual recovery.
"""

from __future__ import annotations

import json
import copy
import math
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


def controller_record(
    session,
    *,
    requested_command,
    requested_at_s,
    delivered_position_rad,
    delivery_time_s,
    command_source,
    safety_events,
):
    """JSON-ready snapshot for the new controller boundary, not legacy NPZ v1.

    Enable session capture explicitly (CPU copies are not a real-time logger).
    Persist the session manifest once alongside these records. Delivered targets
    are host-reported command delivery, NOT measured motion/torque. None means
    not delivered/unknown, never inferred from a successful policy computation.
    Replay requested BODY COMMANDS with live sensing for future closed-loop
    comparison; this function neither replays motor actions nor controls hardware.
    """
    from parkour_lab.learning.controller import finite_tensor

    if session.record is None:
        raise ValueError(
            "No captured validated controller input; capture must be enabled"
        )
    record = copy.deepcopy(session.record)
    batch = len(record["applied_command"])
    finite_tensor(requested_command, (batch, 3))
    if (
        not math.isfinite(requested_at_s)
        or not 0 <= requested_at_s <= record["applied_at_s"]
        or not isinstance(command_source, str)
        or not command_source
        or not isinstance(safety_events, (tuple, list))
        or any(not isinstance(event, str) or not event for event in safety_events)
    ):
        raise ValueError("Invalid command source, timestamp or safety events")
    requested = requested_command.detach().cpu().tolist()
    if requested != record["applied_command"] and not safety_events:
        raise ValueError("Changed applied command requires an explicit event/reason")
    if (delivered_position_rad is None) != (delivery_time_s is None):
        raise ValueError("Delivery target and time must both be known or both absent")
    delivered = None
    if delivered_position_rad is not None:
        if record["status"] == "FAULT":
            raise ValueError("Faulted inference cannot claim target delivery")
        finite_tensor(delivered_position_rad, (batch, len(record["joint_names"])))
        if not math.isfinite(delivery_time_s) or delivery_time_s < record["time_s"]:
            raise ValueError("Invalid delivery time")
        delivered = delivered_position_rad.detach().cpu().tolist()
        if delivered != record["requested_position_rad"] and not safety_events:
            raise ValueError(
                "Changed delivered target requires an explicit event/reason"
            )
    record.update(
        version="operator_controller_record_v1",
        controller_artifact_sha256=session.manifest["artifact_sha256"],
        requested_command=requested,
        requested_at_s=requested_at_s,
        command_source=command_source,
        safety_events=list(safety_events),
        delivered_position_rad=delivered,
        delivery_time_s=delivery_time_s,
        delivery_evidence=(
            "NOT_REPORTED"
            if delivered is None
            else "HOST_REPORTED_TARGET_NOT_MEASURED_MOTION"
        ),
    )
    return record


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


def native_reward_snapshot(env, names, weights):
    """Read once-computed Isaac Lab 2.3.2 rewards before reset; never recompute terms."""
    names, weights = tuple(names), tuple(weights)
    if (
        env.cfg.decimation != 4
        or not math.isclose(env.step_dt, DT, rel_tol=0, abs_tol=1e-9)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or len(weights) != len(names)
        or not np.isfinite(weights).all()
    ):
        raise ValueError("Invalid native reward timing or term specification")
    manager = env.reward_manager
    if (
        tuple(manager.active_terms) != names
        or tuple(float(manager.get_term_cfg(name).weight) for name in names) != weights
    ):
        raise ValueError("Native reward order or weights changed during capture")
    # The pinned manager stores weighted rates, not per-step contributions.
    contribution = manager._step_reward * env.step_dt
    total = env.reward_buf
    if (
        contribution.shape != (env.num_envs, len(names))
        or total.shape != (env.num_envs,)
        or not torch.isfinite(contribution).all()
        or not torch.isfinite(total).all()
    ):
        raise ValueError("Invalid native reward matrix or total")
    if not torch.allclose(contribution.sum(-1), total, atol=2e-6, rtol=1e-5):
        raise ValueError("Native reward decomposition does not sum to reward_buf")
    return {"reward_contribution": contribution, "reward_total": total}


class OperatorControlTrace:
    """Pair the actual pre-action frame with the post-physics/pre-reset state."""

    def __init__(self, env, reference=None, *, native_rewards=False, env_ids=None):
        if reference is not None and (
            reference.training or any(p.requires_grad for p in reference.parameters())
        ):
            raise ValueError("Diagnostic reference must be frozen and in eval mode")
        self.env, self.reference = env, reference
        self.pending = None
        self.samples = []
        self.native_rewards = native_rewards
        self.env_ids = slice(None) if env_ids is None else env_ids
        self.substeps = []
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
        if reference is None:
            self.metadata.pop("reference_action")
        if native_rewards:
            if env.cfg.decimation != 4 or abs(env.step_dt - DT) > 1e-9:
                raise ValueError("Native capture requires the stock four-substep motor")
            manager = env.reward_manager
            names = list(manager.active_terms)
            weights = [float(manager.get_term_cfg(n).weight) for n in names]
            if (
                not names
                or len(set(names)) != len(names)
                or not np.isfinite(weights).all()
            ):
                raise ValueError("Invalid native reward terms")
            self.metadata.update(
                schema_version="operator_native_control_trace_v1",
                env_ids=(
                    list(range(env.num_envs)) if env_ids is None else env_ids.tolist()
                ),
                body_names=bodies,
                reward_names=names,
                reward_weights=weights,
                reward_contribution="Native weighted rate times step_dt, computed once by RewardManager; sum checked against reward_buf",
                omitted_training_objective="persistent_tilt impulse (gate disabled in evaluation), PPO entropy/value/retention losses; these are evaluation returns, not training returns",
                sampling="50 Hz post-step/pre-reset joint/contact/body state; four chronological 200 Hz computed/applied actuator torque commands (N m), not measured hardware torques",
                height_scan_post="Unclipped world hit z (m); nonfinite rays retained, not flat-support evidence",
                body_position_post="Body origins in world meters; NOT contact-point locations",
            )

    def snapshot(self, values):
        return _snapshot({name: value[self.env_ids] for name, value in values.items()})

    def before_step(self, observation, action):
        if self.pending is not None:
            raise RuntimeError("Previous diagnostic action lacks a pre-reset capture")
        if (
            observation.ndim != 2
            or observation.shape[1] != 48
            or action.shape != (len(observation), 12)
            or not all(torch.isfinite(x).all() for x in (observation, action))
        ):
            raise ValueError("Invalid diagnostic observation or action")
        robot = self.env.scene["robot"].data
        self.pending = self.snapshot(
            {
                "observation_pre": observation,
                "action": action,
                "joint_position_pre": robot.joint_pos,
                "joint_velocity_pre": robot.joint_vel,
            }
        )
        if self.reference is not None:
            with torch.inference_mode():
                reference_action = self.reference(observation)
            if (
                reference_action.shape != action.shape
                or not torch.isfinite(reference_action).all()
            ):
                self.pending = None
                raise ValueError("Invalid diagnostic reference action")
            self.pending.update(self.snapshot({"reference_action": reference_action}))
        self.substeps = []

    def after_substep(self):
        if self.native_rewards:
            if self.pending is None or len(self.substeps) >= 4:
                raise RuntimeError("Unpaired or extra actuator substep")
            # Native hook precedes scene.update: only explicit actuator commands
            # are current here, NOT cached post-physics body/joint/contact state.
            robot = self.env.scene["robot"].data
            self.substeps.append(
                self.snapshot(
                    {
                        "computed_torque_substeps": robot.computed_torque,
                        "applied_torque_substeps": robot.applied_torque,
                    }
                )
            )

    def after_step(self):
        if self.pending is None:
            raise RuntimeError("Post-physics diagnostics lack the delivered action")
        robot = self.env.scene["robot"].data
        sample = {
            **self.pending,
            **self.snapshot(
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
        if self.native_rewards:
            if len(self.substeps) != 4:
                raise RuntimeError("Missing actuator substeps")
            sample.update(
                self.snapshot(
                    {
                        **native_reward_snapshot(
                            self.env,
                            self.metadata["reward_names"],
                            self.metadata["reward_weights"],
                        ),
                        "body_position_post": robot.body_pos_w,
                        "height_scan_post": self.env.scene[
                            "height_scanner"
                        ].data.ray_hits_w[..., 2],
                    }
                )
            )
            for name in self.substeps[0]:
                sample[name] = np.stack([s[name] for s in self.substeps], axis=1)
            if any(
                not np.isfinite(v).all()
                for k, v in sample.items()
                if k != "height_scan_post"
            ):
                raise ValueError("Nonfinite native control data")
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
