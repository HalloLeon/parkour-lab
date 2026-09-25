"""Frozen ROA evaluation and explicit simulator-input diagnostics, never a gate."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

import torch

from parkour_lab.learning.operator_roa import set_phase, state_sha256

VERSION = "operator_roa_history_evaluation_v1"
COMMAND_TAPE = (
    {"name": "initial_stop", "steps": 100, "command": (0.0, 0.0, 0.0)},
    {"name": "forward", "steps": 200, "command": (0.4, 0.0, 0.0)},
    {"name": "middle_stop", "steps": 150, "command": (0.0, 0.0, 0.0)},
    {"name": "pivot_positive", "steps": 100, "command": (0.0, 0.0, 0.5)},
    {"name": "pivot_negative", "steps": 100, "command": (0.0, 0.0, -0.5)},
    {"name": "forward_arc", "steps": 100, "command": (0.35, 0.0, 0.3)},
    {"name": "final_stop", "steps": 150, "command": (0.0, 0.0, 0.0)},
)
EVALUATION_STEPS = sum(phase["steps"] for phase in COMMAND_TAPE)


def command_tape_sha256(tape):
    return hashlib.sha256(
        json.dumps(tape, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


COMMAND_TAPE_SHA256 = command_tape_sha256(COMMAND_TAPE)


def _finite(value, shape, name):
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"Invalid finite evaluation {name}")


def _mask(value, count, device, name):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (count,)
        or value.dtype != torch.bool
        or value.device != device
    ):
        raise ValueError(f"Invalid evaluation {name} mask")


def _hash_tensors(values):
    digest = hashlib.sha256()
    for name in sorted(values.keys()):
        value = values[name].detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _terrain_receipt(env):
    scene = getattr(env, "scene", None)
    terrain = getattr(scene, "terrain", None)
    if terrain is None and isinstance(scene, dict):
        terrain = scene.get("terrain")
    result = {"column_ids": None, "level_ids": None, "profile_by_column": None}
    if terrain is None or getattr(terrain, "terrain_types", None) is None:
        return {"status": "UNAVAILABLE_NOT_INFERRED", **result}
    for field, name in (
        ("column_ids", "terrain_types"),
        ("level_ids", "terrain_levels"),
    ):
        value = getattr(terrain, name, None)
        if value is None and field == "level_ids":
            continue
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (env.num_envs,)
            or value.dtype not in (torch.int32, torch.int64)
            or (value < 0).any()
        ):
            raise ValueError(f"Invalid evaluation terrain assignment: {name}")
        result[field] = value.detach().cpu().tolist()
    generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    columns = getattr(generator, "num_cols", None)
    if type(columns) is int and max(result["column_ids"]) >= columns:
        raise ValueError("Evaluation terrain column exceeds the native generator")
    sub_terrains = getattr(generator, "sub_terrains", None)
    if (
        getattr(generator, "curriculum", None) is True
        and type(columns) is int
        and columns > 0
        and isinstance(sub_terrains, dict)
        and len(sub_terrains) == columns
        and all(
            getattr(value, "proportion", None) == 1 / columns
            and isinstance(getattr(value, "profile", None), str)
            and value.profile
            for value in sub_terrains.values()
        )
    ):
        result["profile_by_column"] = [value.profile for value in sub_terrains.values()]
    return {"status": "RECORDED_NATIVE_ASSIGNMENTS", **result}


def _native_state_hash(env, name, width):
    scene = getattr(env, "scene", None)
    robot = getattr(scene, "robot", None)
    if robot is None and scene is not None:
        try:
            robot = scene["robot"]
        except (KeyError, TypeError):
            pass
    state = getattr(getattr(robot, "data", None), name, None)
    if state is None:
        return None  # Explicitly unavailable in a generic CPU host, never inferred.
    _finite(state, (env.num_envs, width), "initial " + name)
    return _hash_tensors({name: state})


def _profile_summary(rows, terrain, period, command_tape):
    if terrain["profile_by_column"] is None:
        return None
    result = {}
    for name in dict.fromkeys(terrain["profile_by_column"]):
        selected = [row for row in rows if row["terrain_profile"] == name]
        decisions = sum(row["decision_count"] for row in selected)
        exposure, start = {}, 0
        for phase in command_tape:
            exposure[phase["name"]] = sum(
                min(max(row["decision_count"] - start, 0), phase["steps"])
                for row in selected
            )
            start += phase["steps"]
        result[name] = {
            "row_count": len(selected),
            "terminated": sum(row["end_kind"] == "terminated" for row in selected),
            "timeouts": sum(row["end_kind"] == "timeout" for row in selected),
            "right_censored": sum(
                row["end_kind"] == "right_censored" for row in selected
            ),
            "exposure_decisions": decisions,
            "exposure_seconds": decisions * period,
            "exposure_decisions_by_phase": exposure,
            "restricted_observed_duration_mean_s": (
                decisions * period / len(selected) if selected else None
            ),
        }
    return result


def _summary(totals):
    count, xy_norm, xy_squared, yaw_abs, yaw_squared, velocity_squared = totals.tolist()
    return {
        "sample_count": int(count),
        "xy_tracking_error_mean_m_s": xy_norm / count if count else None,
        "xy_tracking_rmse_m_s": math.sqrt(xy_squared / count) if count else None,
        "yaw_tracking_error_mean_rad_s": yaw_abs / count if count else None,
        "yaw_tracking_rmse_rad_s": math.sqrt(yaw_squared / count) if count else None,
        "estimated_velocity_component_rmse_m_s": (
            math.sqrt(velocity_squared / (3 * count)) if count else None
        ),
    }


def _input_diagnostic_report(
    records, mode, terrain, output, command_tape, spatial=None
):
    """Compare predictions and actions on this rollout, never a shadow counterfactual."""
    import numpy as np

    arrays = {
        name: np.stack([record[name] for record in records]) for name in records[0]
    }
    steps, count = arrays["first_attempt_valid"].shape
    phase_index = np.repeat(
        np.arange(len(command_tape)), [phase["steps"] for phase in command_tape]
    )[:steps]
    arrays["phase_index"] = phase_index
    near_entry = None
    if spatial is not None:
        root, entry = spatial["root_local_m"], spatial["entry_x_m"]
        width = spatial["corridor_half_width_m"]
        if not (
            root.shape == (steps, count, 3)
            and entry.shape == (count,)
            and np.isfinite(root).all()
            and np.isfinite(entry).all()
            and np.ndim(width) == 0
            and np.isfinite(width)
            and width > 0
            and spatial["first_attempt_valid"].dtype == np.bool_
            and np.array_equal(
                spatial["first_attempt_valid"], arrays["first_attempt_valid"]
            )
        ):
            raise ValueError("Input telemetry and spatial pre-action samples differ")
        arrays.update(
            root_local_m=root, entry_x_m=entry, corridor_half_width_m=np.asarray(width)
        )
        near_entry = (
            (root[..., 0] >= entry - 0.5)
            & (root[..., 0] < entry)
            & (np.abs(root[..., 1]) <= width)
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite prior evidence, even when called outside the screen CLI.
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    profiles = (
        [terrain["profile_by_column"][column] for column in terrain["column_ids"]]
        if terrain["profile_by_column"] is not None
        else ["unmapped"] * arrays["first_attempt_valid"].shape[1]
    )
    moving = np.any(arrays["command_b"][..., :2] != 0, axis=-1)
    forward = (
        moving
        & np.array([p["name"] == "forward" for p in command_tape])[phase_index, None]
    )
    low_speed = np.linalg.norm(arrays["true_velocity_b_m_s"][..., :2], axis=-1) < 0.05

    def summarize(mask):
        predicted = arrays["estimated_velocity_b_m_s"][mask].astype(np.float64)
        actual = arrays["true_velocity_b_m_s"][mask].astype(np.float64)
        command = arrays["command_b"][mask].astype(np.float64)
        delta = arrays["applied_raw_action"][mask].astype(np.float64) - arrays[
            "history_raw_action"
        ][mask].astype(np.float64)
        error = predicted - actual
        means = {
            "estimated_velocity_mean_b_m_s": predicted,
            "true_velocity_mean_b_m_s": actual,
            "command_mean_b": command,
            "estimated_minus_true_velocity_bias_b_m_s": error,
            "command_minus_estimated_xy_mean_m_s": command[:, :2] - predicted[:, :2],
            "command_minus_true_xy_mean_m_s": command[:, :2] - actual[:, :2],
        }
        return {
            "sample_count": len(predicted),
            **{
                name: value.mean(0).tolist() if len(predicted) else None
                for name, value in means.items()
            },
            "estimated_minus_true_velocity_rmse_b_m_s": (
                np.sqrt(np.square(error).mean(0)).tolist() if len(predicted) else None
            ),
            "velocity_component_rmse_m_s": (
                float(np.sqrt(np.square(error).mean())) if len(predicted) else None
            ),
            "raw_action_delta_abs_mean": (
                float(np.abs(delta).mean()) if len(predicted) else None
            ),
            "raw_action_delta_abs_max": (
                float(np.abs(delta).max()) if len(predicted) else None
            ),
        }

    groups = {}
    for profile in dict.fromkeys(profiles):
        selected = np.array([name == profile for name in profiles])[None, :]
        first = arrays["first_attempt_valid"] & selected
        groups[profile] = {
            "first_episode": summarize(first),
            "moving_first_episode": summarize(first & moving),
            "by_phase": {
                phase["name"]: summarize(first & (phase_index[:, None] == i))
                for i, phase in enumerate(command_tape)
            },
            "by_condition": {
                "forward_low_speed": summarize(first & forward & low_speed),
                "forward_moving": summarize(first & forward & ~low_speed),
                "forward_near_entry": (
                    summarize(first & forward & near_entry)
                    if near_entry is not None
                    else None
                ),
                "forward_near_entry_low_speed": (
                    summarize(first & forward & near_entry & low_speed)
                    if near_entry is not None
                    else None
                ),
            },
        }
    return {
        "version": "operator_roa_input_diagnostic_v2",
        "mode": mode or "causal_history",
        "deployable": False,
        "causal_actor_inputs_only": mode is None,
        "scope": "Frozen input telemetry, not an oracle upper bound, deployment validation or acceptance evidence",
        "sampling": "Pre-action decision index times 0.02s; first_attempt_valid includes the decision causing first termination, excludes all later episodes",
        "comparison": (
            "History actions are shadows on the INTERVENED trajectory, not an independent baseline rollout"
            if mode
            else "Executed causal history actions; no input replacement or privileged actor inputs"
        ),
        "replaced": (
            "Only motor code[:3]: native current body-COM velocity in m/s"
            if mode == "true_velocity"
            else (
                "Only motor code[3:]: privileged dynamics encoder latent"
                if mode
                else None
            )
        ),
        "conditions": {
            "forward": "Tape phase named forward with nonzero XY command; all conditions use first-episode decisions",
            "low_speed": "Native body-COM XY speed < 0.05 m/s at this instant, not a sustained stall diagnosis",
            "near_entry": "entry_x - 0.5 <= root_local_x < entry_x; abs(root_local_y) <= corridor_half_width; no foot contact or support inference",
            "spatial_available": spatial is not None,
            "interpretation": "Conditional populations differ after intervention; compare matched full-rollout outcomes, not conditional means as causal effects",
            "command_units": "Body [vx m/s, vy m/s, yaw rad/s]; velocity bias is estimated minus true [x,y,z] m/s",
        },
        "by_profile": groups,
        "trace": {
            "path": str(output.resolve()),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        },
    }


def evaluate_history(
    host,
    policy,
    *,
    seed: int,
    steps: int = EVALUATION_STEPS,
    controller=None,
    command_tape=COMMAND_TAPE,
    observer=None,
    diagnostic_input=None,
    diagnostic_output=None,
):
    """Evaluate a fixed tape without updates; shorter prefixes are diagnostic only.

    Samples are pre-action, not auto-reset observations attributed to a terminal
    state. First-episode totals include the decision causing the first end and no
    later decisions for that row. Timeouts are host-filtered to exclude physical
    terminations. There is no survivor-only mean or zero-sample success claim.
    If supplied, the exported controller drives the existing host motor bridge;
    the full policy is only a frozen parity/diagnostic reference. The caller must
    first validate the exported portable motor contract against the native host.
    A diagnostic_output alone records observer-only telemetry on the causal
    rollout. An explicit diagnostic replaces one motor-code slice after shadow export
    parity checking. It never supplies privilege to the exported controller or
    changes history frames. Applied diagnostic actions feed the next frame.
    """
    if (
        diagnostic_input not in (None, "true_velocity", "privileged_latent")
        or (diagnostic_input is not None and diagnostic_output is None)
        or (diagnostic_output is not None and controller is None)
    ):
        raise ValueError(
            "Input diagnostic requires a known mode, trace path and shadow controller"
        )
    if diagnostic_output is not None and Path(diagnostic_output).exists():
        raise FileExistsError(f"Diagnostic trace already exists: {diagnostic_output}")
    if type(seed) is not int or seed < 0:
        raise ValueError("Evaluation seed must be a nonnegative integer")
    if type(steps) is not int or not 1 <= steps <= EVALUATION_STEPS:
        raise ValueError("Evaluation steps must be a positive fixed-tape prefix")
    command_tape = deepcopy(command_tape)
    if (
        not command_tape
        or sum(p["steps"] for p in command_tape) != EVALUATION_STEPS
        or len({p["name"] for p in command_tape}) != len(command_tape)
        or any(
            type(p["steps"]) is not int
            or p["steps"] < 1
            or not isinstance(p["name"], str)
            or not p["name"]
            or len(p["command"]) != 3
            or any(not math.isfinite(v) or abs(v) > 1 for v in p["command"])
            for p in command_tape
        )
    ):
        raise ValueError("Require a finite named 900-step development command tape")
    period = host.env.step_dt
    count = host.env.num_envs
    if type(count) is not int or count < 1 or period != 0.02:
        raise ValueError("Evaluation requires nonempty native 50Hz environments")
    policy.eval()
    set_phase(policy, "frozen")
    before = state_sha256(policy)
    controller_session = None
    controller_hash = None
    diagnostic_records = []
    if controller is not None:
        from parkour_lab.learning.controller import ControllerSession, Sample
        from parkour_lab.learning.operator_roa_runtime import roa_tensor_sha256

        sensor_fields = {
            "base_ang_vel": (slice(0, 3), "rad/s", "body"),
            "projected_gravity": (slice(3, 6), "unitless", "body"),
            "joint_position_relative_default": (slice(9, 21), "rad", "joint"),
            "joint_velocity": (slice(21, 33), "rad/s", "joint"),
            "stock_previous_raw_action": (slice(33, 45), "unitless", "joint"),
        }
        if (
            set(controller.spec.sensors) != set(sensor_fields)
            or controller.spec.period_s != period
        ):
            raise ValueError("Evaluation controller must declare the causal ROA frame")
        for name, (indices, units, frame_name) in sensor_fields.items():
            spec = controller.spec.sensors[name]
            if (spec.shape, spec.units, spec.frame, spec.required, spec.privileged) != (
                (indices.stop - indices.start,),
                units,
                frame_name,
                True,
                False,
            ):
                raise ValueError("Evaluation controller sensor semantics differ")
        controller_session = ControllerSession(
            controller,
            joint_names=tuple(host.bridge.joint_names),
            # The caller validates the portable motor contract against the native
            # bridge. Full archived profiles may differ only by source batch size.
            actuator_profile=controller.spec.actuator_profile,
            allow_privileged=False,
        )
        controller_hash = roa_tensor_sha256(controller.motor, controller.estimator)
        if any(
            module.training or any(p.requires_grad for p in module.parameters())
            for module in (controller.motor, controller.estimator)
        ):
            raise ValueError("Evaluation requires a frozen exported controller")
    tape = [
        (phase_index, phase["command"])
        for phase_index, phase in enumerate(command_tape)
        for _ in range(phase["steps"])
    ]
    with torch.no_grad():
        observations, reset = host.reset(seed=seed, command=tape[0][1])
        device = observations["policy"].device
        _mask(reset, count, device, "initial reset")
        if not reset.all():
            raise ValueError("Evaluation must reset every environment initially")
        for name, width in (
            ("policy", 45),
            ("history", 1125),
            ("critic_state", 48),
            ("terrain", 264),
            ("dynamics", 7),
        ):
            _finite(observations[name], (count, width), name)
        initial_hash = _hash_tensors(observations)
        initial_group_hashes = {
            name: _hash_tensors({name: observations[name]})
            for name in sorted(observations.keys())
        }
        dynamics_hash = _hash_tensors({"dynamics": observations["dynamics"]})
        terrain = _terrain_receipt(host.env)
        root_state_hash = _native_state_hash(host.env, "root_state_w", 13)
        joint_pos_hash = _native_state_hash(host.env, "joint_pos", 12)
        joint_vel_hash = _native_state_hash(host.env, "joint_vel", 12)
        alive = torch.ones(count, dtype=torch.bool, device=device)
        if observer is not None:
            observer.observe(0, alive)
        durations = torch.zeros(count, dtype=torch.int64, device=device)
        first_end = torch.full((count,), -1, dtype=torch.int64, device=device)
        first_terminated = torch.zeros_like(alive)
        first_timeout = torch.zeros_like(alive)
        # Populations: all decisions / first episode. Groups: regimes then phases.
        totals = torch.zeros(
            (2, 3 + len(command_tape), 6), dtype=torch.float64, device=device
        )
        end_counts = torch.zeros(3, dtype=torch.int64, device=device)
        for index, (phase, command) in enumerate(tape[:steps]):
            frame, flat_history, clean = (
                observations["policy"],
                observations["history"],
                observations["critic_state"],
            )
            for name, value, width in (
                ("policy", frame, 45),
                ("history", flat_history, 1125),
                ("critic_state", clean, 48),
            ):
                _finite(value, (count, width), name)
            history = flat_history.reshape(count, 25, 45)
            expected_command = frame.new_tensor(command).expand(count, 3)
            if not (
                torch.equal(frame, history[:, -1])
                and torch.equal(frame[:, 6:9], expected_command)
                and torch.equal(clean[:, 9:12], expected_command)
            ):
                raise ValueError(
                    "Evaluation pre-action history or command alignment failed"
                )
            estimate = policy.actor.estimate(history)
            _finite(estimate, (count, 11), "velocity/latent estimate")
            raw = policy.actor.history_action(frame, history)
            _finite(raw, (count, 12), "action")
            if controller_session is not None:
                time_s = index * period
                sensors = {
                    name: Sample(
                        frame[:, indices],
                        time_s,
                        torch.ones_like(reset),
                        units,
                        frame_name,
                    )
                    for name, (indices, units, frame_name) in sensor_fields.items()
                }
                output = controller_session.step(
                    time_s=time_s,
                    command=frame[:, 6:9],
                    command_time_s=time_s,
                    sensors=sensors,
                    reset_mask=reset,
                )
                _finite(output.raw_action, (count, 12), "controller raw action")
                controller_estimate = controller.last_estimate
                controller_history = controller.history_frames
                _finite(controller_estimate, (count, 11), "controller estimate")
                _finite(controller_history, (count, 25, 45), "controller history")
                if not (
                    torch.equal(output.raw_action, raw)
                    and torch.equal(
                        output.position_rad, host.bridge.default + 0.25 * raw
                    )
                    and torch.equal(controller_estimate, estimate)
                    and torch.equal(controller_history, history)
                ):
                    raise RuntimeError(
                        "Exported causal controller differs from frozen reference"
                    )
                raw = output.raw_action.detach().clone()
            code, applied = estimate, raw
            if diagnostic_input is not None:
                code = estimate.clone()
                if diagnostic_input == "true_velocity":
                    code[:, :3] = clean[:, :3]
                else:
                    code[:, 3:] = policy.actor.encode(observations["dynamics"])
                applied = policy.actor.motor(frame, code)
                _finite(applied, (count, 12), "diagnostic action")
            if diagnostic_output is not None:
                diagnostic_records.append(
                    {
                        name: value.detach().cpu().numpy().copy()
                        for name, value in {
                            "estimated_velocity_b_m_s": estimate[:, :3],
                            "true_velocity_b_m_s": clean[:, :3],
                            "history_latent": estimate[:, 3:],
                            "applied_latent": code[:, 3:],
                            "history_raw_action": raw,
                            "applied_raw_action": applied,
                            "first_attempt_valid": alive,
                            "command_b": expected_command,
                        }.items()
                    }
                )
            raw = applied
            # Own the metrics before native buffers can change in step().
            xy = clean[:, :2].double() - expected_command[:, :2].double()
            yaw = clean[:, 5].double() - expected_command[:, 2].double()
            velocity_error = estimate[:, :3].double() - clean[:, :3].double()
            values = torch.stack(
                (
                    torch.ones(count, dtype=torch.float64, device=device),
                    xy.norm(dim=1),
                    xy.square().sum(1),
                    yaw.abs(),
                    yaw.square(),
                    velocity_error.square().sum(1),
                ),
                dim=1,
            )
            regime = (
                1
                if command[0] != 0 or command[1] != 0
                else (2 if command[2] != 0 else 0)
            )
            all_values, first_values = values.sum(0), values[alive].sum(0)
            next_command = tape[index + 1][1] if index + 1 < len(tape) else command
            next_observations, reward, done, extras = host.step(
                raw,
                (
                    f"diagnostic_{diagnostic_input}"
                    if diagnostic_input
                    else "frozen_history_evaluation"
                ),
                next_command=next_command,
            )
            _finite(reward, (count,), "reward")
            _mask(done, count, device, "done")
            timeout = extras["time_outs"]
            _mask(timeout, count, device, "timeout")
            if (timeout & ~done).any():
                raise ValueError("Evaluation timeout must be an ended transition")
            if _terrain_receipt(host.env) != terrain:
                raise RuntimeError("Evaluation terrain assignment changed")
            terminated = done & ~timeout
            for group in (regime, 3 + phase):
                totals[0, group] += all_values
                totals[1, group] += first_values
            durations += alive
            newly_ended = alive & done
            first_end[newly_ended] = index
            first_terminated |= alive & terminated
            first_timeout |= alive & timeout
            alive &= ~done
            if observer is not None:
                observer.observe(index + 1, alive)
            end_counts += torch.stack((done.sum(), terminated.sum(), timeout.sum()))
            observations = next_observations
            reset = done.detach().clone()
    after = state_sha256(policy)
    if (
        after != before
        or any(p.requires_grad for p in policy.parameters())
        or policy.training
    ):
        raise RuntimeError("Frozen evaluation changed the policy or its frozen mode")
    if controller is not None:
        controller_after = roa_tensor_sha256(controller.motor, controller.estimator)
        if controller_after != controller_hash or any(
            module.training or any(p.requires_grad for p in module.parameters())
            for module in (controller.motor, controller.estimator)
        ):
            raise RuntimeError("Frozen evaluation changed the exported controller")
    regimes = ("stopped", "moving", "pivot")
    tracking = {}
    for population, values in zip(
        ("all_decisions", "first_episode"), totals, strict=True
    ):
        tracking[population] = {
            "by_regime": {name: _summary(values[i]) for i, name in enumerate(regimes)},
            "by_phase": {
                phase["name"]: _summary(values[3 + i])
                for i, phase in enumerate(command_tape)
            },
        }
    rows = [
        {
            "environment_index": row,
            "decision_count": decisions,
            "observed_duration_s": decisions * period,
            "first_end_step": end if end >= 0 else None,
            "end_kind": (
                "terminated" if term else "timeout" if timeout else "right_censored"
            ),
            "terrain_column_id": (
                terrain["column_ids"][row]
                if terrain["column_ids"] is not None
                else None
            ),
            "terrain_level_id": (
                terrain["level_ids"][row] if terrain["level_ids"] is not None else None
            ),
            "terrain_profile": (
                terrain["profile_by_column"][terrain["column_ids"][row]]
                if terrain["profile_by_column"] is not None
                else None
            ),
        }
        for row, (decisions, end, term, timeout) in enumerate(
            zip(
                durations.tolist(),
                first_end.tolist(),
                first_terminated.tolist(),
                first_timeout.tolist(),
                strict=True,
            )
        )
    ]
    report = {
        "version": VERSION,
        "scope": "Seeded reset on training geometry/dynamics; not held-out terrain or a behavior gate",
        "seed": seed,
        "control_steps": steps,
        "num_envs": count,
        "environment_transitions": steps * count,
        "period_s": period,
        "command_tape": command_tape,
        "command_tape_sha256": command_tape_sha256(command_tape),
        "full_tape_steps": EVALUATION_STEPS,
        "complete_tape": steps == EVALUATION_STEPS,
        "initial_observation_sha256": initial_hash,
        "initial_observation_group_sha256": initial_group_hashes,
        "initial_dynamics_sha256": dynamics_hash,
        "initial_joint_pos_sha256": joint_pos_hash,
        "initial_joint_vel_sha256": joint_vel_hash,
        "initial_root_state_sha256": root_state_hash,
        "initial_root_state_status": (
            "RECORDED_ROOT_STATE_W" if root_state_hash else "UNAVAILABLE_NOT_INFERRED"
        ),
        "terrain_assignment": terrain,
        "policy_state_sha256_before": before,
        "policy_state_sha256_after": after,
        "controller_parity": (
            {"status": "REFERENCE_HISTORY_ACTOR_ONLY"}
            if controller_session is None
            else {
                "status": (
                    "EXACT_SHADOW_NOT_APPLIED"
                    if diagnostic_input
                    else "EXACT_ON_EXECUTED_DECISIONS"
                ),
                "decision_count": steps,
                "interface_sha256": controller_session.interface_sha256,
                "state_sha256_before": controller_hash,
                "state_sha256_after": controller_after,
                "compared": [
                    "raw_action",
                    "absolute_joint_targets",
                    "estimated_velocity_and_latent",
                    "causal_history",
                ],
                "inputs": "Only noisy frame sensors and current command; no true velocity, dynamics or terrain",
            }
        ),
        "all_transition_counts": dict(
            zip(("resets", "terminated", "timeouts"), end_counts.tolist(), strict=True)
        ),
        "first_episode": {
            "rows": rows,
            "terminated": int(first_terminated.sum()),
            "timeouts": int(first_timeout.sum()),
            "right_censored": int(alive.sum()),
            "restriction_horizon_s": steps * period,
            "restricted_observed_duration_mean_s": float(durations.double().mean())
            * period,
            "includes_first_end_transition": True,
            "by_profile": _profile_summary(rows, terrain, period, command_tape),
            "unmapped_profile_row_count": (
                count if terrain["profile_by_column"] is None else 0
            ),
        },
        "sampling": "Pre-action true COM xy velocity, yaw rate and current command; returned auto-reset states are not terminal tracking samples",
        "regimes": "Exact commands: stopped xyz=0; moving xy!=0; pivot xy=0,yaw!=0. Empty groups have null metrics, not success.",
        "tracking": tracking,
    }
    if diagnostic_input is not None:
        report.update(
            diagnostic_input=diagnostic_input,
            action_source="PRIVILEGED_SIMULATION_DIAGNOSTIC_NOT_DEPLOYABLE",
        )
    if diagnostic_output is not None:
        spatial_samples = getattr(observer, "input_diagnostic_samples", None)
        report["input_diagnostic"] = _input_diagnostic_report(
            diagnostic_records,
            diagnostic_input,
            terrain,
            diagnostic_output,
            command_tape,
            spatial_samples(steps) if spatial_samples is not None else None,
        )
    json.dumps(report, allow_nan=False)  # Reject overflow as well as nonfinite inputs.
    return report
