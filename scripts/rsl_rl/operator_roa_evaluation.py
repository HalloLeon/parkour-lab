"""Frozen causal ROA diagnostics on the training geometry, never a behavior gate."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

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
COMMAND_TAPE_SHA256 = hashlib.sha256(
    json.dumps(COMMAND_TAPE, sort_keys=True, allow_nan=False).encode()
).hexdigest()


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


def _root_state_hash(env):
    scene = getattr(env, "scene", None)
    robot = getattr(scene, "robot", None)
    if robot is None and scene is not None:
        try:
            robot = scene["robot"]
        except (KeyError, TypeError):
            pass
    state = getattr(getattr(robot, "data", None), "root_state_w", None)
    if state is None:
        return None  # Explicitly unavailable in a generic CPU host, never inferred.
    _finite(state, (env.num_envs, 13), "initial root_state_w")
    return _hash_tensors({"root_state_w": state})


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


def evaluate_history(host, policy, *, seed: int, steps: int = EVALUATION_STEPS):
    """Evaluate a fixed tape without updates; shorter prefixes are diagnostic only.

    Samples are pre-action, not auto-reset observations attributed to a terminal
    state. First-episode totals include the decision causing the first end and no
    later decisions for that row. Timeouts are host-filtered to exclude physical
    terminations. There is no survivor-only mean or zero-sample success claim.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("Evaluation seed must be a nonnegative integer")
    if type(steps) is not int or not 1 <= steps <= EVALUATION_STEPS:
        raise ValueError("Evaluation steps must be a positive fixed-tape prefix")
    period = host.env.step_dt
    count = host.env.num_envs
    if type(count) is not int or count < 1 or period != 0.02:
        raise ValueError("Evaluation requires nonempty native 50Hz environments")
    policy.eval()
    set_phase(policy, "frozen")
    before = state_sha256(policy)
    tape = [
        (phase_index, phase["command"])
        for phase_index, phase in enumerate(COMMAND_TAPE)
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
        dynamics_hash = _hash_tensors({"dynamics": observations["dynamics"]})
        terrain = _terrain_receipt(host.env)
        root_state_hash = _root_state_hash(host.env)
        alive = torch.ones(count, dtype=torch.bool, device=device)
        durations = torch.zeros(count, dtype=torch.int64, device=device)
        first_end = torch.full((count,), -1, dtype=torch.int64, device=device)
        first_terminated = torch.zeros_like(alive)
        first_timeout = torch.zeros_like(alive)
        # Populations: all decisions / first episode. Groups: regimes then phases.
        totals = torch.zeros(
            (2, 3 + len(COMMAND_TAPE), 6), dtype=torch.float64, device=device
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
                raw, "frozen_history_evaluation", next_command=next_command
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
            end_counts += torch.stack((done.sum(), terminated.sum(), timeout.sum()))
            observations = next_observations
    after = state_sha256(policy)
    if (
        after != before
        or any(p.requires_grad for p in policy.parameters())
        or policy.training
    ):
        raise RuntimeError("Frozen evaluation changed the policy or its frozen mode")
    regimes = ("stopped", "moving", "pivot")
    tracking = {}
    for population, values in zip(
        ("all_decisions", "first_episode"), totals, strict=True
    ):
        tracking[population] = {
            "by_regime": {name: _summary(values[i]) for i, name in enumerate(regimes)},
            "by_phase": {
                phase["name"]: _summary(values[3 + i])
                for i, phase in enumerate(COMMAND_TAPE)
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
        "command_tape": deepcopy(COMMAND_TAPE),
        "command_tape_sha256": COMMAND_TAPE_SHA256,
        "full_tape_steps": EVALUATION_STEPS,
        "complete_tape": steps == EVALUATION_STEPS,
        "initial_observation_sha256": initial_hash,
        "initial_dynamics_sha256": dynamics_hash,
        "initial_root_state_sha256": root_state_hash,
        "initial_root_state_status": (
            "RECORDED_ROOT_STATE_W" if root_state_hash else "UNAVAILABLE_NOT_INFERRED"
        ),
        "terrain_assignment": terrain,
        "policy_state_sha256_before": before,
        "policy_state_sha256_after": after,
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
        },
        "sampling": "Pre-action true COM xy velocity, yaw rate and current command; returned auto-reset states are not terminal tracking samples",
        "regimes": "Exact commands: stopped xyz=0; moving xy!=0; pivot xy=0,yaw!=0. Empty groups have null metrics, not success.",
        "tracking": tracking,
    }
    json.dumps(report, allow_nan=False)  # Reject overflow as well as nonfinite inputs.
    return report
