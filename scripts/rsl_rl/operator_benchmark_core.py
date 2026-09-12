"""Simulator-independent command sequences and scoring for the stock Go2 reference.

This is an engineering screen, not a certificate of operator reliability. Commands
are body-frame (vx, vy, wz); no waypoint, planner, or command assistance is used.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import numpy as np
import yaml


DT = 0.02
DURATION_S = 20.0
STEPS = round(DURATION_S / DT)
OBSERVATION_TERMS = (
    "base_lin_vel",
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
)


@dataclass(frozen=True)
class Phase:
    name: str
    duration_s: float
    command: tuple[float, float, float]


@dataclass(frozen=True)
class Thresholds:
    settling_s: float = 1.0
    block_s: float = 0.4
    planar_error_m_s: float = 0.10
    pivot_yaw_error_rad_s: float = 0.10
    moving_yaw_error_rad_s: float = 0.15
    wrong_sign_fraction: float = 0.05
    stationary_speed_m_s: float = 0.08
    stationary_onset_excursion_m: float = 0.15
    stationary_drift_2s_m: float = 0.10
    zero_twist_heading_excursion_rad: float = 0.15
    flat_attitude_rad: float = np.deg2rad(15.0)


def profiles() -> dict[str, tuple[Phase, ...]]:
    """Fixed, complete 20-s trajectories; initial standing also tests cold start."""
    stop = (0.0, 0.0, 0.0)
    forward = (0.55, 0.0, 0.0)
    result = {"stand": (Phase("stand", 20.0, stop),)}
    for name, command in (
        ("stop_restart", stop),
        ("pivot_positive", (0.0, 0.0, 0.5)),
        ("pivot_negative", (0.0, 0.0, -0.5)),
        ("arc_positive", (0.55, 0.0, 0.5)),
        ("arc_negative", (0.55, 0.0, -0.5)),
        ("reverse", (-0.3, 0.0, 0.0)),
        ("lateral_positive", (0.0, 0.2, 0.0)),
        ("lateral_negative", (0.0, -0.2, 0.0)),
    ):
        result[name] = (
            Phase("initial_stand", 2.0, stop),
            Phase("forward", 4.0, forward),
            Phase(name, 6.0, command),
            Phase("stop", 4.0, stop),
            Phase("restart", 4.0, forward),
        )
    # A continuous stop must keep ONE drift anchor, not restart the holding
    # allowance at an artificial phase boundary with an unchanged command.
    result["stop_restart"] = (
        Phase("initial_stand", 2.0, stop),
        Phase("forward", 4.0, forward),
        Phase("stop", 10.0, stop),
        Phase("restart", 4.0, forward),
    )
    result["yaw_reversal"] = (
        Phase("initial_stand", 2.0, stop),
        Phase("pivot_positive", 6.0, (0.0, 0.0, 0.5)),
        Phase("pivot_negative", 6.0, (0.0, 0.0, -0.5)),
        Phase("stop", 3.0, stop),
        Phase("restart", 3.0, forward),
    )
    return result


def command_schedule(repetitions: int) -> tuple[list[str], np.ndarray]:
    if isinstance(repetitions, bool) or not 1 <= repetitions <= 100:
        raise ValueError("repetitions must be in [1, 100]")
    labels, schedules = [], []
    for name, phases in profiles().items():
        rows = np.concatenate(
            [np.tile(p.command, (round(p.duration_s / DT), 1)) for p in phases]
        ).astype(np.float32)
        if rows.shape != (STEPS, 3):
            raise ValueError(f"Invalid schedule: {name}")
        labels.extend([name] * repetitions)
        schedules.extend([rows] * repetitions)
    return labels, np.stack(schedules, axis=1)


def read_yaml_data(path: Path) -> dict:
    # BaseLoader constructs only strings/lists/dicts, even for !!python tags.
    # Never construct Python objects or import callables from supplied YAML.
    result = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
    if not isinstance(result, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return result


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config_differences(saved: dict, current: dict) -> list[str]:
    """Reject physics/observation/action drift, excluding only documented metadata.

    Both inputs use BaseLoader scalar strings. Compare BEFORE benchmark overrides.
    Scene construction fills terrain.num_envs/env_spacing in the archived config.
    """

    def contract(source):
        data = copy.deepcopy(source)
        selected = {
            k: data[k]
            for k in (
                "decimation",
                "episode_length_s",
                "actions",
                "observations",
                "events",
                "terminations",
                "rewards",
                "curriculum",
            )
        }
        selected["sim"] = {
            k: data["sim"][k]
            for k in (
                "dt",
                "gravity",
                "physx",
                "physics_material",
                "use_fabric",
                "create_stage_in_memory",
                "render_interval",
            )
        }
        selected["scene"] = {
            k: data["scene"][k]
            for k in (
                "robot",
                "terrain",
                "height_scanner",
                "contact_forces",
                "replicate_physics",
                "filter_collisions",
                "lazy_sensor_update",
                "clone_in_fabric",
            )
        }
        for k in ("num_envs", "env_spacing", "visual_material", "debug_vis"):
            selected["scene"]["terrain"].pop(k, None)
        # InteractiveScene resolves this known placeholder in-place during
        # construction. Archived configs contain paths; fresh ones use the token.
        for name in ("robot", "contact_forces"):
            asset = selected["scene"][name]
            if "prim_path" in asset:
                asset["prim_path"] = asset["prim_path"].replace(
                    "{ENV_REGEX_NS}", "/World/envs/env_.*"
                )
        # Observation order is part of the contract, unlike ordinary dict equality.
        selected["observation_order"] = [
            k
            for k, v in data["observations"]["policy"].items()
            if isinstance(v, dict) and "func" in v
        ]
        return selected

    def differences(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            result = []
            for key in sorted(a.keys() | b.keys()):
                if key not in a or key not in b:
                    result.append(f"{path}.{key} (missing)")
                else:
                    result.extend(differences(a[key], b[key], f"{path}.{key}"))
            return result
        return [] if a == b else [path]

    return differences(contract(saved), contract(current), "env")


def load_reference_checkpoint(checkpoint: Path, agent: dict):
    """Load the narrow stock actor/critic contract with restricted unpickling.

    A deliberately narrow contract prevents accidentally loading a parkour/RMA
    checkpoint, normalization state, recurrent policy, or another action interface.
    """
    import torch

    policy = agent["policy"]
    expected = {
        "class_name": "ActorCritic",
        "actor_hidden_dims": ["128"] * 3,
        "critic_hidden_dims": ["128"] * 3,
        "activation": "elu",
        "actor_obs_normalization": "false",
        "critic_obs_normalization": "false",
        "state_dependent_std": "false",
        "noise_std_type": "scalar",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"Unsupported reference policy.{key}: {policy.get(key)}")
    if agent.get("clip_actions") != "null" or agent.get(
        "empirical_normalization"
    ) not in ("null", "false"):
        raise ValueError("Reference requires unclipped actions and no normalization")
    if agent.get("obs_groups") not in (
        {},
        {"policy": ["policy"], "critic": ["policy"]},
    ):
        raise ValueError("Reference requires the stock policy observation group")
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = data["model_state_dict"]
    expected_keys = {"std"}
    for prefix in ("actor", "critic"):
        expected_keys.update(
            f"{prefix}.{i}.{kind}" for i in (0, 2, 4, 6) for kind in ("weight", "bias")
        )
    if set(state) != expected_keys or any(
        not torch.isfinite(v).all() for v in state.values()
    ):
        raise ValueError("Unexpected or nonfinite reference checkpoint tensors")
    for prefix, last_dim in (("actor", 12), ("critic", 1)):
        for index, (inputs, outputs) in enumerate(
            zip((48, 128, 128, 128), (128, 128, 128, last_dim))
        ):
            if state[f"{prefix}.{2 * index}.weight"].shape != (
                outputs,
                inputs,
            ) or state[f"{prefix}.{2 * index}.bias"].shape != (outputs,):
                raise ValueError(f"Invalid reference {prefix} layer {index} shape")
    if state["std"].shape != (12,) or (state["std"] <= 0).any():
        raise ValueError("Reference action standard deviations must be positive")
    if type(data.get("iter")) is not int or data["iter"] < 0:
        raise ValueError(
            "Reference checkpoint must have a nonnegative integer iteration"
        )
    return data


def load_reference_actor(checkpoint: Path, agent: dict):
    """Load just the validated deterministic 48→12 ELU motor."""
    import torch

    data = load_reference_checkpoint(checkpoint, agent)
    state = data["model_state_dict"]
    layers = []
    for index, (inputs, outputs) in enumerate(
        zip((48, 128, 128, 128), (128, 128, 128, 12))
    ):
        if index:
            layers.append(torch.nn.ELU())
        layers.append(torch.nn.Linear(inputs, outputs))
    actor = torch.nn.Sequential(*layers)
    actor.load_state_dict(
        {
            k.removeprefix("actor."): v
            for k, v in state.items()
            if k.startswith("actor.")
        },
        strict=True,
    )
    return actor.eval().requires_grad_(False), int(data["iter"])


def score_trace(
    trace: dict[str, np.ndarray], labels: list[str], thresholds=None
) -> dict:
    """Score every trial, retaining failures even when the simulator auto-resets.

    Post-step samples must be captured BEFORE reset; pre-step positions define
    stopping excursion. After the first termination no later samples are scored.
    Errors are computed from physical signals, never training reward integrals.
    """
    th = thresholds or Thresholds()
    n = len(labels)
    shapes = {
        "command": (STEPS, n, 3),
        "position": (STEPS, n, 3),
        "quaternion": (STEPS, n, 4),
        "linear_velocity_b": (STEPS, n, 3),
        "angular_velocity_b": (STEPS, n, 3),
        "angular_velocity_w": (STEPS, n, 3),
        "terminated": (STEPS, n),
        "time_out": (STEPS, n),
        "initial_position": (n, 3),
        "initial_quaternion": (n, 4),
    }
    for key, shape in shapes.items():
        if key not in trace or trace[key].shape != shape:
            raise ValueError(f"Incomplete trace: {key} must have shape {shape}")
    if n == 0 or n % len(profiles()):
        raise ValueError("Missing operator profiles")
    expected_labels, schedule = command_schedule(n // len(profiles()))
    if labels != expected_labels or not np.allclose(
        trace["command"], schedule, rtol=0, atol=1e-6
    ):
        raise ValueError("Executed command sequence differs from the benchmark")
    for key in ("terminated", "time_out"):
        if trace[key].dtype != np.bool_:
            raise ValueError(f"{key} must be a boolean mask")
    results = []
    block = round(th.block_s / DT)
    settling = round(th.settling_s / DT)
    for env_id, label in enumerate(labels):
        failures, phase_reports = [], []
        sequence_failures = []
        terminal = np.flatnonzero(
            trace["terminated"][:, env_id] | trace["time_out"][:, env_id]
        )
        length = int(terminal[0] + 1) if len(terminal) else STEPS
        if trace["terminated"][:length, env_id].any():
            failures.append("physical termination")
        if trace["time_out"][:length, env_id].any():
            failures.append("unexpected timeout within benchmark")
        if length < STEPS:
            failures.append(f"sequence interrupted at {length * DT:.2f} s")
        signals = [
            trace[k][:length, env_id]
            for k in shapes
            if k
            not in ("initial_position", "initial_quaternion", "terminated", "time_out")
        ]
        if not all(np.isfinite(x).all() for x in signals) or not all(
            np.isfinite(trace[k][env_id]).all()
            for k in ("initial_position", "initial_quaternion")
        ):
            failures.append("nonfinite physical state")
        else:
            quat = trace["quaternion"][:length, env_id]
            all_quats = np.vstack((trace["initial_quaternion"][env_id], quat))
            if not np.allclose(
                np.linalg.norm(all_quats, axis=-1), 1.0, atol=1e-3, rtol=0
            ):
                failures.append("invalid quaternion")
            w, x, y, z = quat.T
            roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
            pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
            qw, qx, qy, qz = all_quats.T
            heading = np.unwrap(
                np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            )
            if max(np.abs(roll).max(), np.abs(pitch).max()) > th.flat_attitude_rad:
                failures.append("flat attitude exceeds 15 degrees")
            sequence_failures = failures.copy()
            start = 0
            for phase in profiles()[label]:
                end = start + round(phase.duration_s / DT)
                if end > length:
                    phase_reports.append({"name": phase.name, "complete": False})
                    start = end
                    continue
                failure_start = len(failures)
                cmd = np.asarray(phase.command)
                velocity = trace["linear_velocity_b"][start:end, env_id, :2]
                yaw = trace["angular_velocity_w"][start:end, env_id, 2]
                body_yaw = trace["angular_velocity_b"][start:end, env_id, 2]
                planar_error = np.linalg.norm(velocity - cmd[:2], axis=-1)
                # Keep both conventions explicit; require body and world-up rate.
                yaw_error = np.maximum(np.abs(yaw - cmd[2]), np.abs(body_yaw - cmd[2]))
                stationary = np.linalg.norm(cmd[:2]) == 0
                yaw_limit = (
                    th.pivot_yaw_error_rad_s
                    if stationary
                    else th.moving_yaw_error_rad_s
                )
                planar_limit = (
                    th.stationary_speed_m_s if stationary else th.planar_error_m_s
                )
                blocks = []
                # The block ENDING at the 1-s deadline must already track;
                # good behavior starting later cannot satisfy acquisition.
                for offset in range(settling - block, end - start, block):
                    tail = slice(offset, min(offset + block, end - start))
                    blocks.append(
                        {
                            "start_s": offset * DT,
                            "duration_s": (tail.stop - offset) * DT,
                            "planar_error_m_s": float(planar_error[tail].mean()),
                            "yaw_error_rad_s": float(yaw_error[tail].mean()),
                        }
                    )
                failed_blocks = [
                    index
                    for index, b in enumerate(blocks)
                    if (
                        b["planar_error_m_s"] > planar_limit
                        or b["yaw_error_rad_s"] > yaw_limit
                    )
                ]
                if failed_blocks:
                    failures.append(f"{phase.name}: tracking not sustained after 1 s")
                details = {
                    "name": phase.name,
                    "complete": True,
                    "start_s": start * DT,
                    "command": list(phase.command),
                    "blocks": blocks,
                    "failed_tracking_blocks": failed_blocks,
                    "acquisition_failed": 0 in failed_blocks,
                    "later_tracking_failed": any(i > 0 for i in failed_blocks),
                    "mean_body_yaw_rate_rad_s": float(body_yaw[settling:].mean()),
                    "mean_world_up_yaw_rate_rad_s": float(yaw[settling:].mean()),
                }
                if cmd[2] != 0:
                    wrong_sign = max(
                        float(np.mean(rate[settling:] * cmd[2] < 0))
                        for rate in (yaw, body_yaw)
                    )
                    details["wrong_sign_fraction"] = wrong_sign
                    if wrong_sign > th.wrong_sign_fraction:
                        failures.append(f"{phase.name}: excessive wrong-sign yaw")
                if stationary:
                    position = trace["position"][start:end, env_id, :2]
                    onset = (
                        trace["initial_position"][env_id, :2]
                        if start == 0
                        else trace["position"][start - 1, env_id, :2]
                    )
                    excursion = float(np.linalg.norm(position - onset, axis=-1).max())
                    drift = 0.0
                    for offset in range(settling, len(position)):
                        window = position[
                            offset : min(offset + round(2 / DT) + 1, len(position))
                        ]
                        drift = max(
                            drift,
                            float(np.linalg.norm(window - window[0], axis=-1).max()),
                        )
                    details.update(
                        onset_excursion_m=excursion, maximum_settled_drift_2s_m=drift
                    )
                    if (
                        excursion > th.stationary_onset_excursion_m
                        or drift > th.stationary_drift_2s_m
                    ):
                        failures.append(
                            f"{phase.name}: excessive stationary drift/braking distance"
                        )
                    if cmd[2] == 0:
                        # heading[start] is the pre-command pose, including at t=0.
                        heading_excursion = float(
                            np.abs(heading[start + 1 : end + 1] - heading[start]).max()
                        )
                        details["heading_onset_excursion_rad"] = heading_excursion
                        if heading_excursion > th.zero_twist_heading_excursion_rad:
                            failures.append(
                                f"{phase.name}: excessive zero-twist heading drift"
                            )
                details["failures"] = failures[failure_start:]
                # Phase-local kinematics never override a whole-sequence fall,
                # invalid state or attitude violation. This is NOT acceptance.
                details["kinematic_passed"] = not details["failures"]
                phase_reports.append(details)
                start = end
        if not phase_reports:
            sequence_failures = failures.copy()
        results.append(
            {
                "env_id": env_id,
                "profile": label,
                "passed": not failures,
                "observed_duration_s": length * DT,
                "failures": failures,
                "sequence_failures": sequence_failures,
                "phases": phase_reports,
            }
        )
    counts = {
        name: {
            "passed": sum(r["passed"] for r in results if r["profile"] == name),
            "total": sum(r["profile"] == name for r in results),
        }
        for name in profiles()
    }
    return {
        "schema_version": 2,
        "status": "PASS" if all(r["passed"] for r in results) else "FAIL",
        "scope": "stock privileged-velocity flat-ground deterministic operator screen; not RMA or obstacle acceptance",
        "thresholds": asdict(th),
        "profiles": counts,
        "phase_summary": summarize_phases(results),
        "trials": results,
    }


def summarize_phases(trials: list[dict]) -> dict:
    """Diagnostic aggregation only; expected phases include unobserved suffixes.

    A failed forward acquisition can fail a reverse *trajectory* while its
    reverse segment tracks correctly. Never count truncated/unobserved phases
    as passing, or conflate local kinematics with whole-trajectory safety.
    """
    rows = {}
    for trial in trials:
        observed = {phase["name"]: phase for phase in trial["phases"]}
        for expected in profiles()[trial["profile"]]:
            row = rows.setdefault(
                expected.name,
                dict(
                    expected=0,
                    complete=0,
                    kinematic_passed=0,
                    acquisition_failed=0,
                    later_tracking_failed=0,
                    excursion_failed=0,
                    heading_failed=0,
                    wrong_sign_failed=0,
                    sequence_failed=0,
                ),
            )
            row["expected"] += 1
            row["sequence_failed"] += bool(trial["sequence_failures"])
            phase = observed.get(expected.name, {})
            if not phase.get("complete", False):
                continue
            row["complete"] += 1
            for key in (
                "kinematic_passed",
                "acquisition_failed",
                "later_tracking_failed",
            ):
                row[key] += phase[key]
            for key, reason in (
                ("excursion_failed", "excessive stationary drift/braking distance"),
                ("heading_failed", "excessive zero-twist heading drift"),
                ("wrong_sign_failed", "excessive wrong-sign yaw"),
            ):
                row[key] += f"{expected.name}: {reason}" in phase["failures"]
    return {
        "scope": "diagnostic phase-local kinematics; NOT trajectory acceptance",
        "acquisition": "first block ending at the unchanged settling deadline",
        "later_tracking": "any subsequent block; can overlap acquisition failure",
        "sequence_failures": "termination, timeout, interruption, nonfinite/invalid state or attitude",
        "phases": rows,
    }
