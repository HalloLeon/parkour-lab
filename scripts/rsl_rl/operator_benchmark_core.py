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
# Reserve new layout seeds prospectively; old training archives remain readable.
PROPRIO_EVALUATION_SEEDS = (43, 44, 45, 46, 47, 48)
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


OUT_AND_BACK_SUPPORT_COVERAGE = {
    "version": "operator_out_and_back_support_evidence_v1",
    "foot_order": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    "normal_contact_force_threshold_n": 1.0,
    "minimum_contact_feet": 2,
    "minimum_contiguous_duration_s": 1.0,
    "final_stop_requires_last_terminal_excluded_second": True,
    "maximum_verified_foot_collision_radius_m": 0.10,
    "extra_mesh_boundary_margin_m": 0.10,
    "flat_height_tolerance_m": 0.0001,
    "flat_normal_xy_tolerance": 0.001,
    "spawn_flat_half_width_m": 1.0,
    "flat_band_half_width_m": 0.6,
    "transition_width_m": 1.0,
    "untapered_outer_half_width_m": 6.0,
    "terminal_samples_receive_coverage_credit": False,
    "scope": (
        "Development coverage only: measured foot-center mesh rays plus conservative "
        "collision-footprint region bounds; not exact contact points, a reconstructed "
        "support patch, physical qualification, or high-step acceptance. Plane controls "
        "must visit the same off-pad core, but their measured surface remains flat."
    ),
}
OUT_AND_BACK_PHASE_NAMES = (
    "cold_stand",
    "outbound",
    "stop_outbound",
    "turnaround",
    "stop_after_turn",
    "return",
    "final_stop",
)


def summarize_out_and_back_coverage(trace, protocol):
    """Replay ordered support-region evidence, never infer a cause for missed return.

    All dynamic inputs are captured post-physics/pre-reset. The first terminal
    sample and every later auto-reset episode receive no coverage credit. The
    ordinary command-phase scorer, not this helper, assesses tracking and drift.
    """
    contract = OUT_AND_BACK_SUPPORT_COVERAGE
    if protocol.get("support_coverage") != contract:
        raise ValueError("Unreviewed out-and-back support coverage contract")
    steps, count, dt = (
        protocol.get("steps"),
        protocol.get("num_envs"),
        protocol.get("period_s"),
    )
    if (
        type(steps) is not int
        or steps <= 0
        or type(count) is not int
        or count <= 0
        or dt != DT
    ):
        raise ValueError("Expected a positive fixed 50 Hz out-and-back trace")
    phases = protocol.get("phases", [])
    if tuple(phase.get("name") for phase in phases) != OUT_AND_BACK_PHASE_NAMES:
        raise ValueError("Unexpected out-and-back phase order")
    bounds, cursor = [], 0
    for phase in phases:
        duration = phase.get("duration_s")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not np.isfinite(duration)
            or duration <= 0
            or not np.isclose(duration / dt, round(duration / dt), atol=1e-9, rtol=0)
        ):
            raise ValueError(
                "Out-and-back phase durations must be control-step aligned"
            )
        end = cursor + round(duration / dt)
        bounds.append((phase["name"], cursor, end))
        cursor = end
    if cursor != steps:
        raise ValueError("Out-and-back phases do not fill the captured horizon")
    shapes = {
        "foot_link_position_w": (steps, count, 4, 3),
        "foot_contact_force_w": (steps, count, 4, 3),
        "foot_ground_hit_w": (steps, count, 4, 3),
        "foot_ground_normal_w": (steps, count, 4, 3),
        "foot_collision_radius_m": (4,),
        "foot_names": (4,),
        "env_origins": (count, 3),
        "terrain_column_id": (count,),
        "terminated": (steps, count),
        "time_out": (steps, count),
        "procedural_workspace": (steps, count),
    }
    for name, shape in shapes.items():
        if (
            name not in trace
            or not isinstance(trace[name], np.ndarray)
            or trace[name].shape != shape
        ):
            raise ValueError(f"Invalid out-and-back coverage field: {name}")
    for name in (
        "foot_link_position_w",
        "foot_contact_force_w",
        "foot_ground_hit_w",
        "foot_ground_normal_w",
        "foot_collision_radius_m",
        "env_origins",
    ):
        if not np.issubdtype(trace[name].dtype, np.floating):
            raise ValueError(
                f"Out-and-back physical fields must be floating point: {name}"
            )
    if np.asarray(trace["foot_names"]).tolist() != contract["foot_order"]:
        raise ValueError("Out-and-back foot order differs from the capture contract")
    for name in ("terminated", "time_out", "procedural_workspace"):
        if trace[name].dtype != np.bool_:
            raise ValueError(f"Out-and-back reset masks must be boolean: {name}")
    if np.any(trace["procedural_workspace"] & ~trace["time_out"]):
        raise ValueError("Workspace censoring requires a native timeout")
    columns = np.asarray(trace["terrain_column_id"])
    if not np.issubdtype(columns.dtype, np.integer) or np.any(
        (columns < 0) | (columns >= 20)
    ):
        raise ValueError("Invalid procedural terrain column identity")
    profiles = columns // 4
    for name in (
        "foot_link_position_w",
        "foot_contact_force_w",
        "foot_collision_radius_m",
        "env_origins",
    ):
        if not np.isfinite(trace[name]).all():
            raise ValueError(f"Nonfinite out-and-back coverage field: {name}")
    radii = np.asarray(trace["foot_collision_radius_m"], dtype=np.float64)
    if np.any(radii <= 0) or np.any(
        radii > contract["maximum_verified_foot_collision_radius_m"]
    ):
        raise ValueError("Unbounded or oversized native foot collision footprint")
    guard = radii + contract["extra_mesh_boundary_margin_m"]
    origins = np.asarray(trace["env_origins"], dtype=np.float64)
    feet = np.asarray(trace["foot_link_position_w"], dtype=np.float64)
    local_xy = feet[..., :2] - origins[None, :, None, :2]
    absolute_xy = np.abs(local_xy)
    largest_xy = absolute_xy.max(axis=-1)
    flat_region = (
        (largest_xy + guard <= contract["spawn_flat_half_width_m"])
        | (absolute_xy[..., 1] + guard <= contract["flat_band_half_width_m"])
    ) & (largest_xy + guard <= contract["untapered_outer_half_width_m"])
    core_region = (
        (
            largest_xy - guard
            >= contract["spawn_flat_half_width_m"] + contract["transition_width_m"]
        )
        & (
            absolute_xy[..., 1] - guard
            >= contract["flat_band_half_width_m"] + contract["transition_width_m"]
        )
        & (largest_xy + guard <= contract["untapered_outer_half_width_m"])
    )
    hit = np.asarray(trace["foot_ground_hit_w"], dtype=np.float64)
    normal = np.asarray(trace["foot_ground_normal_w"], dtype=np.float64)
    ray_valid = np.isfinite(hit).all(axis=-1) & np.isfinite(normal).all(axis=-1)
    safe_normal = np.where(ray_valid[..., None], normal, 0.0)
    if np.any(
        ray_valid
        & (
            ~np.isclose(np.linalg.norm(safe_normal, axis=-1), 1.0, atol=1e-3, rtol=0)
            | (safe_normal[..., 2] <= 0)
        )
    ):
        raise ValueError("Expected upward unit normals from the actual terrain mesh")
    # Native float32 world coordinates lose precision far from the scene origin.
    xy_tolerance = np.maximum(
        2e-5, 4 * np.finfo(np.float32).eps * np.abs(feet[..., :2])
    )
    if np.any(
        ray_valid[..., None]
        & (
            np.abs(np.where(ray_valid[..., None], hit, feet)[..., :2] - feet[..., :2])
            > xy_tolerance
        )
    ):
        raise ValueError(
            "Foot ground rays are not vertically aligned with the named feet"
        )
    height = np.where(ray_valid, hit[..., 2] - origins[None, :, None, 2], 0.0)
    mesh_flat = (
        ray_valid
        & (np.abs(height) <= contract["flat_height_tolerance_m"])
        & (
            np.linalg.norm(safe_normal[..., :2], axis=-1)
            <= contract["flat_normal_xy_tolerance"]
        )
    )
    contact = (
        np.linalg.norm(trace["foot_contact_force_w"].astype(np.float64), axis=-1)
        >= contract["normal_contact_force_threshold_n"]
    )
    minimum_feet = contract["minimum_contact_feet"]
    central_flat_support = flat_region.all(axis=-1) & (
        (contact & mesh_flat).sum(axis=-1) >= minimum_feet
    )
    expected_core_mesh = np.where(
        profiles[None, :, None] == 0, mesh_flat, ray_valid & ~mesh_flat
    )
    core_support = (contact & core_region & expected_core_mesh).sum(
        axis=-1
    ) >= minimum_feet
    minimum_samples = round(contract["minimum_contiguous_duration_s"] / dt)
    names = ("plane", "rough_flat", "hills", "step_hills", "tilted_ramps")
    trials = []
    for env_id in range(count):
        done = np.flatnonzero(
            trace["terminated"][:, env_id] | trace["time_out"][:, env_id]
        )
        terminal = int(done[0]) if len(done) else None
        eligible_end = steps if terminal is None else terminal
        physical = terminal is not None and bool(trace["terminated"][terminal, env_id])
        workspace = terminal is not None and bool(
            trace["procedural_workspace"][terminal, env_id]
        )
        complete = not physical and not workspace and terminal in (None, steps - 1)
        phase_results = []
        for name, start, end in bounds:
            support = (
                central_flat_support
                if name in ("cold_stand", "final_stop")
                else core_support
            )
            selected = support[start : min(end, max(start, eligible_end)), env_id]
            edges = np.flatnonzero(
                np.diff(np.r_[False, selected, False].astype(np.int8))
            )
            longest = int(np.max(edges[1::2] - edges[::2])) if len(edges) else 0
            observed = longest >= minimum_samples
            if name == "final_stop":
                observed = len(selected) >= minimum_samples and bool(
                    selected[-minimum_samples:].all()
                )
            phase_results.append(
                {
                    "name": name,
                    "required_region": (
                        "central_flat"
                        if name in ("cold_stand", "final_stop")
                        else "off_pad_core"
                    ),
                    "eligible_samples": int(len(selected)),
                    "support_samples": int(selected.sum()),
                    "longest_contiguous_support_s": longest * dt,
                    "coverage_window": (
                        "last_terminal_excluded_second"
                        if name == "final_stop"
                        else "any_contiguous_second"
                    ),
                    "observed": observed,
                }
            )
        uncovered = [phase["name"] for phase in phase_results if not phase["observed"]]
        observed = complete and not uncovered
        status = (
            "physical_failure"
            if physical
            else (
                "workspace_timeout"
                if workspace
                else (
                    "early_timeout"
                    if not complete
                    else "coverage_observed" if observed else "coverage_uncovered"
                )
            )
        )
        trials.append(
            {
                "env_id": env_id,
                "profile": names[int(profiles[env_id])],
                "plane_control": bool(profiles[env_id] == 0),
                "status": status,
                "first_terminal_step": terminal,
                "first_attempt_complete": complete,
                "ordered_coverage_observed": observed,
                "uncovered_phases": uncovered,
                "phases": phase_results,
            }
        )
    return {
        "version": contract["version"],
        "policy_acceptance": False,
        "scope": contract["scope"],
        "missed_return_interpretation": (
            "Uncovered route, not proof of terrain incapability: open-loop yaw or translation "
            "tracking can prevent return. Inspect separate command-phase diagnostics; no "
            "feedback steering, reset assistance or retry earns coverage."
        ),
        "ordered_coverage_observed": sum(
            trial["ordered_coverage_observed"] for trial in trials
        ),
        "total": count,
        "profiles": {
            name: {
                "ordered_coverage_observed": sum(
                    t["ordered_coverage_observed"]
                    for t in trials
                    if t["profile"] == name
                ),
                "total": sum(t["profile"] == name for t in trials),
            }
            for name in names
        },
        "trials": trials,
    }


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


def score_course_trace(trace: dict[str, np.ndarray], course: dict) -> dict:
    """First-attempt transfer accounting; production route gates own completion.

    Preserve the pre-reset outcome, even if a later auto-reset episode succeeds.
    This is privileged, waypoint-guided motor transfer, never joint acceptance.
    """
    n = len(trace["initial_position"])
    if n < 3:
        raise ValueError("Course transfer requires at least three initial trials")
    masks = (
        "terminated",
        "time_out",
        "base_contact",
        "course_chassis",
        "course_fall",
        "course_off_route",
        "course_success",
    )
    shapes = {k: (STEPS, n) for k in masks}
    shapes.update(
        {
            k: (STEPS, n, 3)
            for k in (
                "position",
                "command",
                "linear_velocity_b",
                "linear_velocity_w",
                "angular_velocity_b",
                "angular_velocity_w",
            )
        }
    )
    shapes.update(
        quaternion=(STEPS, n, 4),
        waypoint_index=(STEPS, n),
        terminal_predicates=(STEPS, n, 10),
        terminal_dwell_s=(STEPS, n),
        feet_position=(STEPS, n, 4, 3),
        feet_force=(STEPS, n, 4, 3),
        base_height_ray=(STEPS, n, 3),
        contact_force_history=(STEPS, n, 3, len(course["contact_body_names"]), 3),
        initial_position=(n, 3),
        initial_quaternion=(n, 4),
    )
    for key, shape in shapes.items():
        if key not in trace or trace[key].shape != shape:
            raise ValueError(f"Incomplete course trace: {key} must have shape {shape}")
    if any(trace[k].dtype != np.bool_ for k in (*masks, "terminal_predicates")):
        raise ValueError("Course event/predicate masks must be boolean")
    if not np.issubdtype(trace["waypoint_index"].dtype, np.integer):
        raise ValueError("Waypoint cursors must be integers")
    final_index = len(course["course"]["waypoints"]) - 1
    origins = np.asarray(course["env_origins"])
    if origins.shape != (n, 3) or not np.isfinite(origins).all():
        raise ValueError("Missing finite physical course origins")
    body_names = course["contact_body_names"]
    chassis_ids = [
        i
        for i, name in enumerate(body_names)
        if name == "base" or name.startswith("Head_")
    ]
    if (
        "base" not in body_names
        or not chassis_ids
        or len(body_names) != len(set(body_names))
    ):
        raise ValueError("Invalid chassis contact identity")
    supports = {
        s["name"]: np.asarray(s["vertices"], dtype=float)
        for s in course["course"]["support_regions"]
    }
    final_waypoint = course["course"]["waypoints"][-1]

    def loaded_feet(env_id, rows):
        # Independent reconciliation against the recorded named support polygon;
        # online progression still uses the authoritative production route code.
        vertices = supports[final_waypoint["support_region_name"]]
        edges = np.roll(vertices, -1, axis=0) - vertices
        normal = np.cross(edges[0], edges[1])
        normal /= np.linalg.norm(normal)
        feet = trace["feet_position"][rows, env_id] - origins[env_id]
        distance = (feet - vertices[0]) @ normal
        projected = feet - distance[..., None] * normal
        inward = np.sum(
            np.cross(edges, projected[..., None, :] - vertices) * normal, axis=-1
        ) / np.linalg.norm(edges, axis=-1)
        supported = (inward >= -0.05).all(axis=-1) & (np.abs(distance) <= 0.12)
        return supported & ((trace["feet_force"][rows, env_id] @ normal) >= 10.0)

    trials = []
    for env_id in range(n):
        terminal = np.flatnonzero(
            trace["terminated"][:, env_id] | trace["time_out"][:, env_id]
        )
        length = int(terminal[0] + 1) if len(terminal) else STEPS
        failures = [name for name in masks[2:-1] if trace[name][:length, env_id].any()]
        physical = (
            "position",
            "quaternion",
            "linear_velocity_b",
            "linear_velocity_w",
            "angular_velocity_b",
            "angular_velocity_w",
            "command",
            "feet_position",
            "feet_force",
            "contact_force_history",
        )
        if not all(np.isfinite(trace[k][:length, env_id]).all() for k in physical):
            failures.append("nonfinite physical state")
        if not np.allclose(
            np.linalg.norm(trace["quaternion"][:length, env_id], axis=-1),
            1.0,
            atol=1e-3,
            rtol=0,
        ):
            failures.append("invalid quaternion")
        indices = trace["waypoint_index"][:length, env_id]
        increments = np.diff(np.r_[0, indices])
        if (
            (indices < 0).any()
            or (indices > final_index).any()
            or not np.isin(increments, [0, 1]).all()
        ):
            raise ValueError("Invalid ordered waypoint progression before reset")
        success = bool(trace["course_success"][length - 1, env_id])
        if trace["course_success"][: length - 1, env_id].any():
            raise ValueError("Course success reported before termination")
        if not success:
            failures.append("course not completed within first attempt")
        elif (
            not trace["terminated"][length - 1, env_id]
            or trace["time_out"][length - 1, env_id]
            or indices[-1] != final_index
            or length < 10
            or not (indices[length - 10 : length] == final_index).all()
            or not trace["terminal_predicates"][length - 10 : length, env_id].all()
            or not np.isfinite(trace["terminal_dwell_s"][length - 1, env_id])
            or trace["terminal_dwell_s"][length - 1, env_id] + 1e-6 < 0.2
        ):
            raise ValueError("Course success lacks ordered supported stable completion")
        if success:
            rows = slice(length - 10, length)
            position = trace["position"][rows, env_id]
            ray = trace["base_height_ray"][rows, env_id]
            velocity = trace["linear_velocity_w"][rows, env_id]
            angular = trace["angular_velocity_w"][rows, env_id]
            quaternion = trace["quaternion"][rows, env_id]
            w, x, y, z = quaternion.T
            tilt_sine = np.hypot(2 * (x * z - w * y), 2 * (y * z + w * x))
            radius = final_waypoint["root_reach_radius"] or 0.20
            valid_finish = (
                np.isfinite(ray).all()
                and (
                    position[:, 2] - ray[:, 2] > course["course"]["min_clearance"]
                ).all()
                and (loaded_feet(env_id, rows).sum(axis=-1) >= 2).all()
                and (
                    np.linalg.norm(
                        position[:, :2]
                        - origins[env_id, :2]
                        - np.asarray(final_waypoint["position"])[:2],
                        axis=-1,
                    )
                    < radius
                ).all()
                and (np.linalg.norm(velocity[:, :2], axis=-1) < 0.2).all()
                and (np.abs(velocity[:, 2]) < 0.2).all()
                and (np.abs(angular[:, 2]) < 0.35).all()
                and (np.linalg.norm(angular[:, :2], axis=-1) < 0.35).all()
                and (tilt_sine < 0.25).all()
            )
            if not valid_finish:
                raise ValueError(
                    "Claimed stable finish contradicts recorded feet/load/clearance/motion"
                )
        chassis_history = trace["contact_force_history"][:length, env_id][
            :, :, chassis_ids
        ]
        if (
            np.linalg.norm(chassis_history, axis=-1) > 1.0
        ).any() and "course_chassis" not in failures:
            raise ValueError(
                "Chassis contact is missing from physical failure accounting"
            )
        trials.append(
            {
                "env_id": env_id,
                "passed": not failures,
                "failures": failures,
                "observed_duration_s": length * DT,
                "final_waypoint_index": int(indices[-1]),
            }
        )
    return {
        "status": "PASS" if all(t["passed"] for t in trials) else "FAIL",
        "policy_acceptance": False,
        "scope": "Privileged waypoint-guided stock motor transfer only; not autonomous navigation, causal student, operator confirmation or critic exit. No flat attitude gate on banked terrain.",
        "course": course,
        "passed": sum(t["passed"] for t in trials),
        "total": n,
        "trials": trials,
    }


def score_command_phase(
    command, velocity, body_yaw, world_yaw, position, heading, *, dt=DT, thresholds=None
):
    """Complete-phase kinematics, not trajectory/terrain acceptance.

    Position and unwrapped heading include the pre-command pose as sample zero;
    velocities contain only post-physics samples. Callers retain failure/censoring.
    """
    th = thresholds or Thresholds()
    cmd = np.asarray(command)
    length = len(velocity)
    block, settling = round(th.block_s / dt), round(th.settling_s / dt)
    if (
        length < settling
        or velocity.shape != (length, 2)
        or body_yaw.shape != (length,)
        or world_yaw.shape != (length,)
        or position.shape != (length + 1, 2)
        or heading.shape != (length + 1,)
        or cmd.shape != (3,)
        or not all(
            np.isfinite(a).all()
            for a in (cmd, velocity, body_yaw, world_yaw, position, heading)
        )
    ):
        raise ValueError(
            "Command scoring requires a complete finite phase and onset pose"
        )
    planar_error = np.linalg.norm(velocity - cmd[:2], axis=-1)
    yaw_error = np.maximum(np.abs(world_yaw - cmd[2]), np.abs(body_yaw - cmd[2]))
    stationary = np.linalg.norm(cmd[:2]) == 0
    yaw_limit = th.pivot_yaw_error_rad_s if stationary else th.moving_yaw_error_rad_s
    planar_limit = th.stationary_speed_m_s if stationary else th.planar_error_m_s
    # The first block ENDS at the acquisition deadline; late recovery cannot pass it.
    blocks = [
        {
            "start_s": offset * dt,
            "duration_s": (min(offset + block, length) - offset) * dt,
            "planar_error_m_s": float(planar_error[offset : offset + block].mean()),
            "yaw_error_rad_s": float(yaw_error[offset : offset + block].mean()),
        }
        for offset in range(settling - block, length, block)
    ]
    failed_blocks = [
        i
        for i, b in enumerate(blocks)
        if b["planar_error_m_s"] > planar_limit or b["yaw_error_rad_s"] > yaw_limit
    ]
    failures = ["tracking not sustained after 1 s"] if failed_blocks else []
    details = {
        "complete": True,
        "command": cmd.tolist(),
        "blocks": blocks,
        "failed_tracking_blocks": failed_blocks,
        "acquisition_failed": 0 in failed_blocks,
        "later_tracking_failed": any(i > 0 for i in failed_blocks),
        "mean_body_yaw_rate_rad_s": (
            float(body_yaw[settling:].mean()) if length > settling else None
        ),
        "mean_world_up_yaw_rate_rad_s": (
            float(world_yaw[settling:].mean()) if length > settling else None
        ),
    }
    if cmd[2] != 0:
        wrong_sign = (
            max(
                float(np.mean(rate[settling:] * cmd[2] < 0))
                for rate in (world_yaw, body_yaw)
            )
            if length > settling
            else None
        )
        details["wrong_sign_fraction"] = wrong_sign
        if wrong_sign is None:
            failures.append("missing settled yaw samples")
        elif wrong_sign > th.wrong_sign_fraction:
            failures.append("excessive wrong-sign yaw")
    if stationary:
        excursion = float(np.linalg.norm(position[1:] - position[0], axis=-1).max())
        drift = None
        if length > settling:
            drift = max(
                float(
                    np.linalg.norm(
                        position[offset : offset + round(2 / dt) + 1]
                        - position[offset],
                        axis=-1,
                    ).max()
                )
                for offset in range(settling + 1, len(position))
            )
        details.update(onset_excursion_m=excursion, maximum_settled_drift_2s_m=drift)
        if excursion > th.stationary_onset_excursion_m or (
            drift is not None and drift > th.stationary_drift_2s_m
        ):
            failures.append("excessive stationary drift/braking distance")
        if cmd[2] == 0:
            excursion = float(np.abs(heading[1:] - heading[0]).max())
            details["heading_onset_excursion_rad"] = excursion
            if excursion > th.zero_twist_heading_excursion_rad:
                failures.append("excessive zero-twist heading drift")
    details.update(failures=failures, kinematic_passed=not failures)
    return details


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
                onset = (
                    trace["initial_position"][env_id, :2]
                    if start == 0
                    else trace["position"][start - 1, env_id, :2]
                )
                details = score_command_phase(
                    phase.command,
                    trace["linear_velocity_b"][start:end, env_id, :2],
                    trace["angular_velocity_b"][start:end, env_id, 2],
                    trace["angular_velocity_w"][start:end, env_id, 2],
                    np.vstack((onset, trace["position"][start:end, env_id, :2])),
                    heading[start : end + 1],
                    thresholds=th,
                )
                details.update(name=phase.name, start_s=start * DT)
                details["failures"] = [
                    f"{phase.name}: {reason}" for reason in details["failures"]
                ]
                failures.extend(details["failures"])
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
