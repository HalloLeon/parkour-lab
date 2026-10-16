"""Read-only saved terrain-attempt verification and kinematic analysis.

Archive integrity and physical receipts are checked without running a controller
or starting the simulator. These diagnostics do not establish qualification.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from parkour_lab.control.command_tape import validate_tape
from parkour_lab.provenance import file_sha256


def analyze_terrain_attempt(run):
    """Verify saved evidence and score one attempt; never execute or replace it.

    This is single-attempt kinematics, not a frozen terrain bank, independently
    validated layout/randomization, cooked-contact certification or qualification.
    """
    from parkour_lab.control.proprioception import FOOT_NAMES

    from .terrain import score_terrain_attempt

    path = Path(run)
    path = path / "report.json" if path.is_dir() else path
    report = json.loads(path.read_text())
    task = report.get("config", {}).get("task", {})
    world = report.get("connected_world", {})
    steps = report.get("control_steps")

    def require(condition, message):
        if not condition:
            raise ValueError(f"{path}: {message}")

    require(
        task.get("terrain") == "connected"
        and task.get("num_envs") == 1
        and world.get("first_attempt_only") is True
        and world.get("foot_names") == list(FOOT_NAMES),
        "require a single connected-world attempt with named feet",
    )
    require(
        report.get("status")
        in {"EVALUATION_COMPLETE_NOT_QUALIFIED", "FIRST_ATTEMPT_ENDED_NOT_QUALIFIED"}
        and report.get("policy_unchanged") is True
        and report.get("cleanup")
        == {"environment": "complete", "application": "complete"}
        and type(steps) is int
        and 0 < steps <= 30000
        and report.get("environment_transitions") == steps,
        "require recorded steps, frozen policy and complete native cleanup",
    )
    actor = report.get("actor_sha256", "")
    require(
        isinstance(actor, str)
        and len(actor) == 64
        and all(c in "0123456789abcdef" for c in actor)
        and isinstance(report.get("package_sources"), dict)
        and bool(report["package_sources"]),
        "missing actor or executed-source identity",
    )
    for name, digest in {
        "manifest.json": report.get("manifest_sha256"),
        **{
            name: report.get("evidence_sha256", {}).get(name)
            for name in ("commands.json", "motion.npz", "tracking.npz", "world.npz")
        },
    }.items():
        evidence = path.parent / name
        require(
            evidence.is_file() and file_sha256(evidence) == digest,
            f"missing or changed evidence: {name}",
        )

    tape = validate_tape(
        json.loads((path.parent / "commands.json").read_text()), require_complete=False
    )
    require(
        tape["steps"] == steps and tape["error"] is None, "inconsistent consumed tape"
    )
    commands = np.asarray(
        [
            segment["command"]
            for segment in tape["segments"]
            for _ in range(segment["start_step"], segment["end_step_exclusive"])
        ]
    )
    with np.load(path.parent / "motion.npz", allow_pickle=False) as archive:
        motion = dict(archive)
    require(
        all(np.isfinite(value).all() for value in motion.values()),
        "nonfinite motion evidence",
    )
    result = score_terrain_attempt(motion, world["source"])
    require(
        motion["command"].shape == (steps, 1, 3)
        and np.allclose(motion["command"][:, 0], commands, rtol=0, atol=1e-6),
        "motion and consumed tape disagree",
    )
    ended = motion["terminated"] | motion["truncated"]
    endings = np.flatnonzero(ended[:, 0])
    require(
        not len(endings) or endings.tolist() == [steps - 1],
        "trace continues after its first ending",
    )
    first_ended = report["status"] == "FIRST_ATTEMPT_ENDED_NOT_QUALIFIED"
    requested = report.get("requested_control_steps")
    require(
        type(requested) is int
        and steps <= requested <= 30000
        and first_ended == bool(len(endings))
        and tape["complete"] == (not first_ended)
        and (first_ended or steps == requested)
        and report.get("terminated_rows") == int(motion["terminated"].sum())
        and report.get("timeout_rows")
        == int((motion["truncated"] & ~motion["terminated"]).sum()),
        "inconsistent first-ending, completion or requested-step receipts",
    )
    motor = report.get("motor_delivery", {})
    require(
        all(
            motor.get(key) == steps
            for key in (
                "encoded_steps",
                "verified_delivery_steps",
                "native_step_returns",
            )
        )
        and motor.get("faulted") is False
        and motor.get("pending_delivery") is False
        and motor.get("excluded_terminal_rows") == len(endings)
        and motor.get("native_verified_rows") == steps - len(endings),
        "incomplete or invalid native motor delivery",
    )
    with np.load(path.parent / "tracking.npz", allow_pickle=False) as archive:
        shapes = {
            "root_com_velocity": (steps, 1, 3),
            "causal_sensor_noise": (steps, 1, 30),
            "initial_raw_sensors": (1, 30),
            "initial_noisy_sensors": (1, 30),
            "foot_contacts": (steps, 1, 4),
            "foot_net_forces_w": (steps, 1, 4, 3),
            "reset_mask": (steps, 1),
        }
        require(
            all(
                name in archive and archive[name].shape == shape
                for name, shape in shapes.items()
            ),
            "incomplete tracking/input evidence",
        )
        reset = archive["reset_mask"]
        require(
            reset.shape == (steps, 1)
            and reset.dtype == np.bool_
            and reset[0, 0]
            and not reset[1:].any(),
            "missing initial reset or replacement attempt",
        )
        require(
            all(
                value.shape[0] == steps
                for name, value in archive.items()
                if not name.startswith("initial_")
            ),
            "incomplete tracking trace",
        )
        require(
            all(np.isfinite(value).all() for value in archive.values()),
            "nonfinite tracking evidence",
        )
        tracking = dict(archive)

    identity = {}
    if (
        task.get("terrain_group") is not None
        or task.get("terrain_attempt_index") is not None
        or report.get("terrain_assignment") is not None
    ):
        try:
            with np.load(path.parent / "world.npz", allow_pickle=False) as archive:
                mesh = dict(archive)
            manifest = json.loads((path.parent / "manifest.json").read_text())
            identity = _verify_terrain_binding(report, manifest, mesh, motion, tracking)
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError(
                f"{path}: incomplete terrain development binding"
            ) from error

    # The short diagnostic can share a valid prefix, but cannot become a 30 s trial.
    prescribed = requested == 1500
    failures = list(result["failures"])
    if not prescribed:
        failures.append("requested tape is not the prescribed 30 s terrain trial")
    return {
        **result,
        "diagnostic_passed": result["kinematic_passed"] and prescribed,
        "failures": failures,
        "requested_control_steps": requested,
        "prescribed_duration": prescribed,
        "evidence_verified": True,
        "report": str(path.resolve()),
        "actor_sha256": actor,
        "executed_package_sources": report["package_sources"],
        **identity,
        "qualified": False,
        "qualification_eligible": False,
        "scope": "Single-attempt recorded kinematics and archive integrity only; not layout/dynamics-bank validation, cooked-contact certification or qualification.",
    }


def _verify_terrain_binding(report, manifest, mesh, motion, tracking):
    """Reconstruct the assigned ID and consumed draws without rebuilding terrain."""
    from parkour_lab.environments.randomization import (
        TaskRandomization,
        terrain_development_attempt,
    )

    from .flat import _same_values

    task = report["config"]["task"]
    assignment = terrain_development_attempt(
        task["terrain_group"], task["terrain_attempt_index"]
    )
    world = report["connected_world"]
    source, native = world["source"], world["native"]

    def require(condition, message):
        if not condition:
            raise ValueError(f"Terrain development binding: {message}")

    require(report["terrain_assignment"] == assignment, "selected assignment changed")
    require(
        task["world"] == assignment["world"]
        and task["dynamics"] == assignment["stratum"],
        "configured world or stratum changed",
    )
    require(
        native["requested_world"] == assignment["world"], "native world request changed"
    )
    require(report["requested_control_steps"] == 1500, "requested tape changed")
    phases = assignment["command_phases"]
    commands = np.asarray(
        [
            command
            for duration, command in phases
            for _ in range(round(duration * assignment["control_hz"]))
        ],
        dtype=np.float32,
    )
    steps = report["control_steps"]
    require(
        np.array_equal(motion["command"][:, 0], commands[:steps]),
        "consumed tape changed",
    )
    requested = assignment["world"]
    family, direction, _ = task["terrain_group"].split("/")
    local_heading = (
        (-np.pi / 2 if direction == "down" else np.pi / 2)
        if family == "ramp"
        else (np.pi if direction == "down" else 0.0)
    )
    heading = local_heading + requested["world_yaw"]
    require(
        source["coarse_seed"] == requested["coarse_seed"]
        and source["fine_seed"] == requested["fine_seed"]
        and source["size_m"] == requested["size"]
        and source["resolution_limit_m"] == requested["resolution"]
        and source["trial"]["family"] == family
        and source["trial"]["orientation"] == direction
        and abs(source["trial"]["heading_world_rad"] - heading) <= 1e-12
        and abs(native["start_heading_rad"] - heading) <= 1e-12,
        "source layout, roughness seeds or supported heading changed",
    )
    for key, name, dtype in (
        ("vertices_float32_sha256", "vertices", "<f4"),
        ("faces_int64_sha256", "faces", "<i8"),
    ):
        digest = hashlib.sha256(mesh[name].astype(dtype).tobytes()).hexdigest()
        require(
            digest == source[key] == native[key], "source/native mesh identity changed"
        )
    require(
        source["trial"]["local_to_world_column_transform"]
        == native["local_to_world_column_transform"]
        and native["material"]
        == dict(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        "source/native rigid transform or material changed",
    )
    sampler = TaskRandomization(task, evaluation=True, nominal_heading=heading)
    sampler.dynamics()
    sampler.sample_start(None)
    expected = sampler.manifest()
    tolerance = 32 * np.finfo(np.float64).eps
    require(
        _same_values(manifest, expected, tolerance),
        "global ID, streams, first start or dynamics changed",
    )
    require(
        _same_values(report["task_realization"]["randomization"], expected, tolerance),
        "applied randomization differs",
    )
    _verify_saved_dynamics(
        report["task_realization"],
        report["motor_verification"],
        expected["attempts"][0]["dynamics"],
    )
    noise = sampler.uniform(None, steps * 30, "observation-noise", -1.0, 1.0).reshape(
        steps, 1, 30
    )
    noise = (noise * ([0.2] * 3 + [0.05] * 3 + [0.01] * 12 + [1.5] * 12)).astype(
        np.float32
    )
    require(
        report["sensor_noise"]["generated_frames"] == steps
        and report["sensor_noise"]["draw_sha256"]
        == hashlib.sha256(noise.tobytes()).hexdigest()
        == hashlib.sha256(tracking["causal_sensor_noise"].tobytes()).hexdigest(),
        "consumed causal noise or receipt changed",
    )
    start = expected["attempts"][0]["start"]
    initial = report["initial_state"]
    quaternion = np.array(
        [[np.cos(start["start_yaw"] / 2), 0, 0, np.sin(start["start_yaw"] / 2)]]
    )
    require(
        np.allclose(
            motion["initial_position_w"], initial["root_position"], rtol=0, atol=2e-6
        )
        and np.allclose(
            motion["initial_position_w"][0, :2],
            np.asarray(native["start_position_m"][:2]) + start["start_xy"],
            rtol=0,
            atol=2e-6,
        )
        and np.allclose(
            tracking["initial_raw_sensors"][0, 6:18],
            start["joint_offset"],
            rtol=0,
            atol=2e-6,
        )
        and np.allclose(
            tracking["initial_noisy_sensors"],
            tracking["initial_raw_sensors"] + noise[0],
            rtol=0,
            atol=2e-6,
        ),
        "initial native state or causal inputs differ from the assigned start",
    )
    require(
        np.allclose(
            motion["initial_quaternion_w"],
            initial["root_orientation"],
            rtol=0,
            atol=2e-6,
        )
        and min(
            np.max(np.abs(motion["initial_quaternion_w"] - quaternion)),
            np.max(np.abs(motion["initial_quaternion_w"] + quaternion)),
        )
        <= 2e-6
        and not np.any(initial["root_velocity"])
        and not np.any(initial["joint_velocity"])
        and not np.any(motion["initial_linear_velocity_b"])
        and not np.any(motion["initial_angular_velocity_b"]),
        "initial attitude or zero-velocity state changed",
    )
    return {
        "terrain_development": {
            "namespace": assignment["namespace"],
            "group_id": assignment["group_id"],
            "attempt_index": assignment["attempt_index"],
            "stratum": assignment["stratum"],
            "binding_verified": True,
            "scope": "Selected development ID and archived source/native identity, reset/dynamics declarations and consumed causal draws; not bank qualification or cooked-contact certification",
        }
    }


def _verify_saved_dynamics(realized, verification, draw):
    """Cross-check archived physical readbacks and joint-wise sampled motor scales."""

    def require(condition, message):
        if not condition:
            raise ValueError(f"Terrain development dynamics: {message}")

    bodies, joints = realized["body_names"], realized["joint_names"]
    require(
        bodies.count("base") == 1 and len(set(joints)) == len(joints) == 12,
        "invalid body/joint identities",
    )
    requested, actual = realized["requested_physics"], realized["physical_readback"]
    for key in ("masses", "inertias", "coms", "materials", "joint_limits"):
        expected, value = np.asarray(requested[key]), np.asarray(actual[key])
        require(
            expected.shape == value.shape
            and expected.size > 0
            and np.isfinite(value).all(),
            "missing or invalid physical readback",
        )
        if key == "coms":
            expected, value = expected[..., :3], value[..., :3]
        require(
            (
                np.allclose(value, expected, rtol=1e-6, atol=1e-7)
                if key in ("masses", "inertias")
                else np.array_equal(value, expected)
            ),
            f"{key} readback differs from requested physics",
        )
    masses = np.asarray(realized["default_masses"], dtype=np.float32)
    require(
        masses.shape == np.asarray(requested["masses"]).shape,
        "default mass baseline differs from the physical readback shape",
    )
    masses[0, bodies.index("base")] += np.float32(draw["added_mass"])
    require(
        np.allclose(requested["masses"], masses, rtol=1e-6, atol=1e-7),
        "assigned added mass was not applied",
    )
    materials = np.asarray(actual["materials"])
    require(
        materials.ndim == 3
        and materials.shape[0] == 1
        and materials.shape[2] == 3
        and np.all(materials[..., 0] == np.float32(draw["static_friction"]))
        and np.all(materials[..., 1] == np.float32(draw["dynamic_friction"]))
        and not np.any(materials[..., 2]),
        "assigned friction/restitution was not applied",
    )
    require(
        all(
            mode in ("average", "min", "multiply")
            for mode in realized["robot_material_combine_modes"].values()
        ),
        "robot material overrides terrain multiply",
    )
    motors = realized["motors"]
    scales = verification["realized_randomization"]
    require(
        bool(motors)
        and set(motors) == set(scales)
        and verification["runtime_num_envs"] == 1,
        "missing motor realization",
    )
    covered = []
    for name, motor in motors.items():
        covered.extend(motor["joint_names"])
        indices = [joints.index(joint) for joint in motor["joint_names"]]
        receipt = scales[name]
        for parameter, key in (
            ("stiffness", "kp_scale"),
            ("damping", "kd_scale"),
            ("motor_strength", "motor_strength"),
        ):
            scale = np.asarray(draw[key], dtype=np.float32)[None, indices]
            require(
                np.array_equal(receipt["scales"][parameter], scale),
                "motor scale differs from the assigned joint draw",
            )
            value = (
                scale
                if parameter == "motor_strength"
                else scale
                * np.asarray(receipt["nominal_gains"][parameter], dtype=np.float32)
            )
            require(
                np.array_equal(motor[parameter], value),
                "applied PD gain/strength differs from assigned scale",
            )
    require(sorted(covered) == sorted(joints), "motors do not cover each joint once")
