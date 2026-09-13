"""Finite, headless evaluation of the stock Go2 motor checkpoint.

Uses installed Isaac Lab packages, not its separately distributed training/play
scripts. The default flat screen is unchanged. Opt-in mesh/course transfer reuses
production geometry and support gates, never the incompatible legacy RMA motor.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import yaml

try:
    from .operator_benchmark_core import (
        DT,
        OBSERVATION_TERMS,
        STEPS,
        command_schedule,
        config_differences,
        file_sha256,
        load_reference_actor,
        read_yaml_data,
        score_trace,
        score_course_trace,
    )
    from .operator_profiles import (
        PROFILES,
        STOCK_FUNCTIONS,
        STATIONARY_FUNCTIONS,
        apply_reward_profile,
        environment_profile,
    )
except ImportError:
    from operator_benchmark_core import (
        DT,
        OBSERVATION_TERMS,
        STEPS,
        command_schedule,
        config_differences,
        file_sha256,
        load_reference_actor,
        read_yaml_data,
        score_trace,
        score_course_trace,
    )
    from operator_profiles import (
        PROFILES,
        STOCK_FUNCTIONS,
        STATIONARY_FUNCTIONS,
        apply_reward_profile,
        environment_profile,
    )


def make_recorder_cfg():
    """Use Isaac Lab's post-step/pre-reset hook; no environment monkeypatching."""
    from isaaclab.managers import (
        DatasetExportMode,
        RecorderManagerBaseCfg,
        RecorderTerm,
        RecorderTermCfg,
    )
    from isaaclab.utils import configclass

    class OperatorCapture(RecorderTerm):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self.enabled = False
            self.samples = []
            self.control_trace = None
            self.motor_parity = False
            self.course = None
            env.operator_capture = self

        def record_pre_step(self):
            if self.enabled:
                self.command = self._env.command_manager.get_command(
                    "base_velocity"
                ).clone()
            return None, None

        def record_post_step(self):
            if self.enabled:
                robot = self._env.scene["robot"].data
                # Copies are essential: the simulator reuses/reset-writes buffers.
                sample = {
                    "command": self.command,
                    "position": robot.root_pos_w,
                    "quaternion": robot.root_quat_w,
                    "linear_velocity_b": robot.root_lin_vel_b,
                    "angular_velocity_b": robot.root_ang_vel_b,
                    "angular_velocity_w": robot.root_ang_vel_w,
                    "terminated": self._env.reset_terminated,
                    "time_out": self._env.reset_time_outs,
                }
                if self.motor_parity:
                    sample.update(
                        observation=self.observation,
                        action=self._env.action_manager.action,
                        joint_target=robot.joint_pos_target,
                        default_joint_position=robot.default_joint_pos,
                    )
                if self.course is not None:
                    route_state = self._env._parkour_runtime.route
                    params = self._env.termination_manager.get_term_cfg(
                        "course_success"
                    ).params
                    feet = params["feet_asset_cfg"].body_ids
                    contacts = self._env.scene["contact_forces"].data
                    sample.update(
                        linear_velocity_w=robot.root_lin_vel_w,
                        waypoint_index=route_state.active_waypoint_indices,
                        terminal_predicates=route_state.terminal_landing_predicates,
                        terminal_dwell_s=route_state.terminal_landing_stable_time_s,
                        feet_position=robot.body_pos_w[:, feet],
                        feet_force=contacts.net_forces_w[
                            :, params["feet_contact_cfg"].body_ids
                        ],
                        contact_force_history=contacts.net_forces_w_history,
                        base_height_ray=self._env.scene[
                            "base_height_scanner"
                        ].data.ray_hits_w[:, 0],
                    )
                    for name in (
                        "base_contact",
                        "course_chassis",
                        "course_fall",
                        "course_off_route",
                        "course_success",
                    ):
                        sample[name] = self._env.termination_manager.get_term(name)
                self.samples.append(
                    {
                        k: v.detach().to("cpu", copy=True).numpy()
                        for k, v in sample.items()
                    }
                )
                if self.control_trace is not None:
                    self.control_trace.after_step()
            # EXPORT_NONE + no return data avoids a parallel HDF5 recording.
            return None, None

        def finish(self):
            if not self.samples:
                raise RuntimeError("No terminal-safe samples captured")
            return {
                key: np.stack([sample[key] for sample in self.samples])
                for key in self.samples[0]
            }

    @configclass
    class CaptureCfg(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        export_in_record_pre_reset = False
        export_in_close = False
        operator = RecorderTermCfg(class_type=OperatorCapture)

    return CaptureCfg()


def command_observation(env, desired):
    """Publish a complete command before constructing this action's observation."""
    import torch

    command = env.command_manager.get_term("base_velocity")
    command.time_left.fill_(float("inf"))
    command.is_standing_env.fill_(False)
    command.vel_command_b.copy_(desired)
    observation = env.observation_manager.compute()["policy"]
    if (
        observation.shape != (desired.shape[0], 48)
        or not torch.isfinite(observation).all()
    ):
        raise RuntimeError("Invalid policy observation")
    if not torch.allclose(observation[:, 9:12], desired, atol=1e-6, rtol=0):
        raise RuntimeError("Policy did not receive the requested body-twist command")
    return observation


def reference_config(saved):
    """Validate the installed physical/motor contract before any task overrides."""
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )

    cfg = UnitreeGo2FlatEnvCfg()
    profile = environment_profile(saved)
    if (
        cfg.rewards.track_ang_vel_z_exp.params.get("std")
        != PROFILES["stock"].yaw_tracking_std
    ):
        raise ValueError(
            "Installed stock yaw-tracking kernel differs from the known reference"
        )
    if profile.stationary_precision:
        for name, expected in STOCK_FUNCTIONS.items():
            term = getattr(cfg.rewards, name)
            function = term.func
            identity = (
                f"{function.__module__}:{function.__name__}"
                if callable(function)
                else function
            )
            if identity != expected or term.params.get("std") != 0.5:
                raise ValueError(
                    "Installed stock tracking contract differs before stationary override"
                )
    # Reconstruct only a known reward variant. Keep the FULL comparison
    # below: do not ignore rewards or trust arbitrary saved function names.
    apply_reward_profile(cfg, profile)
    # No function from the archived YAML is executed. Its complete relevant
    # contract is compared to this installed, known stock environment instead.
    current = yaml.load(
        yaml.dump(cfg.to_dict(), sort_keys=False), Loader=yaml.BaseLoader
    )
    if profile.stationary_precision:
        # The CLI and `python -m` import the same two repository functions under
        # different package prefixes. Only these explicit identities are aliases.
        for name, function in STATIONARY_FUNCTIONS.items():
            aliases = (
                f"operator_rewards:{function}",
                f"scripts.rsl_rl.operator_rewards:{function}",
            )
            if current["rewards"][name]["func"] in aliases:
                current["rewards"][name]["func"] = saved["rewards"][name]["func"]
    differences = config_differences(saved, current)
    if differences:
        raise ValueError(
            "Installed stock environment differs from training: "
            + ", ".join(differences)
        )
    return cfg


def prepare_config(saved, *, seed, num_envs, device):
    cfg = reference_config(saved)
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 0.0
    command.ranges.heading = None
    command.ranges.lin_vel_x = (-0.3, 0.7)
    command.ranges.lin_vel_y = (-0.2, 0.2)
    command.ranges.ang_vel_z = (-0.8, 0.8)
    command.debug_vis = False
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.device = device
    cfg.observations.policy.enable_corruption = False
    # Retain the original mass/friction/reset events and physical terminations.
    # A guard step beyond the benchmark makes any timeout within it unexpected.
    cfg.episode_length_s = (STEPS + 1) * DT
    cfg.recorders = make_recorder_cfg()
    return cfg


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


COURSE_FAMILIES = ("tilted_ramps", "high_step", "gap", "hurdle")


def configure_terrain(cfg, terrain, level):
    """Overlay only evaluation geometry/route instrumentation on validated stock."""
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    cfg.scene.terrain.terrain_type = (
        "generator"  # Replaces the plane; never adds one beneath gaps.
    )
    if terrain == "mesh-flat":
        from isaaclab.terrains import MeshPlaneTerrainCfg, TerrainGeneratorCfg

        # Preserve grid origins already during scene construction, not after
        # overlapping articulations or terrain-origin RNG draws were created.
        cfg.scene.terrain.use_terrain_origins = False
        extent = max(
            64.0, 2 * np.ceil(np.sqrt(cfg.scene.num_envs)) * cfg.scene.env_spacing
        )
        cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            seed=cfg.seed,
            size=(extent, extent),
            num_rows=1,
            num_cols=1,
            sub_terrains={"flat": MeshPlaneTerrainCfg(proportion=1.0)},
            use_cache=False,
            border_width=0.0,
        )
        return None

    from parkour_lab.tasks.manager_based.parkour_lab.parkour_lab_env_cfg import (
        ParkourLabEnvCfg,
    )
    from parkour_lab.tasks.manager_based.parkour_lab import mdp

    source_file = Path(sys.modules[ParkourLabEnvCfg.__module__].__file__).resolve()
    expected_source = (
        Path(__file__).resolve().parents[2]
        / "source/parkour_lab/parkour_lab/tasks/manager_based/parkour_lab/parkour_lab_env_cfg.py"
    )
    if source_file != expected_source:
        raise ValueError("Course package must use this repository's reviewed source")
    course_cfg = ParkourLabEnvCfg()
    course_cfg.scene.num_envs = cfg.scene.num_envs
    course_cfg.configure_evaluation(terrain, level, seed=cfg.seed, geometry_variant=0)
    cfg.scene.terrain.terrain_generator = course_cfg.scene.ground.terrain_generator
    cfg.scene.terrain.use_terrain_origins = True
    cfg.scene.terrain.max_init_terrain_level = level
    # Keep stock robot, actions, observations, gains, randomizers and BOTH
    # materials. Copy no legacy intent command, reward or active-motion timer.
    cfg.scene.waypoint_marker = course_cfg.scene.waypoint_marker
    cfg.scene.base_height_scanner = course_cfg.scene.base_height_scanner
    cfg.scene.base_height_scanner.mesh_prim_paths = [cfg.scene.terrain.prim_path]
    cfg.events.initialize_terrain_levels = course_cfg.events.initialize_terrain_levels
    cfg.events.reset_routes = course_cfg.events.reset_routes
    # Explicit course-aligned root jitter; preserve the stock joint reset.
    cfg.events.reset_base.params["pose_range"] = {
        k: (-0.05, 0.05) for k in ("x", "y", "yaw")
    }
    cfg.events.reset_base.params["velocity_range"] = {
        k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.curriculum = None
    cfg.terminations.course_off_route = course_cfg.terminations.off_route
    cfg.terminations.course_fall = TerminationTermCfg(func=mdp.fell_below_course)
    chassis = SceneEntityCfg("contact_forces", body_names="base|Head_.*")
    cfg.terminations.course_chassis = TerminationTermCfg(
        func=mdp.chassis_contact_done, params={"sensor_cfg": chassis, "threshold": 1.0}
    )
    cfg.terminations.course_success = course_cfg.terminations.success
    params = cfg.terminations.course_success.params
    params["feet_contact_cfg"].name = "contact_forces"
    params["chassis_contact_cfg"] = copy.deepcopy(chassis)
    params["require_stable_finish"] = True
    return course_cfg.evaluation_course_metadata()


def mesh_identity(env):
    """Fail closed if the requested ground is not the sole collidable triangle mesh."""
    from pxr import Usd, UsdPhysics

    root = env.scene.stage.GetPrimAtPath(env.cfg.scene.terrain.prim_path)
    colliders = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
    if len(colliders) != 1 or colliders[0].GetTypeName() != "Mesh":
        raise ValueError("Expected one generated ground mesh, with no underlying plane")
    prim = colliders[0]
    if not prim.GetAttribute("physics:collisionEnabled").Get() or prim.GetAttribute(
        "physics:approximation"
    ).Get() not in (None, "none"):
        raise ValueError("Terrain must collide as triangles, not a convex hull")
    points = np.asarray(prim.GetAttribute("points").Get(), dtype=np.float32)
    faces = np.asarray(prim.GetAttribute("faceVertexIndices").Get(), dtype=np.int32)
    if points.size == 0 or faces.size == 0 or not np.isfinite(points).all():
        raise ValueError("Empty or invalid generated mesh")
    return {
        "prim_path": str(prim.GetPath()),
        "collision_enabled": True,
        "vertices": len(points),
        "indices": len(faces),
        "sha256": hashlib.sha256(points.tobytes() + faces.tobytes()).hexdigest(),
    }


def course_command(env, course, step, finished):
    """Explicit privileged waypoint guidance; no policy/action replacement."""
    import torch
    from isaaclab.managers import SceneEntityCfg
    from parkour_lab.tasks.manager_based.parkour_lab.mdp.navigation import route

    target = route.active_waypoint_positions(env, SceneEntityCfg("waypoint_marker"))
    delta = (
        target[:, :2]
        - (env.scene["robot"].data.root_pos_w - env.scene.env_origins)[:, :2]
    )
    quat = env.scene["robot"].data.root_quat_w
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    error = torch.atan2(delta[:, 1], delta[:, 0]) - yaw
    error = torch.atan2(torch.sin(error), torch.cos(error))
    distance = torch.linalg.vector_norm(delta, dim=-1)
    final = route.active_waypoint_is_final(env)
    speed = torch.full_like(distance, course["course"]["target_speed"])
    speed = torch.where(final, torch.minimum(speed, distance), speed)
    command = torch.stack(
        (
            speed * torch.cos(error).clamp_min(0),
            torch.zeros_like(speed),
            (1.5 * error).clamp(-0.8, 0.8),
        ),
        dim=-1,
    )
    stop = finished | (
        final & (distance < 0.75 * route.active_waypoint_root_reach_radii(env))
    )
    command[stop | (step < round(2.0 / DT))] = 0.0
    return command


def validate_mesh_flat_report(path, checkpoint):
    """Require an actual matched mesh-flat PASS before measuring course transfer."""
    report = json.loads(path.read_text())
    if report.get("terrain") != "mesh-flat" or report.get("seed") != 43:
        raise ValueError("Expected the predeclared seed-43 mesh-flat control")
    from dataclasses import asdict

    try:
        from .operator_benchmark_core import Thresholds
    except ImportError:
        from operator_benchmark_core import Thresholds
    if report.get("thresholds") != asdict(Thresholds()):
        raise ValueError("Mesh-flat gates changed")
    expected = {
        "checkpoint": checkpoint,
        "env.yaml": checkpoint.parent / "params/env.yaml",
        "agent.yaml": checkpoint.parent / "params/agent.yaml",
        "benchmark": Path(__file__),
        "scoring": Path(__file__).with_name("operator_benchmark_core.py"),
    }
    if any(
        report.get("provenance", {}).get("sha256", {}).get(k) != file_sha256(p)
        for k, p in expected.items()
    ):
        raise ValueError("Mesh-flat checkpoint/configuration/implementation mismatch")
    with np.load(path.with_name("trace.npz"), allow_pickle=False) as archive:
        raw = dict(archive)
    replay = score_trace(raw, command_schedule(10)[0])
    if (
        report.get("status") != "PASS"
        or replay["status"] != "PASS"
        or report.get("worker") != {"returncode": 0, "timed_out": False}
    ):
        raise ValueError("Mesh-flat control did not pass cleanly")
    validate_motor_trace(raw)
    audit = report.get("student_interface_audit", {})
    if any(
        audit.get(k) != v
        for k, v in {
            "status": "ORACLE_PARITY_PASS",
            "control_steps": STEPS,
            "action_comparisons": STEPS * 100,
            "cold_reset_frames": 100,
            "later_reset_frames": 0,
            "exact_action_equality": True,
            "student_status": "UNTRAINED_NOT_RUN",
        }.items()
    ):
        raise ValueError("Mesh-flat control lacks complete oracle parity evidence")
    try:
        from .operator_student_bridge import interface_manifest
    except ImportError:
        from operator_student_bridge import interface_manifest
    interface = json.loads(path.with_name("student_interface.json").read_text())
    expected_interface = interface_manifest(
        teacher_sha256=file_sha256(checkpoint),
        env_sha256=file_sha256(checkpoint.parent / "params/env.yaml"),
        joint_names=report["runtime"]["joint_names"],
    )
    if (
        interface != expected_interface
        or report.get("simulation_steps") != STEPS
        or report.get("mesh", {}).get("collision_enabled") is not True
    ):
        raise ValueError("Mesh-flat motor/geometry interface evidence mismatch")
    return {
        "report": str(path),
        "sha256": file_sha256(path),
        "packages": report["provenance"]["packages"],
        "joint_names": report["runtime"]["joint_names"],
    }


def validate_motor_trace(trace):
    """Check delivered stock joint targets on every pre-reset transition."""
    action = trace["action"]
    target = trace["joint_target"]
    expected = trace["default_joint_position"] + 0.25 * action
    observation = trace["observation"]
    previous = np.zeros_like(action)
    previous[1:] = action[:-1]
    previous[1:][trace["terminated"][:-1] | trace["time_out"][:-1]] = 0
    if (
        observation.shape != (*action.shape[:2], 48)
        or not np.isfinite(observation).all()
        or not np.allclose(observation[..., 9:12], trace["command"], atol=1e-6, rtol=0)
        or not np.allclose(observation[..., 36:48], previous, atol=1e-6, rtol=0)
    ):
        raise ValueError("Delivered observation/command/previous-action parity failed")
    if (
        action.shape != (*trace["terminated"].shape, 12)
        or target.shape != action.shape
        or trace["default_joint_position"].shape != action.shape
        or not np.isfinite(action).all()
        or not np.allclose(target, expected, rtol=0, atol=1e-6)
    ):
        raise ValueError("Stock joint-target parity failed")
    return {
        "status": "PASS",
        "joint_target_comparisons": int(action.size),
        "max_joint_target_error_rad": float(np.abs(target - expected).max()),
    }


def run_benchmark(args, output, agent, saved, *, stop_probe=None):
    # The separate diagnostic CLI owns mixed-controller experiments. The normal
    # benchmark never constructs a probe and retains its original action path.
    if stop_probe is not None and (
        getattr(args, "audit_student_interface", False)
        or getattr(args, "diagnostic_reference", None) is None
    ):
        raise ValueError(
            "Stop probes require control capture, not an oracle parity audit"
        )
    from isaaclab.app import AppLauncher

    # Streaming cannot be enabled accidentally by the LIVESTREAM environment var.
    launcher = AppLauncher(headless=True, livestream=0, device=args.device)
    app = launcher.app
    env = None
    capture = None
    initial = None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv

        actor, iteration = load_reference_actor(args.checkpoint, agent)
        diagnostic_reference = getattr(args, "diagnostic_reference", None)
        reference_actor = None
        if diagnostic_reference is not None:
            try:
                from .operator_control_trace import (
                    OperatorControlTrace,
                    load_diagnostic_reference,
                    summarize_control_trace,
                )
            except ImportError:
                from operator_control_trace import (
                    OperatorControlTrace,
                    load_diagnostic_reference,
                    summarize_control_trace,
                )
            reference_actor, reference_identity = load_diagnostic_reference(
                args.checkpoint, diagnostic_reference
            )
        labels, schedule_np = command_schedule(args.repetitions)
        terrain = getattr(args, "terrain", "plane")
        course_mode = terrain in COURSE_FAMILIES
        if course_mode:
            labels = [terrain] * args.repetitions
        cfg = prepare_config(
            saved, seed=args.seed, num_envs=len(labels), device=args.device
        )
        course = (
            configure_terrain(cfg, terrain, args.level) if terrain != "plane" else None
        )
        env = ManagerBasedRLEnv(cfg=cfg)
        mesh = mesh_identity(env) if terrain != "plane" else None
        (output / "resolved_env.yaml").write_text(
            yaml.dump(cfg.to_dict(), sort_keys=False)
        )
        if abs(env.step_dt - DT) > 1e-9:
            raise ValueError("Benchmark requires the trained 50-Hz action interface")
        if tuple(env.observation_manager.active_terms["policy"]) != OBSERVATION_TERMS:
            raise ValueError(
                "Runtime observation order differs from the 48-D stock actor"
            )
        if env.action_manager.total_action_dim != 12:
            raise ValueError("Runtime action dimension differs from the stock motor")
        actor.to(env.device)
        schedule = torch.as_tensor(schedule_np, device=env.device)
        if (
            course_mode
            and list(env.scene["robot"].joint_names)
            != args.mesh_flat_evidence["joint_names"]
        ):
            raise ValueError("Course joint order differs from the mesh-flat motor")
        env.reset(seed=args.seed)
        audit = None
        if getattr(args, "audit_student_interface", False) or terrain != "plane":
            try:
                from .operator_student_bridge import (
                    OperatorOracleAudit,
                    interface_manifest,
                )
            except ImportError:
                from operator_student_bridge import (
                    OperatorOracleAudit,
                    interface_manifest,
                )
            audit = OperatorOracleAudit(actor)
            write_json(
                output / "student_interface.json",
                interface_manifest(
                    teacher_sha256=file_sha256(args.checkpoint),
                    env_sha256=file_sha256(args.checkpoint.parent / "params/env.yaml"),
                    joint_names=list(env.scene["robot"].joint_names),
                ),
            )
            reset_mask = torch.ones(len(labels), dtype=torch.bool, device=env.device)
        capture = env.operator_capture
        capture.motor_parity = terrain != "plane"
        capture.course = course
        if course_mode:
            from parkour_lab.tasks.manager_based.parkour_lab.mdp.navigation import route

            if not torch.all(route.active_difficulty_indices(env) == args.level):
                raise ValueError(
                    "Generated terrain row does not match requested course"
                )
            course["env_origins"] = env.scene.env_origins.detach().cpu().tolist()
            course["terrain_columns"] = (
                env.scene.terrain.terrain_types.detach().cpu().tolist()
            )
            course["course_indices"] = (
                route.active_course_indices(env).detach().cpu().tolist()
            )
            course["guidance"] = (
                "Privileged waypoint bearing, yaw gain 1.5 clipped +/-0.8 rad/s; nominal course speed times nonnegative heading cosine, terminal speed capped by distance; stop within 75% final radius; initial 2s stand."
            )
            course["reset_override"] = (
                "Root x/y/yaw +/-0.05 m/m/rad; zero initial root twist; stock joint reset and physical randomizers retained."
            )
            course["require_stable_finish"] = True
            course["contact_body_names"] = list(env.scene["contact_forces"].body_names)
            finished = torch.zeros(len(labels), dtype=torch.bool, device=env.device)
        diagnostics = None
        if reference_actor is not None:
            reference_actor.to(env.device)
            diagnostics = OperatorControlTrace(env, reference_actor)
            if stop_probe is not None:
                stop_probe.reference = reference_actor
                diagnostics.metadata["action"] = (
                    "executed raw mean: learner except declared reference stop windows"
                )
                diagnostics.metadata["scope"] = stop_probe.scope
            capture.control_trace = diagnostics
            write_json(
                output / "control_interface.json",
                {
                    **diagnostics.metadata,
                    "reference": reference_identity,
                    "learner_sha256": file_sha256(args.checkpoint),
                },
            )
        initial = {
            "initial_position": env.scene["robot"]
            .data.root_pos_w.detach()
            .to("cpu", copy=True)
            .numpy(),
            "initial_quaternion": env.scene["robot"]
            .data.root_quat_w.detach()
            .to("cpu", copy=True)
            .numpy(),
        }
        capture.enabled = True
        with torch.inference_mode():
            for step in range(STEPS):
                if not app.is_running():
                    raise RuntimeError(f"Simulator closed at step {step}/{STEPS}")
                # Fresh command reaches THIS action, never one control tick later.
                desired = (
                    course_command(env, course, step, finished)
                    if course_mode
                    else schedule[step]
                )
                observation = command_observation(env, desired)
                if capture.motor_parity:
                    capture.observation = observation.detach().clone()
                action = actor(observation)
                if not torch.isfinite(action).all():
                    raise RuntimeError(f"Nonfinite policy action at step {step}")
                if stop_probe is not None:
                    action = stop_probe.select(step, observation, action)
                if audit is not None:
                    audit.observe(observation, action, reset_mask)
                if diagnostics is not None:
                    diagnostics.before_step(observation, action)
                stepped = env.step(action)
                if course_mode:
                    finished |= stepped[2] | stepped[3]
                if stop_probe is not None:
                    stop_probe.observe_done(stepped[2] | stepped[3])
                if audit is not None:
                    # ManagerBasedRLEnv returns termination/timeout masks for the
                    # transition just executed, with new-episode observations.
                    reset_mask = (stepped[2] | stepped[3]).detach().clone()
        capture.enabled = False
        trace = capture.finish()
        trace.update(initial)
        np.savez_compressed(output / "trace.npz", **trace)
        result = (
            score_course_trace(trace, course)
            if course_mode
            else score_trace(trace, labels)
        )
        if terrain != "plane":
            result["terrain"] = terrain
            result["motor_interface"] = validate_motor_trace(trace)
            result["policy_acceptance"] = False
            result["mesh"] = mesh
        if course_mode:
            result["mesh_flat_control"] = args.mesh_flat_evidence
        result["checkpoint_iteration"] = iteration
        result["seed"] = args.seed
        result["simulation_steps"] = STEPS
        if diagnostics is not None:
            control = diagnostics.finish()
            np.savez_compressed(output / "control_trace.npz", **control)
            control_report = summarize_control_trace(
                control, trace, labels, diagnostics.metadata
            )
            control_report["reference"] = reference_identity
            control_report["sha256"] = {
                "learner_checkpoint": file_sha256(args.checkpoint),
                "physical_trace": file_sha256(output / "trace.npz"),
                "control_trace": file_sha256(output / "control_trace.npz"),
                "interface": file_sha256(output / "control_interface.json"),
            }
            write_json(output / "control_report.json", control_report)
            result["control_diagnostics"] = {
                "status": control_report["status"],
                "report": "control_report.json",
                "sha256": file_sha256(output / "control_report.json"),
                "reference": reference_identity,
                "scope": diagnostics.metadata["scope"],
            }
        if audit is not None:
            result["student_interface_audit"] = audit.report()
        result["runtime"] = {
            "joint_names": list(env.scene["robot"].joint_names),
            "observation_terms": list(OBSERVATION_TERMS),
            "actor_observation": "includes simulator base linear velocity; NOT RMA",
            "capture": "RecorderTerm.record_post_step, before ManagerBasedRLEnv auto-reset",
            "evaluation_changes": [
                "deterministic actions",
                "observation corruption disabled",
                "scripted body-twist commands",
                "parallel environment count",
                "evaluation seed",
                "timeout moved one step beyond 20-s benchmark",
            ],
        }
        if terrain != "plane":
            result["runtime"]["evaluation_changes"].append(
                "Plane replaced by verified collidable triangles; stock robot and materials retained"
            )
        if course_mode:
            result["runtime"]["evaluation_changes"].extend(
                [
                    course["reset_override"],
                    course["guidance"],
                    "Production course failure/ordered support gates; stricter final two-foot load and 0.2s stability dwell for every family",
                ]
            )
        if stop_probe is not None:
            result = stop_probe.publish(output, result, trace, control)
        # Kit's app.close() can terminate Python with os._exit(0). Publish before
        # cleanup; a supervising process maps the measured status to an exit code.
        write_json(output / "measurement_report.json", result)
        return result
    except Exception as error:
        result = {
            "status": "ERROR",
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        # Retain partial physical evidence, but never score it as a complete trial.
        if capture is not None and capture.samples:
            try:
                partial = capture.finish()
                if initial is not None:
                    partial.update(initial)
                np.savez_compressed(output / "partial_trace.npz", **partial)
                result["captured_steps"] = len(capture.samples)
            except Exception as capture_error:
                result["partial_capture_error"] = str(capture_error)
        write_json(output / "measurement_report.json", result)
        return result
    finally:
        for name, resource in (("environment", env), ("application", app)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    # Persist each error immediately; a later app.close can exit.
                    write_json(
                        output / f"{name}_cleanup_error.json",
                        {"error": str(error), "traceback": traceback.format_exc()},
                    )


def supervise(
    command,
    output,
    *,
    timeout_s=300,
    report_filename="measurement_report.json",
    valid_statuses=("PASS", "FAIL", "ERROR"),
):
    """Fail closed on incomplete/native-failed workers, including exit(0) in Kit."""
    try:
        process = subprocess.run(command, timeout=timeout_s, check=False)
        worker = {"returncode": process.returncode, "timed_out": False}
    except subprocess.TimeoutExpired:
        worker = {"returncode": None, "timed_out": True}
    except OSError as error:
        worker = {"returncode": None, "timed_out": False, "error": str(error)}
    write_json(output / "worker_status.json", worker)
    try:
        report = json.loads((output / report_filename).read_text())
        if not isinstance(report, dict):
            raise ValueError("Worker report must be an object")
    except (OSError, ValueError):
        report = {
            "status": "ERROR",
            "error": "Worker did not publish a complete report",
        }
    cleanup = {}
    for name in ("environment", "application"):
        path = output / f"{name}_cleanup_error.json"
        if path.exists():
            try:
                cleanup[name] = json.loads(path.read_text())
            except (OSError, ValueError):
                cleanup[name] = {"error": "Unreadable cleanup failure report"}
    if (
        worker["returncode"] != 0
        or cleanup
        or report.get("status") not in valid_statuses
    ):
        report = {
            "status": "ERROR",
            "error": "Worker exit/cleanup failed or report invalid",
            "measurement_result": report,
            "cleanup_errors": cleanup,
        }
    report["worker"] = worker
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=10,
        help="Trials per profile, all run in parallel (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
        help="Development/regression seed; use untouched seeds for new confirmation",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--terrain",
        choices=("plane", "mesh-flat", *COURSE_FAMILIES),
        default="plane",
        help="Opt-in mesh-flat control or privileged course transfer; default operator screen unchanged",
    )
    parser.add_argument(
        "--level",
        type=int,
        choices=range(7),
        help="Explicit course level; only level 6 can contribute target-obstacle evidence",
    )
    parser.add_argument(
        "--mesh-flat-report",
        type=Path,
        help="Matched seed-43 mesh-flat report.json required before course transfer",
    )
    parser.add_argument(
        "--diagnostic-reference",
        type=Path,
        help="Shadow this frozen checkpoint on the actual observations and capture actions/joints/contacts; no action override or training",
    )
    parser.add_argument(
        "--audit-student-interface",
        action="store_true",
        help="Shadow oracle/history audit on delivered observations; original actor still controls; no student rollout",
    )
    parser.add_argument("--output-parent", type=Path)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="CPU-only checkpoint/config-shape check; no simulation",
    )
    args = parser.parse_args(argv)
    reference_identity = None
    try:
        import torch

        args.checkpoint = args.checkpoint.resolve(strict=True)
        command_schedule(args.repetitions)
        course_mode = args.terrain in COURSE_FAMILIES
        if course_mode != (args.level is not None) or course_mode != (
            args.mesh_flat_report is not None
        ):
            raise ValueError(
                "Course transfer requires both --level and --mesh-flat-report; neither applies to flat screens"
            )
        if args.terrain != "plane" and args.diagnostic_reference is not None:
            raise ValueError(
                "Mesh/course transfer uses the original motor only, not control-reference diagnostics"
            )
        if course_mode and args.repetitions < 3:
            raise ValueError("Course transfer requires at least three trials")
        if args.terrain == "mesh-flat" and (args.seed != 43 or args.repetitions != 10):
            raise ValueError(
                "Mesh-flat control is predeclared at seed 43, ten repetitions"
            )
        agent_path = args.checkpoint.parent / "params/agent.yaml"
        env_path = args.checkpoint.parent / "params/env.yaml"
        agent, saved = read_yaml_data(agent_path), read_yaml_data(env_path)
        # Reject corrupt/incompatible learner tensors before creating an output
        # run or launching Kit. Construction must not consume simulation RNG.
        with torch.random.fork_rng(devices=[]):
            _, iteration = load_reference_actor(args.checkpoint, agent)
        if course_mode:
            args.mesh_flat_report = args.mesh_flat_report.resolve(strict=True)
            args.mesh_flat_evidence = validate_mesh_flat_report(
                args.mesh_flat_report, args.checkpoint
            )
        if args.diagnostic_reference is not None:
            try:
                from .operator_control_trace import load_diagnostic_reference
            except ImportError:
                from operator_control_trace import load_diagnostic_reference
            args.diagnostic_reference = args.diagnostic_reference.resolve(strict=True)
            _, reference_identity = load_diagnostic_reference(
                args.checkpoint, args.diagnostic_reference
            )
    except Exception as error:
        # Exit 1 is reserved for a measured behavioral FAIL. This covers path,
        # YAML, tensor-contract and restricted-unpickling errors, not interrupts.
        parser.error(f"Preflight failed: {type(error).__name__}: {error}")
    if args.worker_output is not None:
        try:
            run_benchmark(args, args.worker_output, agent, saved)
        except Exception as error:
            # Covers failure before AppLauncher returns an application object.
            write_json(
                args.worker_output / "measurement_report.json",
                {
                    "status": "ERROR",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        return 0  # Only the supervisor owns behavioral exit semantics.
    if args.validate_only:
        print(
            f"Checkpoint iteration {iteration}: stock 48→12 mean actor loaded. Simulator contract/behavior NOT checked."
        )
        if reference_identity is not None:
            print("Diagnostic reference validated; no capture or behavior checked.")
        return 0
    provenance = {
        "checkpoint": str(args.checkpoint),
        "sha256": {
            "checkpoint": file_sha256(args.checkpoint),
            "agent.yaml": file_sha256(agent_path),
            "env.yaml": file_sha256(env_path),
            "benchmark": file_sha256(Path(__file__)),
            "scoring": file_sha256(
                Path(__file__).with_name("operator_benchmark_core.py")
            ),
            "reward_profiles": file_sha256(
                Path(__file__).with_name("operator_profiles.py")
            ),
            "operator_rewards": file_sha256(
                Path(__file__).with_name("operator_rewards.py")
            ),
            "operator_student_bridge": file_sha256(
                Path(__file__).with_name("operator_student_bridge.py")
            ),
        },
    }
    provenance["packages"] = {}
    if reference_identity is not None:
        provenance["diagnostic_reference"] = reference_identity
        provenance["sha256"]["operator_control_trace"] = file_sha256(
            Path(__file__).with_name("operator_control_trace.py")
        )
    for package in ("isaaclab", "isaaclab_tasks", "isaacsim", "rsl-rl-lib", "torch"):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][package] = "unknown"
    if course_mode and provenance["packages"] != args.mesh_flat_evidence["packages"]:
        parser.error("Course runtime packages differ from the mesh-flat control")
    if course_mode:
        source = (
            Path(__file__).resolve().parents[2]
            / "source/parkour_lab/parkour_lab/tasks/manager_based/parkour_lab"
        )
        provenance["course_source_sha256"] = {
            str(p.relative_to(source)): file_sha256(p)
            for p in sorted(source.rglob("*.py"))
        }
    parent = args.output_parent or args.checkpoint.parent
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="operator_screen_", dir=parent))
    print(f"Operator benchmark (headless, streaming off): {output}", flush=True)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    report = supervise(
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            str(args.checkpoint),
            "--worker-output",
            str(output.resolve()),
            "--seed",
            str(args.seed),
            "--repetitions",
            str(args.repetitions),
            "--device",
            args.device,
            "--terrain",
            args.terrain,
            *(
                [
                    "--level",
                    str(args.level),
                    "--mesh-flat-report",
                    str(args.mesh_flat_report),
                ]
                if course_mode
                else []
            ),
            *(["--audit-student-interface"] if args.audit_student_interface else []),
            *(
                ["--diagnostic-reference", str(args.diagnostic_reference)]
                if reference_identity is not None
                else []
            ),
        ],
        output,
    )
    if reference_identity is not None and report.get("status") != "ERROR":
        try:
            try:
                from .operator_control_trace import validate_control_artifacts
            except ImportError:
                from operator_control_trace import validate_control_artifacts
            validate_control_artifacts(
                output, report, reference_identity, provenance["sha256"]["checkpoint"]
            )
        except (OSError, ValueError, TypeError) as error:
            report = {
                "status": "ERROR",
                "error": f"Control diagnostics invalid: {error}",
                "measurement_result": report,
                "worker": report.get("worker"),
            }
    report["provenance"] = provenance
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for name, row in report.get("profiles", {}).items():
        print(f"  {name}: {row['passed']}/{row['total']} passed")
    phase_rows = report.get("phase_summary", {}).get("phases", {})
    if phase_rows:
        print("Phase-local diagnostics (not whole-trajectory acceptance):")
        for name, row in phase_rows.items():
            if row["kinematic_passed"] != row["expected"] or row["sequence_failed"]:
                print(
                    f"  {name}: {row['kinematic_passed']}/{row['expected']} kinematic pass; "
                    f"acquisition={row['acquisition_failed']}, later tracking={row['later_tracking_failed']}, "
                    f"excursion={row['excursion_failed']}, heading={row['heading_failed']}, "
                    f"wrong sign={row['wrong_sign_failed']}, "
                    f"incomplete={row['expected'] - row['complete']}, "
                    f"sequence violations={row['sequence_failed']}"
                )
    if report["status"] == "ERROR":
        print(report.get("traceback", report.get("error")))
    print(f"{report['status']}: {output / 'report.json'}", flush=True)
    return {"PASS": 0, "FAIL": 1, "ERROR": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
