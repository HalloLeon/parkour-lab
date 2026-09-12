"""Finite, headless operator evaluation of the stock Go2 flat reference checkpoint.

Uses installed Isaac Lab packages, not its separately distributed training/play
scripts. Does not modify or train a checkpoint, or load the custom parkour task.
"""

from __future__ import annotations

import argparse
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


def run_benchmark(args, output, agent, saved):
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
        cfg = prepare_config(
            saved, seed=args.seed, num_envs=len(labels), device=args.device
        )
        env = ManagerBasedRLEnv(cfg=cfg)
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
        env.reset(seed=args.seed)
        audit = None
        if getattr(args, "audit_student_interface", False):
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
        diagnostics = None
        if reference_actor is not None:
            reference_actor.to(env.device)
            diagnostics = OperatorControlTrace(env, reference_actor)
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
                observation = command_observation(env, schedule[step])
                action = actor(observation)
                if not torch.isfinite(action).all():
                    raise RuntimeError(f"Nonfinite policy action at step {step}")
                if audit is not None:
                    audit.observe(observation, action, reset_mask)
                if diagnostics is not None:
                    diagnostics.before_step(observation, action)
                stepped = env.step(action)
                if audit is not None:
                    # ManagerBasedRLEnv returns termination/timeout masks for the
                    # transition just executed, with new-episode observations.
                    reset_mask = (stepped[2] | stepped[3]).detach().clone()
        capture.enabled = False
        trace = capture.finish()
        trace.update(initial)
        np.savez_compressed(output / "trace.npz", **trace)
        result = score_trace(trace, labels)
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
        help="Development/regression seed 43; reserve 44/45 for confirmation",
    )
    parser.add_argument("--device", default="cuda:0")
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
        agent_path = args.checkpoint.parent / "params/agent.yaml"
        env_path = args.checkpoint.parent / "params/env.yaml"
        agent, saved = read_yaml_data(agent_path), read_yaml_data(env_path)
        # Reject corrupt/incompatible learner tensors before creating an output
        # run or launching Kit. Construction must not consume simulation RNG.
        with torch.random.fork_rng(devices=[]):
            _, iteration = load_reference_actor(args.checkpoint, agent)
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
    parent = args.output_parent or args.checkpoint.parent
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="operator_screen_", dir=parent))
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
