"""Bounded command-curriculum refinement of a stock Go2 reference checkpoint.

No external Isaac Lab training script is needed. Actor, critic and action noise
are restored exactly. Command sampling and versioned reward/entropy profiles
are explicit; Adam starts fresh unless an evidence-bound retention resume is
requested. This is NOT an RMA or obstacle policy.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import yaml

try:
    from .operator_benchmark import reference_config, supervise, write_json
    from .operator_benchmark_core import (
        DT,
        OBSERVATION_TERMS,
        file_sha256,
        load_reference_checkpoint,
        read_yaml_data,
    )
    from .operator_curriculum import (
        OperatorExposureWrapper,
        VERSIONS,
        VERSION,
        curriculum_manifest,
    )
    from .operator_profiles import (
        PROFILES,
        apply_reward_profile,
        profile_manifest,
        select_profile,
        source_profile,
    )
    from .run_provenance import write_run_provenance
    from .operator_retention import (
        install_moving_retention,
        retention_manifest,
        restore_adam_state,
        validate_adam_state,
    )
except ImportError:
    from operator_benchmark import reference_config, supervise, write_json
    from operator_benchmark_core import (
        DT,
        OBSERVATION_TERMS,
        file_sha256,
        load_reference_checkpoint,
        read_yaml_data,
    )
    from operator_curriculum import (
        OperatorExposureWrapper,
        VERSIONS,
        VERSION,
        curriculum_manifest,
    )
    from operator_profiles import (
        PROFILES,
        apply_reward_profile,
        profile_manifest,
        select_profile,
        source_profile,
    )
    from run_provenance import write_run_provenance
    from operator_retention import (
        install_moving_retention,
        retention_manifest,
        restore_adam_state,
        validate_adam_state,
    )


def training_configs(saved, agent, args):
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
        UnitreeGo2FlatPPORunnerCfg,
    )

    try:
        from .operator_command import OperatorTransitionCommand, OperatorVelocityCommand
    except ImportError:
        from operator_command import OperatorTransitionCommand, OperatorVelocityCommand

    source = source_profile(saved, agent)
    selected = select_profile(
        saved, agent, getattr(args, "refinement_profile", "source")
    )
    cfg = reference_config(saved)
    runner_cfg = UnitreeGo2FlatPPORunnerCfg().to_dict()
    known_algorithm = yaml.load(
        yaml.dump(runner_cfg["algorithm"]), Loader=yaml.BaseLoader
    )
    known_algorithm["entropy_coef"] = str(source.entropy_coef)
    # Never execute class/function names from an archived YAML through RSL eval.
    # Only the named profile's entropy and the explicit optimizer protocol differ.
    for key in known_algorithm.keys() | agent["algorithm"].keys():
        if key not in ("learning_rate", "schedule") and agent["algorithm"].get(
            key
        ) != known_algorithm.get(key):
            raise ValueError(f"Unsupported source algorithm.{key}")
    apply_reward_profile(cfg, selected)
    command = cfg.commands.base_velocity
    version = getattr(args, "curriculum", VERSION)
    curriculum_manifest(version)
    command.class_type = (
        OperatorVelocityCommand if version == VERSION else OperatorTransitionCommand
    )
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 0.0
    command.ranges.heading = None
    command.ranges.lin_vel_x = (-0.3, 0.7)
    command.ranges.lin_vel_y = (-0.2, 0.2)
    command.ranges.ang_vel_z = (-0.8, 0.8)
    command.resampling_time_range = (2.0, 12.0)
    command.debug_vis = False
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    # Retain observation corruption, 20-s episodes, all reward weights, action
    # scaling, mass/friction/reset events, collision geometry and terminations.
    runner_cfg.update(
        seed=args.seed,
        device=args.device,
        max_iterations=args.iterations,
        experiment_name="go2_operator_refinement",
        run_name=version,
        logger="tensorboard",
        save_interval=50,
        resume=False,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
    )
    runner_cfg["algorithm"].update(
        learning_rate=1.0e-4, schedule="fixed", entropy_coef=selected.entropy_coef
    )
    return cfg, runner_cfg


def restore_reference(runner, data):
    """Exact actor/critic/std handoff, fresh optimizer, unambiguous next update."""
    import torch

    if runner.alg.optimizer.state:
        raise ValueError("Refinement requires a newly constructed, empty optimizer")
    runner.alg.policy.load_state_dict(data["model_state_dict"], strict=True)
    actual = runner.alg.policy.state_dict()
    if any(
        not torch.equal(actual[key].detach().cpu(), value)
        for key, value in data["model_state_dict"].items()
    ):
        raise RuntimeError("Initial actor/critic/std differ from the source checkpoint")
    runner.current_learning_iteration = data["iter"] + 1
    return {
        "source_iteration": data["iter"],
        "first_update": runner.current_learning_iteration,
        "restored": ["actor", "critic", "action_standard_deviation"],
        "initial_state_verified_exact": True,
        "optimizer": "fresh Adam; source optimizer deliberately not restored",
        "learning_rate": runner.alg.learning_rate,
        "schedule": runner.alg.schedule,
    }


def run_training(args, output, agent, saved):
    resume = getattr(args, "retention_resume", None)
    reference = None
    if resume is not None:
        reference_path = args.resume_retention_reference
        reference = load_reference_checkpoint(
            reference_path, read_yaml_data(reference_path.parent / "params/agent.yaml")
        )
        if file_sha256(reference_path) != resume["reference_sha256"]:
            raise ValueError("Retention reference changed after preflight")
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, livestream=0, device=args.device)
    app = launcher.app
    env = None
    runner = None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        cfg, runner_cfg = training_configs(saved, agent, args)
        raw_env = ManagerBasedRLEnv(cfg=cfg)
        env = raw_env  # Keep ownership even if wrapper construction fails.
        if abs(raw_env.step_dt - DT) > 1e-9:
            raise ValueError("Reference refinement requires the 50-Hz action interface")
        if (
            tuple(raw_env.observation_manager.active_terms["policy"])
            != OBSERVATION_TERMS
        ):
            raise ValueError("Runtime observation order differs from the reference")
        if raw_env.action_manager.total_action_dim != 12:
            raise ValueError("Runtime action dimension differs from the reference")
        env = OperatorExposureWrapper(RslRlVecEnvWrapper(raw_env, clip_actions=None))
        params = output / "params"
        params.mkdir(exist_ok=True)
        (params / "env.yaml").write_text(yaml.dump(cfg.to_dict(), sort_keys=False))
        (params / "agent.yaml").write_text(yaml.dump(runner_cfg, sort_keys=False))
        source = load_reference_checkpoint(args.checkpoint, agent)
        if (
            resume is not None
            and file_sha256(args.checkpoint) != resume["checkpoint_sha256"]
        ):
            raise ValueError("Resume checkpoint changed after preflight")

        class RefinementRunner(OnPolicyRunner):
            def save(self, path, infos=None):
                exposure = env.exposure.report()
                super().save(path, {"handoff": handoff, "command_exposure": exposure})
                write_json(output / "command_exposure.json", exposure)

        runner = RefinementRunner(
            env, copy.deepcopy(runner_cfg), str(output), args.device
        )
        handoff = restore_reference(runner, source)
        retention = None
        if getattr(args, "moving_retention", False):
            # Freeze the exactly restored source once. Never re-anchor at an
            # intermediate checkpoint; keep a single uninterrupted Adam run.
            retention = install_moving_retention(
                runner.alg,
                reference_state=None
                if reference is None
                else reference["model_state_dict"],
            )
            if resume is not None:
                restore_adam_state(runner.alg, source, resume["adam_steps"])
                handoff["optimizer"] = (
                    "restored Adam moments, counters and options exactly"
                )
                handoff["restored"].append("optimizer_state")
                handoff["retention_resume"] = resume
        refinement = profile_manifest(
            saved, agent, getattr(args, "refinement_profile", "source")
        )
        handoff.update(
            source_checkpoint=str(args.checkpoint),
            source_sha256=file_sha256(args.checkpoint),
            additional_updates=args.iterations,
            final_iteration=source["iter"] + args.iterations,
            environment_transitions=args.num_envs
            * runner.num_steps_per_env
            * args.iterations,
            curriculum=curriculum_manifest(getattr(args, "curriculum", VERSION)),
            refinement_profile=refinement,
            command_metric_warning=(
                "Stock error_vel_xy/error_vel_yaw accumulators divide by max command duration: "
                "12 s here versus 4 s in the original reference. Identical physical errors "
                "can log 3x smaller values. Compare unchanged physical benchmark errors instead."
            ),
            unchanged=[
                "network",
                "observations",
                "actions",
                "physics",
                "resets",
                "terminations",
                "reward_weights",
            ],
        )
        if not refinement["reward_parameters_changed"]:
            handoff["unchanged"].append("rewards")
        elif refinement["reward_functions_changed"]:
            handoff["unchanged"].append("rewards_except_versioned_velocity_tracking")
        else:
            handoff["unchanged"].append("reward_parameters_except_yaw_tracking_std")
        if retention is not None:
            handoff["moving_retention"] = retention_manifest()
            handoff["moving_retention"]["reference_sha256"] = (
                handoff["source_sha256"]
                if resume is None
                else resume["reference_sha256"]
            )
            handoff["moving_retention"]["reference_iteration"] = (
                source["iter"] if reference is None else reference["iter"]
            )
            handoff["moving_retention"]["check_offsets"] = [100, 200]
            handoff["moving_retention"]["check_timing"] = (
                "after the uninterrupted 200-update worker exits"
            )
        write_json(params / "operator_training.json", handoff)
        # The pre-update checkpoint includes the fresh OR exactly resumed Adam.
        # Never overwrite or renumber the source run's model files.
        torch.save(
            {
                "model_state_dict": runner.alg.policy.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": source["iter"],
                "infos": {"handoff": handoff, "pre_update": True},
            },
            output / f"model_{source['iter']}.pt",
        )
        print(
            f"Restored actor/critic/std exactly from iteration {source['iter']}; "
            f"training updates {handoff['first_update']}–{handoff['final_iteration']}, "
            f"{handoff['optimizer']}; fixed 1e-4, streaming OFF.",
            flush=True,
        )
        # Normal reset ages preserve long standing windows from the first rollout.
        print(handoff["command_metric_warning"], flush=True)
        print(
            f"Refinement profile: {refinement['source']['name']} → "
            f"{refinement['selected']['name']}; "
            f"yaw reward std={refinement['selected']['yaw_tracking_std']}, "
            f"entropy coefficient={refinement['selected']['entropy_coef']}, "
            f"stationary precision={refinement['selected']['stationary_precision']}. "
            "Acceptance thresholds unchanged.",
            flush=True,
        )
        runner.learn(
            num_learning_iterations=args.iterations, init_at_random_ep_len=False
        )
        final = output / f"model_{handoff['final_iteration']}.pt"
        checked = load_reference_checkpoint(
            final, read_yaml_data(params / "agent.yaml")
        )
        if checked["iter"] != handoff["final_iteration"]:
            raise RuntimeError("Training did not reach the requested final checkpoint")
        if resume is not None:
            validate_adam_state(
                checked["optimizer_state_dict"],
                checked["model_state_dict"],
                resume["adam_steps"]
                + args.iterations
                * runner.alg.num_learning_epochs
                * runner.alg.num_mini_batches,
            )
        if (
            env.exposure.report()["environment_transitions"]
            != handoff["environment_transitions"]
        ):
            raise RuntimeError("Executed training budget differs from requested budget")
        if retention is not None and retention.updates != args.iterations:
            raise RuntimeError("Moving retention was not applied to every PPO update")
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        write_json(
            output / "training_status.json",
            {
                "status": "COMPLETED",
                "handoff": handoff,
                "final_checkpoint": str(final),
                "final_sha256": file_sha256(final),
                "behavior_validated": False,
            },
        )
    except Exception as error:
        write_json(
            output / "training_status.json",
            {
                "status": "ERROR",
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        # Kit can hard-exit Python with exit(0). Publish evidence before close;
        # only the independent parent owns success/failure exit semantics.
        for name, resource in (("environment", env), ("application", app)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    write_json(
                        output / f"{name}_cleanup_error.json",
                        {
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    )


def run_final_check(checkpoint, output, device, *, audit_student_interface=False):
    """Require both the benchmark exit code and its checkpoint-bound artifact."""
    result = {"status": "ERROR", "output_parent": str(output), "seed": 43}
    try:
        check = subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("operator_benchmark.py")),
                str(checkpoint),
                "--output-parent",
                str(output),
                "--device",
                device,
                *(["--audit-student-interface"] if audit_student_interface else []),
            ],
            check=False,
            timeout=600,
        )
        reports = list(output.glob("operator_screen_*/report.json"))
        if len(reports) != 1 or check.returncode not in (0, 1):
            raise ValueError("Benchmark failed or did not produce exactly one report")
        measured = json.loads(reports[0].read_text())
        if not isinstance(measured, dict):
            raise ValueError("Benchmark report must be an object")
        expected_status = "PASS" if check.returncode == 0 else "FAIL"
        if (
            measured.get("status") != expected_status
            or measured.get("seed") != 43
            or measured.get("provenance", {}).get("sha256", {}).get("checkpoint")
            != file_sha256(checkpoint)
        ):
            raise ValueError("Benchmark result/seed/checkpoint identity mismatch")
        result.update(status=expected_status, report=str(reports[0]))
        return result, check.returncode
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
        return result, 2


def retention_preflight(args, agent, saved):
    """Bind a bounded repair to replayed development evidence, before Kit starts."""
    try:
        from .operator_checkpoint_screen import replay_report
    except ImportError:
        from operator_checkpoint_screen import replay_report
    if (
        args.iterations != 200
        or args.curriculum != "operator_transitions_v2"
        or args.refinement_profile != "source"
        or source_profile(saved, agent).name != "stationary_twist_v1"
        or args.skip_check
        or args.baseline_report is None
        or importlib.metadata.version("rsl-rl-lib") != "3.1.2"
    ):
        raise ValueError(
            "Moving retention requires source stationary_twist_v1, v2 commands, RSL 3.1.2, 200 updates, a baseline report and checks enabled"
        )
    baseline = replay_report(args.baseline_report.resolve(strict=True), args.checkpoint)
    phases = baseline["phase_summary"]["phases"]
    if (
        baseline["passed"] < 90
        or any(p["sequence_failed"] for p in phases.values())
        or any(
            phases[name]["kinematic_passed"] != phases[name]["expected"]
            for name in ("forward", "restart")
        )
        or baseline["iteration"] % 50
    ):
        raise ValueError(
            "Repair source must have >=90/100 passes, complete forward/restart retention, no sequence violation, and a 50-aligned checkpoint"
        )
    return baseline


def retention_resume_preflight(args, agent, saved, baseline):
    """Permit one 200-update continuation of a completed initial retention run.

    Physics/rollout state is not checkpointed. This restores the learning state,
    not the exact random stream or trajectory of an uninterrupted 400-update run.
    """
    source = load_reference_checkpoint(args.checkpoint, agent)
    reference_path = args.resume_retention_reference.resolve(strict=True)
    reference = load_reference_checkpoint(
        reference_path, read_yaml_data(reference_path.parent / "params/agent.yaml")
    )
    reference_hash = file_sha256(reference_path)
    run = args.checkpoint.parent
    training_status = json.loads((run / "training_status.json").read_text())
    handoff = json.loads((run / "params/operator_training.json").read_text())
    provenance = json.loads((run / "source_provenance.json").read_text())
    loss = handoff.get("moving_retention", {})
    # Torch preserves tuples; JSON sidecars encode those same tuples as lists.
    checkpoint_handoff = json.loads(
        json.dumps(source.get("infos", {}).get("handoff"), allow_nan=False)
    )
    curriculum = json.loads(json.dumps(curriculum_manifest("operator_transitions_v2")))
    if (
        training_status.get("status") != "COMPLETED"
        or training_status.get("final_sha256") != baseline["sha256"]["checkpoint"]
        or training_status.get("handoff") != handoff
        or checkpoint_handoff != handoff
        or source["iter"] != handoff.get("final_iteration")
        or source["iter"] != reference["iter"] + 200
        or handoff.get("additional_updates") != 200
        or handoff.get("source_iteration") != reference["iter"]
        or handoff.get("first_update") != reference["iter"] + 1
        or handoff.get("source_sha256") != reference_hash
        or handoff.get("retention_resume") is not None
        or handoff.get("curriculum") != curriculum
        or any(loss.get(k) != v for k, v in retention_manifest().items())
        or loss.get("reference_sha256") != reference_hash
        or loss.get("reference_iteration") != reference["iter"]
        or loss.get("check_offsets") != [100, 200]
        or provenance.get("sha256", {}).get("checkpoint") != reference_hash
        or baseline["packages"].get("rsl-rl-lib") != "3.1.2"
    ):
        raise ValueError(
            "Resume requires the completed initial 200-update moving_anchor_v1 run and its exact original reference"
        )
    if (
        args.seed != int(agent["seed"])
        or args.seed != int(saved["seed"])
        or args.num_envs != int(saved["scene"]["num_envs"])
        or int(agent["num_steps_per_env"]) != 24
        or int(agent["save_interval"]) != 50
        or agent["algorithm"]["schedule"] != "fixed"
        or float(agent["algorithm"]["learning_rate"]) != 1e-4
    ):
        raise ValueError(
            "Retention resume must preserve training seed, environment count and optimizer/rollout settings"
        )
    for name in (
        "operator_curriculum.py",
        "operator_command.py",
        "operator_profiles.py",
        "operator_rewards.py",
    ):
        if provenance["sha256"].get(name) != file_sha256(
            Path(__file__).with_name(name)
        ):
            raise ValueError(
                f"Learning implementation changed since retention training: {name}"
            )
    steps = (
        200
        * int(agent["algorithm"]["num_learning_epochs"])
        * int(agent["algorithm"]["num_mini_batches"])
    )
    validate_adam_state(
        source["optimizer_state_dict"], source["model_state_dict"], steps
    )
    return {
        "checkpoint_sha256": baseline["sha256"]["checkpoint"],
        "reference_checkpoint": str(reference_path),
        "reference_sha256": reference_hash,
        "reference_iteration": reference["iter"],
        "adam_steps": steps,
        "additional_updates": 200,
        "cumulative_retention_updates": 400,
        "optimizer": "exact saved moments, counters, parameter order and options",
        "not_restored": [
            "simulator state",
            "command ages",
            "random generator state",
            "rollout buffers",
        ],
        "scope": "One bounded learning-state continuation; environment restarts, not bitwise uninterrupted-run equivalence",
        "evidence_sha256": {
            name: file_sha256(run / name)
            for name in (
                "training_status.json",
                "params/operator_training.json",
                "source_provenance.json",
            )
        },
    }


def retention_decision(measured, baseline):
    """Do not rescue a physical failure with an audit or a pooled score."""
    if measured["status"] == "PASS":
        return "DEVELOPMENT_PASS"
    phases = measured["phase_summary"]["phases"]
    if (
        measured["passed"] < baseline["passed"]
        or any(p["sequence_failed"] for p in phases.values())
        or any(
            phases[name]["kinematic_passed"] != phases[name]["expected"]
            for name in ("forward", "restart")
        )
    ):
        return "REGRESSED_CANDIDATE"
    return "NO_DEVELOPMENT_PASS"


def run_retention_checks(checkpoints, output, device, baseline):
    """At most two screens, only after the training worker has released the GPU."""
    try:
        from .operator_checkpoint_screen import replay_report
    except ImportError:
        from operator_checkpoint_screen import replay_report
    result = {
        "status": "ERROR",
        "promoted": False,
        "baseline": baseline,
        "candidates": [],
        "scope": "200 training updates already completed; only evaluation stops early. Selection seed43, not held-out confirmation, RMA or hardware acceptance.",
    }
    try:
        output.mkdir(parents=True, exist_ok=True)
        for checkpoint in checkpoints:
            check, code = run_final_check(
                checkpoint,
                output / checkpoint.stem,
                device,
                audit_student_interface=True,
            )
            if code not in (0, 1):
                raise RuntimeError(f"Retention screen execution failed: {check}")
            measured = replay_report(Path(check["report"]), checkpoint)
            audit = measured["student_interface_audit"]
            if not isinstance(audit, dict) or any(
                audit.get(k) != v
                for k, v in {
                    "status": "ORACLE_PARITY_PASS",
                    "control_steps": 1000,
                    "action_comparisons": 100000,
                    "exact_action_equality": True,
                    "student_status": "UNTRAINED_NOT_RUN",
                }.items()
            ):
                raise ValueError("Missing or incomplete shadow oracle audit")
            if measured["packages"] != baseline["packages"]:
                raise ValueError("Runtime packages changed from the repair baseline")
            measured["repair_classification"] = retention_decision(measured, baseline)
            result["candidates"].append(measured)
            result["status"] = (
                "DEVELOPMENT_PASS"
                if measured["status"] == "PASS"
                else "NO_DEVELOPMENT_PASS"
            )
            write_json(output / "report.json", result)
            if result["status"] == "DEVELOPMENT_PASS":
                break
        return result, 0 if result["status"] == "DEVELOPMENT_PASS" else 1
    except Exception as error:
        result.update(status="ERROR", error=str(error), error_type=type(error).__name__)
        return result, 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--refinement-profile",
        choices=("source", *PROFILES),
        default="source",
        help="Preserve the source profile, or explicitly select a versioned reward/entropy objective",
    )
    parser.add_argument(
        "--curriculum",
        choices=VERSIONS,
        default=VERSION,
        help="Versioned command distribution; v2 focuses on live stops and reverse",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=300,
        help="Additional PPO updates (default: 300)",
    )
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--moving-retention",
        action="store_true",
        help="Opt-in bounded source-mean retention repair; requires --baseline-report, v2, 200 updates",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Checkpoint-bound development report for --moving-retention",
    )
    parser.add_argument(
        "--resume-retention-reference",
        type=Path,
        help="Resume completed retention checkpoint AND Adam, keeping this original frozen reference (requires --moving-retention)",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("logs/rsl_rl/go2_operator_refinement"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="Training worker time limit, seconds",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Training smoke only; do not claim behavioral acceptance",
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (
        args.iterations < 1
        or args.num_envs < 4
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("Use positive iterations/timeout and at least four environments")
    if args.seed < 0:
        parser.error("Training seed must be a nonnegative integer")
    if args.seed in (43, 44, 45):
        parser.error(
            "Seeds 43–45 are reserved for evaluation; choose a different training seed"
        )
    try:
        args.checkpoint = args.checkpoint.resolve(strict=True)
        agent_path, env_path = (
            args.checkpoint.parent / "params" / name
            for name in ("agent.yaml", "env.yaml")
        )
        agent, saved = read_yaml_data(agent_path), read_yaml_data(env_path)
    except Exception as error:
        parser.error(f"Invalid source files ({type(error).__name__}): {error}")
    if (
        args.baseline_report is not None or args.resume_retention_reference is not None
    ) and not args.moving_retention:
        parser.error(
            "--baseline-report/--resume-retention-reference require --moving-retention"
        )
    baseline = None
    args.retention_resume = None
    if args.moving_retention:
        try:
            baseline = retention_preflight(args, agent, saved)
            if args.resume_retention_reference is not None:
                args.retention_resume = retention_resume_preflight(
                    args, agent, saved, baseline
                )
        except Exception as error:
            parser.error(str(error))
    if args.worker_output is not None:
        try:
            run_training(args, args.worker_output, agent, saved)
        except Exception as error:
            write_json(
                args.worker_output / "training_status.json",
                {
                    "status": "ERROR",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        return 0

    try:
        select_profile(saved, agent, args.refinement_profile)
        source = load_reference_checkpoint(args.checkpoint, agent)
    except Exception as error:
        parser.error(str(error))  # Reject before Kit/worker/output creation.
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(prefix="operator_refine_", dir=args.output_parent)
    ).resolve()
    print(f"Operator refinement (headless, streaming off): {output}", flush=True)
    write_run_provenance(output, Path(__file__))
    if baseline is not None:
        write_json(
            output / "retention_protocol.json",
            {
                "baseline": baseline,
                "loss": retention_manifest(),
                "training_updates": 200,
                "check_offsets": [100, 200],
                "seed": args.seed,
                "evaluation_seed": 43,
                "optimizer": "exact restored Adam; fixed original reference"
                if args.retention_resume
                else "one fresh Adam at the initial source; uninterrupted through all 200 updates",
                "retention_resume": args.retention_resume,
                "evaluation_timing": "after training worker exit; never concurrent Kit workers",
                "stop_screening": [
                    "complete unchanged physical PASS",
                    "execution/integrity error",
                ],
                "regressed_candidate": "reject individually, but still screen the already-trained +200 candidate; learning can be non-monotonic",
                "not_promoted": True,
            },
        )
    provenance = {
        "source_checkpoint": str(args.checkpoint),
        "sha256": {
            "checkpoint": file_sha256(args.checkpoint),
            "agent.yaml": file_sha256(agent_path),
            "env.yaml": file_sha256(env_path),
        },
        "packages": {},
    }
    for name in (
        "operator_train.py",
        "operator_curriculum.py",
        "operator_command.py",
        "operator_profiles.py",
        "operator_rewards.py",
        "operator_retention.py",
        "operator_benchmark.py",
        "operator_benchmark_core.py",
    ):
        provenance["sha256"][name] = file_sha256(Path(__file__).with_name(name))
    for package in ("isaaclab", "isaacsim", "rsl-rl-lib", "torch"):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][package] = "unknown"
    write_json(output / "source_provenance.json", provenance)
    if args.retention_resume is not None and provenance["packages"] != {
        name: baseline["packages"].get(name) for name in provenance["packages"]
    }:
        write_json(
            output / "report.json",
            {
                "status": "ERROR",
                "error": "Runtime packages changed since the retention baseline",
            },
        )
        print(
            f"ERROR: runtime packages changed; see {output / 'report.json'}", flush=True
        )
        return 2
    report = supervise(
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            str(args.checkpoint),
            "--worker-output",
            str(output),
            "--iterations",
            str(args.iterations),
            "--num-envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--curriculum",
            args.curriculum,
            "--refinement-profile",
            args.refinement_profile,
            *(
                [
                    "--moving-retention",
                    "--baseline-report",
                    str(args.baseline_report.resolve()),
                ]
                if args.moving_retention
                else []
            ),
            *(
                [
                    "--resume-retention-reference",
                    str(args.resume_retention_reference.resolve()),
                ]
                if args.retention_resume is not None
                else []
            ),
        ],
        output,
        timeout_s=args.timeout,
        report_filename="training_status.json",
        valid_statuses=("COMPLETED", "ERROR"),
    )
    final = output / f"model_{source['iter'] + args.iterations}.pt"
    if report["status"] == "COMPLETED" and (
        report.get("final_checkpoint") != str(final) or not final.is_file()
    ):
        report = {
            "status": "ERROR",
            "error": "Worker did not save the expected final checkpoint",
            "training_result": report,
        }
    write_json(output / "report.json", report)
    if report["status"] == "ERROR":
        print(report.get("traceback", report.get("error")), flush=True)
        return 2
    if baseline is not None:
        if report.get("handoff", {}).get("source_sha256") != baseline["sha256"][
            "checkpoint"
        ] or (
            args.retention_resume is not None
            and report.get("handoff", {}).get("retention_resume")
            != args.retention_resume
        ):
            report.update(
                status="ERROR", error="Training source differs from retention baseline"
            )
            write_json(output / "report.json", report)
            return 2
        checkpoints = [
            output / f"model_{source['iter'] + offset}.pt" for offset in (100, 200)
        ]
        print(
            "Training worker finished. Screening only the predeclared +100/+200 checkpoints.",
            flush=True,
        )
        report["operator_check"], code = run_retention_checks(
            checkpoints, output / "operator_check", args.device, baseline
        )
        try:
            (output / "operator_check").mkdir(parents=True, exist_ok=True)
            write_json(output / "operator_check/report.json", report["operator_check"])
            write_json(output / "report.json", report)
        except OSError as error:
            # A launch may fail before it creates a screen directory. Preserve
            # ERROR/exit2 even when nested evidence itself cannot be published.
            code = 2
            report["operator_check"].update(
                status="ERROR", publication_error=str(error)
            )
            try:
                write_json(output / "report.json", report)
            except OSError as parent_error:
                print(
                    f"Could not publish retention report: {parent_error}",
                    file=sys.stderr,
                )
        print(
            f"{report['operator_check']['status']}: {output / 'report.json'}",
            flush=True,
        )
        return code
    print(f"Training completed: {final}", flush=True)
    if args.skip_check:
        print("NOT behaviorally checked (--skip-check).", flush=True)
        return 0
    print("Running the unchanged 100-trial operator screen, seed 43.", flush=True)
    report["operator_check"], exit_code = run_final_check(
        final, output / "operator_check", args.device
    )
    write_json(output / "report.json", report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
