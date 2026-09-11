"""Bounded command-curriculum refinement of a stock Go2 reference checkpoint.

No external Isaac Lab training script is needed. Actor, critic and action noise
are restored exactly; only commands and the explicitly fresh PPO optimizer
change. This is a simulator-velocity teacher, NOT an RMA or obstacle policy.
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
    from .operator_curriculum import OperatorExposureWrapper, curriculum_manifest
    from .run_provenance import write_run_provenance
except ImportError:
    from operator_benchmark import reference_config, supervise, write_json
    from operator_benchmark_core import (
        DT,
        OBSERVATION_TERMS,
        file_sha256,
        load_reference_checkpoint,
        read_yaml_data,
    )
    from operator_curriculum import OperatorExposureWrapper, curriculum_manifest
    from run_provenance import write_run_provenance


def training_configs(saved, agent, args):
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
        UnitreeGo2FlatPPORunnerCfg,
    )

    try:
        from .operator_command import OperatorVelocityCommand
    except ImportError:
        from operator_command import OperatorVelocityCommand

    cfg = reference_config(saved)
    runner_cfg = UnitreeGo2FlatPPORunnerCfg().to_dict()
    known_algorithm = yaml.load(
        yaml.dump(runner_cfg["algorithm"]), Loader=yaml.BaseLoader
    )
    # Never execute class/function names from an archived YAML through RSL eval.
    # Changing the optimizer is intentional; all other PPO settings stay stock.
    for key in known_algorithm.keys() | agent["algorithm"].keys():
        if key not in ("learning_rate", "schedule") and agent["algorithm"].get(
            key
        ) != known_algorithm.get(key):
            raise ValueError(f"Unsupported source algorithm.{key}")
    command = cfg.commands.base_velocity
    command.class_type = OperatorVelocityCommand
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
        run_name="operator_modes_v1",
        logger="tensorboard",
        save_interval=50,
        resume=False,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
    )
    runner_cfg["algorithm"].update(learning_rate=1.0e-4, schedule="fixed")
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

        class RefinementRunner(OnPolicyRunner):
            def save(self, path, infos=None):
                exposure = env.exposure.report()
                super().save(path, {"handoff": handoff, "command_exposure": exposure})
                write_json(output / "command_exposure.json", exposure)

        runner = RefinementRunner(
            env, copy.deepcopy(runner_cfg), str(output), args.device
        )
        handoff = restore_reference(runner, source)
        handoff.update(
            source_checkpoint=str(args.checkpoint),
            source_sha256=file_sha256(args.checkpoint),
            additional_updates=args.iterations,
            final_iteration=source["iter"] + args.iterations,
            environment_transitions=args.num_envs
            * runner.num_steps_per_env
            * args.iterations,
            curriculum=curriculum_manifest(),
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
                "rewards",
                "resets",
                "terminations",
            ],
        )
        write_json(params / "operator_training.json", handoff)
        # A pre-update checkpoint is explicit and inspectable, with a fresh Adam
        # state. Never overwrite or renumber the source run's model files.
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
            "fresh Adam at fixed 1e-4, streaming OFF.",
            flush=True,
        )
        # Normal reset ages preserve long standing windows from the first rollout.
        print(handoff["command_metric_warning"], flush=True)
        runner.learn(
            num_learning_iterations=args.iterations, init_at_random_ep_len=False
        )
        final = output / f"model_{handoff['final_iteration']}.pt"
        checked = load_reference_checkpoint(
            final, read_yaml_data(params / "agent.yaml")
        )
        if checked["iter"] != handoff["final_iteration"]:
            raise RuntimeError("Training did not reach the requested final checkpoint")
        if (
            env.exposure.report()["environment_transitions"]
            != handoff["environment_transitions"]
        ):
            raise RuntimeError("Executed training budget differs from requested budget")
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


def run_final_check(checkpoint, output, device):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
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
    args.checkpoint = args.checkpoint.resolve(strict=True)
    agent_path, env_path = (
        args.checkpoint.parent / "params" / name for name in ("agent.yaml", "env.yaml")
    )
    agent, saved = read_yaml_data(agent_path), read_yaml_data(env_path)
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

    source = load_reference_checkpoint(
        args.checkpoint, agent
    )  # Reject before starting Kit.
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(prefix="operator_refine_", dir=args.output_parent)
    ).resolve()
    print(f"Operator refinement (headless, streaming off): {output}", flush=True)
    write_run_provenance(output, Path(__file__))
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
