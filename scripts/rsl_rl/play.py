# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a checkpoint of an RL agent trained with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


def positive_int(value: str) -> int:
    """Parse a strictly positive integer argument."""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, received: {value}")
    return parsed_value


# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record an evaluation video.")
parser.add_argument(
    "--video_length",
    type=positive_int,
    default=None,
    help="Length of the recorded video in policy steps. Defaults to one full environment episode.",
)
parser.add_argument(
    "--video_output_dir",
    type=str,
    default=None,
    help="Base directory for evaluation videos and metrics. Defaults to the checkpoint run directory.",
)
parser.add_argument(
    "--difficulty_level",
    type=int,
    default=None,
    help="Fixed logical difficulty level. Supported environments provide their configured default when omitted.",
)
parser.add_argument(
    "--eval_episodes",
    type=positive_int,
    default=10,
    help="Number of completed episodes to evaluate.",
)
parser.add_argument(
    "--export_policy",
    action="store_true",
    default=False,
    help="Export the loaded policy to JIT and ONNX.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import parkour_lab.tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

installed_version = metadata.version("rsl-rl-lib")


def configure_evaluation_difficulty(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    requested_level: int | None,
) -> tuple[int | None, Any]:
    """Apply a task-specific fixed evaluation difficulty when the config supports it."""
    set_difficulty = getattr(env_cfg, "set_evaluation_difficulty", None)
    if not callable(set_difficulty):
        if requested_level is not None:
            raise ValueError(
                f"Task '{args_cli.task}' does not support --difficulty_level because its environment config "
                "does not define set_evaluation_difficulty()."
            )
        return None, {}

    # None lets the task select its own maximum/default after Hydra overrides
    # have been synchronized.
    result = set_difficulty(requested_level, seed=env_cfg.seed)
    effective_level = getattr(env_cfg, "evaluation_level", requested_level)
    if effective_level is None and isinstance(result, int):
        effective_level = result

    metadata_fn = getattr(env_cfg, "evaluation_level_metadata", None)
    metadata = metadata_fn() if callable(metadata_fn) else {}
    return effective_level, metadata


def path_component(value: Any, default: str) -> str:
    """Convert a value to a filesystem-safe path component."""
    text = default if value is None else str(value)
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in text)


def sha256_file(path: str) -> str:
    """Return a stable identity for the checkpoint contents."""

    digest = hashlib.sha256()
    with open(path, "rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    """Return repository identity without making Git a runtime requirement."""

    repository_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def manager_term_mask(
    base_env: Any, names: tuple[str, ...], reference: torch.Tensor
) -> tuple[torch.Tensor, str | None]:
    """Read the first available termination-manager term without recomputing it."""
    termination_manager = getattr(base_env, "termination_manager", None)
    if termination_manager is None:
        return torch.zeros_like(reference, dtype=torch.bool), None

    active_terms = set(getattr(termination_manager, "active_terms", ()))
    for name in names:
        if name in active_terms:
            return termination_manager.get_term(name).to(device=reference.device, dtype=torch.bool), name
    return torch.zeros_like(reference, dtype=torch.bool), None


def timeout_mask(base_env: Any, reference: torch.Tensor) -> tuple[torch.Tensor, str | None]:
    """Read the current timeout mask, preferring the named manager term."""
    mask, source = manager_term_mask(base_env, ("time_out", "timeout"), reference)
    if source is not None:
        return mask, source

    termination_manager = getattr(base_env, "termination_manager", None)
    manager_timeouts = getattr(termination_manager, "time_outs", None)
    if manager_timeouts is not None:
        return manager_timeouts.to(device=reference.device, dtype=torch.bool), "time_outs"

    reset_timeouts = getattr(base_env, "reset_time_outs", None)
    if reset_timeouts is not None:
        return reset_timeouts.to(device=reference.device, dtype=torch.bool), "reset_time_outs"

    return torch.zeros_like(reference, dtype=torch.bool), None


def reset_recurrent_policy_state(policy: Any, policy_nn: Any, dones: torch.Tensor) -> None:
    """Reset recurrent inference state when the loaded policy exposes a reset hook."""
    if not torch.any(dones):
        return
    for candidate in (policy, policy_nn):
        reset_fn = getattr(candidate, "reset", None)
        if callable(reset_fn):
            reset_fn(dones)
            return


def to_jsonable(value: Any) -> Any:
    """Recursively convert tensors and config objects to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_jsonable(to_dict())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(value)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Evaluate an RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # freeze task difficulty before constructing the simulator scene
    evaluation_level, evaluation_level_metadata = configure_evaluation_difficulty(env_cfg, args_cli.difficulty_level)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    checkpoint_path = os.path.abspath(resume_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    log_dir = os.path.dirname(checkpoint_path)

    # keep each checkpoint/level/seed evaluation in a self-describing folder
    checkpoint_stem = path_component(os.path.splitext(os.path.basename(checkpoint_path))[0], "checkpoint")
    checkpoint_id = f"{checkpoint_stem}-{checkpoint_sha256[:8]}"
    level_component = path_component(evaluation_level, "default")
    seed_component = path_component(env_cfg.seed, "default")
    evaluation_kind = "video" if args_cli.video else "metrics"
    evaluation_settings = f"episodes_{args_cli.eval_episodes}"
    if args_cli.video:
        evaluation_settings += f"-steps_{args_cli.video_length or 'full'}"
    evaluation_run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    artifact_root = (
        os.path.abspath(os.path.expanduser(args_cli.video_output_dir))
        if args_cli.video_output_dir is not None
        else os.path.join(log_dir, "evaluation")
    )
    artifact_dir = os.path.join(
        artifact_root,
        checkpoint_id,
        f"level_{level_component}",
        f"seed_{seed_component}",
        evaluation_kind,
        evaluation_settings,
        evaluation_run_id,
    )
    os.makedirs(artifact_dir, exist_ok=True)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = artifact_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_length = args_cli.video_length if args_cli.video_length is not None else int(env.unwrapped.max_episode_length)
    video_name_prefix = f"{checkpoint_stem}-level_{level_component}-seed_{seed_component}"

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": artifact_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "name_prefix": video_name_prefix,
            "disable_logger": True,
        }
        print("[INFO] Recording an evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(checkpoint_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    policy_nn = None
    if version.parse(installed_version) < version.parse("4.0.0"):
        # RSL-RL before 4.0 exports and resets through the policy module.
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

    if args_cli.export_policy:
        export_model_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")
        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        else:
            if policy_nn is None:
                raise RuntimeError("RSL-RL policy module is unavailable for export.")

            # extract the normalizer used by older RSL-RL policy modules
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
        print(f"[INFO] Exported policy to: {export_model_dir}")

    dt = env.unwrapped.step_dt

    num_envs = env.num_envs

    # the RSL-RL wrapper reset the environment during construction
    obs = env.get_observations()
    episode_returns = torch.zeros(num_envs, device=env.unwrapped.device, dtype=torch.float32)
    episode_lengths = torch.zeros(num_envs, device=env.unwrapped.device, dtype=torch.long)
    episode_results: list[dict[str, Any]] = []
    termination_sources: dict[str, str | None] = {
        "success": None,
        "trunk_contact": None,
        "timeout": None,
    }

    try:
        while simulation_app.is_running() and len(episode_results) < args_cli.eval_episodes:
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)

            rewards = rewards.reshape(-1).to(device=episode_returns.device)
            dones = dones.reshape(-1).to(device=episode_returns.device)
            done_mask = dones.to(dtype=torch.bool)
            episode_returns += rewards
            episode_lengths += 1

            # Read the masks produced by this step before the next manager compute.
            success_mask, success_source = manager_term_mask(env.unwrapped, ("success",), done_mask)
            trunk_contact_mask, trunk_contact_source = manager_term_mask(
                env.unwrapped, ("trunk_contact", "base_contact"), done_mask
            )
            current_timeout_mask, current_timeout_source = timeout_mask(env.unwrapped, done_mask)
            if success_source is not None:
                termination_sources["success"] = success_source
            if trunk_contact_source is not None:
                termination_sources["trunk_contact"] = trunk_contact_source
            if current_timeout_source is not None:
                termination_sources["timeout"] = current_timeout_source

            reset_recurrent_policy_state(policy, policy_nn, dones)

            done_indices = torch.nonzero(done_mask, as_tuple=False).flatten().tolist()
            for env_index in done_indices:
                if len(episode_results) >= args_cli.eval_episodes:
                    break
                episode_results.append(
                    {
                        "episode": len(episode_results) + 1,
                        "environment_index": env_index,
                        "return": float(episode_returns[env_index].item()),
                        "length_steps": int(episode_lengths[env_index].item()),
                        "length_seconds": float(episode_lengths[env_index].item() * dt),
                        "success": (bool(success_mask[env_index].item()) if success_source is not None else None),
                        "trunk_contact": (
                            bool(trunk_contact_mask[env_index].item()) if trunk_contact_source is not None else None
                        ),
                        "timeout": (
                            bool(current_timeout_mask[env_index].item()) if current_timeout_source is not None else None
                        ),
                    }
                )

            # Isaac Lab already reset completed environments; reset only our accumulators.
            episode_returns[done_mask] = 0.0
            episode_lengths[done_mask] = 0

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        # Closing also finalizes a partial or completed RecordVideo recording.
        env.close()

    def mean_metric(name: str) -> float | None:
        values = [float(result[name]) for result in episode_results if result[name] is not None]
        return sum(values) / len(values) if values else None

    def rate_metric(name: str) -> float | None:
        values = [bool(result[name]) for result in episode_results if result[name] is not None]
        return sum(values) / len(values) if values else None

    summary = {
        "success_rate": rate_metric("success"),
        "trunk_contact_rate": rate_metric("trunk_contact"),
        "timeout_rate": rate_metric("timeout"),
        "mean_return": mean_metric("return"),
        "mean_episode_length_steps": mean_metric("length_steps"),
        "mean_episode_length_seconds": mean_metric("length_seconds"),
    }
    metrics = {
        "task": args_cli.task,
        "checkpoint": checkpoint_path,
        "loaded_checkpoint": checkpoint_path,
        "checkpoint_stem": checkpoint_stem,
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "rsl_rl_version": installed_version,
        "git": git_state(),
        "evaluation_kind": evaluation_kind,
        "evaluation_run_id": evaluation_run_id,
        "seed": env_cfg.seed,
        "difficulty_level": evaluation_level,
        "difficulty_metadata": evaluation_level_metadata,
        "num_envs": num_envs,
        "requested_episodes": args_cli.eval_episodes,
        "completed_episodes": len(episode_results),
        "complete": len(episode_results) == args_cli.eval_episodes,
        "step_dt_seconds": dt,
        "video": {
            "enabled": args_cli.video,
            "length_steps": video_length if args_cli.video else None,
            "name_prefix": video_name_prefix if args_cli.video else None,
            "artifact_directory": artifact_dir,
        },
        "termination_term_sources": termination_sources,
        "summary": summary,
        "episodes": episode_results,
    }
    metrics_path = os.path.join(artifact_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(to_jsonable(metrics), metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")

    def format_metric(value: float | None, *, rate: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{100.0 * value:.1f}%" if rate else f"{value:.4f}"

    print("[RESULT] Evaluation summary")
    print(f"  Episodes: {len(episode_results)}/{args_cli.eval_episodes}")
    print(f"  Success rate: {format_metric(summary['success_rate'], rate=True)}")
    print(f"  Trunk-contact rate: {format_metric(summary['trunk_contact_rate'], rate=True)}")
    print(f"  Timeout rate: {format_metric(summary['timeout_rate'], rate=True)}")
    print(f"  Mean return: {format_metric(summary['mean_return'])}")
    print(f"  Mean episode length (steps): {format_metric(summary['mean_episode_length_steps'])}")
    print(f"  Mean episode length (seconds): {format_metric(summary['mean_episode_length_seconds'])}")
    print(f"  Metrics: {metrics_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
