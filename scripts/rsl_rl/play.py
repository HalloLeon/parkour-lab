# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a checkpoint of an RL agent trained with RSL-RL."""

# Launch Isaac Sim before importing modules that depend on it.

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args


# Define evaluation arguments.
parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record an evaluation video.")
parser.add_argument(
    "--video_length",
    type=int,
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
    type=int,
    default=10,
    help="Number of completed episodes to evaluate.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)
# Playback needs checkpoint selection, not training-only RSL-RL options such as
# resume state, run naming, or logger configuration.
cli_args.add_rsl_rl_checkpoint_args(parser)
# Add Isaac Lab application arguments.
AppLauncher.add_app_launcher_args(parser)
# Split recognized CLI options from the remaining Hydra configuration overrides.
args_cli, hydra_args = parser.parse_known_args()
for argument_name in ("video_length", "eval_episodes"):
    argument_value = getattr(args_cli, argument_name)
    if argument_value is not None and argument_value <= 0:
        parser.error(f"--{argument_name} must be a positive integer.")
# Enable cameras when recording video.
if args_cli.video:
    args_cli.enable_cameras = True

# ``hydra_task_config`` reads the global ``sys.argv`` when the decorated
# ``main`` is called later. Leave it only the script name and unparsed Hydra
# overrides, excluding options already consumed by argparse and AppLauncher.
sys.argv = [sys.argv[0]] + hydra_args

# Launch the Omniverse application.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# The remaining imports require the running simulation application.

import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from typing import TypedDict

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import parkour_lab.tasks  # noqa: F401
import torch
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from parkour_lab.learning.distillation.contracts import (
    TEACHER_OBSERVATION_GROUPS,
    assert_teacher_interface_matches,
    build_teacher_interface,
    interface_sha256,
    load_teacher_checkpoint,
    sha256_file,
)
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict


class _EvaluationSummary(TypedDict):
    """Aggregate metrics calculated from completed episodes."""

    success_rate: float | None
    trunk_contact_rate: float | None
    timeout_rate: float | None
    mean_return: float | None
    mean_episode_length_steps: float | None
    mean_episode_length_seconds: float | None


class _EvaluationReport(TypedDict):
    """Complete evaluation report written to ``metrics.json``."""

    # Registered Gym task used to create the evaluation environment.
    task: str | None

    # Absolute path identifying the checkpoint evaluated in this report.
    checkpoint: str

    # Complete SHA-256 hash identifying the exact checkpoint file contents.
    checkpoint_sha256: str

    # Reconstructed teacher observation, action, terrain, and timing interface.
    teacher_interface: dict[str, object] | None

    # SHA-256 identity of the reconstructed teacher-interface description.
    teacher_interface_sha256: str | None

    # Random seed used by the evaluated environment.
    seed: int | None

    # Fixed curriculum level selected for this evaluation, when supported.
    difficulty_level: int | None

    # Task-specific description of the selected difficulty level.
    difficulty_metadata: dict[str, object]

    # Number of parallel simulation environments used during evaluation.
    num_envs: int

    # Target number of completed episodes requested on the command line.
    requested_episodes: int

    # Number of completed episodes actually included in the aggregate metrics.
    completed_episodes: int

    # Aggregate returns, episode lengths, and termination rates.
    summary: _EvaluationSummary


@dataclass(frozen=True)
class _ArtifactInfo:
    """Paths and names shared by evaluation outputs."""

    # Directory receiving ``metrics.json`` and any recorded video.
    directory: str

    # Descriptive filename prefix containing the checkpoint, level, and seed.
    video_name_prefix: str


@dataclass(frozen=True)
class _CheckpointInfo:
    """Resolved identity of the evaluated checkpoint."""

    # Absolute path of the checkpoint loaded by RSL-RL.
    path: str

    # SHA-256 hash of the checkpoint contents, used to distinguish files that
    # share a name but contain different model weights.
    sha256: str

    # Filesystem-safe checkpoint filename without its extension.
    stem: str

    # Directory containing the checkpoint and its training artifacts.
    log_dir: str


@dataclass(frozen=True)
class _InterfaceInfo:
    """Teacher interface reconstructed for fixed evaluation."""

    # Runtime description of teacher observations, preprocessing, actions, and
    # control timing; ``None`` for a policy without the privileged-teacher route.
    teacher_interface: dict[str, object] | None

    # Hash of ``teacher_interface`` used to identify its exact contents.
    teacher_interface_sha256: str | None


@dataclass
class _RolloutResult:
    """Aggregate statistics collected from completed evaluation episodes."""

    completed_episodes: int = 0
    return_sum: float = 0.0
    length_steps_sum: int = 0
    success_count: int = 0
    trunk_contact_count: int = 0
    timeout_count: int = 0

    def record_completed(
        self,
        requested_episodes: int,
        done_mask: torch.Tensor,
        episode_returns: torch.Tensor,
        episode_lengths: torch.Tensor,
        outcomes: dict[str, torch.Tensor],
    ) -> None:
        """Add completed episodes without exceeding the requested total."""

        remaining = requested_episodes - self.completed_episodes
        completed_indices = torch.nonzero(done_mask, as_tuple=False).flatten()[:remaining]
        if completed_indices.numel() == 0:
            return

        self.completed_episodes += int(completed_indices.numel())
        self.return_sum += float(episode_returns[completed_indices].sum().item())
        self.length_steps_sum += int(episode_lengths[completed_indices].sum().item())
        self.success_count += int(outcomes["success"][completed_indices].sum().item())
        self.trunk_contact_count += int(outcomes["trunk_contact"][completed_indices].sum().item())
        self.timeout_count += int(outcomes["timeout"][completed_indices].sum().item())

    def summary(self, step_dt: float) -> _EvaluationSummary:
        """Calculate means and rates from the accumulated totals."""

        if self.completed_episodes == 0:
            return {
                "success_rate": None,
                "trunk_contact_rate": None,
                "timeout_rate": None,
                "mean_return": None,
                "mean_episode_length_steps": None,
                "mean_episode_length_seconds": None,
            }

        count = self.completed_episodes
        mean_length_steps = self.length_steps_sum / count
        return {
            "success_rate": self.success_count / count,
            "trunk_contact_rate": self.trunk_contact_count / count,
            "timeout_rate": self.timeout_count / count,
            "mean_return": self.return_sum / count,
            "mean_episode_length_steps": mean_length_steps,
            "mean_episode_length_seconds": mean_length_steps * step_dt,
        }


# Capture the task ID and agent entry-point name now. The returned wrapper later
# loads their registered configuration defaults, lets Hydra consume the retained
# ``sys.argv`` overrides, and calls this function as ``main(env_cfg, agent_cfg)``.
@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Evaluate an RSL-RL agent."""
    agent_cfg = _configure_runtime(env_cfg, agent_cfg)
    evaluation_level, level_metadata = _configure_evaluation_difficulty(env_cfg, args_cli.difficulty_level)
    checkpoint = _resolve_checkpoint(agent_cfg)
    artifacts = _create_artifacts(checkpoint, evaluation_level, env_cfg.seed)
    env = _create_environment(env_cfg, agent_cfg, artifacts)
    num_envs = env.num_envs
    step_dt = env.unwrapped.step_dt

    try:
        observations = env.get_observations()
        interface = _validate_teacher_interface(env.unwrapped, observations, agent_cfg, checkpoint.path)
        policy = _load_policy(env, agent_cfg, checkpoint.path)
        rollout = _run_evaluation(env, observations, policy)
    finally:
        # Closing also finalizes a partial or completed RecordVideo recording.
        env.close()

    metrics = _build_metrics(
        env_cfg=env_cfg,
        checkpoint=checkpoint,
        interface=interface,
        evaluation_level=evaluation_level,
        level_metadata=level_metadata,
        num_envs=num_envs,
        step_dt=step_dt,
        rollout=rollout,
    )
    metrics_path = _write_metrics(artifacts.directory, metrics)
    _print_summary(metrics, metrics_path)


def _build_metrics(
    *,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    checkpoint: _CheckpointInfo,
    interface: _InterfaceInfo,
    evaluation_level: int | None,
    level_metadata: dict[str, object],
    num_envs: int,
    step_dt: float,
    rollout: _RolloutResult,
) -> _EvaluationReport:
    """Assemble the complete JSON-compatible evaluation report."""

    return {
        "task": args_cli.task,
        "checkpoint": checkpoint.path,
        "checkpoint_sha256": checkpoint.sha256,
        "teacher_interface": interface.teacher_interface,
        "teacher_interface_sha256": interface.teacher_interface_sha256,
        "seed": env_cfg.seed,
        "difficulty_level": evaluation_level,
        "difficulty_metadata": level_metadata,
        "num_envs": num_envs,
        "requested_episodes": args_cli.eval_episodes,
        "completed_episodes": rollout.completed_episodes,
        "summary": rollout.summary(step_dt),
    }


def _configure_evaluation_difficulty(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    requested_level: int | None,
) -> tuple[int | None, dict[str, object]]:
    """Apply a task-specific fixed evaluation difficulty when the config supports it."""

    set_difficulty = getattr(env_cfg, "set_evaluation_difficulty", None)
    if not callable(set_difficulty):
        if requested_level is not None:
            raise ValueError(
                f"Task '{args_cli.task}' does not support --difficulty_level because its environment config "
                "does not define set_evaluation_difficulty()."
            )
        return None, {}

    # None lets the task select its own maximum or default after Hydra overrides
    # have been synchronized.
    result = set_difficulty(requested_level, seed=env_cfg.seed)
    effective_level = getattr(env_cfg, "evaluation_level", requested_level)
    if effective_level is None and isinstance(result, int):
        effective_level = result

    metadata_fn = getattr(env_cfg, "evaluation_level_metadata", None)
    metadata = metadata_fn() if callable(metadata_fn) else {}
    return effective_level, metadata


def _configure_runtime(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> RslRlBaseRunnerCfg:
    """Apply CLI overrides needed before constructing the environment."""

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    return agent_cfg


def _create_artifacts(
    checkpoint: _CheckpointInfo,
    evaluation_level: int | None,
    seed: int | None,
) -> _ArtifactInfo:
    """Create one collision-free artifact directory for this evaluation."""

    level_component = _path_component(evaluation_level, "default")
    seed_component = _path_component(seed, "default")
    evaluation_kind = "video" if args_cli.video else "metrics"
    evaluation_settings = f"episodes_{args_cli.eval_episodes}"
    if args_cli.video:
        evaluation_settings += f"-steps_{args_cli.video_length or 'full'}"
    # Build a readable UTC identifier as ``run_YYYYMMDD_HHMMSS``: ``%Y`` is
    # the year, ``%m`` the month, ``%d`` the day, ``%H`` the hour, ``%M`` the
    # minute, and ``%S`` the second.
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    artifact_root = (
        os.path.abspath(os.path.expanduser(args_cli.video_output_dir))
        if args_cli.video_output_dir is not None
        else os.path.join(checkpoint.log_dir, "evaluation")
    )
    directory = os.path.join(
        artifact_root,
        f"{checkpoint.stem}-{checkpoint.sha256[:8]}",
        f"level_{level_component}",
        f"seed_{seed_component}",
        evaluation_kind,
        evaluation_settings,
        run_id,
    )
    os.makedirs(directory, exist_ok=True)
    return _ArtifactInfo(
        directory=directory,
        video_name_prefix=f"{checkpoint.stem}-level_{level_component}-seed_{seed_component}",
    )


def _create_environment(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    artifacts: _ArtifactInfo,
) -> RslRlVecEnvWrapper:
    """Create, optionally record, and adapt the evaluation environment."""

    env_cfg.log_dir = artifacts.directory
    # Instantiate the registered Gym task with the resolved Isaac Lab
    # configuration, requesting rendered RGB frames only when recording video.
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    video_length = args_cli.video_length or int(env.unwrapped.max_episode_length)
    if args_cli.video:
        video_kwargs = {
            "video_folder": artifacts.directory,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "name_prefix": artifacts.video_name_prefix,
            "disable_logger": True,
        }
        print("[INFO] Recording an evaluation video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    return RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)


def _load_policy(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint_path: str,
) -> Callable[[TensorDict], torch.Tensor]:
    """Load the feed-forward PPO teacher and return its inference callable."""

    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(
            "play.py supports only OnPolicyRunner teacher checkpoints; "
            "stock DistillationRunner checkpoints are not part of this project."
        )
    device = env.unwrapped.device
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint_path)
    return runner.get_inference_policy(device=device)


def _path_component(value: str | int | None, default: str) -> str:
    """Convert a value to a filesystem-safe path component."""

    text = default if value is None else str(value)
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in text)


def _print_summary(metrics: _EvaluationReport, metrics_path: str) -> None:
    """Print the concise human-readable evaluation summary."""

    def format_metric(value: float | None, *, rate: bool = False) -> str:
        """Format one optional scalar for terminal output."""

        if value is None:
            return "n/a"
        return f"{100.0 * value:.1f}%" if rate else f"{value:.4f}"

    summary = metrics["summary"]
    print("[RESULT] Evaluation summary")
    print(f"  Episodes: {metrics['completed_episodes']}/{metrics['requested_episodes']}")
    print(f"  Success rate: {format_metric(summary['success_rate'], rate=True)}")
    print(f"  Trunk-contact rate: {format_metric(summary['trunk_contact_rate'], rate=True)}")
    print(f"  Timeout rate: {format_metric(summary['timeout_rate'], rate=True)}")
    print(f"  Mean return: {format_metric(summary['mean_return'])}")
    print(f"  Mean episode length (steps): {format_metric(summary['mean_episode_length_steps'])}")
    print(f"  Mean episode length (seconds): {format_metric(summary['mean_episode_length_seconds'])}")
    print(f"  Metrics: {metrics_path}")


def _resolve_checkpoint(agent_cfg: RslRlBaseRunnerCfg) -> _CheckpointInfo:
    """Resolve the requested checkpoint and calculate its stable identity."""

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # Match Isaac Lab's official playback behavior: an explicit checkpoint is
    # a complete path and takes precedence over run-based automatic lookup.
    resume_path = (
        retrieve_file_path(args_cli.checkpoint)
        if args_cli.checkpoint
        else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    path = os.path.abspath(resume_path)
    checkpoint_sha256 = sha256_file(path)
    stem = _path_component(os.path.splitext(os.path.basename(path))[0], "checkpoint")
    return _CheckpointInfo(
        path=path,
        sha256=checkpoint_sha256,
        stem=stem,
        log_dir=os.path.dirname(path),
    )


def _run_evaluation(
    env: RslRlVecEnvWrapper,
    observations: TensorDict,
    policy: Callable[[TensorDict], torch.Tensor],
) -> _RolloutResult:
    """Collect fixed-evaluation episodes using deterministic policy actions."""

    step_dt = env.unwrapped.step_dt
    episode_returns = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.float32)
    episode_lengths = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.long)
    rollout = _RolloutResult()

    while simulation_app.is_running() and rollout.completed_episodes < args_cli.eval_episodes:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(observations)
            observations, rewards, dones, _ = env.step(actions)

        rewards = rewards.reshape(-1).to(device=episode_returns.device)
        dones = dones.reshape(-1).to(device=episode_returns.device)
        done_mask = dones.to(dtype=torch.bool)
        episode_returns += rewards
        episode_lengths += 1
        outcomes = _termination_outcomes(env.unwrapped, done_mask)
        rollout.record_completed(
            args_cli.eval_episodes,
            done_mask,
            episode_returns,
            episode_lengths,
            outcomes,
        )
        episode_returns[done_mask] = 0.0
        episode_lengths[done_mask] = 0

        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    return rollout


def _termination_outcomes(
    base_env: ManagerBasedRLEnv | DirectRLEnv,
    done_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Read outcome masks produced by the current environment step."""

    if not isinstance(base_env, ManagerBasedRLEnv):
        raise TypeError("Parkour evaluation outcomes require a ManagerBasedRLEnv.")

    termination_manager = base_env.termination_manager
    return {
        "success": termination_manager.get_term("success").to(device=done_mask.device, dtype=torch.bool),
        "trunk_contact": termination_manager.get_term("trunk_contact").to(device=done_mask.device, dtype=torch.bool),
        "timeout": termination_manager.get_term("time_out").to(device=done_mask.device, dtype=torch.bool),
    }


def _to_jsonable(value: object) -> object:
    """Recursively convert tensors and config objects to JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _to_jsonable(to_dict())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_jsonable(item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _validate_teacher_interface(
    base_env: ManagerBasedRLEnv | DirectRLEnv,
    observations: TensorDict,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint_path: str,
) -> _InterfaceInfo:
    """Verify that the checkpoint receives its recorded teacher interface."""

    if tuple(agent_cfg.obs_groups.get("policy", ())) != TEACHER_OBSERVATION_GROUPS:
        return _InterfaceInfo(None, None)

    teacher_checkpoint = load_teacher_checkpoint(checkpoint_path)
    teacher_interface = build_teacher_interface(base_env, observations, agent_cfg)
    teacher_interface_hash = interface_sha256(teacher_interface)
    assert_teacher_interface_matches(
        teacher_checkpoint.teacher_interface,
        teacher_interface,
        context="Fixed-evaluation runtime",
    )
    return _InterfaceInfo(teacher_interface, teacher_interface_hash)


def _write_metrics(artifact_dir: str, metrics: _EvaluationReport) -> str:
    """Write the evaluation report and return its path."""

    metrics_path = os.path.join(artifact_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(_to_jsonable(metrics), metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")
    return metrics_path


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
