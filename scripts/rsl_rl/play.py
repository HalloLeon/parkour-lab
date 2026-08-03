# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a checkpoint of an RL agent trained with RSL-RL."""

# Launch Isaac Sim before importing modules that depend on it.

import argparse
import os
import sys

import cli_args
from isaaclab.app import AppLauncher

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
    "--all_courses",
    action="store_true",
    default=False,
    help="Evaluate every configured obstacle-family and difficulty combination.",
)
parser.add_argument(
    "--difficulty_level",
    type=int,
    default=None,
    help="Fixed logical difficulty level. Supported environments provide their configured default when omitted.",
)
parser.add_argument(
    "--terrain_family",
    type=str,
    default=None,
    help="Fixed obstacle family. Supported environments provide their configured default when omitted.",
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
_original_cli_args = tuple(sys.argv[1:])
_launch_working_directory = os.getcwd()
_launch_script_path = os.path.abspath(__file__)
args_cli, hydra_args = parser.parse_known_args()
for argument_name in ("video_length", "eval_episodes"):
    argument_value = getattr(args_cli, argument_name)
    if argument_value is not None and argument_value <= 0:
        parser.error(f"--{argument_name} must be a positive integer.")
if args_cli.all_courses and (args_cli.terrain_family is not None or args_cli.difficulty_level is not None):
    parser.error("--all_courses cannot be combined with --terrain_family or --difficulty_level.")
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
_simulation_app_closed = False

# The remaining imports require the running simulation application.

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypedDict

import gymnasium as gym
import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401
import parkour_lab.tasks  # noqa: F401
import torch
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
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
    write_json,
)
from parkour_lab.learning.distillation.teacher.rsl_rl import (
    register_rsl_rl_teacher_actor_critic,
)
from parkour_lab.tasks.manager_based.parkour_lab.mdp.navigation.route import (
    last_episode_max_course_progress_m,
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
    mean_max_course_progress_m: float | None


class _EvaluationReport(TypedDict):
    """Complete evaluation report written to ``metrics.json``."""

    task: str | None
    checkpoint: str
    checkpoint_sha256: str
    teacher_interface: dict[str, object] | None
    teacher_interface_sha256: str | None
    seed: int | None
    terrain_family: str | None
    difficulty_level: int | None
    difficulty_metadata: dict[str, object]
    num_envs: int
    requested_episodes: int
    completed_episodes: int
    summary: _EvaluationSummary


@dataclass(frozen=True)
class _ArtifactInfo:
    """Output directory and video filename prefix for one evaluation."""

    directory: str
    video_name_prefix: str


@dataclass(frozen=True)
class _CheckpointInfo:
    """Resolved identity of the evaluated checkpoint."""

    path: str
    sha256: str
    stem: str
    log_dir: str


@dataclass(frozen=True)
class _InterfaceInfo:
    """Validated runtime interface metadata for the evaluated checkpoint."""

    teacher_interface: dict[str, object] | None
    teacher_interface_sha256: str | None


@dataclass
class _RolloutResult:
    """Mutable accumulator for completed-episode statistics."""

    completed_episodes: int = 0
    return_sum: float = 0.0
    length_steps_sum: int = 0
    success_count: int = 0
    trunk_contact_count: int = 0
    timeout_count: int = 0
    max_course_progress_m_sum: float = 0.0

    def record_completed(
        self,
        requested_episodes: int,
        done_mask: torch.Tensor,
        episode_returns: torch.Tensor,
        episode_lengths: torch.Tensor,
        outcomes: dict[str, torch.Tensor],
        episode_max_course_progress_m: torch.Tensor,
    ) -> None:
        """Accumulate newly completed episodes, capped at the requested total."""

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
        self.max_course_progress_m_sum += float(episode_max_course_progress_m[completed_indices].sum().item())

    def summary(self, step_dt: float) -> _EvaluationSummary:
        """Return aggregate means and rates for the completed episodes."""

        if self.completed_episodes == 0:
            return {
                "success_rate": None,
                "trunk_contact_rate": None,
                "timeout_rate": None,
                "mean_return": None,
                "mean_episode_length_steps": None,
                "mean_episode_length_seconds": None,
                "mean_max_course_progress_m": None,
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
            "mean_max_course_progress_m": self.max_course_progress_m_sum / count,
        }


# Capture the task ID and agent entry-point name now. The returned wrapper later
# loads their registered configuration defaults, lets Hydra consume the retained
# ``sys.argv`` overrides, and calls this function as ``main(env_cfg, agent_cfg)``.
@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Evaluate a checkpoint on one course or the complete course matrix."""
    agent_cfg = _apply_cli_overrides(env_cfg, agent_cfg)
    checkpoint = _resolve_checkpoint(agent_cfg)
    _evaluate_requested_courses(env_cfg, agent_cfg, checkpoint)


def _build_evaluation_report(
    *,
    env_cfg: ManagerBasedRLEnvCfg,
    checkpoint: _CheckpointInfo,
    interface: _InterfaceInfo,
    evaluation_family: str | None,
    evaluation_level: int | None,
    level_metadata: dict[str, object],
    num_envs: int,
    step_dt: float,
    rollout: _RolloutResult,
) -> _EvaluationReport:
    """Build the JSON-compatible report for one fixed evaluation course."""

    return {
        "task": args_cli.task,
        "checkpoint": checkpoint.path,
        "checkpoint_sha256": checkpoint.sha256,
        "teacher_interface": interface.teacher_interface,
        "teacher_interface_sha256": interface.teacher_interface_sha256,
        "seed": env_cfg.seed,
        "terrain_family": evaluation_family,
        "difficulty_level": evaluation_level,
        "difficulty_metadata": level_metadata,
        "num_envs": num_envs,
        "requested_episodes": args_cli.eval_episodes,
        "completed_episodes": rollout.completed_episodes,
        "summary": rollout.summary(step_dt),
    }


def _configure_evaluation_course(
    env_cfg: ManagerBasedRLEnvCfg,
    requested_family: str | None,
    requested_level: int | None,
) -> tuple[str | None, int | None, dict[str, object]]:
    """Freeze the config to one course and return its resolved metadata."""

    set_course = getattr(env_cfg, "set_evaluation_course", None)
    if not callable(set_course):
        if requested_family is not None or requested_level is not None:
            raise ValueError(
                f"Task '{args_cli.task}' does not support fixed parkour matrix selection because its "
                "environment config does not define set_evaluation_course()."
            )
        return None, None, {}

    # None lets the task select its own default family and maximum difficulty
    # after Hydra overrides have been synchronized.
    set_course(requested_family, requested_level, seed=env_cfg.seed)
    effective_family = getattr(env_cfg, "evaluation_family", requested_family)
    effective_level = getattr(env_cfg, "evaluation_level", requested_level)

    metadata_fn = getattr(env_cfg, "evaluation_course_metadata", None)
    metadata = metadata_fn() if callable(metadata_fn) else {}
    return effective_family, effective_level, metadata


def _apply_cli_overrides(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> RslRlBaseRunnerCfg:
    """Apply agent, environment-count, seed, and device CLI overrides."""

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    return agent_cfg


def _prepare_evaluation_artifacts(
    checkpoint: _CheckpointInfo,
    evaluation_family: str | None,
    evaluation_level: int | None,
    seed: int | None,
) -> _ArtifactInfo:
    """Create one output directory and derive its video filename prefix."""

    family_component = _path_component(evaluation_family, "default")
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
        f"family_{family_component}",
        f"level_{level_component}",
        f"seed_{seed_component}",
        evaluation_kind,
        evaluation_settings,
        run_id,
    )
    os.makedirs(directory, exist_ok=True)
    return _ArtifactInfo(
        directory=directory,
        video_name_prefix=(
            f"{checkpoint.stem}-family_{family_component}-level_{level_component}-seed_{seed_component}"
        ),
    )


def _configure_evaluation_stage(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Register the USD-context stage before Isaac Lab 2.3.1 initializes PhysX."""

    env_cfg.sim.create_stage_in_memory = False
    sim_utils.get_current_stage_id()


def _create_evaluation_environment(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    artifacts: _ArtifactInfo,
) -> RslRlVecEnvWrapper:
    """Instantiate one course and attach video and RSL-RL wrappers."""

    _configure_evaluation_stage(env_cfg)
    env_cfg.log_dir = artifacts.directory
    # Instantiate the registered Gym task with the resolved Isaac Lab
    # configuration, requesting rendered RGB frames only when recording video.
    gym_env: gym.Env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if args_cli.video:
        video_length = args_cli.video_length or int(gym_env.unwrapped.max_episode_length)
        video_kwargs = {
            "video_folder": artifacts.directory,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "name_prefix": artifacts.video_name_prefix,
            "disable_logger": True,
        }
        print("[INFO] Recording an evaluation video.")
        print_dict(video_kwargs, nesting=4)
        gym_env = gym.wrappers.RecordVideo(gym_env, **video_kwargs)

    return RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)


def _evaluate_course(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint: _CheckpointInfo,
    requested_family: str | None,
    requested_level: int | None,
) -> tuple[_EvaluationReport, str]:
    """Evaluate one fixed course, finalize its video, and write its report."""

    evaluation_family, evaluation_level, level_metadata = _configure_evaluation_course(
        env_cfg,
        requested_family,
        requested_level,
    )
    artifacts = _prepare_evaluation_artifacts(
        checkpoint,
        evaluation_family,
        evaluation_level,
        env_cfg.seed,
    )
    env = _create_evaluation_environment(env_cfg, agent_cfg, artifacts)
    num_envs = env.num_envs
    step_dt = env.unwrapped.step_dt

    try:
        observations = env.get_observations()
        interface = _validate_teacher_interface(
            env.unwrapped,
            observations,
            agent_cfg,
            checkpoint.path,
        )
        policy = _load_inference_policy(env, agent_cfg, checkpoint.path)
        rollout = _collect_rollout_statistics(env, observations, policy)
    finally:
        # Closing also finalizes a partial or completed RecordVideo recording.
        env.close()

    report = _build_evaluation_report(
        env_cfg=env_cfg,
        checkpoint=checkpoint,
        interface=interface,
        evaluation_family=evaluation_family,
        evaluation_level=evaluation_level,
        level_metadata=level_metadata,
        num_envs=num_envs,
        step_dt=step_dt,
        rollout=rollout,
    )
    report_path = _write_evaluation_report(artifacts.directory, report)
    return report, report_path


def _resolve_evaluation_courses(
    env_cfg: ManagerBasedRLEnvCfg,
) -> tuple[tuple[str | None, int | None], ...]:
    """Resolve the CLI selection to one course or every configured matrix cell."""

    if not args_cli.all_courses:
        return ((args_cli.terrain_family, args_cli.difficulty_level),)

    set_course = getattr(env_cfg, "set_evaluation_course", None)
    curriculum_cfg = getattr(env_cfg, "parkour_curriculum", None)
    validate_curriculum = getattr(curriculum_cfg, "validate_configuration", None)
    if callable(validate_curriculum):
        # Hydra/OmegaConf may reconstruct nested configclass entries as plain
        # mappings.  Normalize them before reading computed matrix properties.
        validate_curriculum()
    family_names = tuple(getattr(curriculum_cfg, "family_names", ()))
    num_difficulties = getattr(curriculum_cfg, "num_difficulties", None)
    if (
        not callable(set_course)
        or not family_names
        or isinstance(num_difficulties, bool)
        or not isinstance(num_difficulties, int)
        or num_difficulties <= 0
    ):
        raise ValueError(
            f"Task '{args_cli.task}' does not expose a parkour family/difficulty matrix for --all_courses."
        )

    return tuple(
        (family_name, difficulty_level) for family_name in family_names for difficulty_level in range(num_difficulties)
    )


def _load_inference_policy(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint_path: str,
) -> Callable[[TensorDict], torch.Tensor]:
    """Restore an OnPolicyRunner checkpoint and return its inference callable."""

    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError(
            "play.py supports only OnPolicyRunner teacher checkpoints; "
            "stock DistillationRunner checkpoints are not part of this project."
        )
    device = env.unwrapped.device
    register_rsl_rl_teacher_actor_critic()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint_path)
    return runner.get_inference_policy(device=device)


def _path_component(value: str | int | None, default: str) -> str:
    """Convert a value to a filesystem-safe path component."""

    text = default if value is None else str(value)
    return "".join(character if character.isalnum() or character in {"-", "_", "."} else "_" for character in text)


def _print_evaluation_summary(report: _EvaluationReport, report_path: str) -> None:
    """Print one course's aggregate metrics and report path."""

    def format_metric(value: float | None, *, rate: bool = False) -> str:
        """Format one optional scalar for terminal output."""

        if value is None:
            return "n/a"
        return f"{100.0 * value:.1f}%" if rate else f"{value:.4f}"

    summary = report["summary"]
    print("[RESULT] Evaluation summary")
    print(f"  Terrain family: {report['terrain_family'] or 'n/a'}")
    print(f"  Difficulty level: {report['difficulty_level']}")
    print(f"  Episodes: {report['completed_episodes']}/{report['requested_episodes']}")
    print(f"  Success rate: {format_metric(summary['success_rate'], rate=True)}")
    print(f"  Trunk-contact rate: {format_metric(summary['trunk_contact_rate'], rate=True)}")
    print(f"  Timeout rate: {format_metric(summary['timeout_rate'], rate=True)}")
    print(f"  Mean return: {format_metric(summary['mean_return'])}")
    print(f"  Mean episode length (steps): {format_metric(summary['mean_episode_length_steps'])}")
    print(f"  Mean episode length (seconds): {format_metric(summary['mean_episode_length_seconds'])}")
    print(f"  Mean maximum course progress (m): {format_metric(summary['mean_max_course_progress_m'])}")
    print(f"  Metrics: {report_path}")


def _resolve_checkpoint(agent_cfg: RslRlBaseRunnerCfg) -> _CheckpointInfo:
    """Resolve the checkpoint path and calculate its stable identity."""

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


def _collect_rollout_statistics(
    env: RslRlVecEnvWrapper,
    observations: TensorDict,
    policy: Callable[[TensorDict], torch.Tensor],
) -> _RolloutResult:
    """Aggregate completed episodes while deterministic inference is running."""

    step_dt = env.unwrapped.step_dt
    episode_returns = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.float32)
    episode_lengths = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.long)
    rollout = _RolloutResult()

    while simulation_app.is_running() and rollout.completed_episodes < args_cli.eval_episodes:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(observations)
            # Advance every parallel environment and return its next observations,
            # per-environment reward, episode-completion flags, and auxiliary data.
            observations, rewards, dones, _ = env.step(actions)

        rewards = rewards.reshape(-1).to(device=episode_returns.device)
        dones = dones.reshape(-1).to(device=episode_returns.device)
        done_mask = dones.to(dtype=torch.bool)
        episode_returns += rewards
        episode_lengths += 1
        outcomes = _read_termination_outcomes(env.unwrapped, done_mask)
        episode_max_course_progress_m = last_episode_max_course_progress_m(env.unwrapped)
        rollout.record_completed(
            args_cli.eval_episodes,
            done_mask,
            episode_returns,
            episode_lengths,
            outcomes,
            episode_max_course_progress_m,
        )
        episode_returns[done_mask] = 0.0
        episode_lengths[done_mask] = 0

        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    return rollout


def _evaluate_requested_courses(
    env_cfg: ManagerBasedRLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint: _CheckpointInfo,
) -> None:
    """Evaluate selected courses, isolating configs for a genuine sweep."""

    requested_courses = _resolve_evaluation_courses(env_cfg)
    if len(requested_courses) > 1:
        _evaluate_courses_in_subprocesses(requested_courses, checkpoint)
        return

    requested_family, requested_level = requested_courses[0]
    # Match Isaac Lab's official playback path for one course by passing the
    # Hydra-populated config directly to exactly one Gym environment.
    report, report_path = _evaluate_course(
        env_cfg,
        agent_cfg,
        checkpoint,
        requested_family,
        requested_level,
    )
    _print_evaluation_summary(report, report_path)


def _evaluate_courses_in_subprocesses(
    requested_courses: tuple[tuple[str | None, int | None], ...],
    checkpoint: _CheckpointInfo,
) -> None:
    """Evaluate every sweep cell in a fresh Isaac Sim process."""

    _close_simulation_application()
    for requested_family, requested_level in requested_courses:
        command = _evaluation_subprocess_command(
            checkpoint.path,
            requested_family,
            requested_level,
        )
        print(
            f"[INFO] Starting isolated course evaluation: family={requested_family}, level={requested_level}",
            flush=True,
        )
        subprocess.run(command, cwd=_launch_working_directory, check=True)


def _evaluation_subprocess_command(
    checkpoint_path: str,
    requested_family: str | None,
    requested_level: int | None,
) -> tuple[str, ...]:
    """Build one child command from the original CLI without sweep selection."""

    forwarded_args: list[str] = []
    skip_next = False
    for argument in _original_cli_args:
        if skip_next:
            skip_next = False
            continue
        if argument == "--all_courses":
            continue
        if argument == "--checkpoint":
            skip_next = True
            continue
        if argument.startswith("--checkpoint="):
            continue
        forwarded_args.append(argument)

    forwarded_args.append(f"--checkpoint={checkpoint_path}")
    if requested_family is not None:
        forwarded_args.append(f"--terrain_family={requested_family}")
    if requested_level is not None:
        forwarded_args.append(f"--difficulty_level={requested_level}")
    return (sys.executable, _launch_script_path, *forwarded_args)


def _close_simulation_application() -> None:
    """Close the parent SimulationApp at most once."""

    global _simulation_app_closed
    if not _simulation_app_closed:
        simulation_app.close()
        _simulation_app_closed = True


def _read_termination_outcomes(
    base_env: ManagerBasedRLEnv,
    done_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Read the per-environment outcome masks for the current step."""

    termination_manager = base_env.termination_manager
    return {
        "success": termination_manager.get_term("success").to(device=done_mask.device, dtype=torch.bool),
        "trunk_contact": termination_manager.get_term("trunk_contact").to(device=done_mask.device, dtype=torch.bool),
        "timeout": termination_manager.get_term("time_out").to(device=done_mask.device, dtype=torch.bool),
    }


def _validate_teacher_interface(
    base_env: ManagerBasedRLEnv,
    observations: TensorDict,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint_path: str,
) -> _InterfaceInfo:
    """Rebuild and validate the teacher interface when the policy uses one."""

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


def _write_evaluation_report(artifact_dir: str, report: _EvaluationReport) -> str:
    """Serialize one course report as ``metrics.json`` and return its path."""

    metrics_path = os.path.join(artifact_dir, "metrics.json")
    write_json(metrics_path, report)
    return metrics_path


if __name__ == "__main__":
    try:
        main()
    finally:
        _close_simulation_application()
