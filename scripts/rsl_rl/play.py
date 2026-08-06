# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a checkpoint of an RL agent trained with RSL-RL."""

# Launch Isaac Sim before importing modules that depend on it.

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

import cli_args
from isaaclab.app import AppLauncher


def _run_isolated_course_matrix(cli_arguments: list[str]) -> None:
    """Resolve the configured matrix, then evaluate each cell in a fresh process."""

    script_path = os.path.abspath(__file__)
    evaluation_arguments = [argument for argument in cli_arguments if argument != "--all_courses"]
    with tempfile.TemporaryDirectory(prefix="parkour_lab_courses_") as temporary_directory:
        manifest_path = os.path.join(temporary_directory, "courses.json")
        print("[INFO] Resolving the configured evaluation course matrix...", flush=True)
        subprocess.run(
            [sys.executable, script_path, *cli_arguments, f"--_course_manifest={manifest_path}"],
            check=True,
        )
        with open(manifest_path, encoding="utf-8") as manifest_file:
            courses = json.load(manifest_file)

    for index, (family, level) in enumerate(courses, start=1):
        print(f"[INFO] Evaluating course {index}/{len(courses)}: {family} level {level}", flush=True)
        subprocess.run(
            [
                sys.executable,
                script_path,
                *evaluation_arguments,
                f"--terrain_family={family}",
                f"--difficulty_level={level}",
            ],
            check=True,
        )


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
parser.add_argument("--_course_manifest", type=str, default=None, help=argparse.SUPPRESS)
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
    "--desired_speed",
    type=float,
    default=None,
    help="Fixed desired speed in m/s. Supported environments use their selected course default when omitted.",
)
parser.add_argument(
    "--eval_episodes",
    type=int,
    default=10,
    help="Number of completed episodes to evaluate.",
)
parser.add_argument(
    "--policy_mode",
    choices=("privileged_mean", "privileged_sampled", "history_mean", "history_sampled"),
    default="privileged_mean",
    help="Teacher action path used for the diagnostic rollout.",
)
parser.add_argument(
    "--reset_profile",
    choices=("canonical", "jitter"),
    default="canonical",
    help="Use the exact reset or isolated narrow initial-state jitter.",
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
cli_arguments = sys.argv[1:]
args_cli, hydra_args = parser.parse_known_args()
for argument_name in ("video_length", "eval_episodes"):
    argument_value = getattr(args_cli, argument_name)
    if argument_value is not None and argument_value <= 0:
        parser.error(f"--{argument_name} must be a positive integer.")
if args_cli.desired_speed is not None and (not math.isfinite(args_cli.desired_speed) or args_cli.desired_speed <= 0.0):
    parser.error("--desired_speed must be finite and positive.")
if args_cli.all_courses and (args_cli.terrain_family is not None or args_cli.difficulty_level is not None):
    parser.error("--all_courses cannot be combined with --terrain_family or --difficulty_level.")
if args_cli.all_courses and args_cli._course_manifest is None:
    _run_isolated_course_matrix(cli_arguments)
    raise SystemExit(0)
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

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from functools import partial
from typing import TypedDict, cast

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
from parkour_lab.learning.distillation.teacher.rsl_rl import (
    register_rsl_rl_teacher_actor_critic,
)
from parkour_lab.tasks.manager_based.parkour_lab.mdp.commands import get_target_speed
from parkour_lab.tasks.manager_based.parkour_lab.mdp.navigation import geometry, route
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict


class _EvaluationSummary(TypedDict):
    """Aggregate metrics calculated from completed episodes."""

    success_rate: float | None
    trunk_contact_rate: float | None
    fell_below_course_rate: float | None
    timeout_rate: float | None
    mean_return: float | None
    mean_episode_length_steps: float | None
    mean_episode_length_seconds: float | None
    mean_max_course_progress_m: float | None
    mean_max_waypoints_reached: float | None
    mean_forward_speed_m_s: float | None
    mean_overspeed_ratio: float | None
    rms_vertical_velocity_m_s: float | None
    all_feet_airborne_fraction: float | None
    mean_feet_edge_contacts_per_step: float | None
    mean_undesired_leg_contacts_per_step: float | None


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

    policy_mode: str
    reset_profile: str
    reset_parameters: dict[str, object]
    training_config: dict[str, dict[str, str]]

    # Fixed obstacle family selected for this independent evaluation report.
    terrain_family: str | None

    # Fixed curriculum level selected for this evaluation, when supported.
    difficulty_level: int | None

    # Deterministic scalar command used for every episode in this report.
    desired_speed_m_s: float | None

    # Task-specific description of the selected family/difficulty matrix cell.
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
    """Output directory and video filename prefix for one evaluation."""

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
    """Validated runtime interface metadata for the evaluated checkpoint."""

    # Runtime description of teacher observations, preprocessing, actions, and
    # control timing; ``None`` for a policy without the privileged-teacher route.
    teacher_interface: dict[str, object] | None

    # Hash of ``teacher_interface`` used to identify its exact contents.
    teacher_interface_sha256: str | None


@dataclass
class _RolloutResult:
    """Mutable accumulator for completed-episode statistics."""

    completed_episodes: int = 0
    return_sum: float = 0.0
    length_steps_sum: int = 0
    success_count: int = 0
    trunk_contact_count: int = 0
    fell_below_course_count: int = 0
    timeout_count: int = 0
    max_course_progress_m_sum: float = 0.0
    max_waypoints_reached_sum: float = 0.0
    forward_speed_m_s_sum: float = 0.0
    overspeed_ratio_sum: float = 0.0
    vertical_velocity_squared_m2_s2_sum: float = 0.0
    all_feet_airborne_sum: float = 0.0
    feet_edge_contacts_sum: float = 0.0
    undesired_leg_contacts_sum: float = 0.0

    def record_completed(
        self,
        requested_episodes: int,
        done_mask: torch.Tensor,
        episode_returns: torch.Tensor,
        episode_lengths: torch.Tensor,
        outcomes: dict[str, torch.Tensor],
        episode_max_course_progress_m: torch.Tensor,
        episode_max_waypoints_reached: torch.Tensor,
        episode_metric_sums: dict[str, torch.Tensor],
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
        self.fell_below_course_count += int(outcomes["fell_below_course"][completed_indices].sum().item())
        self.timeout_count += int(outcomes["timeout"][completed_indices].sum().item())
        self.max_course_progress_m_sum += float(episode_max_course_progress_m[completed_indices].sum().item())
        reached = episode_max_waypoints_reached[completed_indices]
        self.max_waypoints_reached_sum += float(reached.sum().item())
        self.forward_speed_m_s_sum += float(episode_metric_sums["forward_speed_m_s"][completed_indices].sum().item())
        self.overspeed_ratio_sum += float(episode_metric_sums["overspeed_ratio"][completed_indices].sum().item())
        self.vertical_velocity_squared_m2_s2_sum += float(
            episode_metric_sums["vertical_velocity_squared_m2_s2"][completed_indices].sum().item()
        )
        self.all_feet_airborne_sum += float(episode_metric_sums["all_feet_airborne"][completed_indices].sum().item())
        self.feet_edge_contacts_sum += float(
            episode_metric_sums["feet_edge_contacts"][completed_indices].sum().item()
        )
        self.undesired_leg_contacts_sum += float(
            episode_metric_sums["undesired_leg_contacts"][completed_indices].sum().item()
        )

    def summary(self, step_dt: float) -> _EvaluationSummary:
        """Return aggregate means and rates for the completed episodes."""

        if self.completed_episodes == 0:
            return cast(_EvaluationSummary, dict.fromkeys(_EvaluationSummary.__annotations__))

        count = self.completed_episodes
        step_count = self.length_steps_sum
        mean_length_steps = self.length_steps_sum / count
        return {
            "success_rate": self.success_count / count,
            "trunk_contact_rate": self.trunk_contact_count / count,
            "fell_below_course_rate": self.fell_below_course_count / count,
            "timeout_rate": self.timeout_count / count,
            "mean_return": self.return_sum / count,
            "mean_episode_length_steps": mean_length_steps,
            "mean_episode_length_seconds": mean_length_steps * step_dt,
            "mean_max_course_progress_m": self.max_course_progress_m_sum / count,
            "mean_max_waypoints_reached": self.max_waypoints_reached_sum / count,
            "mean_forward_speed_m_s": self.forward_speed_m_s_sum / step_count,
            "mean_overspeed_ratio": self.overspeed_ratio_sum / step_count,
            "rms_vertical_velocity_m_s": (self.vertical_velocity_squared_m2_s2_sum / step_count) ** 0.5,
            "all_feet_airborne_fraction": self.all_feet_airborne_sum / step_count,
            "mean_feet_edge_contacts_per_step": self.feet_edge_contacts_sum / step_count,
            "mean_undesired_leg_contacts_per_step": self.undesired_leg_contacts_sum / step_count,
        }


# Capture the task ID and agent entry-point name now. The returned wrapper later
# loads their registered configuration defaults, lets Hydra consume the retained
# ``sys.argv`` overrides, and calls this function as ``main(env_cfg, agent_cfg)``.
@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Evaluate a checkpoint on one course or the complete course matrix."""
    _run_requested_action(env_cfg, agent_cfg)


def _run_requested_action(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Write a matrix manifest or evaluate the requested single course."""

    if args_cli._course_manifest is not None:
        _write_course_manifest(env_cfg, args_cli._course_manifest)
        return

    agent_cfg = _apply_cli_overrides(env_cfg, agent_cfg)
    checkpoint = _resolve_checkpoint(agent_cfg)
    _evaluate_requested_course(env_cfg, agent_cfg, checkpoint)


def _build_evaluation_report(
    *,
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    checkpoint: _CheckpointInfo,
    interface: _InterfaceInfo,
    evaluation_family: str | None,
    evaluation_level: int | None,
    desired_speed: float | None,
    level_metadata: dict[str, object],
    num_envs: int,
    step_dt: float,
    rollout: _RolloutResult,
) -> _EvaluationReport:
    """Build the JSON-compatible report for one fixed evaluation course."""

    reset_base = getattr(getattr(env_cfg, "events", None), "reset_base", None)
    return {
        "task": args_cli.task,
        "checkpoint": checkpoint.path,
        "checkpoint_sha256": checkpoint.sha256,
        "teacher_interface": interface.teacher_interface,
        "teacher_interface_sha256": interface.teacher_interface_sha256,
        "seed": env_cfg.seed,
        "policy_mode": args_cli.policy_mode,
        "reset_profile": args_cli.reset_profile,
        "reset_parameters": getattr(reset_base, "params", {}),
        "training_config": _training_config_provenance(checkpoint.log_dir),
        "terrain_family": evaluation_family,
        "difficulty_level": evaluation_level,
        "desired_speed_m_s": desired_speed,
        "difficulty_metadata": level_metadata,
        "num_envs": num_envs,
        "requested_episodes": args_cli.eval_episodes,
        "completed_episodes": rollout.completed_episodes,
        "summary": rollout.summary(step_dt),
    }


def _configure_evaluation_course(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    requested_family: str | None,
    requested_level: int | None,
    requested_speed: float | None,
) -> tuple[str | None, int | None, float | None, dict[str, object]]:
    """Freeze the config to one course and return its resolved metadata."""

    set_course = getattr(env_cfg, "set_evaluation_course", None)
    if not callable(set_course):
        if requested_family is not None or requested_level is not None or requested_speed is not None:
            raise ValueError(
                f"Task '{args_cli.task}' does not support fixed parkour evaluation because its environment "
                "config does not define set_evaluation_course()."
            )
        return None, None, None, {}

    # None lets the task select its own default family and maximum difficulty
    # after Hydra overrides have been synchronized.
    set_course(requested_family, requested_level, seed=env_cfg.seed)
    if requested_speed is not None:
        set_speed = getattr(env_cfg, "set_evaluation_speed", None)
        if not callable(set_speed):
            raise ValueError(f"Task '{args_cli.task}' does not support --desired_speed.")
        set_speed(requested_speed)
    if args_cli.reset_profile != "canonical":
        set_reset_profile = getattr(env_cfg, "set_evaluation_reset_profile", None)
        if not callable(set_reset_profile):
            raise ValueError(f"Task '{args_cli.task}' does not support --reset_profile jitter.")
        set_reset_profile(args_cli.reset_profile)
    effective_family = getattr(env_cfg, "evaluation_family", requested_family)
    effective_level = getattr(env_cfg, "evaluation_level", requested_level)
    resolved_speed = getattr(env_cfg, "resolved_evaluation_speed", None)
    effective_speed = (
        resolved_speed() if callable(resolved_speed) else getattr(env_cfg, "evaluation_desired_speed", requested_speed)
    )

    metadata_fn = getattr(env_cfg, "evaluation_course_metadata", None)
    metadata = metadata_fn() if callable(metadata_fn) else {}
    return effective_family, effective_level, effective_speed, metadata


def _apply_cli_overrides(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
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
    *,
    desired_speed: float | None = None,
) -> _ArtifactInfo:
    """Create one output directory and derive its video filename prefix."""

    family_component = _path_component(evaluation_family, "default")
    level_component = _path_component(evaluation_level, "default")
    speed_component = _path_component(desired_speed, "default")
    seed_component = _path_component(seed, "default")
    evaluation_kind = "video" if args_cli.video else "metrics"
    evaluation_settings = f"{args_cli.policy_mode}-{args_cli.reset_profile}-episodes_{args_cli.eval_episodes}"
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
        f"speed_{speed_component}",
        f"seed_{seed_component}",
        evaluation_kind,
        evaluation_settings,
        run_id,
    )
    os.makedirs(directory, exist_ok=True)
    return _ArtifactInfo(
        directory=directory,
        video_name_prefix=(
            f"{checkpoint.stem}-{args_cli.policy_mode}-{args_cli.reset_profile}-"
            f"family_{family_component}-level_{level_component}-speed_{speed_component}-seed_{seed_component}"
        ),
    )


def _create_evaluation_environment(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    artifacts: _ArtifactInfo,
) -> RslRlVecEnvWrapper:
    """Instantiate one course and attach video and RSL-RL wrappers."""

    env_cfg.log_dir = artifacts.directory
    # Instantiate the registered Gym task with the resolved Isaac Lab
    # configuration, requesting rendered RGB frames only when recording video.
    gym_env: gym.Env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if isinstance(gym_env.unwrapped, DirectMARLEnv):
        gym_env = multi_agent_to_single_agent(gym_env)

    video_length = args_cli.video_length or int(gym_env.unwrapped.max_episode_length)
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
        gym_env = gym.wrappers.RecordVideo(gym_env, **video_kwargs)

    return RslRlVecEnvWrapper(gym_env, clip_actions=agent_cfg.clip_actions)


def _evaluate_course(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint: _CheckpointInfo,
    requested_family: str | None,
    requested_level: int | None,
) -> tuple[_EvaluationReport, str]:
    """Evaluate one fixed course, finalize its video, and write its report."""

    evaluation_family, evaluation_level, desired_speed, level_metadata = _configure_evaluation_course(
        env_cfg,
        requested_family,
        requested_level,
        args_cli.desired_speed,
    )
    artifacts = _prepare_evaluation_artifacts(
        checkpoint,
        evaluation_family,
        evaluation_level,
        env_cfg.seed,
        desired_speed=desired_speed,
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
        desired_speed=desired_speed,
        level_metadata=level_metadata,
        num_envs=num_envs,
        step_dt=step_dt,
        rollout=rollout,
    )
    report_path = _write_evaluation_report(artifacts.directory, report)
    return report, report_path


def _configured_evaluation_courses(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
) -> tuple[tuple[str | None, int | None], ...]:
    """Return every cell in the Hydra-resolved parkour course matrix."""

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


def _write_course_manifest(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    manifest_path: str,
) -> None:
    """Write the Hydra-resolved course matrix for the process coordinator."""

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(_configured_evaluation_courses(env_cfg), manifest_file)


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
    runner.load(checkpoint_path, load_optimizer=False)
    inference_policy = runner.get_inference_policy(device=device)
    policy = runner.alg.policy
    if args_cli.policy_mode == "privileged_mean":
        return inference_policy
    if args_cli.policy_mode == "privileged_sampled":
        return policy.act
    history_policy = getattr(policy, "act_inference_from_history", None)
    if not callable(history_policy):
        raise TypeError("History evaluation requires a privileged teacher checkpoint.")
    return history_policy if args_cli.policy_mode == "history_mean" else partial(policy.act, use_history=True)


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
    print(f"  Policy mode: {report['policy_mode']}")
    print(f"  Reset profile: {report['reset_profile']}")
    print(f"  Terrain family: {report['terrain_family'] or 'n/a'}")
    print(f"  Difficulty level: {report['difficulty_level']}")
    print(f"  Desired speed (m/s): {format_metric(report['desired_speed_m_s'])}")
    print(f"  Episodes: {report['completed_episodes']}/{report['requested_episodes']}")
    print(f"  Success rate: {format_metric(summary['success_rate'], rate=True)}")
    print(f"  Trunk-contact rate: {format_metric(summary['trunk_contact_rate'], rate=True)}")
    print(f"  Fell-below-course rate: {format_metric(summary['fell_below_course_rate'], rate=True)}")
    print(f"  Timeout rate: {format_metric(summary['timeout_rate'], rate=True)}")
    print(f"  Mean return: {format_metric(summary['mean_return'])}")
    print(f"  Mean episode length (steps): {format_metric(summary['mean_episode_length_steps'])}")
    print(f"  Mean episode length (seconds): {format_metric(summary['mean_episode_length_seconds'])}")
    print(f"  Mean maximum course progress (m): {format_metric(summary['mean_max_course_progress_m'])}")
    print(f"  Mean maximum waypoints reached: {format_metric(summary['mean_max_waypoints_reached'])}")
    print(f"  Mean forward speed (m/s): {format_metric(summary['mean_forward_speed_m_s'])}")
    print(f"  Mean overspeed ratio: {format_metric(summary['mean_overspeed_ratio'], rate=True)}")
    print(f"  RMS vertical velocity (m/s): {format_metric(summary['rms_vertical_velocity_m_s'])}")
    airborne_fraction = format_metric(summary["all_feet_airborne_fraction"], rate=True)
    print(f"  All-feet-airborne fraction: {airborne_fraction}")
    print(f"  Mean feet-edge contacts per step: {format_metric(summary['mean_feet_edge_contacts_per_step'])}")
    print(
        "  Mean undesired-leg contacts per step: "
        f"{format_metric(summary['mean_undesired_leg_contacts_per_step'])}"
    )
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
    """Aggregate completed episodes for the selected policy mode."""

    step_dt = env.unwrapped.step_dt
    episode_returns = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.float32)
    episode_lengths = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.long)
    episode_max_waypoints_reached = torch.zeros(
        env.num_envs,
        device=env.unwrapped.device,
        dtype=torch.long,
    )
    episode_metric_sums = {
        name: torch.zeros_like(episode_returns)
        for name in (
            "forward_speed_m_s",
            "overspeed_ratio",
            "vertical_velocity_squared_m2_s2",
            "all_feet_airborne",
            "feet_edge_contacts",
            "undesired_leg_contacts",
        )
    }
    rollout = _RolloutResult()

    while simulation_app.is_running() and rollout.completed_episodes < args_cli.eval_episodes:
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(observations)
            active_waypoint_indices = route.active_waypoint_indices(env.unwrapped)
            step_metrics = _read_step_metrics(env.unwrapped)
            # Advance every parallel environment and return its next observations,
            # per-environment reward, episode-completion flags, and auxiliary data.
            observations, rewards, dones, _ = env.step(actions)

        rewards = rewards.reshape(-1).to(device=episode_returns.device)
        dones = dones.reshape(-1).to(device=episode_returns.device)
        done_mask = dones.to(dtype=torch.bool)
        episode_returns += rewards
        episode_lengths += 1
        for name, values in step_metrics.items():
            episode_metric_sums[name] += values
        outcomes = _read_termination_outcomes(env.unwrapped, done_mask)
        # The cursor counts prior targets; success adds its still-active final target.
        episode_max_waypoints_reached = torch.maximum(
            episode_max_waypoints_reached,
            active_waypoint_indices + outcomes["success"].to(dtype=active_waypoint_indices.dtype),
        )
        episode_max_course_progress_m = route.last_episode_max_course_progress_m(env.unwrapped)
        rollout.record_completed(
            args_cli.eval_episodes,
            done_mask,
            episode_returns,
            episode_lengths,
            outcomes,
            episode_max_course_progress_m,
            episode_max_waypoints_reached,
            episode_metric_sums,
        )
        episode_returns[done_mask] = 0.0
        episode_lengths[done_mask] = 0
        episode_max_waypoints_reached[done_mask] = 0
        for values in episode_metric_sums.values():
            values[done_mask] = 0.0

        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    return rollout


def _read_step_metrics(base_env: ManagerBasedRLEnv | DirectRLEnv) -> dict[str, torch.Tensor]:
    """Read evaluation signals before the environment can auto-reset."""

    if not isinstance(base_env, ManagerBasedRLEnv):
        raise TypeError("Parkour metrics require a ManagerBasedRLEnv.")

    forward_speed = geometry._velocity_along_active_waypoint_xy(base_env)
    target_speed = get_target_speed(base_env).to(device=forward_speed.device, dtype=forward_speed.dtype)
    vertical_velocity = base_env.scene["robot"].data.root_lin_vel_w[:, 2]
    all_feet_airborne = torch.all(
        base_env.scene["feet_contact"].data.current_air_time > 0.0,
        dim=-1,
    )

    def raw_reward_term(name: str) -> torch.Tensor:
        term_cfg = base_env.reward_manager.get_term_cfg(name)
        values = term_cfg.func(base_env, **term_cfg.params)
        if values.shape != (base_env.num_envs,):
            raise ValueError(f"Reward term '{name}' must return one value per environment.")
        return values

    return {
        "forward_speed_m_s": forward_speed,
        "overspeed_ratio": torch.relu(forward_speed - target_speed)
        / target_speed.clamp_min(torch.finfo(target_speed.dtype).eps),
        "vertical_velocity_squared_m2_s2": vertical_velocity.square(),
        "all_feet_airborne": all_feet_airborne.to(dtype=forward_speed.dtype),
        "feet_edge_contacts": raw_reward_term("feet_edge"),
        "undesired_leg_contacts": raw_reward_term("leg_contact"),
    }


def _evaluate_requested_course(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint: _CheckpointInfo,
) -> None:
    """Evaluate the one course assigned to this Isaac Sim process."""

    report, report_path = _evaluate_course(
        env_cfg,
        agent_cfg,
        checkpoint,
        args_cli.terrain_family,
        args_cli.difficulty_level,
    )
    _print_evaluation_summary(report, report_path)


def _read_termination_outcomes(
    base_env: ManagerBasedRLEnv | DirectRLEnv,
    done_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Read the per-environment outcome masks for the current step."""

    if not isinstance(base_env, ManagerBasedRLEnv):
        raise TypeError("Parkour evaluation outcomes require a ManagerBasedRLEnv.")

    termination_manager = base_env.termination_manager

    def term(name: str) -> torch.Tensor:
        return (
            termination_manager.get_term(name).to(
                device=done_mask.device,
                dtype=torch.bool,
            )
            & done_mask
        )

    trunk_contact = term("trunk_contact")
    fell_below_course = term("fell_below_course")
    return {
        "success": term("success") & (~trunk_contact) & (~fell_below_course),
        "trunk_contact": trunk_contact,
        "fell_below_course": fell_below_course,
        "timeout": term("time_out"),
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


def _training_config_provenance(log_dir: str) -> dict[str, dict[str, str]]:
    """Identify the archived environment and agent configurations."""

    provenance = {}
    for name in ("env.yaml", "agent.yaml"):
        path = os.path.join(log_dir, "params", name)
        if os.path.isfile(path):
            provenance[name] = {"path": path, "sha256": sha256_file(path)}
    return provenance


def _validate_teacher_interface(
    base_env: ManagerBasedRLEnv | DirectRLEnv,
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
    if teacher_checkpoint.teacher_interface_sha256 != teacher_interface_hash:
        print(
            "[WARNING] Evaluation terrain provenance differs from the teacher's training domain; "
            "loading is safe, but the result is out-of-distribution."
        )
    return _InterfaceInfo(teacher_interface, teacher_interface_hash)


def _write_evaluation_report(artifact_dir: str, report: _EvaluationReport) -> str:
    """Serialize one course report as ``metrics.json`` and return its path."""

    metrics_path = os.path.join(artifact_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as metrics_file:
        json.dump(_to_jsonable(report), metrics_file, indent=2, sort_keys=True)
        metrics_file.write("\n")
    return metrics_path


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
