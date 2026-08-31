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

cli_args.require_runtime_versions()

from isaaclab.app import AppLauncher


def _run_isolated_course_matrix(cli_arguments: list[str]) -> None:
    """Resolve the configured matrix, then evaluate each cell in a fresh process."""

    script_path = os.path.abspath(__file__)
    evaluation_arguments = [
        argument for argument in cli_arguments if argument != "--all_courses"
    ]
    with tempfile.TemporaryDirectory(
        prefix="parkour_lab_courses_"
    ) as temporary_directory:
        manifest_path = os.path.join(temporary_directory, "courses.json")
        print("[INFO] Resolving the configured evaluation course matrix...", flush=True)
        subprocess.run(
            [
                sys.executable,
                script_path,
                *cli_arguments,
                f"--_course_manifest={manifest_path}",
            ],
            check=True,
        )
        with open(manifest_path, encoding="utf-8") as manifest_file:
            courses = json.load(manifest_file)

    for index, (family, level) in enumerate(courses, start=1):
        print(
            f"[INFO] Evaluating course {index}/{len(courses)}: {family} level {level}",
            flush=True,
        )
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
parser.add_argument(
    "--video", action="store_true", default=False, help="Record an evaluation video."
)
parser.add_argument(
    "--video_length",
    type=cli_args.positive_int,
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
    "--_course_manifest", type=str, default=None, help=argparse.SUPPRESS
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
    "--geometry_variant",
    type=int,
    default=None,
    help="Fixed within-family geometry variant. Defaults to canonical variant zero.",
)
parser.add_argument(
    "--desired_speed",
    type=float,
    default=None,
    help="Fixed desired speed in m/s. Supported environments use their selected course default when omitted.",
)
parser.add_argument(
    "--desired_yaw_rate",
    type=float,
    default=None,
    help="Signed rad/s for deterministic level-0 translate-pivot-translate pulses.",
)
parser.add_argument(
    "--eval_episodes",
    type=cli_args.positive_int,
    default=10,
    help="Number of completed episodes to evaluate.",
)
parser.add_argument(
    "--policy_mode",
    choices=(
        "privileged_mean",
        "privileged_sampled",
        "history_mean",
        "history_sampled",
    ),
    default="privileged_mean",
    help="Teacher action path used for the diagnostic rollout.",
)
parser.add_argument(
    "--reset_profile",
    choices=("canonical", "jitter"),
    default="canonical",
    help="Use the exact reset or isolated narrow initial-state jitter.",
)
parser.add_argument(
    "--num_envs",
    type=cli_args.positive_int,
    default=None,
    help="Number of environments to simulate.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment."
)
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
if args_cli.desired_speed is not None and (
    not math.isfinite(args_cli.desired_speed) or args_cli.desired_speed < 0.0
):
    parser.error("--desired_speed must be finite and non-negative.")
if args_cli.desired_yaw_rate is not None and not math.isfinite(
    args_cli.desired_yaw_rate
):
    parser.error("--desired_yaw_rate must be finite.")
if args_cli.desired_yaw_rate not in (None, 0.0) and args_cli.desired_speed == 0.0:
    parser.error(
        "A nonzero --desired_yaw_rate requires positive --desired_speed or omission."
    )
if args_cli.all_courses and args_cli.desired_yaw_rate not in (None, 0.0):
    parser.error("Nonzero --desired_yaw_rate cannot be combined with --all_courses.")
if args_cli.all_courses and (
    args_cli.terrain_family is not None or args_cli.difficulty_level is not None
):
    parser.error(
        "--all_courses cannot be combined with --terrain_family or --difficulty_level."
    )
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
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from functools import partial

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import parkour_lab.tasks  # noqa: F401
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from parkour_lab.learning.distillation.contracts import (
    assert_teacher_interface_matches,
    build_teacher_interface,
    interface_sha256,
    load_teacher_checkpoint,
    sha256_file,
    terrain_curriculum_matches,
)
from parkour_lab.learning.distillation.teacher.rsl_rl import (
    register_rsl_rl_teacher_actor_critic,
)
from parkour_lab.learning.rsl_rl import RslRlHistoryWrapper
from parkour_lab.tasks.manager_based.parkour_lab.mdp.commands import (
    EVALUATION_PIVOT_WINDOW_DURATION_S,
    PROVISIONAL_ORACLE_RESIDUAL_THRESHOLD_RAD,
    get_preferred_speed,
)
from parkour_lab.tasks.manager_based.parkour_lab.mdp._shared.go2 import GO2_FOOT_NAMES
from parkour_lab.tasks.manager_based.parkour_lab.mdp.diagnostics import (
    latest_evaluation_step,
)
from parkour_lab.tasks.manager_based.parkour_lab.mdp.navigation import route
from parkour_lab.tasks.manager_based.parkour_lab.mdp.navigation.state import (
    TERMINAL_LANDING_PREDICATE_NAMES,
)
from parkour_lab.tasks.manager_based.parkour_lab.parkour_lab_env_cfg import (
    ParkourLabEnvCfg,
)
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

STOP_SETTLED_PLANAR_SPEED_M_S = 0.10
STOP_SETTLED_ABS_YAW_RATE_RAD_S = 0.20
STOP_SETTLED_WITHIN_S = 1.0
STOP_DRIFT_HORIZON_S = 2.0

EPISODE_SUM_METRICS = (
    "forward_speed_m_s",
    "planar_speed_m_s",
    "abs_yaw_rate_rad_s",
    "moving_speed_absolute_error_m_s",
    "stopped_planar_speed_m_s",
    "stopped_abs_yaw_rate_rad_s",
    "pivot_planar_speed_m_s",
    "pivot_yaw_rate_absolute_error_rad_s",
    "pivot_wrong_way",
    "stationary_command",
    "pivot_command",
    "movement_direction_error_rad",
    "movement_direction_valid",
    "absolute_oracle_residual_rad",
    "oracle_residual_threshold_exceedance",
    "moving_command",
    "overspeed_ratio",
    "vertical_velocity_squared_m2_s2",
    "all_feet_airborne",
    "feet_edge_contacts",
    "undesired_body_contacts",
)


_EvaluationSummary = dict[str, object]
_EvaluationReport = dict[str, object]


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

    # Runtime description of teacher observations, preprocessing, actions, and control timing.
    teacher_interface: dict[str, object]

    # Hash of ``teacher_interface`` used to identify its exact contents.
    teacher_interface_sha256: str


@dataclass
class _EpisodeFootGaitState:
    """Per-environment gait buffers in canonical FL/FR/RL/RR order."""

    contact_step_counts: torch.Tensor
    touchdown_counts: torch.Tensor
    current_noncontact_step_counts: torch.Tensor
    max_noncontact_step_counts: torch.Tensor
    world_z_force_sums: torch.Tensor


@dataclass
class _EpisodeRouteCrossTrackState:
    """Bounded post-physics navigation samples retained until episode termination."""

    samples_m: torch.Tensor
    oracle_residual_rad: torch.Tensor
    moving: torch.Tensor
    pivoting: torch.Tensor
    pivot_yaw_rate_error_rad_s: torch.Tensor
    movement_direction_error_rad: torch.Tensor
    movement_direction_valid: torch.Tensor
    waypoint_transition: torch.Tensor
    sample_counts: torch.Tensor


@dataclass
class _EpisodeStopState:
    """Per-environment stop and pivot transition aggregates."""

    previous_moving: torch.Tensor
    previous_pivoting: torch.Tensor
    stop_elapsed_s: torch.Tensor
    settled_elapsed_s: torch.Tensor
    settle_position_xy: torch.Tensor
    current_drift_max_m: torch.Tensor
    stop_window_counts: torch.Tensor
    settled_stop_counts: torch.Tensor
    settled_within_1s_counts: torch.Tensor
    settling_time_sums: torch.Tensor
    settling_time_maxima: torch.Tensor
    drift_2s_counts: torch.Tensor
    drift_2s_sums: torch.Tensor
    drift_2s_maxima: torch.Tensor
    had_restart: torch.Tensor
    pivot_start_position_xy: torch.Tensor
    had_pivot: torch.Tensor
    pivot_excursion_maxima: torch.Tensor
    had_pivot_restart: torch.Tensor


@dataclass
class _RolloutResult:
    """Mutable accumulator for completed-episode statistics."""

    soft_route_half_width_m: float
    hard_route_half_width_m: float
    completed_episodes: int = 0
    return_sum: float = 0.0
    length_steps_sum: int = 0
    success_count: int = 0
    chassis_contact_count: int = 0
    fell_below_course_count: int = 0
    off_route_count: int = 0
    timeout_count: int = 0
    active_timeout_count: int = 0
    wall_only_timeout_count: int = 0
    max_course_progress_m_sum: float = 0.0
    max_waypoints_reached_sum: float = 0.0
    forward_speed_m_s_sum: float = 0.0
    planar_speed_m_s_sum: float = 0.0
    abs_yaw_rate_rad_s_sum: float = 0.0
    moving_speed_absolute_error_m_s_sum: float = 0.0
    stopped_planar_speed_m_s_sum: float = 0.0
    stopped_abs_yaw_rate_rad_s_sum: float = 0.0
    pivot_planar_speed_m_s_sum: float = 0.0
    pivot_yaw_rate_absolute_error_rad_s_sum: float = 0.0
    pivot_wrong_way_sum: float = 0.0
    stationary_command_sum: float = 0.0
    pivot_command_sum: float = 0.0
    pivot_episode_count: int = 0
    pivot_maximum_xy_excursion_m_sum: float = 0.0
    pivot_xy_excursion_m_maximum: float = 0.0
    pivot_restart_episode_count: int = 0
    successful_pivot_restart_episode_count: int = 0
    movement_direction_error_rad_sum: float = 0.0
    movement_direction_valid_sum: float = 0.0
    stop_window_count: int = 0
    settled_stop_count: int = 0
    settled_within_1s_count: int = 0
    settling_time_s_sum: float = 0.0
    settling_time_s_maximum: float = 0.0
    drift_2s_count: int = 0
    drift_2s_m_sum: float = 0.0
    drift_2s_m_maximum: float = 0.0
    restart_episode_count: int = 0
    successful_restart_episode_count: int = 0
    absolute_oracle_residual_rad_sum: float = 0.0
    oracle_residual_threshold_exceedance_sum: float = 0.0
    moving_command_sum: float = 0.0
    overspeed_ratio_sum: float = 0.0
    vertical_velocity_squared_m2_s2_sum: float = 0.0
    all_feet_airborne_sum: float = 0.0
    feet_edge_contacts_sum: float = 0.0
    undesired_body_contacts_sum: float = 0.0
    gait_episode_count: int = 0
    foot_contact_duty_sum: list[float] = field(
        default_factory=lambda: [0.0] * len(GO2_FOOT_NAMES)
    )
    foot_touchdown_count_sum: list[float] = field(
        default_factory=lambda: [0.0] * len(GO2_FOOT_NAMES)
    )
    foot_touchdown_rate_hz_sum: list[float] = field(
        default_factory=lambda: [0.0] * len(GO2_FOOT_NAMES)
    )
    foot_zero_touchdown_episode_count: list[float] = field(
        default_factory=lambda: [0.0] * len(GO2_FOOT_NAMES)
    )
    foot_max_noncontact_duration_s_sum: list[float] = field(
        default_factory=lambda: [0.0] * len(GO2_FOOT_NAMES)
    )
    foot_max_noncontact_duration_s_max: list[float] = field(
        default_factory=lambda: [0.0] * len(GO2_FOOT_NAMES)
    )
    foot_world_z_load_share_sum: list[float] = field(
        default_factory=lambda: [0.0] * len(GO2_FOOT_NAMES)
    )
    world_z_load_valid_episode_count: int = 0
    contact_balance_valid_episode_count: int = 0
    rear_contact_balance_valid_episode_count: int = 0
    rear_world_z_load_balance_valid_episode_count: int = 0
    minimum_foot_world_z_load_share_sum: float = 0.0
    absolute_rear_contact_imbalance_sum: float = 0.0
    absolute_front_rear_contact_imbalance_sum: float = 0.0
    front_minus_rear_contact_imbalance_sum: float = 0.0
    absolute_rear_world_z_load_imbalance_sum: float = 0.0
    absolute_front_rear_world_z_load_imbalance_sum: float = 0.0
    front_minus_rear_world_z_load_imbalance_sum: float = 0.0
    successful_route_cross_track_episode_count: int = 0
    successful_route_cross_track_p50_m_sum: float = 0.0
    successful_route_cross_track_p95_m_sum: float = 0.0
    successful_route_cross_track_maximum_m: float = 0.0
    successful_route_cross_track_soft_exceedance_fraction_sum: float = 0.0
    waypoint_transition_route_cross_track_samples_m: list[float] = field(
        default_factory=list
    )
    oracle_residual_samples_rad: list[float] = field(default_factory=list)
    successful_oracle_residual_samples_rad: list[float] = field(default_factory=list)
    failed_oracle_residual_samples_rad: list[float] = field(default_factory=list)
    waypoint_transition_oracle_residual_samples_rad: list[float] = field(
        default_factory=list
    )
    movement_direction_error_samples_rad: list[float] = field(default_factory=list)
    pivot_yaw_rate_error_samples_rad_s: list[float] = field(default_factory=list)
    terminal_landing_active_sample_count: int = 0
    terminal_landing_active_episode_count: int = 0
    terminal_landing_predicate_pass_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in TERMINAL_LANDING_PREDICATE_NAMES}
    )
    terminal_landing_episode_max_dwell_s_sum: float = 0.0
    terminal_landing_max_dwell_s: float = 0.0

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
        episode_foot_metrics: dict[str, torch.Tensor] | None = None,
        episode_route_cross_track: _EpisodeRouteCrossTrackState | None = None,
        episode_stop_state: _EpisodeStopState | None = None,
        episode_terminal_landing_active_steps: torch.Tensor | None = None,
        episode_terminal_landing_predicate_counts: torch.Tensor | None = None,
        episode_terminal_landing_max_dwell_s: torch.Tensor | None = None,
    ) -> None:
        """Accumulate newly completed episodes, capped at the requested total."""

        remaining = requested_episodes - self.completed_episodes
        completed_indices = torch.nonzero(done_mask, as_tuple=False).flatten()[
            :remaining
        ]
        if completed_indices.numel() == 0:
            return

        self.completed_episodes += int(completed_indices.numel())
        self.return_sum += float(episode_returns[completed_indices].sum().item())
        self.length_steps_sum += int(episode_lengths[completed_indices].sum().item())
        self.success_count += int(outcomes["success"][completed_indices].sum().item())
        self.chassis_contact_count += int(
            outcomes["chassis_contact"][completed_indices].sum().item()
        )
        self.fell_below_course_count += int(
            outcomes["fell_below_course"][completed_indices].sum().item()
        )
        self.off_route_count += int(
            outcomes["off_route"][completed_indices].sum().item()
        )
        self.timeout_count += int(outcomes["timeout"][completed_indices].sum().item())
        if "active_timeout" in outcomes:
            self.active_timeout_count += int(
                outcomes["active_timeout"][completed_indices].sum().item()
            )
            self.wall_only_timeout_count += int(
                outcomes["wall_only_timeout"][completed_indices].sum().item()
            )
        self.max_course_progress_m_sum += float(
            episode_max_course_progress_m[completed_indices].sum().item()
        )
        reached = episode_max_waypoints_reached[completed_indices]
        self.max_waypoints_reached_sum += float(reached.sum().item())
        self.forward_speed_m_s_sum += float(
            episode_metric_sums["forward_speed_m_s"][completed_indices].sum().item()
        )
        scalar_metric_sums = {
            "planar_speed_m_s": "planar_speed_m_s_sum",
            "abs_yaw_rate_rad_s": "abs_yaw_rate_rad_s_sum",
            "moving_speed_absolute_error_m_s": ("moving_speed_absolute_error_m_s_sum"),
            "stopped_planar_speed_m_s": "stopped_planar_speed_m_s_sum",
            "stopped_abs_yaw_rate_rad_s": "stopped_abs_yaw_rate_rad_s_sum",
            "pivot_planar_speed_m_s": "pivot_planar_speed_m_s_sum",
            "pivot_yaw_rate_absolute_error_rad_s": (
                "pivot_yaw_rate_absolute_error_rad_s_sum"
            ),
            "pivot_wrong_way": "pivot_wrong_way_sum",
            "stationary_command": "stationary_command_sum",
            "pivot_command": "pivot_command_sum",
            "movement_direction_error_rad": "movement_direction_error_rad_sum",
            "movement_direction_valid": "movement_direction_valid_sum",
            "absolute_oracle_residual_rad": "absolute_oracle_residual_rad_sum",
            "oracle_residual_threshold_exceedance": (
                "oracle_residual_threshold_exceedance_sum"
            ),
            "moving_command": "moving_command_sum",
        }
        for metric_name, attribute_name in scalar_metric_sums.items():
            if metric_name not in episode_metric_sums:
                continue
            selected_sum = float(
                episode_metric_sums[metric_name][completed_indices].sum().item()
            )
            setattr(self, attribute_name, getattr(self, attribute_name) + selected_sum)
        self.overspeed_ratio_sum += float(
            episode_metric_sums["overspeed_ratio"][completed_indices].sum().item()
        )
        self.vertical_velocity_squared_m2_s2_sum += float(
            episode_metric_sums["vertical_velocity_squared_m2_s2"][completed_indices]
            .sum()
            .item()
        )
        self.all_feet_airborne_sum += float(
            episode_metric_sums["all_feet_airborne"][completed_indices].sum().item()
        )
        self.feet_edge_contacts_sum += float(
            episode_metric_sums["feet_edge_contacts"][completed_indices].sum().item()
        )
        self.undesired_body_contacts_sum += float(
            episode_metric_sums["undesired_body_contacts"][completed_indices]
            .sum()
            .item()
        )
        if episode_foot_metrics is not None:
            self.gait_episode_count += int(completed_indices.numel())
            self._record_completed_foot_metrics(completed_indices, episode_foot_metrics)
        if episode_route_cross_track is not None:
            self._record_completed_navigation(
                completed_indices,
                outcomes["success"],
                episode_route_cross_track,
            )
        if episode_stop_state is not None:
            self._record_completed_stops(
                completed_indices, outcomes["success"], episode_stop_state
            )
        if (
            episode_terminal_landing_active_steps is not None
            and episode_terminal_landing_predicate_counts is not None
            and episode_terminal_landing_max_dwell_s is not None
        ):
            self._record_completed_terminal_landings(
                completed_indices,
                episode_terminal_landing_active_steps,
                episode_terminal_landing_predicate_counts,
                episode_terminal_landing_max_dwell_s,
            )

    def _record_completed_terminal_landings(
        self,
        completed_indices: torch.Tensor,
        active_steps: torch.Tensor,
        predicate_counts: torch.Tensor,
        max_dwell_s: torch.Tensor,
    ) -> None:
        """Aggregate cached terminal-gate evidence for completed episodes."""

        selected_active_steps = active_steps[completed_indices]
        active_episodes = selected_active_steps > 0
        self.terminal_landing_active_sample_count += int(
            selected_active_steps.sum().item()
        )
        active_episode_count = int(active_episodes.sum().item())
        self.terminal_landing_active_episode_count += active_episode_count
        selected_predicates = predicate_counts[completed_indices].sum(dim=0)
        for index, name in enumerate(TERMINAL_LANDING_PREDICATE_NAMES):
            self.terminal_landing_predicate_pass_counts[name] += int(
                selected_predicates[index].item()
            )
        if active_episode_count > 0:
            selected_dwell = max_dwell_s[completed_indices][active_episodes]
            self.terminal_landing_episode_max_dwell_s_sum += float(
                selected_dwell.sum().item()
            )
            self.terminal_landing_max_dwell_s = max(
                self.terminal_landing_max_dwell_s,
                float(selected_dwell.max().item()),
            )

    def _record_completed_foot_metrics(
        self,
        completed_indices: torch.Tensor,
        episode_foot_metrics: dict[str, torch.Tensor],
    ) -> None:
        """Accumulate gait summaries after selecting complete episodes."""

        per_foot_sums = {
            "contact_duty": self.foot_contact_duty_sum,
            "touchdown_count": self.foot_touchdown_count_sum,
            "touchdown_rate_hz": self.foot_touchdown_rate_hz_sum,
            "zero_touchdown": self.foot_zero_touchdown_episode_count,
            "max_noncontact_duration_s": self.foot_max_noncontact_duration_s_sum,
            "world_z_load_share": self.foot_world_z_load_share_sum,
        }
        for metric_name, accumulator in per_foot_sums.items():
            selected = episode_foot_metrics[metric_name][completed_indices]
            for foot_index in range(len(GO2_FOOT_NAMES)):
                accumulator[foot_index] += float(selected[:, foot_index].sum().item())

        selected_max_noncontact = episode_foot_metrics["max_noncontact_duration_s"][
            completed_indices
        ]
        for foot_index in range(len(GO2_FOOT_NAMES)):
            maximum = float(selected_max_noncontact[:, foot_index].max().item())
            self.foot_max_noncontact_duration_s_max[foot_index] = max(
                self.foot_max_noncontact_duration_s_max[foot_index],
                maximum,
            )

        scalar_sums = {
            "minimum_world_z_load_share": "minimum_foot_world_z_load_share_sum",
            "absolute_rear_contact_imbalance": "absolute_rear_contact_imbalance_sum",
            "absolute_front_rear_contact_imbalance": "absolute_front_rear_contact_imbalance_sum",
            "front_minus_rear_contact_imbalance": "front_minus_rear_contact_imbalance_sum",
            "absolute_rear_world_z_load_imbalance": "absolute_rear_world_z_load_imbalance_sum",
            "absolute_front_rear_world_z_load_imbalance": "absolute_front_rear_world_z_load_imbalance_sum",
            "front_minus_rear_world_z_load_imbalance": "front_minus_rear_world_z_load_imbalance_sum",
        }
        for metric_name, attribute_name in scalar_sums.items():
            selected_sum = float(
                episode_foot_metrics[metric_name][completed_indices].sum().item()
            )
            setattr(self, attribute_name, getattr(self, attribute_name) + selected_sum)

        self.contact_balance_valid_episode_count += int(
            episode_foot_metrics["contact_balance_valid"][completed_indices]
            .sum()
            .item()
        )
        self.rear_contact_balance_valid_episode_count += int(
            episode_foot_metrics["rear_contact_balance_valid"][completed_indices]
            .sum()
            .item()
        )
        self.rear_world_z_load_balance_valid_episode_count += int(
            episode_foot_metrics["rear_world_z_load_balance_valid"][completed_indices]
            .sum()
            .item()
        )
        self.world_z_load_valid_episode_count += int(
            episode_foot_metrics["world_z_load_valid"][completed_indices].sum().item()
        )

    def _record_completed_navigation(
        self,
        completed_indices: torch.Tensor,
        success: torch.Tensor,
        state: _EpisodeRouteCrossTrackState,
    ) -> None:
        """Accumulate successful route metrics and raw moving oracle residuals."""

        for env_index in completed_indices.detach().cpu().tolist():
            sample_count = int(state.sample_counts[env_index].item())
            samples = state.samples_m[env_index, :sample_count].detach().cpu().tolist()
            transitions = (
                state.waypoint_transition[env_index, :sample_count]
                .detach()
                .cpu()
                .tolist()
            )
            successful = bool(success[env_index].item())
            if successful:
                p50_m, p95_m, maximum_m, soft_exceedance, transition_samples = (
                    _summarize_route_cross_track_episode(
                        samples,
                        transitions,
                        soft_half_width_m=self.soft_route_half_width_m,
                    )
                )
                self.successful_route_cross_track_episode_count += 1
                self.successful_route_cross_track_p50_m_sum += p50_m
                self.successful_route_cross_track_p95_m_sum += p95_m
                self.successful_route_cross_track_maximum_m = max(
                    self.successful_route_cross_track_maximum_m,
                    maximum_m,
                )
                self.successful_route_cross_track_soft_exceedance_fraction_sum += (
                    soft_exceedance
                )
                self.waypoint_transition_route_cross_track_samples_m.extend(
                    transition_samples
                )

            residuals = (
                state.oracle_residual_rad[env_index, :sample_count]
                .detach()
                .cpu()
                .tolist()
            )
            moving = state.moving[env_index, :sample_count].detach().cpu().tolist()
            moving_residuals = [
                float(value)
                for value, selected in zip(residuals, moving, strict=True)
                if selected
            ]
            self.oracle_residual_samples_rad.extend(moving_residuals)
            outcome_samples = (
                self.successful_oracle_residual_samples_rad
                if successful
                else self.failed_oracle_residual_samples_rad
            )
            outcome_samples.extend(moving_residuals)
            self.waypoint_transition_oracle_residual_samples_rad.extend(
                float(value)
                for value, is_moving, at_transition in zip(
                    residuals, moving, transitions, strict=True
                )
                if is_moving and at_transition
            )
            direction_errors = (
                state.movement_direction_error_rad[env_index, :sample_count]
                .detach()
                .cpu()
                .tolist()
            )
            direction_valid = (
                state.movement_direction_valid[env_index, :sample_count]
                .detach()
                .cpu()
                .tolist()
            )
            self.movement_direction_error_samples_rad.extend(
                float(value)
                for value, valid in zip(direction_errors, direction_valid, strict=True)
                if valid
            )
            pivoting = state.pivoting[env_index, :sample_count].detach().cpu().tolist()
            pivot_errors = (
                state.pivot_yaw_rate_error_rad_s[env_index, :sample_count]
                .detach()
                .cpu()
                .tolist()
            )
            self.pivot_yaw_rate_error_samples_rad_s.extend(
                float(value)
                for value, selected in zip(pivot_errors, pivoting, strict=True)
                if selected
            )

    def _record_completed_stops(
        self,
        completed_indices: torch.Tensor,
        success: torch.Tensor,
        state: _EpisodeStopState,
    ) -> None:
        """Accumulate stop trials only after their containing episode completes."""

        def selected_sum(values: torch.Tensor) -> float:
            return float(values[completed_indices].sum().item())

        self.stop_window_count += int(selected_sum(state.stop_window_counts))
        self.settled_stop_count += int(selected_sum(state.settled_stop_counts))
        self.settled_within_1s_count += int(
            selected_sum(state.settled_within_1s_counts)
        )
        self.settling_time_s_sum += selected_sum(state.settling_time_sums)
        self.drift_2s_count += int(selected_sum(state.drift_2s_counts))
        self.drift_2s_m_sum += selected_sum(state.drift_2s_sums)
        self.settling_time_s_maximum = max(
            self.settling_time_s_maximum,
            float(state.settling_time_maxima[completed_indices].max().item()),
        )
        self.drift_2s_m_maximum = max(
            self.drift_2s_m_maximum,
            float(state.drift_2s_maxima[completed_indices].max().item()),
        )
        restarted = state.had_restart[completed_indices]
        self.restart_episode_count += int(restarted.sum().item())
        self.successful_restart_episode_count += int(
            (restarted & success[completed_indices]).sum().item()
        )
        pivoted = state.had_pivot[completed_indices]
        pivot_count = int(pivoted.sum().item())
        self.pivot_episode_count += pivot_count
        if pivot_count > 0:
            excursions = state.pivot_excursion_maxima[completed_indices][pivoted]
            self.pivot_maximum_xy_excursion_m_sum += float(excursions.sum().item())
            self.pivot_xy_excursion_m_maximum = max(
                self.pivot_xy_excursion_m_maximum,
                float(excursions.max().item()),
            )
        pivot_restarted = state.had_pivot_restart[completed_indices]
        self.pivot_restart_episode_count += int(pivot_restarted.sum().item())
        self.successful_pivot_restart_episode_count += int(
            (pivot_restarted & success[completed_indices]).sum().item()
        )

    def summary(self, step_dt: float) -> _EvaluationSummary:
        """Return aggregate means and rates for the completed episodes."""

        count = self.completed_episodes
        gait_count = self.gait_episode_count
        step_count = self.length_steps_sum
        contact_balance_count = self.contact_balance_valid_episode_count
        rear_contact_balance_count = self.rear_contact_balance_valid_episode_count
        rear_world_z_load_balance_count = (
            self.rear_world_z_load_balance_valid_episode_count
        )
        world_z_load_count = self.world_z_load_valid_episode_count
        moving_step_count = self.moving_command_sum
        stopped_step_count = self.stationary_command_sum
        pivot_step_count = self.pivot_command_sum
        route_count = self.successful_route_cross_track_episode_count

        def ratio(numerator: float | int, denominator: float | int) -> float | None:
            return numerator / denominator if denominator > 0 else None

        mean_length_steps = ratio(self.length_steps_sum, count)
        direction_errors = sorted(self.movement_direction_error_samples_rad)
        direction_p95 = None
        if direction_errors:
            position = 0.95 * (len(direction_errors) - 1)
            lower = int(position)
            weight = position - lower
            direction_p95 = (
                direction_errors[lower] * (1.0 - weight)
                + direction_errors[min(lower + 1, len(direction_errors) - 1)] * weight
            )
        pivot_errors = sorted(self.pivot_yaw_rate_error_samples_rad_s)
        pivot_error_p95 = None
        if pivot_errors:
            position = 0.95 * (len(pivot_errors) - 1)
            lower = int(position)
            weight = position - lower
            pivot_error_p95 = (
                pivot_errors[lower] * (1.0 - weight)
                + pivot_errors[min(lower + 1, len(pivot_errors) - 1)] * weight
            )

        def mean_per_foot(values: list[float], denominator: int) -> dict[str, float]:
            return {
                foot_name: values[foot_index] / denominator
                for foot_index, foot_name in enumerate(GO2_FOOT_NAMES)
            }

        return {
            "success_rate": ratio(self.success_count, count),
            "chassis_contact_rate": ratio(self.chassis_contact_count, count),
            "fell_below_course_rate": ratio(self.fell_below_course_count, count),
            "off_route_rate": ratio(self.off_route_count, count),
            "timeout_rate": ratio(self.timeout_count, count),
            "active_timeout_rate": ratio(self.active_timeout_count, count),
            "wall_only_timeout_rate": ratio(self.wall_only_timeout_count, count),
            "mean_return": ratio(self.return_sum, count),
            "mean_episode_length_steps": mean_length_steps,
            "mean_episode_length_seconds": (
                mean_length_steps * step_dt if mean_length_steps is not None else None
            ),
            "mean_max_course_progress_m": ratio(self.max_course_progress_m_sum, count),
            "mean_max_waypoints_reached": ratio(self.max_waypoints_reached_sum, count),
            "mean_forward_speed_m_s": ratio(self.forward_speed_m_s_sum, step_count),
            "mean_planar_speed_m_s": ratio(self.planar_speed_m_s_sum, step_count),
            "mean_abs_yaw_rate_rad_s": ratio(self.abs_yaw_rate_rad_s_sum, step_count),
            "mean_planar_path_length_m": ratio(
                self.planar_speed_m_s_sum * step_dt, count
            ),
            "mean_moving_speed_absolute_error_m_s": ratio(
                self.moving_speed_absolute_error_m_s_sum, moving_step_count
            ),
            "mean_stopped_planar_speed_m_s": ratio(
                self.stopped_planar_speed_m_s_sum, stopped_step_count
            ),
            "mean_stopped_abs_yaw_rate_rad_s": ratio(
                self.stopped_abs_yaw_rate_rad_s_sum, stopped_step_count
            ),
            "mean_pivot_planar_speed_m_s": ratio(
                self.pivot_planar_speed_m_s_sum, pivot_step_count
            ),
            "mean_pivot_yaw_rate_absolute_error_rad_s": ratio(
                self.pivot_yaw_rate_absolute_error_rad_s_sum, pivot_step_count
            ),
            "p95_pivot_yaw_rate_absolute_error_rad_s": pivot_error_p95,
            "pivot_wrong_way_fraction": ratio(
                self.pivot_wrong_way_sum, pivot_step_count
            ),
            "pivot_command_fraction": ratio(pivot_step_count, step_count),
            "pivot_episode_count": self.pivot_episode_count,
            "mean_pivot_maximum_xy_excursion_m": ratio(
                self.pivot_maximum_xy_excursion_m_sum, self.pivot_episode_count
            ),
            "maximum_pivot_xy_excursion_m": (
                self.pivot_xy_excursion_m_maximum
                if self.pivot_episode_count > 0
                else None
            ),
            "pivot_restart_episode_count": self.pivot_restart_episode_count,
            "pivot_restart_success_fraction": ratio(
                self.successful_pivot_restart_episode_count,
                self.pivot_restart_episode_count,
            ),
            "mean_movement_direction_error_rad": ratio(
                self.movement_direction_error_rad_sum,
                self.movement_direction_valid_sum,
            ),
            "p95_movement_direction_error_rad": direction_p95,
            "stop_window_count": self.stop_window_count,
            "settled_stop_count": self.settled_stop_count,
            "stop_settled_within_1s_fraction": ratio(
                self.settled_within_1s_count, self.stop_window_count
            ),
            "mean_stop_settling_time_s": ratio(
                self.settling_time_s_sum, self.settled_stop_count
            ),
            "maximum_stop_settling_time_s": (
                self.settling_time_s_maximum if self.settled_stop_count > 0 else None
            ),
            "stop_drift_2s_sample_count": self.drift_2s_count,
            "mean_stop_drift_2s_m": ratio(self.drift_2s_m_sum, self.drift_2s_count),
            "maximum_stop_drift_2s_m": (
                self.drift_2s_m_maximum if self.drift_2s_count > 0 else None
            ),
            "restart_episode_count": self.restart_episode_count,
            "restart_success_fraction": ratio(
                self.successful_restart_episode_count, self.restart_episode_count
            ),
            "mean_absolute_oracle_residual_rad": ratio(
                self.absolute_oracle_residual_rad_sum, moving_step_count
            ),
            "oracle_residual_threshold_exceedance_fraction": ratio(
                self.oracle_residual_threshold_exceedance_sum, moving_step_count
            ),
            "moving_command_fraction": ratio(moving_step_count, step_count),
            "mean_overspeed_ratio": ratio(self.overspeed_ratio_sum, step_count),
            "rms_vertical_velocity_m_s": (
                ratio(self.vertical_velocity_squared_m2_s2_sum, step_count) ** 0.5
                if step_count > 0
                else None
            ),
            "all_feet_airborne_fraction": ratio(self.all_feet_airborne_sum, step_count),
            "mean_feet_edge_contacts_per_step": ratio(
                self.feet_edge_contacts_sum, step_count
            ),
            "mean_undesired_body_contacts_per_step": ratio(
                self.undesired_body_contacts_sum, step_count
            ),
            "mean_foot_contact_duty": (
                mean_per_foot(self.foot_contact_duty_sum, gait_count)
                if gait_count > 0
                else None
            ),
            "mean_foot_touchdown_count": (
                mean_per_foot(self.foot_touchdown_count_sum, gait_count)
                if gait_count > 0
                else None
            ),
            "mean_foot_touchdown_rate_hz": (
                mean_per_foot(self.foot_touchdown_rate_hz_sum, gait_count)
                if gait_count > 0
                else None
            ),
            "foot_zero_touchdown_episode_fraction": (
                mean_per_foot(self.foot_zero_touchdown_episode_count, gait_count)
                if gait_count > 0
                else None
            ),
            "mean_foot_max_noncontact_duration_s": (
                mean_per_foot(self.foot_max_noncontact_duration_s_sum, gait_count)
                if gait_count > 0
                else None
            ),
            "maximum_foot_noncontact_duration_s": (
                {
                    foot_name: self.foot_max_noncontact_duration_s_max[foot_index]
                    for foot_index, foot_name in enumerate(GO2_FOOT_NAMES)
                }
                if gait_count > 0
                else None
            ),
            "mean_foot_world_z_load_share": (
                mean_per_foot(self.foot_world_z_load_share_sum, world_z_load_count)
                if world_z_load_count > 0
                else None
            ),
            "mean_minimum_foot_world_z_load_share": (
                self.minimum_foot_world_z_load_share_sum / world_z_load_count
                if world_z_load_count > 0
                else None
            ),
            "mean_absolute_rear_contact_imbalance": (
                self.absolute_rear_contact_imbalance_sum / rear_contact_balance_count
                if rear_contact_balance_count > 0
                else None
            ),
            "mean_absolute_front_rear_contact_imbalance": (
                self.absolute_front_rear_contact_imbalance_sum / contact_balance_count
                if contact_balance_count > 0
                else None
            ),
            "mean_front_minus_rear_contact_imbalance": (
                self.front_minus_rear_contact_imbalance_sum / contact_balance_count
                if contact_balance_count > 0
                else None
            ),
            "mean_absolute_rear_world_z_load_imbalance": (
                self.absolute_rear_world_z_load_imbalance_sum
                / rear_world_z_load_balance_count
                if rear_world_z_load_balance_count > 0
                else None
            ),
            "mean_absolute_front_rear_world_z_load_imbalance": (
                self.absolute_front_rear_world_z_load_imbalance_sum / world_z_load_count
                if world_z_load_count > 0
                else None
            ),
            "mean_front_minus_rear_world_z_load_imbalance": (
                self.front_minus_rear_world_z_load_imbalance_sum / world_z_load_count
                if world_z_load_count > 0
                else None
            ),
            "successful_route_cross_track_episode_count": route_count,
            "mean_successful_episode_route_cross_track_p50_m": (
                self.successful_route_cross_track_p50_m_sum / route_count
                if route_count > 0
                else None
            ),
            "mean_successful_episode_route_cross_track_p95_m": (
                self.successful_route_cross_track_p95_m_sum / route_count
                if route_count > 0
                else None
            ),
            "maximum_successful_route_cross_track_m": (
                self.successful_route_cross_track_maximum_m if route_count > 0 else None
            ),
            "mean_successful_episode_route_cross_track_soft_exceedance_fraction": (
                self.successful_route_cross_track_soft_exceedance_fraction_sum
                / route_count
                if route_count > 0
                else None
            ),
        }

    def route_cross_track_report(self) -> dict[str, object]:
        """Describe enforced widths and successful waypoint-transition samples."""

        return {
            "definition": "root_xy_distance_to_nearest_finite_approved_route_segment",
            "selection": "successful_episodes_only",
            "sampling": "post_physics_pre_reset_including_terminal",
            "quantile_method": "linear",
            "soft_half_width_m": self.soft_route_half_width_m,
            "hard_half_width_m": self.hard_route_half_width_m,
            "training_enforcement": "moving_soft_squared_cost_and_hard_failure",
            "waypoint_transition_samples_m": (
                self.waypoint_transition_route_cross_track_samples_m
            ),
            "waypoint_transition_semantics": "active_waypoint_changed_this_step",
        }

    def terminal_landing_report(self) -> dict[str, object]:
        """Summarize the exact cached predicates used by the completion gate."""

        samples = self.terminal_landing_active_sample_count
        episodes = self.terminal_landing_active_episode_count
        return {
            "sampling": "post_physics_pre_reset_including_terminal",
            "active_sample_count": samples,
            "active_episode_count": episodes,
            "predicate_pass_fractions": {
                name: (
                    self.terminal_landing_predicate_pass_counts[name] / samples
                    if samples > 0
                    else None
                )
                for name in TERMINAL_LANDING_PREDICATE_NAMES
            },
            "mean_episode_max_dwell_s": (
                self.terminal_landing_episode_max_dwell_s_sum / episodes
                if episodes > 0
                else None
            ),
            "max_dwell_s": (
                self.terminal_landing_max_dwell_s if episodes > 0 else None
            ),
        }

    def oracle_residual_report(self) -> dict[str, object]:
        """Describe the unclamped compatibility diagnostic and its raw tails."""

        threshold = PROVISIONAL_ORACLE_RESIDUAL_THRESHOLD_RAD
        return {
            "definition": "wrap(active_waypoint_heading_minus_reference_heading)",
            "sampling": "post_physics_pre_reset_moving_commands_only",
            "stopped_commands_excluded": True,
            "clamping": "none",
            "quantile_method": "linear",
            "provisional_absolute_threshold_rad": threshold,
            "all": _summarize_oracle_residual_samples(
                self.oracle_residual_samples_rad, threshold
            ),
            "successful_episodes": _summarize_oracle_residual_samples(
                self.successful_oracle_residual_samples_rad, threshold
            ),
            "failed_episodes": _summarize_oracle_residual_samples(
                self.failed_oracle_residual_samples_rad, threshold
            ),
            "waypoint_transitions": _summarize_oracle_residual_samples(
                self.waypoint_transition_oracle_residual_samples_rad, threshold
            ),
        }

    @staticmethod
    def stop_response_report() -> dict[str, object]:
        """State evaluator-only stop metric thresholds and eligibility."""

        return {
            "transition": "translation_to_zero_speed_and_zero_yaw_to_translation",
            "pivot_commands_excluded": True,
            "sampling": "post_physics_pre_reset_including_terminal",
            "settled_planar_speed_threshold_m_s": STOP_SETTLED_PLANAR_SPEED_M_S,
            "settled_absolute_yaw_rate_threshold_rad_s": (
                STOP_SETTLED_ABS_YAW_RATE_RAD_S
            ),
            "settling_semantics": "first_sample_strictly_below_both_thresholds",
            "settled_within_s": STOP_SETTLED_WITHIN_S,
            "drift_horizon_after_settling_s": STOP_DRIFT_HORIZON_S,
            "drift_semantics": (
                "maximum_root_xy_excursion_from_settle_position_during_horizon"
            ),
            "restart_success_denominator": (
                "completed_episodes_containing_a_stop_to_move_transition"
            ),
            "pivot_excursion_semantics": (
                "maximum_root_xy_excursion_from_pivot_onset_per_episode"
            ),
            "fixed_pivot_window_duration_s": EVALUATION_PIVOT_WINDOW_DURATION_S,
            "pivot_yaw_rate_error_sampling": (
                "entire_pivot_command_window_including_acquisition"
            ),
            "pivot_restart_success_denominator": (
                "completed_episodes_containing_a_pivot_to_translation_transition"
            ),
            "pivot_restart_success_definition": (
                "eventual_course_success_after_at_least_one_pivot_restart"
            ),
        }


# Capture the task ID and agent entry-point name now. The returned wrapper later
# loads their registered configuration defaults, lets Hydra consume the retained
# ``sys.argv`` overrides, and calls this function as ``main(env_cfg, agent_cfg)``.
@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ParkourLabEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Evaluate a checkpoint on one course or the complete course matrix."""
    _run_requested_action(env_cfg, agent_cfg)


def _run_requested_action(
    env_cfg: ParkourLabEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Write a matrix manifest or evaluate the requested single course."""

    if args_cli._course_manifest is not None:
        _write_course_manifest(env_cfg, args_cli._course_manifest)
        return

    agent_cfg = _apply_cli_overrides(env_cfg, agent_cfg)
    checkpoint = _resolve_checkpoint(agent_cfg)
    _evaluate_requested_course(env_cfg, agent_cfg, checkpoint)


def _evaluate_requested_course(
    env_cfg: ParkourLabEnvCfg,
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


def _evaluate_course(
    env_cfg: ParkourLabEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint: _CheckpointInfo,
    requested_family: str | None,
    requested_level: int | None,
) -> tuple[_EvaluationReport, str]:
    """Evaluate one fixed course, finalize its video, and write its report."""

    (
        evaluation_family,
        evaluation_level,
        geometry_variant,
        desired_speed,
        desired_yaw_rate,
        level_metadata,
    ) = _configure_evaluation_course(
        env_cfg,
        requested_family,
        requested_level,
        args_cli.desired_speed,
        args_cli.desired_yaw_rate,
        args_cli.geometry_variant,
    )
    artifacts = _prepare_evaluation_artifacts(
        checkpoint,
        evaluation_family,
        evaluation_level,
        geometry_variant,
        env_cfg.seed,
        desired_speed=desired_speed,
        desired_yaw_rate=desired_yaw_rate,
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
            checkpoint.sha256,
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
        geometry_variant=geometry_variant,
        desired_speed=desired_speed,
        desired_yaw_rate=desired_yaw_rate,
        level_metadata=level_metadata,
        num_envs=num_envs,
        step_dt=step_dt,
        rollout=rollout,
    )
    report_path = _write_evaluation_report(artifacts.directory, report)
    return report, report_path


def _apply_cli_overrides(
    env_cfg: ParkourLabEnvCfg,
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


def _build_evaluation_report(
    *,
    env_cfg: ParkourLabEnvCfg,
    checkpoint: _CheckpointInfo,
    interface: _InterfaceInfo,
    evaluation_family: str | None,
    evaluation_level: int | None,
    geometry_variant: int | None,
    desired_speed: float | None,
    desired_yaw_rate: float | None,
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
        "policy_mode": args_cli.policy_mode,
        "reset_profile": args_cli.reset_profile,
        "reset_parameters": env_cfg.events.reset_base.params,
        "training_config": _training_config_provenance(checkpoint.log_dir),
        "terrain_family": evaluation_family,
        "difficulty_level": evaluation_level,
        "geometry_variant_index": geometry_variant,
        "desired_speed_m_s": desired_speed,
        "desired_yaw_rate_rad_s": desired_yaw_rate,
        "difficulty_metadata": level_metadata,
        "num_envs": num_envs,
        "gait_foot_order": list(GO2_FOOT_NAMES),
        "gait_world_z_load_definition": (
            "per-foot share of summed abs(world-frame contact-force z); "
            "not terrain-normal support load"
        ),
        "requested_episodes": args_cli.eval_episodes,
        "completed_episodes": rollout.completed_episodes,
        "route_cross_track": rollout.route_cross_track_report(),
        "terminal_landing": rollout.terminal_landing_report(),
        "oracle_residual": rollout.oracle_residual_report(),
        "stop_response": rollout.stop_response_report(),
        "summary": rollout.summary(step_dt),
    }


def _configure_evaluation_course(
    env_cfg: ParkourLabEnvCfg,
    requested_family: str | None,
    requested_level: int | None,
    requested_speed: float | None,
    requested_yaw_rate: float | None,
    requested_geometry_variant: int | None,
) -> tuple[
    str | None,
    int | None,
    int | None,
    float | None,
    float | None,
    dict[str, object],
]:
    """Freeze the config to one course and return its resolved metadata."""

    # None lets the task select its own default family and maximum difficulty
    # after Hydra overrides have been synchronized.
    env_cfg.configure_evaluation(
        requested_family,
        requested_level,
        seed=env_cfg.seed,
        geometry_variant=requested_geometry_variant,
        speed=requested_speed,
        yaw_rate=requested_yaw_rate,
    )
    if args_cli.reset_profile != "canonical":
        env_cfg.set_evaluation_reset_profile(args_cli.reset_profile)
    return (
        env_cfg.evaluation_family,
        env_cfg.evaluation_level,
        env_cfg.evaluation_geometry_variant,
        env_cfg.resolved_evaluation_speed(),
        env_cfg.resolved_evaluation_yaw_rate(),
        env_cfg.evaluation_course_metadata(),
    )


def _prepare_evaluation_artifacts(
    checkpoint: _CheckpointInfo,
    evaluation_family: str | None,
    evaluation_level: int | None,
    geometry_variant: int | None,
    seed: int | None,
    *,
    desired_speed: float | None = None,
    desired_yaw_rate: float | None = None,
) -> _ArtifactInfo:
    """Create one output directory and derive its video filename prefix."""

    family_component = _path_component(evaluation_family, "default")
    level_component = _path_component(evaluation_level, "default")
    variant_component = _path_component(geometry_variant, "default")
    speed_component = _path_component(desired_speed, "default")
    yaw_rate_component = _path_component(desired_yaw_rate, "default")
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
        f"variant_{variant_component}",
        f"speed_{speed_component}",
        f"yaw_rate_{yaw_rate_component}",
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
            f"family_{family_component}-level_{level_component}-variant_{variant_component}-"
            f"speed_{speed_component}-yaw_rate_{yaw_rate_component}-seed_{seed_component}"
        ),
    )


def _create_evaluation_environment(
    env_cfg: ParkourLabEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
    artifacts: _ArtifactInfo,
) -> RslRlHistoryWrapper:
    """Instantiate one course and attach video and RSL-RL wrappers."""

    env_cfg.log_dir = artifacts.directory
    # Instantiate the registered Gym task with the resolved Isaac Lab
    # configuration, requesting rendered RGB frames only when recording video.
    gym_env: gym.Env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )
    if not isinstance(gym_env.unwrapped, ManagerBasedRLEnv):
        gym_env.close()
        raise TypeError("Parkour evaluation requires a ManagerBasedRLEnv.")

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

    return RslRlHistoryWrapper(gym_env, clip_actions=agent_cfg.clip_actions)


def _configured_evaluation_courses(
    env_cfg: ParkourLabEnvCfg,
) -> tuple[tuple[str | None, int | None], ...]:
    """Return every cell in the Hydra-resolved parkour course matrix."""

    curriculum_cfg = env_cfg.parkour_curriculum
    curriculum_cfg.validate_configuration()

    return tuple(
        (family_name, difficulty_level)
        for family_name in curriculum_cfg.family_names
        for difficulty_level in range(curriculum_cfg.num_difficulties)
    )


def _write_course_manifest(
    env_cfg: ParkourLabEnvCfg,
    manifest_path: str,
) -> None:
    """Write the Hydra-resolved course matrix for the process coordinator."""

    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(_configured_evaluation_courses(env_cfg), manifest_file)


def _load_inference_policy(
    env: RslRlHistoryWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint_path: str,
) -> Callable[[TensorDict], torch.Tensor]:
    """Restore an OnPolicyRunner checkpoint and return its inference callable."""

    print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError("play.py supports only OnPolicyRunner teacher checkpoints.")
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
    return (
        history_policy
        if args_cli.policy_mode == "history_mean"
        else partial(policy.act, use_history=True)
    )


def _path_component(value: str | int | None, default: str) -> str:
    """Convert a value to a filesystem-safe path component."""

    text = default if value is None else str(value)
    return "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in text
    )


def _print_evaluation_summary(report: _EvaluationReport, report_path: str) -> None:
    """Print the qualification signals; retain full detail in ``metrics.json``."""

    def metric(value: object, *, rate: bool = False) -> str:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "n/a"
        return f"{100.0 * value:.1f}%" if rate else f"{value:.4f}"

    summary = report["summary"]
    route_report = report["route_cross_track"]
    print("[RESULT] Parkour qualification summary")
    print(
        f"  Course: {report['terrain_family']} level {report['difficulty_level']} "
        f"variant {report['geometry_variant_index']} | policy={report['policy_mode']} "
        f"reset={report['reset_profile']}"
    )
    print(
        f"  Episodes: {report['completed_episodes']}/{report['requested_episodes']} | "
        f"success={metric(summary['success_rate'], rate=True)}"
    )
    print(
        "  Failures (chassis/below/off-route/active-timeout/wall-timeout): "
        f"{metric(summary['chassis_contact_rate'], rate=True)} / "
        f"{metric(summary['fell_below_course_rate'], rate=True)} / "
        f"{metric(summary['off_route_rate'], rate=True)} / "
        f"{metric(summary['active_timeout_rate'], rate=True)} / "
        f"{metric(summary['wall_only_timeout_rate'], rate=True)}"
    )
    print(
        "  Progress (m / waypoints): "
        f"{metric(summary['mean_max_course_progress_m'])} / "
        f"{metric(summary['mean_max_waypoints_reached'])}"
    )
    print(
        "  Command tracking (speed MAE / direction p95 / oracle residual mean): "
        f"{metric(summary['mean_moving_speed_absolute_error_m_s'])} m/s / "
        f"{metric(summary['p95_movement_direction_error_rad'])} rad / "
        f"{metric(summary['mean_absolute_oracle_residual_rad'])} rad"
    )
    print(
        "  Route tail (successful p95 / maximum / soft-width exceedance): "
        f"{metric(summary['mean_successful_episode_route_cross_track_p95_m'])} m / "
        f"{metric(summary['maximum_successful_route_cross_track_m'])} m / "
        f"{metric(summary['mean_successful_episode_route_cross_track_soft_exceedance_fraction'], rate=True)} "
        f"at {metric(route_report['soft_half_width_m'])} m"
    )
    print(
        "  Stop response (settled <=1 s / max drift / restart success): "
        f"{metric(summary['stop_settled_within_1s_fraction'], rate=True)} / "
        f"{metric(summary['maximum_stop_drift_2s_m'])} m / "
        f"{metric(summary['restart_success_fraction'], rate=True)}"
    )
    print(
        "  Pivot response (yaw MAE mean/p95 / wrong-way / max excursion): "
        f"{metric(summary['mean_pivot_yaw_rate_absolute_error_rad_s'])} / "
        f"{metric(summary['p95_pivot_yaw_rate_absolute_error_rad_s'])} rad/s / "
        f"{metric(summary['pivot_wrong_way_fraction'], rate=True)} / "
        f"{metric(summary['maximum_pivot_xy_excursion_m'])} m"
    )
    print(
        "  Safety (airborne / edge contacts / undesired contacts per step): "
        f"{metric(summary['all_feet_airborne_fraction'], rate=True)} / "
        f"{metric(summary['mean_feet_edge_contacts_per_step'])} / "
        f"{metric(summary['mean_undesired_body_contacts_per_step'])}"
    )
    print(f"  Metrics: {report_path}")


def _resolve_checkpoint(agent_cfg: RslRlBaseRunnerCfg) -> _CheckpointInfo:
    """Resolve the checkpoint path and calculate its stable identity."""

    log_root_path = os.path.abspath(
        os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # An explicit checkpoint is a complete path, matching train.py. Without
    # one, ``load_run`` retains RSL-RL's automatic run/checkpoint lookup.
    resume_path = (
        retrieve_file_path(args_cli.checkpoint)
        if args_cli.checkpoint
        else get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
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


# Route cross-track episode telemetry.


def _create_episode_route_cross_track_state(
    base_env: ManagerBasedRLEnv,
) -> _EpisodeRouteCrossTrackState:
    """Allocate bounded per-episode route-distance buffers."""

    sample_shape = (base_env.num_envs, int(base_env.max_episode_length) + 1)
    return _EpisodeRouteCrossTrackState(
        samples_m=torch.zeros(sample_shape, device=base_env.device),
        oracle_residual_rad=torch.zeros(sample_shape, device=base_env.device),
        moving=torch.zeros(sample_shape, device=base_env.device, dtype=torch.bool),
        pivoting=torch.zeros(sample_shape, device=base_env.device, dtype=torch.bool),
        pivot_yaw_rate_error_rad_s=torch.zeros(sample_shape, device=base_env.device),
        movement_direction_error_rad=torch.zeros(sample_shape, device=base_env.device),
        movement_direction_valid=torch.zeros(
            sample_shape, device=base_env.device, dtype=torch.bool
        ),
        waypoint_transition=torch.zeros(
            sample_shape, device=base_env.device, dtype=torch.bool
        ),
        sample_counts=torch.zeros(
            base_env.num_envs, device=base_env.device, dtype=torch.long
        ),
    )


def _reset_episode_route_cross_track(
    state: _EpisodeRouteCrossTrackState,
    done_mask: torch.Tensor,
) -> None:
    """Forget completed rows; stale samples are ignored by sample count."""

    state.sample_counts[done_mask] = 0


def _summarize_route_cross_track_episode(
    samples_m: list[float],
    waypoint_transition: list[bool],
    *,
    soft_half_width_m: float,
) -> tuple[float, float, float, float, list[float]]:
    """Summarize one successful episode without timestep-pooling across runs."""

    ordered = sorted(float(value) for value in samples_m)

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    sample_count = len(ordered)
    return (
        quantile(0.50),
        quantile(0.95),
        ordered[-1],
        sum(value > soft_half_width_m for value in ordered) / sample_count,
        [
            float(value)
            for value, selected in zip(samples_m, waypoint_transition, strict=True)
            if selected
        ],
    )


def _summarize_oracle_residual_samples(
    samples_rad: list[float], threshold_rad: float
) -> dict[str, float | int | None]:
    """Summarize raw signed residuals and absolute-tail compatibility."""

    if not samples_rad:
        return {
            "sample_count": 0,
            "signed_min_rad": None,
            "signed_max_rad": None,
            "absolute_p50_rad": None,
            "absolute_p95_rad": None,
            "absolute_p99_rad": None,
            "absolute_p99_9_rad": None,
            "absolute_max_rad": None,
            "absolute_threshold_exceedance_fraction": None,
        }

    signed = [float(value) for value in samples_rad]
    absolute = sorted(abs(value) for value in signed)

    def quantile(probability: float) -> float:
        position = probability * (len(absolute) - 1)
        lower = int(position)
        upper = min(lower + 1, len(absolute) - 1)
        weight = position - lower
        return absolute[lower] * (1.0 - weight) + absolute[upper] * weight

    return {
        "sample_count": len(signed),
        "signed_min_rad": min(signed),
        "signed_max_rad": max(signed),
        "absolute_p50_rad": quantile(0.50),
        "absolute_p95_rad": quantile(0.95),
        "absolute_p99_rad": quantile(0.99),
        "absolute_p99_9_rad": quantile(0.999),
        "absolute_max_rad": absolute[-1],
        "absolute_threshold_exceedance_fraction": sum(
            value > threshold_rad for value in absolute
        )
        / len(absolute),
    }


def _update_episode_route_cross_track(
    state: _EpisodeRouteCrossTrackState,
    cross_track_error_m: torch.Tensor,
    waypoint_changed: torch.Tensor,
    oracle_residual_rad: torch.Tensor,
    moving: torch.Tensor,
    pivoting: torch.Tensor,
    pivot_yaw_rate_error_rad_s: torch.Tensor,
    movement_direction_error_rad: torch.Tensor,
    movement_direction_valid: torch.Tensor,
) -> None:
    """Capture one bounded post-physics navigation sample."""

    rows = torch.arange(state.sample_counts.shape[0], device=state.sample_counts.device)
    columns = state.sample_counts.clamp_max(state.samples_m.shape[1] - 1)
    state.samples_m[rows, columns] = cross_track_error_m
    state.oracle_residual_rad[rows, columns] = oracle_residual_rad
    state.moving[rows, columns] = moving
    state.pivoting[rows, columns] = pivoting
    state.pivot_yaw_rate_error_rad_s[rows, columns] = pivot_yaw_rate_error_rad_s
    state.movement_direction_error_rad[rows, columns] = movement_direction_error_rad
    state.movement_direction_valid[rows, columns] = movement_direction_valid
    state.waypoint_transition[rows, columns] = waypoint_changed
    state.sample_counts.add_(1).clamp_max_(state.samples_m.shape[1])


# Stop-window episode telemetry.


def _create_episode_stop_state(base_env: ManagerBasedRLEnv) -> _EpisodeStopState:
    """Allocate per-environment state for moving-stop-moving trials."""

    shape = (base_env.num_envs,)

    def zeros() -> torch.Tensor:
        return torch.zeros(shape, device=base_env.device)

    return _EpisodeStopState(
        previous_moving=get_preferred_speed(base_env) > 0.0,
        # Treat an episode-start pivot as a new window so fixed-pivot trials
        # receive the same excursion accounting as sampled mid-episode pivots.
        previous_pivoting=torch.zeros(shape, device=base_env.device, dtype=torch.bool),
        stop_elapsed_s=-torch.ones(shape, device=base_env.device),
        settled_elapsed_s=-torch.ones(shape, device=base_env.device),
        settle_position_xy=torch.zeros((base_env.num_envs, 2), device=base_env.device),
        current_drift_max_m=zeros(),
        stop_window_counts=zeros(),
        settled_stop_counts=zeros(),
        settled_within_1s_counts=zeros(),
        settling_time_sums=zeros(),
        settling_time_maxima=zeros(),
        drift_2s_counts=zeros(),
        drift_2s_sums=zeros(),
        drift_2s_maxima=zeros(),
        had_restart=torch.zeros(shape, device=base_env.device, dtype=torch.bool),
        pivot_start_position_xy=torch.zeros(
            (base_env.num_envs, 2), device=base_env.device
        ),
        had_pivot=torch.zeros(shape, device=base_env.device, dtype=torch.bool),
        pivot_excursion_maxima=zeros(),
        had_pivot_restart=torch.zeros(shape, device=base_env.device, dtype=torch.bool),
    )


def _update_episode_stop_state(
    state: _EpisodeStopState,
    translating: torch.Tensor,
    stationary: torch.Tensor,
    pivoting: torch.Tensor,
    planar_speed_m_s: torch.Tensor,
    abs_yaw_rate_rad_s: torch.Tensor,
    root_position_xy: torch.Tensor,
    step_dt: float,
) -> None:
    """Advance evaluator-only stop settling, drift, and restart state."""

    # Preserve the normal tensors allocated before the inference-only rollout.
    # Rebinding a field here would make the replacement unsafe to reset afterward.
    falling = state.previous_moving & stationary
    previously_active = state.stop_elapsed_s >= 0.0
    rising = (~state.previous_moving) & translating & previously_active
    state.stop_window_counts += falling.to(dtype=state.stop_window_counts.dtype)
    state.stop_elapsed_s.copy_(
        torch.where(
            falling, torch.zeros_like(state.stop_elapsed_s), state.stop_elapsed_s
        )
    )
    state.settled_elapsed_s.copy_(
        torch.where(
            falling,
            -torch.ones_like(state.settled_elapsed_s),
            state.settled_elapsed_s,
        )
    )
    state.current_drift_max_m.copy_(
        torch.where(
            falling,
            torch.zeros_like(state.current_drift_max_m),
            state.current_drift_max_m,
        )
    )

    active = (state.stop_elapsed_s >= 0.0) & stationary
    newly_settled = (
        active
        & (state.settled_elapsed_s < 0.0)
        & (planar_speed_m_s < STOP_SETTLED_PLANAR_SPEED_M_S)
        & (abs_yaw_rate_rad_s < STOP_SETTLED_ABS_YAW_RATE_RAD_S)
    )
    settling_time = state.stop_elapsed_s
    settled = newly_settled.to(dtype=state.settled_stop_counts.dtype)
    state.settled_stop_counts += settled
    state.settled_within_1s_counts += (
        newly_settled & (settling_time <= STOP_SETTLED_WITHIN_S)
    ).to(dtype=state.settled_within_1s_counts.dtype)
    state.settling_time_sums += torch.where(
        newly_settled, settling_time, torch.zeros_like(settling_time)
    )
    state.settling_time_maxima.copy_(
        torch.maximum(
            state.settling_time_maxima,
            torch.where(newly_settled, settling_time, torch.zeros_like(settling_time)),
        )
    )
    state.settle_position_xy.copy_(
        torch.where(
            newly_settled.unsqueeze(-1), root_position_xy, state.settle_position_xy
        )
    )
    state.current_drift_max_m.copy_(
        torch.where(
            newly_settled,
            torch.zeros_like(state.current_drift_max_m),
            state.current_drift_max_m,
        )
    )

    pending_drift = (
        active
        & torch.isfinite(state.settled_elapsed_s)
        & (state.settled_elapsed_s >= 0.0)
    )
    next_settled_elapsed = state.settled_elapsed_s + step_dt
    drift_m = torch.linalg.norm(root_position_xy - state.settle_position_xy, dim=-1)
    state.current_drift_max_m.copy_(
        torch.where(
            pending_drift,
            torch.maximum(state.current_drift_max_m, drift_m),
            state.current_drift_max_m,
        )
    )
    record_drift = pending_drift & (next_settled_elapsed >= STOP_DRIFT_HORIZON_S)
    completed_drift_m = torch.where(
        record_drift,
        state.current_drift_max_m,
        torch.zeros_like(state.current_drift_max_m),
    )
    state.drift_2s_counts += record_drift.to(dtype=state.drift_2s_counts.dtype)
    state.drift_2s_sums += completed_drift_m
    state.drift_2s_maxima.copy_(
        torch.maximum(
            state.drift_2s_maxima,
            completed_drift_m,
        )
    )
    state.settled_elapsed_s.copy_(
        torch.where(
            record_drift,
            torch.full_like(state.settled_elapsed_s, math.inf),
            torch.where(pending_drift, next_settled_elapsed, state.settled_elapsed_s),
        )
    )
    state.settled_elapsed_s.copy_(
        torch.where(
            newly_settled,
            torch.zeros_like(state.settled_elapsed_s),
            state.settled_elapsed_s,
        )
    )
    state.stop_elapsed_s.copy_(
        torch.where(active, state.stop_elapsed_s + step_dt, state.stop_elapsed_s)
    )

    state.had_restart |= rising
    state.stop_elapsed_s.copy_(
        torch.where(
            rising, -torch.ones_like(state.stop_elapsed_s), state.stop_elapsed_s
        )
    )
    state.settled_elapsed_s.copy_(
        torch.where(
            rising, -torch.ones_like(state.settled_elapsed_s), state.settled_elapsed_s
        )
    )
    pivot_rising = (~state.previous_pivoting) & pivoting
    pivot_restart = state.previous_pivoting & translating
    state.had_pivot |= pivot_rising
    state.pivot_start_position_xy.copy_(
        torch.where(
            pivot_rising.unsqueeze(-1),
            root_position_xy,
            state.pivot_start_position_xy,
        )
    )
    pivot_excursion = torch.linalg.norm(
        root_position_xy - state.pivot_start_position_xy, dim=-1
    )
    state.pivot_excursion_maxima.copy_(
        torch.where(
            pivoting | state.previous_pivoting,
            torch.maximum(state.pivot_excursion_maxima, pivot_excursion),
            state.pivot_excursion_maxima,
        )
    )
    state.had_pivot_restart |= pivot_restart
    state.previous_moving.copy_(translating)
    state.previous_pivoting.copy_(pivoting)


def _reset_episode_stop_state(
    state: _EpisodeStopState,
    done_mask: torch.Tensor,
    translating: torch.Tensor,
) -> None:
    """Clear completed stop trials and seed the auto-reset command state."""

    state.previous_moving[done_mask] = translating[done_mask]
    state.previous_pivoting[done_mask] = False
    state.stop_elapsed_s[done_mask] = -1.0
    state.settled_elapsed_s[done_mask] = -1.0
    state.settle_position_xy[done_mask] = 0.0
    state.current_drift_max_m[done_mask] = 0.0
    for values in (
        state.stop_window_counts,
        state.settled_stop_counts,
        state.settled_within_1s_counts,
        state.settling_time_sums,
        state.settling_time_maxima,
        state.drift_2s_counts,
        state.drift_2s_sums,
        state.drift_2s_maxima,
        state.pivot_excursion_maxima,
    ):
        values[done_mask] = 0.0
    state.had_restart[done_mask] = False
    state.pivot_start_position_xy[done_mask] = 0.0
    state.had_pivot[done_mask] = False
    state.had_pivot_restart[done_mask] = False


def _create_episode_foot_gait_state(
    base_env: ManagerBasedRLEnv,
) -> _EpisodeFootGaitState:
    """Allocate canonical per-foot episode buffers."""

    shape = (base_env.num_envs, len(GO2_FOOT_NAMES))
    return _EpisodeFootGaitState(
        contact_step_counts=torch.zeros(
            shape, device=base_env.device, dtype=torch.float32
        ),
        touchdown_counts=torch.zeros(
            shape, device=base_env.device, dtype=torch.float32
        ),
        current_noncontact_step_counts=torch.zeros(
            shape, device=base_env.device, dtype=torch.float32
        ),
        max_noncontact_step_counts=torch.zeros(
            shape, device=base_env.device, dtype=torch.float32
        ),
        world_z_force_sums=torch.zeros(
            shape, device=base_env.device, dtype=torch.float32
        ),
    )


def _episode_foot_gait_metrics(
    *,
    contact_step_counts: torch.Tensor,
    touchdown_counts: torch.Tensor,
    max_noncontact_step_counts: torch.Tensor,
    world_z_force_sums: torch.Tensor,
    episode_lengths: torch.Tensor,
    step_dt: float,
) -> dict[str, torch.Tensor]:
    """Reduce canonical FL/FR/RL/RR episode gait buffers."""

    safe_steps = episode_lengths.to(dtype=torch.float32).clamp_min(1.0).unsqueeze(-1)
    contact_duty = contact_step_counts / safe_steps
    touchdown_rate_hz = touchdown_counts / (safe_steps * step_dt)

    total_world_z_force = world_z_force_sums.sum(dim=-1)
    force_epsilon = torch.finfo(world_z_force_sums.dtype).eps
    world_z_load_valid = total_world_z_force > force_epsilon
    world_z_load_share = torch.where(
        world_z_load_valid.unsqueeze(-1),
        world_z_force_sums / total_world_z_force.clamp_min(force_epsilon).unsqueeze(-1),
        torch.zeros_like(world_z_force_sums),
    )

    total_contact_duty = contact_duty.sum(dim=-1)
    contact_epsilon = torch.finfo(contact_duty.dtype).eps
    contact_balance_valid = total_contact_duty > contact_epsilon
    front_contact = contact_duty[:, 0] + contact_duty[:, 1]
    rear_contact = contact_duty[:, 2] + contact_duty[:, 3]
    rear_contact_balance_valid = rear_contact > contact_epsilon
    front_minus_rear_contact = (front_contact - rear_contact) / (
        total_contact_duty.clamp_min(contact_epsilon)
    )

    front_world_z_load = world_z_load_share[:, 0] + world_z_load_share[:, 1]
    rear_world_z_load = world_z_load_share[:, 2] + world_z_load_share[:, 3]
    rear_world_z_load_balance_valid = rear_world_z_load > force_epsilon
    front_minus_rear_world_z_load = front_world_z_load - rear_world_z_load

    return {
        "contact_duty": contact_duty,
        "touchdown_count": touchdown_counts,
        "touchdown_rate_hz": touchdown_rate_hz,
        "zero_touchdown": (touchdown_counts <= 0.0).to(dtype=torch.float32),
        "max_noncontact_duration_s": max_noncontact_step_counts * step_dt,
        "world_z_load_share": world_z_load_share,
        "minimum_world_z_load_share": world_z_load_share.min(dim=-1).values,
        "absolute_rear_contact_imbalance": torch.where(
            rear_contact_balance_valid,
            torch.abs(contact_duty[:, 2] - contact_duty[:, 3])
            / rear_contact.clamp_min(contact_epsilon),
            torch.zeros_like(total_contact_duty),
        ),
        "absolute_front_rear_contact_imbalance": torch.where(
            contact_balance_valid,
            torch.abs(front_minus_rear_contact),
            torch.zeros_like(total_contact_duty),
        ),
        "front_minus_rear_contact_imbalance": torch.where(
            contact_balance_valid,
            front_minus_rear_contact,
            torch.zeros_like(total_contact_duty),
        ),
        "absolute_rear_world_z_load_imbalance": torch.where(
            rear_world_z_load_balance_valid,
            torch.abs(world_z_load_share[:, 2] - world_z_load_share[:, 3])
            / rear_world_z_load.clamp_min(force_epsilon),
            torch.zeros_like(total_world_z_force),
        ),
        "absolute_front_rear_world_z_load_imbalance": torch.where(
            world_z_load_valid,
            torch.abs(front_minus_rear_world_z_load),
            torch.zeros_like(total_world_z_force),
        ),
        "front_minus_rear_world_z_load_imbalance": torch.where(
            world_z_load_valid,
            front_minus_rear_world_z_load,
            torch.zeros_like(total_world_z_force),
        ),
        "contact_balance_valid": contact_balance_valid.to(dtype=torch.float32),
        "rear_contact_balance_valid": rear_contact_balance_valid.to(
            dtype=torch.float32
        ),
        "rear_world_z_load_balance_valid": rear_world_z_load_balance_valid.to(
            dtype=torch.float32
        ),
        "world_z_load_valid": world_z_load_valid.to(dtype=torch.float32),
    }


def _reset_episode_foot_gait(
    state: _EpisodeFootGaitState, done_mask: torch.Tensor
) -> None:
    """Clear gait history belonging to auto-reset environments."""

    for values in (
        state.contact_step_counts,
        state.touchdown_counts,
        state.current_noncontact_step_counts,
        state.max_noncontact_step_counts,
        state.world_z_force_sums,
    ):
        values[done_mask] = 0.0


def _update_episode_foot_gait(
    state: _EpisodeFootGaitState,
    foot_contact: torch.Tensor,
    foot_touchdown: torch.Tensor,
    foot_world_z_force: torch.Tensor,
) -> None:
    """Accumulate one post-physics contact snapshot."""

    state.contact_step_counts += foot_contact.to(dtype=torch.float32)
    state.touchdown_counts += foot_touchdown.to(dtype=torch.float32)
    current_noncontact_steps = torch.where(
        foot_contact,
        torch.zeros_like(state.current_noncontact_step_counts),
        state.current_noncontact_step_counts + 1.0,
    )
    state.current_noncontact_step_counts.copy_(current_noncontact_steps)
    state.max_noncontact_step_counts.copy_(
        torch.maximum(state.max_noncontact_step_counts, current_noncontact_steps)
    )
    state.world_z_force_sums += foot_world_z_force


# Rollout collection and signal readers.


def _collect_rollout_statistics(
    env: RslRlHistoryWrapper,
    observations: TensorDict,
    policy: Callable[[TensorDict], torch.Tensor],
) -> _RolloutResult:
    """Aggregate completed episodes for the selected policy mode."""

    base_env = env.unwrapped
    step_dt = base_env.step_dt
    episode_returns = torch.zeros(
        env.num_envs, device=base_env.device, dtype=torch.float32
    )
    episode_lengths = torch.zeros(
        env.num_envs, device=base_env.device, dtype=torch.long
    )
    episode_max_waypoints_reached = torch.zeros(
        env.num_envs,
        device=base_env.device,
        dtype=torch.long,
    )
    episode_metric_sums = {
        name: torch.zeros_like(episode_returns) for name in EPISODE_SUM_METRICS
    }
    episode_foot_gait = _create_episode_foot_gait_state(base_env)
    episode_route_cross_track = _create_episode_route_cross_track_state(base_env)
    episode_stop_state = _create_episode_stop_state(base_env)
    curriculum = base_env.cfg.parkour_curriculum
    rollout = _RolloutResult(
        soft_route_half_width_m=float(curriculum.soft_route_half_width_m),
        hard_route_half_width_m=float(curriculum.hard_route_half_width_m),
    )

    while (
        simulation_app.is_running()
        and rollout.completed_episodes < args_cli.eval_episodes
    ):
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(observations)
            observations, rewards, dones, _ = env.step(actions)
            # The reward-phase diagnostic retains terminal physics before
            # Isaac Lab auto-resets completed environments.
            transition = latest_evaluation_step(env.unwrapped)
            step_metrics = transition.metrics
            moving = step_metrics["moving_command"].to(dtype=torch.bool)
            stationary = step_metrics["stationary_command"].to(dtype=torch.bool)
            pivoting = step_metrics["pivot_command"].to(dtype=torch.bool)
            _update_episode_stop_state(
                episode_stop_state,
                moving,
                stationary,
                pivoting,
                step_metrics["planar_speed_m_s"],
                step_metrics["abs_yaw_rate_rad_s"],
                transition.root_position_xy,
                step_dt,
            )
            _update_episode_foot_gait(
                episode_foot_gait,
                transition.foot_contact,
                transition.foot_touchdown,
                transition.foot_world_z_force,
            )
            _update_episode_route_cross_track(
                episode_route_cross_track,
                transition.route_cross_track_error_m,
                transition.waypoint_changed,
                step_metrics["oracle_residual_rad"],
                moving,
                pivoting,
                step_metrics["pivot_yaw_rate_absolute_error_rad_s"],
                step_metrics["movement_direction_error_rad"],
                step_metrics["movement_direction_valid"].to(dtype=torch.bool),
            )
        rewards = rewards.reshape(-1).to(device=episode_returns.device)
        dones = dones.reshape(-1).to(device=episode_returns.device)
        done_mask = dones.to(dtype=torch.bool)
        episode_returns += rewards
        episode_lengths += 1
        for name, episode_sum in episode_metric_sums.items():
            episode_sum += step_metrics[name]
        outcomes = _read_termination_outcomes(base_env, done_mask)
        # The cursor counts prior targets; success adds its still-active final target.
        episode_max_waypoints_reached = torch.maximum(
            episode_max_waypoints_reached,
            transition.active_waypoint_indices
            + outcomes["success"].to(dtype=transition.active_waypoint_indices.dtype),
        )
        episode_max_course_progress_m = route.last_episode_max_course_progress_m(
            base_env
        )
        episode_foot_metrics = _episode_foot_gait_metrics(
            contact_step_counts=episode_foot_gait.contact_step_counts,
            touchdown_counts=episode_foot_gait.touchdown_counts,
            max_noncontact_step_counts=episode_foot_gait.max_noncontact_step_counts,
            world_z_force_sums=episode_foot_gait.world_z_force_sums,
            episode_lengths=episode_lengths,
            step_dt=step_dt,
        )
        rollout.record_completed(
            args_cli.eval_episodes,
            done_mask,
            episode_returns,
            episode_lengths,
            outcomes,
            episode_max_course_progress_m,
            episode_max_waypoints_reached,
            episode_metric_sums,
            episode_foot_metrics,
            episode_route_cross_track,
            episode_stop_state,
            transition.terminal_landing_active_step_count,
            transition.terminal_landing_predicate_pass_count,
            transition.terminal_landing_max_dwell_s,
        )
        episode_returns[done_mask] = 0.0
        episode_lengths[done_mask] = 0
        episode_max_waypoints_reached[done_mask] = 0
        for values in episode_metric_sums.values():
            values[done_mask] = 0.0
        _reset_episode_foot_gait(episode_foot_gait, done_mask)
        _reset_episode_route_cross_track(episode_route_cross_track, done_mask)
        _reset_episode_stop_state(
            episode_stop_state,
            done_mask,
            get_preferred_speed(base_env) > 0.0,
        )
        sleep_time = step_dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    return rollout


def _read_termination_outcomes(
    base_env: ManagerBasedRLEnv,
    done_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Read the per-environment outcome masks for the current step."""

    termination_manager = base_env.termination_manager

    def term(name: str) -> torch.Tensor:
        return (
            termination_manager.get_term(name).to(
                device=done_mask.device,
                dtype=torch.bool,
            )
            & done_mask
        )

    chassis_contact = term("chassis_contact")
    fell_below_course = term("fell_below_course") & (~chassis_contact)
    off_route = term("off_route") & (~chassis_contact) & (~fell_below_course)
    success = term("success") & (~chassis_contact) & (~fell_below_course) & (~off_route)
    remaining = (~success) & (~chassis_contact) & (~fell_below_course) & (~off_route)
    active_timeout = term("time_out") & remaining
    wall_only_timeout = term("wall_time_out") & remaining & (~active_timeout)
    return {
        "success": success,
        "chassis_contact": chassis_contact,
        "fell_below_course": fell_below_course,
        "off_route": off_route,
        "active_timeout": active_timeout,
        "wall_only_timeout": wall_only_timeout,
        "timeout": active_timeout | wall_only_timeout,
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
    base_env: ManagerBasedRLEnv,
    observations: TensorDict,
    agent_cfg: RslRlBaseRunnerCfg,
    checkpoint_path: str,
    checkpoint_sha256: str,
) -> _InterfaceInfo:
    """Rebuild and validate the exact checkpoint actor interface."""

    teacher_checkpoint = load_teacher_checkpoint(
        checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    teacher_interface = build_teacher_interface(base_env, observations, agent_cfg)
    teacher_interface_hash = interface_sha256(teacher_interface)
    assert_teacher_interface_matches(
        teacher_checkpoint.teacher_interface,
        teacher_interface,
        context="Fixed-evaluation runtime",
    )
    if not terrain_curriculum_matches(
        teacher_checkpoint.teacher_interface,
        teacher_interface,
    ):
        print(
            "[WARNING] Evaluation terrain/curriculum provenance differs from the teacher's training domain; "
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
