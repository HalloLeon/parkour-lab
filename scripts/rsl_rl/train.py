# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

# Launch Isaac Sim before importing modules that depend on it.

import argparse
import os
import subprocess
import sys

import cli_args

cli_args.require_runtime_versions()

from isaaclab.app import AppLauncher


def _require_tracked_training_sources() -> None:
    """Reject source files that RSL-RL's tracked-file diff cannot archive."""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        repo_root = subprocess.check_output(
            ["git", "-C", script_dir, "rev-parse", "--show-toplevel"],
            text=True,
        ).strip()
        output = subprocess.check_output(
            [
                "git",
                "-C",
                repo_root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "scripts",
                "source",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Training provenance requires this project to be in a Git worktree."
        ) from exc

    untracked = output.splitlines()
    if untracked:
        paths = "\n  ".join(untracked)
        raise RuntimeError(
            "RSL-RL cannot archive untracked training source. Stage or commit these files before training:\n  "
            + paths
        )


parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=cli_args.positive_int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=cli_args.positive_int,
    default=2000,
    help="Interval between video recordings (in steps).",
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
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--max_iterations",
    type=cli_args.positive_int,
    default=None,
    help="RL Policy training iterations.",
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
# Add RSL-RL command-line arguments.
cli_args.add_rsl_rl_args(parser)
# Add the explicit staged domain-randomization selector.
cli_args.add_domain_randomization_args(parser)
# Add Isaac Lab application arguments.
AppLauncher.add_app_launcher_args(parser)
# Parse this script's known options into ``args_cli`` and retain unrecognized
# configuration overrides, such as ``env.decimation=8``, in ``hydra_args``.
args_cli, hydra_args = parser.parse_known_args()
_require_tracked_training_sources()
if os.environ.get("WORLD_SIZE", "1") != "1":
    parser.error("Distributed training is not supported; run one training process.")

# Enable cameras when recording video.
if args_cli.video:
    args_cli.enable_cameras = True

# Replace the process-wide argument list with only the script name and Hydra
# overrides. When the decorated ``main()`` is called later, Isaac Lab's wrapper
# invokes ``hydra.main()``, which reads these overrides from ``sys.argv`` and
# applies them to the environment and agent configurations.
sys.argv = [sys.argv[0]] + hydra_args

# Launch the Omniverse application.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# The remaining imports require the running simulation application.

import json
from datetime import datetime, timezone

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import parkour_lab.tasks  # noqa: F401
import torch
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from parkour_lab.learning.distillation.contracts import (
    assert_teacher_interface_matches,
    build_teacher_interface,
    interface_sha256,
    load_teacher_checkpoint,
    terrain_curriculum_matches,
    write_json,
)
from parkour_lab.learning.distillation.teacher.rsl_rl import (
    register_rsl_rl_teacher_actor_critic,
)
from parkour_lab.learning.rsl_rl import RslRlHistoryWrapper
from parkour_lab.tasks.manager_based.parkour_lab.parkour_lab_env_cfg import (
    ParkourLabEnvCfg,
)
from rsl_rl.runners import OnPolicyRunner

# Use faster TF32 arithmetic for float32 matrix multiplications on supported
# NVIDIA GPUs, at the cost of some numerical precision.
torch.backends.cuda.matmul.allow_tf32 = True

# Likewise, permit cuDNN operations such as convolutions to use TF32.
torch.backends.cudnn.allow_tf32 = True

# Allow cuDNN to use faster algorithms that may not reproduce bit-identical
# results between otherwise identical runs.
torch.backends.cudnn.deterministic = False

# Do not benchmark several cuDNN algorithms at runtime to select the fastest
# one for each input shape.
torch.backends.cudnn.benchmark = False

_CURRICULUM_STATE_KEY = "parkour_curriculum"


# Training setup helpers.


def _print_runtime_layout(env: object) -> None:
    """Record resolved timing and semantic body/action order in one line."""

    action_term = env.action_manager.get_term("joint_pos")
    layout = {
        "action_joint_order": list(action_term.IO_descriptor.joint_names),
        "control_dt_s": env.step_dt,
        "feet_contact_body_order": list(env.scene["feet_contact"].body_names),
        "physics_dt_s": env.physics_dt,
        "robot_body_order": list(env.scene["robot"].body_names),
    }
    print(f"[INFO] Parkour runtime layout: {json.dumps(layout, sort_keys=True)}")


# Curriculum checkpoint integration.


def _parkour_curriculum_term(env: object):
    """Return the environment's stateful parkour curriculum term."""

    return env.curriculum_manager.cfg.terrain_levels.func


class ParkourOnPolicyRunner(OnPolicyRunner):
    """Store adaptive parkour curriculum memory with ordinary PPO checkpoints."""

    def save(self, path: str, infos: dict | None = None) -> None:
        checkpoint_infos = dict(infos or {})
        term = _parkour_curriculum_term(self.env.unwrapped)
        checkpoint_infos[_CURRICULUM_STATE_KEY] = term.state_dict()
        super().save(path, checkpoint_infos)

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Restore a checkpoint and RSL-RL's Python-side adaptive LR state."""

        infos = super().load(
            path,
            load_optimizer=load_optimizer,
            map_location=map_location,
        )
        if load_optimizer:
            self.alg.learning_rate = self.alg.optimizer.param_groups[0]["lr"]
        return infos


def _restore_parkour_curriculum(env: object, infos: object) -> None:
    """Restore curriculum memory and begin fresh episodes at that frontier."""

    term = _parkour_curriculum_term(env.unwrapped)
    if not isinstance(infos, dict) or _CURRICULUM_STATE_KEY not in infos:
        raise ValueError("Checkpoint has no parkour curriculum state.")
    state = infos[_CURRICULUM_STATE_KEY]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint parkour curriculum state must be a dictionary.")
    term.load_state_dict(state)
    env.reset()


# Capture the task ID and agent entry-point name now. The returned wrapper later
# loads their registered configuration defaults, lets Hydra consume the retained
# ``sys.argv`` overrides, and calls this function as ``main(env_cfg, agent_cfg)``.
@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ParkourLabEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
) -> None:
    """Train with RSL-RL agent."""
    # Apply command-line overrides that are not handled by Hydra.
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if (
        args_cli.checkpoint is not None or args_cli.load_run is not None
    ) and not agent_cfg.resume:
        raise ValueError("--checkpoint and --load_run require --resume.")
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    agent_cfg.max_iterations = (
        args_cli.max_iterations
        if args_cli.max_iterations is not None
        else agent_cfg.max_iterations
    )
    # Apply this after Hydra so an explicit CLI stage has final precedence.
    cli_args.apply_domain_randomization_stage(env_cfg, args_cli)
    env_cfg.synchronize_curriculum_config()
    env_cfg.synchronize_domain_randomization_config()

    # Set the seed before constructing the environment because initialization
    # may randomize state.
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )
    # Build the experiment and run directories.
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # Build a readable UTC identifier as ``run_YYYYMMDD_HHMMSS``: ``%Y`` is
    # the year, ``%m`` the month, ``%d`` the day, ``%H`` the hour, ``%M`` the
    # minute, and ``%S`` the second.
    log_dir = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    # Ray Tune extracts the experiment name from this exact logging line.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Configure optional environment-interface export.
    env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    env_cfg.io_descriptors_output_dir = log_dir

    # Make the run directory available to the environment.
    env_cfg.log_dir = log_dir

    # Create the Isaac Lab environment.
    gym_env: gym.Env = gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    # Resolve the checkpoint selected for resuming PPO training.
    if agent_cfg.resume:
        # An explicit checkpoint is a complete path, matching play.py. Without
        # one, ``load_run`` retains RSL-RL's automatic run/checkpoint lookup.
        resume_path = (
            retrieve_file_path(args_cli.checkpoint)
            if args_cli.checkpoint
            else get_checkpoint_path(
                log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
            )
        )
    # Add video recording before the final RSL-RL wrapper.
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        gym_env = gym.wrappers.RecordVideo(gym_env, **video_kwargs)

    # Adapt the Isaac Lab environment to the RSL-RL vector interface.
    env = RslRlHistoryWrapper(
        gym_env,
        clip_actions=agent_cfg.clip_actions,
    )
    _print_runtime_layout(env.unwrapped)

    # The compact interface carries only checkpoint semantics and a terrain
    # hash; the resolved env/agent YAML files below archive the full domain.
    teacher_interface = build_teacher_interface(
        env.unwrapped, env.get_observations(), agent_cfg
    )
    write_json(
        os.path.join(log_dir, "params", "teacher_interface.json"),
        {
            "teacher_interface": teacher_interface,
            "teacher_interface_sha256": interface_sha256(teacher_interface),
        },
    )

    resume_terrain_matches = True
    if agent_cfg.resume:
        resume_checkpoint = load_teacher_checkpoint(resume_path)
        assert_teacher_interface_matches(
            resume_checkpoint.teacher_interface,
            teacher_interface,
            context="PPO resume runtime",
        )
        resume_terrain_matches = terrain_curriculum_matches(
            resume_checkpoint.teacher_interface,
            teacher_interface,
        )

    if agent_cfg.class_name != "OnPolicyRunner":
        raise ValueError("train.py supports only OnPolicyRunner.")
    register_rsl_rl_teacher_actor_critic()
    runner = ParkourOnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device
    )
    runner.add_git_repo_to_log(__file__)
    # Load the selected checkpoint when continuing an existing run.
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        checkpoint_infos = runner.load(resume_path)
        if resume_terrain_matches:
            _restore_parkour_curriculum(env, checkpoint_infos)
            print("[INFO] Restored adaptive parkour curriculum state.")
        else:
            print(
                "[WARNING] Checkpoint curriculum belongs to a different terrain; "
                "using the configured bootstrap rows."
            )
    # Save the resolved configurations with the checkpoints.
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    try:
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            # Keep every environment's initial episode counter at zero. If this option
            # were ``True``, RSL-RL would randomize those counters so the first batch of
            # environments timed out at different, artificially shortened lengths.
            # Full first episodes keep those early timeouts meaningful to the curriculum.
            init_at_random_ep_len=False,
        )
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
