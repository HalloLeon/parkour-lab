# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with a random-action agent."""

# Launch Isaac Sim before importing modules that depend on it.

import argparse

from isaaclab.app import AppLauncher

# Define script-specific command-line arguments.
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# Add Isaac Lab application arguments.
AppLauncher.add_app_launcher_args(parser)
# Parse all command-line arguments.
args_cli = parser.parse_args()

# Launch the Omniverse application.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# The remaining imports require the running simulation application.

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import parkour_lab.tasks  # noqa: F401
import torch
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    """Random actions agent with Isaac Lab environment."""

    # Resolve the selected task's environment configuration.
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # Instantiate the registered Gym environment.
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Report the vectorized observation and action spaces.
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Initialize every environment before simulation begins.
    env.reset()

    # Step the environments until the simulation application stops.
    while simulation_app.is_running():
        # Disable gradient tracking because this script does not train a model.
        with torch.inference_mode():
            # Sample each action component uniformly from [-1, 1].
            actions = (
                2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
            )
            env.step(actions)

    # Release environment resources.
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
