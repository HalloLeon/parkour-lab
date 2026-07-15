# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # Group RSL-RL options in the generated help text.
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # Experiment arguments.
    arg_group.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment folder where logs will be stored.",
    )
    arg_group.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name suffix to the log directory.",
    )
    # Checkpoint-loading arguments.
    arg_group.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Whether to resume from a checkpoint.",
    )
    _add_rsl_rl_checkpoint_args(arg_group)
    # Logger arguments.
    arg_group.add_argument(
        "--logger",
        type=str,
        default=None,
        choices={"wandb", "tensorboard", "neptune"},
        help="Logger module to use.",
    )
    arg_group.add_argument(
        "--log_project_name",
        type=str,
        default=None,
        help="Name of the logging project when using wandb or neptune.",
    )


def add_rsl_rl_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Add only the checkpoint-selection arguments needed for evaluation."""

    arg_group = parser.add_argument_group(
        "rsl_rl_checkpoint",
        description="Arguments for selecting an RSL-RL checkpoint.",
    )
    _add_rsl_rl_checkpoint_args(arg_group)


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # Apply only options owned by the RSL-RL runner configuration. The calling
    # scripts handle environment, simulator, video, and Hydra options because
    # those values belong to other configuration objects or runtime setup.
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # Sample a seed when ``-1`` requests nondeterministic selection.
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    experiment_name = getattr(args_cli, "experiment_name", None)
    if experiment_name is not None:
        agent_cfg.experiment_name = experiment_name
    resume = getattr(args_cli, "resume", None)
    if resume is not None:
        agent_cfg.resume = resume
    load_run = getattr(args_cli, "load_run", None)
    if load_run is not None:
        agent_cfg.load_run = load_run
    checkpoint = getattr(args_cli, "checkpoint", None)
    if checkpoint is not None:
        agent_cfg.load_checkpoint = checkpoint
    run_name = getattr(args_cli, "run_name", None)
    if run_name is not None:
        agent_cfg.run_name = run_name
    logger = getattr(args_cli, "logger", None)
    if logger is not None:
        agent_cfg.logger = logger
    # Use one project name for either supported remote logger.
    log_project_name = getattr(args_cli, "log_project_name", None)
    if agent_cfg.logger in {"wandb", "neptune"} and log_project_name:
        agent_cfg.wandb_project = log_project_name
        agent_cfg.neptune_project = log_project_name

    return agent_cfg


def _add_rsl_rl_checkpoint_args(arg_group: argparse._ArgumentGroup) -> None:
    """Register checkpoint arguments on an existing parser group."""

    arg_group.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Name or regular-expression pattern of the run folder to load.",
    )
    arg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for playback, or run-local filename/pattern when resuming training.",
    )
