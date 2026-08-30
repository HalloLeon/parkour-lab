# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

from parkour_lab import runtime_versions

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg


DOMAIN_RANDOMIZATION_STAGES = ("off", "narrow", "wide")
REQUIRED_RUNTIME_VERSIONS = runtime_versions.REQUIRED_RUNTIME_VERSIONS
require_runtime_versions = runtime_versions.require_runtime_versions


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def add_rsl_rl_args(parser: argparse.ArgumentParser) -> None:
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # Group RSL-RL options in the generated help text.
    arg_group = parser.add_argument_group(
        "rsl_rl", description="Arguments for RSL-RL agent."
    )
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
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable checkpoint resume, overriding the configured value.",
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


def add_domain_randomization_args(parser: argparse.ArgumentParser) -> None:
    """Add the staged domain-randomization option to a script parser."""

    arg_group = parser.add_argument_group(
        "domain_randomization",
        description="Arguments for staged environment domain randomization.",
    )
    arg_group.add_argument(
        "--domain_randomization_stage",
        choices=DOMAIN_RANDOMIZATION_STAGES,
        default=None,
        help=(
            "Select the off, narrow, or wide parkour domain-randomization stage. "
            "Overrides env.domain_randomization.stage when provided."
        ),
    )


def add_rsl_rl_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Add only the checkpoint-selection arguments needed for evaluation."""

    arg_group = parser.add_argument_group(
        "rsl_rl_checkpoint",
        description="Arguments for selecting an RSL-RL checkpoint.",
    )
    _add_rsl_rl_checkpoint_args(arg_group)


def apply_domain_randomization_stage(
    env_cfg: object,
    args_cli: argparse.Namespace,
) -> None:
    """Apply the optional CLI stage to the parkour environment."""

    if args_cli.domain_randomization_stage is not None:
        env_cfg.domain_randomization.stage = args_cli.domain_randomization_stage


def update_rsl_rl_cfg(
    agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace
) -> RslRlBaseRunnerCfg:
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
    # ``checkpoint`` is a complete runtime path owned by train.py/play.py. Do
    # not copy it into RSL-RL's run-local ``load_checkpoint`` pattern.
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

    selectors = arg_group.add_mutually_exclusive_group()
    selectors.add_argument(
        "--load_run",
        type=str,
        default=None,
        help="Run-folder name or pattern used for automatic checkpoint lookup.",
    )
    selectors.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Complete path of the checkpoint to load.",
    )
