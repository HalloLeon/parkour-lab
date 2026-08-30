# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Register Parkour Lab's Gym environments on explicit import."""

import gymnasium as gym

_TASK_PACKAGE = f"{__package__}.manager_based.parkour_lab"

# Registry values are lazy ``module.path:ClassName`` references stored in the
# task's ``gym.EnvSpec.kwargs``. Isaac Lab resolves them when the task is loaded:
#
# - ``env_cfg_entry_point`` selects the environment configuration.
# - ``rsl_rl_cfg_entry_point`` selects the default terrain-aware PPO teacher.
# - ``rsl_rl_privileged_critic_cfg_entry_point`` selects the critic-only terrain
#   ablation.
# - ``rsl_rl_baseline_cfg_entry_point`` selects the no-terrain baseline.
#
# The task ID chooses the training or playback environment. The scripts' optional
# ``--agent`` argument chooses an RSL-RL entry (the default is
# ``rsl_rl_cfg_entry_point``); Hydra and CLI overrides are applied afterward.

# Training environment.
if "Parkour-Lab-v0" not in gym.registry:
    gym.register(
        id="Parkour-Lab-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{_TASK_PACKAGE}.parkour_lab_env_cfg:ParkourLabEnvCfg"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{_TASK_PACKAGE}.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
            ),
            "rsl_rl_privileged_critic_cfg_entry_point": (
                f"{_TASK_PACKAGE}.agents.rsl_rl_ppo_cfg:PPOPrivilegedCriticRunnerCfg"
            ),
            "rsl_rl_baseline_cfg_entry_point": (
                f"{_TASK_PACKAGE}.agents.rsl_rl_ppo_cfg:PPOBaselineRunnerCfg"
            ),
        },
    )

# Playback environment.
if "Parkour-Lab-Play-v0" not in gym.registry:
    gym.register(
        id="Parkour-Lab-Play-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": (
                f"{_TASK_PACKAGE}.parkour_lab_env_cfg:ParkourLabEnvCfgPlay"
            ),
            "rsl_rl_cfg_entry_point": (
                f"{_TASK_PACKAGE}.agents.rsl_rl_ppo_cfg:PPORunnerCfg"
            ),
            "rsl_rl_privileged_critic_cfg_entry_point": (
                f"{_TASK_PACKAGE}.agents.rsl_rl_ppo_cfg:PPOPrivilegedCriticRunnerCfg"
            ),
            "rsl_rl_baseline_cfg_entry_point": (
                f"{_TASK_PACKAGE}.agents.rsl_rl_ppo_cfg:PPOBaselineRunnerCfg"
            ),
        },
    )
