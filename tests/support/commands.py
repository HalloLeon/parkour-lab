"""Load native command definitions with simulator-free SDK command doubles."""

import sys
from contextlib import contextmanager
from types import ModuleType
from unittest.mock import patch

import torch


@contextmanager
def native_commands():
    class UniformVelocityCommand:
        def __init__(self, cfg, env):
            self.cfg, self._env = cfg, env
            self.num_envs, self.device = env.num_envs, env.device
            self.command_counter = torch.zeros(env.num_envs, dtype=torch.int64)
            self.vel_command_b = torch.zeros(env.num_envs, 3)
            self.is_heading_env = torch.zeros(env.num_envs, dtype=torch.bool)
            self.is_standing_env = torch.zeros(env.num_envs, dtype=torch.bool)
            self.time_left = torch.zeros(env.num_envs)

        def _update_metrics(self):
            pass

        def _update_command(self):
            pass

    command_module = ModuleType("isaaclab.envs.mdp.commands")
    command_module.UniformVelocityCommand = UniformVelocityCommand
    with patch.dict(
        sys.modules,
        {"isaaclab.envs.mdp.commands": command_module},
    ):
        yield command_module
