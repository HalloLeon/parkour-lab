"""Load native command definitions with simulator-free SDK command doubles."""

import importlib.util
import sys
from contextlib import contextmanager
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from tests.support import SOURCE_ROOT


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

    command_module, managers = (
        ModuleType("isaaclab.envs.mdp.commands"),
        ModuleType("isaaclab.managers"),
    )
    command_module.UniformVelocityCommand = UniformVelocityCommand
    managers.SceneEntityCfg = lambda name: NS(name=name)
    path = SOURCE_ROOT / "parkour_lab/environments/commands/native.py"
    spec = importlib.util.spec_from_file_location(
        "parkour_lab.environments.commands._commands_test", path
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {"isaaclab.envs.mdp.commands": command_module, "isaaclab.managers": managers},
    ):
        spec.loader.exec_module(module)
        yield module
