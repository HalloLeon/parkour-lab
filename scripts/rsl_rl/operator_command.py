"""Isaac Lab binding for the operator curriculum. Import after AppLauncher."""

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand

try:
    from .operator_curriculum import OperatorCommandSampler, VERSION
except ImportError:
    from operator_curriculum import OperatorCommandSampler, VERSION


class OperatorVelocityCommand(UniformVelocityCommand):
    """Retain the stock 3-D command interface, changing only its sampling law."""

    def __init__(self, cfg, env):
        if cfg.heading_command or cfg.rel_heading_envs or cfg.rel_standing_envs:
            raise ValueError(
                "Operator modes own exact stops and yaw; disable stock overrides"
            )
        super().__init__(cfg, env)
        self.sampler = OperatorCommandSampler(self.device)
        self.category = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        category, command, duration = self.sampler.sample(len(env_ids))
        self.category[env_ids] = category
        self.vel_command_b[env_ids] = command
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = category == 0
        # CommandTerm._resample sets its default timer BEFORE this hook.
        self.time_left[env_ids] = duration

    def __str__(self):
        return f"OperatorVelocityCommand ({VERSION}): body [vx, vy, wz], no heading assistance"
