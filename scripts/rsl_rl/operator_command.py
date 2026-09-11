"""Isaac Lab binding for the operator curriculum. Import after AppLauncher."""

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand

try:
    from .operator_curriculum import OperatorCommandSampler, TRANSITION_VERSION, VERSION
except ImportError:
    from operator_curriculum import OperatorCommandSampler, TRANSITION_VERSION, VERSION


class OperatorVelocityCommand(UniformVelocityCommand):
    """Retain the stock 3-D command interface, changing only its sampling law."""

    curriculum_version = VERSION

    def __init__(self, cfg, env):
        if cfg.heading_command or cfg.rel_heading_envs or cfg.rel_standing_envs:
            raise ValueError(
                "Operator modes own exact stops and yaw; disable stock overrides"
            )
        super().__init__(cfg, env)
        self.sampler = OperatorCommandSampler(self.device, self.curriculum_version)
        self.category = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device
        )

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        category, command, duration = self.sampler.sample(
            len(env_ids),
            previous_category=self.category[env_ids],
            # CommandTerm.reset zeroes the counter BEFORE invoking this hook;
            # _resample increments it only AFTER the hook. Old episode commands
            # must not influence the first draw of a new physical episode.
            new_episode=self.command_counter[env_ids] == 0,
        )
        self.category[env_ids] = category
        self.vel_command_b[env_ids] = command
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = category == 0
        # CommandTerm._resample sets its default timer BEFORE this hook.
        self.time_left[env_ids] = duration

    def __str__(self):
        return (
            f"OperatorVelocityCommand ({self.curriculum_version}): "
            "body [vx, vy, wz], no heading assistance"
        )


class OperatorTransitionCommand(OperatorVelocityCommand):
    """Versioned training-only braking/reverse distribution; same 3-D interface."""

    curriculum_version = TRANSITION_VERSION
