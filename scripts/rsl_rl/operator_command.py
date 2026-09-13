"""Isaac Lab binding for the operator curriculum. Import after AppLauncher."""

from functools import wraps

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from parkour_lab.tasks.manager_based.parkour_lab import mdp as parkour_mdp

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


def initialize_readiness_levels(env, env_ids, curriculum_cfg, terrain_layout):
    """Pin alternating environments to L0/L1 in the production family layout."""
    from parkour_lab.tasks.manager_based.parkour_lab import mdp

    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else torch.as_tensor(env_ids, device=env.device)
    )
    for level in (0, 1):
        mdp.initialize_parkour_terrain_levels(
            env, ids[ids % 2 == level], terrain_layout, curriculum_cfg, level
        )


# Preserve the production signatures for Isaac Lab's argument checks, but keep
# our own module/name for faithful configuration serialization. SceneEntityCfg
# parameters remain top-level, so the native manager resolves their body IDs.
@wraps(parkour_mdp.completed_course_done, assigned=("__doc__",))
def readiness_course_success(env, **params):
    return parkour_mdp.completed_course_done(env, **params) & (
        env.scene.terrain.terrain_levels > 0
    )


@wraps(parkour_mdp.off_route, assigned=("__doc__",))
def readiness_course_off_route(env, **params):
    return parkour_mdp.off_route(env, **params) & (env.scene.terrain.terrain_levels > 0)


class TerrainReadinessCommand(UniformVelocityCommand):
    """Training-plumbing fixture: L0 operator profiles and L1 route guidance.

    This deterministic 20-second fixture is not a training curriculum or live
    operator arbitration. CommandManager publishes it before the next policy
    observation, including after physical resets.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        try:
            from .operator_benchmark_core import command_schedule
        except ImportError:
            from operator_benchmark_core import command_schedule

        if self.num_envs != 80:
            raise ValueError("Terrain readiness requires exactly 80 environments.")
        self.labels, schedule = command_schedule(4)
        self.schedule = torch.as_tensor(schedule, device=self.device)
        self.flat_ids = torch.arange(0, self.num_envs, 2, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample(self, env_ids):
        # Reset packets are exact zero; do not sample stock heading/stand modes
        # or advance RNG for unused commands in this deterministic fixture.
        self.time_left[env_ids] = float("inf")
        self.vel_command_b[env_ids] = 0.0
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = False
        self.command_counter[env_ids] += 1

    def _update_command(self):
        try:
            from .operator_benchmark import course_command
        except ImportError:
            from operator_benchmark import course_command

        steps = self._env.episode_length_buf
        desired = course_command(
            self._env, {"course": {"target_speed": 0.55}}, steps, self.finished
        )
        indices = steps[self.flat_ids].clamp(0, self.schedule.shape[0] - 1)
        desired[self.flat_ids] = self.schedule[
            indices, torch.arange(len(self.flat_ids), device=self.device)
        ]
        self.vel_command_b.copy_(desired)

    def __str__(self):
        return "TerrainReadinessCommand: L0 operator / L1 guidance, body [vx, vy, wz]"
