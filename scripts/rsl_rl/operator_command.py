"""Isaac Lab binding for the operator curriculum. Import after AppLauncher."""

from functools import wraps
import copy
import inspect
import math

import torch

from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
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


class ProceduralTerrainCommand(OperatorTransitionCommand):
    """Training packet draws, never terrain-dependent steering or arbitration.

    Level/rough-flat tiles retain the v3 operator distribution. Other tiles
    exercise forward motion, both yaw signs, pivots and exact stops, matching
    the declared forward-only rough-terrain acquisition envelope. This class
    is not a live-operator adapter: externally supplied packets must bypass
    sampling entirely, not be clamped or redirected by terrain profile.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from parkour_lab.tasks.manager_based.parkour_lab.mdp.terrain.operator_terrain import (
            PROFILE_BY_COLUMN,
        )

        try:
            from .operator_sequences import ReversalSequencePlan
        except ImportError:
            from operator_sequences import ReversalSequencePlan

        generator = env.cfg.scene.terrain.terrain_generator
        if (
            not generator.curriculum
            or generator.num_cols != len(PROFILE_BY_COLUMN)
            or type(generator.num_rows) is not int
            or generator.num_rows not in (1, 3)
            or getattr(env.cfg.curriculum, "terrain_levels", None) is not None
            or tuple(
                getattr(sub, "profile", None) for sub in generator.sub_terrains.values()
            )
            != PROFILE_BY_COLUMN
            or any(
                not math.isclose(sub.proportion, 1 / len(PROFILE_BY_COLUMN))
                for sub in generator.sub_terrains.values()
            )
        ):
            raise ValueError(
                "Procedural command sampling requires its static one- or three-row profile layout"
            )
        # Difficulty rows share profile columns; they must not change commands.
        flat_columns = torch.tensor(
            [profile in ("plane", "rough_flat") for profile in PROFILE_BY_COLUMN],
            dtype=torch.bool,
            device=self.device,
        )
        self.flat_command_rows = flat_columns[env.scene.terrain.terrain_types]
        self.sequence_plan = ReversalSequencePlan(self.num_envs, self.device)
        self.rough_sampler = OperatorCommandSampler(self.device, TRANSITION_VERSION)
        # Same mode identities, magnitudes and live-stop replacement as v2;
        # only its training draw probabilities differ. No reverse/lateral
        # packet is drawn on non-flat terrain, including the coverage branch.
        probabilities = self.rough_sampler.probability.new_tensor(
            (0.15, 0.075, 0.075, 0.40, 0.15, 0.15, 0.0, 0.0, 0.0)
        )
        self.rough_sampler.probability.copy_(probabilities)
        self.rough_sampler.target_probability.copy_(probabilities)

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not len(ids):
            return
        flat_ids = ids[self.flat_command_rows[ids]]
        if len(flat_ids):
            new_episode = self.command_counter[flat_ids] == 0
            super()._resample_command(flat_ids)
            selected, commands, categories, seconds = self.sequence_plan.resample(
                flat_ids, new_episode
            )
            self.category[selected] = categories
            self.vel_command_b[selected] = commands
            self.is_standing_env[selected] = categories == 0
            self.time_left[selected] = seconds
        rough_ids = ids[~self.flat_command_rows[ids]]
        if len(rough_ids):
            categories, commands, seconds = self.rough_sampler.sample(
                len(rough_ids),
                previous_category=self.category[rough_ids],
                new_episode=self.command_counter[rough_ids] == 0,
            )
            self.category[rough_ids] = categories
            self.vel_command_b[rough_ids] = commands
            self.is_heading_env[rough_ids] = False
            self.is_standing_env[rough_ids] = categories == 0
            self.time_left[rough_ids] = seconds

    def __str__(self):
        return "ProceduralTerrainCommand: operator body twist on every row; no route guidance"


def procedural_physical_failure(
    env,
    minimum_m: float,
    minimum_surface_z_m: float,
    fall_margin_m: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_tilt_rad: float | None = None,
):
    """Terrain-relative physical failure; downhill elevation is not a fall.

    A center-ray miss over a real gap is not fabricated into a support height.
    A plunge below the lowest possible generated surface minus a declared
    margin is nevertheless a failure, even for a floorless hole. This bound
    must come from the geometry envelope, not a fixed offset below spawn.
    An optional total body tilt bound rejects tipped equilibria even when the
    center ray clears the floor. None preserves archived v1/v2 episode rules.
    """
    if (
        not math.isfinite(minimum_m)
        or minimum_m <= 0
        or not math.isfinite(minimum_surface_z_m)
        or minimum_surface_z_m > 0
        or not math.isfinite(fall_margin_m)
        or fall_margin_m <= 0
        or (
            max_tilt_rad is not None
            and (
                isinstance(max_tilt_rad, bool)
                or not math.isfinite(max_tilt_rad)
                or not 0 < max_tilt_rad < math.pi / 2
            )
        )
    ):
        raise ValueError(
            "Procedural clearance, geometry or tilt fall bounds are invalid"
        )
    hits = env.scene[sensor_cfg.name].data.ray_hits_w
    if hits.shape != (env.num_envs, 1, 3):
        raise ValueError("Procedural base clearance requires exactly one center ray")
    root_position = env.scene[asset_cfg.name].data.root_pos_w
    if not torch.isfinite(root_position).all() or torch.isnan(hits).any():
        raise RuntimeError("Nonfinite robot state or corrupt procedural clearance rays")
    valid = torch.isfinite(hits[:, 0]).all(dim=-1)
    clearance = root_position[:, 2] - hits[:, 0, 2]
    below_geometry = (
        root_position[:, 2] - env.scene.env_origins[:, 2]
        < minimum_surface_z_m - fall_margin_m
    )
    failed = (valid & (clearance < minimum_m)) | below_geometry
    if max_tilt_rad is not None:
        gravity = env.scene[asset_cfg.name].data.projected_gravity_b
        if gravity.shape != (env.num_envs, 3) or not torch.isfinite(gravity).all():
            raise RuntimeError("Invalid projected gravity for procedural tilt failure")
        # Isaac Lab bad_orientation's gravity-angle criterion, in cosine form
        # to avoid acos domain NaNs from floating-point roundoff at +/-1.
        failed |= gravity[:, 2] > -math.cos(max_tilt_rad)
    return failed


def procedural_workspace(env, margin_m: float):
    """Censor tile departures without steering, mastery credit or fall masking."""
    size = env.cfg.scene.terrain.terrain_generator.size
    if not math.isfinite(margin_m) or not 0 < margin_m < min(size) / 2:
        raise ValueError("Procedural workspace requires a finite positive tile margin")
    half = torch.as_tensor(size, device=env.device) / 2 - margin_m
    local = (
        env.scene["robot"].data.body_pos_w[..., :2] - env.scene.env_origins[:, None, :2]
    )
    boundary = (local.abs() >= half).any(dim=-1).any(dim=-1)
    # The config installs this term last, after every physical failure.
    return boundary & ~env.termination_manager.terminated


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


def operator_role(env):
    """Persistent lane ownership, independent of current/replayed terrain level."""
    return torch.arange(env.num_envs, device=env.device) % 4 == 0


def terrain_operator_mask(env):
    """Rollout-only retention metadata, excluded from BOTH policy input groups."""
    return operator_role(env).float().unsqueeze(-1)


def terrain_critic_context(env):
    """Value-only task state; never a command, actor input or family/level label.

    Reuse the native privileged critic's distance scale and route phase. Clear
    route fields on operator lanes, including after resets; operator control
    has no route objective. ObservationManager supplies this alongside both
    current and next/reset observations for native PPO value bootstrapping.
    """
    role = terrain_operator_mask(env)
    course = torch.cat(
        (
            0.25 * parkour_mdp.active_waypoint_distance_xy(env),
            parkour_mdp.route_phase(env),
        ),
        dim=-1,
    )
    return torch.cat((role, torch.where(role.bool(), 0.0, course)), dim=-1)


def teacher_operator_workspace(env, margin_m: float):
    """Training-only flat-workspace censoring, not failure or behavioral credit.

    Check every rigid-body center with explicit padding, not just commanded
    travel distance. Evaluate LAST, so a real physical failure is never turned
    into a bootstrapped workspace timeout. Fixed development gates omit this.
    """
    roles = operator_role(env)
    if (env.scene.terrain.terrain_levels[roles] != 0).any():
        raise RuntimeError("Operator lanes left their assigned L0 row")
    size = env.cfg.scene.terrain.terrain_generator.size
    if tuple(size) != (8.0, 4.0) or not 0 < margin_m < min(size) / 2:
        raise ValueError(
            "Teacher operator workspace requires the production 8 x 4 m tile and valid padding"
        )
    half = torch.as_tensor(size, device=env.device) / 2 - margin_m
    local = (
        env.scene["robot"].data.body_pos_w[..., :2] - env.scene.env_origins[:, None, :2]
    )
    boundary = (local.abs() >= half).any(dim=-1).any(dim=-1)
    return roles & boundary & ~env.termination_manager.terminated


def initialize_teacher_levels(env, env_ids, curriculum_cfg, terrain_layout):
    ids = torch.arange(env.num_envs, device=env.device)
    if env_ids is not None:
        ids = (
            ids[env_ids]
            if isinstance(env_ids, slice)
            else torch.as_tensor(env_ids, device=env.device)
        )
    if env.cfg.terrain_teacher_evaluation:
        # Ten independent resets of each canonical family/level. The native
        # 40-column mesh remains unchanged; only assignment selects variant 0.
        terrain = env.scene.terrain
        terrain.terrain_types[ids] = (terrain.terrain_types[ids] // 10) * 10
        for slot, level in enumerate((0, 1, 3, 6)):
            parkour_mdp.initialize_parkour_terrain_levels(
                env, ids[ids % 4 == slot], terrain_layout, curriculum_cfg, level
            )
    else:
        parkour_mdp.initialize_parkour_terrain_levels(
            env, ids, terrain_layout, curriculum_cfg, 0
        )


@wraps(parkour_mdp.completed_course_done, assigned=("__doc__",))
def teacher_course_success(env, **params):
    return parkour_mdp.completed_course_done(env, **params) & ~operator_role(env)


@wraps(parkour_mdp.off_route, assigned=("__doc__",))
def teacher_course_off_route(env, **params):
    return parkour_mdp.off_route(env, **params) & ~operator_role(env)


class TerrainTeacherCommand(OperatorTransitionCommand):
    """Randomized v3 operator chains and native course guidance on disjoint lanes.

    This is teacher acquisition, not live arbitration or within-episode handoff.
    Evaluation disables sampling; the existing benchmark publishes every packet.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        try:
            from .operator_sequences import ReversalSequencePlan
        except ImportError:
            from operator_sequences import ReversalSequencePlan
        self.sequence_plan = ReversalSequencePlan(self.num_envs, self.device)
        self.operator_role = operator_role(env)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample(self, env_ids):
        ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        self.vel_command_b[ids] = 0
        self.time_left[ids] = float("inf")
        self.is_heading_env[ids] = False
        self.is_standing_env[ids] = False
        if self._env.cfg.terrain_teacher_evaluation:
            self.command_counter[ids] += 1
            return
        selected = ids[self.operator_role[ids]]
        super()._resample(selected)

    def _resample_command(self, env_ids):
        new_episode = self.command_counter[env_ids] == 0
        super()._resample_command(env_ids)
        selected, commands, categories, seconds = self.sequence_plan.resample(
            env_ids, new_episode
        )
        self.category[selected] = categories
        self.vel_command_b[selected] = commands
        self.is_standing_env[selected] = categories == 0
        self.time_left[selected] = seconds

    def _update_command(self):
        if self._env.cfg.terrain_teacher_evaluation:
            return
        try:
            from .operator_benchmark import course_command
        except ImportError:
            from operator_benchmark import course_command
        desired = course_command(
            self._env,
            {"course": {"target_speed": 0.55}},
            self._env.episode_length_buf,
            self.finished,
        )
        self.vel_command_b[~self.operator_role] = desired[~self.operator_role]

    def __str__(self):
        return "TerrainTeacherCommand: fixed operator v3-chain/v2-background lanes and course guidance"


class RoleReward(ManagerTermBase):
    """Mask an existing stateless reward without modifying its physical formula."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.term = copy.deepcopy(cfg.params["term"])
        if not inspect.isfunction(self.term.func):
            raise ValueError("Role rewards support existing stateless functions only")
        # Native managers only resolve top-level SceneEntityCfg parameters.
        for value in self.term.params.values():
            if isinstance(value, SceneEntityCfg):
                value.resolve(env.scene)
        self.mask = operator_role(env) == cfg.params["operator"]

    def __call__(self, env, term, operator):
        value = self.term.func(env, **self.term.params)
        return torch.where(self.mask, value, 0.0)


class PersistentTilt(ManagerTermBase):
    """Training-only fall cutoff; short obstacle maneuvers remain possible."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.seconds = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids=None):
        self.seconds[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env, angle_deg: float, duration_s: float):
        tilted = env.scene["robot"].data.projected_gravity_b[:, 2] > -math.cos(
            math.radians(angle_deg)
        )
        self.seconds.copy_(torch.where(tilted, self.seconds + env.step_dt, 0.0))
        return self.seconds >= duration_s


class TerrainTeacherCurriculum(parkour_mdp.ParkourTerrainCurriculum):
    """Reuse production promotion/replay, excluding permanent operator lanes."""

    outcome_names = (
        "success",
        "chassis",
        "fall",
        "off_route",
        "timeout",
        "other_failure",
    )

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.outcomes = torch.zeros(40, 7, 6, dtype=torch.long, device=env.device)
        env.terrain_teacher_curriculum = self

    def __call__(self, env, env_ids, curriculum_cfg, terrain_layout):
        from parkour_lab.tasks.manager_based.parkour_lab.mdp.curriculums.curriculums import (
            _terminal_event_masks,
        )

        ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        ids = ids[~operator_role(env)[ids]]
        if not len(ids):
            return {}
        success, _, _, chassis, fall, off_route, timeout, other = _terminal_event_masks(
            env, ids
        )
        terrain = env.scene.terrain
        # Count the ATTEMPTED column/level before native promotion or replay.
        index = (terrain.terrain_types[ids] * 7 + terrain.terrain_levels[ids]) * 6
        for outcome, mask in enumerate(
            (success, chassis, fall, off_route, timeout, other)
        ):
            self.outcomes.view(-1).scatter_add_(0, index + outcome, mask.long())
        super().__call__(env, ids, curriculum_cfg, terrain_layout)
        # Native population metrics include fixed operator lanes. Do not publish
        # their diluted mean as course progress; actual denominators are saved.
        return {"course_completed_attempts": self.outcomes.sum().float()}

    def state_dict(self):
        return {
            "curriculum": super().state_dict(),
            "outcomes": self.outcomes.detach().cpu().clone(),
        }

    def load_state_dict(self, state):
        values = state["outcomes"]
        if (
            values.shape != self.outcomes.shape
            or values.dtype != torch.long
            or (values < 0).any()
        ):
            raise ValueError("Invalid attempted-course outcome counters")
        super().load_state_dict(state["curriculum"])
        self.outcomes.copy_(values)
