"""Native commands and physical termination terms; no route or waypoint state."""

import math
import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.managers import SceneEntityCfg
from .sampling import OperatorCommandSampler
from .sequences import ReversalSequencePlan


class FlatVelocityCommand(UniformVelocityCommand):
    """Independent uniform body twists every four seconds, with 10% exact stops."""

    def __init__(self, cfg, env):
        if cfg.heading_command or cfg.rel_heading_envs or cfg.rel_standing_envs:
            raise ValueError(
                "Flat command sampling owns stops; disable heading overrides"
            )
        super().__init__(cfg, env)
        from .dynamics import get_randomization

        self.randomization = get_randomization(env)
        self.remaining_ticks = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.sampling_report = {
            "version": "flat_uniform_twist_v1",
            "period_steps": 200,
            "samples": 0,
            "stops": 0,
        }

    def compute(self, dt):
        if dt != 0.02:
            raise ValueError("Flat commands require 50 Hz control")
        self._update_metrics()
        # Same-step autoreset rows have not executed their new command yet.
        sampled = torch.isfinite(self.time_left)
        self.remaining_ticks -= (sampled & (self._env.episode_length_buf > 0)).long()
        self._resample((sampled & (self.remaining_ticks <= 0)).nonzero().flatten())
        self.time_left[sampled] = self.remaining_ticks[sampled] * dt
        self._update_command()

    def _resample(self, env_ids):
        # Avoid even an unused SDK timer draw from the learner's global RNG.
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids):
            self.remaining_ticks[env_ids] = 200
            self.time_left[env_ids] = 4.0
            self._resample_command(env_ids)
            self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids):
        ids = (
            env_ids.detach().cpu().numpy()
            if isinstance(env_ids, torch.Tensor)
            else env_ids
        )
        draws = self.randomization.uniform(ids, 4, "commands")
        values = torch.as_tensor(
            draws, device=self.device, dtype=self.vel_command_b.dtype
        )
        command = values[:, :3] * values.new_tensor(
            (0.7, 0.4, 1.0)
        ) + values.new_tensor((-0.2, -0.2, -0.5))
        stops = values[:, 3] < 0.1
        command[stops] = 0
        self.vel_command_b[env_ids] = command
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = stops
        self.sampling_report["samples"] += len(env_ids)
        self.sampling_report["stops"] += int(stops.sum())


class RoughVelocityCommand(FlatVelocityCommand):
    """Seeded body twists; steep curriculum rows draw forward-only packets.

    The row upper bound conservatively narrows uncertain boundary rows. Terrain
    labels only select training distributions; external packets bypass sampling.
    Opt-in pivots share the existing unrestricted packet and RNG draws.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.generator = env.cfg.scene.terrain.terrain_generator
        self.pivot_fraction = env.cfg.parkour_task["rough_pivot_fraction"]
        terrains = list(self.generator.sub_terrains.values())
        proportions = torch.tensor(
            [sub.proportion for sub in terrains],
            device=self.device,
            dtype=torch.float64,
        )
        # Match TerrainGenerator's curriculum column assignment, including its offset.
        columns = (
            torch.arange(
                self.generator.num_cols, device=self.device, dtype=torch.float64
            )
            / self.generator.num_cols
            + 0.001
        )
        indices = torch.searchsorted(
            (proportions / proportions.sum()).cumsum(0), columns, right=True
        )
        thresholds = torch.full_like(proportions, float("inf"))
        for index, sub in enumerate(terrains):
            if hasattr(sub, "step_height_range"):
                low, high = sub.step_height_range
                thresholds[index] = (0.08 - low) / (high - low)
            elif hasattr(sub, "grid_height_range"):
                # Neighbors and the raised platform can differ by twice the amplitude.
                low, high = sub.grid_height_range
                thresholds[index] = (0.04 - low) / (high - low)
            elif hasattr(sub, "slope_range"):
                low, high = sub.slope_range
                # Stock square pyramid: bound both gradient axes and one height-rounding step.
                scale = self.generator.horizontal_scale
                pixels = (
                    int(self.generator.size[0] / scale)
                    + 1
                    - 2 * (int(sub.border_width / scale) + 1)
                )
                factor = pixels / (2 * (pixels // 2))
                limit = (
                    math.tan(math.radians(10)) / math.sqrt(2)
                    - self.generator.vertical_scale / scale
                ) / factor
                thresholds[index] = (limit - low) / (high - low)
        self.threshold_by_column = thresholds[indices]
        self.sampling_report["version"] = "rough_uniform_twist_v1"
        self.sampling_report["restricted_samples"] = 0
        self.sampling_report["pivots"] = 0

    def _resample_command(self, env_ids):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        values = torch.as_tensor(
            self.randomization.uniform(env_ids.cpu().numpy(), 4, "commands"),
            device=self.device,
            dtype=self.vel_command_b.dtype,
        )
        terrain = self._env.scene.terrain
        low, high = self.generator.difficulty_range
        upper = (
            low
            + (high - low)
            * (terrain.terrain_levels[env_ids].to(self.threshold_by_column) + 1)
            / self.generator.num_rows
        )
        restricted = upper > self.threshold_by_column[terrain.terrain_types[env_ids]]
        command = values[:, :3] * values.new_tensor(
            (0.7, 0.4, 1.0)
        ) + values.new_tensor((-0.2, -0.2, -0.5))
        command[restricted, 0] = 0.2 + 0.3 * values[restricted, 0]
        command[restricted, 1:] = 0
        stops = (values[:, 3] < 0.1) & ~restricted
        command[stops] = 0
        pivots = (
            (values[:, 3] >= 0.1)
            & (values[:, 3] < 0.1 + self.pivot_fraction)
            & ~restricted
        )
        yaw_draw = values[pivots, 2]
        command[pivots, :2] = 0
        command[pivots, 2] = torch.where(
            yaw_draw < 0.5,
            -0.3 - 0.4 * yaw_draw,
            0.3 + 0.4 * (yaw_draw - 0.5),
        )
        self.vel_command_b[env_ids] = command
        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = stops
        self.sampling_report["samples"] += len(env_ids)
        self.sampling_report["stops"] += int(stops.sum())
        self.sampling_report["restricted_samples"] += int(restricted.sum())
        self.sampling_report["pivots"] += int(pivots.sum())


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
        return "OperatorVelocityCommand: body [vx, vy, wz], no heading assistance"


class ProceduralTerrainCommand(OperatorVelocityCommand):
    """Training packet draws, never terrain-dependent steering or arbitration.

    Level/rough-flat tiles retain the current mixed command distribution. Other tiles
    exercise forward motion, both yaw signs, pivots and exact stops, matching
    the declared forward-only rough-terrain acquisition envelope. This class
    is not a live-operator adapter: externally supplied packets must bypass
    sampling entirely, not be clamped or redirected by terrain profile.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        from parkour_lab.environments.terrain import (
            PROFILE_BY_COLUMN,
        )

        generator = env.cfg.scene.terrain.terrain_generator
        if (
            not generator.curriculum
            or generator.num_cols != len(PROFILE_BY_COLUMN)
            or type(generator.num_rows) is not int
            or generator.num_rows not in (1, 3, 5)
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
                "Procedural command sampling requires its static one-, three- or five-row profile layout"
            )
        # Difficulty rows share profile columns; they must not change commands.
        flat_columns = torch.tensor(
            [profile in ("plane", "rough_flat") for profile in PROFILE_BY_COLUMN],
            dtype=torch.bool,
            device=self.device,
        )
        self.flat_command_rows = flat_columns[env.scene.terrain.terrain_types]
        self.sequence_plan = ReversalSequencePlan(self.num_envs, self.device)
        self.rough_sampler = OperatorCommandSampler(self.device)
        # Same mode identities, magnitudes and live-stop replacement;
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

    A center-ray miss is not fabricated into a support height.
    A plunge below the lowest possible generated surface minus a declared
    margin is nevertheless a failure. This bound
    must come from the geometry envelope, not a fixed offset below spawn.
    An optional total body tilt bound rejects tipped equilibria even when the
    center ray clears the floor. The task factory supplies its declared bound.
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


class OperatorReversalSequenceCommand(OperatorVelocityCommand):
    """Keep mixed sampling for background episodes and after a sequence finishes."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sequence_plan = ReversalSequencePlan(self.num_envs, self.device)

    def _resample_command(self, env_ids):
        if not len(env_ids):
            return
        new_episode = self.command_counter[env_ids] == 0
        super()._resample_command(env_ids)
        selected, commands, categories, seconds = self.sequence_plan.resample(
            env_ids, new_episode
        )
        self.category[selected] = categories
        self.vel_command_b[selected] = commands
        self.is_standing_env[selected] = categories == 0
        self.time_left[selected] = seconds

    def __str__(self):
        return "OperatorReversalSequenceCommand: randomized training-only chains plus mixed commands"
