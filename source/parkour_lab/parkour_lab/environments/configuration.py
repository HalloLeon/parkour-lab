"""Fixed Go2 motor, causal sensing, reward and dynamics configuration."""

from __future__ import annotations

from copy import deepcopy


def build_environment_config(*, num_envs, seed, device, training):
    """Build the shared native base after AppLauncher, before family selection."""
    if type(num_envs) is not int or num_envs <= 0:
        raise ValueError("Require a positive integer environment count")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Require a uint32 seed")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("Require an explicit device")
    if type(training) is not bool:
        raise ValueError("Require an explicit boolean training role")

    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors import RayCasterCfg, patterns
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
        UnitreeGo2RoughEnvCfg,
    )

    from parkour_lab.runtime.native import foot_contacts

    from .commands.episode import EpisodeVelocityCommand
    from .dynamics import configure_dynamics

    cfg = UnitreeGo2FlatEnvCfg()
    cfg.sim.dt = 0.005
    cfg.decimation = 4
    cfg.sim.render_interval = 4
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.sim.device = device
    cfg.episode_length_s = 20.0
    cfg.scene.contact_forces.update_period = 0.005
    cfg.scene.contact_forces.history_length = 3
    cfg.curriculum.terrain_levels = None
    if cfg.scene.robot.soft_joint_pos_limit_factor != 0.9:
        raise ValueError("Require the stock 0.9 soft joint-limit factor")
    cfg.rewards = deepcopy(UnitreeGo2RoughEnvCfg().rewards)

    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_heading_envs = command.rel_standing_envs = 0.0
    command.ranges.heading = None
    command.ranges.lin_vel_x = (-0.2, 0.5)
    command.ranges.lin_vel_y = (-0.2, 0.2)
    command.ranges.ang_vel_z = (-0.5, 0.5)
    command.debug_vis = False
    if not training:
        command.ranges.lin_vel_x = (0.0, 0.0)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
    command.class_type = EpisodeVelocityCommand
    command.resampling_time_range = (20.0, 20.0)

    cfg.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.15,
            size=(1.65, 1.50),
            direction=(0.0, 0.0, -1.0),
            ordering="xy",
        ),
        mesh_prim_paths=[cfg.scene.terrain.prim_path],
        max_distance=25.0,
        update_period=0.0,
        debug_vis=False,
    )
    cfg.observations.proprio = deepcopy(cfg.observations.policy)
    cfg.observations.proprio.base_lin_vel = None
    cfg.observations.proprio.foot_contacts = ObservationTermCfg(func=foot_contacts)
    cfg.observations.proprio.enable_corruption = False
    cfg.observations.policy = None
    configure_dynamics(
        cfg,
        dict(num_envs=num_envs, seed=seed, dynamics="randomized"),
        evaluation=not training,
    )
    return cfg
