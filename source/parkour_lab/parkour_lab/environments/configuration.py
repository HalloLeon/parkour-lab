"""Fresh Go2 task construction, with no checkpoint or experiment reconstruction.

The installed stock Go2 flat task supplies motors and root-COM velocity terms.
Flat tasks use independently seeded commands, starts and bounded dynamics.
The actor receives 49 causal values, including four
noiseless foot-contact flags. Steps/traversal are diagnostic geometries, not an
approved behavioral gate.
"""

from __future__ import annotations

import copy
import math

from parkour_lab.config import TaskConfig


def build_environment_config(task: TaskConfig, *, evaluation: bool = False):
    """Construct a native config after AppLauncher; importing this module is inert.

    Flat sensor noise is applied at the shared causal-input boundary, not here.
    Evaluation changes command ownership, not reward arithmetic. Flat uses a real privileged
    plane scan with the same 264-D schema as procedural terrain. Traversal has a
    fixed diagnostic layout and must be externally commanded during evaluation.
    """
    if not isinstance(task, TaskConfig) or type(evaluation) is not bool:
        raise ValueError("Require TaskConfig and an explicit boolean evaluation flag")
    if task.terrain == "procedural" and task.num_rows not in (1, 3, 5):
        raise ValueError("Procedural command sampling supports one, three or five rows")
    if task.terrain == "steps" and (
        task.num_rows != 3 or task.difficulty_range != (0.15, 0.55)
    ):
        raise ValueError(
            "Step diagnostic geometry requires three rows and difficulty (0.15, 0.55)"
        )
    if task.terrain == "traversal" and (
        not evaluation or task.num_rows != 1 or task.difficulty_range != (1.0, 1.0)
    ):
        raise ValueError(
            "Traversal is evaluation-only fixed geometry: one row, difficulty (1.0, 1.0)"
        )

    from isaaclab.managers import (
        ObservationGroupCfg,
        ObservationTermCfg,
        SceneEntityCfg,
        TerminationTermCfg,
    )
    from isaaclab.sensors import RayCasterCfg, patterns
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
        UnitreeGo2FlatEnvCfg,
    )

    from . import observations
    from .commands import (
        FlatVelocityCommand,
        ProceduralTerrainCommand,
        procedural_physical_failure,
        procedural_workspace,
    )
    from .terrain import BORDER_WIDTH, ENVELOPES, make_operator_terrain_generator
    from parkour_lab.runtime.native import foot_contacts

    cfg = UnitreeGo2FlatEnvCfg()
    cfg.scene.num_envs = task.num_envs
    cfg.seed = task.seed
    cfg.sim.device = task.device
    cfg.episode_length_s = task.episode_length_s
    cfg.scene.contact_forces.update_period = cfg.sim.dt
    cfg.curriculum.terrain_levels = None
    if cfg.scene.robot.soft_joint_pos_limit_factor != 0.9:
        raise ValueError(
            "The current acquisition task requires the stock 0.9 soft joint-limit factor"
        )

    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_heading_envs = command.rel_standing_envs = 0.0
    command.ranges.heading = None
    command.ranges.lin_vel_x = (-0.3, 0.7)
    command.ranges.lin_vel_y = (-0.2, 0.2)
    command.ranges.ang_vel_z = (-0.8, 0.8)
    command.resampling_time_range = (2.0, 12.0)
    command.debug_vis = False
    command.class_type = (
        FlatVelocityCommand if task.terrain == "flat" else ProceduralTerrainCommand
    )
    cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.2, 0.2),
        "y": (-0.2, 0.2),
        "yaw": (-math.pi, math.pi),
    }
    from .rewards import configure_rewards

    configure_rewards(cfg, task)

    if task.terrain != "flat":
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = make_operator_terrain_generator(
            seed=task.seed,
            num_rows=task.num_rows,
            difficulty_range=task.difficulty_range,
            curriculum=True,
        )
        cfg.scene.terrain.max_init_terrain_level = task.num_rows - 1

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
        update_period=0.02,
        debug_vis=False,
    )
    cfg.scene.base_height_scanner = cfg.scene.height_scanner.replace(
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)
        )
    )
    terrain_observations = ObservationGroupCfg(
        enable_corruption=False, concatenate_terms=True
    )
    terrain_observations.height_scan = ObservationTermCfg(
        func=observations.terrain_height_scan,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "obs_cfg": observations.HeightScanObservationCfg(
                num_rays=132, vertical_offset=0.3, clip=0.5
            ),
        },
    )
    cfg.observations.terrain = terrain_observations
    cfg.terminations.procedural_physical_failure = TerminationTermCfg(
        func=procedural_physical_failure,
        params={
            "minimum_m": 0.12,
            "minimum_surface_z_m": (
                0.0
                if task.terrain == "flat"
                else -max(height for height, _ in ENVELOPES.values())
                * task.difficulty_range[1]
            ),
            "fall_margin_m": 0.5,
            "max_tilt_rad": math.pi / 4,
        },
        time_out=False,
    )
    # Keep every physical failure before this censoring term.
    if task.terrain != "flat":
        cfg.terminations.procedural_workspace = TerminationTermCfg(
            func=procedural_workspace,
            params={"margin_m": BORDER_WIDTH + 0.25},
            time_out=True,
        )

    # Copy sensor noise before making privileged clean state noiseless. The actor
    # retains the native term order minus the three oracle linear velocities.
    cfg.observations.policy.foot_contacts = ObservationTermCfg(func=foot_contacts)
    cfg.observations.proprio = copy.deepcopy(cfg.observations.policy)
    cfg.observations.proprio.base_lin_vel = None
    cfg.observations.policy.enable_corruption = False

    if task.terrain == "flat":
        from .dynamics import configure_dynamics

        configure_dynamics(cfg, task, evaluation=evaluation)
        command.ranges.lin_vel_x = (-0.2, 0.5)
        command.ranges.ang_vel_z = (-0.5, 0.5)
        command.resampling_time_range = (4.0, 4.0)
        cfg.observations.proprio.enable_corruption = False

    if task.terrain == "steps":
        from .step_field import configure

        configure(cfg)
    elif task.terrain == "traversal":
        from .traversal import configure, preflight

        configure(
            cfg,
            task.seed,
            preflight(task.seed, task.traversal_layout),
            layout=task.traversal_layout,
        )
    if evaluation:
        from parkour_lab.runtime.native import configure_external_command

        configure_external_command(command)
        cfg.observations.proprio.enable_corruption = False
    return cfg
