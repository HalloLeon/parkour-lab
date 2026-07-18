# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from . import mdp

##
# Pre-defined configs
##


PARKOUR_CURRICULUM = mdp.curriculums_config.DEFAULT_PARKOUR_CURRICULUM

DEFAULT_LEVEL = PARKOUR_CURRICULUM.levels[PARKOUR_CURRICULUM.initial_level]

INITIAL_WAYPOINT_POS = DEFAULT_LEVEL.waypoints[0].position

##
# Scene definition
##


@configclass
class ParkourLabSceneCfg(InteractiveSceneCfg):
    """Configuration for a parkour lab scene."""

    ground: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/Ground",
        terrain_type="generator",
        terrain_generator=mdp.curriculums_config.PARKOUR_TERRAIN_GENERATOR_CFG,
        max_init_terrain_level=PARKOUR_CURRICULUM.initial_level,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.48, 0.35), roughness=0.8),
    )

    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.CylinderCfg(
            radius=PARKOUR_CURRICULUM.waypoint_reach_threshold,
            height=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=INITIAL_WAYPOINT_POS),
    )

    robot: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    feet_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3, track_air_time=True
    )

    leg_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_(thigh|calf)",
        history_length=3,
    )

    base_contact: ContactSensorCfg = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/trunk", history_length=3)

    # One downward terrain ray at the trunk origin provides geometry-agnostic
    # base clearance for flat ground, slopes, and arbitrary terrain meshes.
    base_height_scanner: RayCasterCfg = RayCasterCfg(
        # Attach the sensor to the trunk so its ray origin follows the robot.
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        # Cast from the trunk origin: the measured hit is therefore the terrain
        # surface directly underneath the base, not a nearby grid sample.
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        # Follow heading while ignoring roll and pitch, keeping the ray vertical
        # even when the trunk tilts. Yaw has no effect on this centered ray.
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            # A zero-size grid contains exactly one ray. Resolution remains a
            # required GridPatternCfg field but does not affect this pattern.
            resolution=1.0,
            size=(0.0, 0.0),
            # Ray directions use the sensor frame; negative Z points downward.
            direction=(0.0, 0.0, -1.0),
        ),
        # Generated terrain and all configured structures are combined under
        # /World/Ground, so the ray measures the real supporting surface.
        mesh_prim_paths=["/World/Ground"],
        # The trunk normally remains well within five meters of the terrain.
        max_distance=5.0,
        # Set debug_vis=True temporarily when inspecting ray placement. Keep it
        # disabled during training to avoid visualization overhead.
    )

    # Dense, forward-looking terrain scan for the Phase 1 teacher actor. The
    # explicit RSL-RL routing supplies this independent terrain group to both
    # actor and critic. It samples a 2-D grid instead of the single point
    # beneath the trunk.
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        # Shift the grid forward for upcoming-terrain coverage and start it high
        # enough that every downward ray begins above the course geometry.
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            # Smaller spacing improves terrain detail at additional ray-cast cost.
            resolution=0.15,
            # The 1.65 m by 1.50 m grid produces 12 * 11 = 132 rays, matching
            # HeightScanObservationCfg(num_rays=132).
            size=(1.65, 1.50),
            direction=(0.0, 0.0, -1.0),
            # Flatten with longitudinal X as the inner/fast-changing index and
            # lateral Y as the outer/slow-changing index.
            ordering="xy",
        ),
        mesh_prim_paths=["/World/Ground"],
        # Reach the terrain from the 20 m vertical offset with ample margin.
        max_distance=25.0,
    )

    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # The policy outputs joint-position target offsets.
    #
    # For Unitree A1 this controls the 12 leg joints.
    # The action is interpreted roughly as:
    #
    # target_joint_pos = default_joint_pos + scale * policy_action

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Teacher, critic, restricted-student, and supervision observations."""

    @configclass
    class DeployablePolicyCfg(ObsGroup):
        """Deployable proprioception and command state shared by both actors."""

        # Body orientation and angular motion.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # The operator-supplied speed is deployable; the oracle waypoint
        # direction is kept in its own group so the future student can replace
        # it with a depth-derived prediction without changing this state order.
        desired_speed = ObsTerm(func=mdp.desired_speed_obs)

        # Joint state.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # Previous action.
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedTerrainCfg(ObsGroup):
        """Simulator ray-cast geometry consumed by the Phase 1 teacher."""

        # Fixed-dimensional robot-relative terrain geometry. The PPO routing
        # explicitly appends this group to both actor and critic inputs.
        height_scan = ObsTerm(
            func=mdp.terrain_height_scan,
            params={
                "obs_cfg": mdp.config.HeightScanObservationCfg(num_rays=132, vertical_offset=0.30, clip=0.50),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Keep missing-hit information separate from the height value so future
        # gap terrain cannot be confused with a surface at the fallback height.
        height_scan_validity = ObsTerm(
            func=mdp.terrain_height_scan_validity,
            params={
                "obs_cfg": mdp.config.HeightScanObservationCfg(num_rays=132, vertical_offset=0.30, clip=0.50),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticPrivilegedCfg(ObsGroup):
        """Simulator-only state appended exclusively to the critic input."""

        # Keep this group limited to state that materially improves value
        # estimation and is absent from both policy and terrain groups.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        base_clearance = ObsTerm(func=mdp.base_clearance_obs, params={"asset_cfg": SceneEntityCfg("robot")})

        # Exact distance to the simulator waypoint can improve value
        # estimation but is not available to the deployed motor policy.
        goal_distance_xy = ObsTerm(
            func=mdp.goal_distance_xy_w,
            params={
                "goal_cfg": SceneEntityCfg("goal"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Isaac Lab derives these contacts from its physics contact sensor.
        # Keep them critic-only until equivalent hardware sensing is defined.
        foot_contacts = ObsTerm(
            func=mdp.foot_contact_state,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg("feet_contact", body_names=".*_foot"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StudentExteroceptionCfg(ObsGroup):
        """Temporary terrain-latent boundary for a future depth encoder."""

        # This zero placeholder remains only for pipeline smoke tests; a later
        # stage will replace it with the output of a trainable depth encoder.
        features = ObsTerm(
            func=mdp.student_exteroception_stub,
            params={"feature_dim": 32},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class OracleHeadingTargetCfg(ObsGroup):
        """Wrap-safe oracle heading used by the teacher and student loss."""

        direction_yaw_xy = ObsTerm(
            func=mdp.goal_direction_yaw_xy,
            params={
                "goal_cfg": SceneEntityCfg("goal"),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # Both teacher and student consume this one deployable state definition.
    # Their terrain and heading sources differ, but the state term order cannot
    # drift because there is no duplicated student-state observation group.
    policy: DeployablePolicyCfg = DeployablePolicyCfg()
    terrain: PrivilegedTerrainCfg = PrivilegedTerrainCfg()
    critic_privileged: CriticPrivilegedCfg = CriticPrivilegedCfg()
    student_exteroception: StudentExteroceptionCfg = StudentExteroceptionCfg()

    # RSL-RL appends this oracle group to the Phase-1 teacher. Distillation also
    # uses it as the heading label, but the student motor receives only its own
    # predicted heading.
    heading_target: OracleHeadingTargetCfg = OracleHeadingTargetCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    initialize_terrain_levels = EventTerm(
        func=mdp.initialize_parkour_terrain_levels,
        mode="startup",
        params={
            "curriculum_cfg": PARKOUR_CURRICULUM,
            "fixed_level": None,
        },
    )

    reset_waypoints_and_commands = EventTerm(
        func=mdp.reset_waypoints_and_commands_from_terrain_level,
        mode="reset",
        params={
            "curriculum_cfg": PARKOUR_CURRICULUM,
            "goal_cfg": SceneEntityCfg("goal"),
        },
    )

    # Reset the robot base at the beginning of each episode.
    #
    # We keep the initial pose deterministic for now:
    # x = 0
    # y = 0
    # yaw = 0
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    # Reset joints to their default positions.
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },  # default_joint_pos
    )


@configclass
class CurriculumCfg:
    """Curriculum terms."""

    terrain_levels = CurrTerm(
        func=mdp.parkour_terrain_levels,
        params={"curriculum_cfg": mdp.curriculums_config.DEFAULT_PARKOUR_CURRICULUM},
    )


@configclass
class RewardsCfg:
    """
    Task, safety, and motion-quality rewards for parkour locomotion.

    Normalized world-frame velocity toward the goal is the only dense progress
    signal. Safety remains separate so low clearance or recovery does not erase
    the directional learning signal. Flight and absolute roll/pitch are not
    penalized directly because both can be necessary on parkour terrain.
    """

    # Goal task.
    velocity_along_goal_xy = RewTerm(
        func=mdp.velocity_along_goal_xy_capped,
        weight=1.0,
        params={
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    goal_heading_misalignment = RewTerm(
        func=mdp.goal_heading_misalignment_l2,
        weight=-0.05,
        params={
            "heading_cfg": mdp.config.GoalHeadingCfg(
                max_heading_error=1.0, min_forward_speed=0.1, full_forward_speed=0.5
            ),
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    completed_course = RewTerm(
        func=mdp.completed_course_reward,
        weight=100.0,
    )

    # Safety.
    illegal_contact = RewTerm(
        func=mdp.base_contact,
        weight=-200.0,
        params={
            "threshold": PARKOUR_CURRICULUM.base_contact_threshold,
            "sensor_cfg": SceneEntityCfg("base_contact", body_names="trunk"),
        },
    )

    leg_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("leg_contact")},
    )

    base_clearance_below = RewTerm(
        func=mdp.base_clearance_below_l2,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # Motion quality and regularization.
    # Keep vertical motion affordable enough for deliberate takeoff and landing.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("feet_contact", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.5,
        params={
            "stumble_cfg": mdp.config.FeetStumbleCfg(lateral_to_vertical_force_ratio=4.0, min_vertical_force=1.0),
            "sensor_cfg": SceneEntityCfg("feet_contact", body_names=".*_foot"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.completed_course_done,
        params={
            "reach_threshold": PARKOUR_CURRICULUM.waypoint_reach_threshold,
            "reach_hold_s": PARKOUR_CURRICULUM.waypoint_reach_hold_s,
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    trunk_contact = DoneTerm(
        func=mdp.base_contact_done,
        params={
            "threshold": PARKOUR_CURRICULUM.base_contact_threshold,
            "sensor_cfg": SceneEntityCfg("base_contact", body_names="trunk"),
        },
    )


##
# Environment configuration
##


@configclass
class ParkourLabEnvCfg(ManagerBasedRLEnvCfg):
    # Single source of truth. synchronize_curriculum_config() propagates any
    # Hydra/programmatic overrides to terrain, events, transitions, and dones.
    parkour_curriculum: mdp.curriculums_config.ParkourCurriculumCfg = PARKOUR_CURRICULUM

    # Scene settings.
    scene: ParkourLabSceneCfg = ParkourLabSceneCfg(num_envs=4096, env_spacing=8.0)
    viewer: ViewerCfg = ViewerCfg(
        eye=(-1.0, -6.0, 2.5),
        lookat=(1.0, 0.0, 0.5),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
        resolution=(1280, 720),
    )

    # Basic settings.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = CurriculumCfg()

    # MDP settings.
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # None during adaptive training; set to an exact logical level for play.
    evaluation_level: int | None = None

    # Post initialization.
    def __post_init__(self) -> None:
        """Post initialization."""

        # Simulation and control timing.
        #
        # sim.dt = 0.005 means physics runs at 200 Hz.
        # decimation = 4 means the policy acts every 4 physics steps.
        # So the policy/control rate is 50 Hz.
        self.decimation = 4
        self.episode_length_s = 10.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Match the simulation material to the terrain material.
        self.sim.physics_material = self.scene.ground.physics_material

        # Contact sensors should update every physics step.
        if self.scene.feet_contact is not None:
            self.scene.feet_contact.update_period = self.sim.dt

        if self.scene.leg_contact is not None:
            self.scene.leg_contact.update_period = self.sim.dt

        if self.scene.base_contact is not None:
            self.scene.base_contact.update_period = self.sim.dt

        # Height scanner updates at policy rate.
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        if self.scene.base_height_scanner is not None:
            self.scene.base_height_scanner.update_period = self.decimation * self.sim.dt

        self.synchronize_curriculum_config()

    def evaluation_level_metadata(self) -> dict[str, object]:
        """Return JSON-friendly metadata for the fixed evaluation level."""

        if self.evaluation_level is None:
            return {}
        level = self.parkour_curriculum.levels[self.evaluation_level]
        return {"index": self.evaluation_level, **level.metadata()}

    def set_evaluation_difficulty(self, level: int | None = None, seed: int | None = None) -> None:
        """Freeze the environment at one reproducible logical difficulty."""

        # ``__post_init__`` ran when this config object was first constructed,
        # before Hydra applied its command-line overrides. Hydra does not call
        # ``__post_init__`` again, so propagate any overridden curriculum values
        # to terrain, events, rewards, and terminations before fixing the level.
        self.synchronize_curriculum_config()
        curriculum_cfg = self.parkour_curriculum
        if level is None:
            level = curriculum_cfg.max_level
        if not 0 <= level <= curriculum_cfg.max_level:
            raise ValueError(f"difficulty level must be in [0, {curriculum_cfg.max_level}], got {level}.")

        self.evaluation_level = level
        self.curriculum = None
        self.events.initialize_terrain_levels.params["fixed_level"] = level
        self.scene.ground.max_init_terrain_level = level

        terrain_generator = self.scene.ground.terrain_generator
        if terrain_generator is not None:
            terrain_generator.num_rows = len(curriculum_cfg.levels)
            # Generate at least one column, but never more columns than there
            # are environments available to occupy the generated terrain.
            terrain_generator.num_cols = max(1, min(terrain_generator.num_cols, self.scene.num_envs))
            terrain_generator.curriculum = True
            if seed is not None:
                terrain_generator.seed = seed

        self.observations.policy.enable_corruption = False

    def synchronize_curriculum_config(self) -> None:
        """Propagate the authoritative curriculum to every manager consumer."""

        curriculum_cfg = self.parkour_curriculum
        curriculum_cfg.validate_configuration()

        terrain_generator = self.scene.ground.terrain_generator
        if terrain_generator is None:
            raise ValueError("ParkourLabEnvCfg requires a generated terrain.")
        if not terrain_generator.curriculum or tuple(terrain_generator.difficulty_range) != (0.0, 1.0):
            raise ValueError(
                "The discrete parkour row mapping requires terrain curriculum mode and difficulty_range=(0.0, 1.0)."
            )
        if "parkour_course" not in terrain_generator.sub_terrains:
            raise ValueError("ParkourLabEnvCfg requires the 'parkour_course' sub-terrain.")

        # Ground support regions use course-local coordinates. Validate them
        # once here, after the scene's actual tile size is known and before
        # the terrain generator invokes the same level configuration per tile.
        for level in curriculum_cfg.levels:
            level.validate_terrain_size(terrain_generator.size)

        # Generate one terrain row per logical level so the row index and
        # curriculum-level index have the same meaning.
        terrain_generator.num_rows = len(curriculum_cfg.levels)

        # Give the terrain generator the same authoritative level definitions
        # used by reset events, commands, rewards, and curriculum updates.
        terrain_generator.sub_terrains["parkour_course"].levels = curriculum_cfg.levels

        # Restrict initial terrain assignment to the configured starting range.
        self.scene.ground.max_init_terrain_level = curriculum_cfg.initial_level

        # The marker starts at waypoint zero. Reset events select the matching
        # route independently for every environment after terrain assignment.
        self.scene.goal.init_state.pos = curriculum_cfg.levels[
            curriculum_cfg.initial_level
        ].waypoints[0].position
        # Keep the visible marker footprint consistent with the configured XY
        # radius used by the waypoint transition condition.
        self.scene.goal.spawn.radius = curriculum_cfg.waypoint_reach_threshold

        # Pass the same curriculum object to reset events so initial terrain
        # assignment, active routes, and commands use the authoritative table.
        self.events.initialize_terrain_levels.params["curriculum_cfg"] = curriculum_cfg
        self.events.reset_waypoints_and_commands.params["curriculum_cfg"] = curriculum_cfg

        # The fixed evaluation configuration disables adaptive curriculum
        # updates, so synchronize this term only when it is present.
        if self.curriculum is not None:
            self.curriculum.terrain_levels.params["curriculum_cfg"] = curriculum_cfg

        # The success term owns route advancement before reward computation.
        # Synchronize its proximity and dwell contract with the authoritative
        # curriculum so only a safely reached final waypoint ends an episode.
        self.terminations.success.params[
            "reach_threshold"
        ] = curriculum_cfg.waypoint_reach_threshold
        self.terminations.success.params[
            "reach_hold_s"
        ] = curriculum_cfg.waypoint_reach_hold_s

        # Likewise, use one contact threshold for both the safety penalty and
        # trunk-contact termination.
        self.rewards.illegal_contact.params["threshold"] = curriculum_cfg.base_contact_threshold
        self.terminations.trunk_contact.params["threshold"] = curriculum_cfg.base_contact_threshold


@configclass
class ParkourLabEnvCfgPlay(ParkourLabEnvCfg):
    """Small, fixed-difficulty configuration for comparable evaluation/video."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.ground.terrain_generator.num_cols = 1
        self.set_evaluation_difficulty(self.parkour_curriculum.max_level)
