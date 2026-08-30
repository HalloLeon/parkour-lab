# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

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
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from . import mdp

##
# Pre-defined configs
##


PARKOUR_CURRICULUM = mdp.curriculums_config.DEFAULT_PARKOUR_CURRICULUM

DEFAULT_TERRAIN_LAYOUT = PARKOUR_CURRICULUM.terrain_layout(
    mdp.curriculums_config.PARKOUR_TERRAIN_GENERATOR_CFG.num_cols
)

DEFAULT_FAMILY_INDEX = 0

DEFAULT_LEVEL = PARKOUR_CURRICULUM.course(
    DEFAULT_FAMILY_INDEX,
    PARKOUR_CURRICULUM.initial_level,
)

INITIAL_WAYPOINT_POS = DEFAULT_LEVEL.waypoints[0].position


def _canonical_command_speed(speed: float, stop_deadband_m_s: float) -> float:
    """Validate a command speed and collapse its ingress deadband to zero."""

    speed = float(speed)
    if not math.isfinite(speed) or speed < 0.0:
        raise ValueError("command speed must be finite and non-negative.")
    return 0.0 if speed <= stop_deadband_m_s else speed


def _canonical_command_yaw_rate(
    yaw_rate: float,
    deadband_rad_s: float,
    maximum_rad_s: float,
) -> float:
    """Validate a signed yaw rate and collapse its ingress deadband to zero."""

    yaw_rate = float(yaw_rate)
    if not math.isfinite(yaw_rate) or abs(yaw_rate) > maximum_rad_s:
        raise ValueError(
            f"command yaw rate must be finite and within +/-{maximum_rad_s} rad/s."
        )
    return 0.0 if abs(yaw_rate) <= deadband_rad_s else yaw_rate


##
# Scene definition
##


@configclass
class ParkourLabSceneCfg(InteractiveSceneCfg):
    """Configuration for a parkour lab scene."""

    # Course assets.
    ground: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/Ground",
        terrain_type="generator",
        terrain_generator=mdp.curriculums_config.PARKOUR_TERRAIN_GENERATOR_CFG,
        max_init_terrain_level=PARKOUR_CURRICULUM.initial_level,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            # The robot-side restitution sampled during domain randomization
            # must not be multiplied by the nominal ground value of zero.
            restitution_combine_mode="max",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.48, 0.35), roughness=0.8
        ),
    )

    waypoint_marker: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WaypointMarker",
        spawn=sim_utils.CylinderCfg(
            radius=PARKOUR_CURRICULUM.waypoint_reach_radius_m,
            height=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=INITIAL_WAYPOINT_POS),
    )

    # Robot.
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Contact sensors.
    chassis_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(base|Head_.*)", history_length=3
    )

    feet_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot", history_length=3, track_air_time=True
    )

    undesired_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_(hip|thigh|calf.*)",
        history_length=3,
    )

    # Ray sensors.
    # One downward terrain ray at the base origin provides geometry-agnostic
    # base clearance for flat ground, slopes, and arbitrary terrain meshes.
    base_height_scanner: RayCasterCfg = RayCasterCfg(
        # Attach the sensor to the base so its ray origin follows the robot.
        prim_path="{ENV_REGEX_NS}/Robot/base",
        # Cast from the base origin: the measured hit is therefore the terrain
        # surface directly underneath the base, not a nearby grid sample.
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        # Follow heading while ignoring roll and pitch, keeping the ray vertical
        # even when the base tilts. Yaw has no effect on this centered ray.
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
        # The base normally remains well within five meters of the terrain.
        max_distance=5.0,
        # Set debug_vis=True temporarily when inspecting ray placement. Keep it
        # disabled during training to avoid visualization overhead.
    )

    # Dense, forward-looking terrain scan for the Phase 1 teacher actor. The
    # explicit RSL-RL routing supplies this independent terrain group to both
    # actor and critic. It samples a 2-D grid instead of the single point
    # beneath the base.
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
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

    # Lighting.
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
    # For Unitree Go2 this controls the 12 leg joints.
    # The action is interpreted roughly as:
    #
    # target_joint_pos = default_joint_pos + scale * policy_action

    joint_pos = mdp.DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        soft_joint_limit_margin_rad=0.05,
        min_delay_steps=0,
        max_delay_steps=0,
    )


@configclass
class CommandsCfg:
    """Deployable intent command."""

    intent = mdp.ParkourIntentCommandCfg()


@configclass
class ObservationsCfg:
    """Source groups for teacher, critic, command, and wrapper-derived history."""

    @configclass
    class DeployablePolicyCfg(ObsGroup):
        """Deployable proprioception and command state used by the motor actor."""

        # Go2 LowState does not expose base linear velocity during low-level
        # control. Keep simulator truth out of the deployable actor; it remains
        # available to the asymmetric critic below.

        # Body orientation and angular motion.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Preferred speed is deployable and remains separate from privileged
        # terrain and dynamics observations.
        desired_speed = ObsTerm(func=mdp.desired_speed_obs)

        # Joint state.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)

        # Previous action.
        last_action = ObsTerm(func=mdp.last_action)

        # Signed gravity-aligned yaw-rate command for pivots.
        desired_yaw_rate = ObsTerm(func=mdp.desired_yaw_rate_obs)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedDynamicsCfg(ObsGroup):
        """Actual persistent physics properties available only in simulation."""

        properties = ObsTerm(func=mdp.recorded_privileged_dynamics)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedTerrainCfg(ObsGroup):
        """Simulator ray-cast geometry consumed by the Phase 1 teacher."""

        # Heights and their validity mask share one term so the ray scan is
        # preprocessed once. Its flattened order remains heights then validity.
        height_scan = ObsTerm(
            func=mdp.terrain_height_scan,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "obs_cfg": mdp.config.HeightScanObservationCfg(
                    num_rays=132, vertical_offset=0.30, clip=0.50
                ),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
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
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=0.5)

        base_clearance = ObsTerm(
            func=mdp.base_clearance_obs,
            params={"asset_cfg": SceneEntityCfg("robot")},
            scale=2.0,
        )

        # Exact distance to the simulator waypoint can improve value
        # estimation but is not available to the deployed motor policy.
        active_waypoint_distance_xy = ObsTerm(
            func=mdp.active_waypoint_distance_xy,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "waypoint_marker_cfg": SceneEntityCfg("waypoint_marker"),
            },
            scale=0.25,
        )

        route_phase = ObsTerm(func=mdp.route_phase)

        # Isaac Lab derives these contacts from its physics contact sensor.
        # Keep them critic-only until equivalent hardware sensing is defined.
        foot_contacts = ObsTerm(
            func=mdp.foot_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("feet_contact", body_names=".*_foot"),
                "threshold": 1.0,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class OracleTravelDirectionCfg(ObsGroup):
        """Wrap-safe active-waypoint direction used by the privileged teacher."""

        active_waypoint_direction_yaw_xy = ObsTerm(
            func=mdp.active_waypoint_direction_yaw_xy,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "waypoint_marker_cfg": SceneEntityCfg("waypoint_marker"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    critic_privileged: CriticPrivilegedCfg = CriticPrivilegedCfg()
    dynamics: PrivilegedDynamicsCfg = PrivilegedDynamicsCfg()
    policy: DeployablePolicyCfg = DeployablePolicyCfg()
    terrain: PrivilegedTerrainCfg = PrivilegedTerrainCfg()

    # RSL-RL appends this local oracle to the Phase-1 teacher actor input.
    oracle_travel_direction: OracleTravelDirectionCfg = OracleTravelDirectionCfg()


@configclass
class EventsCfg:
    """Configuration for events."""

    # Optional domain-randomization terms are populated from the selected
    # nominal, narrow, or wide stage before the environment is constructed.
    add_base_mass: EventTerm | None = None
    push_robot: EventTerm | None = None
    randomize_actuator_gains: EventTerm | None = None
    randomize_base_com: EventTerm | None = None
    randomize_robot_material: EventTerm | None = None

    # Startup initialization.
    initialize_terrain_levels = EventTerm(
        func=mdp.initialize_parkour_terrain_levels,
        mode="startup",
        params={
            "curriculum_cfg": PARKOUR_CURRICULUM,
            "initial_level_override": None,
            "terrain_layout": DEFAULT_TERRAIN_LAYOUT,
        },
    )

    # Episode resets. This declaration order is behavioral: reset the route
    # only after the robot base and joints have reached their new state.
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
        },
    )

    # Reset the route after curriculum updates select the new terrain row. The
    # physical terrain column continues to identify the obstacle family.
    reset_routes = EventTerm(
        func=mdp.reset_routes,
        mode="reset",
        params={
            "curriculum_cfg": PARKOUR_CURRICULUM,
            "terrain_layout": DEFAULT_TERRAIN_LAYOUT,
            "waypoint_marker_cfg": SceneEntityCfg("waypoint_marker"),
        },
    )

    # Startup recording. Keep this final so every persistent randomizer has
    # already written the runtime properties captured by the privileged target.
    record_privileged_dynamics = EventTerm(
        func=mdp.PrivilegedDynamicsRecorder,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["base"],
                joint_names=[".*"],
                preserve_order=True,
            )
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms."""

    terrain_levels = CurrTerm(
        func=mdp.ParkourTerrainCurriculum,
        params={
            "curriculum_cfg": mdp.curriculums_config.DEFAULT_PARKOUR_CURRICULUM,
            "terrain_layout": DEFAULT_TERRAIN_LAYOUT,
        },
    )

    training_diagnostics = CurrTerm(
        func=mdp.report_training_diagnostics,
        params={"reward_term_name": "training_diagnostics"},
    )


@configclass
class RewardsCfg:
    """
    Task, safety, and motion-quality rewards for parkour locomotion.

    Signed waypoint progress saturates at the commanded speed. On obstacle
    rows, a phase-local ceiling permits bounded traversal speedups.
    Heading provides directional guidance without prescribing a
    terrain-specific gait.
    One-shot physical milestones and completion bonuses make discrete progress
    unambiguous, while safety remains separate.
    """

    # Active-waypoint task.
    waypoint_velocity_tracking = RewTerm(
        func=mdp.waypoint_velocity_tracking_exp,
        weight=1.5,
        params={
            "approach_allowance_distance_m": 0.6,
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_speed_cap_multiplier": 1.5,
            "std": 0.5,
            "waypoint_marker_cfg": SceneEntityCfg("waypoint_marker"),
        },
    )

    waypoint_heading_alignment = RewTerm(
        func=mdp.waypoint_heading_alignment_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "waypoint_marker_cfg": SceneEntityCfg("waypoint_marker"),
        },
    )

    stationary_velocity_tracking = RewTerm(
        func=mdp.stationary_velocity_tracking_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "planar_speed_std": 0.15,
            "yaw_rate_std": 0.5,
        },
    )

    route_cross_track_excess = RewTerm(
        func=mdp.route_cross_track_excess_l2,
        weight=-1.0,
        params={
            "soft_half_width_m": PARKOUR_CURRICULUM.soft_route_half_width_m,
            "hard_half_width_m": PARKOUR_CURRICULUM.hard_route_half_width_m,
        },
    )

    # Explicit physical milestones split one conservative +2 shaping budget;
    # this includes the two supported flat-bootstrap progress targets.
    completed_course = RewTerm(func=mdp.completed_course_reward, weight=4.0)

    intermediate_milestone = RewTerm(
        func=mdp.intermediate_milestone_reward,
        weight=2.0,
    )

    # Safety.
    base_clearance_below = RewTerm(
        func=mdp.base_clearance_below_l2,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    chassis_contact = RewTerm(
        func=mdp.chassis_contact,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("chassis_contact"),
            "threshold": PARKOUR_CURRICULUM.contact_force_threshold,
            "timestep_independent": True,
        },
    )

    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("undesired_contact"),
            "threshold": 1.0,
        },
    )

    # Motion quality and regularization. Mild posture priors price persistent
    # lean and folded limbs without prescribing a periodic gait. Keep vertical
    # motion affordable enough for deliberate takeoff and landing.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.025)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.25)
    joint_deviation_l2 = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(mdp.GO2_JOINT_NAMES),
                preserve_order=True,
            ),
        },
    )
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0002)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)

    # Foot-placement safety and contact quality.
    feet_edge = RewTerm(
        func=mdp.feet_edge,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=list(mdp.GO2_FOOT_NAMES),
                preserve_order=True,
            ),
            "curriculum_cfg": PARKOUR_CURRICULUM,
            "sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=list(mdp.GO2_FOOT_NAMES),
                preserve_order=True,
            ),
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=list(mdp.GO2_FOOT_NAMES),
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=list(mdp.GO2_FOOT_NAMES),
                preserve_order=True,
            ),
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.5,
        params={
            "lateral_to_vertical_force_ratio": 4.0,
            "min_force": 1.0,
            "sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=list(mdp.GO2_FOOT_NAMES),
                preserve_order=True,
            ),
        },
    )

    # Logging-only manager term. Its callable samples post-physics state and
    # returns exactly zero, so these diagnostics never affect policy reward.
    training_diagnostics = RewTerm(
        func=mdp.TrainingDiagnostics,
        weight=1.0,
        params={
            "action_term_name": "joint_pos",
            "capture_evaluation_step": False,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=list(mdp.GO2_FOOT_NAMES),
                joint_names=list(mdp.GO2_JOINT_NAMES),
                preserve_order=True,
            ),
            "base_height_sensor_cfg": SceneEntityCfg("base_height_scanner"),
            "contact_threshold": PARKOUR_CURRICULUM.contact_force_threshold,
            "feet_sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=list(mdp.GO2_FOOT_NAMES),
                preserve_order=True,
            ),
            "joint_velocity_limit_ratio": 0.95,
            "reverse_speed_threshold_mps": 0.05,
            "torque_clip_tolerance_nm": 1.0e-3,
            "waypoint_marker_cfg": SceneEntityCfg("waypoint_marker"),
        },
    )

    def __post_init__(self) -> None:
        """Validate fixed reward scalars once, outside the control loop."""

        params = self.waypoint_velocity_tracking.params
        if not math.isfinite(params["std"]) or params["std"] <= 0.0:
            raise ValueError("waypoint velocity std must be finite and positive.")
        if (
            not math.isfinite(params["obstacle_speed_cap_multiplier"])
            or params["obstacle_speed_cap_multiplier"] < 1.0
        ):
            raise ValueError(
                "obstacle speed cap multiplier must be finite and at least 1.0."
            )
        if (
            not math.isfinite(params["approach_allowance_distance_m"])
            or params["approach_allowance_distance_m"] <= 0.0
        ):
            raise ValueError("waypoint approach allowance must be finite and positive.")


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(
        func=mdp.active_motion_time_out,
        params={"max_active_motion_time_s": 25.0},
        time_out=True,
    )
    wall_time_out = DoneTerm(func=mdp.time_out, time_out=True)

    off_route = DoneTerm(
        func=mdp.off_route,
        params={
            "hard_half_width_m": PARKOUR_CURRICULUM.hard_route_half_width_m,
        },
    )

    success = DoneTerm(
        func=mdp.completed_course_done,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "feet_asset_cfg": SceneEntityCfg(
                "robot",
                body_names=list(mdp.GO2_FOOT_NAMES),
                preserve_order=True,
            ),
            "feet_contact_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=list(mdp.GO2_FOOT_NAMES),
                preserve_order=True,
            ),
            "chassis_contact_cfg": SceneEntityCfg("chassis_contact"),
            "waypoint_marker_cfg": SceneEntityCfg("waypoint_marker"),
            "contact_threshold": PARKOUR_CURRICULUM.contact_force_threshold,
            "max_completion_tilt_sine": 0.5,
            "max_completion_vertical_speed_m_s": 0.5,
            "support_margin": 0.05,
            "support_plane_tolerance": 0.12,
            "terminal_support_load_threshold_n": (
                PARKOUR_CURRICULUM.terminal_support_load_threshold_n
            ),
            "terminal_min_support_feet": 2,
            "terminal_stability_dwell_s": 0.2,
            "terminal_max_planar_speed_m_s": 0.2,
            "terminal_max_vertical_speed_m_s": 0.2,
            "terminal_max_yaw_rate_rad_s": 0.35,
            "terminal_max_roll_pitch_rate_rad_s": 0.35,
            "terminal_max_tilt_sine": 0.25,
            "progress_route_half_width_m": (
                PARKOUR_CURRICULUM.progress_route_half_width_m
            ),
            "hard_route_half_width_m": PARKOUR_CURRICULUM.hard_route_half_width_m,
        },
    )

    chassis_contact = DoneTerm(
        func=mdp.chassis_contact_done,
        params={
            "sensor_cfg": SceneEntityCfg("chassis_contact"),
            "threshold": PARKOUR_CURRICULUM.contact_force_threshold,
        },
    )

    fell_below_course = DoneTerm(
        func=mdp.fell_below_course,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "minimum_height": -0.5,
        },
    )

    def __post_init__(self) -> None:
        """Validate fixed completion scalars once, outside the control loop."""

        params = self.success.params
        positive = (
            "max_completion_vertical_speed_m_s",
            "support_plane_tolerance",
            "terminal_support_load_threshold_n",
            "terminal_stability_dwell_s",
            "terminal_max_planar_speed_m_s",
            "terminal_max_vertical_speed_m_s",
            "terminal_max_yaw_rate_rad_s",
            "terminal_max_roll_pitch_rate_rad_s",
            "progress_route_half_width_m",
        )
        if any(
            not math.isfinite(params[name]) or params[name] <= 0.0 for name in positive
        ):
            raise ValueError(
                "Route completion distances, loads, times, and bounds must be positive."
            )
        if params["contact_threshold"] < 0.0 or params["support_margin"] < 0.0:
            raise ValueError(
                "Route contact threshold and support margin must be non-negative."
            )
        if any(
            not 0.0 < params[name] <= 1.0
            for name in ("max_completion_tilt_sine", "terminal_max_tilt_sine")
        ):
            raise ValueError("Route completion tilt-sine bounds must be in (0, 1].")
        feet = params["terminal_min_support_feet"]
        if (
            isinstance(feet, bool)
            or not isinstance(feet, int)
            or not 1 <= feet <= len(mdp.GO2_FOOT_NAMES)
        ):
            raise ValueError(
                "terminal_min_support_feet must select between one and four feet."
            )


##
# Environment configuration
##


@configclass
class ParkourLabEnvCfg(ManagerBasedRLEnvCfg):
    # Single source of truth. synchronize_curriculum_config() propagates any
    # Hydra/programmatic overrides to terrain, events, transitions, and dones.
    parkour_curriculum: mdp.curriculums_config.ParkourCurriculumCfg = PARKOUR_CURRICULUM

    # Keep nominal learning deterministic, then opt into ``narrow`` and
    # ``wide`` perturbations on resumed runs.
    domain_randomization: mdp.DomainRandomizationCfg = mdp.DomainRandomizationCfg()

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

    # Manager settings.
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg | None = CurriculumCfg()
    events: EventsCfg = EventsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # None during adaptive training; fixed evaluation selects one matrix cell.
    evaluation_family: str | None = None
    evaluation_level: int | None = None
    evaluation_geometry_variant: int | None = None
    evaluation_desired_speed: float | None = None
    evaluation_desired_yaw_rate: float | None = None

    # Post initialization.
    def __post_init__(self) -> None:
        """Post initialization."""

        # Simulation and control timing.
        #
        # sim.dt = 0.005 means physics runs at 200 Hz.
        # decimation = 4 means the policy acts every 4 physics steps.
        # So the policy/control rate is 50 Hz.
        self.decimation = 4
        # Twenty-five seconds of translation-or-pivot time plus a separate
        # wall-clock cap that bounds stationary command windows.
        self.episode_length_s = 32.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation

        # Match the simulation material to the terrain material.
        self.sim.physics_material = self.scene.ground.physics_material

        # Contact sensors should update every physics step.
        if self.scene.chassis_contact is not None:
            self.scene.chassis_contact.update_period = self.sim.dt

        if self.scene.feet_contact is not None:
            self.scene.feet_contact.update_period = self.sim.dt

        if self.scene.undesired_contact is not None:
            self.scene.undesired_contact.update_period = self.sim.dt

        # Terrain rays used by observations and rewards update at policy rate.
        for sensor_name in ("base_height_scanner", "height_scanner"):
            sensor = getattr(self.scene, sensor_name)
            if sensor is not None:
                sensor.update_period = self.decimation * self.sim.dt

        self.synchronize_curriculum_config()
        self.synchronize_domain_randomization_config()

    def evaluation_course_metadata(self) -> dict[str, object]:
        """Return JSON-friendly metadata for the fixed matrix cell."""

        if self.evaluation_family is None or self.evaluation_level is None:
            return {}
        family_index = self.parkour_curriculum.family_index(self.evaluation_family)
        geometry_variant = self.evaluation_geometry_variant or 0
        course = self.parkour_curriculum.course(
            family_index,
            self.evaluation_level,
            geometry_variant,
        )
        return {
            "family": self.evaluation_family,
            "family_index": family_index,
            "difficulty_index": self.evaluation_level,
            "geometry_variant_index": geometry_variant,
            "course": course.metadata(),
        }

    def resolved_evaluation_speed(self) -> float | None:
        """Return the explicit override or the selected course's nominal speed."""

        if self.evaluation_family is None or self.evaluation_level is None:
            return None
        if self.evaluation_desired_speed is not None:
            return self.evaluation_desired_speed
        family_index = self.parkour_curriculum.family_index(self.evaluation_family)
        geometry_variant = self.evaluation_geometry_variant or 0
        return self.parkour_curriculum.course(
            family_index,
            self.evaluation_level,
            geometry_variant,
        ).target_speed

    def resolved_evaluation_yaw_rate(self) -> float | None:
        """Return the fixed pulse rate, or zero when no rate was requested."""

        if self.evaluation_family is None or self.evaluation_level is None:
            return None
        return self.evaluation_desired_yaw_rate or 0.0

    def configure_evaluation(
        self,
        family: str | None = None,
        level: int | None = None,
        seed: int | None = None,
        *,
        geometry_variant: int | None = None,
        speed: float | None = None,
        yaw_rate: float | None = None,
    ) -> None:
        """Freeze one course and command configuration with a single rebuild."""

        curriculum_cfg = self.parkour_curriculum
        if speed is not None:
            speed = _canonical_command_speed(
                speed,
                self.commands.intent.stop_deadband_m_s,
            )
        if yaw_rate is not None:
            yaw_rate = _canonical_command_yaw_rate(
                yaw_rate,
                self.commands.intent.yaw_rate_deadband_rad_s,
                self.commands.intent.max_external_yaw_rate_rad_s,
            )
        if family is None:
            family = curriculum_cfg.family_names[0]
        curriculum_cfg.family_index(family)
        if yaw_rate not in (None, 0.0) and level is None:
            level = 0
        if level is None:
            level = curriculum_cfg.max_level
        if not 0 <= level <= curriculum_cfg.max_level:
            raise ValueError(
                f"difficulty level must be in [0, {curriculum_cfg.max_level}], got {level}."
            )
        if geometry_variant is None:
            geometry_variant = 0
        if (
            isinstance(geometry_variant, bool)
            or not isinstance(geometry_variant, int)
            or not 0 <= geometry_variant < curriculum_cfg.num_geometry_variants
        ):
            raise ValueError(
                "geometry variant must be in "
                f"[0, {curriculum_cfg.num_geometry_variants - 1}], got {geometry_variant!r}."
            )

        self.evaluation_family = family
        self.evaluation_level = level
        self.evaluation_geometry_variant = geometry_variant
        self.evaluation_desired_speed = speed
        self.evaluation_desired_yaw_rate = yaw_rate
        self.curriculum = None
        self.domain_randomization.stage = "off"

        terrain_generator = self.scene.ground.terrain_generator
        if terrain_generator is not None:
            terrain_generator.num_rows = curriculum_cfg.num_difficulties
            # Give every parallel evaluation environment its own tile column.
            # All columns use the selected family, while the fixed row selects
            # its difficulty.
            terrain_generator.num_cols = max(1, self.scene.num_envs)
            # This flag controls terrain generation: row N must represent
            # difficulty N. It does not enable adaptive level updates, which
            # are disabled for evaluation by self.curriculum = None above.
            terrain_generator.curriculum = True
            if seed is not None:
                terrain_generator.seed = seed

        # Synchronize after changing the column count so terrain generation and
        # the startup event receive the same fixed-family column mapping.
        self.synchronize_curriculum_config()
        self.synchronize_domain_randomization_config()

    def set_evaluation_reset_profile(self, profile: str) -> None:
        """Select exact resets or isolated narrow initial-state jitter."""

        try:
            scale = {"canonical": 0.0, "jitter": 0.5}[profile]
        except KeyError as error:
            raise ValueError(
                f"Unsupported evaluation reset profile: {profile!r}."
            ) from error
        self._set_initial_state_randomization(scale)

    def synchronize_curriculum_config(self) -> None:
        """Validate and propagate the authoritative parkour curriculum.

        ``parkour_curriculum`` is the single source of truth, but the terrain,
        events, curriculum term, rewards, and terminations are separate nested
        configs that are initially constructed from default values. Hydra or
        programmatic overrides therefore do not automatically keep those
        consumers synchronized.

        This method rebuilds the physical terrain layout so row indices remain
        difficulty indices and columns remain obstacle-family assignments. It
        then passes the same curriculum object and its shared thresholds to
        every dependent manager term. Fixed evaluation uses the same path, with
        all columns mapped to the selected family and startup pinned to the
        selected difficulty.

        Call this after changing the curriculum, terrain column count, or
        evaluation selection and before constructing the environment. It only
        updates and validates configuration; runtime promotion and demotion are
        performed separately by the curriculum manager.
        """

        curriculum_cfg = self.parkour_curriculum
        curriculum_cfg.validate_configuration()

        if self.evaluation_desired_speed is not None:
            self.evaluation_desired_speed = _canonical_command_speed(
                self.evaluation_desired_speed,
                self.commands.intent.stop_deadband_m_s,
            )
        if self.evaluation_desired_yaw_rate is not None:
            self.evaluation_desired_yaw_rate = _canonical_command_yaw_rate(
                self.evaluation_desired_yaw_rate,
                self.commands.intent.yaw_rate_deadband_rad_s,
                self.commands.intent.max_external_yaw_rate_rad_s,
            )
        if self.evaluation_desired_yaw_rate not in (None, 0.0):
            if self.evaluation_level != 0:
                raise ValueError(
                    "Nonzero evaluation yaw rate is supported only at level 0."
                )
            speed = self.resolved_evaluation_speed()
            if speed is None or speed <= self.commands.intent.stop_deadband_m_s:
                raise ValueError(
                    "Nonzero evaluation yaw rate requires a positive translation speed for restart trials."
                )

        terrain_generator = self.scene.ground.terrain_generator
        if terrain_generator is None:
            raise ValueError("ParkourLabEnvCfg requires a generated terrain.")
        if not terrain_generator.curriculum or tuple(
            terrain_generator.difficulty_range
        ) != (0.0, 1.0):
            raise ValueError(
                "The discrete parkour row mapping requires terrain curriculum mode and difficulty_range=(0.0, 1.0)."
            )
        # Ground support regions use course-local coordinates. Validate them
        # once here, after the scene's actual tile size is known and before
        # the terrain generator invokes the same level configuration per tile.
        for course in curriculum_cfg.courses:
            course.validate_terrain_size(terrain_generator.size)

        # Generate one terrain row per difficulty so row changes never change
        # obstacle family.
        terrain_generator.num_rows = curriculum_cfg.num_difficulties

        # Build the authoritative semantic mapping for the physical terrain
        # columns. Training produces balanced family blocks; fixed evaluation
        # maps every column to the selected family. Deriving the generated
        # family set from this same mapping keeps mesh generation and runtime
        # route selection synchronized.
        terrain_layout = curriculum_cfg.terrain_layout(
            terrain_generator.num_cols,
            family_name=self.evaluation_family,
            geometry_variant_index=self.evaluation_geometry_variant,
        )
        terrain_generator.sub_terrains = mdp.curriculums_config.parkour_sub_terrains(
            curriculum_cfg,
            terrain_layout,
        )

        # Restrict training to the configured starting range; evaluation pins
        # startup to its requested difficulty row.
        initial_level = (
            curriculum_cfg.initial_level
            if self.evaluation_level is None
            else self.evaluation_level
        )
        self.scene.ground.max_init_terrain_level = initial_level

        # The static marker visualizes the default root radius. Per-waypoint
        # overrides remain runtime metadata and do not resize the USD object.
        self.scene.waypoint_marker.spawn.radius = curriculum_cfg.waypoint_reach_radius_m

        # Pass the same curriculum object to reset events so initial terrain
        # assignment and active routes use the authoritative table.
        self.events.initialize_terrain_levels.params["curriculum_cfg"] = curriculum_cfg
        self.events.initialize_terrain_levels.params["terrain_layout"] = terrain_layout
        self.events.initialize_terrain_levels.params["initial_level_override"] = (
            self.evaluation_level
        )
        self.events.reset_routes.params["curriculum_cfg"] = curriculum_cfg
        self.events.reset_routes.params["terrain_layout"] = terrain_layout
        evaluation_speed = self.resolved_evaluation_speed()
        if evaluation_speed is not None:
            fixed_range = (evaluation_speed, evaluation_speed)
            self.commands.intent.flat_speed_range_m_s = fixed_range
            self.commands.intent.obstacle_speed_range_m_s = fixed_range
        evaluation_yaw_rate = self.resolved_evaluation_yaw_rate()
        self.commands.intent.fixed_yaw_rate_rad_s = (
            evaluation_yaw_rate if evaluation_yaw_rate not in (None, 0.0) else None
        )
        active_budget = float(
            self.terminations.time_out.params["max_active_motion_time_s"]
        )
        if (
            not math.isfinite(active_budget)
            or active_budget <= 0.0
            or not math.isfinite(self.episode_length_s)
            or self.episode_length_s <= active_budget
        ):
            raise ValueError(
                "The wall-clock episode cap must exceed the positive active-motion timeout."
            )

        # The fixed evaluation configuration disables adaptive curriculum
        # updates, so synchronize this term only when it is present.
        if self.curriculum is not None:
            self.curriculum.terrain_levels.params["curriculum_cfg"] = curriculum_cfg
            self.curriculum.terrain_levels.params["terrain_layout"] = terrain_layout

        # The success term owns route advancement before reward computation.
        self.terminations.success.params["contact_threshold"] = (
            curriculum_cfg.contact_force_threshold
        )
        self.terminations.success.params["terminal_support_load_threshold_n"] = (
            curriculum_cfg.terminal_support_load_threshold_n
        )
        self.terminations.success.params["progress_route_half_width_m"] = (
            curriculum_cfg.progress_route_half_width_m
        )
        self.rewards.route_cross_track_excess.params["soft_half_width_m"] = (
            curriculum_cfg.soft_route_half_width_m
        )
        self.rewards.route_cross_track_excess.params["hard_half_width_m"] = (
            curriculum_cfg.hard_route_half_width_m
        )
        self.terminations.off_route.params["hard_half_width_m"] = (
            curriculum_cfg.hard_route_half_width_m
        )
        self.terminations.success.params["hard_route_half_width_m"] = (
            curriculum_cfg.hard_route_half_width_m
        )

        # Likewise, use one contact threshold for safety, route transitions,
        # and the contact-duration diagnostics.
        self.rewards.chassis_contact.params["threshold"] = (
            curriculum_cfg.contact_force_threshold
        )
        self.scene.feet_contact.force_threshold = curriculum_cfg.contact_force_threshold
        if self.rewards.training_diagnostics is not None:
            self.rewards.training_diagnostics.params["contact_threshold"] = (
                curriculum_cfg.contact_force_threshold
            )
        self.terminations.chassis_contact.params["threshold"] = (
            curriculum_cfg.contact_force_threshold
        )

        # Edge geometry, unlike simple level gates, needs the full course table.
        self.rewards.feet_edge.params["curriculum_cfg"] = curriculum_cfg

        # Hydra mutates the constructed config in place. Revalidate fixed term
        # scalars here, once after all overrides and curriculum wiring.
        self.rewards.__post_init__()
        self.terminations.__post_init__()

    def synchronize_domain_randomization_config(self) -> None:
        """Propagate the selected randomization stage to all manager terms.

        Hydra applies overrides after ``__post_init__``, so training entry
        points call this method again before ``gym.make``. Fixed evaluation
        selects ``off`` and follows the same path, removing every stochastic
        event, observation corruption, and control delay without changing any
        observation or action dimensions.
        """

        cfg = self.domain_randomization
        scale = cfg.stage_scale
        enabled = scale > 0.0

        action_delay = mdp.scaled_delay(cfg.max_action_delay_steps, scale)
        self.actions.joint_pos.min_delay_steps = 0
        self.actions.joint_pos.max_delay_steps = action_delay

        policy_group = self.observations.policy
        policy_group.enable_corruption = enabled
        proprioception_delay = mdp.scaled_delay(
            cfg.max_proprioception_delay_steps,
            scale,
        )
        proprioception_terms = {
            "base_ang_vel": cfg.angular_velocity_noise,
            "joint_pos": cfg.joint_position_noise_rad,
            "joint_vel": cfg.joint_velocity_noise_rad_s,
            "projected_gravity": cfg.gravity_noise,
        }
        for term_name, full_noise in proprioception_terms.items():
            term_cfg = getattr(policy_group, term_name)
            term_cfg.modifiers = (
                [
                    mdp.ProprioceptionDelayCfg(
                        min_delay_steps=0,
                        max_delay_steps=proprioception_delay,
                    )
                ]
                if enabled
                else None
            )
            noise = scale * full_noise
            term_cfg.noise = (
                UniformNoiseCfg(n_min=-noise, n_max=noise) if enabled else None
            )

        self._set_initial_state_randomization(scale)

        if not enabled:
            self.events.add_base_mass = None
            self.events.push_robot = None
            self.events.randomize_actuator_gains = None
            self.events.randomize_base_com = None
            self.events.randomize_robot_material = None
            return

        # Add a fixed per-environment base payload at simulator startup.
        self.events.add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": mdp.scaled_range(
                    cfg.added_base_mass_range_kg,
                    scale,
                ),
                "operation": "add",
            },
        )
        # Apply intermittent planar velocity impulses to train disturbance recovery.
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=cfg.push_interval_range_s,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "velocity_range": {
                    "x": mdp.scaled_range(
                        cfg.push_velocity_range_m_s,
                        scale,
                    ),
                    "y": mdp.scaled_range(
                        cfg.push_velocity_range_m_s,
                        scale,
                    ),
                },
            },
        )
        # Scale joint stiffness and damping to cover actuator-model error.
        gain_scale = mdp.scaled_range(
            cfg.actuator_gain_scale_range,
            scale,
            center=1.0,
        )
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": gain_scale,
                "damping_distribution_params": gain_scale,
                "operation": "scale",
            },
        )
        # Shift the base center of mass to model uneven payload placement.
        self.events.randomize_base_com = EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {
                    "x": mdp.scaled_range(cfg.base_com_x_range_m, scale),
                    "y": mdp.scaled_range(cfg.base_com_y_range_m, scale),
                    "z": mdp.scaled_range(cfg.base_com_z_range_m, scale),
                },
            },
        )
        # Vary robot contact behavior through friction and restitution.
        self.events.randomize_robot_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": mdp.scaled_range(
                    cfg.static_friction_range,
                    scale,
                    center=1.0,
                ),
                "dynamic_friction_range": mdp.scaled_range(
                    cfg.dynamic_friction_range,
                    scale,
                    center=1.0,
                ),
                "restitution_range": mdp.scaled_range(
                    cfg.restitution_range,
                    scale,
                ),
                "num_buckets": 64,  # Reuse 64 sampled material triples across shapes.
                "make_consistent": True,  # Keep dynamic friction at most static friction.
            },
        )

    def _set_initial_state_randomization(self, scale: float) -> None:
        """Apply only the configured initial root-state perturbations."""

        cfg = self.domain_randomization
        linear = mdp.scaled_range(cfg.initial_linear_velocity_range_m_s, scale)
        angular = mdp.scaled_range(cfg.initial_angular_velocity_range_rad_s, scale)
        xy = mdp.scaled_range(cfg.initial_xy_range_m, scale)
        self.events.reset_base.params["pose_range"] = {
            "x": xy,
            "y": xy,
            "yaw": mdp.scaled_range(cfg.initial_yaw_range_rad, scale),
        }
        self.events.reset_base.params["velocity_range"] = {
            axis: linear if axis in {"x", "y", "z"} else angular
            for axis in ("x", "y", "z", "roll", "pitch", "yaw")
        }


@configclass
class ParkourLabEnvCfgPlay(ParkourLabEnvCfg):
    """Small, fixed-difficulty configuration for comparable evaluation/video."""

    def __post_init__(self) -> None:
        self.scene.num_envs = 1
        self.scene.ground.terrain_generator.num_cols = 1
        self.evaluation_family = self.parkour_curriculum.family_names[0]
        self.evaluation_level = self.parkour_curriculum.max_level
        self.evaluation_geometry_variant = 0
        self.curriculum = None
        self.domain_randomization.stage = "off"
        super().__post_init__()
        # Retain the post-physics reward term so play.py can consume its
        # terminal-safe transition snapshots without re-running reward terms.
        self.rewards.training_diagnostics.params["capture_evaluation_step"] = True
