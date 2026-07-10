# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors import patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG
from . import mdp

##
# Pre-defined configs
##


PARKOUR_CURRICULUM = mdp.curriculums_config.DEFAULT_PARKOUR_CURRICULUM

DEFAULT_LEVEL = PARKOUR_CURRICULUM.levels[PARKOUR_CURRICULUM.initial_level]

GOAL_POS = (
    DEFAULT_LEVEL.goal_pos[0],
    DEFAULT_LEVEL.goal_pos[1],
    DEFAULT_LEVEL.goal_pos[2]
)

TARGET_SPEED = DEFAULT_LEVEL.target_speed
MIN_CLEARANCE = DEFAULT_LEVEL.min_clearance


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
            dynamic_friction=1.0
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.48, 0.35),
            roughness=0.8
        )
    )

    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.CylinderCfg(
            radius=0.25,
            height=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.8, 0.1)
            )
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=GOAL_POS)
    )

    robot: ArticulationCfg = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    feet_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_foot",
        history_length=3,
        track_air_time=True
    )

    leg_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_(thigh|calf)",
        history_length=3,
    )

    base_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        history_length=3
    )

    height_scanner: RayCasterCfg = RayCasterCfg(
        # Attach the ray sensor frame to the robot trunk.
        #
        # The scanner moves with the robot. Because the prim path is the trunk,
        # the scan pattern is defined relative to the trunk frame, then transformed
        # into the world during simulation.
        prim_path="{ENV_REGEX_NS}/Robot/trunk",

        # Local offset of the ray-pattern origin relative to the trunk frame.
        #
        # x = 0.375:
        #   Shift the scan grid forward in front of the robot. This is useful for
        #   parkour/obstacle traversal because we care more about upcoming terrain
        #   than terrain behind the robot.
        #
        # y = 0.0:
        #   Keep the scan centered laterally.
        #
        # z = 20.0:
        #   Start the rays high above the robot/terrain so all downward rays can
        #   safely hit the terrain mesh.
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.375, 0.0, 20.0)
        ),

        # Align the ray pattern with the robot's yaw only.
        #
        # This means:
        #   - if the robot turns left/right, the scan turns with it,
        #   - if the robot rolls or pitches, the scan does NOT tilt with the body.
        #
        # This is usually what we want for terrain height scans, because terrain
        # perception should stay horizontal in the world instead of rolling/pitching
        # with the trunk.
        ray_alignment="yaw",

        pattern_cfg=patterns.GridPatternCfg(
            # Distance between neighboring ray sample points.
            #
            # Smaller resolution:
            #   more rays, better terrain detail, more computation.
            #
            # Larger resolution:
            #   fewer rays, cheaper, less terrain detail.
            resolution=0.15,

            # Physical size of the scan grid in meters: (x_size, y_size).
            #
            # x_size = 1.65:
            #   scan covers roughly 1.65 m in the robot-forward direction.
            #
            # y_size = 1.50:
            #   scan covers roughly 1.50 m sideways.
            #
            # With resolution 0.15, this should produce:
            #   x samples: 12
            #   y samples: 11
            #   total rays: 12 * 11 = 132
            #
            # This matches HeightScanObservationCfg(num_rays=132).
            size=(1.65, 1.50),

            # Direction of each ray in the scanner frame.
            #
            # (0, 0, -1) means all rays point downward.
            direction=(0.0, 0.0, -1.0)
        ),

        # The mesh that the rays are allowed to hit.
        #
        # Important:
        #   RayCasterCfg supports raycasting against the terrain mesh.
        #   The obstacle is baked into /World/Ground,
        #   so this scanner can see both the flat ground and the obstacle.
        #
        # If the obstacle were a separate RigidObject, this would NOT see it.
        mesh_prim_paths=["/World/Ground"],

        # Maximum ray length.
        #
        # Rays start 20 m above the trunk and point downward.
        # max_distance=25.0 is enough to reach the terrain even if the robot moves
        # over small height variations.
        max_distance=25.0,

        # Optional:
        # Set this to True temporarily when debugging to visualize the rays.
        # Turn it off for training because visualization is expensive.
        # debug_vis=True
    )

    dome_light: AssetBaseCfg = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0)
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
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Deployable actor observations."""

        # Body orientation and angular motion.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # Goal-relative task information.
        goal_direction_body_xy = ObsTerm(
            func=mdp.goal_direction_body_xy,
            params={
                "goal_cfg": SceneEntityCfg("goal"),
                "asset_cfg": SceneEntityCfg("robot")
            }
        )

        goal_distance_xy = ObsTerm(
            func=mdp.goal_distance_xy_w,
            params={
                "goal_cfg": SceneEntityCfg("goal"),
                "asset_cfg": SceneEntityCfg("robot")
            }
        )

        desired_speed = ObsTerm(
            func=mdp.desired_speed_obs
        )

        # Joint state.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # Previous action.
        last_action = ObsTerm(func=mdp.last_action)

        # Contact state.
        foot_contacts = ObsTerm(
            func=mdp.foot_contact_state,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg(
                    "feet_contact",
                    body_names=".*_foot"
                )
            }
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Privileged critic observations."""

        # Same core observations as the actor.
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        goal_direction_body_xy = ObsTerm(
            func=mdp.goal_direction_body_xy,
            params={
                "goal_cfg": SceneEntityCfg("goal"),
                "asset_cfg": SceneEntityCfg("robot")
            }
        )

        goal_distance_xy = ObsTerm(
            func=mdp.goal_distance_xy_w,
            params={
                "goal_cfg": SceneEntityCfg("goal"),
                "asset_cfg": SceneEntityCfg("robot")
            }
        )

        desired_speed = ObsTerm(
            func=mdp.desired_speed_obs
        )

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)

        foot_contacts = ObsTerm(
            func=mdp.foot_contact_state,
            params={
                "threshold": 1.0,
                "sensor_cfg": SceneEntityCfg(
                    "feet_contact",
                    body_names=".*_foot"
                )
            }
        )

        # Privileged state.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        base_clearance = ObsTerm(
            func=mdp.base_clearance_obs,
            params={
                "asset_cfg": SceneEntityCfg("robot")
            }
        )

        height_scan = ObsTerm(
            func=mdp.height_scan_or_zeros,
            params={
                "obs_cfg": mdp.config.HeightScanObservationCfg(
                    num_rays=132,
                    vertical_offset=0.30,
                    clip=0.50
                ),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot")
            }
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_goal_and_commands = EventTerm(
        func=mdp.reset_goal_and_commands_from_terrain_level,
        mode="reset",
        params={
            "curriculum_cfg": mdp.curriculums_config.DEFAULT_PARKOUR_CURRICULUM,
            "goal_cfg": SceneEntityCfg("goal")
        }
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
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0)
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0)
            }
        }
    )

    # Reset joints to their default positions.
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),  # default_joint_pos
            "velocity_range": (0.0, 0.0)
        }
    )

    # reset_constant_parkour_commands = EventTerm(
    #     func=mdp.reset_constant_parkour_commands,
    #     mode="reset",
    #     params={
    #         "target_speed": TARGET_SPEED,
    #         "min_clearance": MIN_CLEARANCE
    #     }
    # )

    # reset_goal_and_obstacle_by_level = EventTerm(
    #     func=mdp.reset_goal_and_obstacle_by_level,
    #     mode="reset",
    #     params={
    #         "curriculum_cfg": mdp.curriculums_config.DEFAULT_PARKOUR_CURRICULUM,
    #         "obstacle_cfg": SceneEntityCfg("obstacle"),
    #         "goal_cfg": SceneEntityCfg("goal")
    #     }
    # )

    # update_levels_and_reset_goal_obstacle = EventTerm(
    #     func=mdp.update_levels_and_reset_goal_obstacle_by_level,
    #     mode="reset",
    #     params={
    #         "curriculum_cfg": mdp.curriculums_config.DEFAULT_PARKOUR_CURRICULUM,
    #         "obstacle_cfg": SceneEntityCfg("obstacle"),
    #         "goal_cfg": SceneEntityCfg("goal")
    #     }
    # )


@configclass
class CurriculumCfg:
    """Curriculum terms."""

    terrain_levels = CurrTerm(
        func=mdp.parkour_terrain_levels,
        params={
            "curriculum_cfg": mdp.curriculums_config.DEFAULT_PARKOUR_CURRICULUM
        }
    )


@configclass
class RewardsCfg:
    # Goal task.
    velocity_along_goal_xy = RewTerm(
        func=mdp.velocity_along_goal_xy_clearance_exp,
        weight=1.0,
        params={
            "tracking_cfg": mdp.config.GoalVelocityCfg(
                target_speed=TARGET_SPEED,
                speed_tracking_scale=0.25,
                slow_down_distance=1.25,
                min_clearance=MIN_CLEARANCE
            ),
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    goal_progress_xy = RewTerm(
        func=mdp.goal_progress_xy_stable,
        weight=3.0,
        params={
            "progress_cfg": mdp.config.StableGoalProgressCfg(
                progress_scale=0.03,
                reset_grace_steps=1,
                max_positive_reward=2.0,
                max_negative_penalty=2.0,
                lateral_drift_weight=0.25,
                max_lateral_penalty=1.0,
                stability=mdp.config.RootStabilityCfg(
                    max_roll_pitch_ang_speed=4.0,
                    max_projected_gravity_xy_norm=0.75,
                    min_clearance=MIN_CLEARANCE
                )
            ),
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    goal_heading_misalignment = RewTerm(
        func=mdp.goal_heading_misalignment_l2,
        weight=-0.05,
        params={
            "heading_cfg": mdp.config.GoalHeadingCfg(
                max_heading_error=1.0,
                min_forward_speed=0.1,
                full_forward_speed=0.5
            ),
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    reached_goal_xy = RewTerm(
        func=mdp.reached_goal_xy,
        weight=300.0,
        params={
            "threshold": 0.30,
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # Safety.
    illegal_contact = RewTerm(
        func=mdp.base_contact,
        weight=-10.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("base_contact", body_names="trunk")
        }
    )

    leg_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("leg_contact")
        }
    )

    base_clearance_below = RewTerm(
        func=mdp.base_clearance_below_l2,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # Stability and regularization.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.0005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    hip_deviation = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.002,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=".*_hip_joint"
            )
        }
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=".*_foot"
            ),
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=".*_foot"
            )
        }
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-0.5,
        params={
            "stumble_cfg": mdp.config.FeetStumbleCfg(
                lateral_to_vertical_force_ratio=4.0,
                min_vertical_force=1.0
            ),
            "sensor_cfg": SceneEntityCfg(
                "feet_contact",
                body_names=".*_foot"
            )
        }
    )

    rapid_feet_motion = RewTerm(
        func=mdp.rapid_feet_motion_l2,
        weight=-0.005,
        params={
            "motion_cfg": mdp.config.FeetMotionCfg(
                max_stance_speed=0.25,
                max_swing_speed=2.0,
                contact_threshold=1.0,
                max_penalty_per_foot=4.0
            ),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("feet_contact", body_names=".*_foot")
        }
    )

    no_feet_contact = RewTerm(
        func=mdp.no_feet_contact,
        weight=-0.2,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("feet_contact", body_names=".*_foot")
        }
    )

    root_chatter = RewTerm(
        func=mdp.root_chatter_l2,
        weight=-0.005,
        params={
            "chatter_cfg": mdp.config.RootMotionChatterCfg(
                small_z_displacement=0.02,
                min_z_reversal_speed=0.05,
                small_tilt_change=0.04,
                min_roll_pitch_reversal_rate=0.75,
                angular_weight=0.25,
                reset_grace_steps=1
            ),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.reached_goal_xy,
        params={
            "threshold": 0.30,
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    trunk_contact = DoneTerm(
        func=mdp.base_contact_done,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("base_contact", body_names="trunk")
        }
    )


##
# Environment configuration
##


@configclass
class ParkourLabEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings.
    scene: ParkourLabSceneCfg = ParkourLabSceneCfg(num_envs=4096, env_spacing=8.0)

    # Basic settings.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # MDP settings.
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

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
