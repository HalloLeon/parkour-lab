# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from . import mdp

##
# Pre-defined configs
##


##
# Scene definition
##


OBSTACLE_POS = (1.0, 0.0, 0.06)
OBSTACLE_SIZE = (0.5, 1.2, 0.03)

GOAL_POS = (2.0, 0.0, 0.01)


@configclass
class ParkourLabSceneCfg(InteractiveSceneCfg):
    """Configuration for a parkour lab scene."""

    ground: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/Ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.55, 0.48, 0.35),
            roughness=0.8
        )
    )

    obstacle: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle",
        spawn=sim_utils.CuboidCfg(
            size=OBSTACLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg()
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=OBSTACLE_POS)
    )

    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.CylinderCfg(
            radius=0.25,
            height=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=None,
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

    base_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/trunk",
        history_length=3
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
            func=mdp.desired_speed_obs,
            params={
                "target_speed": 0.5
            }
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
            func=mdp.desired_speed_obs,
            params={
                "target_speed": 0.5
            }
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

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    alive = RewTerm(func=mdp.is_alive, weight=0.05)

    # Horizontal goal-reaching.
    forward_velocity_xy = RewTerm(
        func=mdp.velocity_towards_goal_xy_l2,
        weight=1.0,
        params={
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    goal_closeness_xy = RewTerm(
        func=mdp.goal_closeness_xy_l2,
        weight=1.0,
        params={
            "max_distance": 2.0,
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    reached_goal_xy = RewTerm(
        func=mdp.reached_goal_xy_l2,
        weight=20.0,
        params={
            "threshold": 0.30,
            "min_base_height": 0.22,
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # Vertical body-shape control.
    base_height_error = RewTerm(
        func=mdp.base_height_error_l2,
        weight=-1.0,
        params={
            "target_height": 0.35,
            "max_error": 0.5,
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # Illegal contact.
    illegal_contact = RewTerm(
        func=mdp.illegal_contact_l2,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("base_contact", body_names="trunk")
        }
    )

    # Stability and regularization.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.1)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
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

    no_feet_contact = RewTerm(
        func=mdp.no_feet_contact_l2,
        weight=-0.25,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("feet_contact", body_names=".*_foot")
        }
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    success = DoneTerm(
        func=mdp.reached_goal_xy,
        params={
            "threshold": 0.35,
            "min_base_height": 0.22,
            "goal_cfg": SceneEntityCfg("goal"),
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    trunk_contact = DoneTerm(
        func=mdp.illegal_contact,
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
    scene: ParkourLabSceneCfg = ParkourLabSceneCfg(num_envs=4096, env_spacing=4.0)

    # Basic settings.
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

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

        if self.scene.base_contact is not None:
            self.scene.base_contact.update_period = self.sim.dt
