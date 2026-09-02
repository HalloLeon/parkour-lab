# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the environment."""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from . import config as config
from .commands import (
    EVALUATION_PIVOT_WINDOW_DURATION_S as EVALUATION_PIVOT_WINDOW_DURATION_S,
    INTENT_COMMAND_NAME as INTENT_COMMAND_NAME,
    PROVISIONAL_ORACLE_RESIDUAL_THRESHOLD_RAD as PROVISIONAL_ORACLE_RESIDUAL_THRESHOLD_RAD,
    ParkourIntentCommand as ParkourIntentCommand,
    ParkourIntentCommandCfg as ParkourIntentCommandCfg,
    active_motion_time_s as active_motion_time_s,
    get_preferred_speed as get_preferred_speed,
    get_requested_travel_direction_yaw_xy as get_requested_travel_direction_yaw_xy,
    get_target_speed as get_target_speed,
    get_target_yaw_rate as get_target_yaw_rate,
    normalize_direction_yaw_xy as normalize_direction_yaw_xy,
    wrapped_heading_residual_rad as wrapped_heading_residual_rad,
)
from .curriculums import curriculums_config as curriculums_config
from .curriculums.curriculums import (
    ParkourTerrainCurriculum as ParkourTerrainCurriculum,
    initialize_parkour_terrain_levels as initialize_parkour_terrain_levels,
    reset_routes as reset_routes,
)
from ._shared.go2 import (
    GO2_FOOT_NAMES as GO2_FOOT_NAMES,
    GO2_JOINT_NAMES as GO2_JOINT_NAMES,
    GO2_JOINT_TYPES as GO2_JOINT_TYPES,
    GO2_LEG_NAMES as GO2_LEG_NAMES,
)
from .diagnostics import (
    TrainingDiagnostics as TrainingDiagnostics,
    latest_evaluation_step as latest_evaluation_step,
    report_training_diagnostics as report_training_diagnostics,
)
from .domain_randomization import (
    DelayedJointPositionAction as DelayedJointPositionAction,
    DelayedJointPositionActionCfg as DelayedJointPositionActionCfg,
    DomainRandomizationCfg as DomainRandomizationCfg,
    ProprioceptionDelay as ProprioceptionDelay,
    ProprioceptionDelayCfg as ProprioceptionDelayCfg,
    PrivilegedDynamicsRecorder as PrivilegedDynamicsRecorder,
    privileged_dynamics_component_names as privileged_dynamics_component_names,
    scaled_delay as scaled_delay,
    scaled_range as scaled_range,
)
from .observations import (
    active_waypoint_direction_yaw_xy as active_waypoint_direction_yaw_xy,
    active_waypoint_distance_xy as active_waypoint_distance_xy,
    base_clearance_obs as base_clearance_obs,
    desired_speed_obs as desired_speed_obs,
    desired_yaw_rate_obs as desired_yaw_rate_obs,
    foot_contact_state as foot_contact_state,
    recorded_privileged_dynamics as recorded_privileged_dynamics,
    route_phase as route_phase,
    terrain_height_scan as terrain_height_scan,
)
from .reward_terms.limb import (
    excessive_foot_air_time_l2 as excessive_foot_air_time_l2,
    feet_edge as feet_edge,
    feet_stumble as feet_stumble,
    joint_deviation_l2 as joint_deviation_l2,
    stable_orientation_l2 as stable_orientation_l2,
)
from .reward_terms.safety import (
    base_clearance_below_l2 as base_clearance_below_l2,
    chassis_contact as chassis_contact,
)
from .reward_terms.waypoint import (
    completed_course_reward as completed_course_reward,
    intermediate_milestone_reward as intermediate_milestone_reward,
    off_route_failure as off_route_failure,
    route_cross_track_excess_l2 as route_cross_track_excess_l2,
    stationary_velocity_tracking_exp as stationary_velocity_tracking_exp,
    waypoint_heading_alignment_exp as waypoint_heading_alignment_exp,
    waypoint_velocity_tracking_exp as waypoint_velocity_tracking_exp,
)
from .terminations import (
    active_motion_time_out as active_motion_time_out,
    chassis_contact_done as chassis_contact_done,
    completed_course_done as completed_course_done,
    fell_below_course as fell_below_course,
    off_route as off_route,
)
