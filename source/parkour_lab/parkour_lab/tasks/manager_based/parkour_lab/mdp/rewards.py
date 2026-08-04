# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public reward terms for the parkour environment.

Implementations are grouped by domain in the ``reward_terms`` package.
They are imported here so references such as ``mdp.base_contact`` continue
to work.
"""

from .reward_terms.waypoint import (
    completed_course_reward,
    intermediate_milestone_reward,
    velocity_along_waypoint_xy_capped,
    velocity_along_waypoint_xy_clearance_capped,
    waypoint_heading_alignment_exp,
    waypoint_heading_misalignment_l2,
    waypoint_progress_xy_stable,
    waypoint_velocity_tracking_exp,
)
from .reward_terms.limb import (
    feet_air_time,
    feet_edge,
    feet_stumble,
    joint_deviation_l2,
    level_scaled_feet_slide,
    no_feet_contact,
    obstacle_only_feet_edge,
    obstacle_only_feet_stumble,
    obstacle_only_undesired_contacts,
    rapid_feet_motion_l2,
)
from .reward_terms.root_motion import root_chatter_l2
from .reward_terms.safety import base_clearance_below_l2, base_contact

__all__ = [
    # Active-waypoint task terms.
    "completed_course_reward",
    "intermediate_milestone_reward",
    "velocity_along_waypoint_xy_capped",
    "velocity_along_waypoint_xy_clearance_capped",
    "waypoint_heading_alignment_exp",
    "waypoint_heading_misalignment_l2",
    "waypoint_progress_xy_stable",
    "waypoint_velocity_tracking_exp",
    # Safety and clearance penalties.
    "base_contact",
    "base_clearance_below_l2",
    # Limb regularizers.
    "feet_air_time",
    "joint_deviation_l2",
    "feet_edge",
    "feet_stumble",
    "level_scaled_feet_slide",
    "no_feet_contact",
    "obstacle_only_feet_edge",
    "obstacle_only_feet_stumble",
    "obstacle_only_undesired_contacts",
    "rapid_feet_motion_l2",
    # Stateful root-motion regularization.
    "root_chatter_l2",
]
