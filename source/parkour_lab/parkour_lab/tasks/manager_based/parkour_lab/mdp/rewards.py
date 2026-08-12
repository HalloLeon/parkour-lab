# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public reward terms for the parkour environment.

Implementations are grouped by domain in the ``reward_terms`` package.
They are imported here so references such as ``mdp.chassis_contact`` continue
to work.
"""

from .reward_terms.limb import (
    feet_edge,
    feet_stumble,
    joint_deviation_l2,
    touchdown_air_time,
)
from .reward_terms.safety import base_clearance_below_l2, chassis_contact
from .reward_terms.waypoint import (
    completed_course_reward,
    intermediate_milestone_reward,
    waypoint_heading_alignment_exp,
    waypoint_velocity_tracking_exp,
)

__all__ = [
    # Active-waypoint task terms.
    "completed_course_reward",
    "intermediate_milestone_reward",
    "waypoint_heading_alignment_exp",
    "waypoint_velocity_tracking_exp",
    # Safety and clearance penalties.
    "base_clearance_below_l2",
    "chassis_contact",
    # Limb rewards and regularizers.
    "feet_edge",
    "feet_stumble",
    "joint_deviation_l2",
    "touchdown_air_time",
]
