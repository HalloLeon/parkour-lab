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

from .reward_terms.goal import (
    completed_course_reward,
    goal_heading_misalignment_l2,
    goal_progress_xy_stable,
    velocity_along_goal_xy_capped,
    velocity_along_goal_xy_clearance_capped,
)
from .reward_terms.limb import (
    feet_stumble,
    joint_deviation_l2,
    no_feet_contact,
    rapid_feet_motion_l2,
)
from .reward_terms.root_motion import root_chatter_l2
from .reward_terms.safety import base_clearance_below_l2, base_contact

__all__ = [
    # Goal-directed task terms.
    "velocity_along_goal_xy_capped",
    "velocity_along_goal_xy_clearance_capped",
    "goal_progress_xy_stable",
    "goal_heading_misalignment_l2",
    "completed_course_reward",
    # Safety and clearance penalties.
    "base_contact",
    "base_clearance_below_l2",
    # Limb regularizers.
    "joint_deviation_l2",
    "feet_stumble",
    "no_feet_contact",
    "rapid_feet_motion_l2",
    # Stateful root-motion regularization.
    "root_chatter_l2",
]
