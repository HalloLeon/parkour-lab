"""Dependency-light semantic ordering for the Unitree Go2 interface."""

GO2_FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
GO2_LEG_NAMES = ("FL", "FR", "RL", "RR")
GO2_JOINT_TYPES = ("hip", "thigh", "calf")
GO2_JOINT_NAMES = tuple(
    f"{leg}_{joint_type}_joint"
    for leg in GO2_LEG_NAMES
    for joint_type in GO2_JOINT_TYPES
)

__all__ = [
    "GO2_FOOT_NAMES",
    "GO2_JOINT_NAMES",
    "GO2_JOINT_TYPES",
    "GO2_LEG_NAMES",
]
