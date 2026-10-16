"""Physical field definitions and metadata for the shared training packet."""

import math

from parkour_lab.control.proprioception import FOOT_NAMES

VERSION = "go2_training_physics_v1"
# Shape excludes the environment batch. Boolean fields count toward width.
FIELD_SPECS = {
    "root_com_linear_velocity_b": ((3,), "float32", "m/s", "root_link"),
    "root_angular_velocity_b": ((3,), "float32", "rad/s", "root_link"),
    "projected_gravity_b": ((3,), "float32", "unitless", "root_link"),
    "joint_position_relative_default": ((12,), "float32", "rad", "joint"),
    "joint_velocity": ((12,), "float32", "rad/s", "joint"),
    "foot_net_forces_b": ((4, 3), "float32", "N", "root_link"),
    "foot_force_valid": ((1,), "bool", "binary", "foot_sample"),
    "base_mass": ((1,), "float32", "kg", "base_link"),
    "base_com_offset_b": ((3,), "float32", "m", "base_link"),
    "material_friction": ((2,), "float32", "unitless", "robot_material"),
    "joint_stiffness": ((12,), "float32", "N*m/rad", "joint"),
    "joint_damping": ((12,), "float32", "N*m*s/rad", "joint"),
    "motor_strength": ((12,), "float32", "unitless", "joint"),
    "terrain_height": ((132,), "float32", "m", "root_yaw_reference_plane"),
    "terrain_valid": ((132,), "bool", "binary", "ray"),
}


def validate_joint_names(names):
    if not isinstance(names, (tuple, list)):
        raise ValueError("Information joint names must be a tuple or list")
    names = tuple(names)
    if (
        len(names) != 12
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != 12
    ):
        raise ValueError("Information requires twelve unique named joints")
    return names


def information_schema(joint_names):
    """Describe every permitted field without including environment handles."""
    names = validate_joint_names(joint_names)
    return {
        "version": VERSION,
        "width": sum(math.prod(spec[0]) for spec in FIELD_SPECS.values()),
        "training_only": True,
        "fields": {
            name: dict(shape=list(shape), dtype=dtype, units=units, frame=frame)
            for name, (shape, dtype, units, frame) in FIELD_SPECS.items()
        },
        "joint_names": list(names),
        "foot_names": list(FOOT_NAMES),
        "material_order": ["static_friction", "dynamic_friction"],
        "definitions": {
            "root_com_linear_velocity_b": "root rigid-body COM, not whole-robot COM",
            "root_link": "x forward, y left, z up; full root-link orientation",
            "projected_gravity_b": "unit gravity direction, not acceleration",
            "material_friction": "uniform robot coefficients, not contact-pair friction",
            "motor_strength": "dimensionless effort scale before the motor limiter",
            "joint_gains": "realized explicit actuator gains, not PhysX drive gains",
        },
        "sampling": {
            "period_s": 0.02,
            "state": "current control boundary; clean, detached owned tensors",
            "contacts": "latest completed physics sample; normal forces only",
            "dynamics": "verified realized startup properties, fixed across resets",
            "final": "outgoing rows captured before automatic reset",
            "reset": "contact force zero and invalid until a completed physics step",
            "noise": "none; capture consumes no random draws",
        },
        "terrain_scan": {
            "grid_shape_yx": [11, 12],
            "xy_m": [
                [-0.825 + 0.15 * x, -0.75 + 0.15 * y]
                for y in range(11)
                for x in range(12)
            ],
            "alignment": "root yaw only; x forward, y left, world-vertical rays",
            "ray_direction_w": [0.0, 0.0, -1.0],
            "ray_start_offset_z_m": 20.0,
            "max_distance_m": 25.0,
            "reference_below_root_link_m": 0.3,
            "formula": "clip(root_link_z - 0.3 - hit_z, -0.5, 0.5)",
            "range_m": [-0.5, 0.5],
            "invalid_fill_m": 0.5,
            "validity": "all hit coordinates finite; finite clipped hits stay valid",
        },
    }
