"""Shared contact-aware Go2 frame; history and reset state belong to each method."""

import torch

from .controller import SensorSpec


FRAME_DIM = 49
ACTION_SLICE = slice(33, 45)
CONTACT_SLICE = slice(45, 49)
FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
CONTACT_THRESHOLD_N = 1.5
FRAME_TERMS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
    ("foot_contacts", 4),
)

# Trusted native sensor semantics, not metadata supplied by a backbone. No
# simulator velocity, terrain, privileged latent or adapter-order assumptions.
NATIVE_SENSORS = {
    "base_ang_vel": ((3,), "rad/s", "body"),
    "projected_gravity": ((3,), "unitless", "body"),
    "joint_position_relative_default": ((12,), "rad", "joint"),
    "joint_position": ((12,), "rad", "joint"),
    "joint_velocity": ((12,), "rad/s", "joint"),
    "stock_previous_raw_action": ((12,), "unitless", "joint"),
    "foot_contacts": ((4,), "binary", "FR_FL_RR_RL"),
}


def proprioceptive_sensor_specs():
    """Return a fresh declaration of the six causal sensor fields."""
    return {
        name: SensorSpec(*NATIVE_SENSORS[name], max_age_s=0.02)
        for name in (
            "base_ang_vel",
            "projected_gravity",
            "joint_position_relative_default",
            "joint_velocity",
            "stock_previous_raw_action",
            "foot_contacts",
        )
    }


def pack_proprioception(
    base_ang_vel,
    projected_gravity,
    command,
    joint_position_relative_default,
    joint_velocity,
    previous_raw_action,
    foot_contacts,
):
    return torch.cat(
        (
            base_ang_vel,
            projected_gravity,
            command,
            joint_position_relative_default,
            joint_velocity,
            previous_raw_action,
            foot_contacts,
        ),
        dim=-1,
    )
