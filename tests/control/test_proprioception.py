"""Ordered proprioceptive frame packing and portable sensor declarations."""

import unittest
from dataclasses import asdict

import torch
from parkour_lab.control.proprioception import (
    FRAME_DIM,
    FRAME_TERMS,
    pack_proprioception,
    proprioceptive_sensor_specs,
)


class ProprioceptionTests(unittest.TestCase):
    def test_frame_packing_preserves_order_shape_and_dtype(self):
        self.assertEqual(FRAME_DIM, 49)
        self.assertEqual(
            FRAME_TERMS,
            (
                ("base_ang_vel", 3),
                ("projected_gravity", 3),
                ("velocity_commands", 3),
                ("joint_pos", 12),
                ("joint_vel", 12),
                ("actions", 12),
                ("foot_contacts", 4),
            ),
        )
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                parts = [
                    (torch.arange(4 * width, dtype=dtype).reshape(4, width) + index)[
                        ::2
                    ]
                    for index, (_, width) in enumerate(FRAME_TERMS)
                ]
                result = pack_proprioception(*parts)
                self.assertEqual(result.shape, (2, FRAME_DIM))
                self.assertTrue(torch.equal(result, torch.cat(parts, dim=-1)))
                self.assertEqual(result.dtype, dtype)

    def test_sensor_declarations_are_independent_and_include_units_and_frames(self):
        expected = {
            "base_ang_vel": ((3,), "rad/s", "body"),
            "projected_gravity": ((3,), "unitless", "body"),
            "joint_position_relative_default": ((12,), "rad", "joint"),
            "joint_velocity": ((12,), "rad/s", "joint"),
            "stock_previous_raw_action": ((12,), "unitless", "joint"),
            "foot_contacts": ((4,), "binary", "FR_FL_RR_RL"),
        }
        first = proprioceptive_sensor_specs()
        self.assertEqual(tuple(first), tuple(expected))
        for name, (shape, units, frame) in expected.items():
            self.assertEqual(
                asdict(first[name]),
                {
                    "shape": shape,
                    "units": units,
                    "frame": frame,
                    "max_age_s": 0.02,
                    "required": True,
                    "privileged": False,
                },
            )
        first.clear()
        self.assertEqual(tuple(proprioceptive_sensor_specs()), tuple(expected))


if __name__ == "__main__":
    unittest.main()
