"""Shared physical information, geometry, ownership and reset contracts."""

import json
import math
import random
import unittest
from types import SimpleNamespace as NS

import numpy as np
import torch

from parkour_lab.benchmark.information import capture_information
from parkour_lab.benchmark.schema import information_schema
from parkour_lab.control.proprioception import FOOT_NAMES


def native_environment():
    """Native-shaped tensors; simulator calls deliberately are not available."""
    count = 2
    joints = tuple(f"joint_{index}" for index in range(12))
    joint_values = torch.arange(12, dtype=torch.float32).repeat(count, 1)
    data = NS(
        root_com_lin_vel_b=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        root_lin_vel_b=torch.full((count, 3), -999.0),
        root_ang_vel_b=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1),
        joint_pos=joint_values + 0.25,
        default_joint_pos=joint_values,
        joint_vel=joint_values + 0.5,
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(count, 1),
        root_link_pos_w=torch.tensor([[0.0, 0.0, 1.0]]).repeat(count, 1),
        joint_stiffness=torch.zeros(count, 12),
    )
    actuators = {}
    for name, ids in (
        ("even", list(range(10, -1, -2))),
        ("odd", list(range(11, 0, -2))),
    ):
        actuators[name] = NS(
            joint_names=[joints[index] for index in ids],
            stiffness=(joint_values + 20)[:, ids],
            damping=(joint_values + 0.5)[:, ids],
            motor_strength=(joint_values * 0.01 + 0.9)[:, ids],
        )
    bodies = ["base", *FOOT_NAMES]
    robot = NS(data=data, joint_names=joints, body_names=bodies, actuators=actuators)
    sensor_names = ["RL_foot", "base", "FR_foot", "RR_foot", "FL_foot"]
    forces = torch.tensor(
        [[4, 5, 6], [0, 0, 0], [1, 0, 0], [0, 0, 3], [0, 2, 0]], dtype=torch.float32
    )
    cache = {
        "masses": torch.tensor([[8.0, 1, 1, 1, 1]]).repeat(count, 1),
        "coms": torch.zeros(count, len(bodies), 7),
        "materials": torch.tensor([0.8, 0.6, 0.0]).repeat(count, 6, 1),
    }
    cache["coms"][:, 0, :3] = torch.tensor([0.01, -0.02, 0.03])
    return NS(
        num_envs=count,
        device="cpu",
        scene={
            "robot": robot,
            "contact_forces": NS(
                body_names=sensor_names,
                data=NS(net_forces_w=forces.repeat(count, 1, 1)),
            ),
            "height_scanner": NS(data=NS(ray_hits_w=torch.zeros(count, 132, 3))),
        },
        episode_length_buf=torch.tensor([1, 2]),
        _parkour_actual_physics=cache,
        _parkour_fixed_physics={"masses": torch.full((count, len(bodies)), 500.0)},
    )


class InformationTests(unittest.TestCase):
    def test_packet_dtype_does_not_depend_on_the_learner_default(self):
        env = native_environment()
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            packet = capture_information(env)
        finally:
            torch.set_default_dtype(previous)
        self.assertTrue(
            all(
                value.dtype == (torch.bool if name.endswith("valid") else torch.float32)
                for name, value in packet.items()
            )
        )

    def test_fixed_schema_and_packet_shapes_units_and_order(self):
        env = native_environment()
        schema = information_schema(env.scene["robot"].joint_names)
        packet = capture_information(env)
        self.assertEqual(schema["version"], "go2_training_physics_v1")
        self.assertEqual(schema["width"], 352)
        self.assertTrue(schema["training_only"])
        self.assertEqual(tuple(schema["fields"]), tuple(packet))
        self.assertEqual(len(packet), 15)
        self.assertEqual(sum(value[0].numel() for value in packet.values()), 352)
        self.assertEqual(schema["foot_names"], list(FOOT_NAMES))
        self.assertEqual(
            schema["material_order"], ["static_friction", "dynamic_friction"]
        )
        self.assertEqual(schema["fields"]["root_com_linear_velocity_b"]["units"], "m/s")
        self.assertEqual(schema["fields"]["foot_net_forces_b"]["frame"], "root_link")
        for name, value in packet.items():
            with self.subTest(name=name):
                spec = schema["fields"][name]
                self.assertEqual(value.shape, (2, *spec["shape"]))
                self.assertEqual(value.dtype, getattr(torch, spec["dtype"]))
                self.assertFalse(value.requires_grad)
                self.assertEqual(value.device, torch.device("cpu"))
        json.dumps(schema, allow_nan=False)
        schema["joint_names"].clear()
        schema["fields"]["joint_velocity"]["shape"].clear()
        self.assertEqual(
            len(information_schema(env.scene["robot"].joint_names)["joint_names"]), 12
        )

    def test_scan_locations_and_metric_clipping_keep_hit_validity(self):
        env = native_environment()
        hits = env.scene["height_scanner"].data.ray_hits_w
        hits[:, :3, 2] = torch.tensor([0.5, 2.0, -2.0])
        hits[:, 3] = float("nan")
        hits[:, 4, 0] = float("inf")
        hits[:, 5, 2] = float("inf")
        packet = capture_information(env)
        torch.testing.assert_close(
            packet["terrain_height"][:, :6],
            torch.tensor([[0.2, -0.5, 0.5, 0.5, 0.5, 0.5]]).repeat(2, 1),
        )
        self.assertTrue(packet["terrain_valid"][:, :3].all())
        self.assertFalse(packet["terrain_valid"][:, 3:6].any())
        scan = information_schema(env.scene["robot"].joint_names)["terrain_scan"]
        self.assertEqual(scan["grid_shape_yx"], [11, 12])
        self.assertEqual(len(scan["xy_m"]), 132)
        np.testing.assert_allclose(
            [scan["xy_m"][0], scan["xy_m"][11], scan["xy_m"][12], scan["xy_m"][-1]],
            [[-0.825, -0.75], [0.825, -0.75], [-0.825, -0.6], [0.825, 0.75]],
            atol=1e-12,
        )

    def test_named_contact_vectors_use_full_root_rotation(self):
        env = native_environment()
        half = math.sqrt(0.5)
        env.scene["robot"].data.root_link_quat_w[:] = torch.tensor(
            [[half, 0, 0, half], [half, half, 0, 0]]
        )
        forces = capture_information(env)["foot_net_forces_b"]
        expected = torch.tensor(
            [
                [[0, -1, 0], [2, 0, 0], [0, 0, 3], [5, -4, 6]],
                [[1, 0, 0], [0, 0, -2], [0, 3, 0], [4, 6, -5]],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(forces, expected, atol=1e-6, rtol=1e-6)
        env.scene["robot"].data.root_link_quat_w.neg_()
        torch.testing.assert_close(
            capture_information(env)["foot_net_forces_b"], forces
        )

    def test_reset_forces_are_zero_invalid_until_post_reset_step(self):
        env = native_environment()
        env.episode_length_buf[0] = 0
        env.scene["contact_forces"].data.net_forces_w[0] = float("nan")
        first = capture_information(env)
        self.assertEqual(first["foot_force_valid"].tolist(), [[False], [True]])
        self.assertEqual(torch.count_nonzero(first["foot_net_forces_b"][0]), 0)
        env.episode_length_buf[0] = 1
        env.scene["contact_forces"].data.net_forces_w[0] = 7.0
        second = capture_information(env)
        self.assertTrue(second["foot_force_valid"].all())
        self.assertTrue((second["foot_net_forces_b"][0] == 7).all())
        torch.testing.assert_close(
            second["foot_net_forces_b"][1], first["foot_net_forces_b"][1]
        )

    def test_realized_dynamics_and_explicit_gains_follow_joint_names(self):
        env = native_environment()
        packet = capture_information(env)
        self.assertTrue((packet["base_mass"] == 8).all())
        torch.testing.assert_close(
            packet["base_com_offset_b"],
            torch.tensor([[0.01, -0.02, 0.03]]).repeat(2, 1),
        )
        torch.testing.assert_close(
            packet["material_friction"], torch.tensor([[0.8, 0.6]]).repeat(2, 1)
        )
        for name, expected in (
            ("joint_stiffness", torch.arange(12) + 20),
            ("joint_damping", torch.arange(12) + 0.5),
            ("motor_strength", torch.arange(12) * 0.01 + 0.9),
        ):
            torch.testing.assert_close(packet[name], expected.float().repeat(2, 1))
        torch.testing.assert_close(
            packet["root_com_linear_velocity_b"],
            env.scene["robot"].data.root_com_lin_vel_b,
        )
        self.assertTrue((packet["joint_position_relative_default"] == 0.25).all())

    def test_owned_detached_values_do_not_consume_rng_or_modify_native_state(self):
        env = native_environment()
        data = env.scene["robot"].data
        data.joint_vel.requires_grad_()
        before_torch, before_python, before_numpy = (
            torch.get_rng_state(),
            random.getstate(),
            np.random.get_state(),
        )
        first = capture_information(env)
        self.assertTrue(torch.equal(torch.get_rng_state(), before_torch))
        self.assertEqual(random.getstate(), before_python)
        after_numpy = np.random.get_state()
        self.assertEqual(before_numpy[0], after_numpy[0])
        np.testing.assert_array_equal(before_numpy[1], after_numpy[1])
        self.assertEqual(before_numpy[2:], after_numpy[2:])
        for value in first.values():
            self.assertFalse(value.requires_grad)
        expected = data.joint_vel.detach().clone()
        first["joint_velocity"].zero_()
        torch.testing.assert_close(data.joint_vel, expected)
        second = capture_information(env)
        first["base_mass"].fill_(123)
        self.assertTrue((second["base_mass"] == 8).all())
        self.assertTrue((env._parkour_actual_physics["masses"][:, 0] == 8).all())
        data.root_link_pos_w[:, :2] += 1000
        data.root_link_pos_w[:, 2] += 10
        env.scene["height_scanner"].data.ray_hits_w[..., 2] += 10
        shifted = capture_information(env)
        for name in second:
            torch.testing.assert_close(shifted[name], second[name])

    def test_malformed_shapes_nonfinite_and_nonuniform_information_are_rejected(self):
        changes = {
            "joint broadcast": lambda e: setattr(
                e.scene["robot"].data, "joint_pos", torch.zeros(12)
            ),
            "default position broadcast": lambda e: setattr(
                e.scene["robot"].data, "default_joint_pos", torch.zeros(12)
            ),
            "joint velocity": lambda e: setattr(
                e.scene["robot"].data, "joint_vel", torch.zeros(2, 11)
            ),
            "nonfinite state": lambda e: e.scene["robot"].data.root_com_lin_vel_b.fill_(
                float("nan")
            ),
            "nonunit rotation": lambda e: e.scene[
                "robot"
            ].data.root_link_quat_w.zero_(),
            "nonfinite forces": lambda e: e.scene[
                "contact_forces"
            ].data.net_forces_w.fill_(float("inf")),
            "missing foot": lambda e: e.scene["contact_forces"].body_names.pop(),
            "invalid counter": lambda e: setattr(
                e, "episode_length_buf", torch.zeros(2)
            ),
            "ray count": lambda e: setattr(
                e.scene["height_scanner"].data, "ray_hits_w", torch.zeros(2, 131, 3)
            ),
            "unverified dynamics": lambda e: delattr(e, "_parkour_actual_physics"),
            "nonuniform friction": lambda e: e._parkour_actual_physics["materials"][
                :, 1, 0
            ].fill_(0.9),
            "nonpositive mass": lambda e: e._parkour_actual_physics["masses"][
                :, 0
            ].zero_(),
            "invalid COM shape": lambda e: e._parkour_actual_physics.update(
                coms=torch.zeros(2, 4, 7)
            ),
            "actuator coverage": lambda e: e.scene["robot"].actuators.pop("odd"),
            "actuator shape": lambda e: setattr(
                e.scene["robot"].actuators["odd"], "stiffness", torch.zeros(2, 5)
            ),
            "missing strength": lambda e: delattr(
                e.scene["robot"].actuators["odd"], "motor_strength"
            ),
            "negative strength": lambda e: e.scene["robot"]
            .actuators["odd"]
            .motor_strength.fill_(-1),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                env = native_environment()
                change(env)
                with self.assertRaises(ValueError):
                    capture_information(env)
        for names in (
            "abcdefghijkl",
            ["joint"] * 12,
            [f"j{i}" for i in range(11)],
            [None] * 12,
        ):
            with self.subTest(names=names), self.assertRaises(ValueError):
                information_schema(names)


if __name__ == "__main__":
    unittest.main()
