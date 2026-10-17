"""The engineering contact check must retain physical evidence before reset."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from parkour_lab.control.proprioception import FOOT_NAMES
from scripts._robustness_checks.contact import check_outer_contact
from scripts._terrain_checks import contacts, geometry


class OuterContactTests(unittest.TestCase):
    def test_simultaneous_ending_uses_owned_outgoing_contact_not_replacement(self):
        force = torch.zeros(2, 4, 3)
        feet = torch.zeros(2, 4, 3)
        pose = torch.zeros(2, 7)
        pose[:, 3] = 1
        robot = NS(
            body_names=FOOT_NAMES,
            data=NS(body_link_pos_w=feet, root_link_pose_w=pose),
            write_root_pose_to_sim=lambda value: pose.copy_(value),
            write_root_velocity_to_sim=lambda value: None,
        )

        class Scene(dict):
            env_origins = torch.zeros(2, 3)

        native = NS(scene=Scene(robot=robot), _capturing=False)

        def reset(ids):
            force[ids] = 0
            feet[ids] = 0

        native._reset_idx = reset
        task = NS(
            _env=native,
            device="cpu",
            num_envs=2,
            interactions=dict(session=0, cumulative=0, uncertain=0),
        )

        def step(action):
            feet[:, :, 0], feet[:, :, 2] = 2.4, 0.02
            force[:, :, 2] = 20
            native._capturing = True
            native._reset_idx(torch.tensor([0]))
            native._capturing = False
            task.interactions = dict(session=2, cumulative=2, uncertain=0)
            return (
                {},
                None,
                torch.tensor([True, False]),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

        def cast(native, mesh, offsets):
            values = native.scene.env_origins[:, None] + offsets
            values[..., 2] = 0
            return values

        def sample(native, ids):
            return dict(
                force_w=force[ids].clone(),
                force_b=force[ids].clone(),
                quaternion_wxyz=pose[ids, 3:].clone(),
            )

        task.step = step
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(geometry, "_mesh", return_value=(None, {})),
            patch.object(geometry, "_cast", side_effect=cast),
            patch.object(contacts, "_contact_sample", side_effect=sample),
        ):
            result = check_outer_contact(task, NS(is_running=lambda: True), Path(tmp))
            receipt = json.loads((Path(tmp) / "outer_contact.json").read_text())
        self.assertTrue(result["passed"])
        self.assertEqual(result["interactions"]["session"], 2)
        self.assertTrue(receipt["witnesses"]["0"]["captured_before_reset"])
        self.assertFalse(receipt["witnesses"]["1"]["captured_before_reset"])
        self.assertEqual(receipt["witnesses"]["0"]["force_w"][0][2], 20)
        self.assertTrue((force[0] == 0).all())
        self.assertEqual(receipt["steps"][0]["terminated"], [True, False])
        self.assertIs(native._reset_idx, reset)


if __name__ == "__main__":
    unittest.main()
