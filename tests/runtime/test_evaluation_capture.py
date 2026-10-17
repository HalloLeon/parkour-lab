"""Evaluator-only native truth, including owned ending reasons and ray misses."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from parkour_lab.control.proprioception import FOOT_NAMES
from parkour_lab.runtime.native import foot_support, motion_state
from tests.support.isaaclab import modules
from tests.support.native import make_native_env


class NativeEvaluationCaptureTests(unittest.TestCase):
    def test_named_foot_support_and_ending_reasons_own_pre_reset_values(self):
        env = make_native_env(1)
        env.capture_evaluation = True
        env.scene["robot"].body_names = ["base", *FOOT_NAMES]
        feet = env.scene["robot"].data.body_pos_w
        feet[0, 1:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        ended = torch.tensor([True])
        env.termination_manager = NS(
            active_terms=["base_contact"], get_term=lambda name: ended
        )
        heights = torch.tensor([[0.1, float("nan"), 0.3, 0.4]])
        valid = torch.tensor([[True, False, True, True]])
        with patch(
            "parkour_lab.runtime.native.foot_support", return_value=(heights, valid)
        ):
            state = motion_state(env)
        feet.zero_()
        heights.zero_()
        valid.zero_()
        ended.zero_()
        self.assertTrue(state["ending_base_contact"].item())
        self.assertEqual(state["foot_position_w"][0, :, 0].tolist(), [1, 2, 3, 4])
        self.assertTrue(state["support_valid"][0, 0])
        self.assertTrue(torch.isnan(state["support_height_w"][0, 1]))

    def test_rays_use_actual_foot_xy_and_preserve_missing_support(self):
        env = make_native_env(1)
        mesh = object()
        env.scene["height_scanner"] = NS(
            cfg=NS(mesh_prim_paths=["/terrain"]), meshes={"/terrain": mesh}
        )
        feet = torch.tensor([[[1.0, 2.0, 3.0]] * 4])
        starts_seen = []

        def raycast(starts, directions, actual, *, max_dist):
            starts_seen.append(starts.clone())
            self.assertIs(actual, mesh)
            self.assertEqual(max_dist, 40.0)
            self.assertTrue((directions[..., 2] == -1).all())
            hits = starts.clone()
            hits[..., 2] = 0.25
            hits[:, 1] = float("inf")
            return hits, None, None, None

        with modules({"isaaclab.utils.warp": NS(raycast_mesh=raycast)}):
            heights, valid = foot_support(env, feet)
        self.assertTrue(torch.equal(starts_seen[0][..., :2], feet[..., :2]))
        self.assertTrue((starts_seen[0][..., 2] == 23).all())
        self.assertEqual(valid.tolist(), [[True, False, True, True]])
        self.assertTrue(torch.isnan(heights[0, 1]))
        self.assertEqual(heights[0, 0], 0.25)
