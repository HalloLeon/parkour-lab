"""Task sampling: seed isolation, fixed dynamics, starts and continuation."""

import hashlib
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from parkour_lab.environments.randomization import (
    NAMESPACE,
    STREAMS,
    TaskRandomization,
)


def sampler(*, count=4, seed=42, evaluation=False, dynamics="randomized"):
    return TaskRandomization(
        SimpleNamespace(
            num_envs=count,
            seed=seed,
            dynamics=dynamics,
        ),
        evaluation=evaluation,
    )


class RandomizationTests(unittest.TestCase):

    def test_full_hash_maps_to_recorded_backend_without_global_randomness(self):
        np.random.seed(11)
        expected_global = np.random.uniform(size=8)
        np.random.seed(11)
        draw = sampler()
        actual = draw.uniform([2], 5, "commands")[0]
        digest = hashlib.sha256(
            f"{NAMESPACE}/train|training/42|2|commands".encode()
        ).hexdigest()
        generator = np.random.Generator(np.random.PCG64(int(digest, 16)))
        mapping = draw.manifest()["attempts"][2]["streams"]["commands"]
        self.assertEqual(mapping["sha256"], digest)
        self.assertEqual(
            mapping["initial_state"], generator.bit_generator.state["state"]
        )
        np.testing.assert_array_equal(actual, generator.uniform(size=5))
        np.testing.assert_array_equal(np.random.uniform(size=8), expected_global)

    def test_draws_are_row_order_and_stream_independent(self):
        first, second = sampler(), sampler()
        expected = first.uniform(None, 8, "observation-noise")
        second.uniform([0, 3], 37, "commands")
        np.testing.assert_array_equal(
            second.uniform([3, 1], 8, "observation-noise"), expected[[3, 1]]
        )
        np.testing.assert_array_equal(
            second.uniform([2, 0], 8, "observation-noise"), expected[[2, 0]]
        )
        self.assertFalse(np.array_equal(expected[0], expected[1]))
        other = sampler(seed=43)
        self.assertFalse(
            np.array_equal(expected, other.uniform(None, 8, "observation-noise"))
        )

    def test_previews_and_manifest_do_not_advance_streams_or_expose_mutable_state(self):
        draw = sampler()
        expected = draw.uniform([1], 7, "start", advance=False)
        draw.manifest()["attempts"][1]["streams"].clear()
        np.testing.assert_array_equal(
            expected, draw.uniform([1], 7, "start", advance=False)
        )
        np.testing.assert_array_equal(expected, draw.uniform([1], 7, "start"))
        self.assertFalse(np.array_equal(expected, draw.uniform([1], 7, "start")))
        self.assertEqual(set(draw.manifest()["attempts"][1]["streams"]), set(STREAMS))

    def test_dynamics_are_independent_bounded_and_never_accumulate(self):
        draw = sampler(count=100, evaluation=True)
        physical = draw.dynamics()
        nominal = sampler(count=100, evaluation=True, dynamics="nominal").dynamics()
        for key in ("motor_strength", "kp_scale", "kd_scale"):
            np.testing.assert_array_equal(nominal[key], np.ones((100, 12)))
            self.assertTrue(((physical[key] >= 0.9) & (physical[key] <= 1.1)).all())
        np.testing.assert_array_equal(nominal["added_mass"], np.zeros(100))
        np.testing.assert_array_equal(nominal["static_friction"], np.full(100, 0.8))
        np.testing.assert_array_equal(
            physical["dynamic_friction"], 0.75 * physical["static_friction"]
        )
        self.assertTrue(
            ((physical["added_mass"] >= -1) & (physical["added_mass"] <= 3)).all()
        )
        self.assertTrue(
            (
                (physical["static_friction"] >= 0.6)
                & (physical["static_friction"] <= 1)
            ).all()
        )
        self.assertFalse(np.array_equal(physical["kp_scale"], physical["kd_scale"]))
        reference = physical["kp_scale"].copy()
        physical["kp_scale"][:] = 0
        draw.sample_start([3, 52])
        np.testing.assert_array_equal(draw.dynamics()["kp_scale"], reference)
        with self.assertRaisesRegex(ValueError, "Joint count"):
            draw.dynamics(6)

    def test_nominal_training_has_fixed_physics_but_varied_starts(self):
        draw = sampler(dynamics="nominal")
        np.testing.assert_array_equal(
            draw.dynamics()["motor_strength"], np.ones((4, 12))
        )
        first, second = draw.sample_start([0]), draw.sample_start([0])
        self.assertFalse(np.array_equal(first["joint_offset"], second["joint_offset"]))
        self.assertEqual(
            draw.manifest()["attempts"][0]["start"]["joint_offset"],
            first["joint_offset"][0].tolist(),
        )

    def test_reset_pose_bounds_and_local_start_frame(self):
        draw = sampler(count=100, evaluation=True)
        start = draw.sample_start(None)
        np.testing.assert_array_equal(start["nominal_heading"], np.zeros(100))
        self.assertTrue((np.abs(start["yaw_offset"]) <= 0.03).all())
        self.assertTrue((np.abs(start["joint_offset"]) <= 0.05).all())
        self.assertTrue((np.abs(start["start_xy_local"]) <= 0.05).all())
        np.testing.assert_allclose(
            np.linalg.norm(start["start_xy"], axis=1),
            np.linalg.norm(start["start_xy_local"], axis=1),
        )
        np.testing.assert_array_equal(
            start["start_yaw"], start["nominal_heading"] + start["yaw_offset"]
        )

    def test_collision_and_malformed_inputs_fail_without_silent_remapping(self):
        with patch("parkour_lab.environments.randomization.hashlib.sha256") as digest:
            digest.return_value.hexdigest.return_value = "0" * 64
            with self.assertRaisesRegex(ValueError, "collision"):
                sampler()
        draw = sampler()
        for ids in ([0, 0], [-1], [4], [True], [1.1]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                draw.uniform(ids, 3, "start")
        self.assertEqual(draw.uniform([], 3, "start").shape, (0, 3))
        for invalid in (
            dict(count=0),
            dict(count=True),
            dict(seed=-1),
            dict(seed=True),
            dict(dynamics="other"),
        ):
            with self.subTest(settings=invalid), self.assertRaises(ValueError):
                sampler(**invalid)

    def test_current_stream_json_round_trip_continues_partial_rows_after_warmup(self):
        original = sampler()
        physical = original.dynamics()
        original.sample_start(None)
        original.uniform([3, 1], 30, "observation-noise")
        original.uniform([2], 4, "commands")
        saved = json.loads(json.dumps(original.state_dict(), allow_nan=False))
        expected_noise = original.uniform([1, 0, 3], 30, "observation-noise", -1, 1)
        expected_command = original.uniform([2, 0], 4, "commands")
        expected_start = original.sample_start([3, 0])
        restored = sampler()
        restored.dynamics()
        restored.sample_start(None)
        restored.uniform(None, 30, "observation-noise")
        restored.load_state_dict(saved)
        self.assertTrue(
            all(row["start"] is None for row in restored.manifest()["attempts"])
        )
        np.testing.assert_array_equal(
            restored.uniform([1, 0, 3], 30, "observation-noise", -1, 1), expected_noise
        )
        np.testing.assert_array_equal(
            restored.uniform([2, 0], 4, "commands"), expected_command
        )
        for name, value in restored.sample_start([3, 0]).items():
            np.testing.assert_array_equal(value, expected_start[name])
        for name, value in restored.dynamics().items():
            np.testing.assert_array_equal(value, physical[name])

    def test_noise_peek_matches_next_selected_draw_without_advancing_any_row(self):
        draw = sampler()
        draw.uniform([1], 30, "observation-noise")
        before = draw.state_dict()
        peeked = draw.uniform([3, 1], 30, "observation-noise", -1, 1, advance=False)
        self.assertEqual(draw.state_dict(), before)
        np.testing.assert_array_equal(
            peeked, draw.uniform([3, 1], 30, "observation-noise", -1, 1)
        )
        self.assertEqual(draw.state_dict()["generators"][0], before["generators"][0])

    def test_bad_late_stream_identity_or_physical_snapshot_leaves_sampler_unchanged(
        self,
    ):
        draw = sampler()
        draw.dynamics()
        draw.sample_start(None)
        saved = draw.state_dict()
        invalid = []
        item = deepcopy(saved)
        item["identity"]["group_id"] = "training/other"
        invalid.append(item)
        item = deepcopy(saved)
        item["generators"][-1]["commands"]["state"]["inc"] = 0
        invalid.append(item)
        item = deepcopy(saved)
        item["generators"][-1]["commands"]["has_uint32"] = True
        invalid.append(item)
        item = deepcopy(saved)
        item["physical"]["static_friction"][0] = 1.5
        invalid.append(item)
        item = deepcopy(saved)
        item["generators"].pop()
        invalid.append(item)
        for state in invalid:
            with self.subTest(state=state), self.assertRaises(ValueError):
                draw.load_state_dict(state)
            self.assertEqual(draw.state_dict(), saved)
        detached = draw.state_dict()
        detached["generators"][0]["commands"]["state"]["state"] = 0
        self.assertEqual(draw.state_dict(), saved)


if __name__ == "__main__":
    unittest.main()
