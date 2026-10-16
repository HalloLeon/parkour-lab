"""Pure task sampling: seed isolation, fixed dynamics, and prospective bank IDs."""

import hashlib
import json
import math
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from parkour_lab.environments.randomization import (
    NAMESPACE,
    STREAMS,
    TERRAIN_GROUPS,
    TaskRandomization,
    terrain_development_attempt,
)


def sampler(*, bank=None, count=4, seed=42, evaluation=False, dynamics="randomized"):
    return TaskRandomization(
        SimpleNamespace(
            num_envs=count,
            seed=seed,
            bank_profile=bank,
            dynamics=dynamics,
        ),
        evaluation=evaluation,
    )


class RandomizationTests(unittest.TestCase):
    def test_connected_global_ids_apply_independent_six_stream_draws(self):
        local_heading = {
            "stairs/up": 0.0,
            "stairs/down": math.pi,
            "ramp/up": math.pi / 2,
            "ramp/down": -math.pi / 2,
            "hill/short-first": 0.0,
            "hill/long-first": 0.0,
        }
        for group in TERRAIN_GROUPS:
            for index in (0, 49, 50, 99):
                with self.subTest(group=group, index=index):
                    assignment = terrain_development_attempt(group, index)
                    heading = (
                        local_heading[group.rsplit("/", 1)[0]]
                        + assignment["world"]["world_yaw"]
                    )
                    task = dict(
                        terrain="connected",
                        num_envs=1,
                        seed=42,
                        terrain_group=group,
                        terrain_attempt_index=index,
                    )
                    draw = TaskRandomization(
                        task, evaluation=True, nominal_heading=heading
                    )
                    manifest = draw.manifest()
                    attempt = manifest["attempts"][0]
                    self.assertEqual(manifest["namespace"], f"{NAMESPACE}/development")
                    self.assertEqual(manifest["group_id"], group)
                    self.assertEqual(attempt["attempt_index"], index)
                    self.assertEqual(attempt["stratum"], assignment["stratum"])
                    self.assertEqual(attempt["streams"], assignment["streams"])
                    self.assertEqual(set(attempt["streams"]), set(STREAMS[:-1]))
                    generators = {}
                    for name in STREAMS[:-1]:
                        digest = hashlib.sha256(
                            f"{NAMESPACE}/development|{group}|{index}|{name}".encode()
                        ).hexdigest()
                        generators[name] = np.random.Generator(
                            np.random.PCG64(int(digest, 16))
                        )
                        self.assertEqual(attempt["streams"][name]["sha256"], digest)
                        self.assertEqual(
                            attempt["streams"][name]["initial_state"],
                            generators[name].bit_generator.state["state"],
                        )
                    # World selection owns geometry; heading does not redraw it.
                    np.testing.assert_array_equal(
                        draw.uniform(None, 7, "geometry", advance=False)[0],
                        generators["geometry"].uniform(size=7),
                    )
                    values = generators["start"].uniform(-0.05, 0.05, size=15)
                    rotation = np.array(
                        [
                            [math.cos(heading), -math.sin(heading)],
                            [math.sin(heading), math.cos(heading)],
                        ]
                    )
                    start = draw.sample_start(None)
                    np.testing.assert_array_equal(
                        start["start_xy_local"][0], values[:2]
                    )
                    # Scalar/vector trig and dot kernels can round differently.
                    roundoff = 4 * np.finfo(values.dtype).eps * np.abs(values[:2]).sum()
                    np.testing.assert_allclose(
                        start["start_xy"][0],
                        rotation @ values[:2],
                        rtol=0,
                        atol=roundoff,
                    )
                    self.assertEqual(start["start_yaw"][0], heading + values[2] * 0.6)
                    self.assertEqual(start["nominal_heading"][0], heading)
                    np.testing.assert_array_equal(start["joint_offset"][0], values[3:])
                    values = generators["dynamics"].uniform(size=38)
                    physical = draw.dynamics()
                    varied = index >= 50
                    self.assertEqual(
                        physical["added_mass"][0], -1 + 4 * values[0] if varied else 0
                    )
                    friction = 0.6 + 0.4 * values[1] if varied else 0.8
                    self.assertEqual(physical["static_friction"][0], friction)
                    self.assertEqual(physical["dynamic_friction"][0], 0.75 * friction)
                    for offset, name in enumerate(
                        ("motor_strength", "kp_scale", "kd_scale")
                    ):
                        expected = (
                            0.9 + 0.2 * values[2 + 12 * offset : 14 + 12 * offset]
                            if varied
                            else np.ones(12)
                        )
                        np.testing.assert_array_equal(physical[name][0], expected)
                    for _ in range(3):
                        np.testing.assert_array_equal(
                            draw.uniform(None, 30, "observation-noise", -1, 1)[0],
                            generators["observation-noise"].uniform(-1, 1, size=30),
                        )
                    other = TaskRandomization(
                        {**task, "seed": 987}, evaluation=True, nominal_heading=heading
                    )
                    self.assertEqual(
                        other.manifest()["attempts"][0]["streams"], attempt["streams"]
                    )

    def test_connected_attempt_rewind_preserves_global_id_heading_and_physics(self):
        task = dict(
            terrain="connected",
            num_envs=1,
            seed=42,
            terrain_group="ramp/down/10deg",
            terrain_attempt_index=50,
            dynamics="randomized",
        )
        draw = TaskRandomization(task, evaluation=True, nominal_heading=-1.2)
        physical = draw.dynamics()
        start = draw.sample_start(None)
        noise = draw.uniform(None, 30, "observation-noise")
        draw.sample_start([0])
        draw.uniform(None, 30, "observation-noise")
        draw.begin_attempts()
        self.assertIsNone(draw.manifest()["attempts"][0]["start"])
        self.assertEqual(draw.manifest()["attempts"][0]["attempt_index"], 50)
        replayed = draw.sample_start(None)
        for name, expected in start.items():
            np.testing.assert_array_equal(replayed[name], expected)
        recorded = draw.manifest()["attempts"][0]["start"]
        draw.sample_start([0])
        self.assertEqual(draw.manifest()["attempts"][0]["start"], recorded)
        np.testing.assert_array_equal(
            draw.uniform(None, 30, "observation-noise"), noise
        )
        for name, expected in physical.items():
            np.testing.assert_array_equal(draw.dynamics()[name], expected)
        with self.assertRaises(ValueError):
            draw.uniform(None, 4, "commands")

    def test_connected_selector_rejects_invalid_or_competing_sampler_settings(self):
        task = dict(
            terrain="connected",
            num_envs=1,
            seed=0,
            terrain_group="stairs/up/04cm",
            terrain_attempt_index=0,
        )
        invalid = (
            dict(terrain_group=None),
            dict(terrain_attempt_index=None),
            dict(terrain_group="qualification"),
            dict(terrain_group=True),
            dict(terrain_attempt_index=True),
            dict(terrain_attempt_index=0.0),
            dict(terrain_attempt_index=-1),
            dict(terrain_attempt_index=100),
            dict(num_envs=2),
            dict(terrain="flat"),
            dict(bank_profile="stand"),
            dict(dynamics="randomized"),
            dict(dynamics="invalid"),
        )
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                TaskRandomization(
                    {**task, **change}, evaluation=True, nominal_heading=0.7
                )
        for evaluation, heading in ((False, 0.7), (True, None), (True, np.nan)):
            with (
                self.subTest(evaluation=evaluation, heading=heading),
                self.assertRaises(ValueError),
            ):
                TaskRandomization(task, evaluation=evaluation, nominal_heading=heading)

    def test_nominal_world_heading_rotates_starts_without_changing_streams(self):
        task = dict(num_envs=2, seed=42, dynamics="nominal")
        flat = TaskRandomization(task, evaluation=True)
        world = TaskRandomization(task, evaluation=True, nominal_heading=1.2)
        a, b = flat.sample_start(None), world.sample_start(None)
        rotation = np.array([[np.cos(1.2), -np.sin(1.2)], [np.sin(1.2), np.cos(1.2)]])
        np.testing.assert_allclose(b["start_xy"], a["start_xy"] @ rotation.T)
        np.testing.assert_array_equal(b["start_xy_local"], a["start_xy_local"])
        np.testing.assert_array_equal(b["joint_offset"], a["joint_offset"])
        np.testing.assert_array_equal(b["start_yaw"], a["start_yaw"] + 1.2)
        np.testing.assert_array_equal(
            world.uniform(None, 30, "observation-noise"),
            flat.uniform(None, 30, "observation-noise"),
        )
        first = world.manifest()["attempts"][0]["start"]
        self.assertEqual(first["nominal_heading"], 1.2)
        self.assertEqual(first["start_xy"], b["start_xy"][0].tolist())

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

    def test_development_ids_have_balanced_strata_and_separate_namespaces(self):
        bank = sampler(bank="stand", count=100, evaluation=True)
        nominal, randomized = bank.randomized[:50], bank.randomized[50:]
        self.assertFalse(nominal.any())
        self.assertTrue(randomized.all())
        manifest = bank.manifest()
        self.assertEqual(manifest["namespace"], f"{NAMESPACE}/development")
        self.assertEqual(manifest["group_id"], "flat/stand")
        self.assertEqual(
            [row["attempt_index"] for row in manifest["attempts"]], list(range(100))
        )
        self.assertNotEqual(sampler().namespace, bank.namespace)
        # Candidate training seed cannot silently select different bank draws.
        other = sampler(bank="stand", count=100, seed=999, evaluation=True)
        np.testing.assert_array_equal(
            bank.uniform(None, 30, "observation-noise"),
            other.uniform(None, 30, "observation-noise"),
        )

    def test_dynamics_are_independent_bounded_and_never_accumulate(self):
        draw = sampler(bank="stand", count=100, evaluation=True)
        physical = draw.dynamics()
        for key in ("motor_strength", "kp_scale", "kd_scale"):
            np.testing.assert_array_equal(physical[key][:50], np.ones((50, 12)))
            self.assertTrue(
                ((physical[key][50:] >= 0.9) & (physical[key][50:] <= 1.1)).all()
            )
        np.testing.assert_array_equal(physical["added_mass"][:50], np.zeros(50))
        np.testing.assert_array_equal(
            physical["static_friction"][:50], np.full(50, 0.8)
        )
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
        self.assertFalse(
            np.array_equal(physical["kp_scale"][50:], physical["kd_scale"][50:])
        )
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

    def test_bank_heading_stratification_and_local_start_frame(self):
        draw = sampler(bank="arc-left", count=100, evaluation=True)
        start = draw.sample_start(None)
        bins = np.floor((start["nominal_heading"] + np.pi) * 50 / (2 * np.pi)).astype(
            int
        )
        np.testing.assert_array_equal(bins, np.arange(100) % 50)
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

    def test_explicit_attempt_start_discards_warmup_not_physical_parameters(self):
        draw = sampler(bank="stand", count=100, evaluation=True)
        physical = draw.dynamics()
        initial = draw.sample_start(None)
        noise = draw.uniform(None, 30, "observation-noise")
        commands = draw.uniform(None, 4, "commands")
        draw.sample_start([3, 70])
        draw.begin_attempts()
        self.assertTrue(
            all(row["start"] is None for row in draw.manifest()["attempts"])
        )
        np.testing.assert_array_equal(
            draw.sample_start(None)["start_yaw"], initial["start_yaw"]
        )
        np.testing.assert_array_equal(
            draw.uniform(None, 30, "observation-noise"), noise
        )
        np.testing.assert_array_equal(draw.uniform(None, 4, "commands"), commands)
        np.testing.assert_array_equal(draw.dynamics()["kp_scale"], physical["kp_scale"])

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
        with self.assertRaises(ValueError):
            sampler(bank="stand", count=100)
        with self.assertRaises(ValueError):
            sampler(bank="stand", evaluation=True)

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
