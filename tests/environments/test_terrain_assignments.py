"""Prospective development assignments; no mesh, simulator or robot execution."""

import hashlib
import json
import unittest
from collections import Counter
from copy import deepcopy
from inspect import signature
from itertools import product
from unittest.mock import patch

import numpy as np

from parkour_lab.environments import randomization
from parkour_lab.environments.terrains import worlds


class TerrainAssignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = randomization.terrain_development_assignments()

    def test_exact_groups_ids_strata_and_development_only(self):
        expected = {
            *(f"stairs/{d}/{h:02d}cm" for d in ("up", "down") for h in (4, 8, 12, 16)),
            *(f"ramp/{d}/{h}deg" for d in ("up", "down") for h in (10, 15, 20)),
            *(
                f"hill/{d}/{h}deg"
                for d in ("short-first", "long-first")
                for h in (10, 15, 20)
            ),
        }
        self.assertEqual(self.plan["namespace"], "go2-roa-exit-v2-r6/development")
        self.assertEqual({g["group_id"] for g in self.plan["groups"]}, expected)
        self.assertEqual(len(self.plan["groups"]), 20)
        for group in self.plan["groups"]:
            self.assertEqual(
                [a["attempt_index"] for a in group["attempts"]], list(range(100))
            )
            self.assertEqual(
                [a["stratum"] for a in group["attempts"]],
                ["nominal"] * 50 + ["randomized"] * 50,
            )
        for invalid in (True, 7, [], "flat/stand", "stairs/up/4cm", "qualification"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                randomization.terrain_development_assignments(invalid)
        for override in (
            dict(namespace="qualification"),
            dict(seed=42),
            dict(attempt_index=0),
        ):
            with self.subTest(override=override), self.assertRaises(TypeError):
                randomization.terrain_development_assignments(**override)

    def test_selected_attempt_preserves_original_global_id_and_detached_assignment(
        self,
    ):
        metadata = (
            "namespace",
            "backend",
            "control_hz",
            "command_phases",
            "prospective_only",
            "native_validated",
            "qualification_eligible",
        )
        for group in self.plan["groups"]:
            for index in (0, 49, 50, 99):
                with self.subTest(group=group["group_id"], index=index):
                    selected = randomization.terrain_development_attempt(
                        group["group_id"], index
                    )
                    self.assertEqual(
                        selected,
                        {
                            **{name: self.plan[name] for name in metadata},
                            "group_id": group["group_id"],
                            **group["attempts"][index],
                        },
                    )
        selected["world"]["size"][0] = 99
        selected["streams"].clear()
        selected["command_phases"][0][1][0] = 99
        self.assertEqual(randomization.terrain_development_assignments(), self.plan)
        self.assertEqual(
            randomization.terrain_development_attempt(group["group_id"], 99)["streams"],
            group["attempts"][99]["streams"],
        )
        invalid = (
            (None, 0),
            (True, 0),
            ([], 0),
            ("qualification", 0),
            ("stairs/up/04cm", None),
            ("stairs/up/04cm", True),
            ("stairs/up/04cm", 0.0),
            ("stairs/up/04cm", -1),
            ("stairs/up/04cm", 100),
        )
        for group, index in invalid:
            with self.subTest(group=group, index=index), self.assertRaises(ValueError):
                randomization.terrain_development_attempt(group, index)

    def test_all_six_full_hashes_backend_mappings_and_roughness_seeds(self):
        streams = {
            "geometry",
            "rough-coarse",
            "rough-fine",
            "start",
            "dynamics",
            "observation-noise",
        }
        digests, backends = set(), set()
        for group in self.plan["groups"]:
            for attempt in group["attempts"]:
                self.assertEqual(set(attempt["streams"]), streams)
                for name, mapping in attempt["streams"].items():
                    identity = f"go2-roa-exit-v2-r6/development|{group['group_id']}|{attempt['attempt_index']}|{name}"
                    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
                    self.assertEqual(mapping["sha256"], digest)
                    expected = np.random.PCG64(int(digest, 16)).state["state"]
                    self.assertEqual(mapping["initial_state"], expected)
                    self.assertNotIn(digest, digests)
                    backend = (expected["state"], expected["inc"])
                    self.assertNotIn(backend, backends)
                    digests.add(digest)
                    backends.add(backend)
                for field, name in (
                    ("coarse_seed", "rough-coarse"),
                    ("fine_seed", "rough-fine"),
                ):
                    self.assertEqual(
                        attempt["world"][field],
                        int(attempt["streams"][name]["sha256"], 16),
                    )
                    self.assertGreater(attempt["world"][field], 2**32)
        self.assertEqual(len(digests), 12000)
        self.assertEqual(self.plan["seed_collisions"], 0)

    def test_digest_and_backend_collisions_reject_across_group_boundary(self):
        original = randomization._stream
        for field in ("sha256", "initial_state"):
            first = None

            def colliding(namespace, group, index, name):
                nonlocal first
                generator, mapping = original(namespace, group, index, name)
                if first is None:
                    first = deepcopy(mapping[field])
                elif (
                    group == randomization.TERRAIN_GROUPS[1]
                    and index == 0
                    and name == "geometry"
                ):
                    mapping[field] = first
                return generator, mapping

            with (
                self.subTest(field=field),
                patch.object(randomization, "_stream", side_effect=colliding),
            ):
                with self.assertRaisesRegex(ValueError, "collision"):
                    randomization.terrain_development_assignments()

    def test_each_stratum_balances_shapes_and_stratifies_tread_and_yaw(self):
        for group in self.plan["groups"]:
            family, direction, tier = group["group_id"].split("/")
            for offset in (0, 50):
                attempts = group["attempts"][offset : offset + 50]
                yaw_bins, tread_bins = [], []
                for attempt in attempts:
                    world = attempt["world"]
                    self.assertEqual(world["target_family"], family)
                    self.assertEqual(
                        world["reverse"], direction in ("down", "long-first")
                    )
                    self.assertEqual(
                        world["tier"],
                        int(tier[:2]) / (100 if family == "stairs" else 1),
                    )
                    self.assertGreaterEqual(world["world_yaw"], -np.pi)
                    self.assertLess(world["world_yaw"], np.pi)
                    yaw_bin = int(
                        np.floor((world["world_yaw"] + np.pi) * 50 / (2 * np.pi))
                    )
                    self.assertEqual(yaw_bin, attempt["geometry_bins"]["world_yaw"])
                    yaw_bins.append(yaw_bin)
                    if family == "stairs":
                        self.assertGreaterEqual(world["tread"], 0.31)
                        self.assertLess(world["tread"], 0.50)
                        tread_bin = int(np.floor((world["tread"] - 0.31) * 50 / 0.19))
                        self.assertEqual(tread_bin, attempt["geometry_bins"]["tread"])
                        tread_bins.append(tread_bin)
                    else:
                        self.assertNotIn("risers", world)
                        self.assertNotIn("tread", world)
                    if family != "ramp":
                        self.assertNotIn("incline_length", world)
                self.assertEqual(sorted(yaw_bins), list(range(50)))
                if family == "stairs":
                    self.assertEqual(sorted(tread_bins), list(range(50)))
                    self.assertEqual(
                        Counter(a["world"]["risers"] for a in attempts),
                        {4: 17, 5: 17, 6: 16},
                    )
                    self.assertNotEqual(tread_bins, yaw_bins)
                elif family == "ramp":
                    self.assertEqual(
                        Counter(a["world"]["incline_length"] for a in attempts),
                        {2: 25, 3: 25},
                    )

    def test_geometry_draw_order_reproduces_ranks_jitters_and_common_placement(self):
        group = self.plan["groups"][0]
        for offset in (0, 50):
            attempts = group["attempts"][offset : offset + 50]
            draws = np.array(
                [
                    np.random.Generator(
                        np.random.PCG64(int(a["streams"]["geometry"]["sha256"], 16))
                    ).uniform(size=7)
                    for a in attempts
                ]
            )
            # Rank independently without copying the implementation's argsort pair.
            for index, attempt in enumerate(attempts):
                world, values = attempt["world"], draws[index]
                rank = [
                    int((draws[:, column] < values[column]).sum())
                    for column in (0, 1, 3)
                ]
                self.assertEqual(world["risers"], 4 + rank[0] % 3)
                self.assertEqual(
                    world["tread"], 0.31 + 0.19 * (rank[1] + values[2]) / 50
                )
                self.assertEqual(
                    world["world_yaw"], -np.pi + 2 * np.pi * (rank[2] + values[4]) / 50
                )
                dx, dy = -0.25 + 0.5 * values[5:7]
                for field, origin, shift in (
                    ("stair_entry", 14, dx),
                    ("hill_entry", 4, dx),
                    ("ramp_route_x", 7, dx),
                    ("ramp_entry", 14, dy),
                    ("stair_route_y", 11, dy),
                    ("hill_center_y", 4, dy),
                ):
                    self.assertEqual(world[field], origin + shift)
                    self.assertLessEqual(abs(world[field] - origin), 0.25)

    def test_every_group_world_kwargs_bind_and_json_preserves_full_integers(self):
        self.assertEqual(json.loads(json.dumps(self.plan, allow_nan=False)), self.plan)
        for group in self.plan["groups"]:
            for attempt in group["attempts"]:
                signature(worlds.build_world).bind(**attempt["world"])
                self.assertNotIn("start", attempt)
                self.assertNotIn("dynamics", attempt)

    def test_single_group_matches_full_repeat_and_mutation_isolation(self):
        for group in reversed(self.plan["groups"]):
            single = randomization.terrain_development_assignments(group["group_id"])
            self.assertEqual(single["groups"], [group])
        single["groups"][0]["attempts"][0]["streams"].clear()
        single["command_phases"][0][1][0] = 99
        self.assertEqual(randomization.terrain_development_assignments(), self.plan)

    def test_global_rng_untouched_and_no_mesh_building(self):
        state = np.random.get_state()
        try:
            np.random.seed(876)
            expected = np.random.uniform(size=10)
            np.random.seed(876)
            with patch.object(
                worlds,
                "build_world",
                side_effect=AssertionError("mesh generation forbidden"),
            ):
                plan = randomization.terrain_development_assignments(
                    "hill/long-first/20deg"
                )
            np.testing.assert_array_equal(np.random.uniform(size=10), expected)
        finally:
            np.random.set_state(state)
        self.assertEqual(plan["status"], "PROSPECTIVE_ASSIGNMENTS_NOT_VALIDATED")
        self.assertTrue(plan["prospective_only"])
        self.assertFalse(plan["native_validated"])
        self.assertFalse(plan["qualification_eligible"])
        self.assertEqual(
            plan["command_phases"],
            [[2.0, [0.0, 0.0, 0.0]], [25.0, [0.35, 0.0, 0.0]], [3.0, [0.0, 0.0, 0.0]]],
        )
        self.assertEqual(
            plan["control_hz"] * sum(p[0] for p in plan["command_phases"]), 1500
        )

    def test_nominal_trial_margins_at_placement_corners_and_shape_extrema(self):
        for group in self.plan["groups"]:
            reference = group["attempts"][0]["world"]
            family = reference["target_family"]
            shapes = (
                [dict(risers=r, tread=t) for r, t in product((4, 5, 6), (0.31, 0.5))]
                if family == "stairs"
                else (
                    [dict(incline_length=n) for n in (2, 3)]
                    if family == "ramp"
                    else [{}]
                )
            )
            for dx, dy, shape in product((-0.25, 0.25), (-0.25, 0.25), shapes):
                kwargs = {
                    **reference,
                    **shape,
                    "stair_entry": 14 + dx,
                    "ramp_entry": 14 + dy,
                    "stair_route_y": 11 + dy,
                    "ramp_route_x": 7 + dx,
                    "hill_entry": 4 + dx,
                    "hill_center_y": 4 + dy,
                }
                bound = signature(worlds.build_world).bind(**kwargs)
                bound.apply_defaults()
                values = bound.arguments
                lips, _ = worlds._stair_lips(
                    values["stair_entry"], values["tread"], values["risers"]
                )
                stair_x, stair_y = lips[0], values["stair_route_y"]
                ramp_x, ramp_y = values["ramp_route_x"], values["ramp_entry"]
                hill_x, hill_y = values["hill_entry"], values["hill_center_y"]
                ramp_end = ramp_y + values["incline_length"]
                size = np.array(values["size"])
                features = {
                    "stairs": dict(
                        axis=0,
                        entry=np.array([stair_x, stair_y]),
                        exit=np.array([lips[-1], stair_y]),
                        footprint=np.array([[stair_x, 0], [lips[-1], ramp_y]]),
                        influence=np.array([[stair_x, 0], size]),
                    ),
                    "ramp": dict(
                        axis=1,
                        entry=np.array([ramp_x, ramp_y]),
                        exit=np.array([ramp_x, ramp_end]),
                        footprint=np.array(
                            [[ramp_x - 1.5, ramp_y], [ramp_x + 1.5, ramp_end]]
                        ),
                        influence=np.array([[0, ramp_y], size]),
                    ),
                    "hill": dict(
                        axis=0,
                        entry=np.array([hill_x, hill_y]),
                        exit=np.array([hill_x + 2.9, hill_y]),
                        footprint=np.array(
                            [[hill_x, hill_y - 1], [hill_x + 2.9, hill_y + 1]]
                        ),
                        influence=np.array(
                            [[hill_x, hill_y - 3], [hill_x + 2.9, hill_y + 3]]
                        ),
                    ),
                }
                with self.subTest(group=group["group_id"], dx=dx, dy=dy, shape=shape):
                    trial = worlds._trial(
                        family, values["reverse"], features, size, values["world_yaw"]
                    )
                    for key in (
                        "minimum_map_margin_m",
                        "minimum_target_side_margin_m",
                        "target_clearance_at_nominal_finish_m",
                    ):
                        self.assertGreater(trial[key], 0)


if __name__ == "__main__":
    unittest.main()
