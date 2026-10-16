"""CPU geometry and native-mesh contracts; no robot traversal or safety claims."""

import hashlib
import json
import math
import sys
import unittest
from copy import deepcopy
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

import parkour_lab.environments.terrains.traversal as terrain
from tests.support.isaaclab import pinned_generator


class TraversalGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surfaces = {
            profile: terrain.build_surface(profile, 47)
            for profile in dict.fromkeys(
                (*terrain.PROFILES, *terrain.profiles_for_layout("step_ladder"))
            )
        }

    def grid(self, profile):
        vertices, faces, metadata = self.surfaces[profile]
        grid = vertices.reshape(-1, 321, 3)
        return grid[:, 0, 0] - 8.0, grid[:, :, 2], vertices, faces, metadata

    def test_four_profiles_are_reproducible_seeded_and_do_not_consume_global_rng(self):
        self.assertEqual(
            terrain.PROFILES, ("ramp_6deg", "ramp_12deg", "step_08m", "step_16m")
        )
        original_rng = np.random.get_state()
        self.addCleanup(np.random.set_state, original_rng)
        np.random.seed(123)
        seeded_rng = np.random.get_state()
        for profile in self.surfaces:
            with self.subTest(profile=profile):
                first = self.surfaces[profile]
                second = terrain.build_surface(profile, 47)
                other = terrain.build_surface(profile, 48)
                np.testing.assert_array_equal(first[0], second[0])
                np.testing.assert_array_equal(first[1], second[1])
                self.assertEqual(first[2], second[2])
                np.testing.assert_array_equal(first[0][:, :2], other[0][:, :2])
                np.testing.assert_array_equal(first[1], other[1])
                self.assertFalse(np.array_equal(first[0][:, 2], other[0][:, 2]))
                self.assertNotEqual(
                    first[2]["vertices_float64_sha256"],
                    other[2]["vertices_float64_sha256"],
                )
        following = np.random.random(10)
        np.random.set_state(seeded_rng)
        np.testing.assert_array_equal(following, np.random.random(10))

    def test_steps_have_exact_full_width_vertical_up_and_down_risers(self):
        for profile, height in (
            (f"step_{cm:02d}m", cm / 100) for cm in (2, 4, 6, 8, 16)
        ):
            with self.subTest(profile=profile):
                x, heights, vertices, faces, metadata = self.grid(profile)
                x = np.round(x, 12)
                self.assertEqual(heights.shape, (323, 321))
                self.assertEqual(metadata["nominal_height_m"], height)
                self.assertTrue(metadata["vertical_risers"])
                self.assertIsNone(metadata["nominal_incline_degrees"])
                for coordinate, expected in (
                    (1.0, (0.0, height)),
                    (2.6, (height, 0.0)),
                ):
                    rows = np.flatnonzero(x == coordinate)
                    self.assertEqual(len(rows), 2)
                    np.testing.assert_array_equal(
                        heights[rows], np.repeat(np.array(expected)[:, None], 321, 1)
                    )
                triangles = vertices[faces]
                normals = np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                )
                vertical = np.abs(normals[:, 2]) < 1e-12
                self.assertEqual(int(vertical.sum()), 2 * 2 * 320)
                self.assertEqual(metadata["vertical_face_count"], int(vertical.sum()))
                self.assertTrue((np.linalg.norm(normals, axis=1) > 0).all())
                for lip_x, sign in ((9.0, -1), (10.6, 1)):
                    lip = vertical & np.isclose(triangles[:, 0, 0], lip_x)
                    self.assertEqual(int(lip.sum()), 640)
                    self.assertTrue((sign * normals[lip, 0] > 0).all())
                    np.testing.assert_allclose(normals[lip, 1:], 0.0, atol=1e-15)
                    np.testing.assert_allclose(triangles[lip, :, 0], lip_x, atol=1e-12)
                    np.testing.assert_allclose(
                        np.ptp(triangles[lip, :, 2], axis=1), height, atol=1e-15
                    )

    def test_ramps_have_exact_nominal_six_and_twelve_degree_seams(self):
        for profile, degrees in (("ramp_6deg", 6.0), ("ramp_12deg", 12.0)):
            with self.subTest(profile=profile):
                x, heights, _, _, metadata = self.grid(profile)
                x = np.round(x, 12)
                height = 1.2 * math.tan(math.radians(degrees))
                self.assertEqual(heights.shape, (321, 321))
                self.assertFalse(metadata["vertical_risers"])
                self.assertEqual(metadata["vertical_face_count"], 0)
                self.assertEqual(metadata["nominal_incline_degrees"], degrees)
                self.assertAlmostEqual(metadata["nominal_height_m"], height, places=15)
                for coordinate, expected in (
                    (1.0, 0),
                    (2.2, height),
                    (3.2, height),
                    (4.4, 0),
                ):
                    row = heights[np.flatnonzero(x == coordinate)[0]]
                    np.testing.assert_allclose(row, expected, rtol=0, atol=2e-15)
                actual_grade = height / (
                    metadata["top_start_x_m"] - metadata["entry_x_m"]
                )
                self.assertAlmostEqual(math.degrees(math.atan(actual_grade)), degrees)

    def test_roughness_is_bounded_tapered_at_seams_and_not_a_flat_lateral_bypass(self):
        for profile in self.surfaces:
            with self.subTest(profile=profile):
                x, heights, _, _, metadata = self.grid(profile)
                x = np.round(x, 12)
                height = metadata["nominal_height_m"]
                if metadata["vertical_risers"]:
                    macro = np.zeros(len(x))
                    first = np.flatnonzero(x == 1.0)[-1]
                    last = np.flatnonzero(x == 2.6)[0]
                    macro[first : last + 1] = height
                else:
                    macro = height * np.minimum(
                        np.clip((x - 1.0) / 1.2, 0, 1),
                        np.clip((4.4 - x) / 1.2, 0, 1),
                    )
                residual = heights - macro[:, None]
                self.assertLessEqual(np.abs(residual).max(), 0.008 + 2e-15)
                self.assertGreater(np.abs(residual).max(), 0.001)
                self.assertGreater(np.std(np.diff(residual, axis=1)), 1e-5)
                seams = {
                    metadata[name]
                    for name in (
                        "entry_x_m",
                        "top_start_x_m",
                        "top_end_x_m",
                        "exit_x_m",
                    )
                }
                for seam in seams:
                    np.testing.assert_allclose(
                        residual[x == seam], 0.0, rtol=0, atol=2e-15
                    )
                outside = (x < metadata["entry_x_m"]) | (x > metadata["exit_x_m"])
                np.testing.assert_array_equal(heights[outside], 0.0)
                # Every lateral column, including both tile edges, must cross
                # the obstacle; there is no route around it inside this tile.
                self.assertTrue((heights.max(axis=0) >= height - 0.008).all())
                top = (x > metadata["top_start_x_m"]) & (x < metadata["top_end_x_m"])
                self.assertTrue((heights[top] >= height - 0.008 - 2e-15).all())
                self.assertEqual(metadata["full_width_m"], 16.0)

    def test_supported_mesh_covers_every_grid_cell_with_nondegenerate_faces(self):
        for profile in self.surfaces:
            with self.subTest(profile=profile):
                _, heights, vertices, faces, metadata = self.grid(profile)
                rows, columns = heights.shape
                self.assertEqual(len(vertices), rows * columns)
                self.assertEqual(len(faces), 2 * (rows - 1) * (columns - 1))
                ids = np.arange(rows * columns).reshape(rows, columns)
                cells = faces.reshape(rows - 1, columns - 1, 2, 3)
                np.testing.assert_array_equal(
                    cells[:, :, 0],
                    np.stack((ids[:-1, :-1], ids[1:, :-1], ids[:-1, 1:]), axis=-1),
                )
                np.testing.assert_array_equal(
                    cells[:, :, 1],
                    np.stack((ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]), axis=-1),
                )
                triangles = vertices[faces]
                normals = np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                )
                self.assertTrue((np.linalg.norm(normals, axis=1) > 0).all())
                self.assertTrue((normals[:, 2] >= 0).all())
                self.assertAlmostEqual(normals[:, 2].sum() / 2, 16.0 * 16.0, places=10)
                nonvertical = normals[:, 2] > 1e-12
                grade = (
                    np.linalg.norm(normals[nonvertical, :2], axis=1)
                    / normals[nonvertical, 2]
                ).max()
                self.assertAlmostEqual(
                    metadata["maximum_nonvertical_triangle_grade"], grade
                )

    def test_receipts_bind_exact_arrays_and_remain_finite_json(self):
        for profile, (vertices, faces, metadata) in self.surfaces.items():
            with self.subTest(profile=profile):
                self.assertEqual(metadata["profile"], profile)
                self.assertEqual(metadata["version"], terrain.VERSION)
                self.assertEqual(metadata["seed"], 47)
                self.assertEqual(metadata["origin_m"], [8.0, 8.0, 0.0])
                self.assertEqual(metadata["mesh_vertex_count"], len(vertices))
                self.assertEqual(metadata["mesh_face_count"], len(faces))
                self.assertEqual(metadata["minimum_height_m"], vertices[:, 2].min())
                self.assertEqual(metadata["maximum_height_m"], vertices[:, 2].max())
                self.assertEqual(
                    metadata["vertices_float64_sha256"],
                    hashlib.sha256(vertices.astype("<f8").tobytes()).hexdigest(),
                )
                self.assertEqual(
                    metadata["faces_int64_sha256"],
                    hashlib.sha256(faces.astype("<i8").tobytes()).hexdigest(),
                )
                json.dumps(metadata, allow_nan=False)
        self.assertEqual(
            terrain.preflight(47), [self.surfaces[p][2] for p in terrain.PROFILES]
        )

    def test_invalid_profile_and_seed_are_rejected(self):
        for profile in ("", "stairs", "step_8m", None, 6):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                terrain.build_surface(profile, 47)
        for seed in (-1, True, np.bool_(False), 47.0, "47", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                terrain.build_surface("step_08m", seed)

    def test_layouts_share_eight_centimeter_anchor_and_match_geometry_specification(
        self,
    ):
        standard = terrain.preflight(47)
        ladder = terrain.preflight(47, layout="step_ladder")
        self.assertEqual(terrain.profiles_for_layout("standard"), terrain.PROFILES)
        self.assertEqual(
            tuple(item["profile"] for item in ladder),
            ("step_02m", "step_04m", "step_06m", "step_08m"),
        )
        self.assertEqual(ladder[-1], standard[2])

        # Independent geometry specification, using the same numerical
        # platform rather than assuming cross-platform float64 mesh hashes.
        def reference_geometry(profile):
            ramp = profile.startswith("ramp")
            angle = (6.0 if profile == "ramp_6deg" else 12.0) if ramp else None
            return dict(
                profile=profile,
                entry_x_m=1.0,
                top_start_x_m=2.2 if ramp else 1.0,
                top_end_x_m=3.2 if ramp else 2.6,
                exit_x_m=4.4 if ramp else 2.6,
                nominal_height_m=(
                    1.2 * math.tan(math.radians(angle))
                    if ramp
                    else 0.08 if profile == "step_08m" else 0.16
                ),
                nominal_incline_degrees=angle,
                vertical_risers=not ramp,
                full_width_m=16.0,
            )

        with patch.object(terrain, "geometry", side_effect=reference_geometry):
            for profile in terrain.PROFILES:
                reference = terrain.build_surface(profile, 47)
                for actual, expected in zip(self.surfaces[profile][:2], reference[:2]):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(self.surfaces[profile][2], reference[2])
        for layout in (None, "", "unknown", 0, [], True):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                terrain.preflight(47, layout=layout)


class TraversalNativeTests(unittest.TestCase):
    layout = "standard"

    @classmethod
    def setUpClass(cls):
        cls.profiles = terrain.profiles_for_layout(cls.layout)
        cls.expected = terrain.preflight(47, layout=cls.layout)

    def setUp(self):
        self.receipts = patch.object(terrain, "_RECEIPTS", [])
        self.receipts.start()
        self.addCleanup(self.receipts.stop)

    def config(self, index=0):
        return SimpleNamespace(
            size=(16.0, 16.0),
            seed=47,
            profile=self.profiles[index],
            expected_geometry=deepcopy(self.expected[index]),
        )

    def test_native_callbacks_use_real_trimesh_and_exact_cpu_arrays(self):
        import trimesh

        for index, profile in enumerate(self.profiles):
            with self.subTest(profile=profile):
                meshes, origin = terrain.traversal_terrain(1.0, self.config(index))
                self.assertEqual(len(meshes), 1)
                self.assertIsInstance(meshes[0], trimesh.Trimesh)
                vertices, faces, metadata = terrain.build_surface(profile, 47)
                np.testing.assert_array_equal(meshes[0].vertices, vertices)
                np.testing.assert_array_equal(meshes[0].faces, faces)
                np.testing.assert_array_equal(origin, metadata["origin_m"])
        self.assertEqual(terrain.native_receipts(), self.expected)
        snapshot = terrain.native_receipts()
        snapshot[0]["origin_m"][0] = -1
        snapshot.clear()
        self.assertEqual(terrain.native_receipts(), self.expected)

    def test_mismatched_preflight_rejected_without_receipt(self):
        for defect in ("hash", "seed", "profile", "height", "missing"):
            cfg = self.config()
            if defect == "hash":
                cfg.expected_geometry["vertices_float64_sha256"] = "0" * 64
            elif defect == "seed":
                cfg.seed = 48
            elif defect == "profile":
                cfg.profile = self.profiles[1]
            elif defect == "height":
                cfg.expected_geometry["nominal_height_m"] += 0.001
            else:
                cfg.expected_geometry = {}
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                terrain.traversal_terrain(1.0, cfg)
            self.assertEqual(terrain.native_receipts(), [])

    def test_wrong_difficulty_or_size_rejected_without_receipt(self):
        for difficulty in (0.0, 0.5, 1.01, math.nan, math.inf, True, np.bool_(True)):
            with self.subTest(difficulty=difficulty), self.assertRaises(ValueError):
                terrain.traversal_terrain(difficulty, self.config())
            self.assertEqual(terrain.native_receipts(), [])
        for size in ((8, 16), (16, 8), (16,), (16, 16, 16), (16, math.nan)):
            cfg = self.config()
            cfg.size = size
            with self.subTest(size=size), self.assertRaises(ValueError):
                terrain.traversal_terrain(1.0, cfg)
            self.assertEqual(terrain.native_receipts(), [])

    def test_duplicate_profile_rejected_before_receipt_mutation(self):
        terrain.traversal_terrain(1.0, self.config())
        before = terrain.native_receipts()
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            terrain.traversal_terrain(1.0, self.config())
        self.assertEqual(terrain.native_receipts(), before)

    def test_fifth_callback_rejected_without_corrupting_completed_receipts(self):
        for index in range(4):
            terrain.traversal_terrain(1.0, self.config(index))
        before = terrain.native_receipts()
        with self.assertRaises(ValueError):
            terrain.traversal_terrain(1.0, self.config(0))
        self.assertEqual(terrain.native_receipts(), before)

    def test_configure_checks_preflight_and_preserves_unrelated_recipe(self):
        terrains = ModuleType("isaaclab.terrains")
        terrains.SubTerrainBaseCfg = SimpleNamespace
        terrains.TerrainGeneratorCfg = SimpleNamespace
        utilities = ModuleType("isaaclab.utils")
        utilities.configclass = lambda cls: cls
        isaaclab = ModuleType("isaaclab")
        isaaclab.terrains, isaaclab.utils = terrains, utilities
        cfg = SimpleNamespace(
            commands=SimpleNamespace(
                base_velocity=SimpleNamespace(ranges=SimpleNamespace())
            ),
            scene=SimpleNamespace(
                terrain=SimpleNamespace(
                    terrain_generator="old", max_init_terrain_level=8
                )
            ),
            events=SimpleNamespace(
                reset_base=SimpleNamespace(
                    params={
                        "pose_range": {"yaw": (-1, 1)},
                        "velocity_range": {"x": (-0.2, 0.2)},
                    }
                )
            ),
            terminations=SimpleNamespace(
                procedural_physical_failure=SimpleNamespace(
                    params={"minimum_surface_z_m": -1, "minimum_m": 0.12}
                )
            ),
            robot="unchanged_motor",
            rewards="unchanged_rewards",
            observations="unchanged_sensors",
        )
        with (
            patch(
                "parkour_lab.runtime.native.configure_external_command"
            ) as external_command,
            patch.dict(
                sys.modules,
                {
                    "isaaclab": isaaclab,
                    "isaaclab.terrains": terrains,
                    "isaaclab.utils": utilities,
                },
            ),
        ):
            bad = deepcopy(self.expected)
            bad[0]["seed"] = 48
            with self.assertRaisesRegex(ValueError, "preflight"):
                terrain.configure(cfg, 47, bad, layout=self.layout)
            self.assertEqual(cfg.scene.terrain.terrain_generator, "old")
            terrain._RECEIPTS.append({"previous_run": True})
            terrain.configure(cfg, 47, deepcopy(self.expected), layout=self.layout)
        external_command.assert_called_once_with(cfg.commands.base_velocity)
        self.assertEqual(terrain.native_receipts(), [])
        generated = cfg.scene.terrain.terrain_generator
        self.assertEqual(
            (generated.seed, generated.num_rows, generated.num_cols), (47, 1, 4)
        )
        self.assertEqual(generated.size, (16.0, 16.0))
        self.assertEqual(generated.difficulty_range, (1.0, 1.0))
        self.assertTrue(generated.curriculum)
        self.assertFalse(generated.use_cache)
        self.assertEqual(list(generated.sub_terrains), list(self.profiles))
        for expected in self.expected:
            subterrain = generated.sub_terrains[expected["profile"]]
            self.assertEqual(subterrain.proportion, 0.25)
            self.assertEqual(subterrain.expected_geometry, expected)
        self.assertEqual(cfg.scene.terrain.max_init_terrain_level, 0)
        self.assertEqual(
            cfg.events.reset_base.params["pose_range"],
            {name: (0.0, 0.0) for name in ("x", "y", "yaw")},
        )
        self.assertEqual(
            cfg.events.reset_base.params["velocity_range"], {"x": (-0.2, 0.2)}
        )
        self.assertEqual(
            cfg.terminations.procedural_physical_failure.params,
            {"minimum_surface_z_m": -0.008, "minimum_m": 0.12},
        )
        self.assertEqual(
            (cfg.robot, cfg.rewards, cfg.observations),
            ("unchanged_motor", "unchanged_rewards", "unchanged_sensors"),
        )

    def test_pinned_native_config_copy_and_generator_preserve_column_profiles(self):
        with (
            pinned_generator() as (configclass, base, Generator),
            patch("parkour_lab.runtime.native.configure_external_command"),
        ):
            namespace = sys.modules["isaaclab.terrains"]
            namespace.SubTerrainBaseCfg = base
            namespace.TerrainGeneratorCfg = SimpleNamespace
            cfg = SimpleNamespace(
                scene=SimpleNamespace(terrain=SimpleNamespace()),
                commands=SimpleNamespace(base_velocity=SimpleNamespace()),
                events=SimpleNamespace(reset_base=SimpleNamespace(params={})),
                terminations=SimpleNamespace(
                    procedural_physical_failure=SimpleNamespace(params={})
                ),
            )
            wrong_layout = "step_ladder" if self.layout == "standard" else "standard"
            with self.assertRaisesRegex(ValueError, "preflight"):
                terrain.configure(cfg, 47, self.expected, layout=wrong_layout)
            terrain.configure(cfg, 47, self.expected, layout=self.layout)
            generator = Generator.__new__(Generator)
            generator.cfg = cfg.scene.terrain.terrain_generator
            generator.cfg.cache_dir = "unused-cache-no-io"
            self.assertEqual(tuple(generator.cfg.sub_terrains), self.profiles)
            for profile, item in generator.cfg.sub_terrains.items():
                item.size = terrain.TILE_SIZE
                copied = item.copy()
                self.assertEqual(copied.profile, profile)
                self.assertEqual(copied.expected_geometry, item.expected_geometry)
                self.assertIs(copied.function, terrain.traversal_terrain)
                mesh, origin = generator._get_terrain_mesh(1.0, item)
                vertices, faces, receipt = terrain.build_surface(profile, 47)
                np.testing.assert_array_equal(mesh.vertices, vertices - [8, 8, 0])
                np.testing.assert_array_equal(mesh.faces, faces)
                np.testing.assert_array_equal(origin, [0, 0, 0])
                self.assertEqual(terrain.native_receipts()[-1], receipt)


class StepLadderNativeTests(TraversalNativeTests):
    """Apply the same native adapter contract to the alternative terrain layout."""

    layout = "step_ladder"


if __name__ == "__main__":
    unittest.main()
