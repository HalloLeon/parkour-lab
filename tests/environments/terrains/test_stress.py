"""CPU contracts for the separate, evaluation-only random-lattice stress mesh."""

import hashlib
import json
import math
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

import parkour_lab.environments.terrains.stress as stress
from parkour_lab.environments.terrains import procedural as training_terrain


class StressSurfaceTests(unittest.TestCase):
    def test_reproducible_different_seeds_and_global_rng_untouched(self):
        np.random.seed(19)
        state = np.random.get_state()
        first = stress.build_stress_surface(0.75, seed=47)
        following = np.random.random(7)
        np.random.set_state(state)
        np.testing.assert_array_equal(following, np.random.random(7))
        np.random.random(50)
        second = stress.build_stress_surface(0.75, seed=47)
        np.testing.assert_array_equal(first["heights"], second["heights"])
        self.assertEqual(first["metadata"], second["metadata"])
        other = stress.build_stress_surface(0.75, seed=48)
        self.assertFalse(np.array_equal(first["heights"], other["heights"]))
        self.assertNotEqual(
            first["metadata"]["height_float64_sha256"],
            other["metadata"]["height_float64_sha256"],
        )

    def test_difficulty_linearly_scales_same_field(self):
        for seed in (0, 47, 150):
            full = stress.build_stress_surface(1.0, seed=seed)
            for difficulty in (0.0, 0.05, 0.35, 0.75):
                with self.subTest(seed=seed, difficulty=difficulty):
                    surface = stress.build_stress_surface(difficulty, seed=seed)
                    np.testing.assert_allclose(
                        surface["heights"], full["heights"] * difficulty, atol=1e-16
                    )
                    self.assertTrue(surface["supported_cells"].all())
                    self.assertLessEqual(
                        surface["metadata"]["maximum_absolute_surface_height_m"],
                        0.08 * difficulty + 1e-12,
                    )
                    self.assertLessEqual(
                        surface["metadata"]["maximum_surface_grade"],
                        0.8 * difficulty + 1e-12,
                    )
                    if difficulty == 0:
                        np.testing.assert_array_equal(surface["heights"], 0.0)

    def test_all_flat_regions_exact_support_and_resolution(self):
        surface = stress.build_stress_surface(0.75, seed=47)
        self.assertEqual(surface["heights"].shape, (321, 321))
        self.assertEqual(surface["supported_cells"].shape, (320, 320))
        self.assertTrue(surface["supported_cells"].all())
        np.testing.assert_allclose(np.diff(surface["x"]), 0.05, atol=2e-15)
        np.testing.assert_allclose(np.diff(surface["y"]), 0.05, atol=2e-15)
        x, y = np.meshgrid(
            np.round(surface["x"] - 8, 12),
            np.round(surface["y"] - 8, 12),
            indexing="ij",
        )
        flat = (
            (np.maximum(np.abs(x), np.abs(y)) <= 1)
            | (np.abs(y) <= 0.6)
            | (np.abs(x) >= 7)
            | (np.abs(y) >= 7)
        )
        np.testing.assert_array_equal(surface["heights"][flat], 0.0)
        np.testing.assert_array_equal(surface["origin"], [8.0, 8.0, 0.0])
        self.assertEqual(surface["metadata"]["gap_area_fraction"], 0)
        self.assertEqual(surface["metadata"]["gap_rectangles"], [])

    def test_both_axes_vary_and_centerline_has_realized_relief(self):
        surface = stress.build_stress_surface(0.75, seed=47)
        for axis in (0, 1):
            self.assertGreater(np.std(np.diff(surface["heights"], axis=axis)), 0.001)
        route = surface["metadata"]["nominal_route_geometry"]
        self.assertEqual(route["samples"], 169)
        self.assertGreater(route["peak_to_peak_surface_height_m"], 0.03)
        self.assertGreater(route["root_mean_square_height_m"], 0.005)
        self.assertGreater(route["maximum_adjacent_triangle_grade"], 0.05)
        self.assertIn("not measured", route["scope"])

    def test_bounds_use_actual_mesh_triangles_including_tapers(self):
        for seed in (0, 1, 2, 47, 100):
            with self.subTest(seed=seed):
                surface = stress.build_stress_surface(1.0, seed=seed)
                vertices, faces = training_terrain.surface_mesh_arrays(surface)
                self.assertEqual(len(faces), 2 * 320 * 320)
                triangles = vertices[faces]
                normals = np.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                )
                self.assertTrue((normals[:, 2] > 0).all())
                mesh_grade = np.max(
                    np.linalg.norm(normals[:, :2], axis=1) / normals[:, 2]
                )
                self.assertAlmostEqual(
                    surface["metadata"]["maximum_surface_grade"], mesh_grade, places=12
                )
                self.assertLessEqual(mesh_grade, 0.8 + 1e-12)
                self.assertLessEqual(np.abs(vertices[:, 2]).max(), 0.08 + 1e-12)

    def test_provenance_records_measured_geometry_and_canonical_hash(self):
        surface = stress.build_stress_surface(0.75, seed=47)
        heights, metadata = surface["heights"], surface["metadata"]
        self.assertEqual(metadata["version"], "operator_stress_surface_v1")
        self.assertEqual(metadata["profile"], "rough_stress")
        self.assertTrue(metadata["evaluation_only"])
        self.assertEqual(metadata["lattice_spacings_m"], [2, 1, 0.5, 0.25])
        self.assertEqual(metadata["resolution_m"], 0.05)
        self.assertEqual(metadata["minimum_surface_height_m"], heights.min())
        self.assertEqual(metadata["maximum_surface_height_m"], heights.max())
        self.assertEqual(
            metadata["root_mean_square_height_m"], np.sqrt(np.mean(heights**2))
        )
        self.assertEqual(
            metadata["maximum_grid_edge_height_change_m"],
            max(np.abs(np.diff(heights, axis=axis)).max() for axis in (0, 1)),
        )
        self.assertEqual(
            metadata["height_float64_sha256"],
            hashlib.sha256(heights.astype("<f8").tobytes(order="C")).hexdigest(),
        )
        json.dumps(metadata, allow_nan=False)

    def test_invalid_inputs_fail_before_geometry(self):
        for difficulty in (
            -0.1,
            1.01,
            math.nan,
            math.inf,
            True,
            np.bool_(False),
            "0.5",
            0.5j,
            [0.5],
        ):
            with self.subTest(difficulty=difficulty), self.assertRaises(ValueError):
                stress.build_stress_surface(difficulty, seed=47)
        for seed in (-1, True, np.bool_(False), 1.0, "47"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                stress.build_stress_surface(0.75, seed=seed)
        for size in (
            None,
            16,
            (16,),
            (16, 16, 16),
            (16, 12),
            (16, 16.01),
            (math.inf, 16),
            (True, 16),
            ("16", 16),
        ):
            with self.subTest(size=size), self.assertRaises(ValueError):
                stress.build_stress_surface(0.75, seed=47, size=size)

    def test_native_callback_requires_matching_cpu_preflight_hash(self):
        stress.clear_stress_terrain_receipt()
        self.addCleanup(stress.clear_stress_terrain_receipt)
        self.assertIsNone(stress.stress_terrain_receipt())
        surface = stress.build_stress_surface(0.75, seed=47)
        cfg = SimpleNamespace(seed=47, size=(16, 16))
        for expected in (None, "bad", "0" * 64):
            cfg.expected_height_sha256 = expected
            with self.assertRaisesRegex(ValueError, "CPU-preflight hash"):
                stress.stress_terrain(0.75, cfg)
            self.assertIsNone(stress.stress_terrain_receipt())
        cfg.expected_height_sha256 = surface["metadata"]["height_float64_sha256"]
        with self.assertRaisesRegex(ValueError, "CPU-preflight hash"):
            stress.stress_terrain(0.5, cfg)
        cfg.seed = 48
        with self.assertRaisesRegex(ValueError, "CPU-preflight hash"):
            stress.stress_terrain(0.75, cfg)
        self.assertIsNone(stress.stress_terrain_receipt())
        cfg.seed = 47
        native_mesh = ModuleType("trimesh")
        native_mesh.Trimesh = lambda **kwargs: SimpleNamespace(**kwargs)
        native_module_name = "parkour_lab.environments.terrains.procedural"
        with patch.dict(
            sys.modules,
            {"trimesh": native_mesh, native_module_name: training_terrain},
        ):
            meshes, origin = stress.stress_terrain(0.75, cfg)
            receipt = stress.stress_terrain_receipt()
            self.assertEqual(receipt["calls"], 1)
            stress.stress_terrain(0.75, cfg)
            self.assertEqual(stress.stress_terrain_receipt()["calls"], 2)
        self.assertEqual(len(meshes), 1)
        self.assertFalse(meshes[0].process)
        self.assertEqual(meshes[0].vertices.shape, (321 * 321, 3))
        self.assertEqual(meshes[0].faces.shape, (2 * 320 * 320, 3))
        np.testing.assert_array_equal(origin, surface["origin"])
        self.assertEqual(receipt["seed"], 47)
        self.assertEqual(receipt["difficulty"], 0.75)
        self.assertEqual(receipt["height_float64_sha256"], cfg.expected_height_sha256)
        self.assertEqual(receipt["mesh_vertex_count"], 321 * 321)
        self.assertEqual(receipt["mesh_face_count"], 2 * 320 * 320)
        for key, values, dtype in (
            ("mesh_vertices_float64_sha256", meshes[0].vertices, "<f8"),
            ("mesh_faces_int64_sha256", meshes[0].faces, "<i8"),
        ):
            self.assertEqual(
                receipt[key], hashlib.sha256(values.astype(dtype).tobytes()).hexdigest()
            )
        receipt["calls"] = 123
        self.assertEqual(stress.stress_terrain_receipt()["calls"], 2)
        stress.clear_stress_terrain_receipt()
        self.assertIsNone(stress.stress_terrain_receipt())


if __name__ == "__main__":
    unittest.main()
