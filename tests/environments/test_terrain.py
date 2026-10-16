"""CPU geometry and native-adapter contracts, without Isaac Sim or training."""

import math
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from parkour_lab.environments import terrain


class ProceduralSurfaceTests(unittest.TestCase):
    def test_default_mixture_has_continuous_support_at_every_difficulty(self):
        expected = ("plane", "rough_flat", "hills", "step_hills", "tilted_ramps")
        self.assertEqual(
            terrain.PROFILE_BY_COLUMN,
            tuple(profile for profile in expected for _ in range(4)),
        )
        envelope = terrain.terrain_envelope()
        self.assertEqual(envelope["version"], "operator_procedural_surface_v2")
        self.assertFalse(envelope["gaps"]["enabled"])
        self.assertEqual(envelope["profile_by_column"], list(terrain.PROFILE_BY_COLUMN))
        self.assertEqual(
            set(envelope["height_and_grade_bounds_at_full_difficulty"]), set(expected)
        )
        # Check the full difficulty range, not just acquisition.
        for seed in (0, 43, 44):
            for difficulty in (0.0, 0.05, 0.15, 0.649, 0.65, 0.85, 1.0):
                for column, profile in enumerate(terrain.PROFILE_BY_COLUMN):
                    with self.subTest(seed=seed, difficulty=difficulty, column=column):
                        surface = terrain.build_operator_surface(
                            difficulty, profile, seed=seed, variant=column
                        )
                        self.assertTrue(surface["supported_cells"].all())
                        self.assertEqual(surface["metadata"]["gap_area_fraction"], 0)
                        self.assertEqual(surface["metadata"]["gap_rectangles"], [])

    def test_deterministic_and_isolated_from_global_numpy_rng(self):
        np.random.seed(17)
        state = np.random.get_state()
        a = terrain.build_operator_surface(0.5, "hills", seed=43, variant=2)
        following = np.random.random(4)
        np.random.set_state(state)
        np.testing.assert_array_equal(following, np.random.random(4))
        np.random.random(100)
        b = terrain.build_operator_surface(0.5, "hills", seed=43, variant=2)
        np.testing.assert_array_equal(a["heights"], b["heights"])
        np.testing.assert_array_equal(a["supported_cells"], b["supported_cells"])
        for change in ({"seed": 44, "variant": 2}, {"seed": 43, "variant": 3}):
            c = terrain.build_operator_surface(0.5, "hills", **change)
            self.assertFalse(np.array_equal(a["heights"], c["heights"]))

    def test_difficulty_scales_same_field_not_only_a_config_label(self):
        for profile in terrain.ENVELOPES:
            with self.subTest(profile=profile):
                zero = terrain.build_operator_surface(0.0, profile, seed=1)
                low = terrain.build_operator_surface(0.25, profile, seed=1)
                high = terrain.build_operator_surface(0.75, profile, seed=1)
                np.testing.assert_array_equal(zero["heights"], 0.0)
                self.assertTrue(zero["supported_cells"].all())
                np.testing.assert_allclose(
                    high["heights"], 3 * low["heights"], atol=1e-15
                )
                if profile != "plane":
                    self.assertGreater(np.std(low["heights"]), 0)

    def test_surface_envelope_includes_fractals_and_all_tapers(self):
        for seed in range(6):
            for profile, (height, grade) in terrain.ENVELOPES.items():
                with self.subTest(profile=profile, seed=seed):
                    result = terrain.build_operator_surface(1.0, profile, seed=seed)
                    self.assertLessEqual(
                        result["metadata"]["maximum_absolute_surface_height_m"],
                        height + 1e-12,
                    )
                    self.assertLessEqual(
                        result["metadata"]["maximum_surface_grade"], grade + 1e-12
                    )
                    self.assertTrue(np.isfinite(result["heights"]).all())
                    self.assertEqual(result["metadata"]["gap_area_fraction"], 0)

    def test_exact_flat_pad_band_and_matching_seams_have_support(self):
        for profile in terrain.ENVELOPES:
            result = terrain.build_operator_surface(1.0, profile, seed=31)
            x, y = np.meshgrid(result["x"] - 8, result["y"] - 8, indexing="ij")
            flat = (
                ((np.abs(x) <= 1) & (np.abs(y) <= 1))
                | (np.abs(y) <= 0.6 - 1e-12)
                | (np.abs(x) >= 7)
                | (np.abs(y) >= 7)
            )
            np.testing.assert_array_equal(result["heights"][flat], 0.0)
            cells = flat[:-1, :-1] & flat[1:, :-1] & flat[:-1, 1:] & flat[1:, 1:]
            self.assertTrue(result["supported_cells"][cells].all())
            np.testing.assert_array_equal(result["origin"], [8.0, 8.0, 0.0])

    def test_exact_plane_replay_is_not_positive_difficulty_row_zero(self):
        for difficulty in (0.0, 0.013, 0.5, 1.0):
            result = terrain.build_operator_surface(difficulty, "plane")
            np.testing.assert_array_equal(result["heights"], 0.0)
            self.assertTrue(result["supported_cells"].all())
        rough = terrain.build_operator_surface(0.013, "rough_flat")
        self.assertGreater(np.abs(rough["heights"]).max(), 0)

    def test_profiles_are_two_dimensional_and_distinct(self):
        fields = []
        for profile in ("rough_flat", "hills", "step_hills", "tilted_ramps"):
            surface = terrain.build_operator_surface(0.5, profile, seed=1)
            field = surface["heights"]
            self.assertGreater(surface["metadata"]["roughness_height_amplitude_m"], 0)
            self.assertGreater(np.std(np.diff(field, axis=0)), 0)
            self.assertGreater(np.std(np.diff(field, axis=1)), 0)
            fields.append(field)
        for i, field in enumerate(fields):
            for other in fields[i + 1 :]:
                self.assertFalse(np.allclose(field, other))

    def test_stepped_hills_report_realized_rise_not_vertical_stair_height(self):
        surface = terrain.build_operator_surface(1.0, "step_hills", seed=43)
        rise = max(
            np.abs(np.diff(surface["heights"], axis=axis)).max() for axis in (0, 1)
        )
        self.assertEqual(surface["metadata"]["maximum_grid_edge_height_change_m"], rise)
        self.assertGreater(rise, 0.005)
        self.assertLessEqual(
            rise, terrain.RESOLUTION * terrain.ENVELOPES["step_hills"][1] + 1e-12
        )
        self.assertIn(
            "not vertical stairs", terrain.terrain_envelope()["stepped_hills"]
        )

    def test_gap_profile_is_unsupported_at_every_difficulty(self):
        for difficulty in (0.0, 0.2, 0.649, 0.65, 0.9, 1.0):
            with self.assertRaisesRegex(ValueError, "Unknown operator terrain profile"):
                terrain.build_operator_surface(difficulty, "gaps")

    def test_mesh_has_exactly_two_upward_triangles_per_supported_cell(self):
        result = terrain.build_operator_surface(1.0, "hills", seed=3)
        vertices, faces = terrain.surface_mesh_arrays(result)
        self.assertTrue(np.isfinite(vertices).all())
        self.assertGreaterEqual(faces.min(), 0)
        self.assertLess(faces.max(), len(vertices))
        triangles = vertices[faces]
        normals = np.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        self.assertTrue((np.linalg.norm(normals, axis=-1) > 0).all())
        top_count = 2 * result["supported_cells"].sum()
        self.assertTrue((normals[:, 2] > 0).all())
        self.assertEqual(len(triangles), top_count)
        self.assertEqual(len(vertices), result["heights"].size)
        self.assertEqual(
            top_count,
            2 * (len(result["x"]) - 1) * (len(result["y"]) - 1),
        )

    def test_mesh_rejects_injected_holes(self):
        result = terrain.build_operator_surface(1.0, "hills", seed=7)
        result["supported_cells"][30, 30] = False
        with self.assertRaisesRegex(ValueError, "continuous support"):
            terrain.surface_mesh_arrays(result)

    def test_invalid_inputs_fail_before_geometry(self):
        for difficulty in (-0.1, 1.01, math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                terrain.build_operator_surface(difficulty, "rough_flat")
        for kwargs in (
            {"seed": -1},
            {"seed": True},
            {"variant": 1.2},
            {"size": (8.0, 16.0)},
            {"size": (16.01, 16.0)},
            {"size": (math.inf, 16)},
        ):
            with self.assertRaises(ValueError):
                terrain.build_operator_surface(0.5, "rough_flat", **kwargs)
        with self.assertRaises(ValueError):
            terrain.build_operator_surface(0.5, "route")

    def test_native_factory_has_exact_fixed_columns_and_no_cache(self):
        class Cfg:
            def __init__(self, **values):
                self.__dict__.update(values)

        native, utils = ModuleType("isaaclab.terrains"), ModuleType("isaaclab.utils")
        native.SubTerrainBaseCfg = native.TerrainGeneratorCfg = Cfg
        utils.configclass = lambda cls: cls
        with patch.dict(
            sys.modules, {"isaaclab.terrains": native, "isaaclab.utils": utils}
        ):
            cfg = terrain.make_operator_terrain_generator(
                seed=43, num_rows=1, difficulty_range=(0.05, 0.15)
            )
        self.assertTrue(cfg.curriculum)
        self.assertFalse(cfg.use_cache)
        self.assertEqual(cfg.difficulty_range, (0.05, 0.15))
        self.assertEqual(
            tuple(item.profile for item in cfg.sub_terrains.values()),
            terrain.PROFILE_BY_COLUMN,
        )
        # Reproduce the exact v2.3.2 native curriculum column selection formula.
        proportions = np.array([item.proportion for item in cfg.sub_terrains.values()])
        selected = [
            np.min(
                np.where(i / 20 + 0.001 < np.cumsum(proportions / proportions.sum()))[0]
            )
            for i in range(20)
        ]
        np.testing.assert_array_equal(selected, np.arange(20))
        self.assertEqual(sum(name == "plane" for name in terrain.PROFILE_BY_COLUMN), 4)
        self.assertEqual(len({item.variant for item in cfg.sub_terrains.values()}), 20)

    def test_native_factory_rejects_random_layout_or_fraction_rounding(self):
        for kwargs in (
            {"num_cols": 19},
            {"num_cols": 21},
            {"num_rows": 0},
            {"curriculum": False},
            {"difficulty_range": (0.2, 0.1)},
            {"difficulty_range": (0.0, math.nan)},
        ):
            with self.assertRaises(ValueError):
                terrain.make_operator_terrain_generator(seed=43, **kwargs)

    def test_native_callback_returns_mesh_list_and_positive_local_origin(self):
        calls = []
        module = ModuleType("trimesh")
        module.Trimesh = lambda **kwargs: calls.append(kwargs) or kwargs
        cfg = SimpleNamespace(profile="hills", seed=43, variant=19, size=(16, 16))
        with patch.dict(sys.modules, {"trimesh": module}):
            meshes, origin = terrain.operator_terrain(1.0, cfg)
        self.assertEqual(len(meshes), 1)
        self.assertFalse(calls[0]["process"])
        np.testing.assert_array_equal(origin, [8, 8, 0])
        self.assertEqual(calls[0]["vertices"][:, :2].min(), 0)
        self.assertEqual(calls[0]["vertices"][:, :2].max(), 16)


if __name__ == "__main__":
    unittest.main()
