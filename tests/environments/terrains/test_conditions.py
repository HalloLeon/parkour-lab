"""Evaluation exclusions and real native support geometry for terrain conditions."""

import json
import sys
import unittest
from itertools import product

import numpy as np
import trimesh

from parkour_lab.environments.terrains.configuration import configure_terrain
from parkour_lab.environments.terrains.generation import (
    generate_terrain,
    native_terrain_config,
)
from parkour_lab.environments.terrains.specification import (
    EVALUATION_LEVELS,
    LEVEL_COUNT,
    TerrainConfig,
    physical_level,
    terrain_specification,
)
from tests.environments.terrains.test_generation import (
    config,
    height_at,
    native_geometry,
    subterrain,
)


def generated(family, subtype, level, condition):
    cfg = subterrain(family, subtype, fixed_level=level)
    cfg.condition = condition
    pieces, origin = generate_terrain(0.25, cfg)
    return pieces, trimesh.util.concatenate(pieces), origin


class TerrainConditionSpecificationTests(unittest.TestCase):
    def test_id_definition_stays_unchanged_and_nonid_never_enters_training(self):
        self.assertNotIn("condition", TerrainConfig().as_dict())
        for family in ("Flat", "Uneven", "Slopes", "Stairs", "Hills", "Mixed"):
            task_id = f"Go2-{family}-v1"
            self.assertNotIn(
                "condition",
                terrain_specification(TerrainConfig(), training=True, task_id=task_id),
            )
            for condition in ("composition", "terrain_ood"):
                with self.subTest(family=family, condition=condition):
                    with self.assertRaisesRegex(ValueError, "evaluation-only"):
                        terrain_specification(
                            TerrainConfig(condition=condition),
                            training=True,
                            task_id=task_id,
                        )
        for value in (None, True, "ood", [], 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TerrainConfig(condition=value)

    def test_every_ood_severity_exceeds_actual_training_parameter_caps(self):
        caps = physical_level(LEVEL_COUNT - 1)
        keys = (
            "uneven_noise_amplitude_m",
            "slope_rise_run",
            "stair_riser_m",
            "hill_native_amplitude_m",
        )
        for level in range(LEVEL_COUNT):
            values = physical_level(level, condition="terrain_ood")
            for key in keys:
                self.assertGreater(values[key], caps[key])
            self.assertEqual(values["stair_tread_m"], caps["stair_tread_m"])
            self.assertEqual(values["hill_wavelength_m"], caps["hill_wavelength_m"])
        receipt = terrain_specification(
            TerrainConfig(condition="terrain_ood", fixed_level=0),
            training=False,
            task_id="Go2-Mixed-v1",
        )
        self.assertEqual(receipt["training_support"]["maximum"], caps)
        self.assertEqual(receipt["fixed_level"], 0)
        self.assertEqual(len(receipt["excluded_training_support"]), 4)
        json.dumps(receipt, allow_nan=False)

    def test_composition_components_are_id_and_invalid_family_labels_rejected(self):
        for level in EVALUATION_LEVELS:
            self.assertEqual(
                physical_level(level, condition="composition"), physical_level(level)
            )
        for family in ("Flat", "Uneven"):
            with self.assertRaisesRegex(ValueError, "composition"):
                terrain_specification(
                    TerrainConfig(condition="composition"),
                    training=False,
                    task_id=f"Go2-{family}-v1",
                )
        with self.assertRaisesRegex(ValueError, "excluded"):
            terrain_specification(
                TerrainConfig(condition="terrain_ood"),
                training=False,
                task_id="Go2-Flat-v1",
            )


class TerrainConditionGeometryTests(unittest.TestCase):
    def test_actual_sdk_copy_and_dispatch_preserve_conditions_and_change_geometry(self):
        with native_geometry():
            generator_type = sys.modules[
                "isaaclab.terrains.terrain_generator"
            ].TerrainGenerator
            for level, (family, subtype) in product(
                (0, 8, 15),
                (
                    ("Slopes", "pyramid"),
                    ("Slopes", "inverted_pyramid"),
                    ("Stairs", "pyramid"),
                    ("Stairs", "inverted_pyramid"),
                    ("Hills", "wave"),
                    ("Uneven", "noise"),
                ),
            ):
                generated = {}
                conditions = (
                    ("id", "terrain_ood")
                    if family == "Uneven"
                    else ("id", "composition", "terrain_ood")
                )
                for condition in conditions:
                    with self.subTest(
                        level=level, family=family, subtype=subtype, condition=condition
                    ):
                        cfg = config()
                        configure_terrain(
                            cfg,
                            TerrainConfig(fixed_level=level, condition=condition),
                            training=False,
                            task_id=f"Go2-{family}-v1",
                            seed=1077,
                        )
                        generator = generator_type.__new__(generator_type)
                        generator.cfg = cfg.scene.terrain.terrain_generator
                        selected = next(
                            sub_cfg
                            for sub_cfg in generator.cfg.sub_terrains.values()
                            if sub_cfg.subtype == subtype
                        )
                        # Match the common-size propagation from SDK __init__,
                        # while avoiding its unused Warp/importer boundaries.
                        selected.size = generator.cfg.size
                        copied = selected.copy()
                        self.assertEqual(getattr(copied, "condition", "id"), condition)
                        if condition == "id":
                            self.assertFalse(hasattr(copied, "condition"))
                        # Exercise the real SDK copy/hash/dispatch/centering path,
                        # not a direct call that can hide lost config fields.
                        generated[condition] = generator._get_terrain_mesh(
                            0.9, selected
                        )
                base, base_origin = generated["id"]
                for condition in conditions[1:]:
                    mesh, origin = generated[condition]
                    if level == 0 and condition == "composition":
                        np.testing.assert_array_equal(mesh.vertices, base.vertices)
                        np.testing.assert_array_equal(mesh.faces, base.faces)
                        np.testing.assert_array_equal(origin, base_origin)
                        continue
                    self.assertFalse(np.array_equal(mesh.vertices, base.vertices))
                    np.testing.assert_array_equal(
                        mesh.bounds[:, :2], base.bounds[:, :2]
                    )
                    if condition == "composition":
                        np.testing.assert_array_equal(origin, base_origin)
                        for x, y in ((0, 0), (0.4, -0.3), (-0.8, 0.7)):
                            self.assertAlmostEqual(
                                height_at(mesh, x, y), height_at(base, x, y), places=7
                            )
                        # Quantized low-amplitude noise can be zero at a few
                        # selected points; probe a spatial grid of real support.
                        changed = any(
                            abs(height_at(mesh, x, y) - height_at(base, x, y)) > 0.0001
                            for x, y in product(
                                (-3.25, -2.25, 2.25, 3.25), (-2.75, -0.75, 0.75, 2.75)
                            )
                        )
                        self.assertTrue(changed, (level, family, subtype))
                    elif family == "Slopes":
                        self.assertGreater(
                            abs(height_at(mesh, 2, 0) - height_at(mesh, 3, 0)), 0.30
                        )
                    elif family == "Stairs":
                        self.assertAlmostEqual(
                            abs(height_at(mesh, 3.05, 0) - height_at(mesh, 3.35, 0)),
                            physical_level(level, condition="terrain_ood")[
                                "stair_riser_m"
                            ],
                            places=7,
                        )
                    elif family == "Hills":
                        self.assertGreater(np.ptp(mesh.vertices[:, 2]), 0.60)

    def test_composition_changes_actual_support_while_preserving_origins_and_starts(
        self,
    ):
        with native_geometry():
            for level in EVALUATION_LEVELS:
                for family, subtype in (
                    ("Slopes", "pyramid"),
                    ("Slopes", "inverted_pyramid"),
                    ("Stairs", "pyramid"),
                    ("Stairs", "inverted_pyramid"),
                    ("Hills", "wave"),
                ):
                    with self.subTest(level=level, family=family, subtype=subtype):
                        _, base, base_origin = generated(family, subtype, level, "id")
                        pieces, composed, origin = generated(
                            family, subtype, level, "composition"
                        )
                        if level == 0:
                            np.testing.assert_array_equal(
                                base.vertices, composed.vertices
                            )
                            np.testing.assert_array_equal(base.faces, composed.faces)
                            np.testing.assert_array_equal(origin, base_origin)
                            np.testing.assert_array_equal(composed.vertices[:, 2], 0)
                            continue
                        np.testing.assert_array_equal(origin, base_origin)
                        np.testing.assert_array_equal(
                            base.bounds[:, :2], composed.bounds[:, :2]
                        )
                        self.assertTrue(np.isfinite(composed.vertices).all())
                        for x, y in ((5.1, 5.1), (6, 6), (6.4, 5.7), (6.9, 6.9)):
                            self.assertAlmostEqual(
                                height_at(base, x, y),
                                height_at(composed, x, y),
                                places=7,
                            )
                        differences = [
                            height_at(composed, x, y) - height_at(base, x, y)
                            for x, y in (
                                (8.15, 6.1),
                                (9.15, 6.1),
                                (9.65, 8.1),
                                (3.15, 2.75),
                            )
                        ]
                        self.assertGreater(
                            max(abs(value) for value in differences), 0.0001
                        )
                        if family == "Stairs":
                            self.assertTrue(
                                all(piece.is_watertight for piece in pieces)
                            )
                            self.assertTrue(all(piece.volume > 0 for piece in pieces))

    def test_irregularity_is_within_one_supporting_tread_not_separate_patches(self):
        with native_geometry():
            _, base, _ = generated("Stairs", "pyramid", 8, "id")
            _, composed, _ = generated("Stairs", "pyramid", 8, "composition")
            samples = [(9.05, 6.05), (9.15, 6.05), (9.25, 6.05)]
            nominal = np.asarray([height_at(base, x, y) for x, y in samples])
            actual = np.asarray([height_at(composed, x, y) for x, y in samples])
            self.assertLess(np.ptp(nominal), 1e-10)
            self.assertGreater(np.ptp(actual), 0.001)

    def test_native_ood_parameters_are_used_in_real_geometry(self):
        with native_geometry():
            for level in EVALUATION_LEVELS:
                values = physical_level(level, condition="terrain_ood")
                cfg = native_terrain_config(
                    "Uneven", "noise", level, size=(12, 12), condition="terrain_ood"
                )
                self.assertEqual(
                    cfg.noise_range,
                    (
                        -values["uneven_noise_amplitude_m"],
                        values["uneven_noise_amplitude_m"],
                    ),
                )
                _, slopes, _ = generated("Slopes", "pyramid", level, "terrain_ood")
                self.assertGreater(
                    abs(height_at(slopes, 8, 6) - height_at(slopes, 9, 6)), 0.30
                )
                _, stairs, _ = generated("Stairs", "pyramid", level, "terrain_ood")
                self.assertAlmostEqual(
                    abs(height_at(stairs, 9.05, 6) - height_at(stairs, 9.35, 6)),
                    values["stair_riser_m"],
                    places=7,
                )
                _, hills, _ = generated("Hills", "wave", level, "terrain_ood")
                self.assertGreater(np.ptp(hills.vertices[:, 2]), 0.60)

    def test_noise_is_reproducible_global_rng_isolated_and_id_columns_unchanged(self):
        with native_geometry():
            np.random.seed(319)
            state = np.random.get_state()
            _, first, origin = generated("Stairs", "inverted_pyramid", 3, "composition")
            after = np.random.random(5)
            np.random.set_state(state)
            np.testing.assert_array_equal(after, np.random.random(5))
            _, second, repeated_origin = generated(
                "Stairs", "inverted_pyramid", 3, "composition"
            )
            np.testing.assert_array_equal(first.vertices, second.vertices)
            np.testing.assert_array_equal(first.faces, second.faces)
            np.testing.assert_array_equal(origin, repeated_origin)
            for condition in ("composition", "terrain_ood"):
                cfg = config()
                receipt = configure_terrain(
                    cfg,
                    TerrainConfig(fixed_level=3, condition=condition),
                    training=False,
                    task_id="Go2-Mixed-v1",
                    seed=7,
                )
                self.assertIsNone(cfg.curriculum.terrain_levels)
                self.assertEqual(receipt["num_rows"], 1)
                self.assertEqual(cfg.scene.terrain.terrain_generator.num_rows, 1)
                for native in cfg.scene.terrain.terrain_generator.sub_terrains.values():
                    self.assertEqual(native.condition, condition)
            for family, subtype in (("Flat", "plane"), ("Uneven", "noise")):
                _, base, base_origin = generated(family, subtype, 3, "id")
                _, changed, changed_origin = generated(
                    family, subtype, 3, "composition"
                )
                np.testing.assert_array_equal(base.vertices, changed.vertices)
                np.testing.assert_array_equal(base.faces, changed.faces)
                np.testing.assert_array_equal(base_origin, changed_origin)


if __name__ == "__main__":
    unittest.main()
