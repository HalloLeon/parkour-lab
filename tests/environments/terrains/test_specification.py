"""Physical candidate mappings and family allocation require no native runtime."""

import json
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from parkour_lab.environments.terrains.specification import (
    EVALUATION_LEVELS,
    FAMILIES,
    LEVEL_COUNT,
    VERTICAL_SCALE_M,
    TerrainConfig,
    columns_for,
    physical_level,
    terrain_specification,
)


class TerrainSpecificationTests(unittest.TestCase):
    def test_fixed_nonzero_flat_share_and_balanced_signed_geometry(self):
        columns = columns_for("Go2-Mixed-v1")
        self.assertEqual(len(columns), 20)
        for family in FAMILIES:
            selected = [kind for name, kind in columns if name == family]
            self.assertEqual(len(selected), 4)
            if family in ("Slopes", "Stairs"):
                self.assertEqual(selected.count("pyramid"), 2)
                self.assertEqual(selected.count("inverted_pyramid"), 2)
        # Reproduce the native deterministic column-selection formula.
        cumulative = np.cumsum(np.full(20, 1 / 20))
        selected = [
            np.flatnonzero(index / 20 + 0.001 < cumulative)[0] for index in range(20)
        ]
        np.testing.assert_array_equal(selected, np.arange(20))

    def test_exact_development_ladder_has_a_flat_entry_and_resolved_millimeters(self):
        rows = [physical_level(level) for level in range(LEVEL_COUNT)]
        expected = {
            "uneven_noise_amplitude_m": np.asarray(
                [0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50]
            )
            / 1000,
            "slope_rise_run": [
                0,
                0.005,
                0.01,
                0.015,
                0.02,
                0.03,
                0.04,
                0.06,
                0.08,
                0.10,
                0.12,
                0.15,
                0.18,
                0.22,
                0.26,
                0.30,
            ],
            "stair_riser_m": [
                0,
                0.005,
                0.01,
                0.015,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.07,
                0.08,
                0.09,
                0.10,
                0.12,
                0.14,
                0.16,
            ],
            "hill_native_amplitude_m": [
                0,
                0.01,
                0.02,
                0.03,
                0.04,
                0.05,
                0.06,
                0.08,
                0.10,
                0.12,
                0.14,
                0.16,
                0.19,
                0.22,
                0.26,
                0.30,
            ],
        }
        self.assertEqual(LEVEL_COUNT, 16)
        self.assertEqual(VERTICAL_SCALE_M, 0.001)
        for key, values in expected.items():
            np.testing.assert_array_equal([row[key] for row in rows], values)
            self.assertTrue(np.all(np.diff([row[key] for row in rows]) > 0))
        np.testing.assert_array_equal(
            [row["uneven_noise_step_m"] for row in rows], 0.001
        )
        spacing = [row["uneven_sample_spacing_m"] for row in rows]
        np.testing.assert_array_equal(spacing[:5], 0.8)
        self.assertTrue(np.all(np.diff(spacing[4:]) < 0))
        self.assertAlmostEqual(spacing[-1], 0.2)
        self.assertEqual(rows[-1]["stair_riser_m"], 0.16)
        self.assertAlmostEqual(rows[-1]["slope_rise_run"], 0.30)

    def test_fixed_evaluation_is_one_physical_row_without_relabeling_native_row(self):
        for level in EVALUATION_LEVELS:
            receipt = terrain_specification(
                TerrainConfig(fixed_level=level), training=False, task_id="Go2-Mixed-v1"
            )
            self.assertEqual(receipt["physical_levels_by_row"], [level])
            self.assertEqual(receipt["num_rows"], 1)
            self.assertEqual(receipt["level_count"], 16)
            self.assertEqual(receipt["training_level_cap"], 15)
            self.assertEqual(
                tuple(receipt["fixed_test_levels"].values()), EVALUATION_LEVELS
            )
            self.assertEqual(receipt["flat_fraction"], 0.2)
            self.assertIn("pending_native_calibration", receipt["status"])
            json.dumps(receipt, allow_nan=False)
        flat = terrain_specification(
            TerrainConfig(), training=False, task_id="Go2-Flat-v1"
        )
        self.assertEqual(flat["physical_levels_by_row"], [None])
        self.assertIsNone(flat["fixed_level"])
        training = terrain_specification(
            TerrainConfig(), training=True, task_id="Go2-Mixed-v1"
        )
        self.assertEqual(training["initial_min_level"], 0)
        self.assertEqual(training["initial_max_level"], 0)
        self.assertEqual(training["physical_levels_by_row"], list(range(16)))
        self.assertEqual(training["version"], "go2-terrain-development-v2")
        self.assertEqual(
            training["initialization"],
            "all training replicas start at level zero; Flat has no difficulty ladder",
        )
        self.assertEqual(
            terrain_specification(
                TerrainConfig(), training=False, task_id="Go2-Mixed-v1"
            )["fixed_level"],
            8,
        )

    def test_options_are_immutable_and_reject_invalid_or_role_conflicting_levels(self):
        with self.assertRaises(FrozenInstanceError):
            TerrainConfig().fixed_level = 1
        for options in (
            {"fixed_level": True},
            {"fixed_level": -1},
            {"fixed_level": 16},
            {"initial_level_range": [0, 1]},
            {"initial_level_range": (2, 1)},
            {"initial_level_range": (False, 1)},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TerrainConfig(**options)
        with self.assertRaisesRegex(ValueError, "evaluation-only"):
            terrain_specification(
                TerrainConfig(fixed_level=3), training=True, task_id="Go2-Mixed-v1"
            )
        for bounds in ((0, 1), (2, 3)):
            with (
                self.subTest(bounds=bounds),
                self.assertRaisesRegex(
                    ValueError, "starts every replica at level zero"
                ),
            ):
                terrain_specification(
                    TerrainConfig(initial_level_range=bounds),
                    training=True,
                    task_id="Go2-Mixed-v1",
                )
        for task_id in ("Go2-Other-v1", "Stairs"):
            with self.assertRaises(ValueError):
                columns_for(task_id)


if __name__ == "__main__":
    unittest.main()
