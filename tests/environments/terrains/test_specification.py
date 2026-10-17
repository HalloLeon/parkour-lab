"""Physical candidate mappings and family allocation require no native runtime."""

import json
import unittest
from dataclasses import FrozenInstanceError

import numpy as np

from parkour_lab.environments.terrains.specification import (
    FAMILIES,
    LEVEL_COUNT,
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

    def test_entry_levels_are_nonflat_and_parameters_increase_explicitly(self):
        rows = [physical_level(level) for level in range(LEVEL_COUNT)]
        for key in (
            "uneven_noise_amplitude_m",
            "slope_rise_run",
            "stair_riser_m",
            "hill_native_amplitude_m",
        ):
            self.assertGreater(rows[0][key], 0)
            self.assertTrue(np.all(np.diff([row[key] for row in rows]) > 0))
        self.assertTrue(
            np.all(np.diff([row["uneven_sample_spacing_m"] for row in rows]) < 0)
        )
        self.assertEqual(rows[-1]["stair_riser_m"], 0.16)
        self.assertAlmostEqual(rows[-1]["slope_rise_run"], 0.30)

    def test_fixed_evaluation_is_one_physical_row_without_relabeling_native_row(self):
        for level in (0, 3, 7):
            receipt = terrain_specification(
                TerrainConfig(fixed_level=level), training=False, task_id="Go2-Mixed-v1"
            )
            self.assertEqual(receipt["physical_levels_by_row"], [level])
            self.assertEqual(receipt["num_rows"], 1)
            self.assertEqual(receipt["level_count"], 8)
            self.assertEqual(receipt["flat_fraction"], 0.2)
            self.assertIn("pending_native_calibration", receipt["status"])
            json.dumps(receipt, allow_nan=False)
        flat = terrain_specification(
            TerrainConfig(), training=False, task_id="Go2-Flat-v1"
        )
        self.assertEqual(flat["physical_levels_by_row"], [None])
        self.assertIsNone(flat["fixed_level"])

    def test_options_are_immutable_and_reject_invalid_or_role_conflicting_levels(self):
        with self.assertRaises(FrozenInstanceError):
            TerrainConfig().fixed_level = 1
        for options in (
            {"fixed_level": True},
            {"fixed_level": -1},
            {"fixed_level": 8},
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
        for task_id in ("Go2-Other-v1", "Stairs"):
            with self.assertRaises(ValueError):
                columns_for(task_id)


if __name__ == "__main__":
    unittest.main()
