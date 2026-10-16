"""Connected source geometry; these tests make no native/robot claims."""

import hashlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from parkour_lab.environments.terrains import structures, worlds


def build(family="stairs", tier=0.16, **kwargs):
    return worlds.build_world(family, tier, coarse_seed=17, fine_seed=23, **kwargs)


def normals(mesh, dtype=np.float64):
    triangle = mesh["vertices"].astype(dtype).astype(np.float64)[mesh["faces"]]
    return np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])


class WorldTests(unittest.TestCase):
    def test_all_tiers_directions_and_shape_boundaries_are_conversion_safe(self):
        cases = [
            (family, tier, reverse, {})
            for family, tiers in (
                ("stairs", (0.04, 0.08, 0.12, 0.16)),
                ("ramp", (10, 15, 20)),
                ("hill", (10, 15, 20)),
            )
            for tier in tiers
            for reverse in (False, True)
        ] + [
            ("stairs", 0.16, False, {"risers": 6, "tread": 0.31}),
            ("stairs", 0.16, True, {"risers": 5, "tread": 0.5}),
            ("stairs", 0.16, True, {"risers": 6, "tread": 0.373}),
            ("ramp", 20, True, {"incline_length": 3}),
        ]
        for family, tier, reverse, shape in cases:
            with self.subTest(family=family, tier=tier, reverse=reverse, shape=shape):
                mesh = build(family, tier, reverse=reverse, **shape)
                meta = mesh["metadata"]
                n = normals(mesh)
                support = mesh["support_faces"]
                self.assertTrue((np.linalg.norm(n, axis=1) > 0).all())
                self.assertTrue((n[support, 2] > 0).all())
                self.assertTrue((n[~support, 2] == 0).all())
                angle = np.degrees(
                    np.arctan2(np.linalg.norm(n[support, :2], axis=1), n[support, 2])
                )
                self.assertLessEqual(angle.max(), 25.001)
                self.assertAlmostEqual(
                    angle.max(), meta["maximum_supporting_grade_degrees"]
                )
                self.assertAlmostEqual(n[support, 2].sum() / 2, 576)
                self.assertLessEqual(meta["maximum_axis_spacing_m"], 0.02)
                self.assertLessEqual(meta["float32_maximum_axis_spacing_m"], 0.02)
                for feature in meta["features"].values():
                    self.assertGreaterEqual(feature["usable_width_m"], 2)
                    self.assertGreaterEqual(feature["float32_usable_width_m"], 2)
                stair = meta["features"]["stairs"]
                for key in ("measured_treads_m", "float32_treads_m"):
                    self.assertTrue(np.all(np.asarray(stair[key]) >= 0.31))
                    self.assertTrue(np.all(np.asarray(stair[key]) <= 0.5))
                for key in ("measured_riser_range_m", "float32_riser_range_m"):
                    self.assertGreaterEqual(min(stair[key]), stair["tier"] - 0.002)
                    self.assertLessEqual(max(stair[key]), stair["tier"])
                for feature_name in ("ramp", "hill"):
                    feature = meta["features"][feature_name]
                    boundaries = np.asarray(feature["profile_boundaries_m"])
                    converted_boundaries = boundaries.astype(np.float32).astype(
                        np.float64
                    )
                    np.testing.assert_array_equal(
                        feature["float32_profile_boundaries_m"], converted_boundaries
                    )
                    np.testing.assert_array_equal(
                        feature["profile_boundary_conversion_error_m"],
                        converted_boundaries - boundaries,
                    )
                    origin = feature["entry"][feature["axis"]]
                    if feature_name == "hill":
                        up = 1.5 if family == "hill" and reverse else 1
                        expected_boundaries = [
                            origin,
                            origin + up,
                            origin + up + 0.4,
                            origin + 2.9,
                        ]
                    else:
                        expected_boundaries = [
                            origin,
                            origin + shape.get("incline_length", 2),
                        ]
                    np.testing.assert_array_equal(boundaries, expected_boundaries)
                    coordinates = mesh["vertices"][:, feature["axis"]]
                    for boundary in boundaries:
                        self.assertTrue((coordinates == boundary).any())
                    for key in (
                        "measured_carrier_grade_degrees",
                        "float32_carrier_grade_degrees",
                    ):
                        self.assertGreaterEqual(feature[key], feature["tier"] - 0.2)
                        self.assertLessEqual(feature[key], feature["tier"])
                samples = mesh["untapered_samples"]
                for key, low in (
                    ("coarse_heights", 0.002),
                    ("fine_heights", 0.0005),
                ):
                    self.assertGreaterEqual(
                        np.sqrt(np.mean(mesh[key][samples] ** 2)), low
                    )
                    self.assertAlmostEqual(mesh[key][samples].mean(), 0, places=14)
                rough = mesh["coarse_heights"] + mesh["fine_heights"]
                self.assertTrue((np.abs(rough) <= mesh["roughness_cap_m"]).all())
                np.testing.assert_array_equal(
                    mesh["vertices"][:, 2], mesh["carrier_heights"] + rough
                )
                q = normals(mesh, np.float32)
                # Exact analytical seam authoring avoids the former near-zero
                # risers that disappeared after conversion (four triangles).
                self.assertTrue((np.linalg.norm(q, axis=1) > 0).all())
                self.assertTrue((q[support, 2] > 0).all())
                self.assertTrue((q[~support, 2] == 0).all())
                converted = float(
                    np.degrees(
                        np.arctan2(
                            np.linalg.norm(q[support, :2], axis=1), q[support, 2]
                        )
                    ).max()
                )
                self.assertLessEqual(converted, 25.001)
                self.assertTrue(meta["local_float32_supporting_slope_within_limit"])
                self.assertTrue(meta["float32_topology_valid"])
                self.assertAlmostEqual(
                    converted, meta["local_float32_supporting_grade_degrees"]
                )
                converted_z = (
                    mesh["vertices"][:, 2].astype(np.float32).astype(np.float64)
                )
                self.assertTrue(
                    (
                        np.abs(converted_z - mesh["carrier_heights"])
                        <= mesh["roughness_cap_m"]
                    ).all()
                )
                error = float(np.max(np.abs(converted_z - mesh["vertices"][:, 2])))
                self.assertEqual(error, meta["maximum_height_conversion_error_m"])
                for measured, low in zip(
                    meta["roughness_rms_m"], structures.MINIMUM_RMS, strict=True
                ):
                    self.assertGreaterEqual(measured - error, low)
                # CPU source validity must not become a false native pass.
                trial = meta["trial"]
                self.assertGreater(trial["minimum_map_margin_m"], 0)
                self.assertGreater(trial["minimum_target_side_margin_m"], 0)
                self.assertGreater(trial["target_clearance_at_nominal_finish_m"], 0)
                self.assertEqual(trial["nominal_flat_travel_m"], 8.75)
                self.assertEqual(
                    trial["command_phases"],
                    [
                        [2.0, [0.0, 0.0, 0.0]],
                        [25.0, [0.35, 0.0, 0.0]],
                        [3.0, [0.0, 0.0, 0.0]],
                    ],
                )
                if family == "hill":
                    self.assertEqual(trial["heading_local_rad"], 0)
                    self.assertEqual(trial["start_xy_local_m"], [2, 4])
                if shape and family == "stairs":
                    self.assertEqual(stair["riser_count"], shape["risers"])
                    self.assertEqual(stair["requested_tread_m"], shape["tread"])
                    if shape["tread"] == 0.31:
                        self.assertTrue(stair["coordinate_headroom_applied"])
                        self.assertEqual(len(set(stair["float32_treads_m"])), 1)
                    else:
                        self.assertFalse(stair["coordinate_headroom_applied"])
                    if reverse:
                        self.assertEqual(trial["heading_local_rad"], math.pi)
                if shape and family == "ramp":
                    ramp = meta["features"]["ramp"]
                    self.assertEqual(ramp["usable_width_m"], 3)
                    self.assertEqual(ramp["float32_horizontal_length_m"], 3)
                    self.assertGreater(
                        min(ramp["shoulder_widths_m"]), ramp["minimum_shoulder_width_m"]
                    )
                    self.assertEqual(ramp["intersection_grade_degrees"], 9.9)
                json.dumps(meta, allow_nan=False)
                del mesh, n, q

    def test_manifold_rectangle_with_no_internal_boundaries_or_unused_vertices(self):
        mesh = build()
        vertices, faces = mesh["vertices"], mesh["faces"]
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edges.sort(axis=1)
        packed = edges[:, 0] * len(vertices) + edges[:, 1]
        unique, counts = np.unique(packed, return_counts=True)
        self.assertTrue(np.isin(counts, (1, 2)).all())
        ends = np.column_stack(
            (unique[counts == 1] // len(vertices), unique[counts == 1] % len(vertices))
        )
        points = vertices[ends]
        boundary = np.zeros(len(ends), dtype=bool)
        for axis in (0, 1):
            boundary |= (points[:, :, axis] == 0).all(axis=1)
            boundary |= (points[:, :, axis] == 24).all(axis=1)
        self.assertTrue(boundary.all())
        _, degrees = np.unique(ends, return_counts=True)
        self.assertTrue((degrees == 2).all())
        self.assertEqual(len(vertices) - len(unique) + len(faces), 1)
        self.assertEqual(len(np.unique(faces)), len(vertices))
        # Together with positive projected support and the one rectangular outer
        # boundary, these rule out separate interior components or seam holes.
        self.assertGreater(mesh["metadata"]["welded_collapsed_riser_vertices"], 0)

    def test_roughness_taper_only_at_actual_risers_and_assigned_pads(self):
        mesh = build()
        xy = mesh["vertices"][:, :2]
        meta = mesh["metadata"]
        taper = np.ones(len(xy))
        for start, end in meta["intentional_riser_segments_local_m"]:
            distance = np.hypot(xy[:, 0] - start[0], np.maximum(xy[:, 1] - end[1], 0))
            taper = np.minimum(taper, np.clip(distance / 0.02, 0, 1))
        for name in ("start_pad_local_m", "final_stop_region_local_m"):
            pad = np.array(meta["trial"][name])
            distance = np.maximum(np.maximum(pad[0] - xy, xy - pad[1]), 0).max(axis=1)
            taper = np.minimum(taper, np.clip(distance / 0.5, 0, 1))
            self.assertEqual(np.ptp(mesh["carrier_heights"][distance <= 0.5]), 0)
        expected = taper**3 * (taper * (6 * taper - 15) + 10)
        np.testing.assert_array_equal(mesh["taper"], expected)
        first = meta["intentional_riser_segments_local_m"][0]
        beyond = (xy[:, 0] == first[0][0]) & (xy[:, 1] > first[1][1] + 0.02)
        self.assertTrue(beyond.any())
        np.testing.assert_array_equal(mesh["taper"][beyond], 1)
        np.testing.assert_array_equal(
            mesh["roughness_cap_m"], np.where(xy[:, 0] >= 14, 0.01, 0.02)
        )

    def test_reproducible_shared_geometry_and_rigid_world_orientation(self):
        first = build()
        hashes = {
            key: hashlib.sha256(first[key].tobytes()).hexdigest()
            for key in first
            if key != "metadata"
        }
        before = np.random.get_state()
        del first
        turned = build(world_yaw=1.2)
        self.assertEqual(
            hashes,
            {
                key: hashlib.sha256(turned[key].tobytes()).hexdigest()
                for key in turned
                if key != "metadata"
            },
        )
        trial = turned["metadata"]["trial"]
        transform = np.array(trial["local_to_world_column_transform"])
        np.testing.assert_allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-15
        )
        local_start = np.array([*trial["start_xy_local_m"], 0, 1])
        expected = transform @ local_start
        # The equivalent 2-D and homogeneous dot kernels need not round identically.
        roundoff = 4 * np.finfo(expected.dtype).eps * np.abs(local_start[:2]).sum()
        np.testing.assert_allclose(
            expected[:2], trial["start_xy_world_m"], rtol=0, atol=roundoff
        )
        self.assertEqual(trial["heading_world_rad"], 1.2)
        for a, b in zip(before, np.random.get_state(), strict=True):
            np.testing.assert_array_equal(a, b)

    def test_tread_representation_preserves_legal_inputs_and_repairs_boundaries(self):
        for risers in (4, 5, 6):
            for entry in (14.0, 14.0 + 2**-21, 14.0 + 3 * 2**-21, 15.0):
                for tread in (0.31, 0.373, 0.4, 0.5):
                    with self.subTest(risers=risers, entry=entry, tread=tread):
                        original = entry + np.arange(risers) * tread
                        converted = original.astype(np.float32).astype(np.float64)
                        was_legal = all(
                            ((np.diff(v) >= 0.31) & (np.diff(v) <= 0.5)).all()
                            for v in (original, converted)
                        )
                        lips, receipt = worlds._stair_lips(entry, tread, risers)
                        for values in (
                            lips,
                            lips.astype(np.float32).astype(np.float64),
                        ):
                            self.assertTrue((np.diff(values) >= 0.31).all())
                            self.assertTrue((np.diff(values) <= 0.5).all())
                        self.assertEqual(
                            receipt["coordinate_headroom_applied"], not was_legal
                        )
                        if was_legal:
                            np.testing.assert_array_equal(lips, original)
                        else:
                            self.assertEqual(len(set(np.diff(lips))), 1)
                            np.testing.assert_array_equal(lips, lips.astype(np.float32))
                            self.assertLessEqual(
                                abs(lips[0] - entry),
                                receipt["coordinate_lattice_m"] / 2,
                            )

    def test_regional_caps_use_largest_common_factor(self):
        zeros = np.zeros((2, 2))
        self.assertEqual(
            structures._shared_scale(
                zeros, zeros, np.array([0.02, 0.005]), np.array([0.02, 0.01])
            ),
            1,
        )
        self.assertAlmostEqual(
            structures._shared_scale(
                zeros, zeros, np.array([0.02, 0.015]), np.array([0.02, 0.01])
            ),
            2 / 3,
        )

    def test_invalid_maps_and_infeasible_roughness_fail_before_use(self):
        for kwargs in (
            {"risers": 1},
            {"tread": 0.3},
            {"resolution": 0.021},
            {"size": (20, 20)},
            {"world_yaw": float("nan")},
            {"hill_center_y": 14},
            {"stair_route_y": 15},
            {"ramp_route_x": 20},
            {"stair_entry": 0.01},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                build(**kwargs)
        with self.assertRaisesRegex(ValueError, "shoulders"):
            build("ramp", 20, incline_length=3, ramp_route_x=3)
        with self.assertRaisesRegex(ValueError, "pad and taper"):
            build("hill", 20, hill_entry=3)
        with self.assertRaisesRegex(ValueError, "support or usable width"):
            build("hill", 20, stair_entry=22.8)
        with (
            patch.object(
                worlds,
                "_apply_roughness",
                side_effect=ValueError("minimum RMS"),
            ),
            patch.object(
                structures, "_noise_points", wraps=structures._noise_points
            ) as noise,
        ):
            with self.assertRaisesRegex(ValueError, "minimum RMS"):
                build()
            self.assertEqual(noise.call_count, 2)  # No noise redraw or retry.


if __name__ == "__main__":
    unittest.main()
