"""Pure local geometry checks; no native collision or robot capability claims."""

import hashlib
import json
import math
import unittest
from unittest.mock import patch

import numpy as np

from parkour_lab.environments.terrains import structures


def build(family="stairs", tier=0.16, **kwargs):
    return structures.build_structure(
        family, tier, coarse_seed=17, fine_seed=23, **kwargs
    )


def normals(mesh, dtype=np.float64):
    triangles = mesh["vertices"].astype(dtype).astype(np.float64)[mesh["faces"]]
    return np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )


class StructureTests(unittest.TestCase):
    def assert_mesh(self, mesh):
        metadata = mesh["metadata"]
        normal = normals(mesh)
        support = mesh["support_faces"]
        self.assertTrue(np.isfinite(mesh["vertices"]).all())
        self.assertTrue((np.linalg.norm(normal, axis=1) > 0).all())
        self.assertTrue((normal[support, 2] > 0).all())
        self.assertTrue((normal[~support, 2] == 0).all())
        angle = np.degrees(
            np.arctan2(np.linalg.norm(normal[support, :2], axis=1), normal[support, 2])
        )
        self.assertLessEqual(angle.max(), 25 + 1e-10)
        self.assertAlmostEqual(
            metadata["maximum_supporting_grade_degrees"], angle.max()
        )
        self.assertAlmostEqual(
            normal[support, 2].sum() / 2, math.prod(metadata["size_m"])
        )
        self.assertLessEqual(
            max(np.diff(mesh["x"]).max(), np.diff(mesh["y"]).max()), 0.02
        )
        for layer, minimum, reported in zip(
            (mesh["coarse_heights"], mesh["fine_heights"]),
            (0.002, 0.0005),
            metadata["roughness_rms_m"],
            strict=True,
        ):
            values = layer[mesh["untapered_samples"]]
            rms = np.sqrt(np.mean(values**2))
            self.assertGreaterEqual(rms, minimum - 1e-12)
            self.assertAlmostEqual(float(rms), reported)
            self.assertAlmostEqual(float(values.mean()), 0, places=14)
            np.testing.assert_array_equal(layer[mesh["taper"] == 0], 0)
        rough = mesh["coarse_heights"] + mesh["fine_heights"]
        self.assertLessEqual(np.abs(rough).max(), metadata["roughness_cap_m"] + 1e-12)
        np.testing.assert_allclose(
            mesh["vertices"][:, 2],
            (mesh["carrier_heights"] + rough).ravel(),
            atol=0,
            rtol=0,
        )

    def test_all_tiers_directions_and_supported_geometry(self):
        for family, tiers in (
            ("stairs", (0.04, 0.08, 0.12, 0.16)),
            ("ramp", (10, 15, 20)),
            ("hill", (10, 15, 20)),
            ("level", (0.01, 0.02)),
        ):
            for tier in tiers:
                for reverse in (False,) if family == "level" else (False, True):
                    with self.subTest(family=family, tier=tier, reverse=reverse):
                        mesh = build(family, tier, reverse=reverse)
                        self.assert_mesh(mesh)
                        metadata = mesh["metadata"]
                        if family in ("ramp", "hill"):
                            self.assertGreaterEqual(
                                metadata["maximum_carrier_grade_degrees"], tier - 0.2
                            )
                            self.assertLessEqual(
                                metadata["maximum_carrier_grade_degrees"], tier
                            )

    def test_stair_counts_treads_and_exposed_riser_orientation(self):
        for count in (1, 4, 5, 6):
            for tread in (0.31, 0.373, 0.5):
                for reverse in (False, True):
                    with self.subTest(count=count, tread=tread, reverse=reverse):
                        mesh = build(risers=count, tread=tread, reverse=reverse)
                        metadata = mesh["metadata"]
                        self.assertEqual(len(metadata["riser_heights_m"]), count)
                        self.assertEqual(len(metadata["tread_lengths_m"]), count - 1)
                        np.testing.assert_allclose(
                            metadata["tread_lengths_m"], tread, rtol=0, atol=1e-15
                        )
                        np.testing.assert_allclose(
                            metadata["riser_heights_m"], 0.159, rtol=0, atol=1e-15
                        )
                        n = normals(mesh)[~mesh["support_faces"]]
                        self.assertTrue(
                            (n[:, 0] > 0).all() if reverse else (n[:, 0] < 0).all()
                        )
                        np.testing.assert_array_equal(n[:, 1:], 0)
                        # Naturally elevated approach for descent; no mid-flight spawn.
                        expected = (count * 0.159, 0) if reverse else (0, count * 0.159)
                        np.testing.assert_allclose(
                            mesh["carrier_heights"][[0, -1], 0],
                            expected,
                            rtol=0,
                            atol=1e-15,
                        )
                        self.assert_mesh(mesh)

    def test_ramp_lengths_and_hill_height_crest_and_reflection(self):
        for length in (2, 3):
            mesh = build("ramp", 20, incline_length=length)
            self.assertAlmostEqual(
                mesh["metadata"]["exit_x_m"] - mesh["metadata"]["entry_x_m"], length
            )
            self.assertAlmostEqual(
                mesh["carrier_heights"][-1, 0], length * math.tan(math.radians(19.9))
            )
        for reverse, up in ((False, 1), (True, 1.5)):
            mesh = build("hill", 20, reverse=reverse)
            x = mesh["x"]
            h = 2 * math.tan(math.radians(19.9)) / math.pi
            crest = (x >= 2 + up) & (x <= 2 + up + 0.4)
            np.testing.assert_allclose(mesh["carrier_heights"][crest], h, atol=1e-15)
            self.assertAlmostEqual(mesh["metadata"]["exit_x_m"] - 2, 2.9)
            np.testing.assert_array_equal(mesh["carrier_heights"][[0, -1]], 0)
            # Roughness does not disappear at crest / flank seams.
            np.testing.assert_array_equal(mesh["taper"][crest], 1)
            self.assertGreater(np.std(mesh["coarse_heights"][crest]), 0.001)

    def test_tapers_only_at_pads_and_two_centimetre_stair_lips(self):
        mesh = build(risers=5, tread=0.373)
        x, taper = mesh["x"], mesh["taper"]
        lips = 2 + np.arange(5) * 0.373
        end = mesh["metadata"]["size_m"][0]
        near_lip = np.abs(x[:, None] - lips).min(axis=1) < 0.02 + 1e-12
        allowed = near_lip | (x < 1.5) | (x > end - 1.5)
        self.assertTrue((taper[~allowed] == 1).all())
        np.testing.assert_array_equal(taper[(x <= 1) | (x >= end - 1)], 0)
        for lip in lips:
            np.testing.assert_array_equal(taper[x == lip], 0)
        # Approach, intermediate treads and landing all retain unevenness.
        for lo, hi in ((1.5, 1.95), (2.02, 2.35), (end - 2, end - 1.5)):
            self.assertGreater(
                np.std(mesh["coarse_heights"][(x > lo) & (x < hi)]), 0.001
            )

    def test_mesh_edges_have_no_internal_cracks_or_nonmanifold_seams(self):
        mesh = build(reverse=True, risers=6, tread=0.373)
        faces = mesh["faces"]
        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        endpoints = mesh["vertices"][edges]
        boundary = np.zeros(len(edges), dtype=bool)
        for axis, end in enumerate(mesh["metadata"]["size_m"]):
            boundary |= (endpoints[:, :, axis] == 0).all(axis=1)
            boundary |= (endpoints[:, :, axis] == end).all(axis=1)
        np.testing.assert_array_equal(counts, np.where(boundary, 1, 2))

    def test_shared_scale_is_maximal_and_allows_cancelling_gradients(self):
        limit = structures.MAXIMUM_GRADE
        for carrier, rough, offset, cap, expected in (
            ([[0, 0]], [[1, 0]], [0.001], 0.02, limit),
            ([[0.3, 0]], [[1, 0]], [0.001], 0.02, limit - 0.3),
            ([[0.3, 0]], [[-1, 0]], [0.001], 0.02, limit + 0.3),
            ([[0, 0]], [[0, 0]], [0.025], 0.02, 0.8),
            ([[0, 0]], [[0, 0]], [0], 0.02, 1),
        ):
            c, r = np.asarray(carrier), np.asarray(rough)
            scale = structures._shared_scale(c, r, np.asarray(offset), cap)
            self.assertAlmostEqual(scale, expected)
            self.assertLessEqual(
                np.linalg.norm(c + scale * r, axis=1).max(), limit + 1e-15
            )
            if scale < 1:
                too_large = scale + 1e-6
                self.assertTrue(
                    np.linalg.norm(c + too_large * r, axis=1).max() > limit
                    or np.max(np.abs(offset)) * too_large > cap
                )

    def test_reproducible_independent_seeded_layers_and_receipt_hashes(self):
        before = np.random.get_state()
        first, again = build(), build()
        for key in ("vertices", "faces", "coarse_heights", "fine_heights"):
            np.testing.assert_array_equal(first[key], again[key])
        self.assertEqual(first["metadata"], again["metadata"])
        for a, b in zip(before, np.random.get_state(), strict=True):
            np.testing.assert_array_equal(a, b)
        other = structures.build_structure("stairs", 0.16, coarse_seed=17, fine_seed=24)
        self.assertEqual(first["metadata"]["roughness_shared_scale"], 1)
        self.assertEqual(other["metadata"]["roughness_shared_scale"], 1)
        np.testing.assert_array_equal(first["coarse_heights"], other["coarse_heights"])
        self.assertFalse(np.array_equal(first["fine_heights"], other["fine_heights"]))
        json.dumps(first["metadata"], allow_nan=False)
        self.assertEqual(
            first["metadata"]["vertices_float64_sha256"],
            hashlib.sha256(first["vertices"].astype("<f8").tobytes()).hexdigest(),
        )

    def test_regional_caps_use_largest_feasible_common_scale(self):
        offset = np.array([0.02, -0.03, 0.0])
        caps = np.array([0.01, 0.02, 0.01])
        carrier = np.zeros((3, 2))
        rough = np.zeros_like(carrier)
        scale = structures._shared_scale(carrier, rough, offset, caps)
        self.assertEqual(scale, 0.5)
        self.assertTrue((np.abs(offset) * scale <= caps).all())
        self.assertTrue((np.abs(offset) * (scale + 1e-6) > caps).any())
        # A global minimum cap would over-shrink the second region to 1/3.
        self.assertGreater(
            scale, structures._shared_scale(carrier, rough, offset, caps.min())
        )
        rough[1, 0] = 1.0
        self.assertEqual(
            structures._shared_scale(carrier, rough, offset, caps),
            structures.MAXIMUM_GRADE,
        )
        self.assertEqual(
            structures._shared_scale(carrier, carrier, np.zeros(3), caps), 1.0
        )

    def test_realized_height_limited_mesh_preserves_both_layer_amplitudes(self):
        mesh = structures.build_structure(
            "hill", 20, coarse_seed=18, fine_seed=19, reverse=True
        )
        self.assert_mesh(mesh)
        metadata = mesh["metadata"]
        scale = metadata["roughness_shared_scale"]
        self.assertLess(scale, 1)
        self.assertGreaterEqual(scale, 2 / 3)
        np.testing.assert_allclose(
            metadata["roughness_rms_m"], np.array((0.003, 0.00075)) * scale
        )
        self.assertAlmostEqual(metadata["maximum_roughness_offset_m"], 0.01)
        larger = (mesh["coarse_heights"] + mesh["fine_heights"]) * (
            (scale + 1e-6) / scale
        )
        self.assertGreater(np.abs(larger).max(), metadata["roughness_cap_m"])

    def test_realized_slope_limited_mesh_uses_maximal_common_scale(self):
        mesh = structures.build_structure(
            "stairs", 0.16, coarse_seed=318, fine_seed=319, risers=6, tread=0.31
        )
        self.assert_mesh(mesh)
        scale = mesh["metadata"]["roughness_shared_scale"]
        self.assertTrue(2 / 3 <= scale < 1)
        self.assertAlmostEqual(mesh["metadata"]["maximum_supporting_grade_degrees"], 25)
        rough = mesh["coarse_heights"] + mesh["fine_heights"]
        larger = dict(mesh, vertices=mesh["vertices"].copy())
        larger["vertices"][:, 2] = (
            mesh["carrier_heights"] + rough * ((scale + 1e-6) / scale)
        ).ravel()
        n = normals(larger)[mesh["support_faces"]]
        self.assertGreater(
            np.degrees(np.arctan2(np.linalg.norm(n[:, :2], axis=1), n[:, 2])).max(), 25
        )

    def test_local_float32_roundtrip_preserves_geometry_with_numerical_tolerance(self):
        # This checks a coordinate conversion, not USD/PhysX import or contacts.
        for family, tier in (("stairs", 0.16), ("ramp", 20), ("hill", 20)):
            mesh = build(family, tier)
            n = normals(mesh, np.float32)
            support = mesh["support_faces"]
            self.assertTrue((np.linalg.norm(n, axis=1) > 0).all())
            self.assertTrue((n[support, 2] > 0).all())
            angle = np.degrees(
                np.arctan2(np.linalg.norm(n[support, :2], axis=1), n[support, 2])
            )
            self.assertLessEqual(angle.max(), 25)
            self.assertLessEqual(np.diff(mesh["x"].astype(np.float32)).max(), 0.02)
            if family == "stairs":
                z = (
                    mesh["vertices"][:, 2]
                    .reshape(len(mesh["x"]), -1)
                    .astype(np.float32)
                )
                rises = np.abs(np.diff(z, axis=0)[np.diff(mesh["x"]) == 0])
                self.assertTrue(((rises >= 0.158) & (rises <= 0.16)).all())

    def test_invalid_inputs_and_infeasible_geometry_fail_before_use(self):
        for family, tier in (
            ("unknown", 10),
            ("stairs", 0.2),
            ("hill", 5),
            ("ramp", True),
            ("level", math.nan),
        ):
            with self.assertRaises(ValueError):
                build(family, tier)
        for kwargs in (
            {"risers": True},
            {"risers": 2},
            {"tread": 0.3},
            {"width": 1.99},
            {"approach": 1.4},
            {"landing": math.inf},
            {"resolution": 0.021},
            {"pad_length": False},
            {"reverse": 1},
            {"incline_length": 3},
            {"width": "2"},
            {"resolution": "0.02"},
        ):
            with self.assertRaises(ValueError):
                build(**kwargs)
        for kwargs in (
            {"coarse_seed": -1, "fine_seed": 2},
            {"coarse_seed": 1, "fine_seed": 1},
        ):
            with self.assertRaises(ValueError):
                structures.build_structure("level", 0.02, **kwargs)
        with self.assertRaises(ValueError):
            build("ramp", 20, risers=6)
        with self.assertRaises(ValueError):
            structures._shared_scale(
                np.array([[1.0, 0]]), np.zeros((1, 2)), np.zeros(1), 0.02
            )
        with self.assertRaisesRegex(ValueError, "positive-area"):
            build("level", 0.02, approach=1.5, landing=1.5)
        with patch.object(structures, "_shared_scale", return_value=0.6):
            with patch.object(
                structures, "_gradient_noise", wraps=structures._gradient_noise
            ) as noise:
                with self.assertRaisesRegex(ValueError, "minimum RMS"):
                    build()
                self.assertEqual(noise.call_count, 2)  # No replacement/reroll.


if __name__ == "__main__":
    unittest.main()
