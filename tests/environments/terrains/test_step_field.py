"""CPU step geometry and mocked native API checks, not simulator/contact evidence."""

import json
import math
import unittest
from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace as NS

import numpy as np
import torch

import parkour_lab.environments.terrains.step_field as field
from parkour_lab.environments.terrains import procedural as terrain
from tests.support.isaaclab import modules, pinned_generator, pinned_raycast


@dataclass
class TerrainConfig:
    profile: str
    variant: int
    proportion: float = 0.05
    function: object = terrain.operator_terrain
    seed: int = 147
    size: tuple = (16.0, 16.0)

    def copy(self):
        return replace(self)


def configuration(seed=147):
    profiles = terrain.PROFILE_BY_COLUMN
    terrains = {
        str(i): TerrainConfig(
            profile=profile,
            variant=i,
            proportion=0.05,
            function=terrain.operator_terrain,
            seed=seed,
            size=(16.0, 16.0),
        )
        for i, profile in enumerate(profiles)
    }
    return NS(
        scene=NS(
            terrain=NS(
                terrain_generator=NS(
                    seed=seed,
                    num_rows=3,
                    num_cols=20,
                    size=(16.0, 16.0),
                    difficulty_range=(0.15, 0.55),
                    curriculum=True,
                    use_cache=False,
                    sub_terrains=terrains,
                )
            )
        ),
        commands=object(),
        events=object(),
    )


def ray_hit(vertices, faces, start, direction):
    """Independent double-precision triangle intersection, no optional rtree."""
    triangles = vertices[faces]
    first, second = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    cross = np.cross(direction, second)
    determinant = np.einsum("ij,ij->i", first, cross)
    valid = np.abs(determinant) > 1e-12
    inverse = np.divide(1.0, determinant, out=np.zeros_like(determinant), where=valid)
    delta = np.asarray(start) - triangles[:, 0]
    u = np.einsum("ij,ij->i", delta, cross) * inverse
    cross = np.cross(delta, first)
    v = (cross @ np.asarray(direction)) * inverse
    distance = np.einsum("ij,ij->i", second, cross) * inverse
    valid &= (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (distance >= 0)
    distance = np.where(valid, distance, np.inf)
    index = distance.argmin()
    normal = np.cross(first[index], second[index])
    return (
        np.asarray(start) + distance[index] * np.asarray(direction),
        float(distance[index]),
        normal / np.linalg.norm(normal),
    )


class StepFieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = []
        for variant in (12, 13, 14, 15):
            for row in range(3):
                difficulty = 0.15 + (row + 0.5) / 3 * 0.4
                vertices, faces, receipt = field.build_surface(
                    difficulty, seed=147, variant=variant
                )
                cls.fixtures.append((vertices, faces, {**receipt, "row": row}))
        cls.records = [item[2] for item in cls.fixtures]

    def test_exact_welded_support_vertical_risers_and_no_beveled_bridge(self):
        for vertices, faces, receipt in self.fixtures:
            with self.subTest(variant=receipt["variant"], row=receipt["row"]):
                self.assertEqual(len(np.unique(vertices, axis=0)), len(vertices))
                triangles = vertices[faces]
                normals = np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                )
                top_count = 32 * 32 * 5 * 5 * 2
                self.assertTrue((normals[:top_count, 2] > 0).all())
                self.assertTrue((normals[top_count:, 2] == 0).all())
                self.assertTrue((np.linalg.norm(normals, axis=1) > 0).all())
                self.assertAlmostEqual(normals[:top_count, 2].sum() / 2, 256.0)
                self.assertGreater(receipt["vertical_face_count"], 100)
                walls = triangles[top_count:]
                vertical_lengths = []
                for a, b in ((0, 1), (1, 2), (2, 0)):
                    vertical = (walls[:, a, :2] == walls[:, b, :2]).all(1)
                    vertical_lengths.extend(
                        np.abs(walls[vertical, a, 2] - walls[vertical, b, 2])
                    )
                self.assertEqual(len(vertical_lengths), len(walls))
                np.testing.assert_allclose(
                    vertical_lengths, receipt["riser_height_m"], atol=1e-14, rtol=0
                )
                self.assertEqual(
                    receipt["mesh_vertex_count"], receipt["welded_vertex_count"]
                )

    def test_pad_and_boundaries_exactly_flat_without_a_through_going_band(self):
        for vertices, _, receipt in self.fixtures:
            local = vertices[:, :2] - 8
            flat = (np.abs(local) <= 1).all(1) | (np.abs(local) >= 7).any(1)
            np.testing.assert_array_equal(vertices[flat, 2], 0)
            band = (np.abs(local[:, 1]) <= 0.1) & (np.abs(local[:, 0]) > 1)
            self.assertTrue((vertices[band, 2] > receipt["riser_height_m"] / 2).any())
            self.assertLessEqual(
                vertices[:, 2].max(),
                2 * receipt["riser_height_m"]
                + receipt["roughness_absolute_bound_m"]
                + 1e-12,
            )
            self.assertGreaterEqual(
                vertices[:, 2].min(), -receipt["roughness_absolute_bound_m"] - 1e-12
            )
            np.testing.assert_array_equal(receipt["origin_m"], [8, 8, 0])

    def test_deterministic_seed_variant_and_si_difficulty(self):
        np.random.seed(17)
        global_state = np.random.get_state()
        vertices, faces, receipt = field.build_surface(0.35, seed=147, variant=12)
        following = np.random.random(4)
        np.random.set_state(global_state)
        np.testing.assert_array_equal(following, np.random.random(4))
        again = field.build_surface(0.35, seed=147, variant=12)
        np.testing.assert_array_equal(vertices, again[0])
        np.testing.assert_array_equal(faces, again[1])
        self.assertEqual(receipt, again[2])
        for kwargs in ({"seed": 148, "variant": 12}, {"seed": 147, "variant": 13}):
            other = field.build_surface(0.35, **kwargs)[2]
            self.assertNotEqual(
                receipt["vertices_float64_sha256"], other["vertices_float64_sha256"]
            )
        for difficulty, rise in (
            (0.0, 0.04),
            (0.15, 0.052),
            (0.55, 0.084),
            (1.0, 0.12),
        ):
            self.assertAlmostEqual(
                field.build_surface(difficulty, seed=147, variant=12)[2][
                    "riser_height_m"
                ],
                rise,
            )
        self.assertEqual(field.envelope()["axis_aligned_cell_tread_m"], 0.5)

    def test_actual_mesh_horizontal_riser_and_downward_pad_rays(self):
        for vertices, faces, receipt in self.fixtures:
            probe = receipt["riser_probe"]
            hit, distance, normal = ray_hit(
                vertices, faces, probe["ray_start_local_m"], probe["ray_direction"]
            )
            np.testing.assert_allclose(hit, probe["expected_hit_local_m"], atol=1e-12)
            self.assertAlmostEqual(distance, 0.1)
            np.testing.assert_allclose(
                normal, probe["expected_face_normal"], atol=1e-12
            )
            hit, distance, normal = ray_hit(vertices, faces, [8, 8, 1], [0, 0, -1])
            np.testing.assert_allclose(hit, [8, 8, 0], atol=1e-12)
            self.assertAlmostEqual(distance, 1)
            np.testing.assert_array_equal(normal, [0, 0, 1])

    def test_geometry_receipts_reject_unsupported_versions(self):
        for retired in ("operator_step_field_bootstrap_v1", "unknown"):
            records = deepcopy(self.records)
            for record in records:
                record["version"] = retired
            with self.assertRaises(ValueError):
                field.verify_native_receipts(records)
        for operation in (
            lambda: field.envelope("operator_step_field_bootstrap_v1"),
            lambda: field.build_surface(0.35, seed=147, variant=12, version="old"),
            lambda: field.configure(configuration(), version="old"),
        ):
            with self.assertRaises(TypeError):
                operation()

    def test_configuration_changes_only_four_callbacks_and_preserves_old_meshes(self):
        cfg = configuration()
        generator = cfg.scene.terrain.terrain_generator
        old = {key: vars(item).copy() for key, item in generator.sub_terrains.items()}
        baseline = terrain.build_operator_surface(
            0.35, "step_hills", seed=147, variant=12
        )
        commands, events = cfg.commands, cfg.events
        field.configure(cfg)
        self.assertIs(cfg.commands, commands)
        self.assertIs(cfg.events, events)
        for key, item in generator.sub_terrains.items():
            if 12 <= item.variant < 16:
                self.assertIs(item.function, field.step_field_terrain)
                self.assertEqual(item.profile, "step_hills")
                self.assertEqual(
                    {k: v for k, v in vars(item).items() if k != "function"},
                    {k: v for k, v in old[key].items() if k != "function"},
                )
            else:
                self.assertEqual(vars(item), old[key])
                expected = terrain.surface_mesh_arrays(
                    terrain.build_operator_surface(
                        0.35, item.profile, seed=item.seed, variant=item.variant
                    )
                )
                meshes, _ = item.function(0.35, item)
                self.assertEqual(meshes[0].vertices.tobytes(), expected[0].tobytes())
                self.assertEqual(meshes[0].faces.tobytes(), expected[1].tobytes())
        after = terrain.build_operator_surface(0.35, "step_hills", seed=147, variant=12)
        for first, second in zip(
            terrain.surface_mesh_arrays(baseline),
            terrain.surface_mesh_arrays(after),
        ):
            self.assertEqual(first.tobytes(), second.tobytes())

    def test_native_callbacks_use_real_trimesh_and_all_twelve_receipts(self):
        cfg = configuration()
        field.configure(cfg)
        with self.assertRaises(ValueError):
            field.verify_native_receipts()
        terrains = cfg.scene.terrain.terrain_generator.sub_terrains
        for vertices, faces, item in self.fixtures:
            meshes, origin = field.step_field_terrain(
                item["difficulty"], terrains[str(item["variant"])].copy()
            )
            np.testing.assert_array_equal(meshes[0].vertices, vertices)
            np.testing.assert_array_equal(meshes[0].faces, faces)
            np.testing.assert_array_equal(origin, [8, 8, 0])
        result = field.verify_native_receipts()
        self.assertEqual(result["tiles"], self.records)
        self.assertEqual(
            result,
            field.verify_native_receipts(json.loads(json.dumps(result["tiles"]))),
        )
        returned = field.native_receipts()
        returned[0]["seed"] = 0
        self.assertEqual(field.native_receipts(), self.records)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            field.step_field_terrain(self.records[0]["difficulty"], terrains["12"])

    def test_invalid_geometry_configuration_and_receipts_fail_closed(self):
        for difficulty in (True, -0.1, 1.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                field.build_surface(difficulty, seed=1, variant=12)
        for kwargs in (
            {"seed": True},
            {"seed": -1},
            {"variant": 1.5},
            {"size": (12, 16)},
        ):
            with self.assertRaises(ValueError):
                field.build_surface(0.35, **({"seed": 1, "variant": 12} | kwargs))
        for key, value in (
            ("num_cols", 4),
            ("num_rows", 1),
            ("difficulty_range", (0.05, 0.15)),
            ("use_cache", True),
            ("curriculum", False),
        ):
            cfg = configuration()
            setattr(cfg.scene.terrain.terrain_generator, key, value)
            with self.assertRaises(ValueError):
                field.configure(cfg)
        for mutate in (
            lambda x: x.pop(),
            lambda x: x.__setitem__(1, deepcopy(x[0])),
            lambda x: x[0].update(seed=1),
            lambda x: x[0].update(difficulty=0.9),
            lambda x: x[0].update(vertices_float64_sha256="0" * 64),
        ):
            records = deepcopy(self.records)
            mutate(records)
            with self.assertRaises(ValueError):
                field.verify_native_receipts(records)
        cfg = configuration()
        field.configure(cfg)
        item = cfg.scene.terrain.terrain_generator.sub_terrains["12"]
        for key, value in (
            ("seed", 99),
            ("size", (12, 16)),
            ("function", terrain.operator_terrain),
        ):
            broken = deepcopy(item)
            setattr(broken, key, value)
            with self.assertRaises(ValueError):
                field.step_field_terrain(0.35, broken)

    def test_pinned_native_config_copy_and_generator_dispatch_all_twelve_tiles(self):
        self.check_pinned_dispatch(self.fixtures)

    def check_pinned_dispatch(self, fixtures):
        with pinned_generator() as (configclass, base, generator_class):

            @configclass
            class OperatorConfig(base):
                profile: str = "plane"
                variant: int = 0

            cfg = configuration()
            generator_cfg = cfg.scene.terrain.terrain_generator
            generator_cfg.cache_dir = "unused-cache-no-io"
            generator_cfg.sub_terrains = {
                key: OperatorConfig(
                    function=value.function,
                    profile=value.profile,
                    variant=value.variant,
                    proportion=value.proportion,
                    size=value.size,
                )
                for key, value in generator_cfg.sub_terrains.items()
            }
            field.configure(cfg)
            generator = generator_class.__new__(generator_class)
            generator.cfg = generator_cfg
            for vertices, faces, receipt in fixtures:
                item = generator_cfg.sub_terrains[str(receipt["variant"])]
                item.seed = 999  # The generator must inject its seed AFTER copying.
                item.undeclared_copy_probe = "must be dropped"
                copied = item.copy()
                self.assertFalse(hasattr(copied, "undeclared_copy_probe"))
                self.assertFalse(hasattr(copied, "seed"))
                self.assertIs(copied.function, field.step_field_terrain)
                mesh, origin = generator._get_terrain_mesh(receipt["difficulty"], item)
                np.testing.assert_array_equal(mesh.vertices, vertices - [8, 8, 0])
                np.testing.assert_array_equal(mesh.faces, faces)
                np.testing.assert_array_equal(origin, [0, 0, 0])
                self.assertEqual(item.seed, 999)
                self.assertFalse(hasattr(item, "difficulty"))
            self.assertEqual(
                field.verify_native_receipts()["tiles"], [item[2] for item in fixtures]
            )

    def imported_fixture(self, fixtures=None):
        """Manufactured Warp/USD responses test API plumbing, not native geometry."""
        fixtures = self.fixtures if fixtures is None else fixtures
        records = [item[2] for item in fixtures]
        field.configure(configuration())
        field._RECEIPTS[:] = deepcopy(records)
        points, faces, _ = fixtures[0]

        def attribute(value):
            return NS(Get=lambda: value)

        state = {"enabled": True, "approximation": None}
        prim = NS(
            IsValid=lambda: True,
            HasAPI=lambda _: True,
            GetPath=lambda: "/World/ground/terrain/mesh",
            GetAttribute=lambda _: attribute(state["approximation"]),
        )
        geom = NS(
            GetPointsAttr=lambda: attribute(points),
            GetFaceVertexIndicesAttr=lambda: attribute(faces.ravel()),
            GetFaceVertexCountsAttr=lambda: attribute(np.full(len(faces), 3)),
        )
        origins = torch.zeros(3, 20, 3)
        for row in range(3):
            for col in range(20):
                origins[row, col] = torch.tensor([16 * (row - 1), 16 * (col - 9.5), 0])

        class Scene(dict):
            pass

        scene = Scene(
            base_height_scanner=NS(
                cfg=NS(mesh_prim_paths=["/World/ground"]),
                meshes={"/World/ground": object()},
            )
        )
        scene.terrain = NS(terrain_origins=origins)
        env = NS(scene=scene, device="cpu")

        def rays(starts, directions, mesh, **kwargs):
            self.assertEqual(starts.shape, (1, 24, 3))
            self.assertEqual(directions.shape, (1, 24, 3))
            hits, distances, normals, expected_starts, expected_directions = (
                [],
                [],
                [],
                [],
                [],
            )
            for tile in records:
                origin = origins[tile["row"], tile["variant"]].numpy()
                probe = tile["riser_probe"]
                hits.extend(
                    (
                        origin + probe["expected_hit_local_m"] - np.array([8, 8, 0]),
                        origin,
                    )
                )
                distances.extend((0.1, 1.0))
                normals.extend((probe["expected_face_normal"], [0, 0, 1]))
                expected_starts.extend(
                    (
                        origin + probe["ray_start_local_m"] - np.array([8, 8, 0]),
                        origin + [0, 0, 1],
                    )
                )
                expected_directions.extend((probe["ray_direction"], [0, 0, -1]))
            torch.testing.assert_close(
                starts,
                torch.tensor(np.array(expected_starts), dtype=torch.float32).unsqueeze(
                    0
                ),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                directions,
                torch.tensor(expected_directions, dtype=torch.float32).unsqueeze(0),
                rtol=0,
                atol=0,
            )
            return (
                torch.tensor(np.array(hits), dtype=torch.float32).unsqueeze(0),
                torch.tensor(distances).unsqueeze(0),
                torch.tensor(normals).unsqueeze(0),
                torch.arange(24, dtype=torch.int32).unsqueeze(0),
            )

        replacements = {
            "pxr": NS(
                UsdGeom=NS(Mesh=lambda _: geom),
                UsdPhysics=NS(
                    CollisionAPI=lambda _: NS(
                        GetCollisionEnabledAttr=lambda: attribute(state["enabled"])
                    )
                ),
            ),
            "isaaclab": NS(sim=NS(get_first_matching_child_prim=lambda *args: prim)),
            "isaaclab.utils.warp": NS(raycast_mesh=rays),
        }
        return env, state, replacements

    def test_native_gate_through_pinned_raycast_wrapper_preserves_scalar_shapes(self):
        env, state, replacements = self.imported_fixture()
        expected_rays = replacements["isaaclab.utils.warp"].raycast_mesh
        mesh = NS(device="cpu", id=149)
        env.scene["base_height_scanner"].meshes["/World/ground"] = mesh
        state["miss"] = False
        launches = []

        def launch(*, kernel, dim, inputs, device):
            self.assertEqual(dim, 24)
            self.assertEqual(device, "cpu")
            self.assertEqual(inputs[0], mesh.id)
            self.assertEqual(inputs[7:], [2.0, 1, 1, 1])
            starts, directions = inputs[1:3]
            self.assertEqual(starts.shape, (24, 3))
            self.assertEqual(directions.shape, (24, 3))
            outputs = expected_rays(starts.unsqueeze(0), directions.unsqueeze(0), mesh)
            for target, value in zip(inputs[3:7], outputs):
                target.copy_(value.reshape(target.shape))
            inputs[5].mul_(3)  # Face-normal magnitude is not a unit-vector contract.
            if state["miss"]:
                inputs[3][0] = math.inf
                inputs[4][0] = math.inf
                inputs[6][0] = -1
            launches.append((starts.clone(), directions.clone()))

        with modules(replacements), pinned_raycast(launch) as raycast:
            report = field.verify_native_geometry(env)
            measurements = report["imported_geometry"]["measurements"]
            for index, (tile, measured) in enumerate(zip(self.records, measurements)):
                self.assertEqual(measured["variant"], tile["variant"])
                self.assertEqual(measured["row"], tile["row"])
                self.assertEqual(measured["riser_face_id"], 2 * index)
                self.assertEqual(measured["pad_face_id"], 2 * index + 1)
                self.assertAlmostEqual(measured["riser_distance_m"], 0.1, places=6)
                self.assertEqual(measured["pad_distance_m"], 1.0)
                self.assertEqual(
                    measured["riser_normal"],
                    tile["riser_probe"]["expected_face_normal"],
                )
                self.assertEqual(measured["pad_normal"], [0, 0, 1])
            # The actual pinned scalar-output reshape rejects the former flat call.
            starts, directions = launches[0]
            with self.assertRaisesRegex(RuntimeError, "shape.*24.*3.*24"):
                raycast(
                    starts,
                    directions,
                    mesh,
                    max_dist=2.0,
                    return_distance=True,
                    return_normal=True,
                    return_face_id=True,
                )
            state["miss"] = True
            with self.assertRaises(ValueError):
                field.verify_native_geometry(env)

    def test_imported_gate_world_offsets_collision_and_serialized_validation(self):
        env, state, replacements = self.imported_fixture()
        with modules(replacements):
            report = field.verify_native_geometry(env)
            self.assertEqual(len(report["imported_geometry"]["measurements"]), 12)
            for key, value in (("enabled", False), ("approximation", "convexHull")):
                before = state[key]
                state[key] = value
                with self.assertRaises(ValueError):
                    field.verify_native_geometry(env)
                state[key] = before
        self.assertEqual(field.validate_geometry_report(report, seed=147), report)
        for mutate in (
            lambda x: x["imported_geometry"].update(collision_enabled=False),
            lambda x: x["imported_geometry"].update(
                collision_approximation="convexHull"
            ),
            lambda x: x["imported_geometry"]["measurements"][0].update(
                riser_distance_m=math.inf
            ),
            lambda x: x["imported_geometry"]["measurements"][0].update(
                riser_normal=[0, 0, 1]
            ),
            lambda x: x["imported_geometry"]["measurements"][0].update(pad_face_id=-1),
        ):
            broken = deepcopy(report)
            mutate(broken)
            with self.assertRaises(ValueError):
                field.validate_geometry_report(broken, seed=147)
        with self.assertRaises(ValueError):
            field.validate_geometry_report(report, seed=148)


if __name__ == "__main__":
    unittest.main()
