"""Execute native terrain config/generator sources on CPU, without Isaac Sim."""

import importlib.util
import sys
import unittest
from contextlib import contextmanager
from types import ModuleType
from types import SimpleNamespace as NS

import numpy as np
import torch
import trimesh

from parkour_lab.environments.terrains.families.configuration import (
    configure_terrain,
    tile_boundary,
)
from parkour_lab.environments.terrains.families.generation import generate_terrain
from parkour_lab.environments.terrains.families.specification import (
    MIXED_COLUMNS,
    TerrainConfig,
)
from tests.support.isaaclab import (
    NativeConfig,
    modules,
    pinned_generator,
    pinned_source_root,
)


@contextmanager
def native_geometry():
    """Real native primitives/configclass; only unused runtime manager types are doubled."""
    root = pinned_source_root()
    with pinned_generator():
        package = ModuleType("isaaclab.terrains.height_field")
        package.__path__ = []
        replacements = {
            package.__name__: package,
            "isaaclab.managers": NS(
                CurriculumTermCfg=NativeConfig, TerminationTermCfg=NativeConfig
            ),
        }
        paths = (
            ("height_field.utils", "height_field/utils.py"),
            ("height_field.hf_terrains", "height_field/hf_terrains.py"),
            ("height_field.hf_terrains_cfg", "height_field/hf_terrains_cfg.py"),
            ("trimesh.utils", "trimesh/utils.py"),
            ("trimesh.mesh_terrains", "trimesh/mesh_terrains.py"),
            ("trimesh.mesh_terrains_cfg", "trimesh/mesh_terrains_cfg.py"),
            ("terrain_generator_cfg", "terrain_generator_cfg.py"),
        )
        with modules(replacements):
            with modules(
                {
                    f"isaaclab.terrains.{name}": ModuleType(f"isaaclab.terrains.{name}")
                    for name, _ in paths
                }
            ):
                native = sys.modules["isaaclab.terrains"]
                native.SubTerrainBaseCfg = sys.modules[
                    "isaaclab.terrains.sub_terrain_cfg"
                ].SubTerrainBaseCfg
                for suffix, path in paths:
                    name = f"isaaclab.terrains.{suffix}"
                    spec = importlib.util.spec_from_file_location(
                        name, root / "terrains" / path
                    )
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[name] = module
                    spec.loader.exec_module(module)
                    for key, value in vars(module).items():
                        if key.endswith("Cfg"):
                            setattr(native, key, value)
                            if key.startswith("Hf"):
                                setattr(package, key, value)
                yield native


def config():
    return NS(
        scene=NS(terrain=NS()),
        commands=NS(base_velocity=NS()),
        curriculum=NS(),
        terminations=NS(),
    )


def subterrain(family, subtype, *, seed=7, column=0, fixed_level=None):
    return NS(
        family=family,
        subtype=subtype,
        fixed_level=fixed_level,
        seed=seed,
        column=column,
        size=(12.0, 12.0),
    )


def height_at(mesh, x, y):
    """Intersect one vertical ray with real contact triangles, without an R-tree."""
    triangles = mesh.triangles
    a = triangles[:, 0]
    u, v = triangles[:, 1] - a, triangles[:, 2] - a
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    valid = np.abs(det) > 1e-12
    a, u, v, det = a[valid], u[valid], v[valid], det[valid]
    dx, dy = x - a[:, 0], y - a[:, 1]
    first = (dx * v[:, 1] - dy * v[:, 0]) / det
    second = (u[:, 0] * dy - u[:, 1] * dx) / det
    inside = (first >= -1e-9) & (second >= -1e-9) & (first + second <= 1 + 1e-9)
    if not inside.any():
        raise AssertionError(f"No terrain support at {(x, y)}")
    return float((a[:, 2] + first * u[:, 2] + second * v[:, 2])[inside].max())


class NativeTerrainGeometryTests(unittest.TestCase):
    def test_real_meshes_have_finite_support_and_both_signed_pyramids(self):
        with native_geometry():
            for level in (0, 3, 7):
                for family, subtype in dict.fromkeys(MIXED_COLUMNS):
                    with self.subTest(level=level, family=family, subtype=subtype):
                        pieces, origin = generate_terrain(
                            (level + 0.4) / 8, subterrain(family, subtype)
                        )
                        mesh = trimesh.util.concatenate(pieces)
                        self.assertTrue(np.isfinite(mesh.vertices).all())
                        self.assertTrue(np.isfinite(origin).all())
                        np.testing.assert_allclose(
                            mesh.bounds[:, :2], [[0, 0], [12, 12]], atol=1e-6
                        )
                        center = height_at(mesh, 6.0, 6.0)
                        outer = height_at(mesh, 9.0, 6.0)
                        for x, y in ((0.2, 0.2), (3, 3), (6, 6), (9, 9), (11.8, 11.8)):
                            self.assertTrue(np.isfinite(height_at(mesh, x, y)))
                        self.assertGreaterEqual(origin[2] + 1e-7, center)
                        if family == "Flat":
                            np.testing.assert_array_equal(mesh.vertices[:, 2], 0)
                        else:
                            self.assertGreater(np.ptp(mesh.vertices[:, 2]), 0)
                        if family in ("Stairs", "Slopes"):
                            sign = 1 if subtype == "pyramid" else -1
                            self.assertGreater(sign * (center - outer), 0)
                            self.assertAlmostEqual(
                                center, height_at(mesh, 6.2, 6.2), places=6
                            )

    def test_uneven_native_randomness_is_reproducible_isolated_and_level_sensitive(
        self,
    ):
        with native_geometry():
            np.random.seed(108)
            state = np.random.get_state()
            first, _ = generate_terrain(0.01, subterrain("Uneven", "noise"))
            after = np.random.random(6)
            np.random.set_state(state)
            np.testing.assert_array_equal(after, np.random.random(6))
            second, _ = generate_terrain(0.1, subterrain("Uneven", "noise"))
            np.testing.assert_array_equal(first[0].vertices, second[0].vertices)
            high, _ = generate_terrain(0.99, subterrain("Uneven", "noise"))
            changed, _ = generate_terrain(0.01, subterrain("Uneven", "noise", seed=8))
            self.assertGreater(
                np.std(high[0].vertices[:, 2]), np.std(first[0].vertices[:, 2])
            )
            self.assertFalse(np.array_equal(changed[0].vertices, first[0].vertices))

    def test_fixed_level_geometry_is_independent_of_native_difficulty_sampling(self):
        with native_geometry():
            for family, subtype in dict.fromkeys(MIXED_COLUMNS):
                options = subterrain(family, subtype, fixed_level=3)
                low, a = generate_terrain(0.0, options)
                high, b = generate_terrain(1.0, options)
                np.testing.assert_array_equal(a, b)
                for left, right in zip(low, high, strict=True):
                    np.testing.assert_array_equal(left.vertices, right.vertices)
                    np.testing.assert_array_equal(left.faces, right.faces)

    def test_native_configs_preserve_columns_ordering_and_one_row_fixed_eval(self):
        with native_geometry():
            train, evaluation = config(), config()
            configure_terrain(
                train, TerrainConfig(), training=True, task_id="Go2-Mixed-v1", seed=5
            )
            configure_terrain(
                evaluation,
                TerrainConfig(fixed_level=7),
                training=False,
                task_id="Go2-Mixed-v1",
                seed=5,
            )
            generator = train.scene.terrain.terrain_generator
            self.assertTrue(generator.curriculum)
            self.assertFalse(generator.use_cache)
            self.assertEqual(generator.num_rows, 8)
            self.assertEqual(train.scene.terrain.max_init_terrain_level, 1)
            self.assertIsNotNone(train.curriculum.terrain_levels)
            fixed = evaluation.scene.terrain.terrain_generator
            self.assertEqual(fixed.num_rows, 1)
            self.assertEqual(evaluation.scene.terrain.max_init_terrain_level, 0)
            self.assertIsNone(evaluation.curriculum.terrain_levels)
            self.assertTrue(fixed.curriculum)
            self.assertEqual(
                tuple((c.family, c.subtype) for c in generator.sub_terrains.values()),
                MIXED_COLUMNS,
            )
            self.assertTrue(
                all(c.fixed_level == 7 for c in fixed.sub_terrains.values())
            )
            flat = config()
            configure_terrain(
                flat, TerrainConfig(), training=True, task_id="Go2-Flat-v1", seed=5
            )
            self.assertEqual(flat.scene.terrain.terrain_type, "plane")
            self.assertIsNone(flat.scene.terrain.terrain_generator)
            self.assertIsNone(flat.curriculum.terrain_levels)

    def test_actual_native_dispatch_preserves_each_family_and_exact_row_band(self):
        with native_geometry():
            cfg = config()
            configure_terrain(
                cfg, TerrainConfig(), training=True, task_id="Go2-Mixed-v1", seed=5
            )
            generator_type = sys.modules[
                "isaaclab.terrains.terrain_generator"
            ].TerrainGenerator
            generator = generator_type.__new__(generator_type)
            generator.cfg = cfg.scene.terrain.terrain_generator
            generator.np_rng = np.random.default_rng(5)
            generated = []

            def capture(difficulty, sub_cfg):
                generated.append((sub_cfg.family, sub_cfg.subtype, int(difficulty * 8)))
                return None, None

            generator._get_terrain_mesh = capture
            generator._add_sub_terrain = lambda *args: None
            generator._generate_curriculum_terrains()
            expected = [
                (family, kind, row)
                for family, kind in MIXED_COLUMNS
                for row in range(8)
            ]
            self.assertEqual(generated, expected)

    def test_boundary_is_first_strict_root_crossing_without_an_inward_margin(self):
        class Scene(dict):
            pass

        origins = torch.tensor([[10.0, 20.0, 0.0]]).repeat(5, 1)
        scene = Scene(
            robot=NS(
                data=NS(
                    root_pos_w=origins
                    + torch.tensor(
                        [
                            [0.0, 0, 0],
                            [6.0, 0, 0],
                            [6.01, 0, 0],
                            [0, -6.01, 0],
                            [4.3, 4.3, 0],
                        ]
                    )
                )
            )
        )
        scene.terrain = NS(env_origins=origins)
        env = NS(scene=scene, cfg=NS(benchmark_terrain={"tile_size_m": [12.0, 12.0]}))
        self.assertEqual(tile_boundary(env).tolist(), [False, False, True, True, False])


if __name__ == "__main__":
    unittest.main()
