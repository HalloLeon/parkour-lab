"""Bounded source/native boundary checks, not cooked PhysX contact evidence."""

from contextlib import contextmanager
import importlib.util
import math
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from parkour_lab.environments import connected
from tests.support.isaaclab import modules, pinned_generator, pinned_source_root


def small_world():
    yaw = 0.37
    matrix = np.eye(4)
    matrix[:2, :2] = [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    return dict(
        vertices=np.array([[1, 1, 0.7], [3, 1, 0.7], [1, 3, 0.7], [3, 3, 0.7]]),
        faces=np.array([[0, 1, 2], [1, 3, 2]]),
        metadata=dict(
            size_m=[24, 24],
            vertices_float32_sha256="points",
            faces_int64_sha256="faces",
            trial=dict(
                start_pad_local_m=[[1, 1], [3, 3]],
                start_xy_world_m=(matrix[:2, :2] @ [2, 2]).tolist(),
                heading_world_rad=yaw - math.pi / 2,
                local_to_world_column_transform=matrix.tolist(),
            ),
        ),
    )


@contextmanager
def native_importer():
    """Execute pinned TerrainImporter.import_mesh; double unavailable USD/Sim."""
    root = pinned_source_root()
    state = dict(matrix=np.eye(4), approximation=None)

    def attribute(name):
        return NS(
            Get=lambda: state[name], Set=lambda value: state.update({name: value})
        )

    prim = NS(HasAPI=lambda api: True, GetPath=lambda: "/World/ground/terrain/mesh")
    mesh = NS(
        GetPointsAttr=lambda: attribute("points"),
        GetFaceVertexIndicesAttr=lambda: attribute("indices"),
        GetFaceVertexCountsAttr=lambda: attribute("counts"),
    )
    material_prim = NS(HasAPI=lambda api: True)
    material = NS(GetPrim=lambda: material_prim)
    usd_material = NS(
        GetStaticFrictionAttr=lambda: attribute("static_friction"),
        GetDynamicFrictionAttr=lambda: attribute("dynamic_friction"),
        GetRestitutionAttr=lambda: attribute("restitution"),
    )
    physx_material = NS(
        GetFrictionCombineModeAttr=lambda: attribute("friction_combine_mode"),
        GetRestitutionCombineModeAttr=lambda: attribute("restitution_combine_mode"),
    )
    collision = Mock(
        return_value=NS(GetCollisionEnabledAttr=lambda: attribute("collision"))
    )
    approximation = NS(
        GetApproximationAttr=lambda: attribute("approximation"),
        CreateApproximationAttr=lambda: attribute("approximation"),
    )
    mesh_collision = Mock(return_value=approximation)
    mesh_collision.Apply.return_value = approximation
    xform = NS(
        ClearXformOpOrder=lambda: None, AddTransformOp=lambda _: attribute("matrix")
    )
    pxr = NS(
        Gf=NS(Matrix4d=lambda *values: np.asarray(values).reshape(4, 4)),
        UsdGeom=NS(
            Mesh=lambda _: mesh,
            Xformable=lambda _: xform,
            XformOp=NS(PrecisionDouble="double"),
            XformCache=lambda: NS(GetLocalToWorldTransform=lambda _: state["matrix"]),
        ),
        UsdPhysics=NS(
            CollisionAPI=collision,
            MeshCollisionAPI=mesh_collision,
            MaterialAPI=lambda _: usd_material,
        ),
        PhysxSchema=NS(PhysxMaterialAPI=lambda _: physx_material),
        UsdShade=NS(
            MaterialBindingAPI=lambda _: NS(
                ComputeBoundMaterial=lambda _: (material, None)
            )
        ),
    )

    def spawn(path, source, *, visual_material, physics_material):
        state.update(
            path=path,
            points=source.vertices.astype(np.float32),
            indices=source.faces.ravel().astype(np.int32),
            counts=np.full(len(source.faces), 3),
            collision=True,
            **vars(physics_material),
        )

    replacements = {
        name: ModuleType(name)
        for name in ("isaaclab", "isaaclab.terrains", "isaaclab.sim")
    }
    for module in replacements.values():
        module.__path__ = []
    sim = replacements["isaaclab.sim"]
    sim.SimulationContext = NS(instance=lambda: NS(device="cpu"))
    sim.get_current_stage = lambda: NS(GetPrimAtPath=lambda _: prim)
    replacements.update(
        {
            "isaaclab.markers": NS(VisualizationMarkers=object),
            "isaaclab.markers.config": NS(FRAME_MARKER_CFG=None),
            "isaaclab.terrains.utils": NS(
                create_prim_from_mesh=Mock(side_effect=spawn)
            ),
            "pxr": pxr,
        }
    )
    name = "isaaclab.terrains.terrain_importer"
    spec = importlib.util.spec_from_file_location(
        name, root / "terrains/terrain_importer.py"
    )
    module = importlib.util.module_from_spec(spec)
    replacements[name] = module
    with modules(replacements):
        spec.loader.exec_module(module)
        replacements["isaaclab.terrains"].TerrainImporter = module.TerrainImporter
        cfg = NS(
            validate=lambda: None,
            num_envs=1,
            world={"target_family": "ramp"},
            prim_path="/World/ground",
            visual_material=None,
            debug_vis=False,
            physics_material=NS(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
            ),
        )
        yield cfg, state, replacements["isaaclab.terrains.utils"].create_prim_from_mesh


class ConnectedTests(unittest.TestCase):
    def test_actual_physical_failure_accepts_descent_but_rejects_below_world(self):
        from tests.environments.test_commands import native_commands

        class Scene(dict):
            env_origins = torch.zeros(3, 3)

        root = torch.tensor([[0.0, 0.0, 1.1], [0.0, 0.0, 0.4], [0.0, 0.0, -0.53]])
        hits = torch.tensor(
            [[[0.0, 0.0, 0.7]], [[0.0, 0.0, 0.0]], [[float("inf")] * 3]]
        )
        env = NS(
            num_envs=3,
            scene=Scene(
                robot=NS(
                    data=NS(
                        root_pos_w=root,
                        projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(
                            3, 1
                        ),
                    )
                ),
                base_height_scanner=NS(data=NS(ray_hits_w=hits)),
            ),
        )
        with native_commands() as commands:
            failed = commands.procedural_physical_failure(
                env,
                minimum_m=0.12,
                minimum_surface_z_m=-0.02,
                fall_margin_m=0.5,
                max_tilt_rad=math.pi / 4,
            )
        self.assertEqual(failed.tolist(), [False, False, True])

    def test_pinned_native_config_installs_single_ray_target_and_real_importer(self):
        with pinned_generator(), native_importer() as (original, _, _):
            sim = sys.modules["isaaclab.sim"]
            sim.PreviewSurfaceCfg = sim.RigidBodyMaterialCfg = NS
            name = "isaaclab.terrains.terrain_importer_cfg"
            spec = importlib.util.spec_from_file_location(
                name, pinned_source_root() / "terrains/terrain_importer_cfg.py"
            )
            module = importlib.util.module_from_spec(spec)
            with modules(
                {name: module, "isaaclab.managers": NS(TerminationTermCfg=NS)}
            ):
                spec.loader.exec_module(module)
                sys.modules[
                    "isaaclab.terrains"
                ].TerrainImporterCfg = module.TerrainImporterCfg
                cfg = NS(scene=NS(terrain=original), terminations=NS())
                connected.configure(cfg, NS(world=original.world))
                terrain_cfg = cfg.scene.terrain
                terrain_cfg.validate()
                self.assertEqual(terrain_cfg.terrain_type, "connected")
                self.assertEqual(terrain_cfg.num_envs, 1)
                self.assertEqual(terrain_cfg.collision_group, -1)
                self.assertEqual(terrain_cfg.world, original.world)
                self.assertEqual(terrain_cfg.prim_path, "/World/ground")
                self.assertEqual(
                    terrain_cfg.class_type.__name__, "ConnectedWorldImporter"
                )
                self.assertIs(
                    cfg.terminations.connected_workspace.func,
                    connected.connected_workspace,
                )
                self.assertTrue(cfg.terminations.connected_workspace.time_out)
                with patch.object(connected, "build_world", return_value=small_world()):
                    terrain = terrain_cfg.class_type(terrain_cfg)
                self.assertEqual(terrain.terrain_prim_paths, ["/World/ground/terrain"])

    def test_actual_fixed_ramp_down_supported_pad_is_native_flat(self):
        world = connected.build_world(
            "ramp",
            10,
            reverse=True,
            world_yaw=0.37,
            coarse_seed=20261002,
            fine_seed=20261003,
        )
        position, heading = connected.supported_start(world)
        self.assertGreater(position[2], 0.3)
        self.assertLess(position[2], 0.4)
        self.assertAlmostEqual(heading, 0.37 - math.pi / 2)
        np.testing.assert_array_equal(
            position[:2], world["metadata"]["trial"]["start_xy_world_m"]
        )
        # One representative full world only; the importer checks use tiny meshes.

    def test_supported_start_rejects_nonflat_or_missing_pad(self):
        world = small_world()
        world["vertices"][0, 2] += 0.01
        with self.assertRaisesRegex(ValueError, "constant native support"):
            connected.supported_start(world)
        world["vertices"][:, :2] += 20
        with self.assertRaisesRegex(ValueError, "constant native support"):
            connected.supported_start(world)

    def test_pinned_import_mesh_keeps_local_identity_and_separate_high_reset(self):
        world = small_world()
        with (
            native_importer() as (cfg, state, spawn),
            patch.object(connected, "build_world", return_value=world) as build,
        ):
            terrain = connected._importer_type()(cfg)
            build.assert_called_once_with(**cfg.world)
            spawn.assert_called_once()
            np.testing.assert_array_equal(
                state["points"], world["vertices"].astype(np.float32)
            )
            np.testing.assert_array_equal(
                state["matrix"],
                np.asarray(
                    world["metadata"]["trial"]["local_to_world_column_transform"]
                ).T,
            )
            self.assertEqual(terrain.terrain_prim_paths, ["/World/ground/terrain"])
            self.assertTrue(torch.equal(terrain.env_origins, torch.zeros(1, 3)))
            self.assertAlmostEqual(terrain.start_position[2], float(np.float32(0.7)))
            self.assertIs(terrain.world, world)
            self.assertEqual(
                terrain.native_receipt["material"]["friction_combine_mode"], "multiply"
            )
            for key, bad in (
                ("points", state["points"] + 0.01),
                ("indices", state["indices"][::-1]),
                ("matrix", np.eye(4)),
                ("collision", False),
                ("approximation", "convexHull"),
                ("dynamic_friction", 0.8),
                ("friction_combine_mode", "average"),
            ):
                with self.subTest(key=key):
                    original = state[key]
                    state[key] = bad
                    with self.assertRaisesRegex(ValueError, "differs"):
                        connected._readback(terrain, state["path"])
                    state[key] = original

    def test_importer_rejects_replication_before_build(self):
        with (
            native_importer() as (cfg, _, _),
            patch.object(connected, "build_world") as build,
        ):
            cfg.num_envs = 2
            with self.assertRaisesRegex(ValueError, "one environment"):
                connected._importer_type()(cfg)
            build.assert_not_called()

    def test_workspace_uses_inverse_yaw_and_does_not_mask_physical_failure(self):
        world = small_world()
        rotation = torch.tensor(
            world["metadata"]["trial"]["local_to_world_column_transform"],
            dtype=torch.float32,
        )[:3, :3]
        local = torch.tensor(
            [[[2.0, 3.0, 0.4]], [[24.0, 3.0, 0.4]], [[24.0, 3.0, 0.4]]]
        )
        scene = {"robot": NS(data=NS(body_pos_w=local @ rotation.T))}

        class Scene(dict):
            terrain = NS(metadata=world["metadata"])

        env = NS(
            scene=Scene(scene),
            device="cpu",
            termination_manager=NS(terminated=torch.tensor([False, False, True])),
        )
        self.assertEqual(
            connected.connected_workspace(env).tolist(), [False, True, False]
        )


if __name__ == "__main__":
    unittest.main()
