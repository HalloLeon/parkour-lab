"""One connected world with an assigned supported start and no route assistance.

The native importer retains local builder points and authors only a rigid USD
transform. One environment avoids duplicating the large mesh or silently using
the stock ray caster on only the first of several independent worlds.
"""

from __future__ import annotations

import numpy as np
import torch

from .worlds import build_world


def supported_start(world):
    """Use the actual converted flat pad, not an analytical obstacle height."""
    metadata = world["metadata"]
    trial = metadata["trial"]
    vertices = world["vertices"].astype(np.float32)
    pad = np.asarray(trial["start_pad_local_m"])
    inside = ((vertices[:, :2] >= pad[0]) & (vertices[:, :2] <= pad[1])).all(-1)
    heights = vertices[inside, 2]
    if not len(heights) or not np.isfinite(heights).all() or np.ptp(heights) != 0:
        raise ValueError("Connected start pad must have constant native support height")
    return (
        np.asarray([*trial["start_xy_world_m"], float(heights[0])]),
        float(trial["heading_world_rad"]),
    )


def connected_workspace(env, margin_m=0.25):
    """Censor map departures in mesh coordinates, never steer the robot."""
    terrain = env.scene.terrain
    transform = torch.as_tensor(
        terrain.metadata["trial"]["local_to_world_column_transform"],
        dtype=env.scene["robot"].data.body_pos_w.dtype,
        device=env.device,
    )
    local = (env.scene["robot"].data.body_pos_w - transform[:3, 3]) @ transform[:3, :3]
    size = torch.as_tensor(terrain.metadata["size_m"], device=env.device)
    outside = (
        ((local[..., :2] < margin_m) | (local[..., :2] > size - margin_m))
        .any(dim=-1)
        .any(dim=-1)
    )
    return outside & ~env.termination_manager.terminated


def _readback(importer, path):
    """Verify native identity/material only; builder owns geometry-bound checks."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

    import isaaclab.sim as sim_utils

    prim = sim_utils.get_current_stage().GetPrimAtPath(path + "/mesh")
    mesh = UsdGeom.Mesh(prim)
    world = importer.world
    matrix = np.asarray(importer.metadata["trial"]["local_to_world_column_transform"])
    if (
        not np.array_equal(
            np.asarray(mesh.GetPointsAttr().Get()), world["vertices"].astype(np.float32)
        )
        or not np.array_equal(
            np.asarray(mesh.GetFaceVertexIndicesAttr().Get()), world["faces"].ravel()
        )
        or not np.array_equal(
            np.asarray(mesh.GetFaceVertexCountsAttr().Get()),
            np.full(len(world["faces"]), 3),
        )
        or not np.array_equal(
            np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)), matrix.T
        )
        or not prim.HasAPI(UsdPhysics.CollisionAPI)
        or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True
        or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "none"
    ):
        raise ValueError(
            "Connected native mesh, rigid transform or collider differs from builder"
        )
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
    if not material:
        raise ValueError("Connected native mesh has no bound physics material")
    material_prim = material.GetPrim()
    usd = UsdPhysics.MaterialAPI(material_prim)
    physx = PhysxSchema.PhysxMaterialAPI(material_prim)
    values = {
        "static_friction": usd.GetStaticFrictionAttr().Get(),
        "dynamic_friction": usd.GetDynamicFrictionAttr().Get(),
        "restitution": usd.GetRestitutionAttr().Get(),
        "friction_combine_mode": physx.GetFrictionCombineModeAttr().Get(),
        "restitution_combine_mode": physx.GetRestitutionCombineModeAttr().Get(),
    }
    if (
        not material_prim.HasAPI(UsdPhysics.MaterialAPI)
        or not material_prim.HasAPI(PhysxSchema.PhysxMaterialAPI)
        or values
        != dict(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        )
    ):
        raise ValueError(
            "Connected native terrain material differs from the declared contract"
        )
    return {
        "scope": "Exact builder float32 mesh, rigid transform and USD collider/material identity; not cooked contact or traversal",
        "mesh_path": str(prim.GetPath()),
        "vertices_float32_sha256": importer.metadata["vertices_float32_sha256"],
        "faces_int64_sha256": importer.metadata["faces_int64_sha256"],
        "local_to_world_column_transform": matrix.tolist(),
        "material": values,
        "start_position_m": importer.start_position.tolist(),
        "start_heading_rad": importer.start_heading,
    }


def _importer_type():
    from isaaclab.terrains import TerrainImporter

    class ConnectedWorldImporter(TerrainImporter):
        def __init__(self, cfg):
            import isaaclab.sim as sim_utils
            import trimesh
            from pxr import Gf, UsdGeom, UsdPhysics

            cfg.validate()
            if cfg.num_envs != 1:
                raise ValueError("Connected worlds currently require one environment")
            self.cfg = cfg
            self.device = sim_utils.SimulationContext.instance().device
            self.terrain_prim_paths = []
            self.terrain_origins = None
            self.env_origins = torch.zeros((1, 3), device=self.device)
            self._terrain_flat_patches = {}
            self.world = build_world(**cfg.world)
            self.metadata = self.world["metadata"]
            self.start_position, self.start_heading = supported_start(self.world)
            self.import_mesh(
                "terrain",
                trimesh.Trimesh(
                    vertices=self.world["vertices"],
                    faces=self.world["faces"],
                    process=False,
                ),
            )
            path = cfg.prim_path + "/terrain"
            stage = sim_utils.get_current_stage()
            prim = stage.GetPrimAtPath(path + "/mesh")
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(
                "none"
            )
            parent = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            parent.ClearXformOpOrder()
            matrix = np.asarray(
                self.metadata["trial"]["local_to_world_column_transform"]
            ).T
            parent.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Matrix4d(*matrix.ravel().tolist())
            )
            self.native_receipt = _readback(self, path)
            self.set_debug_vis(cfg.debug_vis)

    return ConnectedWorldImporter


def configure(cfg, task):
    """Use the ordinary scene lifecycle with one source-identity native terrain."""
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.terrains import TerrainImporterCfg
    from isaaclab.utils import configclass

    @configclass
    class ConnectedWorldCfg(TerrainImporterCfg):
        class_type = _importer_type()
        terrain_type = "connected"
        world: dict = {}

    previous = cfg.scene.terrain
    cfg.scene.terrain = ConnectedWorldCfg(
        prim_path=previous.prim_path,
        physics_material=previous.physics_material,
        world=task.world,
        num_envs=1,
        debug_vis=False,
    )
    cfg.terminations.connected_workspace = TerminationTermCfg(
        func=connected_workspace, time_out=True
    )
