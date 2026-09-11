"""Replay-only terrain collision bisection, applied before physics initialization.

Keep the generated mesh (including ray-caster geometry), robot, reset and motor
pipeline. Change only the terrain's collisionEnabled attribute, not robot
colliders or mechanical properties. This is not a usable locomotion environment.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math


KINDS = ("native", "ground_off")
GROUND_PATH = "/World/Ground"
MESH_PATH = GROUND_PATH + "/terrain/mesh"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _runtime():
    from types import SimpleNamespace

    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    from isaaclab.sim import SimulationContext
    from isaaclab.sim.utils.stage import get_current_stage

    return SimpleNamespace(
        np=np,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        PhysxSchema=PhysxSchema,
        sim=SimulationContext.instance(),
        stage=get_current_stage(),
    )


def _offset_snapshot(attribute):
    """Preserve USD's automatic -inf sentinel without emitting nonfinite JSON.

    The sentinel asks PhysX to choose an offset; it is not a measured numeric
    backend offset. In particular, never author a replacement value here.
    """
    value = attribute.Get()
    authored = attribute.HasAuthoredValueOpinion()
    _require(type(authored) is bool, "Missing collision offset authoring status")
    _require(
        type(value) in (int, float) and (math.isfinite(value) or value == -math.inf),
        "Missing or invalid collision offset",
    )
    return {
        "usd_value": "-inf" if value == -math.inf else float(value),
        "mode": "simulation_default" if value == -math.inf else "numeric",
        "has_authored_value": authored,
    }


def _validate_offset_snapshot(value):
    _require(
        isinstance(value, dict)
        and set(value) == {"usd_value", "mode", "has_authored_value"}
        and type(value["has_authored_value"]) is bool,
        "Invalid collision offset metadata",
    )
    raw = value["usd_value"]
    _require(
        (value["mode"] == "simulation_default" and raw == "-inf")
        or (
            value["mode"] == "numeric"
            and type(raw) in (int, float)
            and math.isfinite(raw)
        ),
        "Invalid collision offset value or automatic sentinel",
    )


def _ground_snapshot(rt):
    root = rt.stage.GetPrimAtPath(GROUND_PATH)
    _require(root.IsValid(), "Ground prim missing")
    colliders = [
        prim
        for prim in rt.Usd.PrimRange(root, rt.Usd.TraverseInstanceProxies())
        if prim.HasAPI(rt.UsdPhysics.CollisionAPI)
    ]
    _require(
        [str(prim.GetPath()) for prim in colliders] == [MESH_PATH],
        "Expected exactly the generated ground mesh collider; unsupported scene",
    )
    prim = colliders[0]
    _require(prim.IsA(rt.UsdGeom.Mesh), "Ground collider must be a mesh")
    mesh = rt.UsdGeom.Mesh(prim)
    digest = hashlib.sha256()
    for label, data, dtype in (
        ("points", mesh.GetPointsAttr().Get(), "<f8"),
        ("counts", mesh.GetFaceVertexCountsAttr().Get(), "<i8"),
        ("indices", mesh.GetFaceVertexIndicesAttr().Get(), "<i8"),
        (
            "world_transform",
            rt.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                rt.Usd.TimeCode.Default()
            ),
            "<f8",
        ),
    ):
        array = rt.np.asarray(data, dtype=dtype)
        _require(
            array.size > 0 and rt.np.isfinite(array).all(), "Invalid ground geometry"
        )
        digest.update(label.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    collider = rt.UsdPhysics.CollisionAPI(prim)
    enabled = collider.GetCollisionEnabledAttr().Get()
    _require(type(enabled) is bool, "Collision-enabled readback unavailable")
    offsets = {}
    _require(
        prim.HasAPI(rt.PhysxSchema.PhysxCollisionAPI), "Missing PhysX collision schema"
    )
    physics = rt.PhysxSchema.PhysxCollisionAPI(prim)
    for name in ("ContactOffset", "RestOffset"):
        offsets[name] = _offset_snapshot(getattr(physics, f"Get{name}Attr")())
    return {
        "mesh_path": MESH_PATH,
        "geometry_world_sha256": digest.hexdigest(),
        "collision_enabled": enabled,
        "offset_attributes_m": offsets,
    }


def validate_collision_metadata(meta):
    _require(isinstance(meta, dict), "Collision probe metadata missing")
    try:
        json.dumps(meta, allow_nan=False)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError("Invalid collision probe metadata") from error
    _require(
        meta.get("kind") == "startup_ground_collision_probe"
        and type(meta.get("schema_version")) is int
        and meta["schema_version"] == 1
        and meta.get("mode") in KINDS
        and meta.get("applied_before_physics") is True
        and meta.get("runtime_verified") is True
        and meta.get("construction_step_index") == 0
        and type(meta.get("construction_step_index")) is int
        and type(meta.get("construction_time_s")) in (int, float)
        and meta.get("construction_time_s") == 0.0,
        "Collision probe was not verified before initialization and at runtime",
    )
    before, after, live = (meta.get(name) for name in ("before", "after", "runtime"))
    for value in (before, after, live):
        _require(
            isinstance(value, dict) and value.get("mesh_path") == MESH_PATH,
            "Missing ground readback",
        )
        digest = value.get("geometry_world_sha256")
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            "Invalid mesh hash",
        )
        _require(type(value.get("collision_enabled")) is bool, "Invalid collision flag")
        offsets = value.get("offset_attributes_m")
        _require(
            isinstance(offsets, dict)
            and set(offsets) == {"ContactOffset", "RestOffset"},
            "Invalid offsets",
        )
        for offset in offsets.values():
            _validate_offset_snapshot(offset)
    expected = copy.deepcopy(before)
    _require(
        before["collision_enabled"] is True, "Source ground collision was not enabled"
    )
    expected["collision_enabled"] = meta["mode"] == "native"
    _require(
        after == expected and live == expected,
        "Ground changed beyond collision participation or readback drifted",
    )
    _require(
        meta.get("enabled_other_external_colliders") == [],
        "Other external colliders remain",
    )
    _require(
        type(meta.get("enabled_robot_collider_count")) is int
        and meta["enabled_robot_collider_count"] > 0,
        "Robot collision shapes must remain enabled",
    )


def configure_collision_probe(env_cfg, mode):
    """Wrap the configured terrain importer; do not alter training defaults."""
    scene = env_cfg.scene
    ground = scene.ground
    generator = ground.terrain_generator
    _require(
        mode in KINDS
        and type(scene.num_envs) is int
        and scene.num_envs == 1
        and env_cfg.evaluation_family == "high_step"
        and type(env_cfg.evaluation_level) is int
        and env_cfg.evaluation_level in (0, 6)
        and env_cfg.evaluation_geometry_variant == 0
        and env_cfg.evaluation_command_profile == "translation_only"
        and env_cfg.curriculum is None
        and ground.prim_path == GROUND_PATH
        and ground.terrain_type == "generator"
        and ground.use_terrain_origins is True
        and generator.num_cols == 1
        and generator.curriculum is True
        and env_cfg.sim.dt == 0.005
        and env_cfg.decimation == 4
        and env_cfg.sim.physx.solver_type == 1,
        "Collision bisection requires fixed uncentered high_step L0/L6, one environment, TGS, 5ms/4",
    )
    base = ground.class_type
    _require(
        isinstance(base, type) and not getattr(base, "_collision_probe", False),
        "Invalid or duplicate collision probe",
    )
    evidence = {
        "kind": "startup_ground_collision_probe",
        "schema_version": 1,
        "mode": mode,
        "applied_before_physics": False,
        "runtime_verified": False,
        "limitations": [
            "No ground support: this is a dynamics diagnostic, never locomotion evaluation or a training preset.",
            "Collision participation is changed before initialization, so contact history and scene topology may both change.",
            "USD collision flags and direct net-force readbacks do not expose all GPU constraint state.",
        ],
    }

    class CollisionProbeTerrainImporter(base):
        _collision_probe = True

        def __init__(self, cfg):
            super().__init__(cfg)
            rt = _runtime()
            _require(
                not rt.sim.is_playing()
                and rt.sim.current_time_step_index == 0
                and rt.sim.current_time == 0.0,
                "Ground edit must precede all physics initialization",
            )
            before = _ground_snapshot(rt)
            _require(
                before["collision_enabled"],
                "Ground unexpectedly lacks collision participation",
            )
            if mode == "ground_off":
                attr = rt.UsdPhysics.CollisionAPI(
                    rt.stage.GetPrimAtPath(MESH_PATH)
                ).GetCollisionEnabledAttr()
                _require(attr.Set(False), "Failed to disable ground collider")
            after = _ground_snapshot(rt)
            expected = {**before, "collision_enabled": mode == "native"}
            _require(after == expected, "Unexpected terrain mutation")
            evidence.update(
                applied_before_physics=True,
                construction_step_index=0,
                construction_time_s=0.0,
                before=before,
                after=after,
            )

    ground.class_type = CollisionProbeTerrainImporter
    return evidence


def verify_collision_probe(env, evidence):
    """Observe the constructed scene, without state writes, forwarding or stepping."""
    rt = _runtime()
    result = copy.deepcopy(evidence)
    root = env.scene["robot"].cfg.prim_path
    _require(
        root == "/World/envs/env_0/Robot" or root == "/World/envs/env_.*/Robot",
        "Unexpected single-environment robot scope",
    )
    root = "/World/envs/env_0/Robot"
    robot_count = 0
    other = []
    for prim in rt.Usd.PrimRange.Stage(rt.stage, rt.Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(rt.UsdPhysics.CollisionAPI):
            continue
        if not rt.UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        path = str(prim.GetPath())
        if path.startswith(root + "/"):
            robot_count += 1
        elif path != MESH_PATH:
            other.append(path)
    result.update(
        runtime=_ground_snapshot(rt),
        runtime_verified=True,
        enabled_robot_collider_count=robot_count,
        enabled_other_external_colliders=sorted(other),
    )
    validate_collision_metadata(result)
    return result
