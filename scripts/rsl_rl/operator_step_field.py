"""Opt-in supported rough step fields, not routes or legacy terrain replacements.

Only the four existing step_hills columns opt in. Per-cell tops and explicit
vertical seams avoid the bevels of a shared-vertex heightfield. The native
callback receipts attest generated triangles, not PhysX contact or capability.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math

import numpy as np

from .operator_stress_terrain import multiscale_field

VERSION = "operator_step_field_v1"
BOOTSTRAP_VERSION = "operator_step_field_bootstrap_v1"
TILE_SIZE = (16.0, 16.0)
DIFFICULTY = (0.15, 0.55)
CELL_SIZE = 0.5
RESOLUTION = 0.1
_CONFIGURATION = None
_RECEIPTS = []


def envelope(version=VERSION):
    if version not in (VERSION, BOOTSTRAP_VERSION):
        raise ValueError("Unknown step-field geometry version")
    result = {
        "version": version,
        "tile_size_m": list(TILE_SIZE),
        "replaced_columns": [12, 13, 14, 15],
        "profile_tag": "step_hills",
        "riser_formula_m": "0.04 + 0.08 * difficulty",
        "acquisition_difficulty_range": list(DIFFICULTY),
        "acquisition_riser_range_m": [0.052, 0.084],
        "axis_aligned_cell_tread_m": CELL_SIZE,
        "maximum_levels": 2,
        "roughness_bound_formula_m": "0.004 * difficulty",
        "exact_spawn_pad_half_width_m": 1.0,
        "flat_border_width_m": 1.0,
        "through_going_flat_band": False,
        "holes_routes_waypoints_success_resets": False,
        "scope": "Partial gentle vertical-step acquisition, not the full 4–24cm curriculum or high-step qualification; tread width is axis-aligned, not every travel direction",
    }
    if version == BOOTSTRAP_VERSION:
        result.update(
            riser_formula_m="0.005 + 0.1 * difficulty",
            acquisition_riser_range_m=[0.02, 0.06],
            scope="2–6cm bootstrap vertical-step acquisition, not the full 4–24cm curriculum or high-step qualification; tread width is axis-aligned, not every travel direction",
        )
    return result


def _levels(seed, variant):
    rng = np.random.default_rng(np.random.SeedSequence((seed, variant)))
    axis = np.arange(32) * CELL_SIZE + CELL_SIZE / 2 - 8
    x, y = np.meshgrid(axis, axis, indexing="ij")
    angle, phase1, phase2 = rng.uniform(-math.pi, math.pi, 3)
    u, v = (
        math.cos(angle) * x + math.sin(angle) * y,
        -math.sin(angle) * x + math.cos(angle) * y,
    )
    field = 0.9 * np.sin(2 * math.pi * u / 6 + phase1)
    field += 0.6 * np.cos(2 * math.pi * v / 5 + phase2)
    levels = np.clip(np.floor(1.5 + field), 0, 2).astype(np.int64)
    # Guard cells keep upper wall vertices outside the closed flat pad/collar.
    flat = (
        ((np.abs(x) <= 1.25) & (np.abs(y) <= 1.25))
        | (np.abs(x) >= 6.75)
        | (np.abs(y) >= 6.75)
    )
    near_flat = flat.copy()
    for axis in (0, 1):
        near_flat |= np.roll(flat, 1, axis) | np.roll(flat, -1, axis)
    levels = np.minimum(levels, np.where(flat, 0, np.where(near_flat, 1, 2)))
    # Checkerboard corners create a four-wall nonmanifold vertical edge. Move
    # only an extreme level toward1; this preserves the one-level neighbor bound
    # and terminates after at most the number of cells that are not already1.
    changed = True
    while changed:
        changed = False
        for i in range(31):
            for j in range(31):
                block = levels[i : i + 2, j : j + 2]
                if (
                    block[0, 0] == block[1, 1]
                    and block[0, 1] == block[1, 0]
                    and block[0, 0] != block[0, 1]
                ):
                    extreme = 0 if block.min() == 0 else 2
                    candidates = np.argwhere(
                        (block == extreme) & ~flat[i : i + 2, j : j + 2]
                    )
                    if not len(candidates):
                        raise ValueError(
                            "Cannot resolve step corner without changing the flat pad"
                        )
                    a, b = candidates[0]
                    block[a, b] = 1
                    changed = True
    if any(np.abs(np.diff(levels, axis=axis)).max() > 1 for axis in (0, 1)):
        raise ValueError("Generated adjacent steps exceed one riser")
    return levels


def _audit_mesh(vertices, faces):
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    if (
        not np.isfinite(vertices).all()
        or (np.linalg.norm(normals, axis=1) <= 1e-12).any()
        or (normals[:, 2] < 0).any()
    ):
        raise ValueError("Invalid step-field surface triangles")
    vertical = normals[:, 2] == 0
    support_area = float(normals[~vertical, 2].sum() / 2)
    if not np.isclose(support_area, math.prod(TILE_SIZE), rtol=0, atol=1e-9):
        raise ValueError("Step field lost projected support coverage")
    welded, inverse = np.unique(vertices, axis=0, return_inverse=True)
    ids = inverse[faces]
    edges = np.concatenate((ids[:, [0, 1]], ids[:, [1, 2]], ids[:, [2, 0]]))
    edges, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    endpoints = welded[edges]
    boundary = np.zeros(len(edges), dtype=bool)
    for axis in (0, 1):
        boundary |= (endpoints[:, :, axis] == 0).all(1) | (
            endpoints[:, :, axis] == TILE_SIZE[axis]
        ).all(1)
    if not np.array_equal(counts, np.where(boundary, 1, 2)):
        raise ValueError("Step field has a crack, T-junction or nonmanifold edge")
    return {
        "vertical_face_count": int(vertical.sum()),
        "projected_support_area_m2": support_area,
        "welded_vertex_count": len(welded),
        "welded_interior_edge_incidence": 2,
        "maximum_nonvertical_triangle_grade": float(
            (
                np.linalg.norm(normals[~vertical, :2], axis=1) / normals[~vertical, 2]
            ).max()
        ),
    }


def build_surface(difficulty, *, seed, variant, size=TILE_SIZE, version=VERSION):
    envelope(version)  # Validate before building or mutating any geometry.
    if (
        isinstance(difficulty, (bool, np.bool_))
        or not np.isscalar(difficulty)
        or not math.isfinite(float(difficulty))
        or not 0 <= difficulty <= 1
        or any(type(value) is not int or value < 0 for value in (seed, variant))
        or tuple(size) != TILE_SIZE
    ):
        raise ValueError(
            "Require finite difficulty[0,1], integer seed/variant and16m tile"
        )
    difficulty = float(difficulty)
    rise = 0.04 + 0.08 * difficulty if version == VERSION else 0.005 + 0.1 * difficulty
    levels = _levels(seed, variant)
    axes = [np.round(np.arange(161) * RESOLUTION, 12)] * 2
    x, y = np.meshgrid(axes[0] - 8, axes[1] - 8, indexing="ij")
    rough = multiscale_field(axes, seed + variant * 1000003)
    rough *= (0.004 * difficulty) / max(float(np.abs(rough).max()), 1e-12)
    # Exact supported spawn and outer collar; no full-width central band.
    taper = np.clip((np.maximum(np.abs(x), np.abs(y)) - 1) / 0.5, 0, 1)
    taper *= np.clip((7 - np.maximum(np.abs(x), np.abs(y))) / 0.5, 0, 1)
    rough *= taper * taper * (3 - 2 * taper)
    vertices, faces = [], []
    local = np.arange(36).reshape(6, 6)
    a, b, c, d = local[:-1, :-1], local[1:, :-1], local[:-1, 1:], local[1:, 1:]
    top_faces = np.stack(
        (np.stack((a, b, c), -1), np.stack((b, d, c), -1)), -2
    ).reshape(-1, 3)
    for i, j in np.ndindex(levels.shape):
        xx, yy = np.meshgrid(
            axes[0][i * 5 : i * 5 + 6], axes[1][j * 5 : j * 5 + 6], indexing="ij"
        )
        zz = levels[i, j] * rise + rough[i * 5 : i * 5 + 6, j * 5 : j * 5 + 6]
        offset = len(vertices)
        vertices.extend(np.stack((xx, yy, zz), -1).reshape(-1, 3))
        faces.extend(top_faces + offset)
    probe = None
    for axis in (0, 1):
        for i, j in np.argwhere(np.diff(levels, axis=axis) != 0):
            before = levels[i, j]
            after = levels[i + (axis == 0), j + (axis == 1)]
            low, high = min(before, after) * rise, max(before, after) * rise
            for sub in range(5):
                if axis == 0:
                    indices = ((i + 1) * 5, j * 5 + sub), ((i + 1) * 5, j * 5 + sub + 1)
                else:
                    indices = (i * 5 + sub, (j + 1) * 5), (i * 5 + sub + 1, (j + 1) * 5)
                first, second = indices
                points = np.array(
                    [
                        [axes[0][first[0]], axes[1][first[1]], low + rough[first]],
                        [axes[0][second[0]], axes[1][second[1]], low + rough[second]],
                        [axes[0][first[0]], axes[1][first[1]], high + rough[first]],
                        [axes[0][second[0]], axes[1][second[1]], high + rough[second]],
                    ]
                )
                indices = np.array([[0, 1, 2], [1, 3, 2]])
                normal = np.cross(points[1] - points[0], points[2] - points[0])
                # Exposed side normal points from the higher cell into lower air.
                if normal[axis] * (before - after) < 0:
                    indices = indices[:, ::-1]
                    normal *= -1
                if probe is None:
                    normal /= np.linalg.norm(normal)
                    hit = points[indices[0]].mean(axis=0)
                    probe = {
                        "ray_start_local_m": (hit + 0.1 * normal).tolist(),
                        "ray_direction": (-normal).tolist(),
                        "expected_hit_local_m": hit.tolist(),
                        "expected_distance_m": 0.1,
                        "expected_face_normal": normal.tolist(),
                    }
                faces.extend(indices + len(vertices))
                vertices.extend(points)
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int64)
    # Emit the same exact indexed topology that we audit, without tolerance-based
    # repair or relying on PhysX to weld independently emitted patch vertices.
    vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
    faces = inverse[faces]
    audit = _audit_mesh(vertices, faces)
    receipt = {
        "version": version,
        "profile": "step_hills",
        "seed": seed,
        "variant": variant,
        "difficulty": difficulty,
        "size_m": list(TILE_SIZE),
        "origin_m": [8.0, 8.0, 0.0],
        "riser_height_m": rise,
        "axis_aligned_cell_tread_m": CELL_SIZE,
        "roughness_absolute_bound_m": 0.004 * difficulty,
        "minimum_surface_height_m": float(vertices[:, 2].min()),
        "maximum_surface_height_m": float(vertices[:, 2].max()),
        "level_cell_counts": np.bincount(levels.ravel(), minlength=3).tolist(),
        "maximum_adjacent_level_difference": 1,
        "riser_probe": probe,
        "mesh_vertex_count": len(vertices),
        "mesh_face_count": len(faces),
        "vertices_float64_sha256": hashlib.sha256(
            vertices.astype("<f8").tobytes()
        ).hexdigest(),
        "faces_int64_sha256": hashlib.sha256(faces.astype("<i8").tobytes()).hexdigest(),
        **audit,
        "scope": "Generated triangle geometry only; no PhysX cooked-collider/contact or robot-capability claim",
    }
    return vertices, faces, receipt


def native_receipts():
    return deepcopy(_RECEIPTS)


def _native_key(difficulty, cfg):
    if _CONFIGURATION is None or (
        cfg.seed != _CONFIGURATION["seed"]
        or tuple(cfg.size) != TILE_SIZE
        or cfg.profile != "step_hills"
        or cfg.variant not in (12, 13, 14, 15)
        or cfg.function is not step_field_terrain
        or isinstance(difficulty, (bool, np.bool_))
        or not math.isfinite(float(difficulty))
        or not DIFFICULTY[0] <= difficulty < DIFFICULTY[1]
    ):
        raise ValueError("Native step field differs from configured acquisition recipe")
    return cfg.variant, min(
        2, int((difficulty - DIFFICULTY[0]) / (DIFFICULTY[1] - DIFFICULTY[0]) * 3)
    )


def step_field_terrain(difficulty, cfg):
    key = _native_key(difficulty, cfg)
    if any((item["variant"], item["row"]) == key for item in _RECEIPTS):
        raise ValueError("Duplicate native step-field variant/row construction")
    vertices, faces, receipt = build_surface(
        difficulty,
        seed=cfg.seed,
        variant=cfg.variant,
        size=cfg.size,
        version=_CONFIGURATION["version"],
    )
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not np.array_equal(mesh.vertices, vertices) or not np.array_equal(
        mesh.faces, faces
    ):
        raise ValueError("Native Trimesh changed step-field triangles")
    _RECEIPTS.append({**receipt, "row": key[1]})
    return [mesh], np.array(receipt["origin_m"])


def configure(cfg, version=VERSION):
    """Opt in four declared callbacks; preserve commands, resets and other columns."""
    global _CONFIGURATION
    envelope(version)
    generator = cfg.scene.terrain.terrain_generator
    terrains = list(generator.sub_terrains.values())
    profiles = tuple(
        profile
        for profile in ("plane", "rough_flat", "hills", "step_hills", "tilted_ramps")
        for _ in range(4)
    )
    if (
        type(generator.seed) is not int
        or generator.seed < 0
        or generator.num_rows != 3
        or generator.num_cols != 20
        or tuple(generator.size) != TILE_SIZE
        or tuple(generator.difficulty_range) != DIFFICULTY
        or generator.curriculum is not True
        or generator.use_cache is not False
        or len(terrains) != 20
        or tuple(item.profile for item in terrains) != profiles
        or any(
            item.variant != i or item.proportion != 0.05
            for i, item in enumerate(terrains)
        )
    ):
        raise ValueError(
            "Require unchanged20-column/3-row procedural acquisition layout"
        )
    for item in terrains[12:16]:
        # configclass.copy() retains declared dataclass fields only. The callback
        # identifies this opt-in; geometry versioning lives in protocol/receipts.
        item.function = step_field_terrain
    _CONFIGURATION = {"seed": generator.seed, "version": version}
    _RECEIPTS.clear()


def verify_native_receipts(receipts=None):
    """Rebuild native or serialized mesh receipts without simulator imports."""
    expected = {(variant, row) for variant in (12, 13, 14, 15) for row in range(3)}
    records = native_receipts() if receipts is None else deepcopy(receipts)
    try:
        seeds = {item["seed"] for item in records}
        versions = {item["version"] for item in records}
        valid = (
            type(records) is list
            and len(records) == 12
            and len(seeds) == 1
            and len(versions) == 1
            and {(item["variant"], item["row"]) for item in records} == expected
        )
        if receipts is None:
            valid = (
                valid
                and _CONFIGURATION is not None
                and seeds == {_CONFIGURATION["seed"]}
                and versions == {_CONFIGURATION["version"]}
            )
        if not valid:
            raise ValueError(
                "Require all12 native step-field variant/row receipts before learning"
            )
        version = records[0]["version"]
        envelope(version)
        for item in records:
            difficulty = item["difficulty"]
            if not DIFFICULTY[0] <= difficulty < DIFFICULTY[1]:
                raise ValueError("Step-field receipt difficulty is outside acquisition")
            row = min(
                2,
                int((difficulty - DIFFICULTY[0]) / (DIFFICULTY[1] - DIFFICULTY[0]) * 3),
            )
            _, _, built = build_surface(
                difficulty, seed=item["seed"], variant=item["variant"], version=version
            )
            if item != {**built, "row": row}:
                raise ValueError(
                    "Step-field receipt differs from its deterministic mesh"
                )
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed native step-field receipts") from error
    return {
        "version": version,
        "status": "NATIVE_GENERATED_TRIANGLES_VERIFIED_NOT_PHYSX_CONTACT",
        "seed": records[0]["seed"],
        "envelope": envelope(version),
        "tiles": records,
        "scope": envelope(version)["scope"],
    }


def _validate_imported_geometry(generated, imported):
    """Finite receipts/tolerances, not proof from an untrusted artifact producer."""
    if (
        imported["status"] != "IMPORTED_USD_WARP_RAYS_VERIFIED_NOT_PHYSX_CONTACT"
        or imported["collision_enabled"] is not True
        or imported["collision_approximation"] not in ("none", "none (USD default)")
        or not isinstance(imported["mesh_prim_path"], str)
        or not imported["mesh_prim_path"].startswith("/")
        or len(imported["measurements"]) != 12
        or any(
            type(imported[key]) is not str
            or len(imported[key]) != 64
            or any(c not in "0123456789abcdef" for c in imported[key])
            for key in ("usd_points_float32_sha256", "usd_faces_int32_sha256")
        )
    ):
        raise ValueError("Invalid imported step-field geometry receipt")
    measurements = {
        (item["variant"], item["row"]): item for item in imported["measurements"]
    }
    if len(measurements) != 12:
        raise ValueError("Duplicate imported tile probes")
    for tile in generated["tiles"]:
        item = measurements[(tile["variant"], tile["row"])]
        origin = np.array([16 * (tile["row"] - 1), 16 * (tile["variant"] - 9.5), 0.0])
        probe = tile["riser_probe"]
        expected = {
            "terrain_origin_world_m": origin,
            "riser_hit_world_m": origin
            + np.asarray(probe["expected_hit_local_m"])
            - [8, 8, 0],
            "riser_distance_m": probe["expected_distance_m"],
            "riser_normal": probe["expected_face_normal"],
            "pad_hit_world_m": origin,
            "pad_distance_m": 1.0,
            "pad_normal": [0.0, 0.0, 1.0],
        }
        for key, target in expected.items():
            value, target = np.asarray(item[key]), np.asarray(target)
            if (
                value.shape != target.shape
                or not np.isfinite(value).all()
                or not np.allclose(value, target, rtol=0, atol=2e-4)
            ):
                raise ValueError(
                    f"Imported step-field {key} differs from generated geometry"
                )
        if any(
            type(item[key]) is not int or item[key] < 0
            for key in ("riser_face_id", "pad_face_id")
        ):
            raise ValueError("Imported step-field ray missed its triangle")


def validate_geometry_report(report, *, seed):
    """Pure serialized admission: generated mesh hashes plus imported probe receipts."""
    try:
        generated = verify_native_receipts(report["tiles"])
        if (
            generated["seed"] != seed
            or {key: report[key] for key in generated} != generated
        ):
            raise ValueError("Step-field geometry report differs from its recipe/seed")
        _validate_imported_geometry(generated, report["imported_geometry"])
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed imported step-field geometry report") from error
    return deepcopy(report)


def verify_native_geometry(env):
    """Before optimization, check the imported USD/Warp mesh and collision settings.

    Warp reads USD triangles, not PhysX's cooked collision representation. These
    probes prevent silently training on a bevel/plane at the declared risers;
    they are not a contact-dynamics or locomotion certification.
    """
    import torch
    from pxr import UsdGeom, UsdPhysics
    from isaaclab import sim as sim_utils
    from isaaclab.utils.warp import raycast_mesh

    generated = verify_native_receipts()
    sensor = env.scene["base_height_scanner"]
    paths = sensor.cfg.mesh_prim_paths
    if len(paths) != 1 or paths[0] not in sensor.meshes:
        raise ValueError("Require the actual initialized terrain RayCaster mesh")
    prim = sim_utils.get_first_matching_child_prim(
        paths[0], lambda item: item.GetTypeName() == "Mesh"
    )
    if prim is None or not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError("Imported terrain lacks a valid enabled mesh collider")
    enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    approximation = prim.GetAttribute("physics:approximation").Get()
    if enabled is not True or approximation not in (None, "none"):
        raise ValueError("Step fields require enabled triangle collision, not a hull")
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype="<f4")
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype="<i4")
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or (np.asarray(mesh.GetFaceVertexCountsAttr().Get()) != 3).any()
    ):
        raise ValueError("Imported terrain must retain finite triangular USD geometry")
    starts, directions, origins = [], [], []
    for tile in generated["tiles"]:
        origin = (
            env.scene.terrain.terrain_origins[tile["row"], tile["variant"]]
            .detach()
            .cpu()
            .numpy()
        )
        probe = tile["riser_probe"]
        starts.extend(
            (
                origin + np.asarray(probe["ray_start_local_m"]) - [8, 8, 0],
                origin + [0, 0, 1],
            )
        )
        directions.extend((probe["ray_direction"], [0, 0, -1]))
        origins.append(origin.tolist())
    # Isaac Lab 2.3.2 restores distance/face IDs as (batch, rays), so keep an
    # explicit batch even though these probes all target one terrain mesh.
    hits, distances, normals, ids = raycast_mesh(
        torch.tensor(
            np.asarray(starts), dtype=torch.float32, device=env.device
        ).unsqueeze(0),
        torch.tensor(
            np.asarray(directions), dtype=torch.float32, device=env.device
        ).unsqueeze(0),
        sensor.meshes[paths[0]],
        max_dist=2.0,
        return_distance=True,
        return_normal=True,
        return_face_id=True,
    )
    hits, distances, normals, ids = (
        value.squeeze(0) for value in (hits, distances, normals, ids)
    )
    # Compare directions, without depending on Warp's face-normal magnitude.
    normals = normals / torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    hits, distances, normals, ids = (
        value.detach().cpu().tolist() for value in (hits, distances, normals, ids)
    )
    imported = {
        "status": "IMPORTED_USD_WARP_RAYS_VERIFIED_NOT_PHYSX_CONTACT",
        "mesh_prim_path": str(prim.GetPath()),
        "collision_enabled": enabled,
        "collision_approximation": (
            "none (USD default)" if approximation is None else approximation
        ),
        "usd_points_float32_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
        "usd_faces_int32_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
        "measurements": [
            {
                "variant": tile["variant"],
                "row": tile["row"],
                "terrain_origin_world_m": origins[index],
                "riser_hit_world_m": hits[2 * index],
                "riser_distance_m": distances[2 * index],
                "riser_normal": normals[2 * index],
                "riser_face_id": ids[2 * index],
                "pad_hit_world_m": hits[2 * index + 1],
                "pad_distance_m": distances[2 * index + 1],
                "pad_normal": normals[2 * index + 1],
                "pad_face_id": ids[2 * index + 1],
            }
            for index, tile in enumerate(generated["tiles"])
        ],
        "scope": "USD collision settings and actual USD-derived Warp mesh rays; NOT PhysX cooked-collider equivalence or contact/behavior certification",
    }
    _validate_imported_geometry(generated, imported)
    return {**generated, "imported_geometry": imported}
