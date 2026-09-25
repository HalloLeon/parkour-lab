"""Four fixed, full-width rough obstacles; evaluation geometry, not a curriculum.

Step lips use duplicate X coordinates with different heights: their triangles
are vertical collision faces, not heightfield ramps. All surfaces are supported
and connected, with no gaps or flat lateral bypass. Simulator imports are lazy.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math

import numpy as np

from .operator_stress_terrain import RESOLUTION, TILE_SIZE, multiscale_field

VERSION = "operator_traversal_geometry_v1"
PROFILES = ("ramp_6deg", "ramp_12deg", "step_08m", "step_16m")
ROUGHNESS_BOUND = 0.008
_RECEIPTS = []


def geometry(profile):
    """Obstacle coordinates relative to the native spawn origin, in metres."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown traversal profile: {profile}")
    ramp = profile.startswith("ramp")
    angle = (6.0 if profile == "ramp_6deg" else 12.0) if ramp else None
    height = (
        1.2 * math.tan(math.radians(angle))
        if ramp
        else (0.08 if profile == "step_08m" else 0.16)
    )
    return {
        "profile": profile,
        "entry_x_m": 1.0,
        "top_start_x_m": 2.2 if ramp else 1.0,
        "top_end_x_m": 3.2 if ramp else 2.6,
        "exit_x_m": 4.4 if ramp else 2.6,
        "nominal_height_m": height,
        "nominal_incline_degrees": angle,
        "vertical_risers": not ramp,
        "full_width_m": TILE_SIZE[1],
    }


def build_surface(profile, seed):
    if type(seed) is not int or seed < 0:
        raise ValueError("Traversal seed must be a nonnegative integer")
    spec = geometry(profile)
    center = np.array(TILE_SIZE) / 2
    axes = [
        np.round(np.arange(round(side / RESOLUTION) + 1) * RESOLUTION, 12)
        for side in TILE_SIZE
    ]
    entry, exit_x = spec["entry_x_m"], spec["exit_x_m"]
    if spec["vertical_risers"]:
        # Insert a second row at each lip. The row before the up lip and after
        # the down lip is ground; the rows between them form the supported top.
        axes[0] = np.sort(np.append(axes[0], [center[0] + entry, center[0] + exit_x]))
    x = np.round(axes[0] - center[0], 12)
    if spec["vertical_risers"]:
        start = np.flatnonzero(x == entry)[-1]
        end = np.flatnonzero(x == exit_x)[0]
        macro = np.zeros_like(x)
        macro[start : end + 1] = spec["nominal_height_m"]
    else:
        macro = spec["nominal_height_m"] * np.minimum(
            np.clip((x - entry) / (spec["top_start_x_m"] - entry), 0, 1),
            np.clip((exit_x - x) / (exit_x - spec["top_end_x_m"]), 0, 1),
        )
    seams = sorted(
        set(
            spec[name]
            for name in ("entry_x_m", "top_start_x_m", "top_end_x_m", "exit_x_m")
        )
    )
    distance = np.min(np.abs(x[:, None] - seams), axis=1)
    taper = np.clip(distance / 0.2, 0, 1)
    taper = taper * taper * (3 - 2 * taper)
    taper *= (x > spec["entry_x_m"]) & (x < spec["exit_x_m"])
    field = multiscale_field(axes, seed)
    roughness = field * (ROUGHNESS_BOUND / max(np.abs(field).max(), 1e-12))
    heights = macro[:, None] + roughness * taper[:, None]
    heights[heights == 0] = 0.0
    xx, yy = np.meshgrid(*axes, indexing="ij")
    vertices = np.stack((xx, yy, heights), axis=-1).reshape(-1, 3)
    ids = np.arange(heights.size).reshape(heights.shape)
    a, b, c, d = ids[:-1, :-1], ids[1:, :-1], ids[:-1, 1:], ids[1:, 1:]
    faces = np.stack((np.stack((a, b, c), -1), np.stack((b, d, c), -1)), -2).reshape(
        -1, 3
    )
    triangles = vertices[faces]
    normal = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    horizontal = np.abs(normal[:, 2]) > 1e-12
    metadata = {
        "version": VERSION,
        **spec,
        "seed": seed,
        "origin_m": [float(center[0]), float(center[1]), 0.0],
        "roughness_absolute_bound_m": ROUGHNESS_BOUND,
        "roughness": "Four independent bilinear 2-D lattice octaves (2, 1, 0.5, 0.25 m); zero at macro seams",
        "minimum_height_m": float(heights.min()),
        "maximum_height_m": float(heights.max()),
        "maximum_nonvertical_triangle_grade": float(
            (
                np.linalg.norm(normal[horizontal, :2], axis=1)
                / np.abs(normal[horizontal, 2])
            ).max()
        ),
        "vertical_face_count": int((~horizontal).sum()),
        "mesh_vertex_count": len(vertices),
        "mesh_face_count": len(faces),
        "vertices_float64_sha256": hashlib.sha256(
            vertices.astype("<f8").tobytes()
        ).hexdigest(),
        "faces_int64_sha256": hashlib.sha256(faces.astype("<i8").tobytes()).hexdigest(),
        "scope": "Connected supported surface; no holes, no lateral flat band. One layout per profile, not independent layouts per robot.",
    }
    return vertices, faces, metadata


def preflight(seed):
    return [build_surface(profile, seed)[2] for profile in PROFILES]


def native_receipts():
    return deepcopy(_RECEIPTS)


def traversal_terrain(difficulty, cfg):
    if (
        isinstance(difficulty, (bool, np.bool_))
        or difficulty != 1.0
        or tuple(cfg.size) != TILE_SIZE
    ):
        raise ValueError("Traversal fixtures require their fixed SI geometry")
    if any(item["profile"] == cfg.profile for item in _RECEIPTS):
        raise ValueError("Unexpected duplicate traversal mesh construction")
    vertices, faces, receipt = build_surface(cfg.profile, cfg.seed)
    if receipt != cfg.expected_geometry:
        raise ValueError("Native traversal geometry differs from CPU preflight")
    import trimesh

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    if not np.array_equal(mesh.vertices, vertices) or not np.array_equal(
        mesh.faces, faces
    ):
        raise ValueError("Native mesh construction changed traversal geometry")
    _RECEIPTS.append(receipt)
    return [mesh], np.array(receipt["origin_m"])


def configure(cfg, seed, expected):
    """Install the fixed fixture and external commands; preserve sensors/motors."""
    from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
    from isaaclab.utils import configclass
    from .operator_runtime import configure_external_command

    @configclass
    class TraversalCfg(SubTerrainBaseCfg):
        function = traversal_terrain
        profile: str = ""
        expected_geometry: dict = {}

    if expected != preflight(seed):
        raise ValueError("Traversal preflight identity changed")
    _RECEIPTS.clear()
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=seed,
        size=TILE_SIZE,
        num_rows=1,
        num_cols=4,
        curriculum=True,
        difficulty_range=(1.0, 1.0),
        horizontal_scale=RESOLUTION,
        border_width=2.0,
        border_height=1.0,
        use_cache=False,
        sub_terrains={
            item["profile"]: TraversalCfg(
                proportion=0.25,
                profile=item["profile"],
                expected_geometry=item,
            )
            for item in expected
        },
    )
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.events.reset_base.params["pose_range"] = {
        name: (0.0, 0.0) for name in ("x", "y", "yaw")
    }
    cfg.terminations.procedural_physical_failure.params["minimum_surface_z_m"] = (
        -ROUGHNESS_BOUND
    )
    # The archived training sampler requires its 20-column profile layout.
    # This four-column evaluation instead uses PilotEnvironment's fixed tape.
    configure_external_command(cfg.commands.base_velocity)
