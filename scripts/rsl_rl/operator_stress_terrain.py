"""Evaluation-only, fully supported synthetic bumpy ground.

This is a separate stress surface, not a change to the checkpoint's training
generator. Independent two-dimensional uniform random lattices are interpolated
and combined at four finite scales. It is neither sine-only terrain nor an
infinitely fractal surface, and does not model holes, loose rocks or vertical
stairs. The builder requires only NumPy; simulator imports are lazy.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from numbers import Real

import numpy as np

VERSION = "operator_stress_surface_v1"
PROFILE = "rough_stress"
TILE_SIZE = (16.0, 16.0)
RESOLUTION = 0.05
LATTICE_SPACINGS = (2.0, 1.0, 0.5, 0.25)
FULL_HEIGHT_BOUND = 0.08
FULL_GRADE_BOUND = 0.8
SPAWN_HALF_WIDTH = 1.0
FLAT_BAND_HALF_WIDTH = 0.6
BORDER_WIDTH = 1.0
TRANSITION_WIDTH = 1.0
_STRESS_TERRAIN_RECEIPT = None


def clear_stress_terrain_receipt():
    """Start a fresh one-process, one-environment native geometry attestation."""
    global _STRESS_TERRAIN_RECEIPT
    _STRESS_TERRAIN_RECEIPT = None


def stress_terrain_receipt():
    """Return an isolated snapshot, or None until a checked native mesh exists."""
    return deepcopy(_STRESS_TERRAIN_RECEIPT)


def stress_terrain_envelope():
    """JSON-safe declared geometry, not a robot capability or acceptance claim."""
    return {
        "version": VERSION,
        "profile": PROFILE,
        "evaluation_only": True,
        "tile_size_m": list(TILE_SIZE),
        "resolution_m": RESOLUTION,
        "lattice_spacings_m": list(LATTICE_SPACINGS),
        "octave_weights": [2.0 ** (-0.7 * octave) for octave in range(4)],
        "roughness": (
            "independent 2-D uniform random lattices with bilinear interpolation; "
            "four finite scales, not infinite fractal"
        ),
        "absolute_height_bound_at_full_difficulty_m": FULL_HEIGHT_BOUND,
        "maximum_triangle_grade_at_full_difficulty": FULL_GRADE_BOUND,
        "spawn_flat_half_width_m": SPAWN_HALF_WIDTH,
        "flat_band_half_width_m": FLAT_BAND_HALF_WIDTH,
        "flat_border_width_m": BORDER_WIDTH,
        "transition_width_m": TRANSITION_WIDTH,
        "support": "continuous triangulated heightfield; no holes",
        "scope": "synthetic bumpy ground; not loose rocks or vertical stairs",
        "height_hash_encoding": "C-order little-endian float64 height bytes, SHA-256",
    }


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _maximum_triangle_grade(heights):
    """Match both triangle orientations in the existing surface mesh adapter."""
    z00, z10 = heights[:-1, :-1], heights[1:, :-1]
    z01, z11 = heights[:-1, 1:], heights[1:, 1:]
    first = np.hypot(z10 - z00, z01 - z00) / RESOLUTION
    second = np.hypot(z11 - z01, z11 - z10) / RESOLUTION
    return float(max(first.max(), second.max()))


def multiscale_field(axes, seed):
    """Independent 2-D lattices; shared by bounded evaluation-only surfaces."""
    rng = np.random.default_rng(seed)
    field = np.zeros(tuple(len(axis) for axis in axes), dtype=np.float64)
    for octave, spacing in enumerate(LATTICE_SPACINGS):
        coarse_axes = [
            np.linspace(0.0, side, round(side / spacing) + 1) for side in TILE_SIZE
        ]
        lattice = rng.uniform(-1.0, 1.0, tuple(len(axis) for axis in coarse_axes))
        along_x = np.column_stack(
            [
                np.interp(axes[0], coarse_axes[0], lattice[:, column])
                for column in range(lattice.shape[1])
            ]
        )
        bilinear = np.stack(
            [np.interp(axes[1], coarse_axes[1], row) for row in along_x]
        )
        field += 2.0 ** (-0.7 * octave) * bilinear
    return field


def _nominal_route_geometry(heights, centered_axes):
    """Describe the demo centerline, without asserting any robot reached it."""
    x, y = centered_axes
    center = int(np.argmin(np.abs(x)))
    route_mask = np.abs(y) <= 4.2
    route = heights[center, route_mask]
    adjacent_cells = heights[center - 1 : center + 2, route_mask]
    return {
        "scope": "geometry only; not measured under-foot support or traversal",
        "centerline_x_m": 0.0,
        "centerline_y_range_m": [-4.2, 4.2],
        "samples": int(route.size),
        "minimum_surface_height_m": float(route.min()),
        "maximum_surface_height_m": float(route.max()),
        "peak_to_peak_surface_height_m": float(np.ptp(route)),
        "root_mean_square_height_m": float(np.sqrt(np.mean(route**2))),
        "maximum_longitudinal_grid_grade": float(
            np.abs(np.diff(route)).max() / RESOLUTION
        ),
        "adjacent_triangle_corridor_half_width_m": RESOLUTION,
        "maximum_adjacent_triangle_grade": _maximum_triangle_grade(adjacent_cells),
    }


def build_stress_surface(difficulty, *, seed, size=TILE_SIZE):
    """Build deterministic 16 m square geometry, independent of global RNG state.

    Difficulty linearly scales the same seeded field after the flat-region
    tapers. Both the absolute height bound and the actual triangulated surface
    grade bound apply to the entire surface, including those tapers.
    """
    if (
        isinstance(difficulty, (bool, np.bool_))
        or not isinstance(difficulty, Real)
        or not math.isfinite(float(difficulty))
        or not 0.0 <= float(difficulty) <= 1.0
    ):
        raise ValueError("Difficulty must be a finite real number in [0, 1]")
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    if (
        not isinstance(size, (tuple, list, np.ndarray))
        or len(size) != 2
        or any(
            isinstance(side, (bool, np.bool_))
            or not isinstance(side, Real)
            or not math.isfinite(float(side))
            or float(side) != expected
            for side, expected in zip(size, TILE_SIZE)
        )
    ):
        raise ValueError("Stress terrain requires exactly a 16 m by 16 m tile")

    difficulty = float(difficulty)
    axes = [
        np.round(np.arange(round(side / RESOLUTION) + 1) * RESOLUTION, 12)
        for side in TILE_SIZE
    ]
    centered_axes = [np.round(axis - side / 2, 12) for axis, side in zip(axes, size)]
    x, y = np.meshgrid(*centered_axes, indexing="ij")
    field = multiscale_field(axes, seed)

    edge_distance = np.minimum(TILE_SIZE[0] / 2 - np.abs(x), 8.0 - np.abs(y))
    envelope = _smoothstep((edge_distance - BORDER_WIDTH) / TRANSITION_WIDTH)
    envelope *= _smoothstep(
        (np.maximum(np.abs(x), np.abs(y)) - SPAWN_HALF_WIDTH) / TRANSITION_WIDTH
    )
    envelope *= _smoothstep((np.abs(y) - FLAT_BAND_HALF_WIDTH) / TRANSITION_WIDTH)
    field *= envelope
    scale = min(
        FULL_HEIGHT_BOUND / max(float(np.abs(field).max()), 1e-12),
        FULL_GRADE_BOUND / max(_maximum_triangle_grade(field), 1e-12),
    )
    heights = np.asarray(field * (scale * difficulty), dtype=np.float64)
    # Canonicalize signed zeros before computing portable geometry identity.
    heights[heights == 0.0] = 0.0
    supported = np.ones((len(axes[0]) - 1, len(axes[1]) - 1), dtype=bool)
    metadata = {
        **stress_terrain_envelope(),
        "seed": seed,
        "difficulty": difficulty,
        "size_m": list(TILE_SIZE),
        "absolute_height_bound_m": FULL_HEIGHT_BOUND * difficulty,
        "maximum_triangle_grade_bound": FULL_GRADE_BOUND * difficulty,
        "minimum_surface_height_m": float(heights.min()),
        "maximum_surface_height_m": float(heights.max()),
        "maximum_absolute_surface_height_m": float(np.abs(heights).max()),
        "maximum_surface_grade": _maximum_triangle_grade(heights),
        "maximum_grid_edge_height_change_m": float(
            max(np.abs(np.diff(heights, axis=axis)).max() for axis in (0, 1))
        ),
        "surface_height_standard_deviation_m": float(heights.std()),
        "root_mean_square_height_m": float(np.sqrt(np.mean(heights**2))),
        "gap_area_fraction": 0.0,
        "gap_rectangles": [],
        "height_float64_sha256": hashlib.sha256(
            heights.astype("<f8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "nominal_route_geometry": _nominal_route_geometry(heights, centered_axes),
    }
    return {
        "x": axes[0],
        "y": axes[1],
        "heights": heights,
        "supported_cells": supported,
        "origin": np.array([8.0, 8.0, 0.0]),
        "metadata": metadata,
    }


def stress_terrain(difficulty, cfg):
    """Native callback; require exact CPU-preflight geometry before mesh creation."""
    global _STRESS_TERRAIN_RECEIPT
    surface = build_stress_surface(difficulty, seed=cfg.seed, size=cfg.size)
    expected = getattr(cfg, "expected_height_sha256", None)
    actual = surface["metadata"]["height_float64_sha256"]
    if expected != actual:
        raise ValueError("Stress terrain differs from the required CPU-preflight hash")

    import trimesh

    from parkour_lab.tasks.manager_based.parkour_lab.mdp.terrain.operator_terrain import (
        surface_mesh_arrays,
    )

    vertices, faces = surface_mesh_arrays(surface)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # Attest the constructed mesh, not just the preflight configuration. Keep
    # one bounded receipt; the caller rejects multiple native callback calls.
    _STRESS_TERRAIN_RECEIPT = {
        "calls": (
            1
            if _STRESS_TERRAIN_RECEIPT is None
            else _STRESS_TERRAIN_RECEIPT["calls"] + 1
        ),
        "seed": cfg.seed,
        "difficulty": float(difficulty),
        "height_float64_sha256": actual,
        "mesh_vertex_count": int(len(mesh.vertices)),
        "mesh_face_count": int(len(mesh.faces)),
        "mesh_vertices_float64_sha256": hashlib.sha256(
            np.asarray(mesh.vertices, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        "mesh_faces_int64_sha256": hashlib.sha256(
            np.asarray(mesh.faces, dtype="<i8").tobytes(order="C")
        ).hexdigest(),
    }
    return [mesh], surface["origin"]
