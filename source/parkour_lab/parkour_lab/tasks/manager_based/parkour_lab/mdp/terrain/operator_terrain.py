# Copyright (c) 2026, Leon Yi Bai
# SPDX-License-Identifier: BSD-3-Clause
"""Seeded, command-agnostic surfaces for provisional operator-terrain acquisition.

The pure NumPy builder is shared by CPU geometry checks and the lazy Isaac Lab
adapter. Heights are metres, grades are rise/run, and holes contain no floor.
This is a development envelope, not a calibrated robot capability or acceptance
gate. Native curriculum layout fixes column identity; it does not promote robots.
"""

from __future__ import annotations

import math

import numpy as np

TILE_SIZE = (16.0, 16.0)
RESOLUTION = 0.1
SPAWN_HALF_WIDTH = 1.0
BORDER_WIDTH = 1.0
FLAT_BAND_HALF_WIDTH = 0.6
TRANSITION_WIDTH = 1.0
GAP_START_DIFFICULTY = 0.65
GAP_WALL_BOTTOM = -2.0
PROFILE_BY_COLUMN = (
    *("plane",) * 4,
    *("rough_flat",) * 4,
    *("hills",) * 4,
    *("step_hills",) * 4,
    *("tilted_ramps",) * 3,
    "gaps",
)
# Full-difficulty surface-height magnitude and maximum triangulated surface grade.
ENVELOPES = {
    "plane": (0.0, 0.0),
    "rough_flat": (0.04, 0.35),
    "hills": (0.32, 0.5),
    "step_hills": (0.24, 0.9),
    "tilted_ramps": (0.28, 0.5),
    "gaps": (0.04, 0.35),
}


def terrain_envelope():
    """Explicit geometry bounds; none are learned-policy success thresholds."""
    return {
        "version": "operator_procedural_surface_v1",
        "status": "provisional_development_geometry_not_robot_acceptance",
        "tile_size_m": list(TILE_SIZE),
        "resolution_m": RESOLUTION,
        "profile_by_column": list(PROFILE_BY_COLUMN),
        "exact_plane_column_fraction": 0.2,
        "spawn_flat_half_width_m": SPAWN_HALF_WIDTH,
        "flat_band_half_width_m": FLAT_BAND_HALF_WIDTH,
        "flat_border_width_m": BORDER_WIDTH,
        "transition_width_m": TRANSITION_WIDTH,
        "height_and_grade_bounds_at_full_difficulty": {
            name: {"absolute_height_m": height, "maximum_surface_grade": grade}
            for name, (height, grade) in ENVELOPES.items()
        },
        "roughness_wavelengths_m": [3.2, 1.6, 0.8, 0.4],
        "roughness": "four seeded directional octaves, H=0.7; band-limited, not infinite fractal",
        "gaps": {
            "start_difficulty": GAP_START_DIFFICULTY,
            "maximum_count_per_gap_tile": 2,
            "maximum_narrow_width_m": 0.2,
            "maximum_length_m": 1.2,
            "maximum_tile_area_fraction": 0.01,
            "side_wall_bottom_m": GAP_WALL_BOTTOM,
            "floor": False,
        },
        "stepped_hills": (
            "rough terraced height field with slanted 0.1 m mesh risers; "
            "not vertical stairs or high-step qualification"
        ),
        "commands_rewards_waypoints": False,
    }


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _maximum_grade(heights, spacing):
    """Maximum slope of either actual triangle in every grid cell."""
    z00, z10 = heights[:-1, :-1], heights[1:, :-1]
    z01, z11 = heights[:-1, 1:], heights[1:, 1:]
    first = np.hypot(z10 - z00, z01 - z00) / spacing
    second = np.hypot(z11 - z01, z11 - z10) / spacing
    return float(max(first.max(), second.max()))


def build_operator_surface(difficulty, profile, *, seed=0, variant=0, size=TILE_SIZE):
    """Build a supported height grid and sparse cell holes without simulator imports.

    Array axes are X,Y. The origin is the centre of an exact supported flat pad.
    Every non-hole surface, including spawn and seam tapers, obeys its reported
    grade bound. Difficulty scales the *same* seeded field linearly; it does not
    silently alter a noise amplitude independent of the curriculum.
    """
    if (
        isinstance(difficulty, (bool, np.bool_))
        or not np.isscalar(difficulty)
        or not math.isfinite(float(difficulty))
        or not 0.0 <= float(difficulty) <= 1.0
    ):
        raise ValueError("Difficulty must be finite and in [0, 1]")
    if profile not in ENVELOPES:
        raise ValueError(f"Unknown operator terrain profile: {profile}")
    if any(type(value) is not int or value < 0 for value in (seed, variant)):
        raise ValueError("Seed and variant must be nonnegative integers")
    if len(size) != 2 or any(
        not math.isfinite(float(value))
        or value < 12.0
        or not math.isclose(value / RESOLUTION, round(value / RESOLUTION))
        for value in size
    ):
        raise ValueError("Tile sides must be finite, at least 12 m and 0.1 m aligned")
    difficulty = float(difficulty)
    rng = np.random.default_rng(np.random.SeedSequence((seed, variant)))
    axes = [np.linspace(0.0, side, round(side / RESOLUTION) + 1) for side in size]
    x, y = np.meshgrid(axes[0] - size[0] / 2, axes[1] - size[1] / 2, indexing="ij")
    rough = np.zeros_like(x)
    for octave, wavelength in enumerate((3.2, 1.6, 0.8, 0.4)):
        angle, phase = rng.uniform(-math.pi, math.pi, 2)
        coordinate = math.cos(angle) * x + math.sin(angle) * y
        rough += 2.0 ** (-0.7 * octave) * np.sin(
            coordinate * (2 * math.pi / wavelength) + phase
        )
    rough /= np.max(np.abs(rough))
    angles = rng.uniform(-math.pi, math.pi, 3)
    u = math.cos(angles[0]) * x + math.sin(angles[0]) * y
    v = -math.sin(angles[0]) * x + math.cos(angles[0]) * y
    hill = 0.6 * np.sin(2 * math.pi * u / 6 + angles[1]) + 0.4 * np.cos(
        2 * math.pi * v / 5 + angles[2]
    )
    if profile == "plane":
        field = np.zeros_like(x)
        rough_weight = 0.0
    elif profile in ("rough_flat", "gaps"):
        field = rough.copy()
        rough_weight = 1.0
    elif profile == "hills":
        field = hill + 0.06 * rough
        rough_weight = 0.06
    elif profile == "step_hills":
        field = np.floor(6 * hill) / 6 + 0.04 * rough
        rough_weight = 0.04
    else:
        # Triangular waves make alternating, tilted planar ramps, not goal-facing
        # slopes. Their intersections and flat transitions remain grade bounded.
        field = (
            0.75 * (1 - 4 * np.abs((u / 5 + angles[1]) % 1 - 0.5))
            + 0.25 * (1 - 4 * np.abs((v / 6 + angles[2]) % 1 - 0.5))
            + 0.04 * rough
        )
        rough_weight = 0.04
    edge_distance = np.minimum(size[0] / 2 - np.abs(x), size[1] / 2 - np.abs(y))
    envelope = _smoothstep((edge_distance - BORDER_WIDTH) / TRANSITION_WIDTH)
    envelope *= _smoothstep(
        (np.maximum(np.abs(x), np.abs(y)) - SPAWN_HALF_WIDTH) / TRANSITION_WIDTH
    )
    envelope *= _smoothstep((np.abs(y) - FLAT_BAND_HALF_WIDTH) / TRANSITION_WIDTH)
    field *= envelope
    height_bound, grade_bound = ENVELOPES[profile]
    scale = min(
        height_bound / max(float(np.max(np.abs(field))), 1e-12),
        grade_bound / max(_maximum_grade(field, RESOLUTION), 1e-12),
    )
    heights = field * (difficulty * scale)
    supported = np.ones((len(axes[0]) - 1, len(axes[1]) - 1), dtype=bool)
    rectangles = []
    if profile == "gaps" and difficulty >= GAP_START_DIFFICULTY:
        # Local finite gaps off the continuous central flat band. Their positions
        # are seeded, do not surround the spawn and do not compel any route.
        width_cells = 1 if difficulty < 0.85 else 2
        for sign in (-1, 1):
            centre = np.array(
                [size[0] / 2 + rng.uniform(-2.5, 2.5), size[1] / 2 + sign * 3.0]
            )
            counts = (width_cells, int(rng.integers(8, 13)))
            if rng.integers(2):
                counts = counts[::-1]
            start = np.rint(centre / RESOLUTION - np.asarray(counts) / 2).astype(int)
            end = start + counts
            supported[start[0] : end[0], start[1] : end[1]] = False
            rectangles.append(
                {
                    "lower_xy_m": (start * RESOLUTION).tolist(),
                    "upper_xy_m": (end * RESOLUTION).tolist(),
                }
            )
    if 1.0 - supported.mean() > 0.01:
        raise ValueError("Generated gap area exceeds the declared envelope")
    return {
        "x": axes[0],
        "y": axes[1],
        "heights": heights,
        "supported_cells": supported,
        "origin": np.array([size[0] / 2, size[1] / 2, 0.0]),
        "metadata": {
            "profile": profile,
            "seed": seed,
            "variant": variant,
            "difficulty": difficulty,
            "size_m": list(size),
            "maximum_absolute_surface_height_m": float(np.max(np.abs(heights))),
            "maximum_surface_grade": _maximum_grade(heights, RESOLUTION),
            "maximum_grid_edge_height_change_m": float(
                max(np.abs(np.diff(heights, axis=axis)).max() for axis in (0, 1))
            ),
            "surface_height_standard_deviation_m": float(heights.std()),
            "roughness_height_amplitude_m": float(
                np.abs(envelope * rough * (rough_weight * scale * difficulty)).max()
            ),
            "gap_area_fraction": float(1.0 - supported.mean()),
            "gap_rectangles": rectangles,
            "scope": "provisional geometry only; no policy acceptance",
        },
    }


def surface_mesh_arrays(surface):
    """Triangulate support and add vertical gap sides, without bridging or floors."""
    x, y = np.meshgrid(surface["x"], surface["y"], indexing="ij")
    heights, supported = surface["heights"], surface["supported_cells"]
    vertices = np.column_stack((x.ravel(), y.ravel(), heights.ravel()))
    index = np.arange(heights.size).reshape(heights.shape)
    a, b, c, d = index[:-1, :-1], index[1:, :-1], index[:-1, 1:], index[1:, 1:]
    faces = np.concatenate(
        (
            np.stack((a, b, c), axis=-1)[supported],
            np.stack((b, d, c), axis=-1)[supported],
        )
    )
    # For each unsupported cell, expose only its edges adjoining support. The
    # ordering points each wall normal into the empty hole, with no bottom face.
    wall_vertices, wall_faces = [], []
    for i, j in np.argwhere(~supported):
        for di, dj, first, second in (
            (-1, 0, index[i, j], index[i, j + 1]),
            (1, 0, index[i + 1, j + 1], index[i + 1, j]),
            (0, -1, index[i + 1, j], index[i, j]),
            (0, 1, index[i, j + 1], index[i + 1, j + 1]),
        ):
            if supported[i + di, j + dj]:
                top = vertices[[first, second]]
                lower = top.copy()
                lower[:, 2] = GAP_WALL_BOTTOM
                offset = len(vertices) + len(wall_vertices)
                wall_vertices.extend((top[0], top[1], lower[0], lower[1]))
                wall_faces.extend(
                    (
                        (offset, offset + 2, offset + 1),
                        (offset + 1, offset + 2, offset + 3),
                    )
                )
    if wall_vertices:
        vertices = np.concatenate((vertices, np.asarray(wall_vertices)))
        faces = np.concatenate((faces, np.asarray(wall_faces, dtype=np.int64)))
    return vertices, faces


def operator_terrain(difficulty, cfg):
    """Isaac Lab v2.3.2 SubTerrainBaseCfg callback; imports Trimesh only here."""
    import trimesh

    surface = build_operator_surface(
        difficulty, cfg.profile, seed=cfg.seed, variant=cfg.variant, size=cfg.size
    )
    vertices, faces = surface_mesh_arrays(surface)
    return [trimesh.Trimesh(vertices=vertices, faces=faces, process=False)], surface[
        "origin"
    ]


def make_operator_terrain_generator(
    *,
    seed,
    num_rows=7,
    num_cols=20,
    size=TILE_SIZE,
    difficulty_range=(0.0, 1.0),
    curriculum=True,
):
    """Create native fixed-column geometry, not an adaptive environment curriculum.

    Native random-layout mode cannot promise a true-flat fraction. Exactly twenty
    columns (or a whole repeated block) therefore use curriculum *layout*, while
    the caller independently controls its environment curriculum and assignments.
    """
    if (
        type(seed) is not int
        or seed < 0
        or type(num_rows) is not int
        or num_rows < 1
        or type(num_cols) is not int
        or num_cols < 20
        or num_cols % 20
        or curriculum is not True
        or len(difficulty_range) != 2
        or not all(math.isfinite(float(v)) for v in difficulty_range)
        or not 0 <= difficulty_range[0] <= difficulty_range[1] <= 1
    ):
        raise ValueError(
            "Require fixed column blocks of 20 and finite ordered difficulty in [0,1]"
        )
    # Validate dimensions before constructing simulator-owned configuration.
    build_operator_surface(0.0, "plane", seed=seed, size=size)
    from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
    from isaaclab.utils import configclass

    @configclass
    class OperatorSubTerrainCfg(SubTerrainBaseCfg):
        function = operator_terrain
        profile: str = "plane"
        variant: int = 0

    terrains = {
        f"{profile}_{column:02d}": OperatorSubTerrainCfg(
            proportion=1 / num_cols, profile=profile, variant=column
        )
        for column in range(num_cols)
        for profile in (PROFILE_BY_COLUMN[column % 20],)
    }
    return TerrainGeneratorCfg(
        seed=seed,
        size=size,
        num_rows=num_rows,
        num_cols=num_cols,
        curriculum=True,
        difficulty_range=difficulty_range,
        horizontal_scale=RESOLUTION,
        border_width=2.0,
        border_height=1.0,
        sub_terrains=terrains,
        use_cache=False,
    )
