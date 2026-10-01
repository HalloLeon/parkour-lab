"""Reusable local rough structures, measured before any simulator import.

These straight supported strips are geometry building blocks, not mixed worlds,
native collision validation, or complete command-tape/evaluation maps. Existing
terrain routes are unchanged. Dimensions and heights are metres; grades are degrees.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real

import numpy as np

SPACINGS = (0.4, 0.2)
TARGET_RMS = (0.003, 0.00075)
MINIMUM_RMS = (0.002, 0.0005)
MAXIMUM_GRADE = math.tan(math.radians(25))


def _positive(value):
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, Real)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _axis(knots, resolution):
    knots = np.unique(knots)
    # Small spacing headroom avoids rounding a nominal 2 cm interval above the
    # limit, including the representative local float32 conversion in tests.
    pieces = [
        np.linspace(a, b, math.ceil((b - a) / (resolution * 0.9999)) + 1)[:-1]
        for a, b in zip(knots[:-1], knots[1:], strict=True)
    ]
    return np.concatenate((*pieces, knots[-1:]))


def _gradient_noise(x, y, spacing, seed):
    """Independent unit lattice gradients, corner dot products, quintic blend."""
    xx, yy = np.meshgrid(x / spacing, y / spacing, indexing="ij")
    return _noise_points(xx, yy, seed)


def _noise_points(xx, yy, seed):
    """Evaluate the common noise field at lattice-scaled mesh coordinates."""
    ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
    tx, ty = xx - ix, yy - iy
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * math.pi, (ix.max() + 2, iy.max() + 2))
    gx, gy = np.cos(angles), np.sin(angles)
    corners = [
        gx[ix + a, iy + b] * (tx - a) + gy[ix + a, iy + b] * (ty - b)
        for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))
    ]
    ux = tx**3 * (tx * (6 * tx - 15) + 10)
    uy = ty**3 * (ty * (6 * ty - 15) + 10)
    low = (1 - ux) * corners[0] + ux * corners[1]
    high = (1 - ux) * corners[2] + ux * corners[3]
    return (1 - uy) * low + uy * high


def _normals(vertices, faces):
    triangles = vertices[faces]
    return np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )


def _gradients(vertices, faces):
    normals = _normals(vertices, faces)
    if (normals[:, 2] <= 0).any():
        raise ValueError("Supporting triangles must have positive projected area")
    return -normals[:, :2] / normals[:, 2, None]


def _shared_scale(carrier_gradient, rough_gradient, offset, cap):
    """Largest common scale in [0,1], including cancellation with carrier slope."""
    a = np.sum(rough_gradient**2, axis=1)
    b = 2 * np.sum(carrier_gradient * rough_gradient, axis=1)
    c = np.sum(carrier_gradient**2, axis=1) - MAXIMUM_GRADE**2
    if (c > 0).any():
        raise ValueError("Carrier already exceeds the supporting-slope limit")
    active = a > 0
    a, b, c = a[active], b[active], c[active]
    discriminant = np.sqrt(b * b - 4 * a * c)
    # Equivalent quadratic roots; the second form avoids cancellation for b>0.
    roots = np.empty(a.shape, dtype=float)
    outward = b > 0
    roots[outward] = -2 * c[outward] / (b[outward] + discriminant[outward])
    roots[~outward] = (-b[~outward] + discriminant[~outward]) / (2 * a[~outward])
    peak = float(np.max(np.abs(offset)))
    if np.ndim(cap):
        absolute = np.abs(offset)
        height_scale = float(
            np.divide(
                cap, absolute, out=np.full_like(absolute, np.inf), where=absolute > 0
            ).min()
        )
    else:
        height_scale = cap / peak if peak else 1.0
    return min(1.0, height_scale, float(roots.min(initial=1.0)))


def _carrier_profile(family, tier, x, entry, lips, reverse, risers, incline_length):
    """Shared one-dimensional carrier, before any world assembly or roughness."""
    carrier = np.zeros_like(x)
    if family == "stairs":
        rise = float(tier) - 0.001
        levels = np.searchsorted(lips, x, side="right")
        for lip in lips:
            levels[np.flatnonzero(x == lip)[0]] -= 1
        carrier = levels * rise
        if reverse:
            carrier = risers * rise - carrier
    elif family in ("ramp", "hill"):
        grade = math.tan(math.radians(float(tier) - 0.1))
        if family == "ramp":
            carrier = np.clip(x - entry, 0, incline_length) * grade
            if reverse:
                carrier = incline_length * grade - carrier
        else:
            up, down = (1.5, 1.0) if reverse else (1.0, 1.5)
            height = 2 * min(up, down) * grade / math.pi
            rise = (1 - np.cos(math.pi * np.clip((x - entry) / up, 0, 1))) / 2
            fall = (
                1 + np.cos(math.pi * np.clip((x - entry - up - 0.4) / down, 0, 1))
            ) / 2
            carrier = height * np.minimum(rise, fall)
    return carrier


def _grid_mesh(x, y, carrier):
    xx, yy = np.meshgrid(x, y, indexing="ij")
    vertices = np.column_stack((xx.ravel(), yy.ravel(), carrier.ravel()))
    ids = np.arange(xx.size).reshape(xx.shape)
    a, b, c, d = ids[:-1, :-1], ids[1:, :-1], ids[:-1, 1:], ids[1:, 1:]
    faces = np.stack((np.stack((a, b, c), -1), np.stack((b, d, c), -1)), -2).reshape(
        -1, 3
    )
    return vertices, faces


def _roughness_layers(x, y, taper, untapered, seeds, *, grid=True):
    layers = []
    for spacing, rms, seed in zip(SPACINGS, TARGET_RMS, seeds, strict=True):
        layer = (
            _gradient_noise(x, y, spacing, seed)
            if grid
            else _noise_points(x / spacing, y / spacing, seed)
        )
        layer -= layer[untapered].mean()
        deviation = float(np.sqrt(np.mean(layer[untapered] ** 2)))
        if deviation == 0 or not math.isfinite(deviation):
            raise ValueError("Degenerate roughness layer")
        layers.append(layer * (rms / deviation) * taper)
    return layers


def _apply_roughness(vertices, faces, support, layers, cap):
    carrier_gradient = _gradients(vertices, faces[support])
    rough = sum(layers)
    rough_vertices = vertices.copy()
    rough_vertices[:, 2] = rough.ravel()
    rough_gradient = _gradients(rough_vertices, faces[support])
    scale = _shared_scale(carrier_gradient, rough_gradient, rough, cap)
    if scale < max(
        low / target for low, target in zip(MINIMUM_RMS, TARGET_RMS, strict=True)
    ):
        raise ValueError(
            "Infeasible roughness: shared slope/height scaling violates minimum RMS"
        )
    layers = [layer * scale for layer in layers]
    vertices[:, 2] += sum(layers).ravel()
    return layers, scale, carrier_gradient


def build_structure(
    family,
    tier,
    *,
    coarse_seed,
    fine_seed,
    reverse=False,
    risers=4,
    tread=0.4,
    incline_length=2.0,
    width=2.0,
    approach=2.0,
    landing=4.0,
    pad_length=1.0,
    resolution=0.02,
):
    """Build a local strip with rough approach/landing and flat endpoint pads.

    ``family`` is stairs/ramp/hill/level; ``tier`` is a riser tier in metres,
    grade tier in degrees, or level-ground roughness cap in metres respectively.
    Reverse descends stairs/ramps or reflects the hill's unequal flank lengths.
    Stairs have exactly ``risers`` vertical faces and ``risers-1`` full treads;
    the final riser meets the supporting landing. An isolated step uses risers=1.

    Pad length is the exactly flat length at each end; the total pad includes
    another 0.5 m taper inward into the pad from the rough surface. Roughness
    vanishes only at these pads and within 0.02 m
    of stair lips. Carrier targets sit at the midpoint of the allowed realized
    tolerances (tier-1 mm or tier-0.1 degrees); actual mesh quantities are reported.
    No native coordinate conversion or world-boundary stitching is certified here.
    """
    if type(family) is not str or family not in ("stairs", "ramp", "hill", "level"):
        raise ValueError("Unknown structure family")
    if not _positive(tier):
        raise ValueError("Tier must be a positive finite number")
    tiers = (0.04, 0.08, 0.12, 0.16) if family == "stairs" else (10, 15, 20)
    if family == "level":
        tiers = (0.01, 0.02)
    if tier not in tiers:
        raise ValueError("Tier is outside the declared structure envelope")
    if (
        type(reverse) is not bool
        or any(type(seed) is not int or seed < 0 for seed in (coarse_seed, fine_seed))
        or coarse_seed == fine_seed
    ):
        raise ValueError(
            "Require a boolean direction and distinct nonnegative layer seeds"
        )
    if type(risers) is not int or risers not in (1, 4, 5, 6):
        raise ValueError("Stairs require one or four through six risers")
    if not _positive(tread) or not 0.31 <= tread <= 0.5:
        raise ValueError("Treads must lie in [0.31,0.50] metres")
    if not _positive(incline_length) or incline_length not in (2, 3):
        raise ValueError("Ramp horizontal length must be two or three metres")
    if (
        (family != "stairs" and (risers != 4 or tread != 0.4))
        or (family != "ramp" and incline_length != 2.0)
        or (family == "level" and reverse)
    ):
        raise ValueError("Nondefault shape options do not apply to this family")
    if (
        not all(
            _positive(value)
            for value in (width, approach, landing, pad_length, resolution)
        )
        or width < 2
        or resolution > 0.02
        or min(approach, landing) < pad_length + 0.5
    ):
        raise ValueError(
            "Require width>=2 m, resolution<=0.02 m and room for pads/tapers"
        )

    entry = float(approach)
    lips = np.array([], dtype=float)
    if family == "stairs":
        lips = entry + np.arange(risers) * tread
        exit_x = float(lips[-1])
        knots = [0, *lips, exit_x + landing]
    elif family == "ramp":
        exit_x = entry + incline_length
        knots = [0, entry, exit_x, exit_x + landing]
    elif family == "hill":
        up, down = (1.5, 1.0) if reverse else (1.0, 1.5)
        exit_x = entry + up + 0.4 + down
        knots = [0, entry, entry + up, entry + up + 0.4, exit_x, exit_x + landing]
    else:
        exit_x = entry
        knots = [0, entry, entry + landing]
    end = exit_x + landing
    knots.extend(
        (pad_length, pad_length + 0.5, end - pad_length - 0.5, end - pad_length)
    )
    if len(lips):
        knots.extend((lips - 0.02).tolist() + (lips + 0.02).tolist())
    x = _axis(knots, resolution)
    if len(lips):
        x = np.sort(np.concatenate((x, lips)))
    y = _axis([0, width], resolution)
    carrier = _carrier_profile(
        family, tier, x, entry, lips, reverse, risers, incline_length
    )

    taper = np.minimum(
        np.clip((x - pad_length) / 0.5, 0, 1),
        np.clip((end - pad_length - x) / 0.5, 0, 1),
    )
    if len(lips):
        distance = np.min(np.abs(x[:, None] - lips), axis=1)
        taper = np.minimum(taper, np.clip(distance / 0.02, 0, 1))
    taper = taper**3 * (taper * (6 * taper - 15) + 10)
    full_intervals = (np.diff(x) > 0) & (taper[:-1] == 1) & (taper[1:] == 1)
    if not full_intervals.any():
        raise ValueError("Structure has no positive-area untapered support")
    # Only samples incident to fully untapered supporting triangles count; a
    # single full-amplitude line between two pad tapers has no supporting area.
    full_rows = np.zeros(len(x), dtype=bool)
    full_rows[:-1] |= full_intervals
    full_rows[1:] |= full_intervals
    untapered = np.broadcast_to(full_rows[:, None], (len(x), len(y)))
    layers = _roughness_layers(
        x, y, taper[:, None], untapered, (coarse_seed, fine_seed)
    )
    carrier = np.broadcast_to(carrier[:, None], (len(x), len(y))).copy()
    vertices, faces = _grid_mesh(x, y, carrier)
    support = np.broadcast_to(
        (np.diff(x) > 0)[:, None, None], (len(x) - 1, len(y) - 1, 2)
    ).ravel()
    cap = float(tier) if family == "level" else (0.01 if tier in (0.16, 20) else 0.02)
    layers, scale, carrier_gradient = _apply_roughness(
        vertices, faces, support, layers, cap
    )
    normals = _normals(vertices, faces)
    if (
        not np.isfinite(vertices).all()
        or (np.linalg.norm(normals, axis=1) == 0).any()
        or (normals[support, 2] <= 0).any()
        or (normals[~support, 2] != 0).any()
    ):
        raise ValueError("Invalid supporting/riser mesh topology")
    grades = np.linalg.norm(_gradients(vertices, faces[support]), axis=1)
    carrier_degrees = math.degrees(
        math.atan(float(np.linalg.norm(carrier_gradient, axis=1).max()))
    )
    actual_rises = np.abs(np.diff(carrier[:, 0])[np.diff(x) == 0])
    actual_rms = [float(np.sqrt(np.mean(layer[untapered] ** 2))) for layer in layers]
    peak = float(np.abs(sum(layers)).max())
    if (
        grades.max() > MAXIMUM_GRADE + 1e-12
        or peak > cap + 1e-12
        or any(
            rms < minimum - 1e-12
            for rms, minimum in zip(actual_rms, MINIMUM_RMS, strict=True)
        )
        or (
            family == "stairs"
            and ((actual_rises < tier - 0.002).any() or (actual_rises > tier).any())
        )
        or (family in ("ramp", "hill") and not tier - 0.2 <= carrier_degrees <= tier)
    ):
        raise ValueError("Realized structure exceeds its geometry/roughness envelope")
    metadata = {
        "version": "rough_local_structure_v1",
        "scope": "Local generated mesh only; not native collision, mixed-world or robot acceptance",
        "family": family,
        "tier": float(tier),
        "reverse": reverse,
        "coarse_seed": coarse_seed,
        "fine_seed": fine_seed,
        "entry_x_m": entry,
        "exit_x_m": exit_x,
        "size_m": [end, float(width)],
        "flat_pad_length_m": float(pad_length),
        "pad_regions_x_m": [[0.0, pad_length + 0.5], [end - pad_length - 0.5, end]],
        "pad_taper_regions_x_m": [
            [pad_length, pad_length + 0.5],
            [end - pad_length - 0.5, end - pad_length],
        ],
        "untapered_support_area_m2": float(np.diff(x)[full_intervals].sum() * width),
        "maximum_axis_spacing_m": float(max(np.diff(x).max(), np.diff(y).max())),
        "riser_heights_m": actual_rises.tolist(),
        "tread_lengths_m": np.diff(lips).tolist(),
        "maximum_carrier_grade_degrees": carrier_degrees,
        "maximum_supporting_grade_degrees": math.degrees(
            math.atan(float(grades.max()))
        ),
        "projected_support_area_m2": float(normals[support, 2].sum() / 2),
        "roughness_lattice_spacing_m": list(SPACINGS),
        "roughness_shared_scale": scale,
        "roughness_rms_m": actual_rms,
        "roughness_cap_m": cap,
        "maximum_roughness_offset_m": peak,
        "vertical_face_count": int((~support).sum()),
        "vertices_float64_sha256": hashlib.sha256(
            vertices.astype("<f8").tobytes()
        ).hexdigest(),
        "faces_int64_sha256": hashlib.sha256(faces.astype("<i8").tobytes()).hexdigest(),
    }
    return {
        "x": x,
        "y": y,
        "vertices": vertices,
        "faces": faces,
        "carrier_heights": carrier,
        "coarse_heights": layers[0],
        "fine_heights": layers[1],
        "taper": taper,
        "untapered_samples": untapered,
        "support_faces": support,
        "metadata": metadata,
    }
