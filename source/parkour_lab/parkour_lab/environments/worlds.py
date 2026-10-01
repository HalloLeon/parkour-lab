"""Connected source-mesh diagnostics with one assigned, unassisted encounter.

This is not a frozen evaluation bank or a native terrain implementation. A single
carrier joins perpendicular stairs and a ramp through their upper envelope, with
a rounded hill on lower ground. One noise field covers that connected surface.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Real

import numpy as np

from .structures import (
    MINIMUM_RMS,
    SPACINGS,
    _apply_roughness,
    _axis,
    _carrier_profile,
    _grid_mesh,
    _normals,
    _positive,
    _roughness_layers,
)


def _stair_lips(entry, tread, risers):
    """Keep legal lips; otherwise realize a constant common-lattice tread.

    A binary32 lattice spanning the flight permits exactly equal authored and
    converted treads. Choose the nearest legal lattice tread and nearest lattice
    entry, with ties to even. Entry movement is at most half that lattice spacing;
    this is coordinate representation, not a search for a different layout.
    """
    lips = entry + np.arange(risers) * tread

    def legal(values):
        converted = values.astype(np.float32).astype(np.float64)
        return np.isfinite(converted).all() and all(
            ((np.diff(v) >= 0.31) & (np.diff(v) <= 0.5)).all()
            for v in (values, converted)
        )

    receipt = {
        "requested_entry_m": float(entry),
        "requested_tread_m": float(tread),
        "realized_entry_m": float(entry),
        "realized_tread_m": float(tread),
        "coordinate_headroom_applied": False,
        "coordinate_lattice_m": None,
    }
    if legal(lips):
        return lips, receipt
    spacing = float(np.spacing(np.float32(lips[-1])))
    for _ in range(4):
        if not math.isfinite(spacing) or spacing <= 0:
            break
        minimum, maximum = math.ceil(0.31 / spacing), math.floor(0.5 / spacing)
        if minimum > maximum:
            break
        step = min(max(round(tread / spacing), minimum), maximum) * spacing
        origin = round(entry / spacing) * spacing
        candidate = origin + np.arange(risers) * step
        needed = float(np.spacing(np.float32(candidate[-1])))
        if needed > spacing:
            spacing = needed
            continue
        if (
            not legal(candidate)
            or abs(origin - entry) > spacing / 2
            or not np.array_equal(candidate, candidate.astype(np.float32))
        ):
            break
        receipt.update(
            realized_entry_m=float(origin),
            realized_tread_m=float(step),
            coordinate_headroom_applied=True,
            coordinate_lattice_m=spacing,
        )
        return candidate, receipt
    raise ValueError("Cannot represent constant in-envelope source/float32 treads")


def _inside(points, bounds):
    return ((points[:, :2] >= bounds[0]) & (points[:, :2] <= bounds[1])).all(axis=1)


def _overlap(a, b):
    return bool((np.maximum(a[0], b[0]) < np.minimum(a[1], b[1])).all())


def _rectangle(center, half_size):
    return np.array([np.asarray(center) - half_size, np.asarray(center) + half_size])


def _trial(family, reverse, features, size, yaw):
    """Fixed tape with an explicit nominal swept-support construction margin."""
    feature = features[family]
    axis = feature["axis"]
    direction = -1 if reverse and family != "hill" else 1
    entry, exit_point = feature["entry"].copy(), feature["exit"].copy()
    if direction < 0:
        entry, exit_point = exit_point, entry
    start = entry.copy()
    start[axis] -= 2 * direction
    finish = start.copy()
    finish[axis] += 8.75 * direction
    heading = math.atan2(direction if axis else 0, direction if not axis else 0)
    # Cover every allowed initial heading/position, not only the centre line.
    # The extra radius is a declared layout margin, not a contact/body model.
    forward = np.array([math.cos(heading), math.sin(heading)])
    lateral = np.array([-forward[1], forward[0]])
    end_points = np.array(
        [
            start + distance * (math.cos(angle) * forward + math.sin(angle) * lateral)
            for distance in (0, 8.75)
            for angle in (-0.03, 0.0, 0.03)
        ]
    )
    envelope = np.array([end_points.min(axis=0) - 0.55, end_points.max(axis=0) + 0.55])
    start_pad, stop_pad = _rectangle(start, 1.0), _rectangle(finish, 1.0)
    if any(
        (pad[0] - 0.5 < 0).any() or (pad[1] + 0.5 > size).any()
        for pad in (start_pad, stop_pad)
    ):
        raise ValueError("Start/final-stop pad and taper must fit inside the map")
    if (envelope[0] <= 0).any() or (envelope[1] >= size).any():
        raise ValueError("Nominal tape and start variation lack supporting-map margin")
    for other, surrounding in features.items():
        if other != family and _overlap(envelope, surrounding["influence"]):
            raise ValueError("Nominal tape encounters an unassigned structure")
    across = 1 - axis
    footprint = feature["footprint"]
    side_margin = min(
        envelope[0, across] - footprint[0, across],
        footprint[1, across] - envelope[1, across],
    )
    crossing_margin = (
        direction * (finish[axis] - exit_point[axis])
        - 0.55
        - 8.75 * (1 - math.cos(0.03))
    )
    if side_margin <= 0 or crossing_margin <= 0:
        raise ValueError(
            "Nominal tape cannot clear the target with its support envelope"
        )
    # Final-stop samples under the nominal motion model must fit the flat pad.
    final_points = end_points[3:]
    stop_envelope = np.array(
        [final_points.min(axis=0) - 0.55, final_points.max(axis=0) + 0.55]
    )
    if not (
        (stop_envelope[0] >= stop_pad[0]).all()
        and (stop_envelope[1] <= stop_pad[1]).all()
    ):
        raise ValueError("Final-stop pad does not contain nominal start/yaw variation")
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]
    )
    transform = np.eye(4)
    transform[:2, :2] = rotation
    landing = footprint.copy()
    landing[:, axis] = sorted((exit_point[axis], finish[axis] + direction))
    return {
        "family": family,
        "orientation": ("long-first" if reverse else "short-first")
        if family == "hill"
        else ("down" if reverse else "up"),
        "start_xy_local_m": start.tolist(),
        "start_xy_world_m": (rotation @ start).tolist(),
        "heading_local_rad": heading,
        "heading_world_rad": heading + yaw,
        "approach_edge_local_m": entry.tolist(),
        "exit_edge_local_m": exit_point.tolist(),
        "target_footprint_local_m": footprint.tolist(),
        "far_side_landing_local_m": landing.tolist(),
        "start_pad_local_m": start_pad.tolist(),
        "final_stop_region_local_m": stop_pad.tolist(),
        "nominal_finish_xy_local_m": finish.tolist(),
        "nominal_swept_support_local_m": envelope.tolist(),
        "nominal_final_support_local_m": stop_envelope.tolist(),
        "minimum_map_margin_m": float(
            min(envelope[0].min(), (size - envelope[1]).min())
        ),
        "minimum_target_side_margin_m": float(side_margin),
        "target_clearance_at_nominal_finish_m": float(crossing_margin),
        "support_radius_margin_m": 0.5,
        "start_xy_offset_bound_m": 0.05,
        "start_yaw_offset_bound_rad": 0.03,
        "command_phases": [
            [2.0, [0.0, 0.0, 0.0]],
            [25.0, [0.35, 0.0, 0.0]],
            [3.0, [0.0, 0.0, 0.0]],
        ],
        "nominal_flat_travel_m": 8.75,
        "motion_model_scope": "Constant prescribed initial heading; not a prediction or guarantee of robot motion",
        "local_to_world_column_transform": transform.tolist(),
    }


def build_world(
    target_family,
    tier,
    *,
    coarse_seed,
    fine_seed,
    reverse=False,
    risers=4,
    tread=0.4,
    incline_length=2.0,
    size=(24.0, 24.0),
    stair_entry=14.0,
    ramp_entry=14.0,
    stair_route_y=11.0,
    ramp_route_x=7.0,
    hill_entry=4.0,
    hill_center_y=4.0,
    world_yaw=0.0,
    resolution=0.02,
):
    """Build one configurable connected diagnostic world, never a policy trial.

    The assigned family uses ``tier``; surrounding stairs/ramp/hill use 8 cm,
    10 degrees and 10 degrees. Hill reversal reflects its unequal flanks while
    travelling +x; stairs/ramp reversal instead changes the supported start side.
    Legal requested stair coordinates are preserved. If binary32 conversion would
    violate a tread bound, realize the flight on a common binary32 lattice before
    authoring the carrier/noise, and report requested versus realized coordinates.
    Shared roughness scaling targets 25 degrees; small rounding errors are accepted.
    Vertices stay local. The recorded rigid transform rotates mesh and start
    together; arbitrary independent feature rotations are intentionally unsupported.
    """
    if target_family not in ("stairs", "ramp", "hill") or not _positive(tier):
        raise ValueError("Require a stairs, ramp or hill target and a valid tier")
    if tier not in (
        (0.04, 0.08, 0.12, 0.16) if target_family == "stairs" else (10, 15, 20)
    ):
        raise ValueError("Target tier is outside the declared envelope")
    if (
        type(reverse) is not bool
        or any(type(seed) is not int or seed < 0 for seed in (coarse_seed, fine_seed))
        or coarse_seed == fine_seed
    ):
        raise ValueError(
            "Require a boolean direction and distinct nonnegative roughness seeds"
        )
    if (
        type(risers) is not int
        or risers not in (4, 5, 6)
        or not _positive(tread)
        or not 0.31 <= tread <= 0.5
    ):
        raise ValueError("Require four through six risers and 0.31–0.50 m treads")
    if not _positive(incline_length) or incline_length not in (2, 3):
        raise ValueError("Ramp horizontal length must be two or three metres")
    if (target_family != "stairs" and (risers != 4 or tread != 0.4)) or (
        target_family != "ramp" and incline_length != 2.0
    ):
        raise ValueError("Shape overrides must apply to the assigned family")
    if (
        not isinstance(size, (tuple, list))
        or len(size) != 2
        or not all(_positive(v) for v in size)
    ):
        raise ValueError("World size must contain two positive finite lengths")
    size = np.asarray(size, dtype=float)
    if (
        not all(
            _positive(v)
            for v in (
                stair_entry,
                ramp_entry,
                stair_route_y,
                ramp_route_x,
                hill_entry,
                hill_center_y,
                resolution,
            )
        )
        or resolution > 0.02
    ):
        raise ValueError("Require positive placements and resolution at most 0.02 m")
    if (
        isinstance(world_yaw, (bool, np.bool_))
        or not isinstance(world_yaw, Real)
        or not math.isfinite(world_yaw)
    ):
        raise ValueError("World yaw must be finite")
    stair_tier = float(tier) if target_family == "stairs" else 0.08
    ramp_tier = float(tier) if target_family == "ramp" else 10.0
    hill_tier = float(tier) if target_family == "hill" else 10.0
    hill_reverse = reverse and target_family == "hill"
    lips, stair_coordinates = _stair_lips(stair_entry, tread, risers)
    stair_entry = float(lips[0])
    ramp_grade = math.tan(math.radians(9.9))
    target_ramp_grade = math.tan(math.radians(ramp_tier - 0.1))
    ramp_core = np.array([ramp_route_x - 1.5, ramp_route_x + 1.5])
    shoulder_widths = np.array([ramp_core[0], stair_entry - ramp_core[1]])
    # This sufficient bound keeps the analytic combined carrier gradient below
    # the named core grade even across the elevated lateral shoulders.
    required_shoulder = (
        incline_length
        * math.pi
        * math.sqrt((target_ramp_grade - ramp_grade) / (2 * target_ramp_grade))
    )
    if (shoulder_widths <= max(required_shoulder, 0)).any():
        raise ValueError(
            "Ramp side shoulders are too narrow for the carrier-slope limit"
        )
    hill_exit = hill_entry + 2.9
    features = {
        "stairs": {
            **stair_coordinates,
            "tier": stair_tier,
            "axis": 0,
            "entry": np.array([stair_entry, stair_route_y]),
            "exit": np.array([lips[-1], stair_route_y]),
            "footprint": np.array([[stair_entry, 0.0], [lips[-1], ramp_entry]]),
            "influence": np.array([[stair_entry, 0.0], size]),
        },
        "ramp": {
            "tier": ramp_tier,
            "axis": 1,
            "entry": np.array([ramp_route_x, ramp_entry]),
            "exit": np.array([ramp_route_x, ramp_entry + incline_length]),
            "footprint": np.array(
                [
                    [ramp_core[0], ramp_entry],
                    [ramp_core[1], ramp_entry + incline_length],
                ]
            ),
            "influence": np.array([[0.0, ramp_entry], size]),
            "core_x_m": ramp_core.tolist(),
            "shoulder_widths_m": shoulder_widths.tolist(),
            "minimum_shoulder_width_m": float(required_shoulder),
            "intersection_grade_degrees": 9.9,
        },
        "hill": {
            "tier": hill_tier,
            "axis": 0,
            "entry": np.array([hill_entry, hill_center_y]),
            "exit": np.array([hill_exit, hill_center_y]),
            "footprint": np.array(
                [[hill_entry, hill_center_y - 1], [hill_exit, hill_center_y + 1]]
            ),
            "influence": np.array(
                [[hill_entry, hill_center_y - 3], [hill_exit, hill_center_y + 3]]
            ),
        },
    }
    if (
        ramp_entry < 2
        or stair_entry < 2
        or lips[-1] + 0.02 >= size[0]
        or any(
            (f["influence"][0] < 0).any() or (f["footprint"][1] > size).any()
            for f in features.values()
        )
    ):
        raise ValueError("Structures lack their required support or usable width")
    if (
        any(
            _overlap(features["hill"]["influence"], features[family]["influence"])
            for family in ("stairs", "ramp")
        )
        or (features["hill"]["influence"][1] > size).any()
    ):
        raise ValueError("Hill sidebanks must lie on separate lower supporting ground")
    trial = _trial(target_family, reverse, features, size, world_yaw)
    up = 1.5 if hill_reverse else 1.0
    stair_levels = np.arange(1, risers + 1) * (stair_tier - 0.001)
    crossings = ramp_entry + stair_levels / ramp_grade
    crossing_levels = stair_levels[crossings < ramp_entry + incline_length]
    crossings = crossings[crossings < ramp_entry + incline_length]
    x_knots = [
        0.0,
        *lips,
        *(lips - 0.02),
        *(lips + 0.02),
        hill_entry,
        hill_entry + up,
        hill_entry + up + 0.4,
        hill_exit,
        *ramp_core,
        size[0],
    ]
    y_knots = [
        0.0,
        ramp_entry,
        ramp_entry + incline_length,
        *crossings,
        hill_center_y - 3,
        hill_center_y - 1,
        hill_center_y + 1,
        hill_center_y + 3,
        size[1],
    ]
    for pad in (trial["start_pad_local_m"], trial["final_stop_region_local_m"]):
        for axis, knots in enumerate((x_knots, y_knots)):
            knots.extend(
                v
                for edge in pad
                for v in (edge[axis] - 0.5, edge[axis], edge[axis] + 0.5)
                if 0 <= v <= size[axis]
            )
    x = np.sort(np.concatenate((_axis(x_knots, resolution), lips)))
    y = _axis(y_knots, resolution)
    axis_spacing = float(max(np.diff(x).max(), np.diff(y).max()))
    float32_spacing = float(
        max(
            np.diff(axis.astype(np.float32).astype(np.float64)).max() for axis in (x, y)
        )
    )
    if axis_spacing > 0.02 or float32_spacing > 0.02:
        raise ValueError("Source/float32 horizontal spacing exceeds 0.02 m")
    stairs = _carrier_profile(
        "stairs", stair_tier, x, stair_entry, lips, False, risers, 2.0
    )
    ramp = _carrier_profile(
        "ramp", 10.0, y, ramp_entry, np.array([]), False, 4, incline_length
    )
    # These knots are exact intersections by construction. Re-evaluating y-entry
    # loses a few ulps and would leave fictitious, near-zero terminal risers.
    for crossing, level in zip(crossings, crossing_levels, strict=True):
        ramp[y == crossing] = level
    # Preserve the named-grade core and blend lateral/upper-plateau support to
    # a moderate ramp before the first stair lip. No roughness is removed here.
    left = np.clip(x / shoulder_widths[0], 0, 1)
    right = np.clip((stair_entry - x) / shoulder_widths[1], 0, 1)
    ramp_weight = np.minimum(
        (1 - np.cos(math.pi * left)) / 2, (1 - np.cos(math.pi * right)) / 2
    )
    ramp_factor = 1 + (target_ramp_grade / ramp_grade - 1) * ramp_weight
    hill = _carrier_profile(
        "hill", hill_tier, x, hill_entry, np.array([]), hill_reverse, 4, 2.0
    )
    shoulder = np.clip((np.abs(y - hill_center_y) - 1) / 2, 0, 1)
    hill_width = (1 + np.cos(math.pi * shoulder)) / 2
    carrier = (
        np.maximum(stairs[:, None], ramp_factor[:, None] * ramp[None, :])
        + hill[:, None] * hill_width[None, :]
    )
    vertices, faces = _grid_mesh(x, y, carrier)
    support = np.broadcast_to(
        (np.diff(x) > 0)[:, None, None], (len(x) - 1, len(y) - 1, 2)
    ).ravel()
    # Collapse only known duplicate-x samples with identical carrier heights.
    # They are the disappearing portions of intentional risers, not mesh repairs.
    mapping = np.arange(len(vertices)).reshape(len(x), len(y))
    for row in np.flatnonzero(np.diff(x) == 0):
        same = carrier[row] == carrier[row + 1]
        mapping[row + 1, same] = mapping[row, same]
    representatives = mapping.ravel()
    retained = representatives == np.arange(len(vertices))
    compact = np.cumsum(retained) - 1
    faces = compact[representatives[faces]]
    valid_faces = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 0] != faces[:, 2])
        & (faces[:, 1] != faces[:, 2])
    )
    removed_faces = int((~valid_faces).sum())
    faces, support, vertices = (
        faces[valid_faces],
        support[valid_faces],
        vertices[retained],
    )
    carrier = vertices[:, 2].copy()
    converted_carrier = vertices.astype(np.float32).astype(np.float64)
    converted_carrier_normals = _normals(converted_carrier, faces)
    if (
        not np.isfinite(converted_carrier).all()
        or (np.linalg.norm(converted_carrier_normals, axis=1) == 0).any()
        or (converted_carrier_normals[support, 2] <= 0).any()
        or (converted_carrier_normals[~support, 2] != 0).any()
    ):
        raise ValueError("Invalid float32 carrier supporting/riser topology")
    converted_carrier_grades = np.degrees(
        np.arctan2(
            np.linalg.norm(converted_carrier_normals[support, :2], axis=1),
            converted_carrier_normals[support, 2],
        )
    )
    del converted_carrier_normals
    taper = np.ones(len(vertices))
    riser_segments = []
    ramp_height = incline_length * ramp_grade
    for index, lip in enumerate(lips):
        high = (index + 1) * (stair_tier - 0.001)
        end_y = size[1] if high > ramp_height else ramp_entry + high / ramp_grade
        riser_segments.append([[float(lip), 0.0], [float(lip), float(end_y)]])
        distance = np.hypot(vertices[:, 0] - lip, np.maximum(vertices[:, 1] - end_y, 0))
        taper = np.minimum(taper, np.clip(distance / 0.02, 0, 1))
    for pad in (trial["start_pad_local_m"], trial["final_stop_region_local_m"]):
        pad = np.asarray(pad)
        distance = np.maximum(
            np.maximum(pad[0] - vertices[:, :2], vertices[:, :2] - pad[1]), 0
        ).max(axis=1)
        pad_support = distance <= 0.5
        if np.ptp(carrier[pad_support]) > 1e-12:
            raise ValueError("Start/final-stop pad overlaps nonlevel carrier")
        taper = np.minimum(taper, np.clip(distance / 0.5, 0, 1))
    taper = taper**3 * (taper * (6 * taper - 15) + 10)
    full_faces = support & (taper[faces] == 1).all(axis=1)
    untapered = np.zeros(len(vertices), dtype=bool)
    untapered[faces[full_faces].ravel()] = True
    if not full_faces.any():
        raise ValueError("World has no positive-area untapered support")
    cap = np.full(len(vertices), 0.02)
    region_masks = {}
    for family, feature in features.items():
        region_masks[family] = _inside(vertices, feature["influence"])
        if feature["tier"] in (0.16, 20.0):
            cap[region_masks[family]] = 0.01
    layers = _roughness_layers(
        vertices[:, 0],
        vertices[:, 1],
        taper,
        untapered,
        (coarse_seed, fine_seed),
        grid=False,
    )
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
        raise ValueError("Invalid connected supporting/riser topology")
    grade = float(
        np.degrees(
            np.arctan2(
                np.linalg.norm(normals[support, :2], axis=1), normals[support, 2]
            )
        ).max()
    )
    rough = sum(layers)
    rms = [float(np.sqrt(np.mean(layer[untapered] ** 2))) for layer in layers]
    if (
        not math.isfinite(grade)
        or grade > 25.001
        or (np.abs(rough) > cap).any()
        or any(value < lower for value, lower in zip(rms, MINIMUM_RMS, strict=True))
    ):
        raise ValueError("World exceeds source roughness/slope limits")
    region_masks["connecting_ground"] = ~np.logical_or.reduce(
        tuple(region_masks.values())
    )
    region_rms = {
        name: [
            float(np.sqrt(np.mean(layer[mask & untapered] ** 2))) for layer in layers
        ]
        for name, mask in region_masks.items()
    }
    # Feature limits apply only to full-height footprints, not clipped seam walls.
    carrier_grades = np.degrees(np.arctan(np.linalg.norm(carrier_gradient, axis=1)))
    for family, feature in features.items():
        footprint_vertices = _inside(vertices, feature["footprint"])
        interior = footprint_vertices[faces[support]].all(axis=1)
        feature["measured_carrier_grade_degrees"] = float(
            carrier_grades[interior].max()
        )
        feature["float32_carrier_grade_degrees"] = float(
            converted_carrier_grades[interior].max()
        )
        across = 1 - feature["axis"]
        feature["usable_width_m"] = float(np.ptp(vertices[footprint_vertices, across]))
        feature["float32_usable_width_m"] = float(
            np.ptp(converted_carrier[footprint_vertices, across])
        )
        if min(feature["usable_width_m"], feature["float32_usable_width_m"]) < 2:
            raise ValueError("Source/float32 feature usable width is below 2 m")
        if family in ("ramp", "hill") and not all(
            feature["tier"] - 0.2 <= feature[key] <= feature["tier"]
            for key in (
                "measured_carrier_grade_degrees",
                "float32_carrier_grade_degrees",
            )
        ):
            raise ValueError("Source/float32 feature carrier grade misses its tier")
        if family == "stairs":
            wall = faces[~support & footprint_vertices[faces].all(axis=1)]
            rises = np.ptp(vertices[wall, 2], axis=1)
            converted_rises = np.ptp(
                vertices[wall, 2].astype(np.float32).astype(np.float64), axis=1
            )
            converted_treads = np.diff(lips.astype(np.float32).astype(np.float64))
            feature["measured_riser_range_m"] = [float(rises.min()), float(rises.max())]
            feature["float32_riser_range_m"] = [
                float(converted_rises.min()),
                float(converted_rises.max()),
            ]
            feature["measured_treads_m"] = np.diff(lips).tolist()
            feature["float32_treads_m"] = converted_treads.tolist()
            feature["riser_count"] = len(lips)
            if not all(
                (
                    (values >= feature["tier"] - 0.002) & (values <= feature["tier"])
                ).all()
                for values in (rises, converted_rises)
            ) or not all(
                ((values >= 0.31) & (values <= 0.5)).all()
                for values in (np.diff(lips), converted_treads)
            ):
                raise ValueError(
                    "Source/float32 feature risers/treads miss their envelope"
                )
        else:
            axis = feature["axis"]
            feature["measured_horizontal_length_m"] = float(
                np.ptp(vertices[footprint_vertices, axis])
            )
            feature["float32_horizontal_length_m"] = float(
                np.ptp(converted_carrier[footprint_vertices, axis])
            )
            boundaries = np.array([ramp_entry, ramp_entry + incline_length])
            if family == "hill":
                boundaries = np.array(
                    [hill_entry, hill_entry + up, hill_entry + up + 0.4, hill_exit]
                )
                feature["measured_flank_crest_flank_m"] = np.diff(boundaries).tolist()
                feature["float32_flank_crest_flank_m"] = np.diff(
                    boundaries.astype(np.float32).astype(np.float64)
                ).tolist()
            converted_boundaries = boundaries.astype(np.float32).astype(np.float64)
            feature["profile_boundaries_m"] = boundaries.tolist()
            feature["float32_profile_boundaries_m"] = converted_boundaries.tolist()
            feature["profile_boundary_conversion_error_m"] = (
                converted_boundaries - boundaries
            ).tolist()
    quantized = vertices.astype(np.float32).astype(np.float64)
    float32_normals = _normals(quantized, faces)
    if (
        not np.isfinite(quantized).all()
        or (np.linalg.norm(float32_normals, axis=1) == 0).any()
        or (float32_normals[support, 2] <= 0).any()
        or (float32_normals[~support, 2] != 0).any()
    ):
        raise ValueError("Invalid float32 supporting/riser topology")
    float32_grade = float(
        np.degrees(
            np.arctan2(
                np.linalg.norm(float32_normals[support, :2], axis=1),
                float32_normals[support, 2],
            )
        ).max()
    )
    conversion_bound = float(np.abs(quantized[:, 2] - vertices[:, 2]).max())
    conservative_rms = [value - conversion_bound for value in rms]
    if (
        not math.isfinite(float32_grade)
        or float32_grade > 25.001
        or (np.abs(quantized[:, 2] - carrier) > cap).any()
        or any(
            value < low
            for value, low in zip(conservative_rms, MINIMUM_RMS, strict=True)
        )
    ):
        raise ValueError("World exceeds float32 roughness/slope limits")
    metadata = {
        "version": "connected_terrace_diagnostic_v1",
        "scope": "Connected source mesh and prospective nominal tape margins only; not native import, contact, robot traversal or a frozen bank",
        "size_m": size.tolist(),
        "resolution_limit_m": float(resolution),
        "maximum_axis_spacing_m": axis_spacing,
        "float32_maximum_axis_spacing_m": float32_spacing,
        "coarse_seed": coarse_seed,
        "fine_seed": fine_seed,
        "roughness_lattice_spacing_m": list(SPACINGS),
        "roughness_shared_scale": scale,
        "boundary_taper": "Quintic 6t^5-15t^4+10t^3 within 0.02 m of actual risers and over 0.5 m at assigned pads",
        "roughness_rms_m": rms,
        "source_layer_rms_minus_conversion_bound_m": conservative_rms,
        "maximum_height_conversion_error_m": conversion_bound,
        "maximum_float32_residual_from_source_carrier_m": float(
            np.abs(quantized[:, 2] - carrier).max()
        ),
        "regional_rms_descriptive_m": region_rms,
        "roughness_cap_rule": "1 cm throughout maximum-tier feature influence (including upper-envelope seams), 2 cm elsewhere; diagnostic construction choice",
        "maximum_supporting_grade_degrees": grade,
        "maximum_carrier_grade_degrees": float(
            np.degrees(np.arctan(np.linalg.norm(carrier_gradient, axis=1).max()))
        ),
        "float32_maximum_carrier_grade_degrees": float(converted_carrier_grades.max()),
        "local_float32_supporting_grade_degrees": float32_grade,
        "local_float32_supporting_slope_within_limit": float32_grade <= 25.001,
        "float32_scope": "Local coordinate conversion only; not native import, cooked collision, contact or traversal evidence",
        "float32_topology_valid": True,
        "projected_support_area_m2": float(normals[support, 2].sum() / 2),
        "untapered_support_area_m2": float(normals[full_faces, 2].sum() / 2),
        "intentional_riser_segments_local_m": riser_segments,
        "riser_scope": "Original stair faces may be clipped where the ramp meets them; only the full-height target footprint carries the named stair tier",
        "welded_collapsed_riser_vertices": int((~retained).sum()),
        "removed_collapsed_riser_triangles": removed_faces,
        "vertical_face_count": int((~support).sum()),
        "features": {
            name: {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in feature.items()
            }
            for name, feature in features.items()
        },
        "trial": trial,
        "vertices_float64_sha256": hashlib.sha256(
            vertices.astype("<f8").tobytes()
        ).hexdigest(),
        "faces_int64_sha256": hashlib.sha256(faces.astype("<i8").tobytes()).hexdigest(),
        "vertices_float32_sha256": hashlib.sha256(
            vertices.astype("<f4").tobytes()
        ).hexdigest(),
    }
    return {
        "vertices": vertices,
        "faces": faces,
        "carrier_heights": carrier,
        "coarse_heights": layers[0],
        "fine_heights": layers[1],
        "taper": taper,
        "untapered_samples": untapered,
        "support_faces": support,
        "roughness_cap_m": cap,
        "metadata": metadata,
    }
