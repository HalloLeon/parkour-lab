"""Pure validation of opt-in development support telemetry, never qualification.

Isaac Lab v2.3.2 records post-step after current scene/termination updates and
before native reset. Its Warp raycast represents misses by paired all-+inf hit
and normal vectors. No simulator, tensor library, force threshold or support
success predicate is needed to replay this capture contract.
"""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np


SUPPORT_CAPTURE_MANIFEST = {
    "version": "operator_support_capture_v1",
    "status": "DEVELOPMENT_ONLY_NOT_SCORED",
    "sample_timing": "post_physics_pre_native_reset",
    "foot_order": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    "position_semantics": "named foot rigid-link origins in world coordinates",
    "force_semantics": "named feet net NORMAL contact force vectors, not full contact wrenches",
    "ray_semantics": "vertical downward rays on the verified native terrain mesh at foot-link XY; paired all-positive-infinity vectors mean missing ray",
    "radius_semantics": "measured link-centered collider extent envelope plus resolved maximum native robot contact offset",
    "geometry_binding": "sha256 of JSON geometry binding with sort_keys=True and allow_nan=False",
    "sample_index": "exact zero-based control-step index, one row per delivered action",
    "terminal_samples_receive_credit": False,
    "later_auto_reset_episodes_receive_credit": False,
    "policy_acceptance": False,
    "support_scoring": "NOT_SCORED",
    "scope": "Development capture completeness only; not contact points, a reconstructed contact patch, force closure, dynamic stability, ordered terrain coverage, stopping/tracking acceptance or qualification.",
}

_FOOT_VECTORS = (
    "foot_link_position_w",
    "foot_contact_force_w",
    "foot_ground_hit_w",
    "foot_ground_normal_w",
)
_PROTOCOL_VERSIONS = {
    "go2_operator_proprio_command_coverage_v1",
    "go2_operator_proprio_command_source_v1",
}
# Existing native geometry adapter's representable diagnostic envelope, not a
# contact/support threshold and not permission to assume a radius.
_MAXIMUM_COLLISION_RADIUS_M = 0.10


def _array(trace, name, shape, kind):
    value = trace.get(name)
    if not isinstance(value, np.ndarray) or value.shape != shape:
        raise ValueError(f"Invalid support capture field shape: {name}")
    if value.dtype.kind not in kind:
        raise ValueError(f"Invalid support capture field dtype: {name}")
    return value


def _sha256(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid support geometry SHA256: {name}")
    return value


def _finite_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
    ):
        raise ValueError(f"Invalid finite support geometry number: {name}")
    return float(value)


def _static_fields(trace):
    names = _array(trace, "foot_names", (4,), "U")
    if names.tolist() != SUPPORT_CAPTURE_MANIFEST["foot_order"]:
        raise ValueError("Support capture foot names/order differ from the contract")
    radii = _array(trace, "foot_collision_radius_m", (4,), "f")
    if (
        not np.isfinite(radii).all()
        or np.any(radii <= 0)
        or np.any(radii > _MAXIMUM_COLLISION_RADIUS_M)
    ):
        raise ValueError("Invalid measured support foot collision radii")
    digest = _array(trace, "foot_geometry_binding_sha256", (), "U").item()
    _sha256(digest, "foot_geometry_binding_sha256")
    return names, radii, digest


def _rays(trace):
    """Validate the exact native ray result encoding and finite-hit geometry."""
    hit = trace["foot_ground_hit_w"]
    normal = trace["foot_ground_normal_w"]
    finite = np.isfinite(hit).all(axis=-1) & np.isfinite(normal).all(axis=-1)
    missing = np.isposinf(hit).all(axis=-1) & np.isposinf(normal).all(axis=-1)
    if not np.all(finite | missing):
        raise ValueError(
            "Support mesh rays require finite paired hits/normals or paired all-positive-infinity misses"
        )
    if np.any(
        ~np.isclose(
            np.linalg.norm(normal[finite].astype(np.float64), axis=-1),
            1,
            atol=1e-3,
            rtol=0,
        )
    ) or np.any(normal[finite, 2] <= 0):
        raise ValueError("Support mesh rays require upward unit normals")
    feet = trace["foot_link_position_w"]
    # A nonfinite physical field is separately reported as invalid evidence;
    # it must not be hidden by dropping that row from a successful summary.
    comparable = finite & np.isfinite(feet).all(axis=-1)
    xy = feet[comparable, :2].astype(np.float64)
    tolerance = np.maximum(2e-5, 4 * np.finfo(np.float32).eps * np.abs(xy))
    if np.any(np.abs(hit[comparable, :2] - xy) > tolerance):
        raise ValueError("Support mesh rays are not vertically aligned with named feet")
    return finite, missing


def validate_support_capture(trace, protocol):
    """Describe first-attempt capture completeness without awarding support credit.

    Structural/capture-contract corruption raises. Nonfinite physical foot state
    is retained as explicit invalid first-attempt evidence, including a terminal
    row. Terminal and later housekeeping rows never contribute ray summaries.
    A complete capture is emphatically not a successful or qualified trial.
    """
    if (
        protocol.get("support_capture") != SUPPORT_CAPTURE_MANIFEST
        or protocol.get("version") not in _PROTOCOL_VERSIONS
        or protocol.get("foot_telemetry") != "post_physics_pre_reset_foot_regions_v1"
    ):
        raise ValueError("Undeclared or incompatible development support capture")
    steps, count, dt = (
        protocol.get("steps"),
        protocol.get("num_envs"),
        protocol.get("period_s"),
    )
    if (
        type(steps) is not int
        or steps <= 0
        or type(count) is not int
        or count <= 0
        or isinstance(dt, bool)
        or dt != 0.02
    ):
        raise ValueError("Support capture requires a positive fixed 50 Hz horizon")
    index = _array(trace, "sample_index", (steps,), "iu")
    if not np.array_equal(index, np.arange(steps, dtype=np.int64)):
        raise ValueError(
            "Support capture sample_index must cover every control step exactly"
        )
    for name in _FOOT_VECTORS:
        _array(trace, name, (steps, count, 4, 3), "f")
    names, _, digest = _static_fields(trace)
    for name in (
        "terminated",
        "time_out",
        "procedural_workspace",
        "valid_first_attempt",
    ):
        _array(trace, name, (steps, count), "b")
    if np.any(trace["procedural_workspace"] & ~trace["time_out"]):
        raise ValueError("Support workspace censoring requires a native timeout")
    done = trace["terminated"] | trace["time_out"]
    expected_valid = np.concatenate(
        (np.ones_like(done[:1]), ~np.maximum.accumulate(done[:-1], axis=0))
    )
    if not np.array_equal(trace["valid_first_attempt"], expected_valid):
        raise ValueError(
            "Support first-attempt mask differs from native terminal history"
        )
    finite_rays, missing_rays = _rays(trace)
    trials = []
    for env_id in range(count):
        valid = expected_valid[:, env_id]
        eligible = valid & ~done[:, env_id]
        terminals = np.flatnonzero(done[:, env_id])
        terminal = int(terminals[0]) if len(terminals) else None
        invalid = {
            name: int((~np.isfinite(trace[name][valid, env_id]).all(axis=(1, 2))).sum())
            for name in ("foot_link_position_w", "foot_contact_force_w")
        }
        invalid = {name: number for name, number in invalid.items() if number}
        missing = int(missing_rays[eligible, env_id].sum())
        complete = terminal is None and not invalid and missing == 0
        trials.append(
            {
                "env_id": env_id,
                "first_terminal_step": terminal,
                "first_attempt_physical_termination": bool(
                    np.any(trace["terminated"][valid, env_id])
                ),
                "first_attempt_time_out": bool(
                    np.any(trace["time_out"][valid, env_id])
                ),
                "first_attempt_workspace_censoring": bool(
                    np.any(trace["procedural_workspace"][valid, env_id])
                ),
                "first_attempt_samples_including_terminal": int(valid.sum()),
                "eligible_nonterminal_samples": int(eligible.sum()),
                "excluded_terminal_samples": int(np.sum(valid & done[:, env_id])),
                "excluded_post_reset_samples": int((~valid).sum()),
                "nonfinite_first_attempt_physical_fields": invalid,
                "physical_evidence_valid": not invalid,
                "finite_ray_count": int(finite_rays[eligible, env_id].sum()),
                "missing_ray_count": missing,
                "missing_ray_count_by_foot": dict(
                    zip(
                        names.tolist(),
                        missing_rays[eligible, env_id].sum(axis=0).astype(int).tolist(),
                        strict=True,
                    )
                ),
                "complete_first_attempt_capture": complete,
                "evidence_status": (
                    "INVALID_PHYSICAL_EVIDENCE"
                    if invalid
                    else (
                        "TERMINAL_CENSORED_CAPTURE"
                        if terminal is not None
                        else (
                            "MISSING_RAY_EVIDENCE"
                            if missing
                            else "CAPTURE_COMPLETE_NOT_SCORED"
                        )
                    )
                ),
            }
        )
    return {
        "version": SUPPORT_CAPTURE_MANIFEST["version"],
        "status": "NOT_SCORED",
        "policy_acceptance": False,
        "qualification_evidence": False,
        "scope": SUPPORT_CAPTURE_MANIFEST["scope"],
        "foot_geometry_binding_sha256": digest,
        "captured_control_steps": steps,
        "total": count,
        "capture_schema_complete": True,
        "complete_first_attempt_captures": sum(
            t["complete_first_attempt_capture"] for t in trials
        ),
        "invalid_physical_evidence_trials": sum(
            not t["physical_evidence_valid"] for t in trials
        ),
        "missing_ray_count": sum(t["missing_ray_count"] for t in trials),
        "trials": trials,
    }


def validate_support_geometry_binding(binding, trace):
    """Bind the recorded native geometry receipt to trace bytes and ray bounds.

    This checks internal consistency, not independent USD remeasurement; native
    geometry was inspected by the capture adapter, not by this replay helper.
    All dynamic ray arrays are checked, but no row receives support credit.
    """
    if (
        not isinstance(binding, dict)
        or binding.get("version") != "operator_four_foot_geometry_v1"
    ):
        raise ValueError("Invalid native support geometry binding version")
    names, radii, digest = _static_fields(trace)
    try:
        actual_digest = hashlib.sha256(
            json.dumps(binding, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Support geometry binding must contain finite JSON data"
        ) from exc
    if digest != actual_digest:
        raise ValueError("Support geometry binding SHA256 differs from the trace")
    positions = trace.get("foot_link_position_w")
    if not isinstance(positions, np.ndarray):
        raise ValueError("Invalid support geometry trace positions")
    shape = positions.shape
    if len(shape) != 4 or shape[2:] != (4, 3) or min(shape[:2]) <= 0:
        raise ValueError("Invalid support geometry trace dimensions")
    for name in _FOOT_VECTORS:
        _array(trace, name, shape, "f")
    if (
        binding.get("foot_names") != names.tolist()
        or type(binding.get("validated_instances")) is not int
        or binding["validated_instances"] != shape[1]
    ):
        raise ValueError("Support geometry binding names or validated instances differ")
    bound_radii = binding.get("foot_collision_radius_m")
    if not isinstance(bound_radii, list) or len(bound_radii) != 4:
        raise ValueError("Invalid support geometry binding radii")
    bound_radii = np.asarray(
        [_finite_number(value, "foot radius") for value in bound_radii]
    )
    if not np.array_equal(bound_radii, radii):
        raise ValueError("Support geometry binding radii differ from trace")
    margin = _finite_number(
        binding.get("maximum_native_robot_contact_offset_m"), "contact offset"
    )
    if margin < 0:
        raise ValueError("Support native contact offset must be nonnegative")
    evidence = binding.get("colliders")
    if not isinstance(evidence, list) or len(evidence) != 4:
        raise ValueError("Support geometry requires four named collider receipts")
    for foot_id, foot in enumerate(evidence):
        if not isinstance(foot, dict) or foot.get("name") != names[foot_id]:
            raise ValueError("Support collider receipt foot order differs")
        colliders = foot.get("colliders")
        if not isinstance(colliders, list) or not colliders:
            raise ValueError("Support foot requires enabled measured colliders")
        paths, measured = set(), 0.0
        for collider in colliders:
            if not isinstance(collider, dict):
                raise ValueError("Invalid support collider receipt")
            path = collider.get("path_relative_to_link")
            if (
                not isinstance(path, str)
                or (path and not path.startswith("/"))
                or path in paths
                or ".." in path.split("/")
            ):
                raise ValueError("Invalid or duplicate relative support collider path")
            paths.add(path)
            kind = collider.get("type")
            if kind not in ("Sphere", "Capsule", "Cylinder", "Cube", "Mesh") or (
                kind == "Mesh"
                and collider.get("approximation") not in (None, "none", "convexHull")
            ):
                raise ValueError("Unsupported support collider geometry")
            try:
                corners = np.asarray(collider.get("link_frame_extent_corners_m"))
            except (TypeError, ValueError) as exc:
                raise ValueError("Invalid measured support collider corners") from exc
            if (
                corners.shape != (8, 3)
                or corners.dtype.kind not in "fi"
                or not np.isfinite(corners).all()
            ):
                raise ValueError("Invalid measured support collider corners")
            measured = max(
                measured,
                float(np.linalg.norm(corners.astype(np.float64), axis=-1).max()),
            )
        recorded = _finite_number(foot.get("geometry_radius_m"), "geometry radius")
        # The recorder validates cloned local geometry to 1e-7 and retains the
        # largest radius across all clones, while storing first-clone corners.
        if (
            recorded <= 0
            or recorded < measured - 1e-12
            or not np.isclose(recorded, measured, atol=2e-7, rtol=0)
        ):
            raise ValueError(
                "Support collider radius does not enclose measured corners"
            )
        if not np.isclose(recorded + margin, radii[foot_id], atol=1e-12, rtol=0):
            raise ValueError(
                "Support collision radius omits geometry or native contact offset"
            )
    mesh = binding.get("mesh")
    if (
        not isinstance(mesh, dict)
        or not isinstance(mesh.get("prim_path"), str)
        or not mesh["prim_path"].startswith("/")
        or mesh.get("collision_enabled") is not True
    ):
        raise ValueError("Invalid collidable support mesh receipt")
    if (
        type(mesh.get("vertices")) is not int
        or mesh["vertices"] < 3
        or type(mesh.get("indices")) is not int
        or mesh["indices"] < 3
        or mesh["indices"] % 3
    ):
        raise ValueError("Invalid support triangle mesh counts")
    _sha256(mesh.get("sha256"), "mesh.sha256")
    _sha256(binding.get("mesh_world_sha256"), "mesh_world_sha256")
    start = _finite_number(binding.get("ray_start_world_z_m"), "ray start")
    distance = _finite_number(binding.get("ray_max_distance_m"), "ray distance")
    if distance < 2.0:
        raise ValueError("Support ray bounds omit the native mesh envelope margins")
    for name in ("radius_method", "scope"):
        if not isinstance(binding.get(name), str) or not binding[name].strip():
            raise ValueError(f"Missing support geometry declaration: {name}")
    finite, _ = _rays(trace)
    z = trace["foot_ground_hit_w"][finite, 2].astype(np.float64)
    tolerance = max(
        2e-5, 4 * np.finfo(np.float32).eps * max(abs(start), abs(start - distance))
    )
    # Adapter starts 1 m above the highest mesh point and extends 1 m below
    # the lowest. Every hit must be within that declared mesh envelope.
    if np.any(z > start - 1.0 + tolerance) or np.any(
        z < start - distance + 1.0 - tolerance
    ):
        raise ValueError(
            "Support mesh-ray hits lie outside recorded geometry ray bounds"
        )
