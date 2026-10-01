"""Native USD collider geometry diagnostics, without robot or contact claims.

Importing this module does not launch or import Isaac Sim. The scene is created
after the CLI's AppLauncher and uses the CLI's normal provenance and cleanup.
"""

from __future__ import annotations

import numpy as np

from parkour_lab.environments.structures import MINIMUM_RMS, build_structure
from parkour_lab.provenance import dependency_identity, file_sha256, write_json

SCOPE = (
    "Separate diagnostic USD triangle colliders; not PhysX contact-response, "
    "connected mixed-world, traversal, or robot qualification evidence"
)


def _progress(message):
    try:
        print(f"[geometry] {message}", flush=True)
    except (OSError, ValueError):
        pass  # Console failure must not prevent native cleanup or report writes.


def diagnostic_fixtures():
    """Fixed public diagnostic inputs; not a development/qualification bank."""
    return [
        dict(family=family, tier=tier, reverse=reverse, coarse_seed=17, fine_seed=23)
        for family, tiers in (
            ("stairs", (0.04, 0.08, 0.12, 0.16)),
            ("ramp", (10, 15, 20)),
            ("hill", (10, 15, 20)),
            ("level", (0.01, 0.02)),
        )
        for tier in tiers
        for reverse in ((False,) if family == "level" else (False, True))
    ]


def validate_readback(source, readback, translation):
    """Measure actual float32 USD points; reject identity or envelope changes.

    Source carrier/layer witnesses remain labelled as witnesses. Quantization
    error bounds their relationship to the final residual; separate roughness
    layers cannot be uniquely recovered from the summed imported heights.
    """
    points = np.asarray(readback["points"])
    indices = np.asarray(readback["face_indices"])
    counts = np.asarray(readback["face_counts"])
    transform = np.asarray(readback["transform"])
    expected_transform = np.eye(4)
    # USD/Gf uses row-vector matrices; translations occupy the last row.
    expected_transform[3, :3] = translation
    checks = {
        "points differ from the exact float32 source conversion": (
            points.dtype == np.float32
            and np.isfinite(points).all()
            and np.array_equal(points, source["vertices"].astype(np.float32))
        ),
        "triangle topology changed": (
            np.issubdtype(indices.dtype, np.integer)
            and np.issubdtype(counts.dtype, np.integer)
            and np.array_equal(indices, source["faces"].ravel())
            and np.array_equal(counts, np.full(len(source["faces"]), 3))
        ),
        "unexpected local-to-world transform": np.array_equal(
            transform, expected_transform
        ),
        "triangle collider is absent, disabled, or approximated": (
            readback["collision_api"] is True
            and readback["collision_enabled"] is True
            and readback["approximation"] == "none"
        ),
    }
    failures = [message for message, passed in checks.items() if not passed]
    result = {"valid": False, "failures": failures}
    if failures:
        return result

    vertices = points.astype(np.float64)
    triangles = vertices[indices.reshape(-1, 3)]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    support = source["support_faces"]
    if (
        (np.linalg.norm(normals, axis=1) == 0).any()
        or (normals[support, 2] <= 0).any()
        or (normals[~support, 2] != 0).any()
    ):
        failures.append("degenerate or incorrectly oriented supporting/riser face")
        return result
    grades = np.degrees(
        np.arctan2(np.linalg.norm(normals[support, :2], axis=1), normals[support, 2])
    )
    grid = vertices.reshape(len(source["x"]), len(source["y"]), 3)
    x, y = grid[:, 0, 0], grid[0, :, 1]
    dx, dy = np.diff(x), np.diff(y)
    lips = np.flatnonzero(dx == 0)
    rises = np.abs(grid[lips + 1, :, 2] - grid[lips, :, 2])
    treads = np.diff(x[lips])
    carrier = source["carrier_heights"].ravel()
    residual = vertices[:, 2] - carrier
    height_error = vertices[:, 2] - source["vertices"][:, 2]
    samples = source["untapered_samples"].ravel()
    conversion_bound = float(np.max(np.abs(height_error)))
    layer_rms = [
        float(np.sqrt(np.mean(source[name].ravel()[samples] ** 2)))
        for name in ("coarse_heights", "fine_heights")
    ]
    metadata = source["metadata"]
    # Carrier geometry is an explicit witness, not a claimed decomposition
    # measured independently in the final, roughened USD collider.
    carrier_vertices = vertices.copy()
    carrier_vertices[:, 2] = carrier.astype(np.float32).astype(np.float64)
    carrier_triangles = carrier_vertices[source["faces"][support]]
    carrier_normals = np.cross(
        carrier_triangles[:, 1] - carrier_triangles[:, 0],
        carrier_triangles[:, 2] - carrier_triangles[:, 0],
    )
    carrier_grade = float(
        np.degrees(
            np.arctan2(
                np.linalg.norm(carrier_normals[:, :2], axis=1), carrier_normals[:, 2]
            )
        ).max()
    )
    measures = {
        "vertex_count": len(vertices),
        "triangle_count": len(normals),
        "size_m": [float(np.ptp(x)), float(np.ptp(y))],
        "maximum_axis_spacing_m": float(max(dx.max(), dy.max())),
        "maximum_supporting_grade_degrees": float(grades.max()),
        "projected_support_area_m2": float(normals[support, 2].sum() / 2),
        "riser_height_ranges_m": (
            np.column_stack((rises.min(axis=1), rises.max(axis=1))).tolist()
            if len(rises)
            else []
        ),
        "tread_lengths_m": treads.tolist(),
        "maximum_residual_from_source_carrier_m": float(np.abs(residual).max()),
        "residual_rms_from_source_carrier_m": float(
            np.sqrt(np.mean(residual[samples] ** 2))
        ),
        "maximum_height_conversion_error_m": conversion_bound,
        "source_layer_rms_witness_m": layer_rms,
        "source_layer_rms_minus_conversion_bound_m": [
            rms - conversion_bound for rms in layer_rms
        ],
        "float32_carrier_witness_grade_degrees": carrier_grade,
        "roughness_scope": (
            "Final USD residual measured against archived source carrier; separate "
            "layer RMS values are source witnesses, not native layer measurements"
        ),
    }
    limits = {
        "supporting grade exceeds 25 degrees": grades.max() <= 25.0,
        "axis spacing exceeds 0.02 m": max(dx.max(), dy.max()) <= 0.02,
        "support width is below 2 m": np.ptp(y) >= 2.0,
        "roughness residual exceeds its cap": np.abs(residual).max()
        <= metadata["roughness_cap_m"],
        "roughness witness loses minimum RMS under conversion uncertainty": all(
            rms - conversion_bound >= minimum
            for rms, minimum in zip(layer_rms, MINIMUM_RMS, strict=True)
        ),
    }
    family, tier = metadata["family"], metadata["tier"]
    if family == "stairs":
        limits.update(
            {
                "riser heights outside frozen tier tolerance": bool(
                    (rises >= tier - 0.002).all() and (rises <= tier).all()
                ),
                "tread lengths outside [0.31,0.50] m": bool(
                    (treads >= 0.31).all() and (treads <= 0.5).all()
                ),
            }
        )
    elif family in ("ramp", "hill"):
        limits["carrier witness grade outside frozen tier tolerance"] = (
            tier - 0.2 <= carrier_grade <= tier
        )
    failures.extend(message for message, passed in limits.items() if not passed)
    result.update(valid=not failures, measurements=measures)
    return result


class GeometryScene:
    """Own only the native simulation scene; the CLI owns the application."""

    def __init__(self, device):
        _progress("Creating simulation context")
        from isaaclab.sim import SimulationCfg, SimulationContext

        try:
            self.sim = SimulationContext(SimulationCfg(device=device, dt=0.005))
        except BaseException:
            partial = SimulationContext.instance()
            if partial is not None:
                try:
                    partial.clear_all_callbacks()
                finally:
                    SimulationContext.clear_instance()
            raise
        _progress("Simulation context ready")

    def _import(self, path, source, translation):
        import trimesh
        from pxr import UsdPhysics

        from isaaclab.terrains.utils import create_prim_from_mesh

        create_prim_from_mesh(
            path,
            trimesh.Trimesh(
                vertices=source["vertices"], faces=source["faces"], process=False
            ),
            translation=translation,
        )
        prim = self.sim.stage.GetPrimAtPath(path + "/mesh")
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("none")

    def _readback(self, path):
        from pxr import UsdGeom, UsdPhysics

        prim = self.sim.stage.GetPrimAtPath(path + "/mesh")
        mesh = UsdGeom.Mesh(prim)
        return {
            "points": np.asarray(mesh.GetPointsAttr().Get()),
            "face_indices": np.asarray(mesh.GetFaceVertexIndicesAttr().Get()),
            "face_counts": np.asarray(mesh.GetFaceVertexCountsAttr().Get()),
            "transform": np.asarray(
                UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            ),
            "collision_api": bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
            "collision_enabled": (
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
            ),
            "approximation": str(
                UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            ),
        }

    def validate(self, output, report):
        # These imports happen only after AppLauncher; no learner dependency is
        # involved in measuring the SDK importer and mesh-conversion toolchain.
        _progress("Recording native dependency identities")
        report["dependencies"] = {
            name: dependency_identity(name, name)
            for name in ("isaaclab", "numpy", "trimesh")
        }
        fixtures = diagnostic_fixtures()
        report["geometry"] = {
            "scope": SCOPE,
            "fixture_inputs": fixtures,
            "expected_fixture_count": len(fixtures),
            "post_reset_physics_steps": 1,
            "native_sim_version": self.sim.get_version(),
            "fixtures": [],
        }
        sources = []
        for index, fixture in enumerate(fixtures):
            _progress(
                f"Importing fixture {index + 1}/{len(fixtures)}: "
                f"{fixture['family']} tier={fixture['tier']} reverse={fixture['reverse']}"
            )
            source = build_structure(**fixture)
            path = f"/World/Geometry/fixture_{index:02d}"
            translation = (0.0, 4.0 * index, 0.0)
            self._import(path, source, translation)
            sources.append((path, translation, source))
        _progress("All meshes imported; resetting physics")
        self.sim.reset()
        _progress("Physics reset complete; stepping once")
        self.sim.step(render=False)
        _progress("Physics step complete; reading back and validating meshes")
        for index, (path, translation, source) in enumerate(sources):
            readback = self._readback(path)
            evidence = output / f"mesh_{index:02d}.npz"
            np.savez_compressed(
                evidence,
                **{
                    f"source_{name}": value
                    for name, value in source.items()
                    if name != "metadata"
                },
                **{f"imported_{name}": value for name, value in readback.items()},
            )
            receipt = {
                "prim_path": path + "/mesh",
                "translation_m": list(translation),
                "source_geometry": source["metadata"],
                "evidence_file": evidence.name,
                "evidence_sha256": file_sha256(evidence),
                **validate_readback(source, readback, translation),
            }
            report["geometry"]["fixtures"].append(receipt)
            write_json(output / "report.json", report)
            _progress(
                f"Fixture {index + 1}/{len(fixtures)}: "
                f"{'valid' if receipt['valid'] else 'INVALID'}; evidence saved"
            )
        if not all(item["valid"] for item in report["geometry"]["fixtures"]):
            raise ValueError("Native USD geometry failed one or more frozen limits")
        report["status"] = "NATIVE_GEOMETRY_VALIDATED_NOT_QUALIFIED"
        _progress("All fixtures validated; starting cleanup")

    def close(self):
        # Match the standard native environment lifecycle. Calling stop() here
        # invokes Isaac Lab's STOP callback, which waits for playback to resume.
        # clear_instance() unsubscribes it before the CLI closes the application.
        _progress("Clearing simulation callbacks")
        try:
            self.sim.clear_all_callbacks()
        finally:
            self.sim.clear_instance()
        _progress("Simulation context released; application close follows")
