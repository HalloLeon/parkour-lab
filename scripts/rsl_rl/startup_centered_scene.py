"""Opt-in world centering for a bounded startup action-replay diagnostic.

Importing this module does not import Isaac Lab, NumPy, Torch or the simulator.
The caller must separately enforce the action-replay-only CLI contract. This is
not a training preset: the generated class and its evidence belong to one
configuration in the current process.
"""

from __future__ import annotations

import hashlib
import json
import math


CENTERING_TOLERANCE_M = 1e-9


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mesh_hash(vertices, faces):
    """Hash canonical arrays with explicit shapes/dtypes, not mesh caches."""
    import numpy as np

    digest = hashlib.sha256()
    for name, values, dtype in (("vertices", vertices, "<f8"), ("faces", faces, "<i8")):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(
            json.dumps([name, list(array.shape), array.dtype.str]).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(array.tobytes())
    return digest.hexdigest()


def _centered_generator_class(base_class, selected_row, evidence):
    """Keep the configured generation algorithm, then translate its outputs."""

    class CenteredStartupTerrainGenerator(base_class):
        def __init__(self, cfg, device="cpu"):
            _require(
                not evidence["applied"],
                "A centered startup configuration may generate its scene only once.",
            )
            super().__init__(cfg=cfg, device=device)
            # Import only after the real generator has finished its ordinary
            # construction, cache handling, coloring and global centering.
            import numpy as np

            origins = np.asarray(self.terrain_origins)
            _require(
                origins.ndim == 3
                and origins.shape == (cfg.num_rows, 1, 3)
                and origins.dtype.kind == "f"
                and np.isfinite(origins).all(),
                "Generated terrain origins must be a finite floating-point (rows, 1, 3) array.",
            )
            source_origin = origins[selected_row, 0].copy()
            translation = -source_origin
            centered_origins = origins + translation
            centered_origin = centered_origins[selected_row, 0]
            _require(
                np.allclose(centered_origin, 0.0, rtol=0, atol=CENTERING_TOLERANCE_M),
                "Selected terrain origin did not center at world zero.",
            )

            mesh = self.terrain_mesh
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)
            _require(
                vertices.ndim == 2
                and vertices.shape[1] == 3
                and len(vertices) > 0
                and np.isfinite(vertices).all(),
                "Generated terrain mesh must have finite (vertices, 3) coordinates.",
            )
            _require(
                faces.ndim == 2
                and faces.shape[1] == 3
                and faces.dtype.kind in "iu"
                and len(faces) > 0
                and faces.min() >= 0
                and faces.max() < len(vertices),
                "Generated terrain mesh must have valid triangular face indices.",
            )
            # Work on the final combined mesh only. Neither source submeshes nor
            # on-disk cache entries may be translated by this diagnostic.
            centered_mesh = mesh.copy()
            centered_mesh.apply_translation(translation)
            centered_vertices = np.asarray(centered_mesh.vertices)
            _require(
                centered_vertices.shape == vertices.shape
                and np.isfinite(centered_vertices).all()
                and np.array_equal(centered_mesh.faces, mesh.faces),
                "Centering unexpectedly changed mesh topology or produced nonfinite geometry.",
            )
            local_geometry_error = float(
                np.max(
                    np.abs(
                        (centered_vertices - centered_origin)
                        - (vertices - source_origin)
                    )
                )
            )
            _require(
                math.isfinite(local_geometry_error)
                and local_geometry_error <= CENTERING_TOLERANCE_M,
                "Centering did not preserve terrain geometry relative to the selected origin.",
            )

            _require(
                isinstance(self.flat_patches, dict),
                "Generated flat_patches must be a dictionary.",
            )
            centered_patches = {}
            if self.flat_patches:
                import torch

                for name, patch in self.flat_patches.items():
                    _require(
                        isinstance(name, str)
                        and isinstance(patch, torch.Tensor)
                        and patch.is_floating_point()
                        and patch.ndim == 4
                        and tuple(patch.shape[:2]) == (cfg.num_rows, 1)
                        and patch.shape[-1] == 3
                        and torch.isfinite(patch).all().item(),
                        "Flat patches must be named finite floating-point (rows, 1, patches, 3) tensors.",
                    )
                    shifted = patch + patch.new_tensor(translation)
                    _require(
                        torch.isfinite(shifted).all().item(),
                        "Centered flat patches are nonfinite.",
                    )
                    centered_patches[name] = shifted

            # Commit the three mutually dependent outputs together, after all
            # validation. TerrainImporter consumes precisely these outputs.
            self.terrain_mesh = centered_mesh
            self.terrain_origins = centered_origins
            self.flat_patches = centered_patches
            evidence.update(
                applied=True,
                source_origin_w_m=source_origin.tolist(),
                translation_w_m=translation.tolist(),
                centered_origin_w_m=centered_origin.tolist(),
                local_geometry_max_abs_error_m=local_geometry_error,
                mesh_vertex_count=int(len(centered_vertices)),
                mesh_face_count=int(len(centered_mesh.faces)),
                source_mesh_sha256=_mesh_hash(vertices, faces),
                centered_mesh_sha256=_mesh_hash(centered_vertices, centered_mesh.faces),
                translated_flat_patch_names=sorted(centered_patches),
            )

    return CenteredStartupTerrainGenerator


def configure_centered_startup_scene(env_cfg):
    """Install replay-only centering after fixed-course configuration.

    Returns a mutable JSON-compatible evidence dictionary. Keep this object:
    the generated class fills its *actual* origin and translation after scene
    generation, even if the environment copies the configuration. ``applied``
    stays false until all mesh, origin and patch validation has succeeded.

    No robot state, actuator, policy, reset event or random generator is changed.
    Translating terrain before construction does change which course surrounds
    world zero during simulator initialization. Thus this is a centered-world
    scene intervention, not proof of translation invariance or robot readiness.
    """
    scene = getattr(env_cfg, "scene", None)
    ground = getattr(scene, "ground", None)
    generator_cfg = getattr(ground, "terrain_generator", None)
    level = getattr(env_cfg, "evaluation_level", None)
    _require(
        type(getattr(scene, "num_envs", None)) is int
        and scene.num_envs == 1
        and getattr(env_cfg, "evaluation_family", None) == "high_step"
        and type(level) is int
        and level in (0, 6)
        and type(getattr(env_cfg, "evaluation_geometry_variant", None)) is int
        and env_cfg.evaluation_geometry_variant == 0
        and getattr(env_cfg, "evaluation_command_profile", None) == "translation_only"
        and getattr(env_cfg, "curriculum", None) is None,
        "Centered startup replay requires one fixed high_step environment, level 0 or 6, variant 0, translation_only.",
    )
    _require(
        getattr(ground, "terrain_type", None) == "generator"
        and getattr(ground, "use_terrain_origins", None) is True
        and generator_cfg is not None
        and getattr(generator_cfg, "curriculum", None) is True
        and type(getattr(generator_cfg, "num_rows", None)) is int
        and generator_cfg.num_rows > level
        and type(getattr(generator_cfg, "num_cols", None)) is int
        and generator_cfg.num_cols == 1,
        "Centered startup replay requires a curriculum-generated single-column terrain with terrain origins enabled.",
    )
    base_class = getattr(generator_cfg, "class_type", None)
    if base_class is None:
        from isaaclab.terrains import TerrainGenerator

        base_class = TerrainGenerator
    _require(
        isinstance(base_class, type), "Terrain generator class_type must be a class."
    )
    _require(
        not getattr(base_class, "_startup_centered_generator", False),
        "Centered startup scene configuration must not be applied twice.",
    )
    robot_init = getattr(getattr(scene, "robot", None), "init_state", None)
    construction_position = getattr(robot_init, "pos", None)
    _require(
        isinstance(construction_position, (tuple, list))
        and len(construction_position) == 3
        and all(
            type(value) in (int, float) and math.isfinite(value)
            for value in construction_position
        ),
        "Centered startup replay requires a finite robot construction position.",
    )
    construction_rotation = getattr(robot_init, "rot", None)
    if construction_rotation is not None:
        _require(
            isinstance(construction_rotation, (tuple, list))
            and len(construction_rotation) == 4
            and all(
                type(value) in (int, float) and math.isfinite(value)
                for value in construction_rotation
            ),
            "Robot construction orientation must be a finite wxyz quaternion.",
        )
    evidence = {
        "kind": "startup_centered_terrain",
        "schema_version": 1,
        "configured": True,
        "applied": False,
        "selected_row": level,
        "selected_column": 0,
        "original_generator_class": f"{base_class.__module__}:{base_class.__qualname__}",
        "centering_abs_tolerance_m": CENTERING_TOLERANCE_M,
        "construction_robot_position_w_m": list(construction_position),
        "construction_robot_orientation_wxyz": (
            list(construction_rotation)
            if construction_rotation is not None
            else {
                "status": "UNAVAILABLE",
                "reason": "Robot config does not expose init_state.rot.",
            }
        ),
        "construction_pose_basis": "Configured robot init_state at the single-environment GridCloner world origin; not a measured reset pose.",
        "limitations": [
            "Replay-only scene intervention; not ordinary policy evaluation or a training preset.",
            "Whole terrain, all origins and flat patches are translated; local course geometry is preserved.",
            "The course surrounding world zero during simulator initialization changes.",
            "This jointly changes scene placement, initialization context and mesh cooking coordinates; it is not an isolated coordinate-precision test.",
            "Matching or diverging replay outcomes alone do not establish a simulator bug or robot readiness.",
        ],
    }
    generator_class = _centered_generator_class(base_class, level, evidence)
    generator_class._startup_centered_generator = True
    generator_cfg.class_type = generator_class
    return evidence
