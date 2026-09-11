"""Opt-in world centering for bounded replay and normal-policy diagnostics.

Importing this module does not import Isaac Lab, NumPy, Torch or the simulator.
The two entry points have separate scope and evidence contracts; the caller
must also enforce the corresponding CLI contract. Neither is a training preset:
the generated class and its evidence belong to one configuration in the current
process.
"""

from __future__ import annotations

import hashlib
import json
import math


CENTERING_TOLERANCE_M = 1e-9
RUNTIME_ORIGIN_TOLERANCE_M = 2e-6


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


def _configure_centered_scene(env_cfg, *, evaluation):
    """Shared generation, with distinct replay and feedback scope contracts."""
    scene = getattr(env_cfg, "scene", None)
    ground = getattr(scene, "ground", None)
    generator_cfg = getattr(ground, "terrain_generator", None)
    level = getattr(env_cfg, "evaluation_level", None)
    _require(
        type(getattr(scene, "num_envs", None)) is int
        and scene.num_envs == 1
        and getattr(env_cfg, "evaluation_family", None) == "high_step"
        and type(level) is int
        and level in ((6,) if evaluation else (0, 6))
        and type(getattr(env_cfg, "evaluation_geometry_variant", None)) is int
        and env_cfg.evaluation_geometry_variant == 0
        and getattr(env_cfg, "evaluation_command_profile", None) == "translation_only"
        and getattr(env_cfg, "curriculum", None) is None,
        (
            "Centered policy evaluation requires one fixed high_step environment, level 6, variant 0, translation_only."
            if evaluation
            else "Centered startup replay requires one fixed high_step environment, level 0 or 6, variant 0, translation_only."
        ),
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
        "Centered scene diagnostics require a curriculum-generated single-column terrain with terrain origins enabled.",
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
        "Centered scene configuration must not be applied twice.",
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
        "Centered scene diagnostics require a finite robot construction position.",
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
        "kind": "evaluation_centered_terrain"
        if evaluation
        else "startup_centered_terrain",
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
    if evaluation:
        evidence.update(
            purpose="normal_policy_evaluation",
            runtime_origin_verified=False,
            runtime_environment_origin_w_m=None,
            runtime_origin_abs_tolerance_m=RUNTIME_ORIGIN_TOLERANCE_M,
        )
        evidence["limitations"][0] = (
            "Normal-policy evaluation with an explicit scene intervention; not a training preset or production fix."
        )
        evidence["limitations"][-1] = (
            "Performance differences alone do not establish a simulator bug or reliable robot operation."
        )
    generator_class = _centered_generator_class(base_class, level, evidence)
    generator_class._startup_centered_generator = True
    generator_cfg.class_type = generator_class
    return evidence


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
    return _configure_centered_scene(env_cfg, evaluation=False)


def configure_centered_evaluation_scene(env_cfg):
    """Install explicit high-step L6 centering for a normal-policy comparison.

    Unlike the replay entry point, this admits only the single fixed L6 course.
    It does not change actions, reset events or control. After scene construction
    and fixed-course initialization, the caller must validate the live origin
    with :func:`validate_centered_evaluation_scene` before policy rollout.

    This intervention changes placement, initialization context and mesh cooking
    coordinates together. It is an experiment, not a production physics fix.
    """
    return _configure_centered_scene(env_cfg, evaluation=True)


def _finite_vector(value, size, field):
    _require(
        isinstance(value, list)
        and len(value) == size
        and all(type(item) in (int, float) and math.isfinite(item) for item in value),
        f"{field} must be a finite {size}-element numeric list.",
    )
    return value


def validate_centered_evaluation_evidence(evidence, require_runtime=True):
    """Fail closed on incomplete or inconsistent captured centering evidence.

    This pure report validator checks generator evidence and, when required,
    the captured live origin. It does not reread the mesh from USD, attest to
    simulator internals, or qualify the robot's behavior.
    """
    _require(
        isinstance(evidence, dict), "Centered evaluation evidence must be a dictionary."
    )
    try:
        json.dumps(evidence, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Centered evaluation evidence must be finite JSON data."
        ) from error
    _require(
        evidence.get("kind") == "evaluation_centered_terrain"
        and evidence.get("purpose") == "normal_policy_evaluation"
        and type(evidence.get("schema_version")) is int
        and evidence["schema_version"] == 1
        and evidence.get("configured") is True
        and evidence.get("applied") is True
        and type(evidence.get("selected_row")) is int
        and evidence["selected_row"] == 6
        and type(evidence.get("selected_column")) is int
        and evidence["selected_column"] == 0,
        "Evidence must describe an applied, fixed high-step L6 centered normal-policy evaluation.",
    )
    _require(
        type(evidence.get("centering_abs_tolerance_m")) in (int, float)
        and evidence["centering_abs_tolerance_m"] == CENTERING_TOLERANCE_M
        and type(evidence.get("runtime_origin_abs_tolerance_m")) in (int, float)
        and evidence["runtime_origin_abs_tolerance_m"] == RUNTIME_ORIGIN_TOLERANCE_M,
        "Centered evaluation evidence must retain the declared geometry and runtime origin tolerances.",
    )
    source = _finite_vector(evidence.get("source_origin_w_m"), 3, "source_origin_w_m")
    shift = _finite_vector(evidence.get("translation_w_m"), 3, "translation_w_m")
    centered = _finite_vector(
        evidence.get("centered_origin_w_m"), 3, "centered_origin_w_m"
    )
    _require(
        max(abs(item) for item in centered) <= CENTERING_TOLERANCE_M
        and max(abs(a + b - c) for a, b, c in zip(source, shift, centered))
        <= CENTERING_TOLERANCE_M,
        "Terrain origin and translation evidence do not consistently center the selected course.",
    )
    error = evidence.get("local_geometry_max_abs_error_m")
    _require(
        type(error) in (int, float) and 0 <= error <= CENTERING_TOLERANCE_M,
        "Centering evidence does not demonstrate local geometry preservation.",
    )
    for key in ("mesh_vertex_count", "mesh_face_count"):
        _require(
            type(evidence.get(key)) is int and evidence[key] > 0,
            f"{key} must be a positive integer.",
        )
    for key in ("source_mesh_sha256", "centered_mesh_sha256"):
        digest = evidence.get(key)
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(c in "0123456789abcdef" for c in digest),
            f"{key} must be a lowercase SHA-256 digest.",
        )
    _require(
        isinstance(evidence.get("original_generator_class"), str)
        and bool(evidence["original_generator_class"]),
        "Original generator class evidence is missing.",
    )
    patches = evidence.get("translated_flat_patch_names")
    _require(
        isinstance(patches, list)
        and all(isinstance(name, str) for name in patches)
        and len(set(patches)) == len(patches),
        "Translated flat patch evidence must contain unique names.",
    )
    _finite_vector(
        evidence.get("construction_robot_position_w_m"),
        3,
        "construction_robot_position_w_m",
    )
    rotation = evidence.get("construction_robot_orientation_wxyz")
    if isinstance(rotation, dict):
        _require(
            rotation.get("status") == "UNAVAILABLE"
            and isinstance(rotation.get("reason"), str)
            and bool(rotation["reason"]),
            "Missing construction orientation needs an explicit unavailable reason.",
        )
    else:
        _finite_vector(rotation, 4, "construction_robot_orientation_wxyz")
    _require(
        isinstance(evidence.get("construction_pose_basis"), str)
        and bool(evidence["construction_pose_basis"])
        and isinstance(evidence.get("limitations"), list)
        and bool(evidence["limitations"])
        and all(
            isinstance(item, str) and bool(item) for item in evidence["limitations"]
        ),
        "Construction pose basis and scene intervention limitations must be recorded.",
    )
    verified = evidence.get("runtime_origin_verified")
    _require(
        type(verified) is bool, "Runtime origin verification status must be explicit."
    )
    if verified:
        runtime = _finite_vector(
            evidence.get("runtime_environment_origin_w_m"),
            3,
            "runtime_environment_origin_w_m",
        )
        _require(
            max(abs(item) for item in runtime) <= RUNTIME_ORIGIN_TOLERANCE_M
            and max(abs(a - b) for a, b in zip(runtime, centered))
            <= RUNTIME_ORIGIN_TOLERANCE_M,
            "The captured environment origin does not match the centered course at world zero.",
        )
    else:
        _require(
            evidence.get("runtime_environment_origin_w_m") is None
            and not require_runtime,
            "Centered evaluation requires a verified live environment origin.",
        )
    return evidence


def validate_centered_evaluation_scene(env, evidence):
    """Validate the generated evidence and capture the live origin, read-only.

    Call after fixed-course initialization, before rollout. Only the evidence
    dictionary is updated; no scene, robot state, reset, step or config is written.
    A failed revalidation clears prior runtime verification in this dictionary.
    """
    _require(
        isinstance(evidence, dict)
        and evidence.get("kind") == "evaluation_centered_terrain"
        and evidence.get("purpose") == "normal_policy_evaluation",
        "Runtime evaluation validation requires evaluation-specific centering evidence.",
    )
    evidence["runtime_origin_verified"] = False
    evidence["runtime_environment_origin_w_m"] = None
    validate_centered_evaluation_evidence(evidence, require_runtime=False)
    cfg = getattr(env, "cfg", None)
    scene_cfg = getattr(cfg, "scene", None)
    _require(
        type(getattr(env, "num_envs", None)) is int
        and env.num_envs == 1
        and type(getattr(scene_cfg, "num_envs", None)) is int
        and scene_cfg.num_envs == 1
        and getattr(cfg, "evaluation_family", None) == "high_step"
        and type(getattr(cfg, "evaluation_level", None)) is int
        and cfg.evaluation_level == 6
        and type(getattr(cfg, "evaluation_geometry_variant", None)) is int
        and cfg.evaluation_geometry_variant == 0
        and getattr(cfg, "evaluation_command_profile", None) == "translation_only"
        and getattr(cfg, "curriculum", None) is None,
        "Live centered policy evaluation must retain the single fixed high-step L6 translation-only scope.",
    )
    import torch

    origins = getattr(getattr(env, "scene", None), "env_origins", None)
    _require(
        isinstance(origins, torch.Tensor)
        and origins.is_floating_point()
        and tuple(origins.shape) == (1, 3)
        and torch.isfinite(origins).all().item(),
        "Live environment origins must be a finite floating-point (1, 3) tensor.",
    )
    runtime_origin = origins.detach().cpu().clone()[0].tolist()
    centered = evidence["centered_origin_w_m"]
    _require(
        max(abs(item) for item in runtime_origin) <= RUNTIME_ORIGIN_TOLERANCE_M
        and max(abs(a - b) for a, b in zip(runtime_origin, centered))
        <= RUNTIME_ORIGIN_TOLERANCE_M,
        "Live environment origin does not match the centered course at world zero.",
    )
    evidence.update(
        runtime_origin_verified=True,
        runtime_environment_origin_w_m=runtime_origin,
    )
    return validate_centered_evaluation_evidence(evidence)
