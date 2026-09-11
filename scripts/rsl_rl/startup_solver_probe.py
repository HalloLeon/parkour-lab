"""Opt-in, solver-only startup diagnostic; never changes simulator state.

Configure before constructing the environment. The runtime readback uses the
Isaac Sim 5.1 SimulationManager getter and constructed USD schema attributes.
Those report the selected configuration, not the GPU's internal execution.
No simulator dependency is imported until runtime readback is requested.
"""

from __future__ import annotations

import copy
import json
import math


_SCENE_ITERATION_GETTERS = {
    "min_position_iteration_count": "GetMinPositionIterationCountAttr",
    "max_position_iteration_count": "GetMaxPositionIterationCountAttr",
    "min_velocity_iteration_count": "GetMinVelocityIterationCountAttr",
    "max_velocity_iteration_count": "GetMaxVelocityIterationCountAttr",
}
_READBACK_PROVENANCE = (
    "SimulationManager getter and constructed USD schemas; "
    "not direct GPU solver execution evidence."
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, expected):
    return type(value) is int and value == expected


def _iteration_bounds(value):
    _require(
        isinstance(value, dict)
        and set(value) == set(_SCENE_ITERATION_GETTERS)
        and all(type(item) is int for item in value.values()),
        "Scene iteration bounds must contain the four integer limits",
    )
    _require(
        1 <= value["min_position_iteration_count"] <= 4
        and 4 <= value["max_position_iteration_count"] <= 255
        and value["min_velocity_iteration_count"] == 0
        and 0 <= value["max_velocity_iteration_count"] <= 255,
        "Scene iteration bounds must preserve the articulation's 4/0 request",
    )


def validate_solver_metadata(metadata, *, require_readback=True):
    """Validate the explicit diagnostic scope; this does not qualify a robot."""
    _require(isinstance(metadata, dict), "Solver probe metadata must be a dictionary")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Solver probe metadata must be finite JSON data") from error
    _require(
        metadata.get("kind") == "startup_solver_probe"
        and _integer(metadata.get("schema_version"), 1)
        and metadata.get("source_solver") == "TGS"
        and _integer(metadata.get("source_solver_type"), 1)
        and metadata.get("requested_solver") in ("TGS", "PGS")
        and _integer(
            metadata.get("selected_solver_type"),
            1 if metadata.get("requested_solver") == "TGS" else 0,
        )
        and metadata.get("configured_before_scene") is True
        and type(metadata.get("readback_verified")) is bool,
        "Invalid or unsupported solver probe selection/provenance",
    )
    dt = metadata.get("physics_dt_s")
    _require(
        type(dt) in (int, float)
        and math.isfinite(dt)
        and dt == 0.005
        and _integer(metadata.get("control_decimation"), 4)
        and _integer(metadata.get("articulation_position_iterations"), 4)
        and _integer(metadata.get("articulation_velocity_iterations"), 0),
        "Solver probe must preserve 0.005s physics, decimation 4, and articulation 4/0",
    )
    path = metadata.get("physics_scene_path")
    _require(
        isinstance(path, str) and path.startswith("/") and len(path) > 1,
        "Solver probe requires an absolute physics scene prim path",
    )
    _iteration_bounds(metadata.get("scene_iteration_bounds"))
    if not metadata["readback_verified"]:
        _require(
            metadata.get("readback") is None,
            "Unverified probe cannot carry verified readback",
        )
        _require(not require_readback, "Solver probe readback has not been verified")
        return
    readback = metadata.get("readback")
    _require(
        isinstance(readback, dict), "Verified solver probe lacks readback evidence"
    )
    _iteration_bounds(readback.get("scene_iteration_bounds"))
    _require(
        readback.get("physics_scene_path") == path
        and readback.get("simulation_manager_solver_type")
        == metadata["requested_solver"]
        and readback.get("usd_scene_solver_type") == metadata["requested_solver"]
        and readback.get("scene_iteration_bounds") == metadata["scene_iteration_bounds"]
        and _integer(readback.get("articulation_position_iterations"), 4)
        and _integer(readback.get("articulation_velocity_iterations"), 0)
        and readback.get("provenance") == _READBACK_PROVENANCE,
        "Solver or iteration readback does not match the declared diagnostic",
    )
    root_path = readback.get("articulation_prim_path")
    _require(
        isinstance(root_path, str) and root_path.startswith("/") and len(root_path) > 1,
        "Verified solver probe lacks the actual articulation prim path",
    )


def configure_solver_probe(env_cfg, requested="pgs"):
    """Change only ``solver_type`` before scene construction, or observe TGS.

    The caller owns the opt-in CLI and fixed-course/replay scope. Calling this
    after construction cannot change a running PhysX scene and is unsupported.
    """
    _require(requested in ("tgs", "pgs"), "Requested solver must be 'tgs' or 'pgs'")
    _require(
        _integer(env_cfg.scene.num_envs, 1),
        "Solver probe requires exactly one environment",
    )
    sim = env_cfg.sim
    props = env_cfg.scene.robot.spawn.articulation_props
    _require(_integer(sim.physx.solver_type, 1), "Solver probe source must be TGS (1)")
    metadata = {
        "kind": "startup_solver_probe",
        "schema_version": 1,
        "source_solver": "TGS",
        "requested_solver": requested.upper(),
        "source_solver_type": 1,
        "selected_solver_type": 1 if requested == "tgs" else 0,
        "configured_before_scene": True,
        "readback_verified": False,
        "physics_scene_path": sim.physics_prim_path,
        "physics_dt_s": sim.dt,
        "control_decimation": env_cfg.decimation,
        "articulation_position_iterations": props.solver_position_iteration_count,
        "articulation_velocity_iterations": props.solver_velocity_iteration_count,
        "scene_iteration_bounds": {
            name: getattr(sim.physx, name) for name in _SCENE_ITERATION_GETTERS
        },
        "readback": None,
        "limitations": [
            "Diagnostic solver selection is not a production fix or robot acceptance.",
            "Readback verifies API/USD selection, not GPU instruction execution or solver accuracy.",
            "Articulation and scene iteration attributes are requests/bounds, not measured per-island iteration counts.",
        ],
    }
    validate_solver_metadata(metadata, require_readback=False)
    # Commit only after all validation: no gain, limit, reset, timing, scene, or
    # additional solver-setting change is hidden in this opt-in helper.
    sim.physx.solver_type = metadata["selected_solver_type"]
    return metadata


def validate_solver_probe_readback(
    env, metadata, *, simulation_manager=None, physx_schema=None
):
    """Read constructed scene/articulation settings without writes or steps.

    The optional dependency arguments are narrow CPU-test seams. Production
    uses Isaac Sim's documented SimulationManager and pxr.PhysxSchema APIs.
    Metadata is updated in place only after successful readback; an attempted
    revalidation first invalidates any old readback to avoid stale evidence.
    """
    _require(isinstance(metadata, dict), "Solver probe metadata must be a dictionary")
    metadata["readback_verified"] = False
    metadata["readback"] = None
    validate_solver_metadata(metadata, require_readback=False)
    _require(_integer(env.num_envs, 1), "Solver readback requires one environment")
    cfg = env.cfg
    props = cfg.scene.robot.spawn.articulation_props
    _require(
        cfg.sim.physics_prim_path == metadata["physics_scene_path"]
        and _integer(cfg.sim.physx.solver_type, metadata["selected_solver_type"])
        and cfg.sim.dt == metadata["physics_dt_s"]
        and _integer(cfg.decimation, metadata["control_decimation"])
        and _integer(props.solver_position_iteration_count, 4)
        and _integer(props.solver_velocity_iteration_count, 0)
        and all(
            _integer(getattr(cfg.sim.physx, name), value)
            for name, value in metadata["scene_iteration_bounds"].items()
        ),
        "Runtime configuration changed after solver probe configuration",
    )
    if simulation_manager is None:
        from isaacsim.core.simulation_manager import SimulationManager

        simulation_manager = SimulationManager
    if physx_schema is None:
        from pxr import PhysxSchema

        physx_schema = PhysxSchema
    path = metadata["physics_scene_path"]
    try:
        manager_solver = simulation_manager.get_solver_type(physics_scene=path)
        stage = env.scene.stage
        scene_prim = stage.GetPrimAtPath(path)
        _require(
            scene_prim.IsValid() and scene_prim.HasAPI(physx_schema.PhysxSceneAPI),
            "Constructed physics scene is missing PhysxSceneAPI",
        )
        scene_api = physx_schema.PhysxSceneAPI(scene_prim)
        scene_solver = scene_api.GetSolverTypeAttr().Get()
        bounds = {
            name: getattr(scene_api, method)().Get()
            for name, method in _SCENE_ITERATION_GETTERS.items()
        }
        paths = list(env.scene["robot"].root_physx_view.prim_paths)
        _require(len(paths) == 1, "Expected one live articulation prim path")
        root_path = paths[0]
        _require(
            isinstance(root_path, str) and root_path.startswith("/"),
            "Live articulation prim path must be absolute",
        )
        root_prim = stage.GetPrimAtPath(root_path)
        _require(
            root_prim.IsValid() and root_prim.HasAPI(physx_schema.PhysxArticulationAPI),
            "Live articulation root is missing PhysxArticulationAPI",
        )
        articulation_api = physx_schema.PhysxArticulationAPI(root_prim)
        position_iterations = (
            articulation_api.GetSolverPositionIterationCountAttr().Get()
        )
        velocity_iterations = (
            articulation_api.GetSolverVelocityIterationCountAttr().Get()
        )
    except (AttributeError, ImportError, RuntimeError) as error:
        raise ValueError(
            "Solver readback unavailable; abort this diagnostic"
        ) from error
    candidate = copy.deepcopy(metadata)
    candidate["readback"] = {
        "physics_scene_path": path,
        "simulation_manager_solver_type": manager_solver,
        "usd_scene_solver_type": scene_solver,
        "articulation_prim_path": root_path,
        "scene_iteration_bounds": bounds,
        "articulation_position_iterations": position_iterations,
        "articulation_velocity_iterations": velocity_iterations,
        "provenance": _READBACK_PROVENANCE,
    }
    candidate["readback_verified"] = True
    validate_solver_metadata(candidate)
    metadata.update(candidate)
    return metadata


def validate_solver_environment_physics(source, current, metadata):
    """Permit only explicit TGS→selected-solver change in physics dictionaries.

    Returns None on valid diagnostic evidence, raises ValueError otherwise.
    This deliberately does not weaken generic replay comparisons outside the
    explicit solver-probe path or permit any other physics/profile differences.
    """
    validate_solver_metadata(metadata)
    _require(
        isinstance(source, dict) and isinstance(current, dict),
        "Physics metadata must be dictionaries",
    )
    _require(
        isinstance(source.get("physx"), dict)
        and _integer(source["physx"].get("solver_type"), 1),
        "Solver probe requires source physics with TGS solver_type=1",
    )
    expected = copy.deepcopy(source)
    expected["physx"]["solver_type"] = metadata["selected_solver_type"]
    try:
        encoded_expected = json.dumps(expected, sort_keys=True, allow_nan=False)
        encoded_current = json.dumps(current, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Physics metadata must be finite JSON data") from error
    _require(
        encoded_expected == encoded_current,
        "Solver probe permits only the declared physx.solver_type change",
    )
    _require(
        current.get("physics_prim_path") == metadata["physics_scene_path"]
        and current.get("dt") == metadata["physics_dt_s"]
        and all(
            _integer(current["physx"].get(name), value)
            for name, value in metadata["scene_iteration_bounds"].items()
        ),
        "Solver probe readback is not bound to this physics configuration",
    )
