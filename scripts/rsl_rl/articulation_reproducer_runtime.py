"""One contact-free Go2 physics step, without policy, task or terrain imports.

AppLauncher belongs to the coordinator and must already be running. Prescribed
backend effort bypasses the explicit actuator calculation, without changing the
asset's physical parameters. This is a diagnostic, not an operating controller.
"""

from __future__ import annotations

import importlib.metadata
import math
from types import SimpleNamespace

try:
    from .articulation_reproducer_core import validate_reproducer_source
except ImportError:
    from articulation_reproducer_core import validate_reproducer_source


_CASES = {"origin": (0.0, 0.0), "translated": (24.0, 24.0), "teleported": (0.0, 24.0)}
_PROPERTY_GETTERS = {
    "body_mass_kg": "get_masses",
    "body_inertia_kg_m2": "get_inertias",
    "body_com_pose": "get_coms",
    "shape_material_static_dynamic_restitution": "get_material_properties",
    "joint_armature": "get_dof_armatures",
    "joint_physx_stiffness": "get_dof_stiffnesses",
    "joint_physx_damping": "get_dof_dampings",
    "joint_max_velocity_rad_s": "get_dof_max_velocities",
    "joint_max_force_nm": "get_dof_max_forces",
    "joint_position_limits_rad": "get_dof_limits",
    "joint_friction_static_dynamic_viscous": "get_dof_friction_properties",
    "joint_legacy_friction_coefficient": "get_dof_friction_coefficients",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _load_runtime():
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
    from isaacsim.core.simulation_manager import SimulationManager
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    return SimpleNamespace(
        torch=torch,
        SimulationCfg=sim_utils.SimulationCfg,
        SimulationContext=sim_utils.SimulationContext,
        Articulation=Articulation,
        go2_cfg=UNITREE_GO2_CFG,
        SimulationManager=SimulationManager,
        PhysxSchema=PhysxSchema,
        Usd=Usd,
        UsdGeom=UsdGeom,
        UsdPhysics=UsdPhysics,
        versions={
            name: importlib.metadata.version(name)
            for name in ("isaaclab", "isaacsim", "torch")
        },
    )


def _configure_simulation(source, rt, directory):
    """Copy only known primitive physics fields, never resolve source callables."""
    cfg = rt.SimulationCfg(device="cuda:0", dt=0.005, render_interval=4)
    physics = source["environment_physics"]
    for name in (
        "physics_prim_path",
        "device",
        "dt",
        "render_interval",
        "gravity",
        "enable_scene_query_support",
        "use_fabric",
        "create_stage_in_memory",
    ):
        value = physics[name]
        default = getattr(cfg, name)
        if name == "gravity":
            _require(
                isinstance(value, list) and len(value) == 3, "Invalid source gravity"
            )
            value = tuple(value)
        _require(
            type(value) is type(default), f"Unsupported source simulation field {name}"
        )
        setattr(cfg, name, value)
    _require(
        cfg.device == "cuda:0"
        and cfg.dt == 0.005
        and cfg.render_interval == 4
        and cfg.create_stage_in_memory is False
        and cfg.use_fabric is True,
        "Reproducer requires the fixed CUDA, 5ms, render4, USD/Fabric configuration",
    )
    defaults = cfg.physx.to_dict()
    _require(
        set(physics["physx"]) == set(defaults),
        "Source PhysX schema differs from installed Isaac Lab",
    )
    for name, default in defaults.items():
        value = physics["physx"][name]
        _require(
            type(default) in (bool, int, float)
            and type(value) is type(default)
            and (type(value) is not float or math.isfinite(value)),
            f"Unsupported source PhysX setting {name}",
        )
        setattr(cfg.physx, name, value)
    # A material's function/class must always come from the installed code.
    for name, value in physics["physics_material"].items():
        if name == "func":
            continue
        default = getattr(cfg.physics_material, name, None)
        _require(
            type(default) in (bool, int, float, str) and type(value) is type(default),
            f"Unsupported source material setting {name}",
        )
        setattr(cfg.physics_material, name, value)
    _require(cfg.physx.solver_type == 1, "Only the source TGS solver is allowed")
    if directory is not None:
        cfg.log_dir = str(directory)
    cfg.validate()
    return cfg


def _tensor_list(value, rt):
    _require(
        isinstance(value, rt.torch.Tensor)
        and value.is_floating_point()
        and value.ndim >= 2
        and value.shape[0] == 1
        and rt.torch.isfinite(value).all().item(),
        "Expected one finite floating-point physics tensor",
    )
    return value.detach().cpu().clone()[0].tolist()


def _clock(sim, measured):
    time = sim.current_time
    index = sim.current_time_step_index
    _require(
        type(time) in (int, float) and math.isfinite(time) and type(index) is int,
        "Simulation clock/step index unavailable",
    )
    return {
        "sim_time_s": float(time),
        "physics_step_index": index,
        "measured_step_count": measured,
    }


def _construction_position(robot, rt):
    """Read authored placement before sim.reset may advance initialization."""
    prim = robot.stage.GetPrimAtPath("/World/Robot")
    _require(
        prim.IsValid(), "Spawned robot prim unavailable before physics initialization"
    )
    matrix = rt.UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        rt.Usd.TimeCode.Default()
    )
    position = [float(value) for value in matrix.ExtractTranslation()]
    _require(
        len(position) == 3 and all(math.isfinite(value) for value in position),
        "Robot construction world position unavailable",
    )
    return position


def _properties(robot, rt):
    result = {
        name: _tensor_list(getattr(robot.root_physx_view, getter)(), rt)
        for name, getter in _PROPERTY_GETTERS.items()
    }
    for name in ("stiffness", "damping"):
        values = getattr(robot.data, f"default_joint_{name}").detach().clone()
        for actuator in robot.actuators.values():
            values[:, actuator.joint_indices] = getattr(actuator, name)
        result[f"joint_operative_{name}"] = _tensor_list(values, rt)
    for name in (
        "joint_physx_stiffness",
        "joint_physx_damping",
        "joint_legacy_friction_coefficient",
    ):
        _require(
            all(value == 0.0 for value in result[name]), f"Unexpected nonzero {name}"
        )
    _require(
        all(
            value == 0.0
            for row in result["joint_friction_static_dynamic_viscous"]
            for value in row
        ),
        "Unexpected nonzero new joint friction",
    )
    return result


def _scene_evidence(sim, robot, rt):
    stage, view = robot.stage, robot.root_physx_view
    root_path = list(view.prim_paths)
    _require(
        len(root_path) == 1 and not robot.is_fixed_base,
        "Expected one floating-base articulation",
    )
    scene_prim = stage.GetPrimAtPath(sim.cfg.physics_prim_path)
    root_prim = stage.GetPrimAtPath(root_path[0])
    _require(
        scene_prim.IsValid()
        and scene_prim.HasAPI(rt.PhysxSchema.PhysxSceneAPI)
        and root_prim.IsValid()
        and root_prim.HasAPI(rt.PhysxSchema.PhysxArticulationAPI),
        "Physics scene/articulation schemas unavailable",
    )
    scene_api = rt.PhysxSchema.PhysxSceneAPI(scene_prim)
    art_api = rt.PhysxSchema.PhysxArticulationAPI(root_prim)
    result = {
        "physics_scene_path": sim.cfg.physics_prim_path,
        "articulation_prim_path": root_path[0],
        "simulation_manager_solver_type": rt.SimulationManager.get_solver_type(
            physics_scene=sim.cfg.physics_prim_path
        ),
        "usd_scene_solver_type": scene_api.GetSolverTypeAttr().Get(),
        "articulation_position_iterations": art_api.GetSolverPositionIterationCountAttr().Get(),
        "articulation_velocity_iterations": art_api.GetSolverVelocityIterationCountAttr().Get(),
        "self_collisions_enabled": art_api.GetEnabledSelfCollisionsAttr().Get(),
        "is_fixed_base": bool(robot.is_fixed_base),
        "gravity_w_m_s2": [
            float(value)
            for value in rt.SimulationManager.get_physics_sim_view().get_gravity()
        ],
        "enabled_robot_collision_prim_paths": [],
        "enabled_external_collision_prim_paths": [],
        "contact_body_names": list(robot.body_names),
        "contact_evidence_basis": "USD collision-scope validation plus direct per-link PhysX net contact forces; no production correctness claim.",
        "solver_readback_basis": "SimulationManager/USD selected solver, not direct GPU execution evidence.",
    }
    _require(
        result["simulation_manager_solver_type"]
        == result["usd_scene_solver_type"]
        == "TGS"
        and result["articulation_position_iterations"] == 4
        and result["articulation_velocity_iterations"] == 0
        and result["self_collisions_enabled"] is False,
        "Actual solver/iteration/self-collision configuration differs from source",
    )
    for prim in stage.Traverse(rt.Usd.TraverseInstanceProxies()):
        if prim.HasAPI(rt.UsdPhysics.CollisionAPI):
            enabled = rt.UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            _require(type(enabled) is bool, "Collider enabled state unavailable")
            if enabled:
                path = prim.GetPath().pathString
                name = (
                    "enabled_robot_collision_prim_paths"
                    if path.startswith("/World/Robot/")
                    else "enabled_external_collision_prim_paths"
                )
                result[name].append(path)
    _require(
        result["enabled_robot_collision_prim_paths"]
        and not result["enabled_external_collision_prim_paths"],
        "Expected robot collision geometry only, with no ground or external colliders",
    )
    return result


def _contact_views(robot, rt):
    paths = list(robot.root_physx_view.link_paths[0])
    _require(
        [path.rsplit("/", 1)[-1] for path in paths] == list(robot.body_names),
        "Actual link paths do not match the declared body order",
    )
    physics = rt.SimulationManager.get_physics_sim_view()
    # One exact path per view makes ordering explicit without wildcard assumptions.
    return [
        physics.create_rigid_contact_view(
            path, filter_patterns=[], max_contact_data_count=16
        )
        for path in paths
    ]


def _state(robot, contacts, rt):
    view = robot.root_physx_view
    state = {
        name: _tensor_list(getattr(view, getter)(), rt)
        for name, getter in (
            ("joint_position_rad", "get_dof_positions"),
            ("joint_velocity_rad_s", "get_dof_velocities"),
            ("root_transform_w_xyzw", "get_root_transforms"),
            ("root_com_velocity_w_m_s_rad_s", "get_root_velocities"),
            ("link_transform_w_xyzw", "get_link_transforms"),
            ("link_com_velocity_w_m_s_rad_s", "get_link_velocities"),
            ("joint_physx_actuation_force_nm", "get_dof_actuation_forces"),
        )
    }
    state["contacts_w_n"] = [
        _tensor_list(contact.get_net_contact_forces(dt=0.005), rt)
        for contact in contacts
    ]
    _require(
        all(len(force) == 3 for force in state["contacts_w_n"]),
        "Contact getter returned unexpected dimensions",
    )
    return state


def run_case(source, case, directory=None):
    """Construct once, restore source state, and measure exactly one physics step."""
    validate_reproducer_source(source)
    _require(case in _CASES, "Unknown articulation reproducer case")
    rt = _load_runtime()
    for name in ("isaaclab", "isaacsim"):
        _require(
            rt.versions[name] == source["source_versions"][name],
            f"Installed {name} version differs from source",
        )
    cfg = _configure_simulation(source, rt, directory)
    robot_cfg = rt.go2_cfg.copy()
    _require(
        robot_cfg.spawn.usd_path == source["robot"]["asset_path"],
        "Installed Go2 asset path differs from source",
    )
    construction_x, reset_x = _CASES[case]
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (construction_x, 0.0, 0.4)
    props = robot_cfg.spawn.articulation_props
    _require(
        props.enabled_self_collisions is False
        and props.solver_position_iteration_count == 4
        and props.solver_velocity_iteration_count == 0,
        "Installed Go2 articulation defaults differ",
    )
    sim = rt.SimulationContext(cfg)
    try:
        robot = rt.Articulation(cfg=robot_cfg)
        construction_position = _construction_position(robot, rt)
        _require(
            all(
                abs(a - b) <= 1e-5
                for a, b in zip(construction_position, robot_cfg.init_state.pos)
            ),
            "Spawned robot world position differs from intended construction placement",
        )
        chronology = {"before_sim_reset": _clock(sim, 0)}
        sim.reset()
        chronology["after_sim_reset"] = _clock(sim, 0)
        _require(
            list(robot.joint_names)
            == source["joint_names"]
            == source["raw_joint_names"]
            and list(robot.body_names) == source["body_names"],
            "Installed Go2 joint/body order differs from source",
        )
        contacts = _contact_views(robot, rt)
        physical_properties = _properties(robot, rt)
        scene_readback = _scene_evidence(sim, robot, rt)
        after_sim_reset_state = _state(robot, contacts, rt)

        def tensor(values):
            return rt.torch.tensor(
                [values], device=robot.device, dtype=rt.torch.float32
            )

        initial = source["initial_state"]
        root_xyzw = tensor(initial["root_transform_w_xyzw"])
        root_xyzw[:, :3] -= tensor(source["source_origin_w_m"])
        root_xyzw[:, 0] += reset_x
        root_wxyz = rt.torch.cat(
            (root_xyzw[:, :3], root_xyzw[:, 6:7], root_xyzw[:, 3:6]), dim=-1
        )
        robot.reset()
        robot.write_root_pose_to_sim(root_wxyz)
        robot.write_root_com_velocity_to_sim(
            tensor(initial["root_com_velocity_w_m_s_rad_s"])
        )
        robot.write_joint_state_to_sim(
            tensor(initial["joint_position_rad"]),
            tensor(initial["joint_velocity_rad_s"]),
        )
        chronology["after_state_writes"] = _clock(sim, 0)
        _require(
            chronology["after_state_writes"] == chronology["after_sim_reset"],
            "Unexpected physics advance between initialization and state writes",
        )
        sim.forward()  # Kinematics/Fabric only; clock invariance checked below.
        chronology["after_sim_forward"] = _clock(sim, 0)
        effort = tensor(source["effort_nm"])
        indices = rt.torch.tensor([0], device=robot.device, dtype=rt.torch.int32)
        robot.root_physx_view.set_dof_actuation_forces(effort, indices)
        # Do NOT write_data_to_sim(): that would overwrite the prescribed effort
        # with the DCMotor calculation using the asset's position-target buffers.
        pre = _state(robot, contacts, rt)
        _require(
            rt.torch.allclose(
                tensor(pre["joint_physx_actuation_force_nm"]), effort, rtol=0, atol=1e-6
            ),
            "Backend effort readback does not match the prescribed source effort",
        )
        chronology["pre_step"] = _clock(sim, 0)
        _require(
            chronology["after_state_writes"]
            == chronology["after_sim_forward"]
            == chronology["pre_step"],
            "Unexpected physics advance after state writes and before measured step",
        )
        sim.step(render=False)
        robot.update(0.005)
        post = _state(robot, contacts, rt)
        chronology["post_step"] = _clock(sim, 1)
        _require(
            chronology["post_step"]["physics_step_index"]
            == chronology["pre_step"]["physics_step_index"] + 1
            and math.isclose(
                chronology["post_step"]["sim_time_s"]
                - chronology["pre_step"]["sim_time_s"],
                0.005,
                rel_tol=0,
                abs_tol=1e-9,
            ),
            "Measured step did not advance exactly one 5-ms physics step",
        )
        physics = cfg.to_dict()
        physics.pop("log_dir", None)
        return {
            "kind": "articulation_reproducer_case",
            "schema_version": 1,
            "case": case,
            "source_sha256": source["source_sha256"],
            "construction_origin_w_m": [construction_x, 0.0, 0.0],
            "reset_origin_w_m": [reset_x, 0.0, 0.0],
            "joint_names": list(robot.joint_names),
            "raw_joint_names": list(robot.joint_names),
            "body_names": list(robot.body_names),
            "environment_physics": physics,
            "runtime_versions": rt.versions,
            "configured_robot_usd_path": robot_cfg.spawn.usd_path,
            "construction_robot_position_w_m": construction_position,
            "configured_construction_robot_position_w_m": list(
                robot_cfg.init_state.pos
            ),
            "construction_pose_readback_basis": "USD /World/Robot world Xform before mandatory sim.reset; not a post-initialization physical root pose.",
            "physical_properties": physical_properties,
            "scene_readback": scene_readback,
            "after_sim_reset_state": after_sim_reset_state,
            "chronology": chronology,
            "measured_physics_steps": 1,
            "steps": [
                {"physics_substep": 0, "physics_dt_s": 0.005, "pre": pre, "post": post}
            ],
            "limitations": [
                "Contact-free direct backend effort; no policy, task, reward, terrain or actuator-compute loop.",
                "Mandatory sim.reset initialization is recorded separately from the one measured step.",
                "Source effort is reused numerically; this is not a source closed-loop trajectory or robot acceptance test.",
            ],
        }
    finally:
        # The coordinator owns AppLauncher.app.close(). Do not step during cleanup.
        sim.stop()
        sim.clear_all_callbacks()
        sim.clear_instance()
