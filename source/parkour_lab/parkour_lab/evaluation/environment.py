"""Case environment construction and motor binding after AppLauncher."""

import gc

import torch

from parkour_lab.benchmark import Go2Config, TerrainConfig, build_config, make
from parkour_lab.benchmark.interface import terrain_metadata
from parkour_lab.control.motor_contract import make_motor_contract
from parkour_lab.control.policy import PolicyController
from parkour_lab.runtime.motor import _runtime_motor_binding
from parkour_lab.runtime.native import NativeControllerSession


def _prepare_case_stage():
    """Own a fresh stage; never replace a caller's active native environment.

    Closing an SDK environment releases its context but does not clear an
    ordinary USD stage. RayCaster also caches meshes by prim path across sensor
    instances, so reused terrain paths must not retain another case's geometry.
    """
    from isaaclab import sim as sim_utils
    from isaaclab.sensors import RayCaster

    gc.collect()
    if sim_utils.SimulationContext.instance() is not None:
        raise RuntimeError(
            "Evaluation requires an application without an active environment"
        )
    sim_utils.create_new_stage()
    RayCaster.meshes.clear()


def make_case_task(case, device):
    """One replica per semantic case keeps draws independent of iteration order.

    A fresh native context is closed after each attempt. Training remains batched;
    this deliberately small evaluator does not introduce a second batch scheduler.
    """
    options = Go2Config(
        task_id=f"Go2-{case.family}-v1",
        num_envs=1,
        seed=case.seed,
        device=device,
        terrain=(
            TerrainConfig()
            if case.family == "Flat"
            else TerrainConfig(fixed_level=case.physical_level)
        ),
    )
    _prepare_case_stage()
    task = make(build_config(options, training=False))
    try:
        env = task._env
        if task.training or env.cfg.curriculum.terrain_levels is not None:
            raise ValueError("Fixed evaluation must disable training and progression")
        env.capture_motion = env.capture_evaluation = True
        _select_column(env, case)
        task.metadata["terrain"] = terrain_metadata(env)
        return task
    except BaseException:
        task.close()
        raise


def _select_column(env, case):
    if case.family == "Flat":
        return
    definition = env.cfg.benchmark_terrain
    column = next(
        index
        for index, item in enumerate(definition["columns"])
        if item == {"family": case.family, "subtype": case.subtype}
    )
    terrain = env.scene.terrain
    terrain.terrain_types.fill_(column)
    terrain.terrain_levels.zero_()
    terrain.env_origins[:] = terrain.terrain_origins[0, column]
    env.scene.env_origins[:] = terrain.env_origins


def bind_policy(env, policy):
    """Adapt an already-loaded frame policy, or verify a caller's Controller."""
    binding, digest = _runtime_motor_binding(env)
    limits = env.scene["robot"].data.soft_joint_pos_limits[0].cpu().tolist()
    profile = "native_motor_sha256:" + digest
    controller = policy
    if not hasattr(policy, "spec"):
        controller = PolicyController(
            policy,
            joint_names=tuple(binding["joint_names"]),
            default_position_rad=env.scene["robot"].data.default_joint_pos[0],
            target_limits_rad=limits,
            actuator_profile=profile,
        )
    contract = make_motor_contract(binding, profile, target_limits_rad=limits)
    return NativeControllerSession(env, controller, contract, preserve_native_raw=True)


def realization(task):
    """Evaluator-only physical receipts; never supplied to a policy or its reset."""
    from parkour_lab.environments.dynamics import dynamics_report, start_report

    env = task._env
    terrain = env.scene.terrain
    return {
        "task": task.metadata,
        "dynamics": dynamics_report(env),
        "start": start_report(env),
        "randomization": env.parkour_randomization.manifest(),
        "terrain_levels": _indices(terrain, "terrain_levels"),
        "terrain_types": _indices(terrain, "terrain_types"),
        "env_origin_w": env.scene.env_origins[0].cpu().tolist(),
        "training_information_available": env.capture_training_information,
    }


def check_fixed_terrain(task, before):
    env = task._env
    for key in ("terrain_levels", "terrain_types"):
        if _indices(env.scene.terrain, key) != before[key]:
            raise RuntimeError("Evaluation changed fixed terrain assignment")
    if env.capture_training_information:
        raise RuntimeError("Training information was enabled during evaluation")


def _indices(terrain, name):
    value = getattr(terrain, name, None)
    return None if value is None else value.cpu().tolist()


def reset_native(task):
    """Use one sensing owner: the native controller host draws the actor noise.

    Calling Go2Task.reset/step as well would prepare additional actor frames and
    consume another noise draw. The task facade still owns configuration/cleanup.
    """
    task._env.reset(seed=None)
    return torch.ones(1, dtype=torch.bool, device=task.device)
