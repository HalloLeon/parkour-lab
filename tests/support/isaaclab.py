"""Native config doubles and optional pinned-source checks, without starting PhysX."""

import importlib.util
import sys
import unittest
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np
import torch

from tests.support import ROOT


@contextmanager
def modules(replacements):
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in replacements}
    sys.modules.update(replacements)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def pinned_source_root():
    """Find the pinned checkout without importing Isaac Sim."""
    workspace = ROOT.parent
    checkout = next(
        (
            workspace / name
            for name in ("IsaacLab2.3.2", "IsaacLab")
            if (workspace / name / "VERSION").is_file()
            and (workspace / name / "VERSION").read_text().strip() == "2.3.2"
        ),
        None,
    )
    if checkout is None:
        raise unittest.SkipTest("Pinned IsaacLab2.3.2 source checkout unavailable")
    return checkout / "source/isaaclab/isaaclab"


@contextmanager
def pinned_generator():
    """Load real v2.3.2 config/copy/hash/dispatch; stub unexecuted Sim utilities."""
    root = pinned_source_root()

    def unused(*args, **kwargs):
        raise AssertionError("Unexecuted simulator/cache boundary unexpectedly called")

    replacements = {}
    for name in (
        "isaaclab",
        "isaaclab.utils",
        "isaaclab.terrains",
        "isaaclab.terrains.trimesh",
    ):
        replacements[name] = ModuleType(name)
        replacements[name].__path__ = []
    for name, attributes in {
        "isaaclab.utils.array": {
            "TENSOR_TYPES": {"numpy": np.ndarray, "torch": torch.Tensor},
            "TENSOR_TYPE_CONVERSIONS": {},
        },
        "isaaclab.utils.io": {"dump_yaml": unused},
        "isaaclab.utils.timer": {"Timer": unused},
        "isaaclab.utils.warp": {"convert_to_warp_mesh": unused},
        "isaaclab.terrains.trimesh.utils": {"make_border": unused},
        "isaaclab.terrains.utils": {
            "color_meshes_by_height": unused,
            "find_flat_patches": unused,
        },
    }.items():
        replacements[name] = ModuleType(name)
        vars(replacements[name]).update(attributes)
    paths = {
        "isaaclab.utils.string": "utils/string.py",
        "isaaclab.utils.dict": "utils/dict.py",
        "isaaclab.utils.configclass": "utils/configclass.py",
        "isaaclab.terrains.sub_terrain_cfg": "terrains/sub_terrain_cfg.py",
        "isaaclab.terrains.terrain_generator": "terrains/terrain_generator.py",
    }
    specs = {
        name: importlib.util.spec_from_file_location(name, root / path)
        for name, path in paths.items()
    }
    replacements.update(
        {name: importlib.util.module_from_spec(spec) for name, spec in specs.items()}
    )
    with modules(replacements):
        for name, spec in specs.items():
            spec.loader.exec_module(replacements[name])
            if name == "isaaclab.utils.configclass":
                replacements["isaaclab.utils"].configclass = replacements[
                    name
                ].configclass
        yield (
            replacements["isaaclab.utils.configclass"].configclass,
            replacements["isaaclab.terrains.sub_terrain_cfg"].SubTerrainBaseCfg,
            replacements["isaaclab.terrains.terrain_generator"].TerrainGenerator,
        )


@contextmanager
def pinned_raycast(launch):
    """Real wrapper/Torch shapes; only Warp bridge/kernel responses are doubled."""
    name = "isaaclab.utils.warp.ops"
    spec = importlib.util.spec_from_file_location(
        name, pinned_source_root() / "utils/warp/ops.py"
    )
    module = importlib.util.module_from_spec(spec)
    package = ModuleType("isaaclab.utils.warp")
    package.__path__ = []
    replacements = {
        name: module,
        "isaaclab.utils.warp": package,
        "isaaclab.utils.warp.kernels": NS(raycast_mesh_kernel=object()),
        "isaaclab.utils.math": NS(convert_quat=None),
        "warp": NS(
            config=NS(quiet=False),
            init=lambda: None,
            device_to_torch=lambda device: device,
            from_torch=lambda tensor, **kwargs: tensor,
            vec3=object(),
            float32=torch.float32,
            int32=torch.int32,
            launch=launch,
            synchronize=lambda: None,
        ),
    }
    with modules(replacements):
        spec.loader.exec_module(module)
        package.raycast_mesh = module.raycast_mesh
        yield module.raycast_mesh


class NativeConfig:
    def __init__(self, name=None, **values):
        if name is not None:
            self.name = name
        self.__dict__.update(values)

    def __eq__(self, other):
        return isinstance(other, NativeConfig) and vars(self) == vars(other)

    def replace(self, **values):
        result = deepcopy(self)
        result.__dict__.update(values)
        return result


class NativeRewardTerm(NativeConfig):
    def __init__(self, *, func, weight, params=None):
        super().__init__(
            func=func, weight=weight, params={} if params is None else params
        )


def configclass(cls):
    cls.__init__ = NativeConfig.__init__
    return cls


class NativeCommand:
    pass


class FlatCommand:
    pass


class RoughCommand:
    pass


class EpisodeCommand:
    pass


class TerrainCommand:
    pass


def scan(*args):
    pass


def failure(*args):
    pass


def workspace(*args):
    pass


def com_velocity(*args):
    pass


def configure_dynamics(cfg, task, *, evaluation=False):
    cfg.parkour_task = asdict(task)
    cfg.parkour_evaluation = evaluation


def stock_config():
    return NS(
        scene=NS(
            num_envs=4096,
            terrain=NS(
                terrain_type="plane", terrain_generator=None, prim_path="/World/ground"
            ),
            robot=NS(soft_joint_pos_limit_factor=0.9),
            height_scanner=None,
            contact_forces=NS(update_period=0.02),
        ),
        seed=None,
        sim=NS(device="cuda:0", dt=0.005),
        decimation=4,
        episode_length_s=20.0,
        curriculum=NS(terrain_levels=None),
        commands=NS(base_velocity=NS(ranges=NS())),
        events=NS(
            reset_base=NS(
                params={"pose_range": {}, "velocity_range": {"x": (0.0, 0.0)}}
            ),
            add_base_mass=NS(params={"mass_distribution_params": (-1.0, 3.0)}),
            physics_material=NS(
                params={
                    "static_friction_range": (0.8, 0.8),
                    "dynamic_friction_range": (0.6, 0.6),
                }
            ),
            push_robot=None,
            base_com=None,
        ),
        rewards=NS(
            track_lin_vel_xy_exp=NS(
                func=com_velocity,
                weight=1.5,
                params={"command_name": "base_velocity", "std": 0.5},
            ),
            track_ang_vel_z_exp=NS(
                func=com_velocity,
                weight=0.75,
                params={"command_name": "base_velocity", "std": 0.5},
            ),
            lin_vel_z_l2=NS(func=com_velocity, weight=-2.0, params={}),
            ang_vel_xy_l2=NS(func=com_velocity, weight=-0.05, params={}),
            dof_torques_l2=NS(func=com_velocity, weight=-0.0002, params={}),
            dof_acc_l2=NS(func=com_velocity, weight=-2.5e-7, params={}),
            action_rate_l2=NS(func=com_velocity, weight=-0.01, params={}),
            feet_air_time=NS(
                func=com_velocity,
                weight=0.25,
                params={
                    "sensor_cfg": NativeConfig("contact_forces", body_names=".*_foot"),
                    "command_name": "base_velocity",
                    "threshold": 0.5,
                },
            ),
            undesired_contacts=None,
            flat_orientation_l2=NS(func=com_velocity, weight=-2.5, params={}),
            dof_pos_limits=NS(func=com_velocity, weight=0.0, params={}),
        ),
        observations=NS(
            policy=NS(
                base_lin_vel=NS(func=com_velocity, noise=(-0.1, 0.1)),
                base_ang_vel=NS(noise=(-0.2, 0.2)),
                projected_gravity=NS(noise=(-0.05, 0.05)),
                velocity_commands=NS(),
                joint_pos=NS(noise=(-0.01, 0.01)),
                joint_vel=NS(noise=(-1.5, 1.5)),
                actions=NS(),
                height_scan=None,
                enable_corruption=True,
                concatenate_terms=True,
            )
        ),
        terminations=NS(time_out=NS(time_out=True), base_contact=NS(time_out=False)),
        actions=NS(joint_pos=NS(scale=0.25, use_default_offset=True)),
    )


def stock_rough_config():
    cfg = stock_config()
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = NativeConfig(
        size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        num_rows=10,
        num_cols=20,
        difficulty_range=(0.0, 1.0),
        curriculum=True,
        sub_terrains={
            "pyramid_stairs": NativeConfig(
                proportion=0.2, step_height_range=(0.05, 0.23)
            ),
            "pyramid_stairs_inv": NativeConfig(
                proportion=0.2, step_height_range=(0.05, 0.23)
            ),
            "boxes": NativeConfig(proportion=0.2, grid_height_range=(0.025, 0.1)),
            "random_rough": NativeConfig(proportion=0.2, noise_range=(0.01, 0.06)),
            "hf_pyramid_slope": NativeConfig(
                proportion=0.1, slope_range=(0.0, 0.4), border_width=0.25
            ),
            "hf_pyramid_slope_inv": NativeConfig(
                proportion=0.1, slope_range=(0.0, 0.4), border_width=0.25
            ),
        },
    )
    cfg.curriculum.terrain_levels = NativeConfig(func=workspace)
    cfg.observations.policy.height_scan = NativeConfig(num_rays=187)
    cfg.rewards.flat_orientation_l2.weight = 0.0
    return cfg


@contextmanager
def native_config_modules(stock=None, *, rough=None):
    ray = type("RayCasterCfg", (NativeConfig,), {"OffsetCfg": NativeConfig})
    replacements = {}
    values = {
        "isaaclab.managers": {
            **{
                name: NativeConfig
                for name in (
                    "ObservationGroupCfg",
                    "ObservationTermCfg",
                    "SceneEntityCfg",
                    "TerminationTermCfg",
                    "CurriculumTermCfg",
                )
            },
            "RewardTermCfg": NativeRewardTerm,
        },
        "isaaclab.sensors": {
            "RayCasterCfg": ray,
            "patterns": NS(GridPatternCfg=NativeConfig),
        },
        "isaaclab.terrains": {
            "SubTerrainBaseCfg": NativeConfig,
            "TerrainGeneratorCfg": NativeConfig,
            "MeshPlaneTerrainCfg": NativeConfig,
        },
        "isaaclab.utils": {"configclass": configclass},
        "isaaclab.envs.mdp.commands": {"UniformVelocityCommand": NativeCommand},
        "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg": {
            "UnitreeGo2FlatEnvCfg": lambda: deepcopy(
                stock if stock is not None else stock_config()
            )
        },
        "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg": {
            "UnitreeGo2RoughEnvCfg": lambda: deepcopy(
                rough if rough is not None else stock_rough_config()
            )
        },
        "parkour_lab.environments.observations": {
            "HeightScanObservationCfg": NativeConfig,
            "terrain_height_scan": scan,
        },
        "parkour_lab.environments.commands.native": {
            "FlatVelocityCommand": FlatCommand,
            "RoughVelocityCommand": RoughCommand,
            "ProceduralTerrainCommand": TerrainCommand,
            "procedural_physical_failure": failure,
            "procedural_workspace": workspace,
        },
        "parkour_lab.environments.commands.episode": {
            "EpisodeVelocityCommand": EpisodeCommand,
        },
        "parkour_lab.environments.dynamics": {"configure_dynamics": configure_dynamics},
    }
    for name, attributes in values.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        replacements[name] = module
    import parkour_lab.environments as environments

    with (
        patch.dict(sys.modules, replacements),
        patch.object(
            environments,
            "observations",
            replacements["parkour_lab.environments.observations"],
            create=True,
        ),
    ):
        yield
