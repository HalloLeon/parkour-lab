"""Public Go2 task definitions and native configuration, independent of learners."""

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType

from parkour_lab.environments.task_config import TaskConfig, positive
from parkour_lab.environments.terrains.families.specification import TerrainConfig

VERSION = "go2-blind-v1"
# Each definition is (families, supported). Definitions do not register native tasks.
TASK_IDS = MappingProxyType(
    {
        "Go2-Flat-v1": (("Flat",), True),
        "Go2-Uneven-v1": (("Uneven",), True),
        "Go2-Slopes-v1": (("Slopes",), True),
        "Go2-Stairs-v1": (("Stairs",), True),
        "Go2-Hills-v1": (("Hills",), True),
        "Go2-Mixed-v1": (("Flat", "Uneven", "Slopes", "Stairs", "Hills"), True),
    }
)


@dataclass(frozen=True)
class Go2Config:
    """Task selection and allocation under the fixed 200/50 Hz, 20-second recipe."""

    task_id: str = "Go2-Flat-v1"
    num_envs: int = 160
    seed: int = 42
    device: str = "cuda:0"
    terrain: TerrainConfig = field(default_factory=TerrainConfig)

    def __post_init__(self):
        if not isinstance(self.task_id, str) or self.task_id not in TASK_IDS:
            raise ValueError(f"Unknown Go2 task: {self.task_id}")
        positive(self.num_envs, "num_envs", integer=True)
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Require a uint32 seed")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("Require an explicit device")
        if not isinstance(self.terrain, TerrainConfig):
            raise ValueError("Require TerrainConfig for terrain selection")
        if self.task_id == "Go2-Mixed-v1" and self.num_envs < 10:
            raise ValueError(
                "Mixed terrain requires at least ten replicas for its subtypes"
            )
        if self.task_id == "Go2-Flat-v1" and self.terrain != TerrainConfig():
            raise ValueError("Flat has no terrain level selection")


def build_config(options: Go2Config, *, training=True):
    """Build after AppLauncher; customize native reward terms before making an env.

    Weights and parameters use the native reward configuration directly. Add an
    ordinary function with ``RewardTermCfg(func=function, weight=..., params=...)``.
    No learning method or model format is required.
    """
    if not isinstance(options, Go2Config) or type(training) is not bool:
        raise ValueError("Require Go2Config and an explicit boolean training flag")
    if not TASK_IDS[options.task_id][1]:
        raise NotImplementedError(f"Native task is not implemented: {options.task_id}")

    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
        UnitreeGo2RoughEnvCfg,
    )

    from parkour_lab.environments.commands.episode import EpisodeVelocityCommand
    from parkour_lab.environments.configuration import build_environment_config
    from parkour_lab.environments.terrains.families.configuration import (
        configure_terrain,
    )

    task = TaskConfig(
        terrain="flat",
        num_envs=options.num_envs,
        seed=options.seed,
        device=options.device,
        physics_hz=200,
        episode_length_s=20.0,
    )
    cfg = build_environment_config(task, evaluation=not training)
    cfg.rewards = deepcopy(UnitreeGo2RoughEnvCfg().rewards)
    cfg.terminations.procedural_physical_failure = None
    cfg.observations.policy = None
    cfg.observations.terrain = None
    cfg.benchmark_training = training
    cfg.benchmark_version = VERSION
    cfg.benchmark_task_id = options.task_id
    cfg.benchmark_terrain_options = options.terrain.as_dict()
    cfg.commands.base_velocity.class_type = EpisodeVelocityCommand
    cfg.commands.base_velocity.resampling_time_range = (20.0, 20.0)
    configure_terrain(
        cfg,
        options.terrain,
        training=training,
        task_id=options.task_id,
        seed=options.seed,
    )
    return cfg
