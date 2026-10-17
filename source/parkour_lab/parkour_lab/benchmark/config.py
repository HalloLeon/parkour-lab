"""Public Go2 task definitions and native configuration, independent of learners."""

from dataclasses import dataclass, field
from types import MappingProxyType

from parkour_lab.environments.faults import FaultConfig
from parkour_lab.environments.terrains.specification import TerrainConfig

VERSION = "go2-blind-v1"
# Definitions select families; native construction remains explicit.
TASK_IDS = MappingProxyType(
    {
        "Go2-Flat-v1": ("Flat",),
        "Go2-Uneven-v1": ("Uneven",),
        "Go2-Slopes-v1": ("Slopes",),
        "Go2-Stairs-v1": ("Stairs",),
        "Go2-Hills-v1": ("Hills",),
        "Go2-Mixed-v1": ("Flat", "Uneven", "Slopes", "Stairs", "Hills"),
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
    fault: FaultConfig | None = None

    def __post_init__(self):
        if not isinstance(self.task_id, str) or self.task_id not in TASK_IDS:
            raise ValueError(f"Unknown Go2 task: {self.task_id}")
        if type(self.num_envs) is not int or self.num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Require a uint32 seed")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("Require an explicit device")
        if not isinstance(self.terrain, TerrainConfig):
            raise ValueError("Require TerrainConfig for terrain selection")
        if self.fault is not None and not isinstance(self.fault, FaultConfig):
            raise ValueError("Require FaultConfig for a timed drive loss")
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
    if training and options.fault is not None:
        raise ValueError("Timed actuator faults are excluded from core training")
    from parkour_lab.environments.configuration import build_environment_config
    from parkour_lab.environments.terrains.configuration import (
        configure_terrain,
    )

    cfg = build_environment_config(
        num_envs=options.num_envs,
        seed=options.seed,
        device=options.device,
        training=training,
    )
    cfg.benchmark_training = training
    cfg.benchmark_version = VERSION
    cfg.benchmark_task_id = options.task_id
    cfg.benchmark_terrain_options = options.terrain.as_dict()
    if options.fault is not None:
        cfg.benchmark_fault = options.fault.as_dict()
    configure_terrain(
        cfg,
        options.terrain,
        training=training,
        task_id=options.task_id,
        seed=options.seed,
    )
    return cfg
