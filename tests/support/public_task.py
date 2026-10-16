"""Public task fixtures with real bounds/noise/counters and doubled simulator I/O."""

from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from parkour_lab.benchmark import Go2Config, build_config, make
from parkour_lab.benchmark.schema import FIELD_SPECS
from parkour_lab.environments.randomization import TaskRandomization
from parkour_lab.runtime.accounting import InteractionCounter
from tests.support.isaaclab import (
    NativeConfig,
    modules,
    native_config_modules,
    stock_config,
)
from tests.support.native import TrainingEnvironmentFixture


def config_dict(value):
    """Match the SDK's config conversion without loading simulator modules."""
    if isinstance(value, (NativeConfig, NS)):
        return {key: config_dict(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {key: config_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [config_dict(item) for item in value]
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    return value


@contextmanager
def public_configs():
    stock = NativeConfig(**vars(stock_config()))
    with (
        native_config_modules(stock=stock),
        patch.object(NativeConfig, "to_dict", config_dict, create=True),
        patch.object(NativeConfig, "validate", lambda _: None, create=True),
    ):
        yield


class PublicTaskFixture(TrainingEnvironmentFixture):
    """Four replicas, distinguishable outgoing/reset states and two ending flags."""

    def __init__(self, num_envs=4):
        super().__init__(num_envs)
        self.env.scene["robot"].data.soft_joint_pos_limits = torch.tensor(
            [[-1.0, 1.0]] * 12
        ).repeat(num_envs, 1, 1)
        self.env.interactions = InteractionCounter(num_envs)
        self.env.close = lambda: setattr(self.env, "closed", True)
        self.native_step = self.step
        self.env.step = self.public_step
        self.env.reset = self.public_reset
        self.env.closed = False
        self.env.cfg.parkour_task = dict(
            num_envs=num_envs, seed=42, dynamics="randomized"
        )
        self.env.parkour_randomization = TaskRandomization(self.env.cfg.parkour_task)
        self.final_information_missing = False

    def clean(self):
        observation = self.observations()
        observation["proprio"][:, :3] -= 0.001 * (self.env.index + 1)
        return observation

    def information(self, env):
        values = {
            name: torch.full(
                (env.num_envs, *shape),
                bool(env.index) if dtype == "bool" else float(env.index),
                dtype=torch.bool if dtype == "bool" else torch.float32,
            )
            for name, (shape, dtype, _, _) in FIELD_SPECS.items()
        }
        values["joint_position_relative_default"] = (
            self.data.joint_pos - self.data.default_joint_pos
        ).clone()
        return values

    def public_reset(self, *, seed=None, env_ids=None):
        if env_ids is None:
            self.reset(seed=seed)
        else:
            self.env.action_manager.action[env_ids] = 0
            self.env.episode_length_buf[env_ids] = 0
            self.data.joint_pos[env_ids] = self.data.default_joint_pos[env_ids]
        return self.clean(), {"native_extra": "hidden"}

    def public_step(self, action):
        # The underlying fixture supplies native delivery and ending/reset flags.
        # Capture is doubled here; real SDK reset ordering has separate tests.
        before = self.information(self.env)
        _, reward, terminated, truncated, raw = self.native_step(action)
        ids = raw["final_env_ids"]
        self.env.interactions.sync(self.env.index, 4 * self.env.index)
        if len(ids) and self.env.capture_training_information:
            raw["final_training_information"] = (
                None
                if self.final_information_missing
                else {name: value[ids].clone() for name, value in before.items()}
            )
        return self.clean(), reward, terminated, truncated, raw

    @contextmanager
    def task(self, *, training=True, options=None):
        def construct(*, cfg, render_mode):
            self.env.cfg = cfg
            if cfg.benchmark_task_id != "Go2-Flat-v1":
                self.env.scene.terrain = NS(
                    terrain_types=torch.div(
                        torch.arange(self.env.num_envs)
                        * len(cfg.benchmark_terrain["columns"]),
                        self.env.num_envs,
                        rounding_mode="floor",
                    )
                )
            return self.env

        with (
            public_configs(),
            modules({"parkour_lab.environments.runtime": NS(LocomotionEnv=construct)}),
            patch("parkour_lab.benchmark.task.reward_recipe", return_value={}),
            patch(
                "parkour_lab.benchmark.information.capture_information",
                side_effect=self.information,
            ),
        ):
            # native_config_modules supplies a deliberately inert dynamics module.
            import sys

            sys.modules["parkour_lab.environments.dynamics"].dynamics_report = (
                lambda _: None
            )
            options = options or Go2Config(num_envs=self.env.num_envs, device="cpu")
            cfg = build_config(options, training=training)
            task = make(cfg)
            try:
                yield task
            finally:
                task.close()
