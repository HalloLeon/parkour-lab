"""Batched external-trainer facade over the owned native step/reset boundary."""

import torch

from parkour_lab.control.controller import finite_tensor
from parkour_lab.environments.rewards import reward_recipe

from .config import VERSION
from .interface import (
    action_transform,
    configuration_id,
    task_spaces,
    validate_task_config,
)
from .observations import task_observation


class Go2Task:
    """Same-step autoreset with blind actors, optional training truth and owned finals.

    The caller owns the application and its learner/checkpoint. Closing this task
    closes only its native environment. Returned info is restricted to episode
    and interaction accounting; simulator diagnostics remain with the host.
    """

    def __init__(self, cfg, *, render_mode=None):
        if (
            getattr(cfg, "benchmark_version", None) != VERSION
            or type(getattr(cfg, "benchmark_training", None)) is not bool
            or cfg.benchmark_task_id != "Go2-Flat-v1"
        ):
            raise ValueError("Use build_config to select a supported Go2 task and role")
        cfg.validate()
        validate_task_config(cfg)
        identity = configuration_id(cfg)
        from parkour_lab.environments.runtime import LocomotionEnv

        self._env = LocomotionEnv(cfg=cfg, render_mode=render_mode)
        self._closed = False
        try:
            self._training = cfg.benchmark_training
            self.num_envs, self.device = self._env.num_envs, self._env.device
            self.step_dt = self._env.step_dt
            self._actions, binding_id = action_transform(self._env)
            (
                self.single_observation_space,
                self.single_action_space,
                self.observation_space,
                self.action_space,
            ) = task_spaces(self._env, self._actions, training=self.training)
            from parkour_lab.environments.dynamics import dynamics_report

            dynamics_report(self._env)
            self._env.capture_training_information = self.training
            self.metadata = {
                "protocol": VERSION,
                "task_id": cfg.benchmark_task_id,
                "configuration_id": identity,
                "seed": cfg.seed,
                "training": self.training,
                "autoreset": "same_step",
                "joint_names": list(self._env.scene["robot"].joint_names),
                "reward_recipe": reward_recipe(self._env),
                "motor_binding_id": binding_id,
                "action_coordinates": "q_target = default_q + 0.25 * bounded_action",
            }
            if self.training:
                from .schema import information_schema

                self.metadata["training_information"] = information_schema(
                    self.metadata["joint_names"]
                )
            self._ready = self._failed = False
            self._actor = None
        except BaseException:
            self.close()
            raise

    @property
    def training(self):
        return self._training

    @property
    def interactions(self):
        """Read the native owner; the facade never increments a second counter."""
        counter = self._env.interactions
        return {
            "session": counter.session,
            "cumulative": counter.cumulative,
            "uncertain": counter.uncertain,
        }

    def _healthy(self):
        if self._closed or self._failed:
            raise RuntimeError(
                "Task is closed or faulted; recreate it before continuing"
            )

    def _info(self, ids=None, final=None):
        if ids is None:
            ids = torch.empty(0, dtype=torch.long, device=self.device)
        return {
            "final_env_ids": ids.detach().clone(),
            "final_observation": final,
            "interactions": self.interactions,
        }

    @torch.no_grad()
    def reset(self, *, seed=None, env_ids=None):
        """Reset physical episodes without resetting cumulative experience or RNG progress."""
        self._healthy()
        if seed is not None and (type(seed) is not int or seed != self._env.cfg.seed):
            raise ValueError("A different run seed requires a new task")
        try:
            # The constructor seeded the task. Re-seeding the SDK here would also
            # rewind the consuming learner's global Torch RNG during continuation.
            native, _ = self._env.reset(seed=None, env_ids=env_ids)
            observation = task_observation(self._env, native, training=self.training)
            self._actor = observation["actor"].clone()
            self._ready = True
            return observation, self._info()
        except BaseException:
            self._failed = True
            raise

    @torch.no_grad()
    def step(self, action):
        self._healthy()
        if not self._ready:
            raise RuntimeError("Reset the task before collecting transitions")
        finite_tensor(action, (self.num_envs, 12), self._actions.default)
        delivered, _ = self._actions(action)
        try:
            native, reward, terminated, truncated, raw = self._env.step(delivered)
            ids = raw["final_env_ids"]
            final = None
            if len(ids):
                final = task_observation(
                    self._env,
                    raw["final_observation"],
                    training=self.training,
                    final_env_ids=ids,
                    information=raw.get("final_training_information"),
                )
            observation = task_observation(self._env, native, training=self.training)
            self._actor = observation["actor"].clone()
            return (
                observation,
                reward.detach().clone(),
                terminated,
                truncated,
                self._info(ids, final),
            )
        except BaseException:
            self._failed = True
            raise

    def progress(self, *, run_id, checkpoint_id, milestones=()):
        """Create JSON progress linked to the caller's full learner checkpoint identity."""
        self._healthy()
        from .progress import capture_progress

        return capture_progress(
            self._env,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            configuration_id=self.metadata["configuration_id"],
            milestones=milestones,
        )

    def restore(
        self,
        record,
        *,
        checkpoint_id,
        charged_interactions=None,
        uncertain_interactions=0,
        reason=None,
    ):
        """Restore benchmark progress before reset; the caller restores its learner separately."""
        self._healthy()
        if self._ready:
            raise RuntimeError("Restore before the first explicit reset")
        from .progress import restore_progress

        return restore_progress(
            self._env,
            record,
            checkpoint_id=checkpoint_id,
            configuration_id=self.metadata["configuration_id"],
            charged_interactions=charged_interactions,
            uncertain_interactions=uncertain_interactions,
            reason=reason,
        )

    @torch.no_grad()
    def set_command(self, command):
        """Return a fresh actor observation with an applied evaluation body twist.

        Sensor noise is retained; changing commands neither resamples sensors nor
        resets causal memory. Training commands belong to the shared sampler.
        """
        self._healthy()
        if self.training:
            raise ValueError("Training commands are sampled by the shared task")
        if not self._ready:
            raise RuntimeError("Reset before setting an evaluation command")
        finite_tensor(command, (self.num_envs, 3), self._actions.default)
        term = self._env.command_manager.get_term("base_velocity")
        term.vel_command_b.copy_(command)
        term.time_left.fill_(float("inf"))
        term.is_standing_env.fill_(False)
        term.is_heading_env.fill_(False)
        self._actor[:, 6:9] = command
        return {"actor": self._actor.clone()}

    def close(self):
        if not self._closed:
            self._closed = True
            self._env.close()
