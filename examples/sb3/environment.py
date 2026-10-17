"""Expose the task's actor-only, same-step autoreset contract as an SB3 VecEnv."""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv


def _numpy(value, shape, *, boolean=False):
    dtype = torch.bool if boolean else torch.float32
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or (not boolean and not torch.isfinite(value).all())
    ):
        raise ValueError(f"Require finite {dtype} task tensor with shape {shape}")
    return value.detach().cpu().numpy().copy()


class Go2VecEnv(VecEnv):
    """Use one native batch; SB3 owns policy scaling, rollouts and replay.

    ``total_interactions`` is an absolute cumulative ceiling, including prior
    collection restored with a learner checkpoint. The guard never hides a
    partially collected vector batch. Native episode resets remain task-owned.
    """

    def __init__(self, task, *, total_interactions=None, app=None):
        actor = task.single_observation_space["actor"]
        action = task.single_action_space
        if (
            not isinstance(actor, spaces.Box)
            or actor.shape != (49,)
            or actor.dtype != np.float32
            or not isinstance(action, spaces.Box)
            or action.shape != (12,)
            or action.dtype != np.float32
            or not np.isfinite(action.low).all()
            or not np.isfinite(action.high).all()
            or not (action.low < action.high).all()
        ):
            raise ValueError("Require the shared actor49 and bounded action12 spaces")
        if total_interactions is not None and (
            type(total_interactions) is not int or total_interactions < 0
        ):
            raise ValueError("Require a nonnegative absolute interaction budget")
        self.task, self.app = task, app
        self.total_interactions = total_interactions
        self._pending = None
        self._closed = self._faulted = False
        self._returns = np.zeros(task.num_envs, dtype=np.float64)
        self._lengths = np.zeros(task.num_envs, dtype=np.int64)
        super().__init__(task.num_envs, actor, action)

    @property
    def interactions(self):
        return dict(self.task.interactions)

    def _healthy(self):
        if self._closed or self._faulted:
            raise RuntimeError("Vector task is closed or faulted; recreate it")

    def reset(self):
        self._healthy()
        self._pending = None
        try:
            observation, _ = self.task.reset()
            actor = _numpy(observation["actor"], (self.num_envs, 49))
        except BaseException:
            self._faulted = True
            raise
        self._returns.fill(0)
        self._lengths.fill(0)
        self.reset_infos = [{} for _ in range(self.num_envs)]
        return actor

    def step_async(self, actions):
        self._healthy()
        if self._pending is not None:
            raise RuntimeError("An action batch is already pending")
        values = np.asarray(actions)
        if values.shape != (self.num_envs, 12) or values.dtype.kind not in "fiu":
            raise ValueError("Require a finite action batch with shape (N, 12)")
        values = np.array(values, dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Require finite actions")
        self._pending = values

    def step_wait(self):
        self._healthy()
        if self._pending is None:
            raise RuntimeError("No action batch is pending")
        actions, self._pending = self._pending, None
        if self.app is not None and not self.app.is_running():
            raise RuntimeError("Native application stopped before collection")
        if (
            self.total_interactions is not None
            and self.interactions["cumulative"] + self.num_envs
            > self.total_interactions
        ):
            raise RuntimeError("Next native batch would exceed the interaction budget")
        before = self.interactions
        try:
            observation, reward, terminated, truncated, info = self.task.step(
                torch.as_tensor(actions, device=self.task.device)
            )
            after = self.interactions
            if (
                any(
                    after[key] - before[key] != self.num_envs
                    for key in ("session", "cumulative")
                )
                or after["uncertain"] != before["uncertain"]
                or info["interactions"] != after
            ):
                raise RuntimeError(
                    "Native interaction count differs from the delivered batch"
                )
            current = _numpy(observation["actor"], (self.num_envs, 49))
            rewards = _numpy(reward, (self.num_envs,))
            terminated = _numpy(terminated, (self.num_envs,), boolean=True)
            truncated = _numpy(truncated, (self.num_envs,), boolean=True)
            return self._outgoing(current, rewards, terminated, truncated, info)
        except BaseException:
            self._faulted = True
            raise

    def _outgoing(self, current, rewards, terminated, truncated, info):
        done = terminated | truncated
        ids = info["final_env_ids"]
        if (
            not isinstance(ids, torch.Tensor)
            or ids.ndim != 1
            or ids.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Require compact integer final environment IDs")
        ids = ids.detach().cpu().numpy()
        if not np.array_equal(np.sort(ids), np.flatnonzero(done)):
            raise ValueError("Final environment IDs differ from episode endings")
        final = info["final_observation"]
        if len(ids):
            final = _numpy(final["actor"], (len(ids), 49))
        elif final is not None:
            raise ValueError("Final observations must be absent without endings")
        self._returns += rewards
        self._lengths += 1
        infos = [
            {"TimeLimit.truncated": bool(timeout and not physical)}
            for timeout, physical in zip(truncated, terminated, strict=True)
        ]
        for compact, index in enumerate(ids):
            infos[index]["terminal_observation"] = final[compact].copy()
            infos[index]["episode"] = {
                "r": float(self._returns[index]),
                "l": int(self._lengths[index]),
            }
        self._returns[done] = 0
        self._lengths[done] = 0
        return current, rewards, done, infos

    def seed(self, seed=None):
        """Assert the one task seed; never rewind native or learner RNG streams."""
        original = self.task.metadata["seed"]
        if seed is not None and (type(seed) is not int or seed != original):
            raise ValueError("A different task seed requires a new task")
        return [original] * self.num_envs

    def get_attr(self, attr_name, indices=None):
        if attr_name != "render_mode":
            raise AttributeError("Only render_mode is exposed through SB3 get_attr")
        return [None for _ in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Task configuration is frozen")

    def set_options(self, options=None):
        if options is not None and options != {}:
            raise NotImplementedError("Task reset options are fixed")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError("Native task internals are not an SB3 env method")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._get_indices(indices)]

    def close(self):
        if not self._closed:
            self._closed = True
            self.task.close()
