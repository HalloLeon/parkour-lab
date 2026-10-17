"""Bounded synthetic task for exercising real external optimizers, without Kit."""

from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces


class FakeTask:
    """49D actor/12D action task with visible outgoing/reset frame differences."""

    def __init__(self, num_envs=4, seed=42, horizon=8):
        self.num_envs, self.device, self.horizon = num_envs, "cpu", horizon
        self.step_dt = 0.02
        self.single_observation_space = spaces.Dict(
            actor=spaces.Box(-np.inf, np.inf, (49,), dtype=np.float32)
        )
        self.single_action_space = spaces.Box(
            np.linspace(-2.0, -0.5, 12, dtype=np.float32),
            np.linspace(0.75, 2.0, 12, dtype=np.float32),
        )
        self.metadata = {
            "protocol": "go2-blind-v1",
            "seed": seed,
            "configuration_id": f"fake-go2-{num_envs}",
            "task_id": "Fake-Go2-v1",
            "training": True,
        }
        self.interactions = dict(session=0, cumulative=0, uncertain=0)
        self.state = torch.zeros(num_envs, 49)
        self.state[:, 48] = torch.arange(num_envs) + 1
        self.age = torch.zeros(num_envs, dtype=torch.long)
        self.reset_calls = self.restore_calls = 0
        self.closed = self.ready = False
        self.next_terminated = self.next_truncated = None
        self.last_actions = None
        self._run_id = None

    def _observation(self, actor=None):
        count = self.num_envs if actor is None else len(actor)
        return {
            "actor": self.state.clone() if actor is None else actor.clone(),
            "training": {"secret": torch.full((count, 1), 123456.0)},
        }

    def reset(self, *, seed=None):
        if seed is not None and seed != self.metadata["seed"]:
            raise ValueError("Wrong seed")
        if self.closed:
            raise RuntimeError("Closed task")
        self.reset_calls += 1
        self.ready = True
        self.age.zero_()
        self.state.zero_()
        self.state[:, 48] = torch.arange(self.num_envs) + 1
        return self._observation(), {}

    def step(self, actions):
        if not self.ready or self.closed:
            raise RuntimeError("Reset an open task before stepping")
        assert actions.shape == (self.num_envs, 12)
        assert actions.dtype == torch.float32 and torch.isfinite(actions).all()
        self.last_actions = actions.detach().clone()
        low = torch.as_tensor(self.single_action_space.low)
        high = torch.as_tensor(self.single_action_space.high)
        delivered = actions.clamp(low, high)
        self.state[:, :12] += 0.02 * delivered
        self.age += 1
        self.state[:, 12] = self.age.float()
        reward = 1 - 0.1 * delivered.square().mean(-1)
        automatic = self.age >= self.horizon
        row = torch.arange(self.num_envs)
        terminated = automatic & (row % 3 != 0)
        truncated = automatic & (row % 3 != 1)
        if self.next_terminated is not None:
            terminated = self.next_terminated.clone()
            self.next_terminated = None
        if self.next_truncated is not None:
            truncated = self.next_truncated.clone()
            self.next_truncated = None
        ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
        final = self._observation(self.state[ids]) if len(ids) else None
        self.state[ids] = 0
        self.state[:, 48] = torch.arange(self.num_envs) + 1
        self.age[ids] = 0
        self.interactions["session"] += self.num_envs
        self.interactions["cumulative"] += self.num_envs
        return (
            self._observation(),
            reward.clone(),
            terminated.clone(),
            truncated.clone(),
            {
                "final_env_ids": ids.clone(),
                "final_observation": final,
                "interactions": dict(self.interactions),
            },
        )

    def progress(self, *, run_id, checkpoint_id, milestones=()):
        if self._run_id is not None and run_id != self._run_id:
            raise ValueError("Run identity changed")
        self._run_id = run_id
        return {
            "version": "fake-progress-v1",
            "run_id": run_id,
            "checkpoint_id": checkpoint_id,
            "configuration_id": self.metadata["configuration_id"],
            "seed": self.metadata["seed"],
            "interactions": dict(self.interactions),
            "checkpoint_interactions": self.interactions["cumulative"],
            "milestones": list(milestones),
        }

    def restore(
        self,
        record,
        *,
        checkpoint_id,
        charged_interactions=None,
        uncertain_interactions=0,
        reason=None,
    ):
        if self.ready or self.interactions["session"]:
            raise RuntimeError("Restore before reset and collection")
        if (
            record["version"] != "fake-progress-v1"
            or record["checkpoint_id"] != checkpoint_id
            or record["seed"] != self.metadata["seed"]
            or record["configuration_id"] != self.metadata["configuration_id"]
        ):
            raise ValueError("Continuation identity differs")
        floor = max(record["interactions"]["cumulative"], charged_interactions or 0)
        if (
            floor > record["interactions"]["cumulative"] or uncertain_interactions
        ) and not reason:
            raise ValueError("Additional spend requires a reason")
        self.interactions.update(
            cumulative=floor + uncertain_interactions,
            uncertain=record["interactions"]["uncertain"] + uncertain_interactions,
        )
        self._run_id = record["run_id"]
        self.restore_calls += 1
        return deepcopy(record)

    def close(self):
        self.closed = True
