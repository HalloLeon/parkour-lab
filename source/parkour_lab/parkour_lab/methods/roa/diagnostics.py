"""Opt-in evidence for the observed raw-action spike and PPO numerical failure.

Capture the first gross action excursion and an existing exception. This does
not clip actions, change learning, skip a batch or create a resumable checkpoint.
"""

from collections.abc import Mapping
from pathlib import Path

import torch


def tensor_copy(value, *, cpu=True):
    """Detach evidence from mutable rollout buffers without retaining graphs."""
    if isinstance(value, torch.Tensor):
        value = value.detach()
        return value.cpu().clone() if cpu else value.clone()
    if isinstance(value, Mapping):
        return {key: tensor_copy(item, cpu=cpu) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_copy(item, cpu=cpu) for item in value]
    return value


class NumericalCapture:
    """One pre/post event, its next PPO rollout, and any terminating exception."""

    def __init__(self, output, method, report):
        self.output, self.method = Path(output) / "numerical_diagnostic", method
        self.report = report["numerical_diagnostic"] = {
            "raw_abs_trigger": 100.0,
            "action_delta_l2_trigger": 10000.0,
            "event_captured": False,
            "files": [],
            "scope": "diagnostic tensors only; no simulator replay or resumable state",
        }
        self.latest_forward = {}
        self.previous_transition = None
        self.rollout_captured = False
        self.handles = []

    def __enter__(self):
        self.output.mkdir()
        self.method.diagnostic = self
        for name in ("estimator", "motor"):

            def remember(module, inputs, result, name=name):
                self.latest_forward[name] = {
                    "inputs": tuple(value.detach() for value in inputs),
                    "output": result.detach(),
                }

            self.handles.append(getattr(self.method.policy.actor, name).register_forward_hook(remember))
        return self

    def __exit__(self, kind, error, traceback):
        try:
            if error is not None:
                try:
                    self.capture_exception(error)
                except Exception as capture_error:
                    # Preserve the original learning error if saving evidence fails.
                    self.report["capture_error"] = repr(capture_error)
        finally:
            self.close()
        return False

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.method.diagnostic = None

    def save(self, name, value):
        torch.save(tensor_copy(value), self.output / name)
        self.report["files"].append(f"numerical_diagnostic/{name}")

    def step(self, environment, observations, action, phase):
        host = environment.host
        delta_l2 = (action - observations["policy"][:, 33:45]).square().sum(-1)
        trigger_rows = (
            (action.abs().amax(-1) > self.report["raw_abs_trigger"])
            | (delta_l2 > self.report["action_delta_l2_trigger"])
            | ~torch.isfinite(action).all(-1)
        )
        first = not self.report["event_captured"] and bool(trigger_rows.any())
        current = tensor_copy(
            {
                "method_updates": self.method.updates,
                "control_step": host.steps,
                "phase": phase,
                "observations": observations,
                "raw_action": action,
                "policy_std": self.method.policy.std,
                "requested_position_rad": host.bridge.default + 0.25 * action,
                "action_delta_l2": delta_l2,
                "forward": self.latest_forward,
            },
            cpu=False,
        )
        if first:
            self.save(
                "first_transition.pt",
                {
                    "trigger_rows": trigger_rows.nonzero().flatten(),
                    "previous": self.previous_transition,
                    "current": current,
                },
            )
            self.report["event_captured"] = True
            self.report["event_phase"] = phase
            self.report["event_method_updates"] = self.method.updates
            print(f"Numerical diagnostic captured first action excursion: {self.output}", flush=True)
        result = environment.step(action, phase)
        _, reward, done, _ = result
        current.update(
            tensor_copy(
                {
                    "reward": reward,
                    "reward_term_names": list(host.env.reward_manager.active_terms),
                    "weighted_reward_rates": host.env.reward_manager._step_reward,
                    "step_dt": host.env.step_dt,
                    "done": done,
                    "encoded_raw_action": host.previous,
                },
                cpu=False,
            )
        )
        if first:
            self.save("first_transition_after.pt", current)
        self.previous_transition = current
        return result

    @property
    def active(self):
        return self.report["event_captured"] and not self.rollout_captured

    def rollout(self, algorithm):
        storage = algorithm.storage
        return {
            "filled_steps": storage.step,
            "method_updates": self.method.updates,
            **{
                name: getattr(storage, name)
                for name in (
                    "observations",
                    "actions",
                    "rewards",
                    "dones",
                    "values",
                    "returns",
                    "advantages",
                    "actions_log_prob",
                    "mu",
                    "sigma",
                )
            },
        }

    def capture_rollout(self, algorithm, observations):
        self.save("first_rollout.pt", self.rollout(algorithm))
        self.rollout_captured = True

    def after_update(self, algorithm):
        pass  # Existing observer interface; no extra computation or state change.

    def capture_exception(self, error):
        # Tensor locals preserve the actual failed minibatch, including non-finite
        # values. Do not serialize frames/modules or call model inference again.
        frames = []
        traceback = error.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            filename = frame.f_code.co_filename
            if "/parkour_lab/" in filename:
                values = {
                    name: tensor_copy(value)
                    for name, value in frame.f_locals.items()
                    if isinstance(value, torch.Tensor)
                    or (
                        isinstance(value, Mapping)
                        and len(value)
                        and all(isinstance(item, torch.Tensor) for item in value.values())
                    )
                }
                frames.append(
                    {
                        "file": filename,
                        "function": frame.f_code.co_name,
                        "line": traceback.tb_lineno,
                        "tensors": values,
                    }
                )
            traceback = traceback.tb_next
        gradients = {
            name: parameter.grad
            for name, parameter in self.method.policy.named_parameters()
            if parameter.grad is not None
        }
        self.save(
            "exception.pt",
            {
                "error": repr(error),
                "method_updates": self.method.updates,
                "policy_state": self.method.policy.state_dict(),
                "frames": frames,
                "last_transition": self.previous_transition,
                "latest_forward": self.latest_forward,
                "rollout": self.rollout(self.method.algorithm),
                "gradients": gradients,
                "gradient_norms_float64": {name: value.detach().double().norm() for name, value in gradients.items()},
                "gradient_timing": "as observed at exception; may be stale if failure preceded backward",
                "storage_timing": "only filled_steps prefix is current; returns may be unfinished if collection failed",
            },
        )
