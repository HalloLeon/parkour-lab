"""Native control transitions with owned observations from before automatic reset.

Import after AppLauncher. Physics, rewards and reset order remain SDK-owned.
Observation functions and noise must be stateless; learner adapters own history.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase
from isaaclab.utils.noise import NoiseCfg

from parkour_lab.runtime.accounting import InteractionCounter


class LocomotionEnv(ManagerBasedRLEnv):
    """Same-step autoreset with compact final rows in ``extras``.

    ``final_observation`` contains native groups indexed by ``final_env_ids``;
    it is None when no episode ends. These owned tensors precede scene reset,
    command resampling and interval events, so they retain the outgoing command.
    The regular step observations are for the next action, including reset rows.
    Both native termination flags are retained, even when they overlap.
    """

    def __init__(self, *args, **kwargs):
        self._capturing = False
        self._final = None
        self.capture_motion = False
        self.capture_evaluation = False
        self.capture_training_information = False
        self._final_physics = None
        self._final_information = None
        self.transition_counts = dict(
            control_steps=0,
            terminated_rows=0,
            truncated_rows=0,
            simultaneous_rows=0,
            final_observation_rows=0,
        )
        super().__init__(*args, **kwargs)
        self.interactions = InteractionCounter(self.num_envs)
        try:
            from parkour_lab.environments.faults import configure_fault

            configure_fault(self)
            self._validate_observations()
        except Exception:
            # The caller cannot close an object whose constructor never returned.
            self.close()
            raise

    def _validate_observations(self):
        # An extra observation read must not advance a filter or return stale
        # native history. Current groups use pure functions and uniform noise.
        manager = self.observation_manager
        for name, terms in manager.active_terms.items():
            group = getattr(manager.cfg, name)
            if group.history_length or not group.concatenate_terms:
                raise ValueError(
                    "Final observations require concatenated stateless frames"
                )
            for term_name in terms:
                term = getattr(group, term_name)
                if (
                    term.history_length
                    or term.modifiers
                    or isinstance(term.func, ManagerTermBase)
                    or (term.noise is not None and not isinstance(term.noise, NoiseCfg))
                ):
                    raise ValueError(
                        "Observation history, modifiers and stateful noise belong in the method adapter"
                    )

    def _capture_physics(self):
        if not self.capture_motion:
            return {}
        from parkour_lab.runtime.native import motion_state

        return motion_state(self)

    def _reset_idx(self, env_ids):
        if self._capturing:
            self._sync_interactions()
            self._final_physics = {
                name: value[env_ids] for name, value in self._capture_physics().items()
            }
            # Only ending episodes need a final sample. Isolate that additional
            # noise draw from the normal reset/survivor observation RNG stream.
            device = torch.device(self.device)
            with torch.random.fork_rng(
                devices=[device] if device.type == "cuda" else []
            ):
                observations = self.observation_manager.compute(update_history=False)
                final = {
                    name: value[env_ids].detach().clone()
                    for name, value in observations.items()
                }
            if any(not torch.isfinite(value).all() for value in final.values()):
                raise ValueError("Nonfinite pre-reset observation")
            self._final = (env_ids.clone(), final)
            if self._training_information_enabled():
                from parkour_lab.benchmark.information import capture_information

                self._final_information = {
                    name: value[env_ids].detach().clone()
                    for name, value in capture_information(self).items()
                }
        from parkour_lab.environments.faults import reset_fault

        reset_fault(self, env_ids)
        super()._reset_idx(env_ids)

    def _training_information_enabled(self):
        return self.capture_training_information and getattr(
            self.cfg, "benchmark_training", False
        )

    def _sync_interactions(self, *, failed=False):
        self.interactions.sync(
            self.common_step_counter, self._sim_step_counter, failed=failed
        )

    def step(self, action):
        self._final = None
        self._final_physics = None
        self._final_information = None
        self._capturing = True
        failed = True
        try:
            observations, reward, terminated, truncated, extras = super().step(action)
            self._sync_interactions()
            ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
            final = None
            if len(ids):
                if self._final is None or not torch.equal(ids, self._final[0]):
                    raise RuntimeError(
                        "Automatic reset did not supply matching final observations"
                    )
                final = self._final[1]
            # Tasks have no interval state perturbations; adding any requires
            # revisiting survivor sampling after the SDK step.
            physics = self._capture_physics()
            if len(ids):
                for name, value in physics.items():
                    value[ids] = self._final_physics[name]
            self.transition_counts["control_steps"] += 1
            self.transition_counts["terminated_rows"] += int(terminated.sum())
            self.transition_counts["truncated_rows"] += int(truncated.sum())
            self.transition_counts["simultaneous_rows"] += int(
                (terminated & truncated).sum()
            )
            self.transition_counts["final_observation_rows"] += len(ids)
            failed = False
            return (
                observations,
                reward,
                terminated.clone(),
                truncated.clone(),
                {
                    **extras,
                    "final_env_ids": ids.clone(),
                    "final_observation": final,
                    "motion_state": physics if self.capture_motion else None,
                    **(
                        {"final_training_information": self._final_information}
                        if self._training_information_enabled()
                        else {}
                    ),
                },
            )
        finally:
            self._capturing = False
            self._final = None
            self._final_physics = None
            self._final_information = None
            self._sync_interactions(failed=failed)
