"""Training-method boundary, separate from the inference controller contract.

Methods own collection, replay/storage, optimization and learning state. The host
owns task/reset semantics, commands and motor delivery. Array types are opaque:
an adapter, not this interface, owns device/framework conversion and preprocessing.
ROA is the first implementation; this is not an implemented Dreamer/DreamFLEX port.
"""

from typing import Any, Mapping, Protocol, runtime_checkable


class TrainingEnvironment(Protocol):
    """Synchronous batched control ticks with explicit auto-reset semantics.

    reset() returns observations and a boolean first-frame mask. step() returns
    (next_observations, rewards, done, info). A done row is already reset in
    next_observations; it MUST NOT be treated as a terminal observation. The
    info['terminated'] and info['truncated'] preserve both native flags;
    info['time_outs'] identifies truncation without termination. Termination
    wins for bootstrapping when both occur.

    info['final_observation'] contains owned pre-reset rows, indexed by
    info['final_env_ids']; None and an empty index tensor mean no episode ended.
    The adapter declares the final schema: ROA supplies only critic inputs,
    while the native provider retains all observation groups for other adapters.
    Sampling precedes reset, command resampling and interval events. Learners
    must copy ordinary observations retained across steps: buffers may be reused.
    """

    def reset(self, *, seed: int | None = None) -> tuple[Any, Any]: ...

    def step(
        self, action: Any, phase: str
    ) -> tuple[Any, Any, Any, Mapping[str, Any]]: ...


@runtime_checkable
class TrainingMethod(Protocol):
    """A learner, not merely a replaceable network inside PPO.

    Collection/update ratios and auxiliary phases belong to each method's
    schedule. update() may use on-policy storage, replay or imagined rollouts;
    neither its internals nor the checkpoint schema are prescribed here.
    Implementations may expose additional method-specific options and phases.
    Collection must reset per-environment learner memory on episode boundaries
    and keep replay sequences from crossing them; a reset is not a transition.

    Learning snapshots include policy/model and optimizer state; callers retain
    configuration/provenance beside them. Restoring learning state alone is NOT
    exact resumption of simulator state, observations, RNG or in-flight rollout.
    Inference/export uses the independent Controller interface.
    """

    def collect(self, environment: TrainingEnvironment, observations: Any) -> Any: ...

    def update(self, observations: Any) -> Mapping[str, Any]: ...

    def advance(
        self, environment: TrainingEnvironment, observations: Any
    ) -> tuple[Any, Mapping[str, Any]]: ...

    def state_dict(self) -> Mapping[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...
