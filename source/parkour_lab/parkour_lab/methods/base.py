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
    mandatory info['time_outs'] mask identifies truncation without termination;
    termination wins if both occur, so terminated = done & ~time_outs.

    Final pre-reset observations are not currently supplied by the ROA host.
    Adapters that need them for replay/bootstrapping must extend the provider or
    reject that configuration, never substitute a reset observation. A method
    must copy any observation retained across steps: buffers may be reused.
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
