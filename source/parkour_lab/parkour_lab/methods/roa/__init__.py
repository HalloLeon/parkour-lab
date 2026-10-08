"""ROA backend: configuration, learning lifecycle and causal export."""

from dataclasses import asdict, dataclass
import io
import math

from parkour_lab.config import construct, positive


@dataclass(frozen=True)
class ROAConfig:
    rollout_steps: int = 24
    history_steps: int = 64
    history_interval: int = 5
    adaptation_epochs: int = 4
    adaptation_batches: int = 4
    adaptation_learning_rate: float = 0.001
    learning_rate: float = 0.0002
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    entropy_coef: float = 0.01
    regularization_coef: float = 0.1
    regularization_start_update: int = 3000
    regularization_end_update: int = 10000
    initial_action_std: float = 1.0
    contact_conditioned: bool = True

    def __post_init__(self):
        for name in (
            "rollout_steps",
            "history_steps",
            "history_interval",
            "adaptation_epochs",
            "adaptation_batches",
            "num_learning_epochs",
            "num_mini_batches",
        ):
            positive(getattr(self, name), name, integer=True)
        for name in ("learning_rate", "adaptation_learning_rate", "initial_action_std"):
            positive(getattr(self, name), name)
        for name in ("entropy_coef", "regularization_coef"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if type(self.contact_conditioned) is not bool:
            raise ValueError("contact_conditioned must be boolean")
        if (
            type(self.regularization_start_update) is not int
            or type(self.regularization_end_update) is not int
            or not 0
            <= self.regularization_start_update
            <= self.regularization_end_update
        ):
            raise ValueError("Regularization updates must satisfy 0 <= start <= end")


def configure(options):
    return asdict(construct(ROAConfig, options))


def validate_training(options, num_envs):
    config = ROAConfig(**options)
    samples = num_envs * config.rollout_steps
    if (
        config.num_mini_batches > samples
        or config.adaptation_batches > num_envs * config.history_steps
    ):
        raise ValueError("A minibatch cannot be empty")
    if samples % config.num_mini_batches:
        raise ValueError("PPO samples must divide evenly into minibatches")


def create(host, options, seed):
    from .training import ROATrainingMethod
    from .model import build_policy
    from .environment import ROAEnvironment

    adapter = ROAEnvironment(host, contact_conditioned=options["contact_conditioned"])
    observations, _ = adapter.reset(seed=seed)
    policy, _ = build_policy(
        observations,
        contact_conditioned=options["contact_conditioned"],
        initial_action_std=options["initial_action_std"],
    )
    method = ROATrainingMethod(
        policy,
        observations,
        num_envs=host.env.num_envs,
        device=host.env.device,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        max_grad_norm=1.0,
        **{
            k: v
            for k, v in options.items()
            if k not in ("contact_conditioned", "initial_action_std")
        },
    )
    return Learner(method, adapter, observations)


class Learner:
    def __init__(self, method, environment, observations):
        self.method, self.environment, self.observations = (
            method,
            environment,
            observations,
        )

    @property
    def updates(self):
        return self.method.updates

    def advance(self):
        self.observations, metrics = self.method.advance(
            self.environment, self.observations
        )
        metrics["gradient_updates"] = (
            self.updates
            * self.method.algorithm.num_learning_epochs
            * self.method.algorithm.num_mini_batches
        )
        return metrics

    def state_dict(self):
        return self.method.state_dict()

    def load_state_dict(self, state):
        self.method.load_state_dict(state)

    def diagnostics(self, output, report):
        from .diagnostics import NumericalCapture

        return NumericalCapture(output, self.method, report)


def _validate(state, kind, options, updates):
    import torch

    _finite_state(state)
    if kind == "training":
        if (
            not isinstance(state, dict)
            or set(state)
            != {
                "policy_state",
                "ppo_optimizer",
                "adaptation_optimizer",
                "updates",
                "adaptation_optimizer_steps",
            }
            or type(state["updates"]) is not int
            or state["updates"] != updates
            or type(state["adaptation_optimizer_steps"]) is not int
            or state["adaptation_optimizer_steps"] < 0
        ):
            raise ValueError("Invalid learning snapshot schema or counters")
        weights = state["policy_state"]
        if (
            not isinstance(weights, dict)
            or not weights
            or any(
                not isinstance(v, torch.Tensor) or v.dtype != torch.float32
                for v in weights.values()
            )
        ):
            raise ValueError("Learning weights must be finite float32 tensors")
        std = weights.get("std")
        if std is None or std.shape != (12,) or (std <= 0).any():
            raise ValueError("Learning snapshot requires positive action std")
        for key in ("ppo_optimizer", "adaptation_optimizer"):
            _validate_optimizer(state[key])
    else:
        _actor_modules(state)


def _finite_state(value):
    """Validate numerical containers; never serialize Python model objects."""
    import torch

    if isinstance(value, torch.Tensor):
        valid = bool(torch.isfinite(value).all())
    elif isinstance(value, dict):
        for item in value.values():
            _finite_state(item)
        return
    elif isinstance(value, (list, tuple)):
        for item in value:
            _finite_state(item)
        return
    else:
        valid = (
            value is None
            or type(value) in (str, bool, int)
            or (type(value) is float and math.isfinite(value))
        )
    if not valid:
        raise ValueError(
            "State must contain only finite numerical tensors and plain containers"
        )


def _validate_optimizer(optimizer):
    """Reject corrupt optimizer containers/moments before native initialization."""
    if (
        not isinstance(optimizer, dict)
        or set(optimizer) != {"state", "param_groups"}
        or not isinstance(optimizer["state"], dict)
        or not isinstance(optimizer["param_groups"], list)
        or not optimizer["param_groups"]
    ):
        raise ValueError("Invalid optimizer state")
    parameters = []
    for group in optimizer["param_groups"]:
        if (
            not isinstance(group, dict)
            or not isinstance(group.get("params"), list)
            or not group["params"]
            or any(type(p) is not int or p < 0 for p in group["params"])
        ):
            raise ValueError("Invalid optimizer parameter groups")
        parameters.extend(group["params"])
    if (
        len(set(parameters)) != len(parameters)
        or set(optimizer["state"]) - set(parameters)
        or any(not isinstance(state, dict) for state in optimizer["state"].values())
    ):
        raise ValueError("Optimizer state does not match its parameter groups")

    _finite_state(optimizer)


def dump(state, kind, options, updates):
    import torch

    _validate(state, kind, options, updates)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def load(payload, kind, options, updates):
    import torch

    state = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    _validate(state, kind, options, updates)
    return state


def export(state, options):
    parameters = state["policy_state"]
    return {
        name: {
            key.removeprefix(f"actor.{name}."): value
            for key, value in parameters.items()
            if key.startswith(f"actor.{name}.")
        }
        for name in ("motor", "estimator")
    }


def _actor_modules(state):
    import torch
    from .runtime import _fixed_modules

    if not isinstance(state, dict) or set(state) != {"motor", "estimator"}:
        raise ValueError("An actor contains only causal motor and estimator weights")
    with torch.random.fork_rng(devices=[]):
        motor, estimator = _fixed_modules()
    for name, module in (("motor", motor), ("estimator", estimator)):
        if not isinstance(state[name], dict) or any(
            not isinstance(v, torch.Tensor)
            or v.dtype != torch.float32
            or not torch.isfinite(v).all()
            for v in state[name].values()
        ):
            raise ValueError("Actor weights must be finite float32 tensors")
        module.load_state_dict(state[name], strict=True)
    return motor, estimator


def controller(state, options, manifest, device):
    import torch
    from .runtime import ROAHistoryController, roa_tensor_sha256

    motor, estimator = (module.to(device) for module in _actor_modules(state))
    return ROAHistoryController(
        motor,
        estimator,
        joint_names=manifest["joint_names"],
        default_position_rad=torch.tensor(
            manifest["configuration"]["default_position_rad"],
            dtype=torch.float32,
            device=device,
        ),
        artifact_sha256=roa_tensor_sha256(motor, estimator),
        actuator_profile=manifest["actuator_profile"],
    )


def dependencies():
    from parkour_lab.provenance import dependency_identity

    return {"rsl_rl": dependency_identity("rsl_rl", "rsl-rl-lib")}
