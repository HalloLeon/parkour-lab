"""Bounded ROA: alternating privileged PPO and supervised history blocks.

The seven dynamics values are declared by the native provider, not inferred here:
base mass ratio minus one, local base COM xyz, and robot-shape mean static
friction, dynamic friction and restitution. Shape means are not effective contact
coefficients. Models and diagnostics alone are not training or acceptance evidence.
An opt-in teacher also reads twelve current body-frame net normal contact-force
features. These simulator labels never enter the causal history actor directly.
"""

from __future__ import annotations

import copy
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .controller import finite_tensor

VERSION = "operator_roa_pilot_v1"
FRAME_DIM = 45
HISTORY_LENGTH = 25
DYNAMICS_DIM = 7
CONTACT_DIM = 12
LATENT_DIM = 8
CODE_DIM = 3 + LATENT_DIM


def _batch(value, width, reference=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or len(value) < 1:
        raise ValueError("Require a nonempty batched tensor")
    finite_tensor(value, (len(value), width), reference)


def _stock_motor(motor):
    if not isinstance(motor, nn.Sequential) or len(motor) != 7:
        raise ValueError("Require the stock 48→128→128→128→12 ELU motor")
    for index, shape in enumerate(((48, 128), (128, 128), (128, 128), (128, 12))):
        layer = motor[index * 2]
        if (
            type(layer) is not nn.Linear
            or (layer.in_features, layer.out_features) != shape
            or layer.bias is None
        ):
            raise ValueError("Unexpected stock motor linear layer")
        if index < 3:
            activation = motor[index * 2 + 1]
            if (
                type(activation) is not nn.ELU
                or activation.alpha != 1
                or activation.inplace
            ):
                raise ValueError("Require unchanged stock ELU activations")
    for parameter in motor.parameters():
        if (
            parameter.dtype != torch.float32
            or parameter.device != motor[0].weight.device
        ):
            raise ValueError("Stock motor parameters require common-device float32")
        finite_tensor(parameter, parameter.shape)


class LatentMotor(nn.Module):
    """Keep the stock 48-column GEMM; condition its first hidden preactivation."""

    def __init__(self, stock_motor):
        super().__init__()
        _stock_motor(stock_motor)
        self.stock_first = copy.deepcopy(stock_motor[0])
        self.tail = nn.Sequential(
            *(copy.deepcopy(layer) for layer in list(stock_motor)[1:])
        )
        self.latent_projection = nn.Linear(LATENT_DIM, 128, bias=False, device="cpu")
        nn.init.zeros_(self.latent_projection.weight)
        self.latent_projection.to(self.stock_first.weight)

    def forward(self, frame, code):
        _batch(frame, FRAME_DIM, self.stock_first.weight)
        finite_tensor(code, (len(frame), CODE_DIM), frame)
        stock = torch.cat((code[:, :3], frame), dim=-1)
        result = self.tail(
            self.stock_first(stock) + self.latent_projection(code[:, 3:])
        )
        finite_tensor(result, (len(frame), 12), frame)
        return result


class ContactEncoderInput(nn.Module):
    """Preserve the seven-column teacher GEMM and add a separate contact projection."""

    def __init__(self, reference):
        super().__init__()
        self.reference = reference
        # Initialization must not perturb the matched experiment's rollout RNG.
        with torch.random.fork_rng(devices=[]):
            self.contact_projection = nn.Linear(
                CONTACT_DIM, reference.out_features, bias=False, device="cpu"
            )
        nn.init.zeros_(self.contact_projection.weight)
        self.contact_projection.to(reference.weight)
        self.contact_projection.requires_grad_(reference.weight.requires_grad)
        self.train(reference.training)

    def forward(self, privilege):
        return self.reference(privilege[:, :DYNAMICS_DIM].contiguous()) + (
            self.contact_projection(privilege[:, DYNAMICS_DIM:])
        )


class ROAActor(nn.Module):
    """Shared motor with privileged μ and causal φ; neither route reads true velocity.

    Both routes insert the same detached velocity estimate into the stock motor.
    μ and φ are aligned by separately directed, unsquared per-row L2 objectives.
    The native scheduler owns alternating blocks; forward is always privileged.
    """

    def __init__(self, stock_motor, contact_conditioned=False):
        super().__init__()
        if type(contact_conditioned) is not bool:
            raise ValueError("Contact conditioning must be an explicit boolean")
        self.motor = LatentMotor(stock_motor)
        self.encoder = nn.Sequential(
            nn.Linear(DYNAMICS_DIM, 64, device="cpu"),
            nn.ELU(),
            nn.Linear(64, 32, device="cpu"),
            nn.ELU(),
            nn.Linear(32, LATENT_DIM, device="cpu"),
            nn.Tanh(),
        ).to(self.motor.stock_first.weight)
        self.estimator = nn.Sequential(
            nn.Linear(HISTORY_LENGTH * FRAME_DIM, 128, device="cpu"),
            nn.ELU(),
            nn.Linear(128, 64, device="cpu"),
            nn.ELU(),
            nn.Linear(64, CODE_DIM, device="cpu"),
        ).to(self.motor.stock_first.weight)
        if contact_conditioned:
            self.enable_contact_conditioning()

    @property
    def contact_conditioned(self):
        return isinstance(self.encoder[0], ContactEncoderInput)

    @property
    def privileged_dim(self):
        return DYNAMICS_DIM + (CONTACT_DIM if self.contact_conditioned else 0)

    @property
    def privileged_obs_groups(self):
        return (
            ["policy", "dynamics"]
            + (["contacts"] if self.contact_conditioned else [])
            + ["history"]
        )

    def enable_contact_conditioning(self):
        if self.contact_conditioned:
            raise ValueError("Contact conditioning is already enabled")
        self.encoder[0] = ContactEncoderInput(self.encoder[0])

    def privileged_input(self, observations):
        dynamics = observations["dynamics"]
        _batch(dynamics, DYNAMICS_DIM, self.motor.stock_first.weight)
        if not self.contact_conditioned:
            return dynamics
        contacts = observations["contacts"]
        finite_tensor(contacts, (len(dynamics), CONTACT_DIM), dynamics)
        return torch.cat((dynamics, contacts), dim=-1)

    def encode(self, dynamics):
        _batch(dynamics, self.privileged_dim, self.motor.stock_first.weight)
        latent = self.encoder(dynamics)
        finite_tensor(latent, (len(dynamics), LATENT_DIM), dynamics)
        return latent

    def forward(self, observations):
        _batch(
            observations,
            FRAME_DIM + self.privileged_dim + HISTORY_LENGTH * FRAME_DIM,
            self.motor.stock_first.weight,
        )
        frame = observations[:, :FRAME_DIM]
        history = observations[:, FRAME_DIM + self.privileged_dim :].reshape(
            -1, HISTORY_LENGTH, FRAME_DIM
        )
        predicted = self._current_estimate(frame, history)
        latent = self.encode(
            observations[:, FRAME_DIM : FRAME_DIM + self.privileged_dim]
        )
        return self.motor(frame, torch.cat((predicted[:, :3].detach(), latent), dim=-1))

    def estimate(self, history):
        if (
            not isinstance(history, torch.Tensor)
            or history.ndim != 3
            or len(history) < 1
        ):
            raise ValueError("Require nonempty causal frame history")
        finite_tensor(
            history,
            (len(history), HISTORY_LENGTH, FRAME_DIM),
            self.motor.stock_first.weight,
        )
        raw = self.estimator(history.detach().clone().flatten(1))
        result = torch.cat((raw[:, :3], raw[:, 3:].tanh()), dim=-1)
        finite_tensor(result, (len(history), CODE_DIM), history)
        return result

    def _current_estimate(self, frame, history):
        _batch(frame, FRAME_DIM, self.motor.stock_first.weight)
        result = self.estimate(history)
        if not torch.equal(frame, history[:, -1]):
            raise ValueError(
                "Newest history frame must equal the delivered current frame"
            )
        return result

    def history_action(self, frame, history):
        predicted = self._current_estimate(frame, history)
        return self.motor(
            frame.detach(),
            torch.cat((predicted[:, :3].detach(), predicted[:, 3:]), dim=-1),
        )

    def regularization_loss(self, observations):
        history = observations["history"].reshape(-1, HISTORY_LENGTH, FRAME_DIM)
        predicted = self._current_estimate(observations["policy"], history)
        return torch.linalg.vector_norm(
            self.encode(self.privileged_input(observations))
            - predicted[:, 3:].detach(),
            dim=-1,
        ).mean()

    def adaptation_losses(self, history, privilege, velocity):
        predicted = self.estimate(history)
        finite_tensor(velocity, (len(history), 3), predicted)
        with torch.no_grad():
            target = self.encode(privilege).detach().clone()
        finite_tensor(target, (len(history), LATENT_DIM), predicted)
        return {
            "latent": torch.linalg.vector_norm(
                predicted[:, 3:] - target, dim=-1
            ).mean(),
            "velocity": F.mse_loss(predicted[:, :3], velocity.detach().clone()),
        }


class CausalHistory:
    """Oldest-to-newest delivered frames, with repeated reset-frame padding."""

    def __init__(self):
        self.frames = None

    def push(self, frame, reset_mask):
        _batch(frame, FRAME_DIM)
        if (
            not isinstance(reset_mask, torch.Tensor)
            or reset_mask.shape != (len(frame),)
            or reset_mask.dtype != torch.bool
            or reset_mask.device != frame.device
        ):
            raise ValueError("Reset mask must be a matching boolean device vector")
        delivered = frame.detach().clone()
        if self.frames is None:
            if not reset_mask.all():
                raise ValueError("First frame must reset every environment")
            self.frames = delivered[:, None].repeat(1, HISTORY_LENGTH, 1)
        else:
            finite_tensor(self.frames, (len(frame), HISTORY_LENGTH, FRAME_DIM), frame)
            self.frames = torch.cat((self.frames[:, 1:], delivered[:, None]), dim=1)
            self.frames[reset_mask] = delivered[reset_mask, None]
        return self.frames.clone()


def state_sha256(module):
    """Deterministic named-state identity, including shape and dtype."""
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_norms(actor):
    def norm(parameters):
        values = [
            p.grad.detach().square().sum() for p in parameters if p.grad is not None
        ]
        return float(torch.stack(values).sum().sqrt()) if values else None

    result = {
        "encoder_gradient_l2": norm(actor.encoder.parameters()),
        "projection_gradient_l2": norm(actor.motor.latent_projection.parameters()),
    }
    if actor.contact_conditioned:
        result["contact_projection_gradient_l2"] = norm(
            actor.encoder[0].contact_projection.parameters()
        )
    return result


@torch.no_grad()
def branch_diagnostics(actor, observations):
    """Permute observed latents only, keeping frame and velocity fixed; no pass flag."""
    predicted = actor.estimate(
        observations["history"].reshape(-1, HISTORY_LENGTH, FRAME_DIM)
    )
    privilege = actor.privileged_input(observations)
    code = torch.cat((predicted[:, :3], actor.encode(privilege)), dim=-1)
    permuted = torch.cat((code[:, :3], code[:, 3:].roll(1, dims=0)), dim=-1)
    delta = actor.motor(observations["policy"], code) - actor.motor(
        observations["policy"], permuted
    )
    spread = code[:, 3:].std(dim=0, unbiased=False)
    result = {
        "samples": len(code),
        "latent_batch_std_mean": float(spread.mean()),
        "latent_batch_std_max": float(spread.max()),
        "latent_permutation_action_abs_max": float(delta.abs().max()),
        **gradient_norms(actor),
    }
    if actor.contact_conditioned:
        without_contacts = privilege.clone()
        without_contacts[:, DYNAMICS_DIM:] = 0
        zero_code = torch.cat((code[:, :3], actor.encode(without_contacts)), dim=-1)
        contact_delta = (
            actor.motor(observations["policy"], code)
            - actor.motor(observations["policy"], zero_code)
        ).abs()
        result.update(
            contact_zero_action_abs_mean=float(contact_delta.mean()),
            contact_zero_action_abs_max=float(contact_delta.max()),
            contact_projection_weight_l2=float(
                actor.encoder[0].contact_projection.weight.norm()
            ),
            contact_feature_abs_mean=float(privilege[:, DYNAMICS_DIM:].abs().mean()),
            contact_feature_abs_max=float(privilege[:, DYNAMICS_DIM:].abs().max()),
        )
    return result


def build_policy(observations, source_state, contact_conditioned=False):
    """Stock motor weights/value warm start; predicted velocity changes actor inputs."""
    from rsl_rl.modules import ActorCritic
    from .distillation.teacher.model import StockTerrainInput

    policy_obs = observations["policy"]
    _batch(policy_obs, FRAME_DIM)
    if policy_obs.dtype != torch.float32:
        raise ValueError("ROA observations require float32")
    for name, width in (
        ("critic_state", 48),
        ("dynamics", DYNAMICS_DIM),
        ("terrain", 264),
        ("history", HISTORY_LENGTH * FRAME_DIM),
    ):
        finite_tensor(observations[name], (len(policy_obs), width), policy_obs)
    if type(contact_conditioned) is not bool:
        raise ValueError("Contact conditioning must be an explicit boolean")
    if contact_conditioned:
        finite_tensor(
            observations["contacts"], (len(policy_obs), CONTACT_DIM), policy_obs
        )
    if not isinstance(source_state, dict) or not source_state:
        raise ValueError("Require the verified stock ActorCritic state dictionary")
    for value in source_state.values():
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
            raise ValueError("Stock source tensors must be finite float32")
        finite_tensor(value, value.shape)
    reference = ActorCritic(
        observations,
        {"policy": ["critic_state"], "critic": ["critic_state"]},
        12,
        actor_hidden_dims=[128] * 3,
        critic_hidden_dims=[128] * 3,
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    reference.load_state_dict(source_state, strict=True)
    if (reference.std <= 0).any():
        raise ValueError("Stock action noise must be positive")
    policy = copy.deepcopy(reference)
    policy.actor = ROAActor(policy.actor, contact_conditioned=contact_conditioned)
    policy.critic[0] = StockTerrainInput(policy.critic[0])
    policy.obs_groups = {
        "policy": policy.actor.privileged_obs_groups,
        "critic": ["critic_state", "terrain"],
    }
    return policy.to(policy_obs), reference.to(policy_obs).eval().requires_grad_(False)


def set_phase(policy, phase):
    """Explicit optimizer ownership for privileged, history, and frozen blocks."""
    if (
        phase not in ("privileged", "history", "frozen")
        or type(policy.actor) is not ROAActor
    ):
        raise ValueError("Unknown ROA phase or actor")
    estimator = {id(p) for p in policy.actor.estimator.parameters()}
    for parameter in policy.parameters():
        parameter.requires_grad_(
            phase == "history" if id(parameter) in estimator else phase == "privileged"
        )
        parameter.grad = None
