"""Versioned, CPU-testable bridge from the stock operator motor to history.

This is interface groundwork, not a qualified policy. The oracle path preserves
the 48-D motor exactly; its legacy flat student estimates velocity from history.
The separate native GRU actor learns directly from 45-D sensor/command frames,
with simulator-only observations confined to its training critic. There is no
automatic checkpoint conversion or trained recurrent checkpoint here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn


VERSION = "go2_operator_velocity_student_v1"
HISTORY_LENGTH = 10
FRAME_DIM = 45
FRAME_TERMS = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
)
CONTROLLER_ROLLOUT_COMMANDS = (
    (0.0, 0.0, 0.0),
    (0.55, 0.0, 0.5),
    (0.55, 0.0, -0.5),
    (0.0, 0.0, 0.0),
    (0.55, 0.0, 0.0),
)
RECURRENT_OPERATOR_VERSION = "go2_operator_proprio_gru_v1"


def recurrent_policy_config():
    """Native RSL-RL policy; fresh learning, not a stock-checkpoint conversion."""
    return {
        "class_name": "ActorCriticRecurrent",
        "rnn_type": "gru",
        "rnn_hidden_dim": 128,
        "rnn_num_layers": 1,
        "actor_hidden_dims": [128, 128, 128],
        "critic_hidden_dims": [128, 128, 128],
        "activation": "elu",
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "state_dependent_std": False,
        "noise_std_type": "log",
        "init_noise_std": 0.5,
    }


def build_recurrent_operator_policy(observations):
    """Construct an unmodified causal actor/asymmetric critic on its input device."""
    from rsl_rl.modules import ActorCriticRecurrent

    proprio = observations["proprio"]
    if proprio.ndim != 2 or len(proprio) < 1:
        raise ValueError("Require batched 45-D proprioceptive observations")
    for name, width in (("proprio", 45), ("policy", 48), ("terrain", 264)):
        value = observations[name]
        _check_tensor(value, (len(proprio), width), name)
        if value.device != proprio.device or value.dtype != proprio.dtype:
            raise ValueError("Recurrent observation device/dtype must agree")
    config = recurrent_policy_config()
    config.pop("class_name")
    return ActorCriticRecurrent(
        observations,
        {"policy": ["proprio"], "critic": ["policy", "terrain"]},
        12,
        **config,
    ).to(proprio)


class RecurrentOperatorAdapter:
    """Extract only a causal actor; private zero-start memory never copies a rollout."""

    def __init__(
        self,
        policy,
        *,
        joint_names,
        default_position_rad,
        artifact_sha256,
        actuator_profile,
    ):
        from rsl_rl.modules import ActorCriticRecurrent
        from parkour_lab.learning.controller import ControllerSpec, SensorSpec

        if (
            type(policy) is not ActorCriticRecurrent
            or policy.obs_groups
            != {"policy": ["proprio"], "critic": ["policy", "terrain"]}
            or policy.actor_obs_normalization
            or type(policy.actor_obs_normalizer) is not nn.Identity
            or policy.state_dependent_std
            or type(policy.memory_a.rnn) is not nn.GRU
            or type(policy.memory_c.rnn) is not nn.GRU
            or policy.memory_a.rnn.input_size != 45
            or policy.memory_c.rnn.input_size != 312
            or policy.memory_a.rnn.hidden_size != 128
            or policy.memory_a.rnn.num_layers != 1
            or policy.memory_a.rnn.batch_first
            or policy.memory_a.rnn.bidirectional
            or policy.memory_a.rnn.dropout != 0
            or len(policy.actor) != 7
            or any(
                type(policy.actor[i]) is not nn.Linear
                or (policy.actor[i].in_features, policy.actor[i].out_features) != shape
                for i, shape in (
                    (0, (128, 128)),
                    (2, (128, 128)),
                    (4, (128, 128)),
                    (6, (128, 12)),
                )
            )
            or any(
                type(policy.actor[i]) is not nn.ELU or policy.actor[i].alpha != 1.0
                for i in (1, 3, 5)
            )
            or any(not torch.isfinite(p).all() for p in policy.parameters())
            or len(joint_names) != 12
            or len(set(joint_names)) != 12
        ):
            raise ValueError(
                "Require the native 45-D GRU operator policy and twelve named joints"
            )
        _check_tensor(default_position_rad, (12,), "default joint position")
        parameter = next(policy.actor.parameters())
        if (
            default_position_rad.device != parameter.device
            or default_position_rad.dtype != parameter.dtype
        ):
            raise ValueError("Default joint position must match policy device/dtype")
        self.memory = copy.deepcopy(policy.memory_a.rnn).eval().requires_grad_(False)
        self.actor = copy.deepcopy(policy.actor).eval().requires_grad_(False)
        self.default_position_rad = default_position_rad.detach().clone()
        self.hidden_state = None
        self.reset_mask = None
        self.spec = ControllerSpec(
            name="recurrent_operator_proprio",
            artifact_sha256=artifact_sha256,
            preprocessing_version=RECURRENT_OPERATOR_VERSION,
            joint_names=tuple(joint_names),
            period_s=0.02,
            actuator_profile=actuator_profile,
            sensors={
                "base_ang_vel": SensorSpec((3,), "rad/s", "body", 0.02),
                "projected_gravity": SensorSpec((3,), "unitless", "body", 0.02),
                "joint_position_relative_default": SensorSpec(
                    (12,), "rad", "joint", 0.02
                ),
                "joint_velocity": SensorSpec((12,), "rad/s", "joint", 0.02),
                "stock_previous_raw_action": SensorSpec(
                    (12,), "unitless", "joint", 0.02
                ),
            },
            raw_action_meaning="stock unscaled action; q_target = default_q + 0.25 * raw_action; no clip",
            configuration={
                "default_position_rad": default_position_rad.tolist(),
                "action_scale": 0.25,
                "action_clip": None,
                "actor": "45 -> GRU(128,1) -> ELU MLP(128,128,128) -> 12",
                "input_order": [name for name, _ in FRAME_TERMS],
                "reset": "zero GRU state; previous delivered raw action must be zero",
                "privileged_critic_exported": False,
            },
        )

    @torch.inference_mode()
    def reset(self, mask):
        if (
            mask.ndim != 1
            or mask.dtype != torch.bool
            or mask.device != self.default_position_rad.device
        ):
            raise ValueError("Require a matching boolean reset mask")
        if self.hidden_state is None:
            if not len(mask) or not mask.all():
                raise ValueError("First recurrent frame must reset every environment")
        else:
            _check_tensor(self.hidden_state, (1, len(mask), 128), "GRU state")
            self.hidden_state[:, mask] = 0
        self.reset_mask = mask.clone()

    @torch.inference_mode()
    def act(self, inputs):
        from parkour_lab.learning.controller import JointTargets

        sensors = inputs.sensors
        previous = sensors["stock_previous_raw_action"].value
        if self.reset_mask is None or torch.any(previous[self.reset_mask] != 0):
            raise ValueError("Previous delivered action must be zero after reset")
        frame = torch.cat(
            (
                sensors["base_ang_vel"].value,
                sensors["projected_gravity"].value,
                inputs.command,
                sensors["joint_position_relative_default"].value,
                sensors["joint_velocity"].value,
                previous,
            ),
            dim=-1,
        )
        _check_tensor(frame, (len(self.reset_mask), FRAME_DIM), "proprioceptive frame")
        memory, hidden = self.memory(frame.unsqueeze(0), self.hidden_state)
        action = self.actor(memory.squeeze(0))
        _check_tensor(action, (len(frame), 12), "recurrent raw action")
        _check_tensor(hidden, (1, len(frame), 128), "GRU state")
        self.hidden_state = hidden.detach()
        return JointTargets(
            self.spec.joint_names, self.default_position_rad + 0.25 * action, action
        )


def interface_manifest(
    *, teacher_sha256: str, env_sha256: str, joint_names: list[str]
) -> dict:
    """Bind explicit tensor semantics to an audited source and resolved joint order.

    Callers must obtain hashes from the validated source checkpoint/config and
    joint_names from the simulator's runtime descriptor, not from a regex order.
    This manifest does not itself qualify the teacher or certify hardware units.
    """
    for value in (teacher_sha256, env_sha256):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Require a full lowercase source SHA-256")
    if (
        len(joint_names) != 12
        or len(set(joint_names)) != 12
        or any(not isinstance(n, str) or not n for n in joint_names)
    ):
        raise ValueError("Require twelve distinct resolved joint names")
    return {
        "version": VERSION,
        "teacher_sha256": teacher_sha256,
        "environment_sha256": env_sha256,
        "motor": {
            "input_order": ["base_lin_vel", *[name for name, _ in FRAME_TERMS]],
            "input_dim": 48,
            "hidden_dims": [128, 128, 128],
            "activation": "elu",
            "output_dim": 12,
            "frozen": True,
            "normalization": "none",
        },
        "student": {
            "frame_terms": [[name, width] for name, width in FRAME_TERMS],
            "frame_dim": FRAME_DIM,
            "history_length": HISTORY_LENGTH,
            "history_layout": "oldest_to_newest; newest equals this action's delivered frame",
            "reset": "repeat first delivered new-episode frame; never inherit old-episode samples",
            "step_dt_s": 0.02,
            "velocity_estimator": [HISTORY_LENGTH * FRAME_DIM, 128, 64, 3],
            "velocity_units": "body-frame m/s; no scaling or clipping",
            "supervision": "current pre-action simulator velocity, detached MSE label only",
            "motor_input": "detached estimate, never the supervision label",
            "forbidden_inputs": [
                "simulator linear velocity",
                "world pose",
                "privileged dynamics",
                "terrain",
                "future frames",
            ],
        },
        "actions": {
            "joint_names": list(joint_names),
            "transform": "default_joint_position + 0.25 * raw_action",
            "clip": None,
            "history_action": "previous delivered raw action, not current or PD-scaled target",
        },
        "scope": "flat-ground velocity-estimation bridge only; NOT full latent RMA, obstacle or hardware acceptance",
    }


def interface_sha256(manifest: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def validate_interface(expected: dict, actual: dict) -> None:
    if expected.get("version") != VERSION or expected != actual:
        raise ValueError(
            "Operator student interface mismatch; implicit conversion is forbidden"
        )


def prepare_bridge(checkpoint: Path, report_path: Path):
    """Construct an UNTRAINED bridge from replay-verified source evidence.

    A failing teacher is allowed for interface audits, never silently qualified.
    No environment or simulator is launched, and source files are not modified.
    """
    try:
        from .operator_benchmark_core import (
            OBSERVATION_TERMS,
            load_reference_actor,
            read_yaml_data,
        )
        from .operator_checkpoint_screen import replay_report
    except ImportError:
        from operator_benchmark_core import (
            OBSERVATION_TERMS,
            load_reference_actor,
            read_yaml_data,
        )
        from operator_checkpoint_screen import replay_report
    evidence = replay_report(report_path, checkpoint)
    runtime = json.loads(report_path.read_text())["runtime"]
    if runtime["observation_terms"] != list(OBSERVATION_TERMS):
        raise ValueError("Runtime observation order differs from the stock teacher")
    saved = read_yaml_data(checkpoint.parent / "params/env.yaml")
    if saved.get("decimation") != "4" or saved.get("sim", {}).get("dt") != "0.005":
        raise ValueError("Require the stock 5-ms physics / 20-ms action timestep")
    if set(saved["observations"]) != {"policy"}:
        raise ValueError("Only the stock policy observation group is supported")
    policy = saved["observations"]["policy"]
    functions = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "generated_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
    )
    order = [k for k, v in policy.items() if isinstance(v, dict) and "func" in v]
    if (
        order != list(OBSERVATION_TERMS)
        or policy.get("history_length") != "null"
        or policy.get("concatenate_terms") != "true"
        or policy.get("concatenate_dim") != "-1"
    ):
        raise ValueError("Unexpected teacher observation layout")
    for name, function in zip(OBSERVATION_TERMS, functions, strict=True):
        term = policy[name]
        params = (
            {"command_name": "base_velocity"} if name == "velocity_commands" else {}
        )
        if (
            term.get("func") != f"isaaclab.envs.mdp.observations:{function}"
            or term.get("params") != params
        ):
            raise ValueError(f"Unsupported observation semantics: {name}")
        if (
            any(term.get(key) != "null" for key in ("scale", "clip", "modifiers"))
            or term.get("history_length") != "0"
        ):
            raise ValueError(f"Unsupported observation transform: {name}")
    action = saved["actions"].get("joint_pos", {})
    if set(saved["actions"]) != {"joint_pos"} or any(
        action.get(key) != value
        for key, value in {
            "class_type": "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction",
            "asset_name": "robot",
            "scale": "0.25",
            "clip": "null",
            "use_default_offset": "true",
            "joint_names": [".*"],
            "preserve_order": "false",
        }.items()
    ):
        raise ValueError("Unsupported stock action transform")
    actor, _ = load_reference_actor(
        checkpoint, read_yaml_data(checkpoint.parent / "params/agent.yaml")
    )
    manifest = interface_manifest(
        teacher_sha256=evidence["sha256"]["checkpoint"],
        env_sha256=evidence["sha256"]["env.yaml"],
        joint_names=runtime["joint_names"],
    )
    return (
        OperatorVelocityStudent(actor),
        manifest,
        {
            "teacher_screen": evidence["status"],
            "teacher_passed": evidence["passed"],
            "student_status": "UNTRAINED_NOT_RUN",
            "behavior_validated": False,
        },
    )


def _check_tensor(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if (
        value.shape != shape
        or not value.is_floating_point()
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be a finite floating tensor of shape {shape}")


class CausalOperatorHistory:
    """Delivered-frame history with an explicit reset mask on every push."""

    def __init__(self):
        self.frames = None

    def push(self, frame: torch.Tensor, reset_mask: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 2 or len(frame) < 1:
            raise ValueError("Expected a nonempty batched sensor frame")
        _check_tensor(frame, (len(frame), FRAME_DIM), "sensor frame")
        if (
            reset_mask.shape != (len(frame),)
            or reset_mask.dtype != torch.bool
            or reset_mask.device != frame.device
        ):
            raise ValueError("Reset mask must be a matching boolean device vector")
        delivered = frame.detach()
        if self.frames is None:
            if not reset_mask.all():
                raise ValueError(
                    "The first delivered frame must reset every environment"
                )
            self.frames = delivered[:, None, :].repeat(1, HISTORY_LENGTH, 1)
        else:
            if (
                self.frames.shape[0] != len(frame)
                or self.frames.device != frame.device
                or self.frames.dtype != frame.dtype
            ):
                raise ValueError(
                    "History batch, device and dtype cannot silently change"
                )
            self.frames = torch.cat((self.frames[:, 1:], delivered[:, None]), dim=1)
            self.frames[reset_mask] = delivered[reset_mask, None]
        # No caller or simulator buffer may mutate the retained history.
        return self.frames.clone()


class OperatorVelocityStudent(nn.Module):
    """Frozen stock motor with an explicit, causally inferred velocity input.

    Estimator starts untrained. There is deliberately no oracle fallback in
    forward(), no hidden scene reference and no action override on a bad estimate.
    This module must not be deployed without a trained and validated estimator.
    """

    def __init__(self, motor: nn.Sequential):
        super().__init__()
        expected = (48, 128, 128, 128, 12)
        if not isinstance(motor, nn.Sequential) or len(motor) != 7:
            raise ValueError("Require the validated stock 48→12 ELU motor")
        for i, (inputs, outputs) in enumerate(
            zip(expected[:-1], expected[1:], strict=True)
        ):
            layer = motor[2 * i]
            if (
                not isinstance(layer, nn.Linear)
                or layer.in_features != inputs
                or layer.out_features != outputs
                or layer.bias is None
            ):
                raise ValueError("Unexpected motor layer contract")
            if i < 3 and (
                type(motor[2 * i + 1]) is not nn.ELU or motor[2 * i + 1].alpha != 1.0
            ):
                raise ValueError("Require stock ELU activations")
        if any(not torch.isfinite(p).all() for p in motor.parameters()):
            raise ValueError("Nonfinite motor parameters")
        self.motor = copy.deepcopy(motor).eval().requires_grad_(False)
        # Initialize on CPU explicitly so the shadow audit can preserve RNG
        # without initializing or consuming random state on other GPU devices.
        with torch.device("cpu"):
            self.velocity_estimator = nn.Sequential(
                nn.Linear(HISTORY_LENGTH * FRAME_DIM, 128),
                nn.ELU(),
                nn.Linear(128, 64),
                nn.ELU(),
                nn.Linear(64, 3),
            ).to(device=motor[0].weight.device, dtype=motor[0].weight.dtype)

    def train(self, mode=True):
        super().train(mode)
        self.motor.eval()
        return self

    def estimate_velocity(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or len(history) < 1:
            raise ValueError("Expected batched history")
        _check_tensor(history, (len(history), HISTORY_LENGTH, FRAME_DIM), "history")
        estimate = self.velocity_estimator(history.detach().flatten(1))
        _check_tensor(estimate, (len(history), 3), "velocity estimate")
        return estimate

    def _motor_action(
        self, frame: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        if frame.ndim != 2 or len(frame) < 1:
            raise ValueError("Expected batched frame")
        _check_tensor(frame, (len(frame), FRAME_DIM), "sensor frame")
        _check_tensor(velocity, (len(frame), 3), "body velocity")
        action = self.motor(torch.cat((velocity.detach(), frame.detach()), dim=-1))
        _check_tensor(action, (len(frame), 12), "motor action")
        return action

    def oracle_action(
        self, frame: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        """Explicit audit-only path. Never called by student forward()."""
        return self._motor_action(frame, velocity)

    def forward(self, frame: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        estimate = self.estimate_velocity(history)
        if not torch.equal(frame, history[:, -1]):
            raise ValueError(
                "Newest history frame must be this action's delivered frame"
            )
        return self._motor_action(frame, estimate)

    def velocity_loss(
        self, history: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        estimate = self.estimate_velocity(history)
        _check_tensor(target, (len(history), 3), "pre-action velocity label")
        return nn.functional.mse_loss(estimate, target.detach())


class OperatorOracleAudit:
    """Shadow-only runtime audit; the original actor still controls the robot.

    Constructing the unused estimator must not perturb reset/randomization RNG.
    No student-forward call or extra simulator step occurs in this audit.
    """

    def __init__(self, motor: nn.Sequential):
        with torch.random.fork_rng(devices=[]):
            self.bridge = OperatorVelocityStudent(motor).eval()
        self.history = CausalOperatorHistory()
        self.previous_action = None
        self.previous_command = None
        self.steps = self.comparisons = self.cold_resets = self.later_resets = 0
        self.command_changes = 0

    def observe(self, observation, original_action, reset_mask):
        _check_tensor(
            observation, (len(observation), 48), "delivered teacher observation"
        )
        _check_tensor(original_action, (len(observation), 12), "original actor action")
        frames = self.history.push(observation[:, 3:], reset_mask)
        expected_previous = (
            torch.zeros_like(original_action)
            if self.previous_action is None
            else self.previous_action.clone()
        )
        expected_previous[reset_mask] = 0
        if not torch.equal(observation[:, 36:48], expected_previous):
            raise RuntimeError(
                "Previous-action observation differs from delivered raw action/reset"
            )
        if not torch.equal(frames[:, -1], observation[:, 3:]):
            raise RuntimeError("Current delivered frame/history timing mismatch")
        shadow = self.bridge.oracle_action(frames[:, -1], observation[:, :3])
        if not torch.equal(shadow, original_action):
            raise RuntimeError("Oracle bridge actions differ from the original actor")
        if self.steps == 0:
            self.cold_resets += int(reset_mask.sum().item())
        else:
            self.later_resets += int(reset_mask.sum().item())
            self.command_changes += int(
                (observation[:, 9:12] != self.previous_command).any(dim=-1).sum().item()
            )
        self.previous_action = original_action.detach().clone()
        self.previous_command = observation[:, 9:12].detach().clone()
        self.steps += 1
        self.comparisons += len(observation)

    def report(self):
        return {
            "status": "ORACLE_PARITY_PASS" if self.steps else "NOT_RUN",
            "interface_version": VERSION,
            "control_steps": self.steps,
            "action_comparisons": self.comparisons,
            "cold_reset_frames": self.cold_resets,
            "later_reset_frames": self.later_resets,
            "command_changes_observed": self.command_changes,
            "exact_action_equality": True if self.steps else None,
            "controller": "original stock actor; audit never selects actions",
            "student_status": "UNTRAINED_NOT_RUN",
            "scope": "same delivered pre-action observations, previous raw actions, commands and observed resets only; NOT student behavior",
        }


class StockOperatorAdapter:
    """Semantic adapter for the existing stock actor, explicitly oracle-only.

    Previous raw action is supplied from actual action-manager delivery, not from
    an unacknowledged prediction. The host supplies measured joint order/default
    pose and a verified actuator-profile identity; this adapter cannot set gains.
    """

    def __init__(
        self,
        actor,
        *,
        joint_names,
        default_position_rad,
        artifact_sha256,
        actuator_profile,
    ):
        from parkour_lab.learning.controller import ControllerSpec, SensorSpec

        if len(joint_names) != 12:
            raise ValueError("Stock adapter requires twelve named joints")
        _check_tensor(default_position_rad, (12,), "default joint position")
        self.actor = copy.deepcopy(actor).eval().requires_grad_(False)
        self.default_position_rad = default_position_rad.detach().clone()
        self.reset_mask = None
        self.spec = ControllerSpec(
            name="stock_operator_oracle",
            artifact_sha256=artifact_sha256,
            preprocessing_version="stock_48_unscaled_v1",
            joint_names=tuple(joint_names),
            period_s=0.02,
            actuator_profile=actuator_profile,
            sensors={
                "oracle_base_lin_vel": SensorSpec(
                    (3,), "m/s", "body", 0.02, privileged=True
                ),
                "base_ang_vel": SensorSpec((3,), "rad/s", "body", 0.02),
                "projected_gravity": SensorSpec((3,), "unitless", "body", 0.02),
                "joint_position_relative_default": SensorSpec(
                    (12,), "rad", "joint", 0.02
                ),
                "joint_velocity": SensorSpec((12,), "rad/s", "joint", 0.02),
                "stock_previous_raw_action": SensorSpec(
                    (12,), "unitless", "joint", 0.02
                ),
            },
            raw_action_meaning="stock unscaled action; q_target = default_q + 0.25 * raw_action; no clip",
            configuration={
                "default_position_rad": default_position_rad.tolist(),
                "action_scale": 0.25,
                "action_clip": None,
            },
        )

    def reset(self, mask):
        # Feed-forward actor has no latent memory. The previous-action input must
        # still be zero on a reset; host action-manager state owns that value.
        self.reset_mask = mask

    def act(self, inputs):
        from parkour_lab.learning.controller import JointTargets

        sensors = inputs.sensors
        previous = sensors["stock_previous_raw_action"].value
        if self.reset_mask is None or torch.any(previous[self.reset_mask] != 0):
            raise ValueError("Stock previous action must be zero after reset")
        observation = torch.cat(
            (
                sensors["oracle_base_lin_vel"].value,
                sensors["base_ang_vel"].value,
                sensors["projected_gravity"].value,
                inputs.command,
                sensors["joint_position_relative_default"].value,
                sensors["joint_velocity"].value,
                previous,
            ),
            dim=-1,
        )
        action = self.actor(observation)
        return JointTargets(
            self.spec.joint_names, self.default_position_rad + 0.25 * action, action
        )


def controller_preflight(checkpoint):
    """Actual checkpoint, synthetic sensor fixture: exact CPU action/target parity.

    No claim about runtime joint mapping, physics, estimator or deployability.
    Keeps this boundary exercised without touching the archived benchmark path.
    """
    from parkour_lab.learning.controller import ControllerSession, Sample

    try:
        from .operator_benchmark_core import (
            load_reference_actor,
            read_yaml_data,
            file_sha256,
        )
        from .operator_control_trace import controller_record
    except ImportError:
        from operator_benchmark_core import (
            load_reference_actor,
            read_yaml_data,
            file_sha256,
        )
        from operator_control_trace import controller_record

    source_hash = file_sha256(checkpoint)
    with torch.random.fork_rng(devices=[]):
        actor, _ = load_reference_actor(
            checkpoint, read_yaml_data(checkpoint.parent / "params/agent.yaml")
        )
    generator = torch.Generator().manual_seed(123)
    default = torch.linspace(-0.3, 0.3, 12)
    adapter = StockOperatorAdapter(
        actor,
        joint_names=tuple(f"fixture_joint_{i}" for i in range(12)),
        default_position_rad=default,
        artifact_sha256=source_hash,
        actuator_profile="CPU_SYNTHETIC_NOT_A_RUNTIME_MOTOR_CONFIGURATION",
    )
    session = ControllerSession(
        adapter,
        joint_names=adapter.spec.joint_names,
        actuator_profile=adapter.spec.actuator_profile,
        allow_privileged=True,
        capture=True,
    )
    previous = torch.zeros(4, 12)
    slices = ((0, 3), (3, 6), (6, 9), (12, 24), (24, 36), (36, 48))
    for step in range(6):
        observation = torch.randn(4, 48, generator=generator)
        reset = torch.tensor([step in (0, 3), step == 0, step == 0, step in (0, 4)])
        previous[reset] = 0
        observation[:, 36:48] = previous
        now = step * 0.02
        samples = {
            name: Sample(
                observation[:, start:end],
                now,
                torch.ones(4, dtype=torch.bool),
                spec.units,
                spec.frame,
                spec.privileged,
            )
            for (name, spec), (start, end) in zip(
                adapter.spec.sensors.items(), slices, strict=True
            )
        }
        command = observation[:, 9:12]
        result = session.step(
            time_s=now,
            command=command,
            command_time_s=now,
            sensors=samples,
            reset_mask=reset,
        )
        with torch.inference_mode():
            expected = actor(observation)
        if not torch.equal(result.raw_action, expected) or not torch.equal(
            result.position_rad, default + 0.25 * expected
        ):
            raise RuntimeError("Stock controller adapter changed actions or targets")
        record = controller_record(
            session,
            requested_command=command,
            requested_at_s=now,
            delivered_position_rad=None,
            delivery_time_s=None,
            command_source="synthetic_cpu_fixture",
            safety_events=(),
        )
        json.dumps(record, allow_nan=False)
        if record["delivered_position_rad"] is not None:
            raise RuntimeError("CPU fixture falsely records actuator delivery")
        previous = result.raw_action.clone()
    if file_sha256(checkpoint) != source_hash:
        raise ValueError("Checkpoint changed during controller preflight")
    return {
        "status": "CPU_ADAPTER_PARITY_PASS",
        "source_sha256": adapter.spec.artifact_sha256,
        "interface_sha256": session.interface_sha256,
        "control_steps": 6,
        "action_comparisons": 24,
        "joint_target_comparisons": 288,
        "exact_action_and_target_equality": True,
        "fixture": "synthetic sensors and joint names, partial resets, changed body twist",
        "privileged": True,
        "actuator_delivery": "NOT_RUN",
        "environment_transitions": 0,
        "learning_updates": 0,
        "exit_allowed": False,
    }


def _runtime_motor_binding(env):
    """Bind the resolved native stock motor, not merely an action tensor width."""
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    cfg, descriptor = term.cfg, term.IO_descriptor
    joints = tuple(robot.joint_names)
    default = robot.data.default_joint_pos
    _check_tensor(default, (env.num_envs, 12), "runtime default joint position")
    _check_tensor(
        robot.data.default_joint_vel, default.shape, "runtime default joint velocity"
    )
    if (
        tuple(env.action_manager.active_terms) != ("joint_pos",)
        or env.action_manager.total_action_dim != 12
        or len(joints) != 12
        or len(set(joints)) != 12
        or tuple(descriptor.joint_names) != joints
        or cfg.asset_name != "robot"
        or cfg.joint_names != [".*"]
        or cfg.preserve_order
        or not cfg.use_default_offset
        or cfg.scale != 0.25
        or descriptor.scale != 0.25
        or cfg.clip is not None
        or descriptor.clip is not None
        or cfg.class_type.__name__ != "JointPositionAction"
        or cfg.class_type.__module__ != "isaaclab.envs.mdp.actions.joint_actions"
        or not torch.equal(default, default[0].expand_as(default))
        or torch.any(robot.data.default_joint_vel != 0)
        or not torch.equal(
            torch.as_tensor(
                descriptor.offset, device=default.device, dtype=default.dtype
            ),
            default[0],
        )
    ):
        raise ValueError(
            "Native stock joint order, default pose/velocity or action transform differs"
        )
    actuators = {}
    covered = []
    for name, actuator in robot.actuators.items():
        covered.extend(actuator.joint_names)
        parameters = {}
        for parameter in (
            "stiffness",
            "damping",
            "effort_limit",
            "velocity_limit",
            "effort_limit_sim",
            "velocity_limit_sim",
            "armature",
            "friction",
        ):
            value = getattr(actuator, parameter)
            _check_tensor(value, (env.num_envs, len(actuator.joint_names)), parameter)
            parameters[parameter] = value.detach().cpu().tolist()
        actuators[name] = {
            "joint_names": list(actuator.joint_names),
            "configuration": actuator.cfg.to_dict(),
            "resolved_parameters": parameters,
        }
    if sorted(covered) != sorted(joints):
        raise ValueError("Native actuators must cover each motor joint exactly once")
    binding = {
        "joint_names": list(joints),
        "default_position_rad": default[0].cpu().tolist(),
        "action": cfg.to_dict(),
        "actuators": actuators,
        "step_dt_s": env.step_dt,
        "physics_dt_s": env.physics_dt,
        "decimation": env.cfg.decimation,
    }
    digest = hashlib.sha256(
        json.dumps(binding, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return binding, digest


def run_controller_rollout(env, actor, checkpoint_sha256, *, is_running, steps=200):
    """Bounded native integration check; caller owns reset, artifacts and cleanup.

    Independent robot samples reach the adapter, whose actions alone drive the
    existing native motor. The original actor is a same-frame shadow comparator.
    No terrain tensor, reward, termination state or environment reaches act().
    """
    import numpy as np
    from parkour_lab.learning.controller import ControllerSession, Sample

    try:
        from .operator_benchmark import command_observation, validate_motor_trace
        from .operator_benchmark_core import DT, OBSERVATION_TERMS
    except ImportError:
        from operator_benchmark import command_observation, validate_motor_trace
        from operator_benchmark_core import DT, OBSERVATION_TERMS

    if (
        type(steps) is not int
        or steps != 200
        or not 20 <= env.num_envs <= 80
        or env.num_envs % 20
        or env.cfg.decimation != 4
        or not math.isclose(env.step_dt, DT, abs_tol=1e-9)
        or not math.isclose(env.physics_dt, 0.005, abs_tol=1e-9)
        or env.cfg.observations.policy.enable_corruption
        or tuple(env.observation_manager.active_terms["policy"]) != OBSERVATION_TERMS
        or actor.training
        or any(p.requires_grad for p in actor.parameters())
    ):
        raise ValueError(
            "Runtime fixture requires 200 steps, 20–80 environments, uncorrupted stock48 and frozen motor"
        )
    command = env.command_manager.get_term("base_velocity")
    if (
        command.cfg.heading_command
        or command.cfg.rel_heading_envs
        or command.cfg.rel_standing_envs
    ):
        raise ValueError(
            "Runtime body twist must not receive heading or standing assistance"
        )
    binding, motor_hash = _runtime_motor_binding(env)
    robot = env.scene["robot"].data
    term = env.action_manager.get_term("joint_pos")
    adapter = StockOperatorAdapter(
        actor,
        joint_names=tuple(binding["joint_names"]),
        default_position_rad=robot.default_joint_pos[0],
        artifact_sha256=checkpoint_sha256,
        actuator_profile="native_motor_sha256:" + motor_hash,
    )
    session = ControllerSession(
        adapter,
        joint_names=adapter.spec.joint_names,
        actuator_profile=adapter.spec.actuator_profile,
        allow_privileged=True,
    )
    capture = env.operator_capture
    if capture.enabled or capture.samples or capture.control_trace is not None:
        raise ValueError("Runtime fixture requires a fresh disabled native recorder")

    class TargetAudit:
        def __init__(self):
            self.target = self.raw = None
            self.current, self.substeps, self.targets = [], [], []

        def after_substep(self):
            # This native hook precedes scene.update. Target/actuator command
            # buffers are current; cached body/joint measurements are not.
            if (
                self.target is None
                or len(self.current) >= 4
                or any(
                    not torch.equal(value, self.target)
                    for value in (robot.joint_pos_target, term.processed_actions)
                )
                or not torch.equal(env.action_manager.action, self.raw)
            ):
                raise RuntimeError(
                    "Native actuator substep differs from controller output"
                )
            self.current.append(robot.joint_pos_target.detach().cpu().numpy().copy())

        def after_step(self):
            if len(self.current) != 4 or not torch.equal(
                robot.joint_pos_target, self.target
            ):
                raise RuntimeError(
                    "Missing native substeps or changed pre-reset target"
                )
            self.substeps.append(np.stack(self.current, axis=1))
            self.targets.append(self.target.detach().cpu().numpy().copy())
            self.current = []
            self.target = self.raw = None

    audit = TargetAudit()
    capture.motor_parity = True
    capture.control_trace = audit
    capture.enabled = True
    reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    forced = torch.zeros_like(reset)
    forced[: env.num_envs // 2] = True
    terrain_samples = []
    partial_resets = 0
    try:
        with torch.inference_mode():
            for step in range(steps):
                if not is_running():
                    raise RuntimeError(
                        f"Simulator closed at controller step {step}/{steps}"
                    )
                desired = robot.joint_pos.new_tensor(
                    CONTROLLER_ROLLOUT_COMMANDS[step // 40]
                ).repeat(env.num_envs, 1)
                if 40 <= step < 120:
                    desired[: env.num_envs // 2, 0] = (
                        0.0  # Predeclared pure pivots alongside the arc rows.
                    )
                observation = command_observation(env, desired)
                previous = env.action_manager.action
                if not torch.equal(observation[:, 36:48], previous) or torch.any(
                    previous[reset] != 0
                ):
                    raise RuntimeError(
                        "Native delivered previous action/reset mismatch"
                    )
                values = (
                    robot.root_lin_vel_b,
                    robot.root_ang_vel_b,
                    robot.projected_gravity_b,
                    robot.joint_pos - robot.default_joint_pos,
                    robot.joint_vel,
                    previous,
                )
                independent = torch.cat((*values[:3], desired, *values[3:]), dim=-1)
                if not torch.equal(independent, observation):
                    raise RuntimeError(
                        "Independent semantic provider differs from native stock48 observation"
                    )
                now = step * DT
                samples = {
                    name: Sample(
                        value,
                        now,
                        torch.ones_like(reset),
                        spec.units,
                        spec.frame,
                        spec.privileged,
                    )
                    for (name, spec), value in zip(
                        adapter.spec.sensors.items(), values, strict=True
                    )
                }
                output = session.step(
                    time_s=now,
                    command=desired,
                    command_time_s=now,
                    sensors=samples,
                    reset_mask=reset,
                )
                if not torch.equal(output.raw_action, actor(observation)):
                    raise RuntimeError(
                        "Controller adapter differs from same-frame shadow actor"
                    )
                capture.observation = observation.detach().clone()
                audit.target, audit.raw = output.position_rad, output.raw_action
                if step == 99:
                    env.episode_length_buf[forced] = env.max_episode_length - 1
                # Native action processing remains the exact stock affine map;
                # its four delivered target buffers must equal the named output.
                observed, _, terminated, timed_out, _ = env.step(output.raw_action)
                if len(audit.targets) != step + 1:
                    raise RuntimeError(
                        "Native post-step/pre-reset controller capture missing"
                    )
                reset = (terminated | timed_out).detach().clone()
                partial_resets += int(reset.any() and not reset.all())
                if step == 99 and (
                    not timed_out[forced].all() or not reset.any() or reset.all()
                ):
                    raise RuntimeError(
                        "Injected timeouts did not exercise a partial native reset"
                    )
                if "terrain" not in observed:
                    raise RuntimeError("Missing privileged terrain fixture")
                terrain = observed["terrain"]
                _check_tensor(
                    terrain, (env.num_envs, 264), "privileged terrain fixture"
                )
                if not torch.all((terrain[:, 132:] == 0) | (terrain[:, 132:] == 1)):
                    raise RuntimeError(
                        "Privileged terrain fixture validity is not binary"
                    )
                terrain_samples.append(terrain.detach().cpu().numpy().copy())
    finally:
        capture.enabled = False
        capture.control_trace = None
    trace = capture.finish()
    motor_interface = validate_motor_trace(trace)
    trace.update(
        adapter_target=np.stack(audit.targets),
        joint_target_substeps=np.stack(audit.substeps),
        decision_time_s=np.arange(steps, dtype=np.float64) * DT,
        forced_timeout=np.zeros((steps, env.num_envs), dtype=np.bool_),
        terrain_observation=np.stack(terrain_samples),
    )
    trace["forced_timeout"][99, : env.num_envs // 2] = True
    if not np.array_equal(trace["adapter_target"], trace["joint_target"]):
        raise RuntimeError("Recorded pre-reset targets differ from adapter outputs")
    return {
        "status": "SIM_ADAPTER_PARITY_PASS",
        "source_sha256": checkpoint_sha256,
        "control_steps": steps,
        "action_comparisons": steps * env.num_envs,
        "substep_target_comparisons": steps * env.num_envs * 4 * 12,
        "forced_timeout_step": 99,
        "forced_timeout_count": env.num_envs // 2,
        "partial_reset_steps": partial_resets,
        "physical_failure_count": int(trace["terminated"].sum()),
        "other_timeout_count": int(
            (trace["time_out"] & ~trace["forced_timeout"]).sum()
        ),
        "motor_interface": motor_interface,
        "motor_binding": binding,
        "controller_manifest": session.manifest,
        "interface_sha256": session.interface_sha256,
        "motor_binding_sha256": motor_hash,
        "command_phases": [list(p) for p in CONTROLLER_ROLLOUT_COMMANDS],
        "pivot_env_ids": list(range(env.num_envs // 2)),
        "pivot_phase_steps": [40, 120],
        "exact_action_and_target_equality": True,
        "phase_steps": 40,
        "policy_inputs": "stock48 with simulator velocity; explicit oracle only",
        "terrain_fixture_steps": len(terrain_samples),
        "terrain_sampling": "env.step returned observation, after auto-reset; never delivered to actor",
        "terrain_actor_access": False,
        "capture": "four native target substeps and post-physics/pre-reset state",
        "environment_transitions": steps * env.num_envs,
        "learning_updates": 0,
        "exit_allowed": False,
        "scope": "interface integration, not locomotion/terrain acceptance or hardware delivery",
    }, trace
