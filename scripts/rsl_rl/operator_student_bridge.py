"""Versioned, CPU-testable bridge from the stock operator motor to history.

This is interface groundwork, not an RMA trainer or qualified student policy.
The oracle path preserves the 48-D motor exactly. The student path accepts only
45-D sensor/command frames and their causal history, replacing the three oracle
linear-velocity inputs with a separately supervised estimate. No terrain or
dynamics latent is implied, and no automatic checkpoint conversion is performed.
"""

from __future__ import annotations

import copy
import hashlib
import json
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
