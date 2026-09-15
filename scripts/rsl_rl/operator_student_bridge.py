"""Operator training/evaluation integration and legacy flat-history diagnostics.

The simulator-independent recurrent policy and causal adapter live in
``parkour_lab.learning.recurrent_operator`` and are re-exported for existing
callers. Simulator capture, training and development scoring remain here. The
legacy oracle path preserves the 48-D motor exactly; its flat student estimates
velocity from history. Neither path implies hardware or terrain acceptance.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

# Compatibility exports for existing training, audit and evaluation callers.
from parkour_lab.learning.recurrent_operator import (
    RECURRENT_OPERATOR_VERSION,
    RecurrentOperatorAdapter,
    _recurrent_actor_sha256,
    build_recurrent_operator_policy as build_recurrent_operator_policy,
    load_recurrent_checkpoint as load_recurrent_checkpoint,
    recurrent_policy_config,
)


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


def recurrent_evaluation_protocol(*, difficulty_range=None):
    """Predeclared clean-sensor first-attempt screen, not an acceptance gate."""
    phases = (
        ("cold_stand", 1, (0, 0, 0)),
        ("forward", 4, (0.55, 0, 0)),
        ("arc_positive", 2, (0.55, 0, 0.45)),
        ("arc_negative", 2, (0.55, 0, -0.45)),
        ("stop_after_arcs", 2, (0, 0, 0)),
        ("reverse_flat_or_forward_rough", 3, (-0.3, 0, 0)),
        ("pivot_positive", 2, (0, 0, 0.5)),
        ("pivot_negative", 2, (0, 0, -0.5)),
        ("final_stop", 2, (0, 0, 0)),
    )
    protocol = {
        "version": "go2_operator_proprio_screen_v1",
        "seed": 43,
        "num_envs": 80,
        "period_s": 0.02,
        "steps": 1000,
        "settling_s": 0.4,
        "noise": False,
        "phases": [
            {
                "name": name,
                "duration_s": seconds,
                "flat_command": list(command),
                "rough_command": list((0.45, 0, 0) if index == 5 else command),
            }
            for index, (name, seconds, command) in enumerate(phases)
        ],
        "trial": "one cold-start first attempt per environment; first termination ends scoring permanently",
        "finished_rows": "zero body-twist packets for excluded housekeeping, not zero motor actions",
        "scope": "deterministic actor mean, fixed packets and seed; clean-sensor development only, not robust sensing or terrain acceptance",
        "exit_allowed": False,
    }
    if difficulty_range is not None:
        if (
            len(difficulty_range) != 2
            or any(
                isinstance(value, bool) or not math.isfinite(value)
                for value in difficulty_range
            )
            or not 0 <= difficulty_range[0] <= difficulty_range[1] <= 1
        ):
            raise ValueError(
                "Evaluation difficulty must be two finite ordered values in [0, 1]"
            )
        protocol.update(
            version="go2_operator_proprio_terrain_probe_v1",
            difficulty_range=list(difficulty_range),
            native_diagnostics="joint_actuator_contact_v1",
            scope=(
                "Frozen mean-policy terrain-amplitude probe on the seed-43 development "
                "layouts and unchanged clean-sensor command tape; no learning, curriculum "
                "promotion, held-out confirmation or stair/terrain acceptance"
            ),
        )
    return protocol


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


def run_recurrent_training(env, runner_cfg, output, *, is_running, iterations):
    """Fresh native PPO with bounded-memory runtime audits; no acceptance claim.

    The caller owns the environment, protocol and artifact publication. Native
    PPO/rollout storage are unchanged. Only observation delivery, update checks,
    metrics and completed-update checkpoint names are adapted here.
    """
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from parkour_lab.learning.controller import ControllerSession, Sample
    from parkour_lab.tasks.manager_based.parkour_lab.mdp.terrain.operator_terrain import (
        BORDER_WIDTH,
        FLAT_BAND_HALF_WIDTH,
        PROFILE_BY_COLUMN,
        SPAWN_HALF_WIDTH,
    )

    expected_groups = {"policy": ["proprio"], "critic": ["policy", "terrain"]}
    command = env.command_manager.get_term("base_velocity")
    if (
        type(iterations) is not int
        or iterations < 1
        or type(runner_cfg.get("save_interval")) is not int
        or runner_cfg["save_interval"] < 1
        or runner_cfg["policy"] != recurrent_policy_config()
        or runner_cfg["obs_groups"] != expected_groups
        or runner_cfg.get("resume")
        or runner_cfg["algorithm"]["class_name"] != "PPO"
        or runner_cfg["algorithm"].get("rnd_cfg") is not None
        or runner_cfg["algorithm"].get("symmetry_cfg") is not None
        or tuple(env.observation_manager.active_terms["proprio"])
        != tuple(n for n, _ in FRAME_TERMS)
        or tuple(env.observation_manager.active_terms["policy"])
        != ("base_lin_vel", *(n for n, _ in FRAME_TERMS))
        or env.cfg.decimation != 4
        or abs(env.step_dt - 0.02) > 1e-9
        or abs(env.physics_dt - 0.005) > 1e-9
        or command.cfg.heading_command
        or command.cfg.rel_heading_envs
        or command.cfg.rel_standing_envs
        or env.cfg.curriculum.terrain_levels is not None
        or env.cfg.scene.terrain.terrain_generator.num_rows != 1
    ):
        raise ValueError(
            "Require fresh native GRU/PPO with direct commands and fixed easy terrain"
        )
    recipe = copy.deepcopy(runner_cfg)
    output = Path(output)
    profiles = tuple(dict.fromkeys(PROFILE_BY_COLUMN))
    columns = env.scene.terrain.terrain_types
    if tuple(profiles) != (
        "plane",
        "rough_flat",
        "hills",
        "step_hills",
        "tilted_ramps",
    ) or torch.any((columns < 0) | (columns >= 20)):
        raise ValueError("Require the declared five-profile supported terrain layout")
    profile_ids = columns // 4
    counts = torch.zeros(5, 9, dtype=torch.int64, device=env.device)
    errors = torch.zeros(5, 3, dtype=torch.float64, device=env.device)
    stats = {
        "learning_updates": 0,
        "control_steps": 0,
        "partial_reset_steps": 0,
        "adapter_comparisons": 0,
    }
    robot, term = env.scene["robot"].data, env.action_manager.get_term("joint_pos")

    def metrics():
        return {
            name: dict(
                zip(
                    (
                        "samples",
                        "moving_commands",
                        "moving_nonflat_region",
                        "physical_failures",
                        "workspace_timeouts",
                        "timeouts",
                        "stop_commands",
                        "pivot_commands",
                        "reverse_commands",
                    ),
                    counts[i].cpu().tolist(),
                ),
                mean_abs_twist_error=(errors[i] / counts[i, 0].clamp_min(1))
                .cpu()
                .tolist(),
            )
            for i, name in enumerate(profiles)
        }

    class DeliveredEnvironment(RslRlVecEnvWrapper):
        def __init__(self):
            super().__init__(env, clip_actions=None)
            # Sample once after the wrapper's native reset, then never resample
            # corruption in get_observations() or in the shadow adapter.
            self.observations = super().get_observations()
            self.validate_observations(self.observations)
            self.done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            self.runner = self.session = None

        def get_observations(self):
            return self.observations

        def validate_observations(self, observation):
            for name, width in (("proprio", 45), ("policy", 48), ("terrain", 264)):
                _check_tensor(observation[name], (self.num_envs, width), name)

        def step(self, actions):
            if not is_running():
                raise RuntimeError("Simulator closed during recurrent training")
            frame = self.observations["proprio"]
            _check_tensor(actions, (self.num_envs, 12), "sampled PPO action")
            desired = command.command
            if (
                not torch.equal(frame[:, 6:9], desired)
                or not torch.equal(frame[:, 33:], env.action_manager.action)
                or torch.any(frame[self.done, 33:] != 0)
            ):
                raise RuntimeError(
                    "Delivered command/previous action/reset differs from the native actor frame"
                )
            if stats["learning_updates"] == 0:
                now = stats["control_steps"] * env.step_dt
                slices = ((0, 3), (3, 6), (9, 21), (21, 33), (33, 45))
                samples = {
                    name: Sample(
                        frame[:, lo:hi],
                        now,
                        torch.ones_like(self.done),
                        spec.units,
                        spec.frame,
                    )
                    for (name, spec), (lo, hi) in zip(
                        self.session.controller.spec.sensors.items(),
                        slices,
                        strict=True,
                    )
                }
                inferred = self.session.step(
                    time_s=now,
                    command=desired,
                    command_time_s=now,
                    sensors=samples,
                    reset_mask=self.done,
                )
                if not torch.allclose(
                    inferred.raw_action,
                    self.runner.alg.policy.action_mean,
                    atol=1e-6,
                    rtol=0,
                ):
                    raise RuntimeError(
                        "Native recurrent actor differs from same-frame extracted adapter"
                    )
                stats["adapter_comparisons"] += self.num_envs
            local = (robot.root_pos_w[:, :2] - env.scene.env_origins[:, :2]).abs()
            nonflat = (
                (profile_ids != 0)
                & (local.max(dim=1).values > SPAWN_HALF_WIDTH)
                & (local[:, 1] > FLAT_BAND_HALF_WIDTH)
                & (local.max(dim=1).values < 8.0 - BORDER_WIDTH)
            )
            moving = desired[:, :2].norm(dim=-1) > 0.1
            exposure = (
                nonflat & moving & (robot.root_lin_vel_b[:, :2].norm(dim=-1) > 0.1)
            )
            tracking = torch.cat(
                (robot.root_lin_vel_b[:, :2], robot.root_ang_vel_b[:, 2:3]), dim=-1
            )
            errors.index_add_(0, profile_ids, (tracking - desired).abs().double())
            counts[:, 0] += torch.bincount(profile_ids, minlength=5)
            counts[:, 1] += torch.bincount(profile_ids[moving], minlength=5)
            counts[:, 2] += torch.bincount(profile_ids[exposure], minlength=5)
            for index, mask in (
                (6, desired.abs().amax(dim=-1) < 1e-6),
                (7, ~moving & (desired[:, 2].abs() > 0.1)),
                (8, desired[:, 0] < -0.05),
            ):
                counts[:, index] += torch.bincount(profile_ids[mask], minlength=5)
            observation, reward, done, extras = super().step(actions)
            # Native wrappers can report both a physical failure and a time
            # limit on the same step. Only genuine truncations bootstrap PPO.
            if "time_outs" in extras:
                extras["time_outs"] = extras["time_outs"] & ~env.reset_terminated
            self.validate_observations(observation)
            _check_tensor(reward, (self.num_envs,), "native reward")
            expected = robot.default_joint_pos + 0.25 * actions
            if not torch.equal(term.processed_actions, expected):
                raise RuntimeError(
                    "Native processed joint target differs from the stock affine map"
                )
            self.done = done.bool()
            if not torch.equal(
                robot.joint_pos_target[~self.done], expected[~self.done]
            ):
                raise RuntimeError("Native surviving joint target buffer differs")
            previous = actions.clone()
            previous[self.done] = 0
            if not torch.equal(observation["proprio"][:, 33:], previous):
                raise RuntimeError(
                    "Next frame does not contain the previous delivered action"
                )
            self.observations = observation
            for index, mask in (
                (3, env.reset_terminated),
                (4, env.termination_manager.get_term("procedural_workspace")),
                (5, env.reset_time_outs),
            ):
                counts[:, index] += torch.bincount(
                    profile_ids[mask.bool()], minlength=5
                )
            stats["partial_reset_steps"] += int(self.done.any() and not self.done.all())
            stats["control_steps"] += 1
            return observation, reward, done, extras

    wrapped = DeliveredEnvironment()
    binding, motor_hash = _runtime_motor_binding(env)

    def extract(policy):
        return RecurrentOperatorAdapter(
            policy,
            joint_names=tuple(binding["joint_names"]),
            default_position_rad=robot.default_joint_pos[0],
            artifact_sha256=_recurrent_actor_sha256(policy),
            actuator_profile="native_motor_sha256:" + motor_hash,
        )

    class TrainingRunner(OnPolicyRunner):
        saved_updates = 0
        published_updates = 0
        checkpoint = None

        def save(self, path, infos=None):
            completed = stats["learning_updates"]
            if completed == self.published_updates or (
                completed != iterations
                and completed % self.save_interval
                and completed % 50
            ):
                return
            metadata = {
                "policy_version": RECURRENT_OPERATOR_VERSION,
                **stats,
                "environment_transitions": stats["control_steps"] * env.num_envs,
                "recipe": recipe,
                "controller_manifest": extract(self.alg.policy).spec.manifest(),
                "actor_artifact_identity": "named_actor_state_tensor_sha256",
                "motor_binding_sha256": motor_hash,
                "metrics": metrics(),
                "exit_allowed": False,
                "resume_supported": False,
            }
            if completed == iterations or completed % self.save_interval == 0:
                self.checkpoint = f"model_{completed}.pt"
                super().save(
                    str(output / self.checkpoint),
                    infos={
                        "learning_updates": completed,
                        "recurrent_training": metadata,
                    },
                )
                self.saved_updates = completed
            # Keep inexpensive progress visible even with sparse checkpoints.
            progress = {
                **metadata,
                "last_checkpoint": self.checkpoint,
                "last_checkpoint_learning_updates": self.saved_updates,
            }
            temporary = output / "training_progress.json.tmp"
            temporary.write_text(json.dumps(progress, indent=2, allow_nan=False) + "\n")
            temporary.replace(output / "training_progress.json")
            self.published_updates = completed

        def log(self, locs, *args, **kwargs):
            super().log(locs, *args, **kwargs)
            self.save(
                None
            )  # Exact completed-update milestones, not native zero-based filenames.

    runner = TrainingRunner(wrapped, copy.deepcopy(runner_cfg), str(output), env.device)
    wrapped.runner = runner
    adapter = extract(runner.alg.policy)
    wrapped.session = ControllerSession(
        adapter,
        joint_names=adapter.spec.joint_names,
        actuator_profile=adapter.spec.actuator_profile,
    )
    groups = ("memory_a.", "memory_c.", "actor.", "critic.")
    initial = {
        prefix: torch.cat(
            [
                p.detach().flatten()
                for name, p in runner.alg.policy.named_parameters()
                if name.startswith(prefix)
            ]
        ).clone()
        for prefix in groups
    }
    native_update = runner.alg.update

    def checked_update():
        losses = native_update()
        if (
            not losses
            or any(not math.isfinite(value) for value in losses.values())
            or not all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
        ):
            raise RuntimeError("Nonfinite native PPO loss or policy parameters")
        if stats["learning_updates"] == 0:
            for prefix, before in initial.items():
                after = torch.cat(
                    [
                        p.detach().flatten()
                        for name, p in runner.alg.policy.named_parameters()
                        if name.startswith(prefix)
                    ]
                )
                if torch.equal(before, after):
                    raise RuntimeError(
                        f"First native PPO update left {prefix} unchanged"
                    )
            initial.clear()
        stats["learning_updates"] += 1
        stats["last_losses"] = losses
        return losses

    runner.alg.update = checked_update
    try:
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
    finally:
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
    if (
        stats["learning_updates"] != iterations
        or stats["control_steps"] != iterations * runner.num_steps_per_env
        or runner.saved_updates != iterations
    ):
        raise RuntimeError("Incomplete recurrent learning budget or final checkpoint")
    optimizer_steps = (
        iterations * runner.alg.num_learning_epochs * runner.alg.num_mini_batches
    )
    if set(runner.alg.optimizer.state) != set(runner.alg.policy.parameters()):
        raise RuntimeError("Native Adam state does not cover every policy parameter")
    for state in runner.alg.optimizer.state.values():
        if state["step"].item() != optimizer_steps or any(
            not torch.isfinite(value).all() for value in state.values()
        ):
            raise RuntimeError("Native Adam update count or moments are invalid")
    return {
        "status": "TRAINING_COMPLETED_NOT_ACCEPTED",
        "policy_version": RECURRENT_OPERATOR_VERSION,
        **stats,
        "environment_transitions": stats["control_steps"] * env.num_envs,
        "checkpoint": runner.checkpoint,
        "optimizer_steps": optimizer_steps,
        "updated_parameter_groups": list(groups),
        "metrics": metrics(),
        "controller_manifest": extract(runner.alg.policy).spec.manifest(),
        "actor_artifact_identity": "named_actor_state_tensor_sha256",
        "motor_binding": binding,
        "motor_binding_sha256": motor_hash,
        "metric_scope": "pooled pre-action tracking and moving exposure outside flat pad/band/border; command counts distinguish stops/pivots/reverse; not terrain traversal or acceptance",
        "adapter_scope": "initial frozen-policy rollout, same noisy frames and natural resets; deterministic means within 1e-6, not sampled actions or hardware validation",
        "target_scope": "native processed affine targets and surviving joint target buffers, not a per-substep actuator-delivery audit",
        "exit_allowed": False,
    }


def summarize_recurrent_evaluation(trace, protocol=None, *, version=3):
    """Pure first-attempt diagnostics; versions 1 and 2 explicitly replay old schemas."""
    import numpy as np

    if type(version) is not int or version not in (1, 2, 3):
        raise ValueError("Unsupported recurrent evaluation summary version")
    protocol = recurrent_evaluation_protocol() if protocol is None else protocol
    native = protocol.get("native_diagnostics") == "joint_actuator_contact_v1"
    if native and version != 3:
        raise ValueError("Native terrain probes require summary version 3")
    names = ("plane", "rough_flat", "hills", "step_hills", "tilted_ramps")
    shape = (protocol["steps"], protocol["num_envs"])
    phase_ids = np.repeat(
        np.arange(len(protocol["phases"])),
        [round(p["duration_s"] / protocol["period_s"]) for p in protocol["phases"]],
    )
    profiles = trace["terrain_profile_id"]
    if (
        profiles.shape != (shape[1],)
        or not np.issubdtype(profiles.dtype, np.integer)
        or np.any((profiles < 0) | (profiles >= len(names)))
        or not np.array_equal(np.bincount(profiles, minlength=5), np.full(5, 16))
        or not np.array_equal(trace["phase_index"], phase_ids)
    ):
        raise ValueError("Evaluation terrain assignment or phase tape differs")
    for key, width in (
        ("command", 3),
        ("position", 3),
        ("pre_position", 3),
        ("quaternion", 4),
        ("pre_quaternion", 4),
        ("linear_velocity_b", 3),
        ("angular_velocity_b", 3),
        ("base_height_ray", 3),
    ):
        if trace[key].shape != (*shape, width):
            raise ValueError(f"Invalid evaluation field shape: {key}")
    for key in ("terminated", "time_out", "procedural_workspace"):
        if trace[key].shape != shape or trace[key].dtype != np.bool_:
            raise ValueError(f"Invalid evaluation termination field: {key}")
    if trace["env_origins"].shape != (shape[1], 3):
        raise ValueError("Invalid evaluation environment origins")
    valid = trace["valid_first_attempt"]
    if (
        valid.shape != (protocol["steps"], protocol["num_envs"])
        or valid.dtype != np.bool_
    ):
        raise ValueError("Invalid first-attempt evaluation mask")
    done = trace["terminated"] | trace["time_out"]
    expected_valid = np.concatenate(
        (np.ones_like(done[:1]), ~np.maximum.accumulate(done[:-1], axis=0))
    )
    if not np.array_equal(valid, expected_valid):
        raise ValueError("First-attempt mask includes post-reset replacement trials")
    expected_command = np.asarray(
        [
            [
                phase["flat_command"] if profile < 2 else phase["rough_command"]
                for profile in profiles
            ]
            for phase in protocol["phases"]
        ],
        dtype=trace["command"].dtype,
    )[phase_ids]
    expected_command[~valid] = 0
    if not np.array_equal(trace["command"], expected_command):
        raise ValueError(
            "Recorded command differs from the first-attempt operator tape"
        )
    if version == 3:
        limits = trace.get("soft_joint_pos_limits")
        if "soft_joint_pos_limits" in trace and (
            not isinstance(limits, np.ndarray)
            or limits.shape != (shape[1], 12, 2)
            or not np.issubdtype(limits.dtype, np.floating)
            or not np.isfinite(limits).all()
            or np.any(limits[..., 0] >= limits[..., 1])
        ):
            raise ValueError("Invalid recorded native soft joint position limits")
    if native:
        contact_names = trace["contact_body_names"]
        columns = trace["terrain_column_id"]
        if (
            contact_names.ndim != 1
            or contact_names.dtype.kind != "U"
            or not len(contact_names)
            or len(set(contact_names)) != len(contact_names)
            or any(not name for name in contact_names)
            or limits is None
        ):
            raise ValueError("Invalid native contact names or missing soft bounds")
        if (
            columns.shape != (shape[1],)
            or not np.issubdtype(columns.dtype, np.integer)
            or np.any((columns < 0) | (columns >= 20))
            or not np.array_equal(columns // 4, profiles)
        ):
            raise ValueError("Invalid native terrain column assignment")
        for key, field_shape in (
            ("joint_position_post", (*shape, 12)),
            ("joint_velocity_post", (*shape, 12)),
            ("computed_torque_substeps", (*shape, 4, 12)),
            ("applied_torque_substeps", (*shape, 4, 12)),
            ("contact_force_norm_n", (*shape, len(contact_names))),
            ("joint_pos_limits", (shape[1], 12, 2)),
        ):
            value = trace[key]
            if (
                value.shape != field_shape
                or not np.issubdtype(value.dtype, np.floating)
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"Invalid native diagnostic field: {key}")
        hard = trace["joint_pos_limits"]
        if (
            np.any(hard[..., 0] >= hard[..., 1])
            or np.any(hard[..., 0] > limits[..., 0])
            or np.any(hard[..., 1] < limits[..., 1])
            or np.any(trace["contact_force_norm_n"] < 0)
        ):
            raise ValueError("Invalid native joint bounds or contact force norms")
    if version >= 2:
        for key, width in (
            ("observation", 45),
            ("action", 12),
            ("joint_target", 12),
            ("default_joint_position", 12),
        ):
            if (
                trace[key].shape != (*shape, width)
                or not np.issubdtype(trace[key].dtype, np.floating)
                or not np.isfinite(trace[key]).all()
            ):
                raise ValueError(f"Invalid evaluation motor field: {key}")
        if not np.array_equal(trace["observation"][..., 6:9], trace["command"]):
            raise ValueError("Actor observation differs from the applied command")
        # Native capture uses these same operations/dtypes, requiring no tolerance.
        if not np.array_equal(
            trace["joint_target"],
            trace["default_joint_position"] + 0.25 * trace["action"],
        ):
            raise ValueError("Recorded target differs from the native affine action")
    for key in (
        "command",
        "position",
        "pre_position",
        "quaternion",
        "pre_quaternion",
        "linear_velocity_b",
        "angular_velocity_b",
        "env_origins",
    ):
        if not np.isfinite(trace[key]).all():
            raise ValueError(f"Nonfinite evaluation field: {key}")
    if np.isnan(trace["base_height_ray"]).any():
        raise ValueError(
            "Corrupt support rays; only finite hits or infinity misses are valid"
        )
    if any(
        not np.allclose(np.linalg.norm(trace[key], axis=-1), 1.0, atol=1e-3, rtol=0)
        for key in ("quaternion", "pre_quaternion")
    ):
        raise ValueError("Invalid evaluation quaternion norm")
    dt, settle = protocol["period_s"], round(
        protocol["settling_s"] / protocol["period_s"]
    )
    twist = np.concatenate(
        (trace["linear_velocity_b"][..., :2], trace["angular_velocity_b"][..., 2:3]),
        axis=-1,
    )
    error = np.abs(twist - trace["command"])
    delta = trace["position"][..., :2] - trace["pre_position"][..., :2]
    distance = np.linalg.norm(delta, axis=-1)
    local = np.abs(trace["position"][..., :2] - trace["env_origins"][None, :, :2])
    moving = np.linalg.norm(trace["command"][..., :2], axis=-1) > 0.1
    # Spatial exclusion avoids awarding the exact support pad, band and border.
    # Finite ray-height variation below is separate evidence, not a profile label.
    nonflat = (
        (trace["terrain_profile_id"][None] != 0)
        & (local.max(-1) > 1.0)
        & (local[..., 1] > 0.6)
        & (local.max(-1) < 7.0)
    )
    exposed = valid & moving & nonflat & (np.linalg.norm(twist[..., :2], axis=-1) > 0.1)
    hits = trace["base_height_ray"]
    hit_valid = np.isfinite(hits).all(-1)
    quaternion = trace["quaternion"]
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    if version >= 2:
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
        measured_joint = (
            trace["observation"][..., 9:21] + trace["default_joint_position"]
        )

    def health(indices, row):
        finite_hits = indices[hit_valid[indices, row]]
        clearance = trace["position"][finite_hits, row, 2] - hits[finite_hits, row, 2]
        result = {
            "center_clearance_min_m": (
                float(clearance.min()) if len(clearance) else None
            ),
            "center_clearance_median_m": (
                float(np.median(clearance)) if len(clearance) else None
            ),
            "abs_roll_p95_rad": (
                float(np.quantile(np.abs(roll[indices, row]), 0.95))
                if len(indices)
                else None
            ),
            "abs_pitch_p95_rad": (
                float(np.quantile(np.abs(pitch[indices, row]), 0.95))
                if len(indices)
                else None
            ),
            "pre_action_joint_position_mean_rad": (
                measured_joint[indices, row].mean(0).tolist() if len(indices) else None
            ),
            "joint_target_mean_rad": (
                trace["joint_target"][indices, row].mean(0).tolist()
                if len(indices)
                else None
            ),
        }
        if version == 3:
            excess = None
            if limits is not None and len(indices):
                joint = measured_joint[indices, row]
                excess = np.maximum(limits[row, :, 0] - joint, 0) + np.maximum(
                    joint - limits[row, :, 1], 0
                )
            result.update(
                pre_action_soft_bound_exceedance_fraction=(
                    (excess > 0).mean(0).tolist() if excess is not None else None
                ),
                pre_action_soft_bound_mean_excess_rad=(
                    excess.mean(0).tolist() if excess is not None else None
                ),
            )
        if native:
            diagnostics = None
            if len(indices):
                joint = trace["joint_position_post"][indices, row]
                excess = np.maximum(hard[row, :, 0] - joint, 0) + np.maximum(
                    joint - hard[row, :, 1], 0
                )
                computed = trace["computed_torque_substeps"][indices, row]
                applied = trace["applied_torque_substeps"][indices, row]
                diagnostics = {
                    "post_joint_hard_bound_exceedance_fraction": (excess > 0)
                    .mean(0)
                    .tolist(),
                    "post_joint_hard_bound_max_excess_rad": excess.max(0).tolist(),
                    "post_joint_velocity_abs_max_rad_s": np.abs(
                        trace["joint_velocity_post"][indices, row]
                    )
                    .max(0)
                    .tolist(),
                    "computed_torque_abs_max_nm": np.abs(computed).max((0, 1)).tolist(),
                    "applied_torque_abs_max_nm": np.abs(applied).max((0, 1)).tolist(),
                    "torque_clipping_fraction": (np.abs(computed - applied) > 1e-5)
                    .mean((0, 1))
                    .tolist(),
                    "contact_force_norm_max_n": trace["contact_force_norm_n"][
                        indices, row
                    ]
                    .max(0)
                    .tolist(),
                }
            result["native_actuator"] = diagnostics
        return result

    w, x, y, z = np.moveaxis(trace["pre_quaternion"], -1, 0)
    pre_heading = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    trials = []
    for row in range(valid.shape[1]):
        count = int(valid[:, row].sum())
        selected = valid[:, row]
        if np.any(trace["terminated"][:, row] & selected):
            outcome = "physical_failure"
        elif np.any(trace["procedural_workspace"][:, row] & selected):
            outcome = "workspace_censored"
        elif count == protocol["steps"]:
            outcome = "horizon_completed"
        else:
            outcome = "incomplete"
        heights = hits[selected & hit_valid[:, row], row, 2]
        phases = []
        for index, phase in enumerate(protocol["phases"]):
            phase_steps = np.flatnonzero(trace["phase_index"] == index)
            surviving = phase_steps[selected[phase_steps]]
            steady = surviving[surviving >= phase_steps[0] + settle]
            final_block = phase_steps[-settle:]
            final_complete = len(final_block) == settle and selected[final_block].all()
            entry = {
                "name": phase["name"],
                "recorded_steps": len(surviving),
                "complete": len(surviving) == len(phase_steps),
                "mean_abs_twist_error_after_settle": (
                    error[steady, row].mean(0).tolist() if len(steady) else None
                ),
                "final_block_mean_abs_twist_error": (
                    error[final_block, row].mean(0).tolist() if final_complete else None
                ),
                "travel_distance_m": float(distance[surviving, row].sum()),
                "moving_nonflat_time_s": float(exposed[surviving, row].sum() * dt),
            }
            if version >= 2:
                entry.update(
                    achieved_twist_after_settle=(
                        twist[steady, row].mean(0).tolist() if len(steady) else None
                    ),
                    final_block_achieved_twist=(
                        twist[final_block, row].mean(0).tolist()
                        if final_complete
                        else None
                    ),
                    health=health(surviving, row),
                )
            if len(surviving) and not np.any(trace["command"][surviving, row]):
                onset = trace["pre_position"][phase_steps[0], row, :2]
                start_heading = pre_heading[phase_steps[0], row]
                change = (
                    np.unwrap(np.r_[start_heading, heading[surviving, row]])[1:]
                    - start_heading
                )
                entry["stop_onset_max_displacement_m"] = float(
                    np.linalg.norm(
                        trace["position"][surviving, row, :2] - onset, axis=-1
                    ).max()
                )
                entry["stop_onset_max_heading_change_rad"] = float(np.abs(change).max())
            phases.append(entry)
        trials.append(
            {
                "env_id": row,
                "terrain": names[int(trace["terrain_profile_id"][row])],
                "outcome": outcome,
                "recorded_steps": count,
                "travel_distance_m": float(distance[selected, row].sum()),
                "moving_nonflat_time_s": float(exposed[:, row].sum() * dt),
                "moving_nonflat_distance_m": float(
                    distance[exposed[:, row], row].sum()
                ),
                "support_height_span_m": (
                    float(np.ptp(heights)) if len(heights) else None
                ),
                "missing_support_ray_steps": int((selected & ~hit_valid[:, row]).sum()),
                "phases": phases,
            }
        )
        if version >= 2:
            trials[-1]["health"] = health(np.flatnonzero(selected), row)
        if native:
            trials[-1]["terrain_column_id"] = int(columns[row])
    profiles = {}
    for name in names:
        group = [trial for trial in trials if trial["terrain"] == name]
        profiles[name] = {
            "initial_trials": len(group),
            **{
                outcome: sum(trial["outcome"] == outcome for trial in group)
                for outcome in (
                    "physical_failure",
                    "workspace_censored",
                    "horizon_completed",
                    "incomplete",
                )
            },
            "trials_with_moving_nonflat_exposure": sum(
                trial["moving_nonflat_time_s"] > 0 for trial in group
            ),
        }
    summary = {"profiles": profiles, "trials": trials}
    if version >= 2:
        summary.update(
            summary_version=f"go2_operator_proprio_summary_v{version}",
            health_scope=(
                "All recorded first-attempt samples, including terminal samples: "
                "post-physics root-to-center-ray vertical clearance and world-referenced "
                "root roll/pitch; pre-action measured joint positions; requested targets "
                "for each step. Not terrain-normal clearance, simultaneous joint tracking "
                "error, terminal joint states, hard-limit violations, contacts or actuator "
                "loads. No health acceptance."
            ),
        )
        if version == 3:
            summary["health_scope"] += (
                " Soft-bound diagnostics use recorded native limits only and strict "
                "exceedance; absent limits yield null."
            )
    if native:
        summary.update(
            contact_body_names=contact_names.tolist(),
            health_scope=(
                "Post-physics/pre-reset center-ray vertical clearance and world roll/pitch; "
                "pre-action joint state and requested targets. Soft-bound diagnostics use "
                "strict exceedance. Additional native_actuator fields include terminal "
                "joint state; see actuator_health_scope. No health acceptance."
            ),
            actuator_health_scope=(
                "First attempts including terminal samples: post-physics/pre-reset joint "
                "position and velocity at 50 Hz; strict excess over recorded simulator "
                "joint bounds; computed and applied actuator COMMAND torques at all four "
                "200 Hz substeps, not hardware measured torque. Clipping threshold 1e-5 Nm. "
                "Contacts are 50 Hz norms of net world force vectors per named body, not "
                "contact-point peaks, terrain-normal loads, slip or support proof. "
                "Requested targets are not measured joint positions. No health acceptance."
            ),
        )
    return summary


def evaluate_recurrent_operator(
    env, policy, checkpoint_sha, metadata, *, is_running, difficulty_range=None
):
    """Replay one clean deterministic first attempt; only the actor drives motors."""
    import numpy as np
    from parkour_lab.learning.controller import ControllerSession, Sample

    protocol = recurrent_evaluation_protocol(difficulty_range=difficulty_range)
    native = difficulty_range is not None
    if native:
        from parkour_lab.tasks.manager_based.parkour_lab.mdp.terrain.operator_terrain import (
            ENVELOPES,
        )

        generator = env.cfg.scene.terrain.terrain_generator
        failure = env.cfg.terminations.procedural_physical_failure
        if (
            tuple(generator.difficulty_range) != tuple(difficulty_range)
            or generator.seed != protocol["seed"]
            or generator.num_rows != 1
            or env.cfg.curriculum.terrain_levels is not None
            or not math.isclose(
                failure.params["minimum_surface_z_m"],
                -max(height for height, _ in ENVELOPES.values()) * difficulty_range[1],
                abs_tol=1e-12,
            )
            or failure.params.get("max_tilt_rad") != math.pi / 4
            or failure.time_out
        ):
            raise ValueError("Native terrain differs from the predeclared frozen probe")
    if (
        env.num_envs != protocol["num_envs"]
        or env.max_episode_length != protocol["steps"]
        or not math.isclose(env.step_dt, protocol["period_s"], abs_tol=1e-9)
        or not math.isclose(env.physics_dt, 0.005, abs_tol=1e-9)
        or env.cfg.decimation != 4
        or env.cfg.observations.proprio.enable_corruption
        or tuple(env.observation_manager.active_terms["proprio"])
        != tuple(name for name, _ in FRAME_TERMS)
        or policy.training
        or any(parameter.requires_grad for parameter in policy.parameters())
        or policy.memory_a.hidden_state is not None
        or policy.memory_c.hidden_state is not None
        or _recurrent_actor_sha256(policy)
        != metadata["controller_manifest"]["artifact_sha256"]
    ):
        raise ValueError(
            "Require a frozen fresh-memory GRU and the clean 80-trial evaluation configuration"
        )
    command = env.command_manager.get_term("base_velocity")
    if (
        command.cfg.heading_command
        or command.cfg.rel_heading_envs
        or command.cfg.rel_standing_envs
    ):
        raise ValueError("Evaluation forbids heading or standing command assistance")
    capture = env.operator_capture
    if (
        capture.enabled
        or capture.samples
        or capture.course is not None
        or capture.control_trace is not None
        or not capture.procedural
    ):
        raise ValueError(
            "Evaluation requires a fresh procedural terminal-safe recorder"
        )
    env.reset(seed=protocol["seed"])
    binding, motor_hash = _runtime_motor_binding(env)
    robot = env.scene["robot"].data
    limits = robot.soft_joint_pos_limits.detach().clone()
    _check_tensor(limits, (env.num_envs, 12, 2), "native soft joint position limits")
    if torch.any(limits[..., 0] >= limits[..., 1]):
        raise ValueError("Native soft joint position limits must be ordered")
    native_trace = {}
    if native:
        hard_limits = robot.joint_pos_limits.detach().clone()
        _check_tensor(
            hard_limits, (env.num_envs, 12, 2), "native joint position limits"
        )
        native_trace.update(
            joint_pos_limits=hard_limits.cpu().numpy().copy(),
            contact_body_names=np.asarray(
                env.scene["contact_forces"].body_names, dtype=str
            ),
        )
    if (
        binding["joint_names"] != metadata["controller_manifest"]["joint_names"]
        or binding["default_position_rad"]
        != metadata["controller_manifest"]["configuration"]["default_position_rad"]
    ):
        raise ValueError(
            "Evaluation joint order or default pose differs from the learned interface"
        )
    columns = env.scene.terrain.terrain_types
    if torch.any((columns < 0) | (columns >= 20)):
        raise ValueError("Unexpected procedural terrain columns")
    profile_ids = columns // 4
    if native:
        native_trace["terrain_column_id"] = columns.cpu().numpy().copy()
    if not torch.equal(
        torch.bincount(profile_ids, minlength=5),
        torch.full((5,), 16, device=env.device),
    ):
        raise ValueError(
            "Evaluation requires sixteen initial trials per terrain profile"
        )
    flat = profile_ids < 2
    adapter = RecurrentOperatorAdapter(
        policy,
        joint_names=tuple(binding["joint_names"]),
        default_position_rad=robot.default_joint_pos[0],
        artifact_sha256=checkpoint_sha,
        actuator_profile="native_motor_sha256:" + motor_hash,
    )
    session = ControllerSession(
        adapter,
        joint_names=adapter.spec.joint_names,
        actuator_profile=adapter.spec.actuator_profile,
    )
    phase_ids = np.repeat(
        np.arange(len(protocol["phases"])),
        [round(p["duration_s"] / env.step_dt) for p in protocol["phases"]],
    )
    valid = []
    finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    reset = torch.ones_like(finished)
    partial_resets = 0
    capture.actuator_diagnostics = native
    capture.motor_parity = capture.enabled = True
    try:
        with torch.inference_mode():
            for step, phase_id in enumerate(phase_ids):
                if not is_running():
                    raise RuntimeError(
                        f"Simulator closed at recurrent evaluation step {step}"
                    )
                phase = protocol["phases"][phase_id]
                desired = robot.joint_pos.new_tensor(phase["rough_command"]).repeat(
                    env.num_envs, 1
                )
                desired[flat] = desired.new_tensor(phase["flat_command"])
                desired[finished] = 0  # Housekeeping packets, never replacement trials.
                command.time_left.fill_(float("inf"))
                command.is_standing_env.fill_(False)
                command.is_heading_env.fill_(False)
                command.vel_command_b.copy_(desired)
                frame = env.observation_manager.compute()["proprio"]
                values = (
                    robot.root_ang_vel_b,
                    robot.projected_gravity_b,
                    robot.joint_pos - robot.default_joint_pos,
                    robot.joint_vel,
                    env.action_manager.action,
                )
                independent = torch.cat((*values[:2], desired, *values[2:]), dim=-1)
                if not torch.equal(frame, independent) or torch.any(
                    values[-1][reset] != 0
                ):
                    raise RuntimeError(
                        "Clean causal frame or previous-action reset differs from native sensors"
                    )
                now = step * env.step_dt
                samples = {
                    name: Sample(
                        value, now, torch.ones_like(reset), spec.units, spec.frame
                    )
                    for (name, spec), value in zip(
                        adapter.spec.sensors.items(), values, strict=True
                    )
                }
                result = session.step(
                    time_s=now,
                    command=desired,
                    command_time_s=now,
                    sensors=samples,
                    reset_mask=reset,
                )
                policy.reset(reset)
                shadow = policy.act_inference({"proprio": frame})
                if not torch.allclose(result.raw_action, shadow, atol=1e-6, rtol=0):
                    raise RuntimeError(
                        "Learned actor-only adapter differs from native deterministic inference"
                    )
                capture.observation = frame.detach().clone()
                valid.append((~finished).cpu().numpy().copy())
                _, reward, terminated, timed_out, _ = env.step(result.raw_action)
                _check_tensor(reward, (env.num_envs,), "evaluation reward")
                if len(capture.samples) != step + 1:
                    raise RuntimeError(
                        "Missing terminal-safe recurrent evaluation sample"
                    )
                if not np.array_equal(
                    capture.samples[-1]["joint_target"],
                    result.position_rad.cpu().numpy(),
                ):
                    raise RuntimeError(
                        "Learned adapter target differs from native pre-reset joint target"
                    )
                reset = (terminated | timed_out).detach().clone()
                partial_resets += int(reset.any() and not reset.all())
                finished |= reset
    finally:
        capture.enabled = capture.motor_parity = False
        capture.actuator_diagnostics = False
    trace = capture.finish()
    if not torch.equal(limits, robot.soft_joint_pos_limits):
        raise RuntimeError(
            "Native soft joint position limits changed during evaluation"
        )
    trace.update(
        **native_trace,
        valid_first_attempt=np.stack(valid),
        phase_index=phase_ids,
        terrain_profile_id=profile_ids.cpu().numpy(),
        env_origins=env.scene.env_origins.detach().cpu().numpy().copy(),
        soft_joint_pos_limits=limits.cpu().numpy().copy(),
    )
    if native and not torch.equal(hard_limits, robot.joint_pos_limits):
        raise RuntimeError("Native joint bounds changed during evaluation")
    if (
        _recurrent_actor_sha256(policy)
        != metadata["controller_manifest"]["artifact_sha256"]
    ):
        raise RuntimeError("Evaluation modified the learned actor weights")
    return {
        "status": "DEVELOPMENT_EVALUATED_NOT_ACCEPTED",
        "policy_version": RECURRENT_OPERATOR_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_learning_updates": metadata["learning_updates"],
        "learning_updates": 0,
        "control_steps": len(phase_ids),
        "environment_transitions": len(phase_ids) * env.num_envs,
        "partial_reset_steps": partial_resets,
        "protocol": protocol,
        "motor_binding": binding,
        "motor_binding_sha256": motor_hash,
        "training_motor_binding_sha256": metadata["motor_binding_sha256"],
        "controller_manifest": session.manifest,
        "interface_sha256": session.interface_sha256,
        **summarize_recurrent_evaluation(trace, protocol),
        "metric_scope": "post-physics/pre-reset first attempts; final block only when fully observed; nonflat-region motion and center-ray height variation are separate descriptive evidence, not traversal acceptance",
        "exit_allowed": False,
    }, trace


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
