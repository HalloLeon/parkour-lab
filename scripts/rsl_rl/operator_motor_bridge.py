"""Deliver named absolute joint targets through the pinned native motor interface.

No controller architecture, observations, command-source logic, physics changes or
hardware I/O. Startup verifies the resolved motor binding once. Per-step
delivery uses the unchanged stock affine JointPositionAction transform and never
interprets a controller-private raw vector unless the trusted host opted into the
exact documented native hint. The caller owns a frozen simulator configuration.
"""

from __future__ import annotations

import hashlib
import json

import torch

from parkour_lab.learning.controller import JointTargets, finite_tensor
from parkour_lab.learning.motor_contract import verify_runtime_motor


VERSION = "native_joint_target_bridge_v1"
NATIVE_RAW_ACTION_MEANING = (
    "stock unscaled action; q_target = default_q + 0.25 * raw_action; no clip"
)


def _named_finite(value, shape, name):
    try:
        finite_tensor(value, shape)
    except ValueError as error:
        raise ValueError(
            f"{name} must be a finite floating tensor of shape {shape}"
        ) from error


def _runtime_motor_binding(env):
    """Bind the resolved native stock motor, not merely an action tensor width."""
    robot = env.scene["robot"]
    term = env.action_manager.get_term("joint_pos")
    cfg, descriptor = term.cfg, term.IO_descriptor
    joints = tuple(robot.joint_names)
    default = robot.data.default_joint_pos
    _named_finite(default, (env.num_envs, 12), "runtime default joint position")
    _named_finite(
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
            _named_finite(value, (env.num_envs, len(actuator.joint_names)), parameter)
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


class NativeJointTargetBridge:
    """One encoded command followed by one verified native delivery.

    Native-hint opt-in is for trusted archive loaders, not a policy-controlled
    capability. Even there, named absolute positions remain authoritative. A
    failure latches the bridge: recover externally, never replace bad output
    with zeros or clamp it. Terminal rows are deliberately excluded from native
    post-step buffer comparisons because auto-reset may already have run.
    """

    def __init__(self, env, contract, manifest, *, preserve_native_raw=False):
        if type(preserve_native_raw) is not bool:
            raise ValueError("Native raw-action preservation must be an explicit bool")
        if (
            preserve_native_raw
            and manifest.get("raw_action_meaning") != NATIVE_RAW_ACTION_MEANING
        ):
            raise ValueError("Native raw hint requires the exact stock action meaning")
        self.binding, _ = _runtime_motor_binding(env)
        self.motor_verification = verify_runtime_motor(contract, manifest, self.binding)
        self.env = env
        self.term = env.action_manager.get_term("joint_pos")
        self._robot_data = env.scene["robot"].data
        self.default = env.scene["robot"].data.default_joint_pos.detach().clone()
        self.joint_names = tuple(self.binding["joint_names"])
        self.num_envs = env.num_envs
        self.preserve_native_raw = preserve_native_raw
        self.faulted = False
        self._pending = None
        self._counts = {
            "encoded_steps": 0,
            "native_hint_steps": 0,
            "target_only_steps": 0,
            "verified_delivery_steps": 0,
            "native_step_returns": 0,
            "native_verified_rows": 0,
            "excluded_terminal_rows": 0,
        }

    def _healthy(self):
        if self.faulted:
            raise RuntimeError(
                "Native joint-target bridge faulted; external recovery required"
            )

    @property
    def pending_delivery(self):
        return self._pending is not None

    @torch.inference_mode()
    def encode(self, result):
        """Return independent native action storage; do not mutate or step the env."""
        self._healthy()
        try:
            if self._pending is not None:
                raise RuntimeError(
                    "Previous encoded targets have not been delivery-verified"
                )
            if (
                not isinstance(result, JointTargets)
                or result.joint_names != self.joint_names
            ):
                raise ValueError(
                    "JointTargets must retain the exact native joint names/order"
                )
            finite_tensor(result.position_rad, (self.num_envs, 12), self.default)
            target = result.position_rad.detach().clone()
            use_hint = self.preserve_native_raw and result.raw_action is not None
            if use_hint:
                finite_tensor(result.raw_action, (self.num_envs, 12), self.default)
                action = result.raw_action.detach().clone()
            else:
                # raw_action is architecture-private diagnostic data by default;
                # it is not inspected, normalized, clipped or used for delivery.
                action = (target - self.default) / 0.25
            finite_tensor(action, (self.num_envs, 12), self.default)
            if not torch.equal(action * 0.25 + self.default, target):
                raise ValueError(
                    "Native affine action cannot exactly reproduce requested joint targets"
                )
            self._pending = (action.detach().clone(), target)
            self._counts["encoded_steps"] += 1
            self._counts["native_hint_steps" if use_hint else "target_only_steps"] += 1
            return action.detach().clone()
        except Exception:
            self.faulted = True
            raise

    @torch.inference_mode()
    def verify_delivery(self, terminated, timed_out):
        """Compare only live rows; never inspect replacement terminal scene state."""
        self._healthy()
        try:
            if self._pending is None:
                raise RuntimeError(
                    "Native delivery verification requires one encoded command"
                )
            # The caller invokes this only after env.step returned. Preserve
            # that completion count even if a buffer/mask check below fails.
            self._counts["native_step_returns"] += 1
            for value in (terminated, timed_out):
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != (self.num_envs,)
                    or value.dtype != torch.bool
                    or value.device != self.default.device
                ):
                    raise ValueError(
                        "Native done masks must match the encoded target batch"
                    )
            selected = ~(terminated | timed_out)
            rows = int(selected.sum().item())
            if rows:
                action, target = self._pending
                delivered = self.env.action_manager.action
                processed = self.term.processed_actions
                joint_target = self._robot_data.joint_pos_target
                for value in (delivered, processed, joint_target):
                    if (
                        not isinstance(value, torch.Tensor)
                        or value.shape != (self.num_envs, 12)
                        or value.device != self.default.device
                        or value.dtype != self.default.dtype
                    ):
                        raise ValueError(
                            "Native delivery buffer differs from the encoded target batch"
                        )
                    finite_tensor(value[selected], (rows, 12), self.default)
                if not torch.equal(delivered[selected], action[selected]):
                    raise ValueError(
                        "Native delivered action differs from the encoded targets"
                    )
                if not torch.equal(processed[selected], target[selected]):
                    raise ValueError(
                        "Native processed joint positions differ from requested targets"
                    )
                if not torch.equal(joint_target[selected], target[selected]):
                    raise ValueError(
                        "Native articulation joint targets differ from requested targets"
                    )
            self._counts["verified_delivery_steps"] += 1
            self._counts["native_verified_rows"] += rows
            self._counts["excluded_terminal_rows"] += self.num_envs - rows
            self._pending = None
        except Exception:
            self.faulted = True
            raise

    def progress(self):
        return {
            "version": VERSION,
            **self._counts,
            "pending_delivery": self.pending_delivery,
            "faulted": self.faulted,
            "qualification_passed": False,
            "exit_allowed": False,
            "scope": "Native affine target encoding and nonterminal delivery equality only; not controller behavior, continuous motor telemetry or hardware safety acceptance.",
        }
