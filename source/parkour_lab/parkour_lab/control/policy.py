"""Adapt an already-loaded blind policy to the named controller boundary."""

from typing import Protocol

import torch

from .action import BOUNDED_MODE, BOUNDED_RAW_ACTION_MEANING, JointActionTransform
from .controller import ControllerSpec, JointTargets, finite_tensor
from .proprioception import (
    FRAME_DIM,
    FRAME_TERMS,
    pack_proprioception,
    proprioceptive_sensor_specs,
)


class FramePolicy(Protocol):
    """Caller-owned inference; preprocessing and episode memory stay upstream.

    ``state_sha256`` fingerprints learned parameters, buffers and normalization,
    excluding evolving episode memory. ``act`` returns encoded raw requests,
    not radians. No model architecture or checkpoint format is prescribed.
    """

    def reset(self, mask: torch.Tensor) -> None: ...

    def act(self, observation: torch.Tensor) -> torch.Tensor: ...

    def state_sha256(self) -> str: ...


def _state_identity(policy):
    identity = policy.state_sha256()
    if (
        not isinstance(identity, str)
        or len(identity) != 64
        or any(character not in "0123456789abcdef" for character in identity)
    ):
        raise ValueError("Policy state identity requires a lowercase SHA-256 digest")
    return identity


class PolicyController:
    """Pack the shared noisy 49D frame and bound caller-owned policy requests.

    The native host supplies only named causal samples. This adapter neither
    reads the simulator nor draws sensor noise; it owns no learner or weights.
    """

    def __init__(
        self,
        policy: FramePolicy,
        *,
        joint_names,
        default_position_rad,
        target_limits_rad,
        actuator_profile,
    ):
        finite_tensor(default_position_rad, (12,))
        if default_position_rad.dtype != torch.float32:
            raise ValueError("The shared policy interface requires float32 tensors")
        names = tuple(joint_names)
        if len(names) != 12 or len(set(names)) != 12:
            raise ValueError("Require twelve distinct native joint names in order")
        default = default_position_rad.detach().clone()
        limits = default.new_tensor(target_limits_rad)
        finite_tensor(limits, (12, 2), default)
        if not (
            (limits[:, 0] < limits[:, 1]).all()
            and (limits[:, 0] <= default).all()
            and (default <= limits[:, 1]).all()
        ):
            raise ValueError(
                "Joint limits must be ordered and contain the default pose"
            )
        self.policy = policy
        self.transform = JointActionTransform(default, limits.tolist())
        self.spec = ControllerSpec(
            name="caller_loaded_frame_policy",
            artifact_sha256=_state_identity(policy),
            preprocessing_version="go2_blind_frame49_v1",
            joint_names=names,
            period_s=0.02,
            actuator_profile=actuator_profile,
            sensors=proprioceptive_sensor_specs(),
            raw_action_meaning=BOUNDED_RAW_ACTION_MEANING,
            configuration={
                "default_position_rad": default.tolist(),
                "action_scale": 0.25,
                "action_clip": {
                    "version": BOUNDED_MODE,
                    "target_limits_rad": limits.tolist(),
                },
                "frame_dim": FRAME_DIM,
                "input_order": [name for name, _ in FRAME_TERMS],
                "reset": "caller-owned episode memory; explicit per-row mask",
            },
        )

    def state_sha256(self):
        return _state_identity(self.policy)

    def reset(self, mask):
        self.policy.reset(mask.detach().clone())

    @torch.inference_mode()
    def act(self, inputs):
        sensors = inputs.sensors
        frame = pack_proprioception(
            sensors["base_ang_vel"].value,
            sensors["projected_gravity"].value,
            inputs.command,
            sensors["joint_position_relative_default"].value,
            sensors["joint_velocity"].value,
            sensors["stock_previous_raw_action"].value,
            sensors["foot_contacts"].value,
        )
        finite_tensor(frame, (len(inputs.command), FRAME_DIM), self.transform.default)
        raw = self.policy.act(frame)
        finite_tensor(raw, (len(frame), 12), frame)
        delivered, targets = self.transform(raw)
        return JointTargets(self.spec.joint_names, targets, delivered)
