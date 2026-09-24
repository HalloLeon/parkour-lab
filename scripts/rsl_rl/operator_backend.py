"""Trusted, statically registered controller decoders; never import artifact code.

File receipts detect mismatches, not trustworthiness. Source checkpoint metadata
is recorded by the artifact producer, not independently reloaded or verified.
"""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Callable


def _sha256(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _json_dict(value):
    if type(value) is not dict:
        raise ValueError("Controller metadata must be a finite JSON object")
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class LoadedController:
    controller: object
    motor_contract: dict
    backend: str
    artifact_sha256: str
    source: dict
    preserve_native_raw: bool
    _receipt_json: str = field(init=False, repr=False)

    def __post_init__(self):
        from parkour_lab.learning.controller import ControllerSession, ControllerSpec
        from parkour_lab.learning.motor_contract import validate_motor_contract

        spec = getattr(self.controller, "spec", None)
        if (
            not isinstance(spec, ControllerSpec)
            or not callable(getattr(self.controller, "act", None))
            or not callable(getattr(self.controller, "reset", None))
            or not isinstance(self.backend, str)
            or not self.backend
            or not _sha256(self.artifact_sha256)
            or type(self.preserve_native_raw) is not bool
        ):
            raise ValueError("Invalid loaded controller identity or interface")
        if self.preserve_native_raw:
            from .operator_motor_bridge import NATIVE_RAW_ACTION_MEANING

            if spec.raw_action_meaning != NATIVE_RAW_ACTION_MEANING:
                raise ValueError(
                    "Native raw preservation requires the stock action meaning"
                )
        try:
            session = ControllerSession(
                self.controller,
                joint_names=spec.joint_names,
                actuator_profile=spec.actuator_profile,
                allow_privileged=False,
            )
            manifest = _json_dict(session.manifest)
            contract, source = _json_dict(self.motor_contract), _json_dict(self.source)
            validate_motor_contract(contract, manifest)
            object.__setattr__(self, "motor_contract", contract)
            object.__setattr__(self, "source", source)
            object.__setattr__(
                self,
                "_receipt_json",
                json.dumps(
                    {
                        "backend": self.backend,
                        "artifact_sha256": self.artifact_sha256,
                        "controller_manifest": manifest,
                        "controller_interface_sha256": session.interface_sha256,
                        "source": source,
                        "motor_contract": contract,
                        "preserve_native_raw": self.preserve_native_raw,
                    },
                    sort_keys=True,
                    allow_nan=False,
                ),
            )
        except (AttributeError, TypeError, KeyError, OverflowError) as error:
            raise ValueError("Malformed controller interface or metadata") from error

    def receipt(self):
        """Return an independent startup snapshot, not mutable controller state."""
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class Backend:
    decode: Callable
    preserve_native_raw: bool = False


def _recurrent_actor_v2(encoded, device):
    from parkour_lab.learning.recurrent_runtime import (
        BOUND_ACTOR_BUNDLE_VERSION,
        _load_actor_bytes,
    )

    controller, metadata, _ = _load_actor_bytes(encoded, device)
    if metadata["format"] != BOUND_ACTOR_BUNDLE_VERSION:
        raise ValueError("recurrent_actor_v2 requires a bound V2 motor artifact")
    return (
        controller,
        metadata["motor_contract"],
        {
            "checkpoint_sha256": metadata["source_checkpoint_sha256"],
            "learning_updates": metadata["source_learning_updates"],
            "verification": "artifact_recorded_not_checkpoint_reloaded",
        },
    )


BACKENDS = MappingProxyType(
    {"recurrent_actor_v2": Backend(_recurrent_actor_v2, preserve_native_raw=True)}
)


def load_controller_artifact(path, *, backend, device="cpu", expected_sha256=None):
    """Read once, check the external receipt, then call only a trusted decoder."""
    if not isinstance(backend, str) or backend not in BACKENDS:
        raise ValueError(f"Unknown controller backend: {backend!r}")
    selected = BACKENDS[backend]
    encoded = Path(path).read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and (
        not _sha256(expected_sha256) or digest != expected_sha256
    ):
        raise ValueError("Controller artifact file hash mismatch")
    controller, contract, source = selected.decode(encoded, device)
    return LoadedController(
        controller, contract, backend, digest, source, selected.preserve_native_raw
    )
