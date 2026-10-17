"""Versioned fixed Go2 cases; geometry seeds are independent of command profiles."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType

from parkour_lab.environments.faults import JOINT_NAMES, FaultConfig

VERSION = "go2_fixed_cases_development_v1"
PERIOD_S = 0.02
DURATION_S = 20.0
STEPS = 1000
PROFILES = MappingProxyType(
    {
        "stand": (0.0, 0.0, 0.0),
        "forward-035": (0.35, 0.0, 0.0),
        "pivot-left": (0.0, 0.0, 0.5),
        "pivot-right": (0.0, 0.0, -0.5),
        "arc-left": (0.35, 0.0, 0.3),
        "arc-right": (0.35, 0.0, -0.3),
    }
)
SUBTYPES = MappingProxyType(
    {
        "Flat": ("plane",),
        "Uneven": ("noise",),
        "Slopes": ("pyramid", "inverted_pyramid"),
        "Stairs": ("pyramid", "inverted_pyramid"),
        "Hills": ("wave",),
    }
)
PHASES = MappingProxyType(
    {
        "stand": (0.0, 2.0),
        "onset": (2.0, 3.0),
        "settled_motion": (3.0, 14.0),
        "braking": (14.0, 15.0),
        "stop_hold": (15.0, 20.0),
    }
)


@dataclass(frozen=True)
class EvaluationCase:
    """One prescribed first attempt; level zero is not an OOD designation."""

    partition: str
    family: str
    physical_level: int | None
    subtype: str
    profile: str
    attempt: int = 0
    condition: str = "id"
    fault: FaultConfig | None = None

    def __post_init__(self):
        if any(
            type(value) is not str
            for value in (self.partition, self.family, self.subtype, self.profile)
        ):
            raise ValueError(
                "Case partition, family, subtype and profile require names"
            )
        if self.partition not in ("train", "validation", "test"):
            raise ValueError("Require a train, validation or test partition")
        if self.family not in SUBTYPES or self.subtype not in SUBTYPES[self.family]:
            raise ValueError("Unknown terrain family/subtype")
        valid_level = (
            self.physical_level is None
            if self.family == "Flat"
            else type(self.physical_level) is int and self.physical_level in (0, 3, 7)
        )
        if not valid_level:
            raise ValueError("Flat has no ladder; other families use levels 0, 3 or 7")
        if self.profile not in PROFILES:
            raise ValueError("Unknown fixed command profile")
        if type(self.attempt) is not int or self.attempt < 0:
            raise ValueError("Require a nonnegative attempt index")
        if self.condition not in ("id", "composition", "terrain_ood", "fault"):
            raise ValueError("Unknown evaluation condition")
        if self.condition != "id" and self.partition == "train":
            raise ValueError("Robustness conditions are excluded from training")
        if self.condition == "composition" and self.family not in (
            "Slopes",
            "Stairs",
            "Hills",
        ):
            raise ValueError("Composition requires Slopes, Stairs or Hills")
        if self.condition == "terrain_ood" and self.family == "Flat":
            raise ValueError("Flat has no terrain extrapolation")
        if self.condition == "fault":
            if (
                not isinstance(self.fault, FaultConfig)
                or self.fault.joint_name not in JOINT_NAMES
                or self.family not in ("Flat", "Uneven")
                or (self.family == "Uneven" and self.physical_level != 3)
                or self.profile != "forward-035"
                or not 6 <= self.fault.onset_s <= 8
                or not math.isclose(
                    self.fault.onset_s / PERIOD_S,
                    round(self.fault.onset_s / PERIOD_S),
                    rel_tol=0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "Fault cases require declared Flat/medium Uneven forward motion and onset in [6, 8] s"
                )
        elif self.fault is not None:
            raise ValueError("Drive loss requires the separate fault condition")

    def as_dict(self):
        """Constructor fields only; identities are recomputed, never trusted input."""
        value = asdict(self)
        if self.fault is None:
            value.pop("fault")
        if self.condition == "id":
            value.pop("condition")
        return value

    @property
    def case_id(self):
        level = "flat" if self.physical_level is None else f"level{self.physical_level}"
        base = (
            f"{VERSION}_{self.partition}_{self.family}_{level}_{self.subtype}"
            f"_{self.profile}_attempt{self.attempt}"
        )
        if self.condition == "id":
            return base
        suffix = self.condition
        if self.fault is not None:
            suffix += f"_{self.fault.joint_name}_drive{self.fault.factor:g}_onset{self.fault.onset_s:g}"
        return f"{base}_{suffix}"

    @property
    def seed(self):
        identity = self.as_dict()
        identity.pop("profile")
        identity.pop("condition", None)
        identity.pop("fault", None)
        text = json.dumps(dict(version=VERSION, **identity), sort_keys=True)
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")

    def command_at(self, time_s):
        if isinstance(time_s, bool) or not isinstance(time_s, (int, float)):
            raise ValueError("Require a finite decision time in seconds")
        if not math.isfinite(time_s) or not 0 <= time_s <= DURATION_S:
            raise ValueError("Decision time lies outside the fixed horizon")
        return PROFILES[self.profile] if 2.0 <= time_s < 14.0 else (0.0, 0.0, 0.0)


def case_from_dict(value):
    """Read exactly the prescribed case fields, without accepting identity aliases."""
    required = {field.name for field in fields(EvaluationCase)} - {"condition", "fault"}
    if (
        type(value) is not dict
        or not required <= set(value)
        or set(value) - required - {"condition", "fault"}
    ):
        raise ValueError("Invalid fixed case fields")
    values = dict(value)
    if "fault" in values:
        fault = values["fault"]
        if type(fault) is not dict or set(fault) != {"joint_name", "factor", "onset_s"}:
            raise ValueError("Invalid drive-loss prescription")
        values["fault"] = FaultConfig(**fault)
    return EvaluationCase(**values)


def case_bank(partition, *, attempts=1):
    """Nineteen geometry cells times six equally weighted profiles per attempt."""
    if type(attempts) is not int or attempts <= 0:
        raise ValueError("Require a positive prescribed attempt count")
    return tuple(
        EvaluationCase(partition, family, level, subtype, profile, attempt)
        for family, subtypes in SUBTYPES.items()
        for level in ((None,) if family == "Flat" else (0, 3, 7))
        for subtype in subtypes
        for profile in PROFILES
        for attempt in range(attempts)
    )
