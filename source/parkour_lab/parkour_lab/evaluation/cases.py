"""Versioned fixed Go2 cases; geometry seeds are independent of command profiles."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from types import MappingProxyType

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

    def as_dict(self):
        """Constructor fields only; identities are recomputed, never trusted input."""
        return asdict(self)

    @property
    def case_id(self):
        level = "flat" if self.physical_level is None else f"level{self.physical_level}"
        return (
            f"{VERSION}_{self.partition}_{self.family}_{level}_{self.subtype}"
            f"_{self.profile}_attempt{self.attempt}"
        )

    @property
    def seed(self):
        identity = self.as_dict()
        identity.pop("profile")
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
    if type(value) is not dict or set(value) != {
        field.name for field in fields(EvaluationCase)
    }:
        raise ValueError("Invalid fixed case fields")
    return EvaluationCase(**value)


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
