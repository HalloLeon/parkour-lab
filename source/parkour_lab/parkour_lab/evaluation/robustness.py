"""Evaluation-only conditions and matched controls under a declared development recipe."""

from dataclasses import replace

from parkour_lab.environments.faults import JOINT_NAMES, FaultConfig

from .cases import case_bank

VERSION = "go2_robustness_development_v2"
JOINTS = JOINT_NAMES
FACTORS = (1.0, 0.5, 0.0)


def fault_onset(case):
    """One seeded 6–8 s control-boundary onset shared by all matched joint cases."""
    return (300 + case.seed % 101) * 0.02


def robustness_bank(partition, *, attempts=1, version="v2"):
    """Core controls, compositions, excluded severities and 72 joint cases.

    The same initial draws, nominal geometry and observation noise are paired
    across conditions. These explicit development severities are not calibrated
    difficulty or a final hidden test. No fault/composition condition is trained.
    """
    if partition not in ("validation", "test"):
        raise ValueError("Robustness banks are excluded from training")
    core = case_bank(partition, attempts=attempts, version=version)
    composed = tuple(
        replace(case, condition="composition")
        for case in core
        if case.family in ("Slopes", "Stairs", "Hills")
    )
    extrapolated = tuple(
        replace(case, condition="terrain_ood") for case in core if case.family != "Flat"
    )
    faults = tuple(
        replace(
            case,
            condition="fault",
            fault=FaultConfig(joint, factor, fault_onset(case)),
        )
        for case in core
        if case.profile == "forward-035"
        and (
            case.family == "Flat"
            or (
                case.family == "Uneven"
                and case.physical_level == (3 if version == "v1" else 8)
            )
        )
        for joint in JOINTS
        for factor in FACTORS
    )
    return core + composed + extrapolated + faults


def matched_control(case):
    """Return the exact prescription for a paired base or healthy-joint control."""
    if case.condition == "id":
        return None
    if case.fault is not None:
        return replace(case, fault=replace(case.fault, factor=1.0))
    return replace(case, condition="id")


def block_name(case):
    if case.fault is not None:
        return f"fault_drive_{case.fault.factor:g}"
    return case.condition
