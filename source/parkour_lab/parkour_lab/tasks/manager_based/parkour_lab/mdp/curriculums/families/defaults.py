"""Composition of the default parkour obstacle-family catalog."""

from __future__ import annotations

from ..levels import ParkourFamilyCfg
from . import gap, high_step, hurdle, tilted_ramps


def build_default_families() -> tuple[ParkourFamilyCfg, ...]:
    """Build the complete family-by-difficulty curriculum matrix."""

    return (
        ParkourFamilyCfg(
            name="gap",
            levels=gap.build_default_levels(),
            level_variants=gap.build_level_variants(),
        ),
        ParkourFamilyCfg(
            name="high_step",
            levels=high_step.build_default_levels(),
            level_variants=high_step.build_level_variants(),
        ),
        ParkourFamilyCfg(
            name="hurdle",
            levels=hurdle.build_default_levels(),
            level_variants=hurdle.build_level_variants(),
        ),
        ParkourFamilyCfg(
            name="tilted_ramps",
            levels=tilted_ramps.build_default_levels(),
            level_variants=tilted_ramps.build_level_variants(),
        ),
    )
