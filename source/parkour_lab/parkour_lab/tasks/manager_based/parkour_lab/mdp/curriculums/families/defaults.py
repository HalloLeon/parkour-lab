"""Composition of the default parkour obstacle-family catalog."""

from __future__ import annotations

from collections.abc import Callable

from ..levels import (
    ParkourFamilyCfg,
    ParkourGeometryVariantCfg,
    ParkourLevelCfg,
)
from . import _shared, gap, high_step, hurdle, tilted_ramps


def _geometry_variants(
    build_levels: Callable[[int], tuple[ParkourLevelCfg, ...]],
) -> tuple[ParkourGeometryVariantCfg, ...]:
    """Build and wrap every deterministic ladder for Hydra serialization."""

    return tuple(
        ParkourGeometryVariantCfg(levels=build_levels(variant_index))
        for variant_index in range(len(_shared.GEOMETRY_VARIANT_OFFSETS))
    )


def build_default_families() -> tuple[ParkourFamilyCfg, ...]:
    """Build the complete family-by-difficulty curriculum matrix."""

    return (
        ParkourFamilyCfg(
            name="gap",
            geometry_variants=_geometry_variants(gap.build_default_levels),
        ),
        ParkourFamilyCfg(
            name="high_step",
            geometry_variants=_geometry_variants(high_step.build_default_levels),
        ),
        ParkourFamilyCfg(
            name="hurdle",
            geometry_variants=_geometry_variants(hurdle.build_default_levels),
        ),
        ParkourFamilyCfg(
            name="tilted_ramps",
            geometry_variants=_geometry_variants(tilted_ramps.build_default_levels),
        ),
    )
