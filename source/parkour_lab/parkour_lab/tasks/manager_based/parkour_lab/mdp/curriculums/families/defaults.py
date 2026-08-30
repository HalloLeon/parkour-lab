"""Immutable built-in parkour course families."""

from __future__ import annotations

from collections.abc import Callable

from ..levels import ParkourFamilyCfg, ParkourLevelCfg
from . import _shared, gap, high_step, hurdle, tilted_ramps


def _geometry_variants(
    build_levels: Callable[[int], tuple[ParkourLevelCfg, ...]],
) -> tuple[tuple[ParkourLevelCfg, ...], ...]:
    """Build every deterministic difficulty ladder."""

    return tuple(
        build_levels(variant_index)
        for variant_index in range(len(_shared.GEOMETRY_VARIANT_OFFSETS))
    )


def _build_course_families() -> tuple[ParkourFamilyCfg, ...]:
    """Build the complete family-by-difficulty curriculum matrix."""

    families = (
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
    difficulty_orders = tuple(
        level.difficulty.order for level in families[0].canonical_levels
    )
    if len({family.name for family in families}) != len(families) or any(
        len(family.geometry_variants) != len(families[0].geometry_variants)
        or tuple(level.difficulty.order for level in family.canonical_levels)
        != difficulty_orders
        for family in families[1:]
    ):
        raise ValueError(
            "Built-in course families must share unique names, variants, and difficulty rows."
        )
    return families


COURSE_FAMILIES = _build_course_families()
