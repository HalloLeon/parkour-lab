"""Thin severity bindings around native Isaac Lab terrain generators."""

from copy import deepcopy

import numpy as np

from .specification import (
    HORIZONTAL_SCALE_M,
    LEVEL_COUNT,
    VERTICAL_SCALE_M,
    physical_level,
)


def generate_terrain(difficulty, cfg):
    """Quantize native row bands to exact levels, then call the native generator.

    Evaluation overrides the band with its declared fixed physical level. Native
    random-uniform interpolation uses NumPy's legacy global RNG; isolate that
    call so generated instances neither depend on nor advance a learner's RNG.
    """
    if not np.isfinite(difficulty) or not 0 <= difficulty <= 1:
        raise ValueError("Terrain difficulty must be finite and in [0, 1]")
    level = min(int(difficulty * LEVEL_COUNT), LEVEL_COUNT - 1)
    if cfg.fixed_level is not None:
        level = cfg.fixed_level
    condition = getattr(cfg, "condition", "id")
    native = native_terrain_config(
        cfg.family, cfg.subtype, level, size=cfg.size, condition=condition
    )
    if level == 0 and condition != "terrain_ood":
        return native.function(0.0, native)
    if cfg.family == "Uneven":
        return _seeded_noise(native, cfg, level)
    pieces, origin = native.function(0.0, native)
    if condition == "composition" and cfg.family in ("Slopes", "Stairs", "Hills"):
        from .composition import compose_surface

        noise = native_terrain_config("Uneven", "noise", level, size=cfg.size)
        roughness, _ = _seeded_noise(noise, cfg, level)
        pieces = compose_surface(pieces, roughness[0], family=cfg.family, size=cfg.size)
    return pieces, origin


def _seeded_noise(native, cfg, level):
    """Native noise sampling retains the existing column/level seed and global RNG."""
    state = np.random.get_state()
    try:
        seed = np.random.SeedSequence((cfg.seed, cfg.column, level)).generate_state(1)[
            0
        ]
        np.random.seed(seed)
        return native.function(0.0, native)
    finally:
        np.random.set_state(state)


def native_terrain_config(family, subtype, level, *, size, condition="id"):
    """Configure native primitives; no copied terrain algorithm or extra noise layer."""
    import isaaclab.terrains as native

    values = physical_level(level, condition=condition)
    common = dict(size=deepcopy(size))
    supported = {
        "Flat": ("plane",),
        "Uneven": ("noise",),
        "Slopes": ("pyramid", "inverted_pyramid"),
        "Stairs": ("pyramid", "inverted_pyramid"),
        "Hills": ("wave",),
    }
    if subtype not in supported.get(family, ()):
        raise ValueError(f"Unsupported terrain family/subtype: {family}/{subtype}")
    if family == "Flat" or (level == 0 and condition != "terrain_ood"):
        return native.MeshPlaneTerrainCfg(**common)
    heightfield = dict(
        **common,
        horizontal_scale=HORIZONTAL_SCALE_M,
        vertical_scale=VERTICAL_SCALE_M,
        border_width=0.0,
        slope_threshold=None,
    )
    if family == "Uneven" and subtype == "noise":
        amplitude = values["uneven_noise_amplitude_m"]
        return native.HfRandomUniformTerrainCfg(
            **heightfield,
            noise_range=(-amplitude, amplitude),
            noise_step=values["uneven_noise_step_m"],
            downsampled_scale=values["uneven_sample_spacing_m"],
        )
    if family == "Slopes" and subtype in ("pyramid", "inverted_pyramid"):
        slope = values["slope_rise_run"]
        return native.HfPyramidSlopedTerrainCfg(
            **heightfield,
            slope_range=(slope, slope),
            platform_width=values["pyramid_platform_width_m"],
            inverted=subtype == "inverted_pyramid",
        )
    if family == "Stairs" and subtype in ("pyramid", "inverted_pyramid"):
        kind = (
            native.MeshPyramidStairsTerrainCfg
            if subtype == "pyramid"
            else native.MeshInvertedPyramidStairsTerrainCfg
        )
        rise = values["stair_riser_m"]
        return kind(
            **common,
            step_height_range=(rise, rise),
            step_width=values["stair_tread_m"],
            platform_width=values["pyramid_platform_width_m"],
            border_width=0.3,
            holes=False,
        )
    if family == "Hills" and subtype == "wave":
        amplitude = values["hill_native_amplitude_m"]
        return native.HfWaveTerrainCfg(
            **heightfield,
            amplitude_range=(amplitude, amplitude),
            num_waves=values["hill_num_waves"],
        )
    raise ValueError(f"Unsupported terrain family/subtype: {family}/{subtype}")
