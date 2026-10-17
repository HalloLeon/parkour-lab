"""Physical development levels; calibration does not follow from a row number."""

import math
from dataclasses import asdict, dataclass

VERSION = "go2-terrain-development-v1"
CONDITION_VERSION = "go2-terrain-conditions-development-v1"
CONDITIONS = ("id", "composition", "terrain_ood")
LEVEL_COUNT = 8
TILE_SIZE_M = (12.0, 12.0)
HORIZONTAL_SCALE_M = 0.1
VERTICAL_SCALE_M = 0.005
FAMILIES = ("Flat", "Uneven", "Slopes", "Stairs", "Hills")
MIXED_COLUMNS = (
    *(("Flat", "plane"),) * 4,
    *(("Uneven", "noise"),) * 4,
    *(("Slopes", "pyramid"),) * 2,
    *(("Slopes", "inverted_pyramid"),) * 2,
    *(("Stairs", "pyramid"),) * 2,
    *(("Stairs", "inverted_pyramid"),) * 2,
    *(("Hills", "wave"),) * 4,
)


@dataclass(frozen=True)
class TerrainConfig:
    """Select fixed evaluation severity or the initial training level interval.

    Geometry is a versioned development recipe, not a calibrated capability claim.
    Fixed levels apply to non-flat views; Flat never gains a difficulty ladder.
    """

    fixed_level: int | None = None
    initial_level_range: tuple[int, int] = (0, 1)
    condition: str = "id"

    def __post_init__(self):
        if type(self.condition) is not str or self.condition not in CONDITIONS:
            raise ValueError(f"Terrain condition must be one of {CONDITIONS}")
        if self.fixed_level is not None and (
            type(self.fixed_level) is not int or not 0 <= self.fixed_level < LEVEL_COUNT
        ):
            raise ValueError("Fixed terrain level must be an integer in [0, 7]")
        bounds = self.initial_level_range
        if (
            type(bounds) is not tuple
            or len(bounds) != 2
            or any(type(value) is not int for value in bounds)
            or not 0 <= bounds[0] <= bounds[1] < LEVEL_COUNT
        ):
            raise ValueError(
                "Initial terrain levels require ordered integers in [0, 7]"
            )

    def as_dict(self):
        values = asdict(self)
        if self.condition == "id":
            values.pop("condition")
        return values


def physical_level(level, *, condition="id"):
    """Return requested native-generator parameters with units and wave convention."""
    if type(level) is not int or not 0 <= level < LEVEL_COUNT:
        raise ValueError("Terrain level must be an integer in [0, 7]")
    if type(condition) is not str or condition not in CONDITIONS:
        raise ValueError(f"Terrain condition must be one of {CONDITIONS}")
    fraction = level / (LEVEL_COUNT - 1)
    amplitude = 0.02 + 0.28 * fraction
    slope = 0.035 + 0.265 * fraction
    values = {
        "level": level,
        "uneven_noise_amplitude_m": 0.005 + 0.045 * fraction,
        "uneven_sample_spacing_m": 0.8 - 0.6 * fraction,
        "uneven_noise_step_m": VERTICAL_SCALE_M,
        "slope_rise_run": slope,
        "slope_angle_deg": math.degrees(math.atan(slope)),
        "stair_riser_m": 0.02 + 0.14 * fraction,
        "stair_tread_m": 0.3,
        "pyramid_platform_width_m": 2.0,
        # Native waves use half this parameter on each of their two axes.
        "hill_native_amplitude_m": amplitude,
        "hill_per_axis_amplitude_bound_m": amplitude / 2,
        "hill_num_waves": 2,
        "hill_wavelength_m": 5.95,
        "hill_analytic_grade_bound": math.sqrt(2) * amplitude * math.pi / 5.95,
    }
    if condition == "terrain_ood":
        slope = 0.35 + 0.10 * fraction
        amplitude = 0.35 + 0.10 * fraction
        values.update(
            uneven_noise_amplitude_m=0.06 + 0.04 * fraction,
            slope_rise_run=slope,
            slope_angle_deg=math.degrees(math.atan(slope)),
            stair_riser_m=0.18 + 0.04 * fraction,
            hill_native_amplitude_m=amplitude,
            hill_per_axis_amplitude_bound_m=amplitude / 2,
            hill_analytic_grade_bound=math.sqrt(2) * amplitude * math.pi / 5.95,
        )
    return values


def columns_for(task_id):
    """The ordered native terrain columns, with equal mixed family allocation."""
    if task_id == "Go2-Mixed-v1":
        return MIXED_COLUMNS
    family = next((name for name in FAMILIES if task_id == f"Go2-{name}-v1"), None)
    if family is None:
        raise ValueError(f"Unknown terrain task: {task_id}")
    if family == "Flat":
        return (("Flat", "plane"),)
    kinds = tuple(pair for pair in MIXED_COLUMNS if pair[0] == family)
    return kinds * (20 // len(kinds))


def terrain_specification(options, *, training, task_id):
    """JSON-safe recipe, independent of native imports and runtime level updates."""
    if not isinstance(options, TerrainConfig) or type(training) is not bool:
        raise ValueError("Require TerrainConfig and a boolean training role")
    if training and options.fixed_level is not None:
        raise ValueError("Fixed terrain levels are evaluation-only")
    if training and options.condition != "id":
        raise ValueError("Terrain compositions and excluded ranges are evaluation-only")
    family = task_id.removeprefix("Go2-").removesuffix("-v1")
    if options.condition == "composition" and family not in (
        "Slopes",
        "Stairs",
        "Hills",
        "Mixed",
    ):
        raise ValueError("Surface composition requires Slopes, Stairs, Hills or Mixed")
    if options.condition == "terrain_ood" and family == "Flat":
        raise ValueError("Flat has no excluded terrain parameter range")
    columns = columns_for(task_id)
    fixed_level = (
        None
        if training or task_id == "Go2-Flat-v1"
        else 3 if options.fixed_level is None else options.fixed_level
    )
    receipt = {
        "version": VERSION,
        "status": "development_geometry_pending_native_calibration",
        "task_id": task_id,
        "columns": [dict(family=family, subtype=kind) for family, kind in columns],
        "level_count": LEVEL_COUNT,
        "initial_min_level": options.initial_level_range[0],
        "initial_max_level": options.initial_level_range[1],
        "fixed_level": fixed_level,
        "num_rows": LEVEL_COUNT if training and task_id != "Go2-Flat-v1" else 1,
        "physical_levels_by_row": (
            (list(range(LEVEL_COUNT)) if training else [fixed_level])
            if task_id != "Go2-Flat-v1"
            else [None]
        ),
        "tile_size_m": list(TILE_SIZE_M),
        "horizontal_scale_m": HORIZONTAL_SCALE_M,
        "vertical_scale_m": VERTICAL_SCALE_M,
        "heightfield_implicit_border_m": HORIZONTAL_SCALE_M,
        "heightfield_interpretation": (
            "requested parameters; native height quantization and cubic Uneven "
            "interpolation affect realized heights/grades; interpolation may overshoot"
        ),
        "levels": [
            physical_level(level, condition=options.condition)
            for level in range(LEVEL_COUNT)
        ],
        "training_level_cap": LEVEL_COUNT - 1,
        "calibration": "entry/support/ID-OOD caps remain uncalibrated until native evidence",
        "fixed_test_levels": {"easy": 0, "medium": 3, "hard": 7},
        "flat_fraction": sum(family == "Flat" for family, _ in columns) / len(columns),
        "initialization": "private seeded level draw; Flat holds level zero",
        "layout": "ordered columns and rows; runtime level updates are separate",
        "boundary": "root crossing tile edge truncates; edge samples disclose footprint spill",
        "exposure": "origin/platform occupancy is not evidence of non-flat contact",
        "direction": "pyramid sign is geometry, not measured ascent/descent motion",
        "roughness_layers": "Uneven only; no added noise on slopes, stairs or hills",
        "instance_variation": (
            "Uneven noise is seeded by level/column; other native primitives repeat "
            "their declared geometry, without hidden phase or roughness variation"
        ),
    }
    if options.condition != "id":
        receipt.update(
            condition=options.condition,
            condition_version=CONDITION_VERSION,
            training_support={
                "minimum": physical_level(0),
                "maximum": physical_level(LEVEL_COUNT - 1),
            },
            exclusion="not available to core training, curriculum or method selection",
        )
        if options.condition == "composition":
            receipt["composition"] = {
                "families": ["Slopes", "Stairs", "Hills"],
                "layer": "native Uneven heights added vertically to the same support mesh",
                "component_level": fixed_level,
                "central_clear_half_width_m": 1.0,
                "central_blend_width_m": 0.5,
                "perimeter_blend_width_m": 0.3,
                "stair_surface_grid_max_spacing_m": HORIZONTAL_SCALE_M,
                "origin": "unchanged native base origin; central support unchanged",
                "bounds": "component parameters ID; realized grades/transitions may exceed ID",
            }
            receipt["roughness_layers"] = "ID Uneven layer on Slopes/Stairs/Hills only"
            receipt["instance_variation"] = (
                "native base plus Uneven layer seeded by the same level/column; "
                "central start and perimeter blends are fixed, not randomized"
            )
        else:
            receipt["excluded_training_support"] = {
                "Uneven": "noise amplitude exceeds 0.05 m",
                "Slopes": "rise/run exceeds 0.30",
                "Stairs": "riser exceeds 0.16 m",
                "Hills": "native wave amplitude exceeds 0.30 m",
            }
    return receipt
