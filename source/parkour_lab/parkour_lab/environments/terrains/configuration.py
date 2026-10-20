"""Lazy configuration of stable native terrain columns and physical severity rows."""

from .generation import generate_terrain
from .specification import (
    HORIZONTAL_SCALE_M,
    TILE_SIZE_M,
    VERTICAL_SCALE_M,
    terrain_specification,
)


def configure_terrain(cfg, options, *, training, task_id, seed):
    """Bind native geometry, progression and tile limits.

    Fixed-level evaluation keeps ordered generation with one explicit severity
    row. Disabling a runtime curriculum term cannot
    accidentally switch this to random family/difficulty generation.
    """
    receipt = terrain_specification(options, training=training, task_id=task_id)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Require a uint32 terrain seed")
    from isaaclab.managers import CurriculumTermCfg, TerminationTermCfg

    from .curriculum import terrain_levels

    cfg.benchmark_terrain = receipt
    cfg.curriculum.terrain_levels = (
        CurriculumTermCfg(func=terrain_levels)
        if training and task_id != "Go2-Flat-v1"
        else None
    )
    terrain = cfg.scene.terrain
    if task_id == "Go2-Flat-v1":
        terrain.terrain_type = "plane"
        terrain.terrain_generator = None
        cfg.curriculum.terrain_levels = None
        cfg.terminations.terrain_boundary = None
        return receipt

    from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
    from isaaclab.utils import configclass

    @configclass
    class PhysicalTerrainCfg(SubTerrainBaseCfg):
        function = generate_terrain
        family: str = "Flat"
        subtype: str = "plane"
        column: int = 0
        fixed_level: int | None = None

    @configclass
    class EvaluationTerrainCfg(PhysicalTerrainCfg):
        # SDK configclass.copy() rebuilds declared dataclass fields. An instance
        # attribute would disappear before the native generator calls our hook.
        condition: str = "id"

    terrain_class = (
        PhysicalTerrainCfg if options.condition == "id" else EvaluationTerrainCfg
    )
    condition = {} if options.condition == "id" else {"condition": options.condition}
    columns = receipt["columns"]
    sub_terrains = {
        f"column_{column:02d}_{definition['family'].lower()}": terrain_class(
            proportion=1 / len(columns),
            family=definition["family"],
            subtype=definition["subtype"],
            column=column,
            fixed_level=receipt["fixed_level"],
            **condition,
        )
        for column, definition in enumerate(columns)
    }
    terrain.terrain_type = "generator"
    terrain.terrain_generator = TerrainGeneratorCfg(
        seed=seed,
        size=TILE_SIZE_M,
        num_rows=receipt["num_rows"],
        num_cols=len(columns),
        curriculum=True,
        difficulty_range=(0.0, 1.0),
        horizontal_scale=HORIZONTAL_SCALE_M,
        vertical_scale=VERTICAL_SCALE_M,
        border_width=0.0,
        use_cache=False,
        sub_terrains=sub_terrains,
    )
    terrain.max_init_terrain_level = receipt["initial_max_level"] if training else 0
    cfg.terminations.terrain_boundary = TerminationTermCfg(
        func=tile_boundary, time_out=True
    )
    return receipt


def tile_boundary(env):
    """Censor the first root-center sample beyond its assigned tile's XY extent.

    The outgoing transition remains observable; feet may already straddle the
    edge. Progression can inspect the outgoing sample before reset. This is not
    obstacle completion.
    """
    offset = (
        env.scene["robot"].data.root_pos_w[:, :2] - env.scene.terrain.env_origins[:, :2]
    )
    half_x, half_y = (side / 2 for side in env.cfg.benchmark_terrain["tile_size_m"])
    return (offset[:, 0].abs() > half_x) | (offset[:, 1].abs() > half_y)
