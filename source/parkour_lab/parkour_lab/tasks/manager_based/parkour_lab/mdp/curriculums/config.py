from __future__ import annotations

from collections.abc import Iterable
from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .difficulty import difficulty_to_level
from .families.defaults import build_default_families
from .levels import (
    ParkourFamilyCfg,
    ParkourLevelCfg,
    ParkourStructureCfg,
    base_ground_structures,
    coerce_family_cfg,
)

_DEFAULT_PARKOUR_FAMILIES = build_default_families()


@configclass
class ParkourTerrainLayout:
    """Map Isaac Lab's physical terrain grid to the curriculum matrix.

    Isaac Lab stores generated tile origins in a grid whose shape is
    ``(num_rows, num_columns, 3)``. This project gives those physical axes the
    following semantic meaning:

    * Physical row ``r`` represents curriculum difficulty ``r``. There must be
      exactly one row for every shared difficulty in the curriculum.
    * Physical columns are sampling slots rather than unique obstacle
      families. ``family_index_by_column[c]`` and
      ``geometry_variant_index_by_column[c]`` jointly identify the exact
      prebuilt course ladder generated in column ``c``.

    Keeping both parts in one value object makes the complete grid contract
    explicit wherever terrain generation and runtime indexing are connected.
    For example, a seven-difficulty, four-family training layout with 40 columns
    has seven rows, ten columns per family, and two columns per one of five
    geometry variants. Fixed-family evaluation instead maps every column to the
    selected family's canonical variant while retaining the same rows.

    The layout is a mutable configclass because it is stored in an event term's
    ``params`` mapping. Isaac Lab's Hydra bridge reconstructs that mapping by
    assigning fields back onto the existing object, even without CLI overrides.

    Attributes:
        num_difficulty_rows: Number of physical terrain rows, equal to the
            number of shared curriculum difficulties.
        family_index_by_column: One curriculum-family index for every physical
            terrain column, ordered by column index.
        geometry_variant_index_by_column: One deterministic within-family
            geometry-variant index for every physical terrain column.
    """

    num_difficulty_rows: int = MISSING
    family_index_by_column: tuple[int, ...] = MISSING
    geometry_variant_index_by_column: tuple[int, ...] = MISSING

    @property
    def num_columns(self) -> int:
        """Return the number of physical terrain columns described."""

        return len(self.family_index_by_column)

    def validate_grid(
        self,
        *,
        curriculum_difficulties: int,
        curriculum_families: int,
        curriculum_geometry_variants: int,
        terrain_columns: int,
        terrain_rows: int,
    ) -> None:
        """Require the semantic layout and generated terrain grid to agree."""

        if self.num_difficulty_rows != curriculum_difficulties:
            raise ValueError(
                "The terrain layout and curriculum must describe the same "
                "number of difficulty rows: "
                f"got {self.num_difficulty_rows} and {curriculum_difficulties}."
            )
        if terrain_rows != self.num_difficulty_rows:
            raise ValueError(
                "Parkour terrain rows and difficulty levels must match "
                f"one-to-one: got {terrain_rows} rows and "
                f"{self.num_difficulty_rows} difficulties."
            )
        if terrain_columns != self.num_columns:
            raise ValueError(
                "The terrain layout must contain one family index per physical "
                f"column: got {self.num_columns} indices and "
                f"{terrain_columns} columns."
            )
        if len(self.geometry_variant_index_by_column) != self.num_columns:
            raise ValueError("The terrain layout must contain one geometry variant index per physical column.")
        if any(family_index < 0 or family_index >= curriculum_families for family_index in self.family_index_by_column):
            raise ValueError("The terrain layout contains an out-of-range family index.")
        if any(
            variant_index < 0 or variant_index >= curriculum_geometry_variants
            for variant_index in self.geometry_variant_index_by_column
        ):
            raise ValueError("The terrain layout contains an out-of-range geometry variant index.")


@configclass
class ParkourCurriculumCfg:
    """Balanced family-variant-difficulty curriculum matrix.

    Terrain rows are shared difficulty indices. Terrain columns are split
    equally among ``families`` so PPO receives the same number of samples from
    gaps, high steps, hurdles, and tilted ramps at every active difficulty.
    """

    families: tuple[ParkourFamilyCfg, ...] = _DEFAULT_PARKOUR_FAMILIES

    # Bootstrap every environment on the shared flat row. Terrain columns
    # already assign future obstacle families, so mastered environments spread
    # asynchronously into the easiest family-specific courses.
    distribute_initial_levels: bool = True
    initial_level: int = 0
    max_level: int = 6

    # Default root XY radius for waypoint transitions. Control targets may also
    # cross their route plane inside this global corridor; physical targets
    # additionally require named-support contact. Individual waypoints may
    # override only the root radius. Final completion also requires safe
    # whole-body state, but no dwell.
    waypoint_reach_threshold: float = 0.20

    # Rolling evidence tolerates occasional exploration failures without letting
    # one lucky completion advance an unmastered policy.
    promotion_window: int = 5
    promotion_successes_required: int = 3
    demotion_window: int = 3
    demotion_failures_required: int = 2

    # A failed frontier attempt is stalled when it passes less than this
    # fraction of its intermediate route waypoints. Replay episodes do not
    # contribute transition evidence, and the first episode after promotion is
    # protected from demotion.
    # Below the ceiling, keep the total replay budget at 25% while retaining
    # both the shared flat bootstrap and the frontier's immediate predecessor.
    # At the ceiling the combined budget is spread uniformly over every lower
    # row, maintaining the whole acquired ladder without increasing replay.
    demotion_progress_fraction: float = 0.60
    post_promotion_grace_episodes: int = 1
    bootstrap_replay_probability: float = 0.10
    predecessor_replay_probability: float = 0.15

    # Shared threshold for named-support evidence and fatal chassis contact.
    contact_force_threshold: float = 1.0

    # A contacted foot within this metric distance of a support boundary is
    # counted by the edge penalty.
    edge_width_threshold: float = 0.05
    foot_edge_contact_threshold: float = 1.0

    def __post_init__(self) -> None:
        self.validate_configuration()

    def course(
        self,
        family_index: int,
        difficulty_index: int,
        geometry_variant_index: int = 0,
    ) -> ParkourLevelCfg:
        """Return one exact cell of the family-variant-difficulty matrix."""

        if not 0 <= family_index < len(self.families):
            raise IndexError("family_index is out of range.")
        if not 0 <= difficulty_index < self.num_difficulties:
            raise IndexError("difficulty_index is out of range.")
        if not 0 <= geometry_variant_index < self.num_geometry_variants:
            raise IndexError("geometry_variant_index is out of range.")
        return self.families[family_index].geometry_variants[geometry_variant_index].levels[difficulty_index]

    def course_index(
        self,
        family_index: int,
        difficulty_index: int,
        geometry_variant_index: int = 0,
    ) -> int:
        """Flatten one matrix cell for vectorized runtime table indexing."""

        self.course(family_index, difficulty_index, geometry_variant_index)
        return (
            family_index * self.num_geometry_variants + geometry_variant_index
        ) * self.num_difficulties + difficulty_index

    @property
    def courses(self) -> tuple[ParkourLevelCfg, ...]:
        """Return family-then-variant-major cells for runtime lookup tables."""

        return tuple(
            level for family in self.families for variant in family.geometry_variants for level in variant.levels
        )

    def family_index(self, family_name: str) -> int:
        """Return the stable index of a configured obstacle family."""

        try:
            return self.family_names.index(family_name)
        except ValueError as error:
            raise ValueError(
                f"Unknown terrain family {family_name!r}; choose one of {list(self.family_names)}."
            ) from error

    @property
    def family_names(self) -> tuple[str, ...]:
        """Return obstacle-family names in their stable terrain-column order."""

        return tuple(family.name for family in self.families)

    def metadata(self) -> dict[str, object]:
        """Return the complete JSON-compatible curriculum matrix."""

        return {
            "family_order": list(self.family_names),
            "num_difficulties": self.num_difficulties,
            "num_geometry_variants": self.num_geometry_variants,
            "families": [family.metadata() for family in self.families],
        }

    @property
    def num_difficulties(self) -> int:
        """Return the shared number of terrain difficulty rows."""

        return len(self.families[0].canonical_levels)

    @property
    def num_geometry_variants(self) -> int:
        """Return the shared number of deterministic geometry variants."""

        return len(self.families[0].geometry_variants)

    def terrain_layout(
        self,
        num_columns: int,
        *,
        family_name: str | None = None,
    ) -> ParkourTerrainLayout:
        """Describe how physical terrain rows and columns encode the matrix.

        Training divides columns into contiguous family blocks and then into
        near-balanced contiguous variant blocks. Supplying ``family_name``
        instead maps every column to that family's canonical variant for fixed
        evaluation. Rows always retain their one-to-one correspondence with
        the shared curriculum difficulties.
        """

        if isinstance(num_columns, bool) or not isinstance(num_columns, int) or num_columns <= 0:
            raise ValueError("num_columns must be a positive integer.")

        if family_name is not None:
            family_index_by_column = (self.family_index(family_name),) * num_columns
            # Fixed evaluation stays nominal and directly comparable across
            # runs; training is what spreads deterministic variants by column.
            geometry_variant_index_by_column = (0,) * num_columns
        else:
            num_families = len(self.families)
            if num_columns % num_families != 0:
                raise ValueError(
                    "Balanced family sampling requires num_columns to be "
                    f"divisible by the {num_families} obstacle families."
                )
            columns_per_family = num_columns // num_families
            family_index_by_column = tuple(column // columns_per_family for column in range(num_columns))
            geometry_variant_index_by_column = tuple(
                min(
                    (column % columns_per_family) * self.num_geometry_variants // columns_per_family,
                    self.num_geometry_variants - 1,
                )
                for column in range(num_columns)
            )

        return ParkourTerrainLayout(
            num_difficulty_rows=self.num_difficulties,
            family_index_by_column=family_index_by_column,
            geometry_variant_index_by_column=geometry_variant_index_by_column,
        )

    def validate_configuration(self) -> None:
        """Validate matrix shape, balance assumptions, and transition settings."""

        # Hydra can turn nested dataclasses into dictionaries. Reconstruct each
        # family once so all downstream consumers receive one typed matrix.
        self.families = tuple(coerce_family_cfg(family) for family in self.families)
        if not self.families:
            raise ValueError("Parkour curriculum families must not be empty.")

        family_names = self.family_names
        if len(family_names) != len(set(family_names)):
            raise ValueError("Parkour curriculum family names must be unique.")

        difficulty_counts = {len(family.canonical_levels) for family in self.families}
        if len(difficulty_counts) != 1:
            raise ValueError("Every obstacle family must define the same difficulty rows.")

        variant_counts = {len(family.geometry_variants) for family in self.families}
        if len(variant_counts) != 1:
            raise ValueError("Every obstacle family must define the same number of geometry variants.")

        difficulty_orders = tuple(level.difficulty.order for level in self.families[0].canonical_levels)
        if any(
            tuple(level.difficulty.order for level in family.canonical_levels) != difficulty_orders
            for family in self.families[1:]
        ):
            raise ValueError("Every obstacle family must use the same difficulty ranks by row.")

        if self.initial_level < 0 or self.initial_level >= self.num_difficulties:
            raise ValueError("initial_level is out of range.")

        if self.max_level < self.initial_level or self.max_level >= self.num_difficulties:
            raise ValueError("max_level is out of range.")

        if not np.isfinite(self.waypoint_reach_threshold) or self.waypoint_reach_threshold <= 0.0:
            raise ValueError("waypoint_reach_threshold must be positive.")

        for field_name in (
            "promotion_window",
            "promotion_successes_required",
            "demotion_window",
            "demotion_failures_required",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        if self.promotion_successes_required > self.promotion_window:
            raise ValueError("promotion_successes_required must not exceed promotion_window.")
        if self.demotion_failures_required > self.demotion_window:
            raise ValueError("demotion_failures_required must not exceed demotion_window.")

        if not np.isfinite(self.demotion_progress_fraction) or not 0.0 < self.demotion_progress_fraction <= 1.0:
            raise ValueError("demotion_progress_fraction must be in (0, 1].")

        if (
            isinstance(self.post_promotion_grace_episodes, bool)
            or not isinstance(self.post_promotion_grace_episodes, int)
            or self.post_promotion_grace_episodes < 0
        ):
            raise ValueError("post_promotion_grace_episodes must be a non-negative integer.")

        for field_name, probability in (
            ("bootstrap_replay_probability", self.bootstrap_replay_probability),
            ("predecessor_replay_probability", self.predecessor_replay_probability),
        ):
            if not np.isfinite(probability) or not 0.0 <= probability < 1.0:
                raise ValueError(f"{field_name} must be in [0, 1).")
        if self.bootstrap_replay_probability + self.predecessor_replay_probability >= 1.0:
            raise ValueError("Replay probabilities must sum to less than 1.")

        if self.contact_force_threshold < 0.0:
            raise ValueError("contact_force_threshold must be non-negative.")

        if not np.isfinite(self.edge_width_threshold) or self.edge_width_threshold <= 0.0:
            raise ValueError("edge_width_threshold must be positive.")

        if not np.isfinite(self.foot_edge_contact_threshold) or self.foot_edge_contact_threshold < 0.0:
            raise ValueError("foot_edge_contact_threshold must be non-negative.")


DEFAULT_PARKOUR_CURRICULUM = ParkourCurriculumCfg()


def parkour_terrain(difficulty: float, cfg: ParkourTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a terrain tile from base-support patches and structures."""

    levels = cfg.levels
    level = levels[difficulty_to_level(difficulty, len(levels))]
    level.validate_terrain_size(cfg.size)
    terrain_center = _terrain_local_center(cfg)
    ground_structures = base_ground_structures(
        level,
        mesh_factory=trimesh.creation.box,
        ground_thickness=cfg.ground_thickness,
    )

    meshes: list[trimesh.Trimesh] = []
    for structure in (*ground_structures, *level.structures):
        meshes.extend(_structure_meshes(structure, terrain_center))

    return meshes, terrain_center.copy()


@configclass
class ParkourTerrainCfg(SubTerrainBaseCfg):
    """Terrain config for courses composed from reusable structures."""

    function = parkour_terrain

    levels: tuple[ParkourLevelCfg, ...] = DEFAULT_PARKOUR_CURRICULUM.families[0].canonical_levels

    ground_thickness: float = 0.05


def parkour_sub_terrains(
    curriculum_cfg: ParkourCurriculumCfg,
    terrain_layout: ParkourTerrainLayout,
    *,
    ground_thickness: float = 0.05,
) -> dict[str, ParkourTerrainCfg]:
    """Build exact sub-terrain ladders in physical column-block order."""

    column_pairs = tuple(
        zip(
            terrain_layout.family_index_by_column,
            terrain_layout.geometry_variant_index_by_column,
            strict=True,
        )
    )
    ordered_pairs = tuple(dict.fromkeys(column_pairs))
    reconstructed = tuple(pair for pair in ordered_pairs for _ in range(column_pairs.count(pair)))
    if reconstructed != column_pairs:
        raise ValueError("Each family-variant terrain selection must occupy one contiguous column block.")

    return {
        f"{curriculum_cfg.families[family_index].name}_variant_{variant_index}": ParkourTerrainCfg(
            proportion=column_pairs.count((family_index, variant_index)) / terrain_layout.num_columns,
            levels=curriculum_cfg.families[family_index].geometry_variants[variant_index].levels,
            ground_thickness=ground_thickness,
        )
        for family_index, variant_index in ordered_pairs
    }


PARKOUR_TERRAIN_GENERATOR_CFG = TerrainGeneratorCfg(
    # Enable Isaac Lab's terrain-curriculum layout.
    #
    # With curriculum=True, terrain rows correspond to difficulty levels.
    # In our case, each row is one parkour curriculum level.
    curriculum=True,
    # Physical size of one terrain tile in meters: (x_size, y_size).
    size=(8.0, 4.0),
    # Extra terrain border around the whole generated terrain.
    border_width=5.0,
    # One terrain row per shared difficulty level.
    num_rows=DEFAULT_PARKOUR_CURRICULUM.num_difficulties,
    # Number of terrain columns per curriculum row.
    num_cols=40,
    # Horizontal resolution used by height-field/mesh terrain utilities.
    #
    # For this custom trimesh terrain, this is not the main geometric control;
    # the actual geometry comes from the configured mesh factories.
    #
    # Keep it reasonably small and standard.
    horizontal_scale=0.05,
    # Vertical resolution used by terrain utilities.
    #
    # Geometry is defined directly by each mesh factory. This value is still
    # required by TerrainGeneratorCfg.
    vertical_scale=0.005,
    # Slope threshold used by some terrain-generation utilities to correct or
    # simplify steep surfaces.
    #
    # Custom meshes may contain steep or vertical faces, so this is not their
    # primary geometry control. Keep it at a conservative default.
    slope_threshold=0.75,
    # Disable terrain cache.
    #
    # use_cache=False is useful while actively developing terrain code, because
    # changes take effect immediately.
    use_cache=False,
    # Isaac Lab assigns sub-terrain types to contiguous column blocks in this
    # stable dictionary order. Every block owns one exact family/variant ladder.
    sub_terrains=parkour_sub_terrains(
        DEFAULT_PARKOUR_CURRICULUM,
        DEFAULT_PARKOUR_CURRICULUM.terrain_layout(40),
    ),
)


def _normalize_mesh_result(result: object, factory: object) -> list[trimesh.Trimesh]:
    """Normalize common Trimesh factory outputs into independent meshes."""

    # Wrap a single mesh in a list so all callers can process one consistent
    # return type. Copy it because the caller subsequently transforms it.
    if isinstance(result, trimesh.Trimesh):
        return [result.copy()]

    # A scene may contain several positioned geometries. ``dump`` resolves its
    # scene graph and returns the geometries as transformed mesh objects.
    if isinstance(result, trimesh.Scene):
        result = result.dump(concatenate=False)
        if isinstance(result, trimesh.Trimesh):
            return [result.copy()]

    # Factories may also return a sequence or generator of meshes. Strings and
    # bytes are iterable too, but cannot represent a valid collection of meshes.
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes)):
        meshes = list(result)
        if all(isinstance(mesh, trimesh.Trimesh) for mesh in meshes):
            # Return independent copies so applying a structure transform does
            # not mutate meshes retained or reused by the factory.
            return [mesh.copy() for mesh in meshes]

    # Report the factory as well as the accepted return types to make malformed
    # custom structure factories straightforward to identify.
    raise TypeError(f"Mesh factory {factory!r} must return a Trimesh, Scene, or iterable of Trimesh objects.")


def _structure_meshes(structure: ParkourStructureCfg, terrain_center: np.ndarray) -> list[trimesh.Trimesh]:
    """Create and rigidly transform all meshes produced by one structure."""

    # Call the configured factory with its declarative keyword arguments.
    # Normalize its possible mesh, scene, or iterable output into independent
    # meshes that are safe to transform.
    meshes = _normalize_mesh_result(
        structure.mesh_factory(**structure.mesh_kwargs),
        structure.mesh_factory,
    )

    # Structure positions are expressed relative to the terrain tile center.
    # Adding the center converts that local offset into tile mesh coordinates.
    translation = terrain_center + np.asarray(structure.position, dtype=np.float64)

    # Build one homogeneous transform containing the structure's XYZ
    # translation and roll-pitch-yaw rotation.
    transform = trimesh.transformations.compose_matrix(
        translate=translation,
        angles=structure.orientation_rpy,
    )

    # Apply the same rigid pose to every mesh returned for this structure.
    for mesh in meshes:
        mesh.apply_transform(transform)
    return meshes


def _terrain_local_center(cfg: ParkourTerrainCfg) -> np.ndarray:
    """Return the terrain tile center used as its environment origin."""

    size_x, size_y = cfg.size
    return np.array([0.5 * size_x, 0.5 * size_y, 0.0], dtype=np.float32)
