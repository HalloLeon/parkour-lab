"""Independent, recorded random streams for each robot and first-attempt bank ID."""

from copy import deepcopy
import hashlib

import numpy as np

NAMESPACE = "go2-roa-exit-v2-r6"
STREAMS = (
    "geometry",
    "rough-coarse",
    "rough-fine",
    "start",
    "dynamics",
    "observation-noise",
    "commands",
)
TERRAIN_GROUPS = tuple(
    f"{family}/{direction}/{tier}"
    for family, directions, tiers in (
        ("stairs", ("up", "down"), ("04cm", "08cm", "12cm", "16cm")),
        ("ramp", ("up", "down"), ("10deg", "15deg", "20deg")),
        ("hill", ("short-first", "long-first"), ("10deg", "15deg", "20deg")),
    )
    for direction in directions
    for tier in tiers
)


def _stream(namespace, group_id, index, name):
    text = f"{namespace}|{group_id}|{index}|{name}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    generator = np.random.Generator(np.random.PCG64(int(digest, 16)))
    mapping = dict(
        sha256=digest, initial_state=generator.bit_generator.state["state"].copy()
    )
    return generator, mapping


class TaskRandomization:
    """Full SHA-256 entropy seeds a PCG64 generator per row and named stream.

    Subset draws advance only the requested rows, independently of batch ordering.
    Physical parameters stay fixed; reset starts and sensor samples advance their
    own streams. No NumPy, torch, learner, or simulator global RNG is consumed.
    """

    def __init__(self, task, *, evaluation=False, nominal_heading=None):
        def setting(name, default=None):
            return (
                task.get(name, default)
                if isinstance(task, dict)
                else getattr(task, name, default)
            )

        self.count = setting("num_envs")
        profile, seed = setting("bank_profile"), setting("seed")
        terrain_group = setting("terrain_group")
        terrain_index = setting("terrain_attempt_index")
        dynamics = setting(
            "dynamics", None if terrain_group is not None else "randomized"
        )
        terrain_attempt = terrain_group is not None or terrain_index is not None
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("Require a positive environment count")
        self._streams = STREAMS
        self._attempt_indices = list(range(self.count))
        if terrain_attempt:
            if (
                not evaluation
                or self.count != 1
                or setting("terrain") != "connected"
                or profile is not None
                or not isinstance(terrain_group, str)
                or terrain_group not in TERRAIN_GROUPS
                or type(terrain_index) is not int
                or not 0 <= terrain_index < 100
                or nominal_heading is None
            ):
                raise ValueError(
                    "Require one connected development terrain ID and supported heading"
                )
            stratum = "nominal" if terrain_index < 50 else "randomized"
            if dynamics not in (None, stratum):
                raise ValueError("Dynamics conflict with the terrain attempt stratum")
            dynamics = stratum
            self._streams = STREAMS[:-1]
            self._attempt_indices = [terrain_index]
        if (
            type(seed) is not int
            or seed < 0
            or dynamics not in ("nominal", "randomized")
        ):
            raise ValueError("Require a nonnegative run seed and declared dynamics")
        if terrain_attempt:
            self.namespace = f"{NAMESPACE}/development"
            self.group_id = terrain_group
            self.randomized = np.asarray([terrain_index >= 50])
        elif profile is not None:
            from parkour_lab.evaluation.flat import PROFILES

            if not evaluation or profile not in PROFILES or self.count != 100:
                raise ValueError(
                    "A development profile requires evaluation of its 100 IDs"
                )
            self.namespace = f"{NAMESPACE}/development"
            self.group_id = f"flat/{profile}"
            self.randomized = np.arange(self.count) >= 50
        else:
            self.namespace = f"{NAMESPACE}/{'development' if evaluation else 'train'}"
            self.group_id = f"{'diagnostic' if evaluation else 'training'}/{seed}"
            self.randomized = np.full(self.count, dynamics == "randomized")
        self._generators, self._mappings = {}, []
        entropy_seen, backend_seen = set(), set()
        for row, index in enumerate(self._attempt_indices):
            mapping = {}
            for stream in self._streams:
                generator, record = _stream(
                    self.namespace, self.group_id, index, stream
                )
                digest, state = record["sha256"], record["initial_state"]
                backend = (state["state"], state["inc"])
                if digest in entropy_seen or backend in backend_seen:
                    raise ValueError("Random-stream seed collision")
                entropy_seen.add(digest)
                backend_seen.add(backend)
                mapping[stream] = record
                self._generators[row, stream] = generator
            self._mappings.append(mapping)
        self._starts = [None] * self.count
        self._physical = None
        self._heading = np.zeros(self.count)
        if nominal_heading is not None:
            if profile is not None or not np.isfinite(nominal_heading):
                raise ValueError(
                    "Require a finite nominal heading outside the flat bank"
                )
            self._heading.fill(nominal_heading)
        if profile is not None:
            offset = self.uniform(np.arange(self.count), 1, "geometry")[:, 0]
            self._heading = (
                -np.pi + 2 * np.pi * (np.arange(self.count) % 50 + offset) / 50
            )

    def _ids(self, ids):
        if ids is None:
            return list(range(self.count))
        ids = np.asarray(ids)
        if (
            ids.ndim != 1
            or (len(ids) and ids.dtype.kind not in "iu")
            or (ids < 0).any()
            or (ids >= self.count).any()
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("Require distinct valid environment IDs")
        return ids.tolist()

    def uniform(self, ids, width, stream, low=0.0, high=1.0, *, advance=True):
        """Return [rows, width] samples; previews leave all stream states untouched."""
        if stream not in self._streams or type(width) is not int or width <= 0:
            raise ValueError("Require a known stream and positive sample width")
        ids = self._ids(ids)
        values = []
        for row in ids:
            generator = self._generators[row, stream]
            if not advance:
                generator = deepcopy(generator)
            values.append(generator.uniform(low, high, size=width))
        return np.asarray(values, dtype=np.float64).reshape(len(ids), width)

    def begin_attempts(self):
        """Discard warm-up sampling before the explicit first evaluation reset.

        This is not recovery or an opportunity to replace an observed outcome.
        Fixed physical draws and nominal bank headings remain unchanged.
        """
        for row, mapping in enumerate(self._mappings):
            for stream in ("start", "observation-noise", "commands"):
                if stream not in mapping:
                    continue
                self._generators[row, stream] = np.random.Generator(
                    np.random.PCG64(int(mapping[stream]["sha256"], 16))
                )
        self._starts = [None] * self.count

    def sample_start(self, ids, joint_count=12):
        """New reset pose around the nominal start; all velocities remain zero."""
        if type(joint_count) is not int or joint_count <= 0:
            raise ValueError("Require a positive joint count")
        ids = self._ids(ids)
        values = self.uniform(ids, joint_count + 3, "start", -0.05, 0.05)
        heading = self._heading[ids]
        cosine, sine = np.cos(heading), np.sin(heading)
        local_xy = values[:, :2]
        result = dict(
            start_xy=np.column_stack(
                (
                    cosine * local_xy[:, 0] - sine * local_xy[:, 1],
                    sine * local_xy[:, 0] + cosine * local_xy[:, 1],
                )
            ),
            start_yaw=heading + values[:, 2] * 0.6,
            nominal_heading=heading.copy(),
            start_xy_local=local_xy,
            yaw_offset=values[:, 2] * 0.6,
            joint_offset=values[:, 3:],
        )
        for index, row in enumerate(ids):
            if self._starts[row] is None:
                self._starts[row] = {
                    key: value[index].tolist() for key, value in result.items()
                }
        return result

    def dynamics(self, joint_count=12):
        """Draw independent mass, shared robot friction, strength and PD scales once."""
        if type(joint_count) is not int or joint_count <= 0:
            raise ValueError("Require a positive joint count")
        if self._physical is None:
            values = self.uniform(
                np.arange(self.count), 2 + 3 * joint_count, "dynamics"
            )
            varied = self.randomized
            self._physical = dict(
                added_mass=np.where(varied, -1 + 4 * values[:, 0], 0),
                static_friction=np.where(varied, 0.6 + 0.4 * values[:, 1], 0.8),
            )
            self._physical["dynamic_friction"] = (
                0.75 * self._physical["static_friction"]
            )
            for index, name in enumerate(("motor_strength", "kp_scale", "kd_scale")):
                start = 2 + index * joint_count
                self._physical[name] = np.where(
                    varied[:, None],
                    0.9 + 0.2 * values[:, start : start + joint_count],
                    1.0,
                )
        if self._physical["motor_strength"].shape[1] != joint_count:
            raise ValueError("Joint count changed after drawing physical parameters")
        return {key: value.copy() for key, value in self._physical.items()}

    def manifest(self):
        """Record backend mapping and realized first starts without advancing RNGs."""
        attempts = []
        for row in range(self.count):
            physical = (
                None
                if self._physical is None
                else {
                    key: values[row].tolist() for key, values in self._physical.items()
                }
            )
            attempts.append(
                dict(
                    attempt_index=self._attempt_indices[row],
                    stratum="randomized" if self.randomized[row] else "nominal",
                    streams=deepcopy(self._mappings[row]),
                    start=deepcopy(self._starts[row]),
                    dynamics=physical,
                )
            )
        return dict(
            version="go2_task_randomization_v1",
            namespace=self.namespace,
            group_id=self.group_id,
            backend="numpy.random.PCG64",
            seed_mapping="SHA-256 UTF-8 namespace|group_id|attempt_index|stream_name; full digest as big-endian integer",
            seed_collisions=0,
            attempts=attempts,
        )


def development_manifest(profile):
    """Canonical first starts and physical draws, independent of candidate seed."""
    sampler = TaskRandomization(
        dict(num_envs=100, seed=0, bank_profile=profile), evaluation=True
    )
    sampler.dynamics()
    sampler.sample_start(None)
    return sampler.manifest()


def terrain_development_assignments(group_id=None):
    """Assign requested terrain layouts and six streams, without building worlds.

    These prospective IDs are not a realized or native-validated bank. In
    particular, generating assignments does not realize starts, apply dynamics
    or consume noise. No candidate or qualification input is accepted. Each
    50-ID stratum independently balances the target dimensions.
    """
    if group_id is not None and (
        not isinstance(group_id, str) or group_id not in TERRAIN_GROUPS
    ):
        raise ValueError("Require a canonical terrain development group")
    namespace = f"{NAMESPACE}/development"
    groups, entropy_seen, backend_seen = [], set(), set()
    for group in TERRAIN_GROUPS if group_id is None else (group_id,):
        family, direction, tier = group.split("/")
        attempts, geometry = [], []
        for index in range(100):
            mappings = {}
            for name in STREAMS[:-1]:
                generator, mapping = _stream(namespace, group, index, name)
                digest, state = mapping["sha256"], mapping["initial_state"]
                backend = (state["state"], state["inc"])
                if digest in entropy_seen or backend in backend_seen:
                    raise ValueError("Random-stream seed collision")
                entropy_seen.add(digest)
                backend_seen.add(backend)
                mappings[name] = mapping
                if name == "geometry":
                    geometry.append(generator.uniform(size=7))
            attempts.append(
                dict(
                    attempt_index=index,
                    stratum="nominal" if index < 50 else "randomized",
                    streams=mappings,
                )
            )
        geometry = np.asarray(geometry)
        for offset in (0, 50):
            draws = geometry[offset : offset + 50]
            # Independent keys permute dimensions without another seed stream.
            ranks = np.argsort(
                np.argsort(draws[:, [0, 1, 3]], axis=0, kind="stable"),
                axis=0,
                kind="stable",
            )
            for local_index, (values, bins) in enumerate(
                zip(draws, ranks, strict=True)
            ):
                attempt = attempts[offset + local_index]
                dx, dy = -0.25 + 0.5 * values[5:7]
                world = dict(
                    target_family=family,
                    tier=int(tier[:2]) / 100 if family == "stairs" else int(tier[:2]),
                    reverse=direction in ("down", "long-first"),
                    coarse_seed=int(attempt["streams"]["rough-coarse"]["sha256"], 16),
                    fine_seed=int(attempt["streams"]["rough-fine"]["sha256"], 16),
                    size=[24.0, 24.0],
                    resolution=0.02,
                    stair_entry=float(14 + dx),
                    ramp_entry=float(14 + dy),
                    stair_route_y=float(11 + dy),
                    ramp_route_x=float(7 + dx),
                    hill_entry=float(4 + dx),
                    hill_center_y=float(4 + dy),
                    world_yaw=float(-np.pi + 2 * np.pi * (bins[2] + values[4]) / 50),
                )
                geometry_bins = dict(world_yaw=int(bins[2]))
                if family == "stairs":
                    world.update(
                        risers=4 + int(bins[0] % 3),
                        tread=float(0.31 + 0.19 * (bins[1] + values[2]) / 50),
                    )
                    geometry_bins.update(risers=int(bins[0] % 3), tread=int(bins[1]))
                elif family == "ramp":
                    world["incline_length"] = 2 + int(bins[0] % 2)
                    geometry_bins["incline_length"] = int(bins[0] % 2)
                attempt.update(world=world, geometry_bins=geometry_bins)
        groups.append(dict(group_id=group, attempts=attempts))
    return dict(
        version="terrain_development_assignments_v1",
        namespace=namespace,
        status="PROSPECTIVE_ASSIGNMENTS_NOT_VALIDATED",
        prospective_only=True,
        native_validated=False,
        qualification_eligible=False,
        scope="Requested layouts and streams only; no realized starts, applied dynamics, generated meshes or native results",
        backend="numpy.random.PCG64",
        seed_mapping="SHA-256 UTF-8 namespace|group_id|attempt_index|stream_name; full digest as big-endian integer",
        seed_collisions=0,
        geometry_draw_order=[
            "shape_rank_key",
            "tread_rank_key",
            "tread_jitter",
            "yaw_rank_key",
            "yaw_jitter",
            "translation_x",
            "translation_y",
        ],
        placement="Common local dx,dy uniform [-0.25,0.25] m; relative feature centres fixed, finite map unchanged",
        control_hz=50,
        command_phases=[
            [2.0, [0.0, 0.0, 0.0]],
            [25.0, [0.35, 0.0, 0.0]],
            [3.0, [0.0, 0.0, 0.0]],
        ],
        groups=groups,
    )


def terrain_development_attempt(group_id, attempt_index):
    """Select one detached prospective development ID without realizing its world."""
    if (
        not isinstance(group_id, str)
        or group_id not in TERRAIN_GROUPS
        or type(attempt_index) is not int
        or not 0 <= attempt_index < 100
    ):
        raise ValueError(
            "Require a canonical terrain group and attempt index in [0, 99]"
        )
    plan = terrain_development_assignments(group_id)
    return deepcopy(
        {
            **{
                name: plan[name]
                for name in (
                    "namespace",
                    "backend",
                    "control_hz",
                    "command_phases",
                    "prospective_only",
                    "native_validated",
                    "qualification_eligible",
                )
            },
            "group_id": group_id,
            **plan["groups"][0]["attempts"][attempt_index],
        }
    )
