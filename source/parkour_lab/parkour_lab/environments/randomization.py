"""Independent recorded random streams for each robot, run seed and task role."""

import hashlib
import json
from copy import deepcopy

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


def _stream(namespace, group_id, index, name):
    text = f"{namespace}|{group_id}|{index}|{name}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    generator = np.random.Generator(np.random.PCG64(int(digest, 16)))
    mapping = dict(
        sha256=digest, initial_state=generator.bit_generator.state["state"].copy()
    )
    return generator, mapping


def _valid_pcg64_state(state):
    if (
        type(state) is not dict
        or set(state) != {"bit_generator", "state", "has_uint32", "uinteger"}
        or state["bit_generator"] != "PCG64"
        or type(state["state"]) is not dict
        or set(state["state"]) != {"state", "inc"}
    ):
        return False
    return (
        all(
            type(value) is int and 0 <= value < 2**128
            for value in state["state"].values()
        )
        and state["state"]["inc"] % 2 == 1
        and type(state["has_uint32"]) is int
        and state["has_uint32"] in (0, 1)
        and type(state["uinteger"]) is int
        and 0 <= state["uinteger"] < 2**32
    )


class TaskRandomization:
    """Full SHA-256 entropy seeds a PCG64 generator per row and named stream.

    Subset draws advance only the requested rows, independently of batch ordering.
    Physical parameters stay fixed; reset starts and sensor samples advance their
    own streams. No NumPy, torch, learner, or simulator global RNG is consumed.
    """

    def __init__(self, task, *, evaluation=False):
        def setting(name, default=None):
            return (
                task.get(name, default)
                if isinstance(task, dict)
                else getattr(task, name, default)
            )

        self.count = setting("num_envs")
        seed = setting("seed")
        dynamics = setting("dynamics", "randomized")
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("Require a positive environment count")
        self._streams = STREAMS
        self._attempt_indices = list(range(self.count))
        if (
            type(seed) is not int
            or seed < 0
            or dynamics not in ("nominal", "randomized")
        ):
            raise ValueError("Require a nonnegative run seed and declared dynamics")
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

    def _state_identity(self):
        return dict(
            namespace=self.namespace,
            group_id=self.group_id,
            count=self.count,
            streams=list(self._streams),
            attempt_indices=self._attempt_indices.copy(),
            randomized=self.randomized.tolist(),
            heading=self._heading.tolist(),
            mapping_sha256=hashlib.sha256(
                json.dumps(self._mappings, sort_keys=True).encode()
            ).hexdigest(),
        )

    def state_dict(self):
        """Detached current streams for continuation, not an in-flight episode."""
        return dict(
            version="go2_task_randomization_state_v1",
            identity=self._state_identity(),
            generators=[
                {
                    stream: deepcopy(self._generators[row, stream].bit_generator.state)
                    for stream in self._streams
                }
                for row in range(self.count)
            ],
            physical=(
                None
                if self._physical is None
                else {name: value.tolist() for name, value in self._physical.items()}
            ),
        )

    def load_state_dict(self, state):
        """Validate everything before replacing streams; next reset starts new episodes."""
        if (
            type(state) is not dict
            or set(state) != {"version", "identity", "generators", "physical"}
            or state["version"] != "go2_task_randomization_state_v1"
            or state["identity"] != self._state_identity()
            or type(state["generators"]) is not list
            or len(state["generators"]) != self.count
        ):
            raise ValueError("Sampler continuation identity or schema differs")
        generators = {}
        for row, streams in enumerate(state["generators"]):
            if type(streams) is not dict or set(streams) != set(self._streams):
                raise ValueError("Sampler continuation stream names differ")
            for name, saved in streams.items():
                if not _valid_pcg64_state(saved):
                    raise ValueError("Invalid PCG64 continuation state")
                generator = np.random.Generator(np.random.PCG64(0))
                generator.bit_generator.state = deepcopy(saved)
                generators[row, name] = generator
        physical = None
        if state["physical"] is not None:
            try:
                physical = {
                    name: np.asarray(value, dtype=np.float64)
                    for name, value in state["physical"].items()
                }
                width = physical["motor_strength"].shape[1]
                canonical = deepcopy(self)
                canonical._physical = None
                for row, mapping in enumerate(self._mappings):
                    canonical._generators[row, "dynamics"] = np.random.Generator(
                        np.random.PCG64(int(mapping["dynamics"]["sha256"], 16))
                    )
                expected = canonical.dynamics(width)
                if set(physical) != set(expected) or any(
                    not np.array_equal(value, expected[name])
                    for name, value in physical.items()
                ):
                    raise ValueError("Physical draws differ from the seeded startup")
            except (AttributeError, KeyError, IndexError, TypeError) as error:
                raise ValueError(
                    "Invalid physical sampler continuation state"
                ) from error
        self._generators, self._physical = generators, physical
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
