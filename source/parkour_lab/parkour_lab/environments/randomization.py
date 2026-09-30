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
        profile, seed = setting("bank_profile"), setting("seed")
        dynamics = setting("dynamics", "randomized")
        if type(self.count) is not int or self.count <= 0:
            raise ValueError("Require a positive environment count")
        if (
            type(seed) is not int
            or seed < 0
            or dynamics not in ("nominal", "randomized")
        ):
            raise ValueError("Require a nonnegative run seed and declared dynamics")
        if profile is not None:
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
        for row in range(self.count):
            mapping = {}
            for stream in STREAMS:
                text = f"{self.namespace}|{self.group_id}|{row}|{stream}"
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                generator = np.random.Generator(np.random.PCG64(int(digest, 16)))
                state = generator.bit_generator.state["state"]
                backend = (state["state"], state["inc"])
                if digest in entropy_seen or backend in backend_seen:
                    raise ValueError("Random-stream seed collision")
                entropy_seen.add(digest)
                backend_seen.add(backend)
                mapping[stream] = dict(sha256=digest, initial_state=state.copy())
                self._generators[row, stream] = generator
            self._mappings.append(mapping)
        self._starts = [None] * self.count
        self._physical = None
        self._heading = np.zeros(self.count)
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
        if stream not in STREAMS or type(width) is not int or width <= 0:
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
                    attempt_index=row,
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
