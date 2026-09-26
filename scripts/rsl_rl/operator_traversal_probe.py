"""Post-step first-attempt spatial diagnostics, never a supported-traversal gate.

The controller cannot read this observer. Native auto-reset terminal states and
later episodes are excluded; their failure counts remain in the main evaluator.
Foot-link positions and measured upward contact forces are recorded explicitly,
not converted into unverified contact patches or support-polygon claims.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import hashlib

import numpy as np

from .operator_traversal_terrain import geometry, profiles_for_layout

COMMAND_TAPE = (
    {"name": "initial_stop", "steps": 100, "command": (0.0, 0.0, 0.0)},
    {"name": "forward", "steps": 700, "command": (0.4, 0.0, 0.0)},
    {"name": "final_stop", "steps": 100, "command": (0.0, 0.0, 0.0)},
)
FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
CORRIDOR_HALF_WIDTH = 2.0
CONTACT_FORCE_Z = 1.0


def motor_diagnostic_protocol():
    return {
        "version": "operator_traversal_motor_diagnostic_v1",
        "sampling": "State sample t follows input-trace action t-1; sample0 and ended/reset rows are excluded by motor_response_valid",
        "torques": "Computed/applied actuator commands from BEFORE the final physics substep of the preceding control interval; not measured torques, interval extrema or PD recomputable from post-step q/qd",
        "targets": "Native joint_pos_target for the verified nondelayed DCMotor; absolute radians, not clipped or altered by this observer",
        "limits": "Hard positions and velocity/effort limits are PhysX bounds; soft positions are reward bounds. Actuator caps and torque-speed configuration are in native_motor_binding, not inferred from PhysX effort limits",
        "feet": "World LINK-origin velocities; air time uses native net-normal-force norm and the recorded sensor threshold, not the traversal upward-force test",
        "scope": "Observer-only mechanical diagnostics; no policy inputs, control changes or qualification",
    }


def _snapshot(value, shape, name):
    result = value.detach().cpu().numpy().copy()
    if (
        result.shape != shape
        or result.dtype.kind != "f"
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"Invalid native motor diagnostic field: {name}")
    return result


class TraversalProbe:
    def __init__(self, env, output, *, layout="standard", motor_binding=None):
        profiles = profiles_for_layout(layout)
        self.env, self.output = env, Path(output)
        self.robot = env.scene["robot"]
        self.contacts = env.scene["contact_forces"]
        self.foot_ids = [self.robot.body_names.index(name) for name in FEET]
        self.contact_ids = [self.contacts.body_names.index(name) for name in FEET]
        generator = getattr(
            getattr(env.scene.terrain, "cfg", None), "terrain_generator", None
        )
        terrains = getattr(generator, "sub_terrains", None)
        if (
            getattr(generator, "curriculum", None) is not True
            or getattr(generator, "num_cols", None) != 4
            or not isinstance(terrains, dict)
            or tuple(
                (getattr(term, "profile", None), getattr(term, "proportion", None))
                for term in terrains.values()
            )
            != tuple((profile, 0.25) for profile in profiles)
        ):
            raise ValueError("Traversal observer layout differs from native columns")
        columns = env.scene.terrain.terrain_types.detach().cpu().numpy()
        if (
            columns.shape != (env.num_envs,)
            or columns.dtype.kind not in "iu"
            or set(columns.tolist()) != set(range(4))
        ):
            raise ValueError("Traversal requires all four native terrain columns")
        self.specs = [geometry(profiles[column]) for column in columns]
        self.origins = env.scene.env_origins.detach().cpu().numpy().copy()
        if (
            self.origins.shape != (env.num_envs, 3)
            or not np.isfinite(self.origins).all()
        ):
            raise ValueError("Invalid traversal origins")
        self.records = []
        self.motor_records = []
        self.motor_binding = deepcopy(motor_binding)
        self.motor_limits = {}
        if motor_binding is not None:
            if (
                list(self.robot.joint_names) != motor_binding["joint_names"]
                or len(set(self.robot.joint_names)) != 12
                or not self.contacts.cfg.track_air_time
                or not np.isfinite(self.contacts.cfg.force_threshold)
                or self.contacts.cfg.force_threshold <= 0
                or any(
                    item["configuration"]["class_type"]
                    != "isaaclab.actuators.actuator_pd:DCMotor"
                    for item in motor_binding["actuators"].values()
                )
            ):
                raise ValueError(
                    "Motor capture requires the verified named DCMotor and air timers"
                )
            for name in (
                "joint_pos_limits",
                "soft_joint_pos_limits",
                "joint_vel_limits",
                "joint_effort_limits",
            ):
                shape = (env.num_envs, 12, 2) if "pos" in name else (env.num_envs, 12)
                value = _snapshot(getattr(self.robot.data, name), shape, name)
                if (
                    (value[..., 0] >= value[..., 1]).any()
                    if "pos" in name
                    else (value <= 0).any()
                ):
                    raise ValueError(f"Invalid native motor limits: {name}")
                self.motor_limits[name] = value

    def observe(self, step, alive):
        """Capture initial state or a nonterminal post-step state, on native clock."""
        data = self.robot.data
        root = data.root_pos_w.detach().cpu().numpy() - self.origins
        feet = (
            data.body_pos_w[:, self.foot_ids].detach().cpu().numpy()
            - self.origins[:, None]
        )
        forces = (
            self.contacts.data.net_forces_w[:, self.contact_ids].detach().cpu().numpy()
        )
        valid = alive.detach().cpu().numpy().copy()
        if not all(np.isfinite(value).all() for value in (root, feet, forces)):
            raise ValueError("Nonfinite native traversal telemetry")
        if (
            root.shape != (self.env.num_envs, 3)
            or feet.shape != forces.shape
            or feet.shape != (self.env.num_envs, 4, 3)
            or valid.shape != (self.env.num_envs,)
            or valid.dtype != np.bool_
        ):
            raise ValueError("Invalid native traversal telemetry shape")
        if step != len(self.records):
            raise ValueError("Traversal requires contiguous completed native steps")
        if self.records and (valid & ~self.records[-1][3]).any():
            raise ValueError("Traversal cannot re-enter a finished first attempt")
        if step == 0:
            quat = data.root_quat_w.detach().cpu().numpy()
            if not (
                valid.all()
                and np.allclose(root[:, :2], 0.0, atol=1e-6, rtol=0)
                and quat.shape == (self.env.num_envs, 4)
                and np.allclose(np.abs(quat[:, 0]), 1.0, atol=1e-6, rtol=0)
                and np.allclose(quat[:, 1:], 0.0, atol=1e-6, rtol=0)
            ):
                raise ValueError(
                    "Traversal requires the fixed native approach origin and heading"
                )
        self.records.append((root.copy(), feet.copy(), forces.copy(), valid))
        if self.motor_binding is not None:
            self._observe_motor(step, valid)

    def _observe_motor(self, step, valid):
        data = self.robot.data
        fields = {
            "joint_pos_rad": data.joint_pos,
            "joint_vel_rad_s": data.joint_vel,
            "joint_target_rad": data.joint_pos_target,
            "computed_torque_nm": data.computed_torque,
            "applied_torque_nm": data.applied_torque,
            "foot_link_velocity_world_m_s": data.body_link_lin_vel_w[:, self.foot_ids],
            "foot_air_time_s": self.contacts.data.current_air_time[:, self.contact_ids],
        }
        response_valid = valid & (step > 0)
        sample = {"motor_response_valid": response_valid.copy()}
        for name, value in fields.items():
            width = (
                (4, 3)
                if name == "foot_link_velocity_world_m_s"
                else ((4,) if name == "foot_air_time_s" else (12,))
            )
            copied = _snapshot(value, (self.env.num_envs, *width), name)
            if name == "foot_air_time_s" and (copied < 0).any():
                raise ValueError("Native foot air time cannot be negative")
            # Never expose reset buffers or the initial stale effort as response
            # evidence. Zero is a placeholder; the mask is authoritative.
            copied[~response_valid] = 0
            sample[name] = copied
        self.motor_records.append(sample)

    def input_diagnostic_samples(self, steps):
        """Align observer states with pre-action decisions, excluding the final state."""
        if type(steps) is not int or steps < 1 or len(self.records) != steps + 1:
            raise ValueError("Input telemetry requires every initial/post-step sample")
        return {
            "root_local_m": np.stack([record[0] for record in self.records[:steps]]),
            "first_attempt_valid": np.stack(
                [record[3] for record in self.records[:steps]]
            ),
            "entry_x_m": np.array([spec["entry_x_m"] for spec in self.specs]),
            "corridor_half_width_m": CORRIDOR_HALF_WIDTH,
        }

    def report(self):
        if len(self.records) != 901:
            raise ValueError("Traversal report requires the complete 900-step tape")
        root, feet, forces, valid = map(np.stack, zip(*self.records, strict=True))
        corridor = (
            (np.abs(root[:, :, 1]) <= CORRIDOR_HALF_WIDTH)
            & (np.abs(feet[:, :, :, 1]).max(axis=2) <= CORRIDOR_HALF_WIDTH)
            & (np.abs(root[:, :, 0]) <= 7.5)
        )
        # Once a route violation occurs, never credit a later return to the lane.
        eligible = valid & np.logical_and.accumulate(corridor, axis=0)
        rows = []
        for row, spec in enumerate(self.specs):
            good = eligible[:, row]
            first = []
            start = 0
            for key in ("entry_x_m", "top_start_x_m", "exit_x_m"):
                found = np.flatnonzero(
                    good & (np.arange(901) >= start) & (root[:, row, 0] >= spec[key])
                )
                reached = int(found[0]) if len(found) else None
                first.append(reached)
                start = reached if reached is not None else 901
            contact = forces[:, row, :, 2] > CONTACT_FORCE_Z
            top = (feet[:, row, :, 0] >= spec["top_start_x_m"]) & (
                feet[:, row, :, 0] <= spec["top_end_x_m"]
            )
            beyond = feet[:, row, :, 0] > spec["exit_x_m"]
            crossed = np.flatnonzero(good & beyond.all(axis=1))
            observed = np.flatnonzero(valid[:, row])
            stop = np.flatnonzero(valid[:, row] & (np.arange(901) >= 800))
            stop_displacement = (
                np.linalg.norm(root[stop, row, :2] - root[stop[0], row, :2], axis=1)
                if len(stop) >= 2
                else None
            )
            final = int(observed[-1])
            rows.append(
                {
                    "environment_index": row,
                    "profile": spec["profile"],
                    "valid_spatial_samples": len(observed),
                    "ordered_root_entry_top_exit_steps": first,
                    "all_foot_links_beyond_exit_step": (
                        int(crossed[0]) if len(crossed) else None
                    ),
                    "first_route_violation_step": next(
                        (int(i) for i in observed if not corridor[i, row]), None
                    ),
                    "maximum_root_x_m": float(root[valid[:, row], row, 0].max()),
                    "last_observed_step": final,
                    "last_root_position_local_m": root[final, row].tolist(),
                    "last_foot_positions_local_m": feet[final, row].tolist(),
                    "last_contact_force_world_n": forces[final, row].tolist(),
                    "top_x_interval_upward_force_samples_by_foot": (
                        top & contact & good[:, None]
                    )
                    .sum(axis=0)
                    .tolist(),
                    "beyond_exit_x_upward_force_samples_by_foot": (
                        beyond & contact & good[:, None]
                    )
                    .sum(axis=0)
                    .tolist(),
                    "final_stop_spatial_samples": len(stop),
                    "final_stop_observed_duration_s": max(len(stop) - 1, 0) * 0.02,
                    "final_stop_complete_window": len(stop) == 101,
                    "final_stop_net_xy_displacement_m": (
                        float(stop_displacement[-1])
                        if stop_displacement is not None
                        else None
                    ),
                    "final_stop_max_xy_displacement_from_start_m": (
                        float(stop_displacement.max())
                        if stop_displacement is not None
                        else None
                    ),
                    "survived_full_tape": bool(valid[-1, row]),
                }
            )
        self.output.parent.mkdir(parents=True, exist_ok=True)
        motor_arrays = {}
        if self.motor_binding is not None:
            if len(self.motor_records) != len(self.records):
                raise ValueError("Incomplete motor diagnostic samples")
            motor_arrays = {
                name: np.stack([sample[name] for sample in self.motor_records])
                for name in self.motor_records[0]
            }
            motor_arrays.update(self.motor_limits)
        np.savez_compressed(
            self.output,
            root_local_m=root,
            feet_local_m=feet,
            foot_force_world_n=forces,
            first_attempt_valid=valid,
            route_eligible=eligible,
            **motor_arrays,
        )
        return {
            "version": "operator_traversal_diagnostic_v1",
            "scope": "Ordered ROOT crossings and named FOOT-LINK/contact observations only; NOT supported completion or qualification",
            "terminal_sampling": "Post-step auto-reset states on ended rows and every later episode excluded; main report retains physical failures/timeouts",
            "sample_clock": "index times 0.02 seconds; sample 0 after initial reset, sample 900 after final step",
            "foot_order": list(FEET),
            "position_semantics": "Native root and foot LINK origins, relative to the fixed environment origin; not COM or contact points",
            "contact_observation": f"Foot-link X inside the named interval AND net world-Z force > {CONTACT_FORCE_Z} N; no surface-height/ray test, not verified top contact, contact patch or support gate",
            "route_corridor_half_width_m": CORRIDOR_HALF_WIDTH,
            "layouts_per_profile": 1,
            "trace": {
                "path": str(self.output),
                "sha256": hashlib.sha256(self.output.read_bytes()).hexdigest(),
            },
            "rows": rows,
            **(
                {
                    "motor_diagnostics": {
                        **motor_diagnostic_protocol(),
                        "native_motor_binding": self.motor_binding,
                        "contact_force_threshold_n": self.contacts.cfg.force_threshold,
                        "valid_response_samples": int(
                            motor_arrays["motor_response_valid"].sum()
                        ),
                        "trace": "Additional named arrays in traversal trace; static limits have no time axis",
                    }
                }
                if self.motor_binding is not None
                else {}
            ),
            "exit_allowed": False,
        }
