"""Post-step first-attempt spatial diagnostics, never a supported-traversal gate.

The controller cannot read this observer. Native auto-reset terminal states and
later episodes are excluded; their failure counts remain in the main evaluator.
Foot-link positions and measured upward contact forces are recorded explicitly,
not converted into unverified contact patches or support-polygon claims.
"""

from __future__ import annotations

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


class TraversalProbe:
    def __init__(self, env, output, *, layout="standard"):
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
        np.savez_compressed(
            self.output,
            root_local_m=root,
            feet_local_m=feet,
            foot_force_world_n=forces,
            first_attempt_valid=valid,
            route_eligible=eligible,
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
            "exit_allowed": False,
        }
