"""Training-only supported starts; no command guidance or reset-on-success."""

from copy import deepcopy
import hashlib
import json

import numpy as np

from . import operator_step_field as geometry

VERSION = "operator_step_support_v1"
PATCHES = 16
OFFSETS = np.array([(x, y, 0.0) for x in (-0.5, 0.0, 0.5) for y in (-0.5, 0.0, 0.5)])
SCOPE = "Training initial-state exposure only; imported USD/Warp support rays, NOT PhysX contact, foot support or climbing qualification"
COUNTERS = (
    "center_step_resets",
    "raised_step_resets",
    "nonstep_resets",
    "level1_reset_starts",
    "level2_reset_starts",
    "center_starts_seen_moving",
    "raised_starts_seen_moving",
    "moving_raised_center_ray_samples",
    "rising_center_ray_transitions",
    "descending_center_ray_transitions",
    "step_samples",
)
COUNTER_SCOPE = "Training only; seen moving requires >0.05m/s and >=0.1m displacement from that reset; center-ray quantized transitions exclude reset boundaries, NOT foot support or climbing success"


def recipe():
    return {
        "version": VERSION,
        "raised_probability": 0.5,
        "patches_per_tile": PATCHES,
        "patches_per_raised_level": PATCHES // 2,
        "same_level_neighborhood_cells": 3,
        "same_level_support_square_m": 1.5,
        "support_validation": "Analytic same-level 1.5m square; nine imported-ray samples in central1m square, not exhaustive contact validation",
        "raised_xy_jitter_m": 0.0,
        "raised_reset_orientation": "Native random Euler yaw; identity default quaternion required",
        "footprint_offsets_m": OFFSETS.tolist(),
        "scope": SCOPE,
    }


def configure(cfg):
    """Keep the native reset parameters; change only the reviewed dispatch."""
    term = cfg.events.reset_base
    if (
        term.func.__module__ != "isaaclab.envs.mdp.events"
        or term.func.__name__ != "reset_root_state_uniform"
    ):
        raise ValueError("Supported starts require the stock uniform reset")
    term.func = reset_root_state


def _tiles(report):
    result = []
    for tile in report["tiles"]:
        levels = geometry._levels(tile["seed"], tile["variant"])
        cells, selected_levels, candidate_counts = [], [], {}
        for level in (1, 2):
            candidates = np.array(
                [
                    (i, j)
                    for i in range(1, 31)
                    for j in range(1, 31)
                    if (levels[i - 1 : i + 2, j - 1 : j + 2] == level).all()
                ]
            )
            if len(candidates) < PATCHES // 2:
                raise ValueError("Insufficient same-level supported start patches")
            candidate_counts[str(level)] = len(candidates)
            cells.extend(
                candidates[
                    np.linspace(0, len(candidates) - 1, PATCHES // 2, dtype=int)
                ].tolist()
            )
            selected_levels.extend([level] * (PATCHES // 2))
        origin = np.array([16 * (tile["row"] - 1), 16 * (tile["variant"] - 9.5), 0.0])
        rise, rough = 0.04 + 0.08 * tile["difficulty"], 0.004 * tile["difficulty"]
        positions = (
            np.column_stack(
                (
                    np.asarray(cells) * 0.5 + 0.25 - 8,
                    np.asarray(selected_levels) * rise + rough,
                )
            )
            + origin
        )
        result.append(
            {
                "row": tile["row"],
                "variant": tile["variant"],
                "difficulty": tile["difficulty"],
                "cells": cells,
                "levels": selected_levels,
                "candidate_counts_by_level": candidate_counts,
                "positions_world_m": positions.tolist(),
            }
        )
    return result


def _recipe(report):
    return {
        **recipe(),
        "seed": report["seed"],
        "geometry_sha256": hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "tiles": _tiles(report),
    }


def validate_receipt(receipt, geometry_report):
    """Reproduce candidates and validate imported measurements, without Isaac Sim."""
    try:
        geometry.validate_geometry_report(geometry_report, seed=geometry_report["seed"])
        expected = _recipe(geometry_report)
        if {key: receipt[key] for key in expected} != expected:
            raise ValueError("Supported-start recipe differs from generated geometry")
        if set(receipt) != {*expected, "support_hits_world_m", "support_face_ids"}:
            raise ValueError("Unexpected supported-start receipt fields")
        positions = np.array([tile["positions_world_m"] for tile in expected["tiles"]])
        nominal = positions[:, :, None, :] + OFFSETS
        hits, faces = np.asarray(receipt["support_hits_world_m"]), np.asarray(
            receipt["support_face_ids"]
        )
        bounds = np.array([0.004 * tile["difficulty"] for tile in expected["tiles"]])[
            :, None, None
        ]
        if (
            hits.shape != nominal.shape
            or faces.shape != nominal.shape[:-1]
            or not np.isfinite(hits).all()
            or not np.issubdtype(faces.dtype, np.integer)
            or (faces < 0).any()
            or not np.allclose(hits[..., :2], nominal[..., :2], rtol=0, atol=2e-4)
            or (hits[..., 2] > nominal[..., 2] + 2e-4).any()
            or (hits[..., 2] < nominal[..., 2] - 2 * bounds - 2e-4).any()
        ):
            raise ValueError("Imported support rays missed the declared raised plateau")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed supported-start receipt") from error
    return deepcopy(receipt)


def install(env, geometry_report):
    """Validate imported support before exposing the native world-space patch table."""
    import torch
    from isaaclab.utils.warp import raycast_mesh

    if (
        hasattr(env, "_operator_step_support")
        or env.scene.terrain.flat_patches.get("init_pos") is not None
    ):
        raise ValueError("Do not overwrite existing supported-start state or init_pos")
    if env.cfg.events.reset_base.func is not reset_root_state:
        raise ValueError("Supported-start reset dispatcher was not configured")
    geometry.validate_geometry_report(
        geometry_report, seed=env.cfg.scene.terrain.terrain_generator.seed
    )
    defaults = env.scene["robot"].data.default_root_state
    if not torch.equal(
        defaults[:, :2], torch.zeros_like(defaults[:, :2])
    ) or not torch.equal(
        defaults[:, 3:7], defaults.new_tensor([1, 0, 0, 0]).expand(env.num_envs, -1)
    ):
        raise ValueError(
            "Native terrain reset requires zero default XY and identity orientation"
        )
    receipt = _recipe(geometry_report)
    positions = np.array([tile["positions_world_m"] for tile in receipt["tiles"]])
    starts = positions[:, :, None, :] + OFFSETS + [0, 0, 1]
    starts = torch.tensor(
        starts.reshape(-1, 9, 3), dtype=torch.float32, device=env.device
    )
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1
    sensor = env.scene["base_height_scanner"]
    paths = sensor.cfg.mesh_prim_paths
    if len(paths) != 1 or paths[0] not in sensor.meshes:
        raise ValueError("Supported starts require the initialized terrain RayCaster")
    hits, _, _, faces = raycast_mesh(
        starts, directions, sensor.meshes[paths[0]], max_dist=2.0, return_face_id=True
    )
    receipt.update(
        support_hits_world_m=hits.reshape(12, PATCHES, 9, 3).cpu().tolist(),
        support_face_ids=faces.reshape(12, PATCHES, 9).cpu().tolist(),
    )
    validate_receipt(receipt, geometry_report)
    origins = env.scene.terrain.terrain_origins
    if origins.shape != (3, 20, 3):
        raise ValueError("Supported starts require the static three-row layout")
    patches = origins[:, :, None, :].repeat(1, 1, PATCHES, 1).clone()
    for tile in receipt["tiles"]:
        expected_origin = origins.new_tensor(
            [16 * (tile["row"] - 1), 16 * (tile["variant"] - 9.5), 0]
        )
        if not torch.allclose(
            origins[tile["row"], tile["variant"]], expected_origin, rtol=0, atol=2e-4
        ):
            raise ValueError("Supported-start terrain origin changed")
        patches[tile["row"], tile["variant"]] = patches.new_tensor(
            tile["positions_world_m"]
        )
    state = SupportState(env, patches, receipt)
    env.scene.terrain.flat_patches["init_pos"] = patches
    env._operator_step_support = state
    return receipt


class SupportState:
    """Environment-local training counters; center-ray levels are not foot support."""

    def __init__(self, env, patches, receipt):
        import torch

        self.env, self.patches, self.active = env, patches, False
        terrain = env.scene.terrain
        self.columns, self.rows = (
            terrain.terrain_types.clone(),
            terrain.terrain_levels.clone(),
        )
        self.step = (self.columns >= 12) & (self.columns < 16)
        difficulty = patches.new_zeros((3, 20))
        for tile in receipt["tiles"]:
            difficulty[tile["row"], tile["variant"]] = tile["difficulty"]
        self.rise = 0.04 + 0.08 * difficulty[self.rows, self.columns]
        self.rough = 0.004 * difficulty[self.rows, self.columns]
        self.previous = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.valid = torch.zeros_like(self.step)
        self.start_kind = torch.full_like(self.previous, -1)
        self.start_xy = env.scene["robot"].data.root_pos_w[:, :2].clone()
        self.moved = torch.zeros_like(self.step)
        self.counts = torch.zeros(11, dtype=torch.long, device=env.device)

    def record_reset(self, ids, raised):
        import torch

        self.valid[ids] = False
        self.moved[ids] = False
        self.start_kind[ids] = torch.where(self.step[ids], raised.long(), -1)
        self.start_xy[ids] = self.env.scene["robot"].data.root_pos_w[ids, :2]
        self.counts[0] += (self.step[ids] & ~raised).sum()
        self.counts[1] += raised.sum()
        self.counts[2] += (~self.step[ids]).sum()
        if raised.any():
            selected = ids[raised]
            local_z = (
                self.env.scene["robot"].data.root_pos_w[selected, 2]
                - self.env.scene.env_origins[selected, 2]
            )
            local_z -= self.env.scene["robot"].data.default_root_state[selected, 2]
            level = (
                ((local_z - self.rough[selected]) / self.rise[selected]).round().long()
            )
            self.counts[3] += (level == 1).sum()
            self.counts[4] += (level == 2).sum()

    def sample(self):
        if not self.active:
            return
        import torch

        env = self.env
        data = env.scene["robot"].data
        height = (
            env.scene["base_height_scanner"].data.ray_hits_w[:, 0, 2]
            - env.scene.env_origins[:, 2]
        )
        level = (height / self.rise).round().long()
        local = data.root_pos_w - env.scene.env_origins
        valid = self.step & torch.isfinite(height) & (local[:, :2].abs() < 7).all(1)
        valid &= (
            (level >= 0)
            & (level <= 2)
            & ((height - level * self.rise).abs() <= self.rough + 2e-4)
        )
        moving = torch.linalg.vector_norm(data.root_lin_vel_b[:, :2], dim=1) > 0.05
        displacement = torch.linalg.vector_norm(
            data.root_pos_w[:, :2] - self.start_xy, dim=1
        )
        new_motion = moving & (displacement >= 0.1) & self.step & ~self.moved
        self.counts[5] += (new_motion & (self.start_kind == 0)).sum()
        self.counts[6] += (new_motion & (self.start_kind == 1)).sum()
        self.counts[7] += (valid & moving & (level > 0)).sum()
        comparable = valid & self.valid
        self.counts[8] += (comparable & (level > self.previous)).sum()
        self.counts[9] += (comparable & (level < self.previous)).sum()
        self.counts[10] += self.step.sum()
        self.moved |= new_motion
        self.previous, self.valid = level, valid

    def report(self):
        import torch

        terrain = self.env.scene.terrain
        if not torch.equal(self.rows, terrain.terrain_levels) or not torch.equal(
            self.columns, terrain.terrain_types
        ):
            raise ValueError("Supported-start terrain assignment changed")
        return {
            "version": VERSION,
            **dict(zip(COUNTERS, self.counts.cpu().tolist())),
            "scope": COUNTER_SCOPE,
        }


def validate_training_report(report, *, num_envs, steps):
    if (
        type(num_envs) is not int
        or num_envs <= 0
        or num_envs % 20
        or type(steps) is not int
        or steps < 1
        or type(report) is not dict
        or set(report) != {"version", "scope", *COUNTERS}
        or report["version"] != VERSION
        or report["scope"] != COUNTER_SCOPE
        or any(type(report[key]) is not int or report[key] < 0 for key in COUNTERS)
        or report["step_samples"] != steps * (num_envs // 5)
        or report["center_step_resets"] + report["raised_step_resets"] < num_envs // 5
        or report["nonstep_resets"] < 4 * num_envs // 5
        or (
            num_envs >= 320
            and min(report["center_step_resets"], report["raised_step_resets"]) == 0
        )
        or report["level1_reset_starts"] + report["level2_reset_starts"]
        != report["raised_step_resets"]
        or report["center_starts_seen_moving"] > report["center_step_resets"]
        or report["raised_starts_seen_moving"] > report["raised_step_resets"]
        or any(report[key] > report["step_samples"] for key in COUNTERS[7:10])
    ):
        raise ValueError("Invalid training supported-start counts")
    return deepcopy(report)


def reset_root_state(env, env_ids, pose_range, velocity_range, asset_cfg=None):
    """Use stock events; inactive/evaluation and nonstep rows retain center resets."""
    import torch
    from isaaclab.envs.mdp.events import (
        reset_root_state_uniform,
        reset_root_state_from_terrain,
    )
    from isaaclab.managers import SceneEntityCfg

    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    )
    if not len(ids):
        return
    state = getattr(env, "_operator_step_support", None)
    if state is None or not state.active:
        reset_root_state_uniform(env, ids, pose_range, velocity_range, asset_cfg)
        if state is not None:
            state.valid[ids] = False
            state.start_kind[ids] = -1
            state.moved[ids] = False
        return
    if env.scene.terrain.flat_patches.get("init_pos") is not state.patches:
        raise ValueError("Installed supported-start table was replaced")
    raised = torch.zeros(len(ids), dtype=torch.bool, device=env.device)
    selected = state.step[ids]
    raised[selected] = torch.rand(int(selected.sum()), device=env.device) < 0.5
    if (~raised).any():
        reset_root_state_uniform(
            env, ids[~raised], pose_range, velocity_range, asset_cfg
        )
    if raised.any():
        reset_root_state_from_terrain(
            env, ids[raised], pose_range, velocity_range, asset_cfg
        )
    state.record_reset(ids, raised)
