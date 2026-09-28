"""Training-only riser approaches; no steering, progress reward or success reset."""

from copy import deepcopy
import math

import numpy as np

from . import operator_step_field as geometry

VERSION = "operator_step_approach_v1"
HOLD_STEPS = 250
FOOTPRINT_RADIUS = 0.5
COUNTERS = (
    "at_risk_decisions",
    "approach_decisions",
    "riser_contact_proxy_decisions",
    "commanded_decisions",
    "stalled_decisions",
    "root_crossing_episodes",
    "foot_link_crossing_episodes",
    "ground_transition_episodes",
    "corridor_departures",
    "physical_ends",
    "timeouts",
    "forward_hold_decisions",
)


def recipe():
    return {
        "version": VERSION,
        "geometry_version": geometry.VERSION,
        "approach_probability_on_step_rows": 0.75,
        "directions": "Uniform ascent/descent of each certified seam",
        "start_distance_m": 0.75,
        "seam_frame_xy_jitter_m": 0.1,
        "yaw_jitter_rad": math.pi / 18,
        "same_level_patch_size_m": 1.5,
        "spawn_footprint_radius_m": FOOTPRINT_RADIUS,
        "link_center_padding_m": 0.05,
        "forward_speed_range_m_s": [0.3, 0.5],
        "initial_command_decisions": HOLD_STEPS,
        "commands": "Episode-owned body vx, vy=wz=0 for250 delivered decisions, then ordinary sampler; no heading feedback",
        "background": "25% step resets and all nonstep resets retain ordinary center starts and free commands",
        "scope": "Fixed training occupancy experiment; analytic patch coverage plus imported rays and nominal link-footprint check, NOT a PhysX collision certificate or supported traversal qualification",
    }


def candidates(report):
    """All straight seams with a complete3x3-cell patch on EACH side.

    Each undirected seam appears once, then in both travel directions. Root
    jitter plus the0.5m footprint stays inside its1.5m square with0.15m margin.
    No assumption is made about support after leaving these two rectangles.
    """
    result = []
    for tile in report["tiles"]:
        levels = geometry._levels(tile["seed"], tile["variant"])
        origin = np.array([16 * (tile["row"] - 1), 16 * (tile["variant"] - 9.5), 0.0])
        count = len(result)
        for axis in (0, 1):
            grid = levels if axis == 0 else levels.T
            for i in range(2, 29):
                for j in range(1, 31):
                    a, b = int(grid[i, j]), int(grid[i + 1, j])
                    if abs(a - b) != 1 or not (
                        (grid[i - 2 : i + 1, j - 1 : j + 2] == a).all()
                        and (grid[i + 1 : i + 4, j - 1 : j + 2] == b).all()
                    ):
                        continue
                    edge = np.array([(i + 1) * 0.5 - 8, (j + 0.5) * 0.5 - 8])
                    if axis == 1:
                        edge = edge[::-1]
                    for sign in (1, -1):
                        normal = np.eye(2)[axis] * sign
                        start, target = (a, b) if sign == 1 else (b, a)
                        result.append(
                            dict(
                                row=tile["row"],
                                variant=tile["variant"],
                                edge_world_xy_m=(edge + origin[:2]).tolist(),
                                normal=normal.tolist(),
                                start_level=start,
                                target_level=target,
                                start_height_m=start * tile["riser_height_m"],
                                target_height_m=target * tile["riser_height_m"],
                                roughness_m=tile["roughness_absolute_bound_m"],
                            )
                        )
        if len(result) == count:
            raise ValueError(
                f"No certified approach seam in tile {tile['row'], tile['variant']}"
            )
    return result


def support_probes(table):
    """Probe both complete analytic patches, not just the root's center ray."""
    result = []
    for item in table:
        normal = np.asarray(item["normal"])
        tangent = normal[::-1] * [-1, 1]
        for side, height in (
            (-1, item["start_height_m"]),
            (1, item["target_height_m"]),
        ):
            for x in (-0.6, 0.0, 0.6):
                for y in (-0.6, 0.0, 0.6):
                    xy = (
                        item["edge_world_xy_m"]
                        + (side * 0.75 + x) * normal
                        + y * tangent
                    )
                    result.append([*xy, height])
    return np.asarray(result).reshape(len(table), 18, 3)


def validate_receipt(receipt, report):
    table = candidates(report)
    if receipt["recipe"] != recipe() or receipt["candidates"] != table:
        raise ValueError("Approach recipe or certified seam table changed")
    expected = support_probes(table)
    hits, faces = np.asarray(receipt["hits_world_m"]), np.asarray(receipt["face_ids"])
    bounds = np.array([item["roughness_m"] for item in table])[:, None]
    if (
        hits.shape != expected.shape
        or faces.shape != expected.shape[:-1]
        or not np.isfinite(hits).all()
        or not np.issubdtype(faces.dtype, np.integer)
        or (faces < 0).any()
        or not np.allclose(hits[..., :2], expected[..., :2], rtol=0, atol=2e-4)
        or (np.abs(hits[..., 2] - expected[..., 2]) > bounds + 2e-4).any()
    ):
        raise ValueError(
            "Imported approach/landing rays disagree with certified patches"
        )
    return deepcopy(receipt)


def configure(cfg):
    from .operator_command import (
        ProceduralStepApproachCommand,
        ProceduralTerrainCommand,
    )

    term = cfg.events.reset_base
    if (
        term.func.__name__ != "reset_root_state_uniform"
        or term.func.__module__ != "isaaclab.envs.mdp.events"
    ):
        raise ValueError("Approaches require the stock reset")
    if cfg.commands.base_velocity.class_type is not ProceduralTerrainCommand:
        raise ValueError("Approaches require the ordinary free-operator sampler")
    # The footprint check is valid only for the unchanged nominal reset posture.
    if cfg.events.reset_robot_joints.params != {
        "position_range": (1.0, 1.0),
        "velocity_range": (0.0, 0.0),
    }:
        raise ValueError("Approaches require nominal joints and zero reset velocity")
    if any(
        tuple(value) != (0.0, 0.0) for value in term.params["velocity_range"].values()
    ):
        raise ValueError("Approaches must not inject launch velocity")
    term.func = reset_root_state
    cfg.commands.base_velocity.class_type = ProceduralStepApproachCommand


def install(env, report):
    import torch
    from isaaclab.utils.warp import raycast_mesh

    if hasattr(env, "_operator_step_approach") or hasattr(
        env, "_operator_step_support"
    ):
        raise ValueError("Do not combine or overwrite reset mechanisms")
    if (
        report["version"] != geometry.VERSION
        or env.cfg.events.reset_base.func is not reset_root_state
    ):
        raise ValueError(
            "Approaches require their configured higher-field geometry/reset"
        )
    origins = env.scene.terrain.terrain_origins
    expected = origins.new_tensor(
        [
            [[16 * (row - 1), 16 * (col - 9.5), 0] for col in range(20)]
            for row in range(3)
        ]
    )
    if origins.shape != expected.shape or not torch.allclose(
        origins, expected, rtol=0, atol=2e-4
    ):
        raise ValueError(
            "Approach world coordinates require the static3x20 terrain layout"
        )
    table = candidates(report)
    probes = torch.tensor(support_probes(table), dtype=torch.float32, device=env.device)
    starts = probes + probes.new_tensor([0, 0, 1])
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1
    sensor = env.scene["base_height_scanner"]
    paths = sensor.cfg.mesh_prim_paths
    if len(paths) != 1 or paths[0] not in sensor.meshes:
        raise ValueError("Approaches require the imported terrain RayCaster")
    hits, _, _, faces = raycast_mesh(
        starts, directions, sensor.meshes[paths[0]], max_dist=2.0, return_face_id=True
    )
    receipt = dict(
        recipe=recipe(),
        candidates=table,
        hits_world_m=hits.cpu().tolist(),
        face_ids=faces.cpu().tolist(),
    )
    validate_receipt(receipt, report)
    env._operator_step_approach = ApproachState(env, table)
    return receipt


class ApproachState:
    """Episode-local exposure, with permanent corridor censoring and reset masks."""

    def __init__(self, env, table):
        import torch
        from .operator_roa_contacts import FEET

        self.env, self.table, self.active = env, table, False
        self.footprint = None
        self.rows = env.scene.terrain.terrain_levels.clone()
        self.columns = env.scene.terrain.terrain_types.clone()
        self.step = (self.columns >= 12) & (self.columns < 16)
        self.edge = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.episode = torch.zeros_like(self.edge)
        self.live = torch.zeros_like(self.step)
        self.seen = torch.zeros((env.num_envs, 3), dtype=torch.bool, device=env.device)
        self.resets = torch.zeros((3, 2), dtype=torch.long, device=env.device)
        self.holds = torch.zeros_like(self.resets)
        self.counts = torch.zeros(
            (2, 3, 2, len(COUNTERS)), dtype=torch.long, device=env.device
        )
        self.samples = 0
        self.pending = self.last = None
        self.feet = [env.scene["robot"].body_names.index(foot) for foot in FEET]
        self.force_ids = [
            env.scene["contact_forces"].body_names.index(foot) for foot in FEET
        ]
        self.values = {
            key: torch.tensor([item[key] for item in table], device=env.device)
            for key in (
                "edge_world_xy_m",
                "normal",
                "start_height_m",
                "target_height_m",
                "roughness_m",
            )
        }
        self.pools = {
            (row, col): torch.tensor(
                [
                    i
                    for i, item in enumerate(table)
                    if item["row"] == row and item["variant"] == col
                ],
                device=env.device,
            )
            for row in range(3)
            for col in range(12, 16)
        }

    def verify_nominal_footprint(self):
        """Called after an explicit native reset/forward, before any approach use."""
        import torch

        if self.footprint is not None:
            return
        data = self.env.scene["robot"].data
        radius = torch.linalg.vector_norm(
            data.body_link_pos_w[..., :2] - data.root_link_pos_w[:, None, :2], dim=-1
        )
        defaults = data.default_root_state
        if (
            not torch.isfinite(radius).all()
            or float(radius.max()) + 0.05 > FOOTPRINT_RADIUS
            or not torch.allclose(
                data.joint_pos, data.default_joint_pos, atol=1e-5, rtol=0
            )
            or torch.count_nonzero(defaults[:, 7:])
            or torch.count_nonzero(data.default_joint_vel)
            or torch.count_nonzero(data.joint_vel)
            or not torch.equal(defaults[:, :2], torch.zeros_like(defaults[:, :2]))
            or not torch.equal(
                defaults[:, 3:7],
                defaults.new_tensor([1, 0, 0, 0]).expand(self.env.num_envs, -1),
            )
        ):
            raise ValueError(
                "Native nominal posture exceeds the certified approach footprint"
            )
        self.footprint = dict(
            maximum_link_center_radius_m=float(radius.max()),
            padding_m=0.05,
            bound_m=FOOTPRINT_RADIUS,
        )

    def before_step(self, stage):
        import torch

        if not self.active:
            return
        if self.pending is not None or stage not in (
            "privileged_ppo",
            "history_adaptation",
        ):
            raise ValueError(
                "Approach telemetry requires consecutive owned training steps"
            )
        phase = int(stage == "history_adaptation")
        env, data = self.env, self.env.scene["robot"].data
        chosen = self.edge >= 0
        values = {
            key: value[self.edge.clamp_min(0)] for key, value in self.values.items()
        }
        normal = values["normal"]
        tangent = normal.flip(-1) * normal.new_tensor([-1, 1])
        relative = data.root_pos_w[:, :2] - values["edge_world_xy_m"]
        along, lateral = (relative * normal).sum(-1), (relative * tangent).sum(-1)
        foot = data.body_link_pos_w[:, self.feet]
        foot_relative = foot[..., :2] - values["edge_world_xy_m"][:, None]
        foot_along = (foot_relative * normal[:, None]).sum(-1)
        foot_lateral = (foot_relative * tangent[:, None]).sum(-1)
        fresh = (
            env.episode_length_buf > 0
        )  # Reset force/kinematic buffers may be stale.
        inside = (
            (along.abs() < 1.5)
            & (lateral.abs() < 0.75)
            & (foot_lateral.abs() < 0.75).all(-1)
        )
        departed = self.live & fresh & ~inside
        self.live &= ~departed
        valid = self.live & fresh
        forces = (
            env.scene["contact_forces"].data.net_forces_w[:, self.force_ids].clone()
        )
        forces[~fresh] = 0
        command = env.command_manager.get_command("base_velocity")
        moving_command = torch.linalg.vector_norm(command[:, :2], dim=-1) > 0.05
        slow = torch.linalg.vector_norm(data.root_lin_vel_b[:, :2], dim=-1) < 0.05
        contact = (
            (foot_along.abs() < 0.12)
            & ((forces[..., :2] * normal[:, None]).sum(-1).abs() > 1)
        ).any(-1)
        heights = (
            env.scene["base_height_scanner"].data.ray_hits_w[:, 0, 2]
            - env.scene.env_origins[:, 2]
        )
        events = torch.stack(
            (
                along >= 0,
                (foot_along > 0).all(-1) & (along >= 0),
                (along >= 0)
                & torch.isfinite(heights)
                & (
                    (heights - values["target_height_m"]).abs()
                    <= values["roughness_m"] + 2e-4
                ),
            ),
            -1,
        )
        new = events & valid[:, None] & ~self.seen
        self.seen |= events & valid[:, None]
        direction = (values["target_height_m"] < values["start_height_m"]).long()
        group = self.rows * 2 + direction
        measurements = torch.stack(
            (
                valid,
                valid & (along < 0) & (along > -0.6),
                valid & contact,
                valid & moving_command,
                valid & moving_command & slow,
                *new.unbind(-1),
                departed,
            ),
            -1,
        ).long()
        self.counts[phase].view(6, -1)[:, :9].index_add_(0, group, measurements)
        holding = env.command_manager.get_term("base_velocity").holding
        self.counts[phase].view(6, -1)[:, 11].index_add_(
            0, group, (chosen & holding).long()
        )
        self.pending = (phase, group.clone(), chosen.clone())
        self.samples += 1
        self.last = dict(
            approach_edge_id=self.edge.clone(),
            approach_episode=self.episode.clone(),
            approach_valid_pre=valid,
            approach_along_pre=along,
            approach_lateral_pre=lateral,
            approach_foot_along_pre=foot_along,
            approach_foot_lateral_pre=foot_lateral,
            approach_foot_height_pre=foot[..., 2] - env.scene.env_origins[:, None, 2],
            approach_foot_force_pre=forces,
            approach_hold_pre=holding.clone(),
        )
        for key, value in self.last.items():
            if value.is_floating_point():
                value[~chosen | ~fresh] = 0
                if not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite approach telemetry: {key}")

    def after_step(self, terminated, timed_out):
        if not self.active:
            return
        import torch

        phase, group, chosen = self.pending
        ends = torch.stack((terminated, timed_out & ~terminated), -1) & chosen[:, None]
        self.counts[phase].view(6, -1)[:, 9:11].index_add_(0, group, ends.long())
        self.pending = None

    def report(self):
        return dict(
            recipe=recipe(),
            nominal_footprint=self.footprint,
            control_steps=self.samples,
            resets_by_level_direction=self.resets.cpu().tolist(),
            forward_windows_by_level_direction=self.holds.cpu().tolist(),
            axes=["ppo/history", "terrain_row0/1/2", "ascent/descent", list(COUNTERS)],
            counts=self.counts.cpu().tolist(),
            pending_step=self.pending is not None,
            scope="Pre-action first-seam exposure; permanently censored outside3x1.5m corridor; link crossings/center-ray transitions are NOT supported success. Contact is a local net-force proxy, not a collider identity. Physical ends include censored approaches; final unfinished episodes remain unfinished.",
        )


def validate_training_report(report, *, steps, num_envs):
    counts, resets = np.asarray(report["counts"]), np.asarray(
        report["resets_by_level_direction"]
    )
    footprint = report["nominal_footprint"]
    holds = np.asarray(report["forward_windows_by_level_direction"])
    if (
        report["recipe"] != recipe()
        or report["control_steps"] != steps
        or report["pending_step"] is not False
        or counts.shape != (2, 3, 2, len(COUNTERS))
        or resets.shape != (3, 2)
        or not np.issubdtype(counts.dtype, np.integer)
        or not np.issubdtype(resets.dtype, np.integer)
        or (counts < 0).any()
        or (resets < 0).any()
        or not np.array_equal(holds, resets)
        or resets.sum() == 0
        or counts[..., 11].sum() == 0
        or counts[..., 0].sum() > steps * (num_envs // 5)
        or (counts[..., 1:8] > counts[..., :1]).any()
        or (counts[..., 4] > counts[..., 3]).any()
        or (counts[..., 5:8].sum(axis=0) > resets[..., None]).any()
        or (counts[..., 9:11].sum(axis=(0, 3)) > resets).any()
        or (counts[..., 11].sum(axis=0) > resets * HOLD_STEPS).any()
        or footprint["padding_m"] != 0.05
        or footprint["bound_m"] != FOOTPRINT_RADIUS
        or not 0 <= footprint["maximum_link_center_radius_m"] <= FOOTPRINT_RADIUS - 0.05
    ):
        raise ValueError("Invalid approach exposure or native footprint receipt")
    return deepcopy(report)


def reset_root_state(env, env_ids, pose_range, velocity_range, asset_cfg=None):
    import torch
    from isaaclab.envs.mdp.events import reset_root_state_uniform
    from isaaclab.managers import SceneEntityCfg

    asset_cfg = SceneEntityCfg("robot") if asset_cfg is None else asset_cfg
    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    )
    reset_root_state_uniform(env, ids, pose_range, velocity_range, asset_cfg)
    state = getattr(env, "_operator_step_approach", None)
    if state is None:
        return
    state.edge[ids] = -1
    state.live[ids] = False
    state.seen[ids] = False
    if not state.active:
        return
    if state.footprint is None:
        raise ValueError(
            "Verify native nominal footprint before enabling approach resets"
        )
    state.episode[ids] += 1
    selected = ids[state.step[ids]]
    selected = selected[torch.rand(len(selected), device=env.device) < 0.75]
    for (row, col), pool in state.pools.items():
        rows = selected[
            (state.rows[selected] == row) & (state.columns[selected] == col)
        ]
        state.edge[rows] = pool[
            torch.randint(len(pool), (len(rows),), device=env.device)
        ]
    if not len(selected):
        return
    state.live[selected] = True
    values = {key: value[state.edge[selected]] for key, value in state.values.items()}
    normal = values["normal"]
    tangent = normal.flip(-1) * normal.new_tensor([-1, 1])
    jitter = torch.rand((len(selected), 3), device=env.device) * 2 - 1
    robot = env.scene[asset_cfg.name]
    pose = robot.data.default_root_state[selected, :7].clone()
    pose[:, :2] = (
        values["edge_world_xy_m"]
        + (-0.75 + 0.1 * jitter[:, :1]) * normal
        + 0.1 * jitter[:, 1:2] * tangent
    )
    pose[:, 2] += values["start_height_m"] + values["roughness_m"]
    yaw = torch.atan2(normal[:, 1], normal[:, 0]) + math.pi / 18 * jitter[:, 2]
    pose[:, 3:7] = 0
    pose[:, 3], pose[:, 6] = torch.cos(yaw / 2), torch.sin(yaw / 2)
    robot.write_root_pose_to_sim(pose, env_ids=selected)
    direction = (values["target_height_m"] < values["start_height_m"]).long()
    state.resets.view(-1).index_add_(
        0, state.rows[selected] * 2 + direction, torch.ones_like(direction)
    )
