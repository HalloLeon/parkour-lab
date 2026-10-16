"""Actual collider/ray readbacks and family-specific surface measurements."""

import hashlib


def assignments(native):
    terrain, definition = native.scene.terrain, native.cfg.benchmark_terrain
    types = terrain.terrain_types.cpu().tolist()
    levels = terrain.terrain_levels.cpu().tolist()
    return [
        dict(env_id=i, column=column, row=row, **definition["columns"][column])
        for i, (column, row) in enumerate(zip(types, levels))
    ]


def _mesh(native):
    import numpy as np
    import omni.usd
    from isaaclab.sim import get_first_matching_child_prim
    from pxr import UsdGeom, UsdPhysics

    scanner = native.scene["height_scanner"]
    path = scanner.cfg.mesh_prim_paths[0]
    prim = get_first_matching_child_prim(path, lambda p: p.GetTypeName() == "Mesh")
    assert prim is not None and prim.HasAPI(UsdPhysics.CollisionAPI)
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    transform = np.asarray(omni.usd.get_world_transform_matrix(mesh)).T
    world = points @ transform[:3, :3].T + transform[:3, 3]
    assert np.isfinite(world).all() and len(indices) % 3 == 0
    receipt = dict(
        prim_path=str(prim.GetPath()),
        vertices=len(points),
        triangles=len(indices) // 3,
        points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
        world_bounds_m=[world.min(0).tolist(), world.max(0).tolist()],
        collision_enabled=bool(
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        ),
    )
    assert receipt["collision_enabled"]
    return scanner.meshes[path], receipt


def _cast(native, mesh, offsets, *, require_hits=True):
    import torch
    from isaaclab.utils.warp import raycast_mesh

    origins = native.scene.env_origins
    points = origins[:, None, :] + offsets
    starts = points.clone()
    starts[..., 2] = origins[:, None, 2] + 20.0
    directions = torch.zeros_like(starts)
    directions[..., 2] = -1
    hits, _, _, _ = raycast_mesh(starts, directions, mesh, max_dist=40.0)
    if require_hits:
        assert torch.isfinite(hits).all(), "A declared terrain probe missed its surface"
    return hits


def _surface_metrics(assignment, height, axis, definition, physical_level):
    import numpy as np

    middle, spacing = len(axis) // 2, float(axis[1] - axis[0])
    center = float(height[middle, middle])
    x, y = np.meshgrid(axis, axis)
    outside = (abs(x) > 1.3) | (abs(y) > 1.3)
    line = height[middle]
    result = dict(
        **assignment,
        physical_level=physical_level,
        height_range_m=[float(height.min()), float(height.max())],
        height_ptp_m=float(np.ptp(height)),
        height_rms_about_mean_m=float(height.std()),
        adjacent_height_rms_m=float(np.sqrt(np.mean(np.diff(height, axis=1) ** 2))),
        outside_platform_fraction=float(outside.mean()),
        varied_outside_fraction=float((abs(height[outside] - center) > 0.0025).mean()),
        probe_extent_xy_m=[float(axis[0]), float(axis[-1])],
        probe_spacing_m=spacing,
    )
    family, subtype = assignment["family"], assignment["subtype"]
    requested = definition["levels"][physical_level]
    if family == "Flat":
        assert result["height_ptp_m"] < 1e-4
    else:
        assert result["height_ptp_m"] > 0.003
        assert result["varied_outside_fraction"] > 0.01
    if family in ("Slopes", "Stairs"):
        sign = 1 if subtype == "pyramid" else -1
        result["center_minus_outer_height_m"] = center - float(line[-1])
        assert sign * result["center_minus_outer_height_m"] > 0.005
    if family == "Slopes":
        region = (axis > 1.4) & (axis < 5.2)
        slope = abs(float(np.polyfit(axis[region], line[region], 1)[0]))
        result["axis_slope_rise_run"] = slope
        assert abs(slope - requested["slope_rise_run"]) < 0.015
    if family == "Stairs":
        steps = np.diff(np.unique(np.round(line, 4)))
        result["measured_riser_m"] = float(np.median(steps[steps > 0.001]))
        assert abs(result["measured_riser_m"] - requested["stair_riser_m"]) < 0.001
    if family == "Hills":
        assert (
            abs(result["height_ptp_m"] - 2 * requested["hill_native_amplitude_m"])
            < 0.025
        )
    return result


def check_geometry(task, output):
    import numpy as np
    import torch

    native, definition = task._env, task._env.cfg.benchmark_terrain
    cases = assignments(native)
    assert [row["column"] for row in cases] == list(range(20))
    assert {row["family"] for row in cases} == {
        "Flat",
        "Uneven",
        "Slopes",
        "Stairs",
        "Hills",
    }
    mesh, mesh_receipt = _mesh(native)
    axis = torch.linspace(-5.4, 5.4, 73, device=task.device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack(
        (x.flatten(), y.flatten(), torch.zeros(x.numel(), device=task.device)), -1
    )
    hits = _cast(native, mesh, offsets)
    heights, positions = hits[..., 2].cpu().numpy(), axis.cpu().numpy()
    starts = native.scene["robot"].data.root_link_pos_w.clone()
    center_hits = _cast(native, mesh, (starts - native.scene.env_origins)[:, None])[
        :, 0
    ]
    clearance = (starts[:, 2] - center_hits[:, 2]).cpu().tolist()
    assert all(0.15 < value < 0.8 for value in clearance)
    from parkour_lab.control.proprioception import FOOT_NAMES

    robot = native.scene["robot"]
    feet = robot.data.body_link_pos_w[
        :, [robot.body_names.index(name) for name in FOOT_NAMES]
    ].clone()
    foot_hits = _cast(native, mesh, feet - native.scene.env_origins[:, None])
    foot_clearance = feet[..., 2] - foot_hits[..., 2]
    assert (
        foot_clearance > -0.05
    ).all(), "Initial foot link centers penetrate the surface by over 5 cm"
    edge_xy = torch.tensor(
        [
            (5.95, 0, 0),
            (-5.95, 0, 0),
            (0, 5.95, 0),
            (0, -5.95, 0),
            (6.1, 0, 0),
            (-6.1, 0, 0),
            (0, 6.1, 0),
            (0, -6.1, 0),
        ],
        device=task.device,
    )
    edge_hits = _cast(native, mesh, edge_xy, require_hits=False)
    np.savez_compressed(
        output / "surface_readback.npz",
        axis_m=positions,
        heights_w_m=heights.reshape(20, 73, 73),
        env_origins_w_m=native.scene.env_origins.cpu().numpy(),
        initial_root_link_position_w_m=starts.cpu().numpy(),
        initial_foot_link_position_w_m=feet.cpu().numpy(),
        initial_foot_surface_hits_w_m=foot_hits.cpu().numpy(),
        edge_offsets_m=edge_xy.cpu().numpy(),
        edge_surface_hits_w_m=edge_hits.cpu().numpy(),
    )
    rows = []
    for row, height in zip(cases, heights):
        physical = definition["physical_levels_by_row"][row["row"]]
        rows.append(
            _surface_metrics(
                row, height.reshape(73, 73), positions, definition, physical
            )
        )
    return dict(
        mesh=mesh_receipt,
        root_clearance_m=clearance,
        foot_link_clearance_m=foot_clearance.cpu().tolist(),
        edge_hit_valid=edge_hits.isfinite().all(-1).cpu().tolist(),
        surfaces=rows,
    )
