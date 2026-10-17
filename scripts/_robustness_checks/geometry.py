"""Read composed/OOD surfaces from the actual collider, with paired base controls."""


def sample_surface(task, output):
    import numpy as np
    import torch

    from parkour_lab.control.proprioception import FOOT_NAMES
    from scripts._terrain_checks.geometry import (
        _cast,
        _mesh,
        _surface_metrics,
        assignments,
    )

    native = task._env
    definition = native.cfg.benchmark_terrain
    rows = assignments(native)
    assert len(rows) == 2
    mesh, collider = _mesh(native)
    axis = torch.linspace(-5.4, 5.4, 73, device=task.device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    offsets = torch.stack(
        (x.flatten(), y.flatten(), torch.zeros(x.numel(), device=task.device)), -1
    )
    heights = _cast(native, mesh, offsets)[..., 2].cpu().numpy().reshape(2, 73, 73)
    robot = native.scene["robot"]
    root = robot.data.root_link_pos_w.clone()
    feet = robot.data.body_link_pos_w[
        :, [robot.body_names.index(name) for name in FOOT_NAMES]
    ].clone()
    origin = native.scene.env_origins
    root_hits = _cast(native, mesh, (root - origin)[:, None])[:, 0]
    foot_hits = _cast(native, mesh, feet - origin[:, None])
    root_clearance, foot_clearance = (
        root[:, 2] - root_hits[:, 2],
        feet[..., 2] - foot_hits[..., 2],
    )
    assert ((root_clearance > 0.15) & (root_clearance < 0.8)).all()
    assert (foot_clearance > -0.05).all()
    positions = axis.cpu().numpy()
    measurements = []
    for row, height in zip(rows, heights, strict=True):
        level = definition["physical_levels_by_row"][row["row"]]
        if definition.get("condition", "id") == "composition":
            # Roughness changes pointwise slopes/risers; do not apply the pure
            # primitive estimator to a composed surface and call that a failure.
            measurements.append(
                dict(**row, physical_level=level, height_ptp_m=float(np.ptp(height)))
            )
        else:
            measurements.append(
                _surface_metrics(row, height, positions, definition, level)
            )
    values = {
        "axis_m": positions,
        "height_w_m": heights,
        "env_origin_w_m": origin.cpu().numpy().copy(),
        "root_position_w_m": root.cpu().numpy(),
        "foot_position_w_m": feet.cpu().numpy(),
        "root_support_w_m": root_hits.cpu().numpy(),
        "foot_support_w_m": foot_hits.cpu().numpy(),
    }
    np.savez_compressed(output / "surface.npz", **values)
    return values, dict(
        collider=collider,
        measurements=measurements,
        root_clearance_m=root_clearance.cpu().tolist(),
        foot_clearance_m=foot_clearance.cpu().tolist(),
        sample_count_per_replica=73 * 73,
        witness="native collision mesh and downward raycasts; no learned traversal claim",
    )


def compare_surfaces(samples, definitions):
    import numpy as np

    base, composed = samples["id"], samples["composition"]
    axis = base["axis_m"]
    x, y = np.meshgrid(axis, axis)
    platform = (abs(x) <= 0.9) & (abs(y) <= 0.9)
    outside = (abs(x) > 1.3) | (abs(y) > 1.3)
    relative = {
        key: sample["height_w_m"] - sample["env_origin_w_m"][:, None, None, 2]
        for key, sample in samples.items()
    }
    delta = relative["composition"] - relative["id"]
    assert np.max(abs(delta[:, platform])) < 2e-4
    changed = np.mean(abs(delta[:, outside]) > 0.0025, axis=1)
    assert (
        changed > 0.01
    ).all(), "Composition did not change the realized outer surface"
    np.testing.assert_allclose(
        base["root_position_w_m"], composed["root_position_w_m"], atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(
        base["foot_position_w_m"], composed["foot_position_w_m"], atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(
        base["root_support_w_m"], composed["root_support_w_m"], atol=1e-5, rtol=0
    )
    np.testing.assert_allclose(
        base["foot_support_w_m"], composed["foot_support_w_m"], atol=1e-5, rtol=0
    )
    from parkour_lab.environments.terrains.specification import physical_level

    cap = physical_level(7)
    level = definitions["terrain_ood"]["fixed_level"]
    values = definitions["terrain_ood"]["levels"][level]
    family = definitions["terrain_ood"]["columns"][0]["family"]
    field = {
        "Slopes": "slope_rise_run",
        "Stairs": "stair_riser_m",
        "Hills": "hill_native_amplitude_m",
    }[family]
    assert values[field] > cap[field]
    return dict(
        paired_platform_maximum_difference_m=float(np.max(abs(delta[:, platform]))),
        composed_outside_changed_fraction=changed.tolist(),
        composed_delta_range_m=[float(delta.min()), float(delta.max())],
        paired_starts_and_support=True,
        ood_field=field,
        ood_value=values[field],
        training_cap=cap[field],
        ood_excluded_support=True,
        traversal_validated=False,
    )
