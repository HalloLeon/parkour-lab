"""First-attempt terrain kinematics from recorded, pre-reset 50 Hz evidence.

This is not a qualification bank: archive integrity, independent layouts and
nominal/randomized attempt assignment are separate from these geometric scores.
"""

import numpy as np

PERIOD_S = 0.02
STEPS = 1500


def _inside(points, bounds):
    return ((points >= bounds[0]) & (points <= bounds[1])).all(axis=-1)


def _segment_box(a, b, bounds):
    """Closed segment/rectangle interval, with the exit edge retained."""
    delta = b - a
    first, last = np.full(2, -np.inf), np.full(2, np.inf)
    for axis in range(2):
        if delta[axis] == 0:
            if not bounds[0, axis] <= a[axis] <= bounds[1, axis]:
                return None
        else:
            times = (bounds[:, axis] - a[axis]) / delta[axis]
            first[axis], last[axis] = min(times), max(times)
    entry, exit = max(0.0, max(first)), min(1.0, min(last))
    if exit < entry:
        return None
    return entry, exit, int(last[1] <= last[0])


def _tracking(velocity, yaw, command):
    if not len(velocity):
        return dict(
            samples=0, planar_error_m_s=None, yaw_error_rad_s=None, passed=False
        )
    planar = float(np.linalg.norm(velocity[:, :2] - command[:, :2], axis=-1).mean())
    angular = float(np.abs(yaw - command[:, 2]).mean())
    return dict(
        samples=len(velocity),
        planar_error_m_s=planar,
        yaw_error_rad_s=angular,
        passed=planar <= 0.20 and angular <= 0.20,
    )


def _crossing(position, feet, trial, velocity, yaw, command):
    footprint = np.asarray(trial["target_footprint_local_m"], dtype=float)
    approach, exit_point = (
        np.asarray(trial[key], dtype=float)
        for key in ("approach_edge_local_m", "exit_edge_local_m")
    )
    axis = int(np.argmax(np.abs(exit_point - approach)))
    across = 1 - axis
    direction = np.sign(exit_point[axis] - approach[axis])
    length = abs(exit_point[axis] - approach[axis])
    if (
        not length
        or approach[across] != exit_point[across]
        or approach[axis] != footprint[0 if direction > 0 else 1, axis]
        or exit_point[axis] != footprint[1 if direction > 0 else 0, axis]
    ):
        raise ValueError("Target approach/exit must be opposite footprint edges")
    route = np.column_stack(
        ((position[:, axis] - approach[axis]) * direction, position[:, across])
    )
    bounds = np.array([[0, footprint[0, across]], [length, footprint[1, across]]])
    landing = np.asarray(trial["far_side_landing_local_m"], dtype=float)
    entry_step = clear_step = far_exit_step = None
    failures = []
    if _inside(route[0], bounds):
        failures.append("base starts inside the footprint")
    else:
        for step, (a, b) in enumerate(zip(route[:-1], route[1:])):
            hit = _segment_box(a, b, bounds)
            if hit is None:
                continue
            if hit[0] == hit[1] and not _inside(b, bounds):
                continue  # A tangential point alone is not an observed entry.
            if not _inside(b, bounds):
                failures.append(
                    "footprint intersection lacks an observed inside sample"
                )
                break
            entry_point = a + hit[0] * (b - a)
            if not (
                a[0] < 0 <= b[0]
                and bounds[0, 1] <= entry_point[1] <= bounds[1, 1]
                and hit[0] == -a[0] / (b[0] - a[0])
            ):
                failures.append("first footprint entry is not from the approach side")
                break
            entry_step = step
            break
    if entry_step is not None:
        for step in range(entry_step, len(feet)):
            a, b = route[step : step + 2]
            hit = _segment_box(a, b, bounds)
            if hit is not None:
                if hit[0] == hit[1] and not _inside(b, bounds):
                    continue
                if far_exit_step is None and not _inside(a, bounds):
                    # Retreat is allowed, but going around and re-entering elsewhere
                    # cannot substitute for traversing from the original approach.
                    if not (
                        _inside(b, bounds)
                        and a[0] < 0 <= b[0]
                        and hit[0] == -a[0] / (b[0] - a[0])
                    ):
                        failures.append(
                            "footprint re-entry is not from the approach side"
                        )
                        break
                _, leave, edge = hit
                if leave < 1 and edge == 1:
                    failures.append("base leaves the footprint through a side edge")
                    break
                if leave < 1 and edge == 0 and b[0] > length and b[0] > a[0]:
                    far_exit_step = step if far_exit_step is None else far_exit_step
            points = np.vstack((position[step + 1], feet[step]))
            beyond = (points[:, axis] - exit_point[axis]) * direction > 0
            if (
                far_exit_step is not None
                and beyond.all()
                and _inside(points[:, :2], landing).all()
            ):
                clear_step = step
                break
    if entry_step is None and not failures:
        failures.append("no observed approach-side footprint entry")
    if clear_step is None:
        failures.append("base and all four feet do not clear the far landing")
    elif clear_step >= 1350:
        failures.append("crossing completes after 27 s")
    tracking = (
        _tracking(
            velocity[entry_step : clear_step + 1],
            yaw[entry_step : clear_step + 1],
            command[entry_step : clear_step + 1],
        )
        if clear_step is not None
        else None
    )
    if tracking is not None and not tracking["passed"]:
        failures.append("crossing tracking exceeds limits")
    return dict(
        entry_step=entry_step,
        entry_time_s=None if entry_step is None else (entry_step + 1) * PERIOD_S,
        clear_step=clear_step,
        clear_time_s=None if clear_step is None else (clear_step + 1) * PERIOD_S,
        base_far_exit_step=far_exit_step,
        complete=clear_step is not None,
        tracking=tracking,
        failures=failures,
        passed=not failures,
    )


def score_terrain_attempt(trace, source_metadata):
    """Score one recorded row once, cutting at its first native ending.

    Missing/malformed arrays reject evidence. Nonfinite first-attempt values and
    incomplete or wrong tapes cannot pass. All reported steps are zero-indexed
    post-physics rows: row k is time (k+1)*0.02 s. No I/O or policy execution.
    """
    position = np.asarray(trace.get("position_w"))
    if (
        position.ndim != 3
        or position.shape[1:] != (1, 3)
        or not 1 <= len(position) <= STEPS
    ):
        raise ValueError("Require one terrain row with 1 through 1500 motion samples")
    supplied = len(position)
    body = np.asarray(trace.get("body_position_w"))
    if (
        body.ndim != 4
        or body.shape[:2] != (supplied, 1)
        or body.shape[-1] != 3
        or not body.shape[2]
    ):
        raise ValueError("Require all native body-link centres")
    shapes = {
        key: (supplied, 1, width)
        for key, width in {
            "position_w": 3,
            "quaternion_w": 4,
            "linear_velocity_b": 3,
            "angular_velocity_b": 3,
            "base_contact_force_w": 3,
            "command": 3,
        }.items()
    }
    shapes.update(
        foot_position_w=(supplied, 1, 4, 3),
        body_position_w=body.shape,
        initial_position_w=(1, 3),
        initial_quaternion_w=(1, 4),
        initial_foot_position_w=(1, 4, 3),
        initial_body_position_w=(1, body.shape[2], 3),
        terminated=(supplied, 1),
        truncated=(supplied, 1),
    )
    arrays = {}
    for name, shape in shapes.items():
        value = np.asarray(trace.get(name))
        if value.shape != shape or value.dtype.kind not in "biuf":
            raise ValueError(
                f"Malformed terrain evidence: {name} must have shape {shape}"
            )
        arrays[name] = value
    if any(arrays[key].dtype != np.bool_ for key in ("terminated", "truncated")):
        raise ValueError("Terrain termination flags must be boolean")
    try:
        trial = source_metadata["trial"]
        transform = np.asarray(trial["local_to_world_column_transform"], dtype=float)
        size = np.asarray(source_metadata["size_m"], dtype=float)
        bounds = {
            key: np.asarray(trial[key], dtype=float)
            for key in (
                "target_footprint_local_m",
                "far_side_landing_local_m",
                "final_stop_region_local_m",
            )
        }
        edges = [
            np.asarray(trial[key], dtype=float)
            for key in ("approach_edge_local_m", "exit_edge_local_m")
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Missing or malformed terrain geometry metadata") from error
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.array_equal(transform[3], [0, 0, 0, 1])
        or not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-8, rtol=0
        )
        or not np.isclose(np.linalg.det(transform[:3, :3]), 1, atol=1e-8, rtol=0)
        or size.shape != (2,)
        or not np.isfinite(size).all()
        or (size <= 0).any()
        or any(
            b.shape != (2, 2) or not np.isfinite(b).all() or (b[1] <= b[0]).any()
            for b in bounds.values()
        )
        or any(e.shape != (2,) or not np.isfinite(e).all() for e in edges)
    ):
        raise ValueError(
            "Require finite rigid transform, map size and target rectangles"
        )
    ends = np.flatnonzero(arrays["terminated"][:, 0] | arrays["truncated"][:, 0])
    length = int(ends[0] + 1) if len(ends) else supplied
    failures = []
    result = dict(
        version="terrain_attempt_kinematics_v1",
        qualified=False,
        qualification_eligible=False,
        complete=length == STEPS,
        supplied_steps=supplied,
        first_attempt_steps=length,
        scope="Single first-attempt kinematics; not an independently assigned qualification bank. Domain checks use body-link centres, not collider extents.",
        failures=failures,
        crossing=None,
        phases=[],
        finish=None,
        failure_events={},
    )
    if length != STEPS:
        failures.append("incomplete 30 s terrain tape")
    for name, label in (
        ("terminated", "native termination"),
        ("truncated", "native timeout"),
    ):
        ids = np.flatnonzero(arrays[name][:length, 0])
        if len(ids):
            failures.append(label)
            result["failure_events"][name] = int(ids[0])
    if not all(
        np.isfinite(value if name.startswith("initial_") else value[:length]).all()
        for name, value in arrays.items()
    ):
        failures.append("nonfinite first-attempt evidence")
        result["kinematic_passed"] = False
        return result
    quaternion = np.vstack(
        (arrays["initial_quaternion_w"], arrays["quaternion_w"][:length, 0])
    ).astype(float)
    if not np.allclose(np.linalg.norm(quaternion, axis=-1), 1, atol=1e-3, rtol=0):
        failures.append("invalid root quaternion")
        result["kinematic_passed"] = False
        return result
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True)
    command = arrays["command"][:length, 0].astype(float)
    schedule = np.zeros((STEPS, 3))
    schedule[100:1350, 0] = 0.35
    if not np.allclose(command, schedule[:length], atol=1e-6, rtol=0):
        failures.append("commands differ from fixed 2/25/3 s terrain tape")
    world_position = np.vstack(
        (arrays["initial_position_w"], position[:length, 0])
    ).astype(float)

    def local(points):
        return (points - transform[:3, 3]) @ transform[:3, :3]

    local_position = local(world_position)
    feet = local(arrays["foot_position_w"][:length, 0])
    local_body = local(
        np.concatenate((arrays["initial_body_position_w"], body[:length, 0]))
    )
    map_bounds = np.array([np.zeros(2), size])
    outside = ~_inside(local_body[..., :2], map_bounds).all(axis=-1) | ~_inside(
        local_position[:, :2], map_bounds
    )
    if outside.any():
        failures.append("body-link centre leaves finite map")
        result["failure_events"]["out_of_domain_step"] = int(
            np.flatnonzero(outside)[0] - 1
        )
    q = quaternion[1:]
    tilt = np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))
    force = np.linalg.norm(arrays["base_contact_force_w"][:length, 0], axis=-1)
    for mask, label in (
        (tilt > np.deg2rad(60), "sustained tilt fall"),
        (force > 20, "sustained base contact"),
    ):
        sustained = (
            np.convolve(mask.astype(int), np.ones(5, dtype=int), mode="valid")
            if length >= 5
            else []
        )
        ids = np.flatnonzero(np.asarray(sustained) == 5)
        if len(ids):
            failures.append(label)
            result["failure_events"][label] = int(ids[0] + 4)
    velocity = arrays["linear_velocity_b"][:length, 0].astype(float)
    yaw = arrays["angular_velocity_b"][:length, 0, 2].astype(float)
    for start, end in ((0, 100), (100, 1350), (1350, 1500)):
        observed = min(end, length)
        tracking = _tracking(
            velocity[start + 50 : observed],
            yaw[start + 50 : observed],
            command[start + 50 : observed],
        )
        phase = dict(
            start_s=start * PERIOD_S,
            end_s=end * PERIOD_S,
            complete=length >= end,
            samples=max(0, observed - start),
            tracking=tracking,
        )
        result["phases"].append(phase)
        if not phase["complete"] or not tracking["passed"]:
            failures.append(
                f"phase {start * PERIOD_S:g}s incomplete or tracking exceeds limits"
            )
    stall = next(
        (
            k
            for k in range(150, min(1350, length) - 150 + 1)
            if np.linalg.norm(world_position[k + 150, :2] - world_position[k, :2])
            < 0.10
        ),
        None,
    )
    if stall is not None:
        failures.append("translating phase stall")
        result["failure_events"]["stall_window_start_s"] = stall * PERIOD_S
    crossing = _crossing(local_position, feet, trial, velocity, yaw, command)
    result["crossing"] = crossing
    failures.extend(crossing["failures"])
    finish = dict(complete=length == STEPS, samples=max(0, length - 1399), passed=False)
    if length == STEPS:
        # Rotate the full body-frame root-COM velocity, including its vertical component.
        vector = q[:, 1:]
        cross = 2 * np.cross(vector, velocity)
        world_velocity = velocity + q[:, :1] * cross + np.cross(vector, cross)
        speed = np.linalg.norm(world_velocity[1399:, :2], axis=-1)
        joint_fraction = float(np.mean((speed <= 0.08) & (np.abs(yaw[1399:]) <= 0.10)))
        points = np.concatenate(
            (local_position[1400:, None, :2], feet[1399:, :, :2]), axis=1
        )
        over_region = bool(_inside(points, bounds["final_stop_region_local_m"]).all())
        max_tilt = float(np.rad2deg(tilt[1399:].max()))
        excursion = float(
            np.linalg.norm(
                world_position[1351:, :2] - world_position[1350, :2], axis=-1
            ).max()
        )
        drift = float(
            np.linalg.norm(world_position[1500, :2] - world_position[1400, :2])
        )
        finish.update(
            over_stop_region=over_region,
            max_tilt_degrees=max_tilt,
            joint_velocity_fraction=joint_fraction,
            onset_excursion_m=excursion,
            net_drift_m=drift,
            passed=over_region
            and max_tilt <= 20
            and joint_fraction >= 0.9
            and excursion <= 0.15
            and drift <= 0.10,
        )
    result["finish"] = finish
    if not finish["passed"]:
        failures.append("stable final stop incomplete or outside limits")
    result["kinematic_passed"] = not failures
    return result
