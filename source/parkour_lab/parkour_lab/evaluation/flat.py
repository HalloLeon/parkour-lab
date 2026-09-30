"""Flat first-attempt kinematics and balanced development-bank summaries.

Ordinary diagnostics remain distinct from the independently seeded, noisy bank.
Neither is final qualification. Positions/headings include the pre-command onset
sample; velocities and termination flags are post-physics, before automatic reset.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from parkour_lab.control.command_tape import PERIOD_S

PROFILES = {
    "stand": (0.0, 0.0, 0.0),
    "forward-02": (0.2, 0.0, 0.0),
    "forward-05": (0.5, 0.0, 0.0),
    "reverse-02": (-0.2, 0.0, 0.0),
    "left-02": (0.0, 0.2, 0.0),
    "right-02": (0.0, -0.2, 0.0),
    "pivot-left": (0.0, 0.0, 0.5),
    "pivot-right": (0.0, 0.0, -0.5),
    "arc-left": (0.35, 0.0, 0.3),
    "arc-right": (0.35, 0.0, -0.3),
}
STEPS = 550


@dataclass(frozen=True)
class Thresholds:
    settling_s: float = 1.0
    block_s: float = 0.4
    planar_error_m_s: float = 0.10
    pivot_yaw_error_rad_s: float = 0.10
    moving_yaw_error_rad_s: float = 0.15
    wrong_sign_fraction: float = 0.05
    stationary_speed_m_s: float = 0.08
    stationary_onset_excursion_m: float = 0.15
    stationary_drift_2s_m: float = 0.10
    zero_twist_heading_excursion_rad: float = 0.15
    flat_attitude_rad: float = math.radians(15)


def profile_commands(profile):
    """Eleven seconds: stop 2 s, named command 6 s, stop 3 s."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown flat profile: {profile}")
    return [(0.0, 0.0, 0.0)] * 100 + [PROFILES[profile]] * 300 + [(0.0, 0.0, 0.0)] * 150


def _pass_counts(passed, total):
    """Observed counts with a Wilson 95% interval; the interval is not a gate."""
    z = 1.959963984540054
    fraction = passed / total
    denominator = 1 + z * z / total
    center = (fraction + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total**2))
        / denominator
    )
    return dict(
        passed=passed,
        total=total,
        fraction=fraction,
        wilson_95=[max(0.0, center - radius), min(1.0, center + radius)],
    )


def summarize_development_profile(profile, trials, manifest):
    """Keep the original 100 IDs and denominator, including unresolved evidence.

    Duplicate IDs are rejected, never selected among. Recovery needs an external
    record of a verified infrastructure fault; this scorer cannot authorize it.
    """
    from parkour_lab.environments.randomization import development_manifest

    if profile not in PROFILES or manifest != development_manifest(profile):
        raise ValueError("Development manifest differs from its canonical 100-ID draws")
    attempts = manifest["attempts"]
    indexed = {}
    for trial in trials:
        index = trial.get("attempt_index")
        if type(index) is not int or not 0 <= index < 100 or index in indexed:
            raise ValueError(
                "Require unique assigned attempt IDs; do not select among reruns"
            )
        if type(trial.get("passed")) is not bool:
            raise ValueError("Each recorded outcome must declare a boolean pass")
        indexed[index] = dict(trial, stratum=attempts[index]["stratum"])
    missing = sorted(set(range(100)) - set(indexed))
    rows = [
        indexed.get(
            index,
            dict(
                attempt_index=index,
                stratum=attempts[index]["stratum"],
                passed=False,
                failures=["unresolved missing evidence"],
            ),
        )
        for index in range(100)
    ]
    strata = {
        name: _pass_counts(sum(row["passed"] for row in rows[start : start + 50]), 50)
        for name, start in (("nominal", 0), ("randomized", 50))
    }
    overall = _pass_counts(sum(row["passed"] for row in rows), 100)
    return dict(
        version="flat_development_bank_v1",
        namespace=manifest["namespace"],
        group_id=f"flat/{profile}",
        manifest_content_sha256=hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
        qualified=False,
        qualification_eligible=False,
        complete=not missing,
        missing_attempt_ids=missing,
        trials=rows,
        overall=overall,
        strata=strata,
        group_passed=not missing
        and overall["passed"] >= 90
        and all(value["passed"] >= 45 for value in strata.values()),
    )


def score_development_profile(trace, profile, manifest):
    """Apply unchanged kinematics to all prospective rows of one noisy flat group."""
    diagnostic = score_flat_profile(trace, profile)
    if diagnostic["total"] != 100:
        raise ValueError("A development profile requires all 100 assigned rows")
    trials = [
        dict(row, attempt_index=row["env_id"], passed=row["diagnostic_passed"])
        for row in diagnostic["trials"]
    ]
    result = summarize_development_profile(profile, trials, manifest)
    starts = [row["start"] for row in manifest["attempts"]]
    xy = trace["initial_position_w"][:, :2] - trace["env_origins"][:, :2]
    roll, pitch, yaw = _angles(trace["initial_quaternion_w"])
    difference = yaw - np.array([row["start_yaw"] for row in starts])
    if (
        not np.allclose(xy, [row["start_xy"] for row in starts], atol=1e-4, rtol=0)
        or not np.allclose(roll, 0, atol=1e-4, rtol=0)
        or not np.allclose(pitch, 0, atol=1e-4, rtol=0)
        or not np.allclose(
            np.arctan2(np.sin(difference), np.cos(difference)), 0, atol=1e-4, rtol=0
        )
    ):
        raise ValueError("Initial native root poses differ from the frozen manifest")
    result["thresholds"] = diagnostic["thresholds"]
    return result


def aggregate_development_bank(profile_results):
    """Recompute counts for one controller; callers verify the trace artifacts."""
    from parkour_lab.environments.randomization import development_manifest

    groups = {}
    controllers = set()
    for result in profile_results:
        group = result.get("group_id")
        if group not in {f"flat/{profile}" for profile in PROFILES} or group in groups:
            raise ValueError("Require unique known flat development groups")
        if result.get("version") != "flat_development_bank_v1":
            raise ValueError(
                "Ordinary diagnostics cannot be promoted to development evidence"
            )
        controller = result.get("controller_state_sha256", "")
        if (
            not isinstance(controller, str)
            or len(controller) != 64
            or any(char not in "0123456789abcdef" for char in controller)
        ):
            raise ValueError("Require each group's verified controller-state identity")
        controllers.add(controller)
        if len(controllers) != 1:
            raise ValueError(
                "All development groups must use the same frozen controller"
            )
        trials = result.get("trials", [])
        missing_ids = result.get("missing_attempt_ids", [])
        if (
            len(trials) != 100
            or [row.get("attempt_index") for row in trials] != list(range(100))
            or any(type(i) is not int or not 0 <= i < 100 for i in missing_ids)
            or len(set(missing_ids)) != len(missing_ids)
        ):
            raise ValueError(
                "Development group does not preserve its assigned attempt IDs"
            )
        profile = group.removeprefix("flat/")
        recomputed = summarize_development_profile(
            profile,
            [row for row in trials if row["attempt_index"] not in missing_ids],
            development_manifest(profile),
        )
        if any(result.get(key) != value for key, value in recomputed.items()):
            raise ValueError("Development group counts or evidence metadata changed")
        groups[group] = result
    missing = sorted({f"flat/{profile}" for profile in PROFILES} - groups.keys())
    complete = not missing and all(row["complete"] for row in groups.values())
    return dict(
        version="flat_development_bank_v1",
        controller_state_sha256=next(iter(controllers), None),
        qualified=False,
        complete=complete,
        missing_groups=missing,
        groups=groups,
        bank_passed=complete and all(row["group_passed"] for row in groups.values()),
    )


def analyze_development_bank(root):
    """Read and rescore a single frozen actor's saved bank; never run or retry jobs."""
    from parkour_lab.control.command_tape import load_tape
    from parkour_lab.environments.randomization import TaskRandomization

    root = Path(root)
    if not root.is_dir():
        raise ValueError("Flat-bank analysis requires a directory of evaluation runs")
    results, receipts, profiles = [], [], set()
    shared = None
    for path in sorted(root.rglob("report.json")):
        report = json.loads(path.read_text())
        task = report.get("config", {}).get("task", {})
        profile = task.get("bank_profile")
        if profile is None:
            continue

        def require(condition, reason):
            if not condition:
                raise ValueError(f"{path}: {reason}")

        require(
            profile in PROFILES and profile not in profiles,
            "unknown or repeated group; recovery is not automatic",
        )
        profiles.add(profile)
        require(
            report.get("status") == "EVALUATION_COMPLETE_NOT_QUALIFIED"
            and report.get("policy_unchanged") is True
            and report.get("control_steps") == STEPS
            and report.get("environment_transitions") == STEPS * 100
            and task.get("terrain") == "flat"
            and task.get("num_envs") == 100
            and report.get("cleanup")
            == {"environment": "complete", "application": "complete"},
            "require a complete, frozen 100-row evaluation and successful cleanup",
        )
        identity = (report.get("actor_sha256"), report.get("package_sources"))
        require(
            isinstance(identity[0], str)
            and len(identity[0]) == 64
            and all(char in "0123456789abcdef" for char in identity[0])
            and isinstance(identity[1], dict)
            and bool(identity[1]),
            "missing actor or executed-source identity",
        )
        require(
            shared is None or shared == identity,
            "actor or executed sources changed across groups",
        )
        shared = identity
        motor = report.get("motor_delivery", {})
        require(
            all(
                motor.get(key) == STEPS
                for key in (
                    "encoded_steps",
                    "verified_delivery_steps",
                    "native_step_returns",
                )
            )
            and motor.get("faulted") is False
            and motor.get("pending_delivery") is False
            and motor.get("native_verified_rows", -1)
            + motor.get("excluded_terminal_rows", -1)
            == STEPS * 100,
            "incomplete or invalid native motor delivery",
        )
        for name, digest in {
            "manifest.json": report.get("manifest_sha256"),
            **{
                name: report.get("evidence_sha256", {}).get(name)
                for name in ("commands.json", "tracking.npz", "motion.npz")
            },
        }.items():
            evidence = path.parent / name
            require(
                evidence.is_file()
                and hashlib.sha256(evidence.read_bytes()).hexdigest() == digest,
                f"missing or changed evidence: {name}",
            )
        tape = load_tape(path.parent / "commands.json")
        commands = [
            segment["command"]
            for segment in tape["segments"]
            for _ in range(segment["start_step"], segment["end_step_exclusive"])
        ]
        require(
            commands == [list(command) for command in profile_commands(profile)],
            "recorded commands differ from the profile",
        )
        manifest = json.loads((path.parent / "manifest.json").read_text())
        with np.load(path.parent / "motion.npz", allow_pickle=False) as archive:
            result = score_development_profile(dict(archive), profile, manifest)
        with np.load(path.parent / "tracking.npz", allow_pickle=False) as archive:
            require(
                {
                    "causal_sensor_noise",
                    "initial_raw_sensors",
                    "initial_noisy_sensors",
                    "root_com_velocity",
                }.issubset(archive.files),
                "incomplete tracking/input evidence",
            )
            require(
                archive["root_com_velocity"].shape == (STEPS, 100, 3)
                and all(np.isfinite(archive[key]).all() for key in archive.files),
                "incomplete or nonfinite tracking/input evidence",
            )
            noise = archive["causal_sensor_noise"]
            amplitudes = np.array([0.2] * 3 + [0.05] * 3 + [0.01] * 12 + [1.5] * 12)
            require(
                noise.shape == (STEPS, 100, 30)
                and noise.dtype == np.float32
                and np.isfinite(noise).all()
                and (np.abs(noise) <= amplitudes + 1e-7).all(),
                "missing, malformed or out-of-bounds causal noise",
            )
            noise_report = report.get("sensor_noise", {})
            require(
                noise_report.get("generated_frames") == STEPS
                and noise_report.get("draw_sha256")
                == hashlib.sha256(noise.tobytes()).hexdigest(),
                "causal noise digest or frame count changed",
            )
            draws = TaskRandomization(
                dict(num_envs=100, seed=0, bank_profile=profile), evaluation=True
            )
            for sample in noise:
                expected = (
                    draws.uniform(None, 30, "observation-noise", -1, 1) * amplitudes
                ).astype(np.float32)
                require(
                    np.array_equal(sample, expected),
                    "noise differs from the assigned stream",
                )
            raw, noisy = (
                archive["initial_raw_sensors"],
                archive["initial_noisy_sensors"],
            )
            require(
                raw.shape == noisy.shape == (100, 30)
                and np.isfinite(raw).all()
                and np.allclose(raw + noise[0], noisy, atol=1e-6, rtol=0),
                "first causal input does not contain the recorded additive noise",
            )
        stored = report.get("flat_development", {})
        require(
            all(stored.get(key) == value for key, value in result.items()),
            "stored development outcome differs from rescored evidence",
        )
        result["controller_state_sha256"] = stored.get("controller_state_sha256")
        require(
            tape["metadata"].get("controller_sha256")
            == result["controller_state_sha256"],
            "command tape and score identify different controllers",
        )
        process = path.parent.parent / "process_exit.json"
        if process.exists():
            status = json.loads(process.read_text())
            require(
                status.get("returncode") == 0 and not status.get("interrupted", False),
                "external process did not exit successfully",
            )
        results.append(result)
        receipts.append(
            dict(
                profile=profile,
                report=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    summary = aggregate_development_bank(results)
    summary.update(actor_sha256=None if shared is None else shared[0], reports=receipts)
    return summary


def score_command_phase(command, velocity, body_yaw, world_yaw, position, heading):
    """Retain the reviewed acquisition/block/drift arithmetic at 50 Hz."""
    th, dt = Thresholds(), PERIOD_S
    cmd = np.asarray(command)
    length = len(velocity)
    block, settling = round(th.block_s / dt), round(th.settling_s / dt)
    if (
        length < settling
        or velocity.shape != (length, 2)
        or body_yaw.shape != (length,)
        or world_yaw.shape != (length,)
        or position.shape != (length + 1, 2)
        or heading.shape != (length + 1,)
        or cmd.shape != (3,)
        or not all(
            np.isfinite(a).all()
            for a in (cmd, velocity, body_yaw, world_yaw, position, heading)
        )
    ):
        raise ValueError(
            "Command scoring requires a complete finite phase and onset pose"
        )
    planar_error = np.linalg.norm(velocity - cmd[:2], axis=-1)
    yaw_error = np.maximum(np.abs(world_yaw - cmd[2]), np.abs(body_yaw - cmd[2]))
    stationary = np.linalg.norm(cmd[:2]) == 0
    planar_limit = th.stationary_speed_m_s if stationary else th.planar_error_m_s
    yaw_limit = th.pivot_yaw_error_rad_s if stationary else th.moving_yaw_error_rad_s
    blocks = [
        {
            "start_s": offset * dt,
            "duration_s": (min(offset + block, length) - offset) * dt,
            "planar_error_m_s": float(planar_error[offset : offset + block].mean()),
            "yaw_error_rad_s": float(yaw_error[offset : offset + block].mean()),
        }
        for offset in range(settling - block, length, block)
    ]
    failed = [
        i
        for i, row in enumerate(blocks)
        if row["planar_error_m_s"] > planar_limit or row["yaw_error_rad_s"] > yaw_limit
    ]
    failures = ["tracking not sustained after 1 s"] if failed else []
    result = dict(
        command=cmd.tolist(),
        blocks=blocks,
        failed_tracking_blocks=failed,
        acquisition_failed=0 in failed,
        later_tracking_failed=any(i > 0 for i in failed),
    )
    if cmd[2] != 0:
        wrong_sign = (
            max(
                float(np.mean(rate[settling:] * cmd[2] < 0))
                for rate in (world_yaw, body_yaw)
            )
            if length > settling
            else None
        )
        result["wrong_sign_fraction"] = wrong_sign
        if wrong_sign is None or wrong_sign > th.wrong_sign_fraction:
            failures.append("missing or excessive wrong-sign yaw")
    if stationary:
        excursion = float(np.linalg.norm(position[1:] - position[0], axis=-1).max())
        drift = (
            max(
                float(
                    np.linalg.norm(
                        position[offset : offset + round(2 / dt) + 1]
                        - position[offset],
                        axis=-1,
                    ).max()
                )
                for offset in range(settling + 1, len(position))
            )
            if length > settling
            else None
        )
        result.update(onset_excursion_m=excursion, maximum_settled_drift_2s_m=drift)
        if excursion > th.stationary_onset_excursion_m or (
            drift is not None and drift > th.stationary_drift_2s_m
        ):
            failures.append("excessive stationary drift/braking distance")
        if cmd[2] == 0:
            excursion = float(np.abs(heading[1:] - heading[0]).max())
            result["heading_onset_excursion_rad"] = excursion
            if excursion > th.zero_twist_heading_excursion_rad:
                failures.append("excessive zero-twist heading drift")
    result.update(failures=failures, kinematic_passed=not failures)
    return result


def _angles(quaternion):
    w, x, y, z = quaternion.T
    return (
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
        np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
        np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))),
    )


def score_flat_profile(trace, profile):
    """Score each original row once; all later auto-reset episodes are excluded.

    Missing/malformed arrays reject the evidence. Nonfinite first-attempt state
    fails that row. No per-group acceptance rates or dynamics strata are inferred
    from this diagnostic's existing startup distribution.
    """
    schedule = np.asarray(profile_commands(profile))
    origins = np.asarray(trace["env_origins"])
    if (
        origins.ndim != 2
        or origins.shape[1] != 3
        or not len(origins)
        or not np.isfinite(origins).all()
    ):
        raise ValueError("Require finite environment origins")
    count = len(origins)
    shapes = {
        key: (STEPS, count, width)
        for key, width in {
            "command": 3,
            "position_w": 3,
            "quaternion_w": 4,
            "linear_velocity_b": 3,
            "angular_velocity_b": 3,
            "angular_velocity_w": 3,
            "base_contact_force_w": 3,
        }.items()
    }
    shapes.update(
        initial_position_w=(count, 3),
        initial_quaternion_w=(count, 4),
        terminated=(STEPS, count),
        truncated=(STEPS, count),
    )
    for key, shape in shapes.items():
        if key not in trace or trace[key].shape != shape:
            raise ValueError(f"Incomplete motion trace: {key} must have shape {shape}")
    body = trace.get("body_position_w")
    if (
        body is None
        or body.ndim != 4
        or body.shape[:2] != (STEPS, count)
        or not body.shape[2]
        or body.shape[3] != 3
    ):
        raise ValueError("Require all native body-link positions")
    if any(trace[key].dtype != np.bool_ for key in ("terminated", "truncated")):
        raise ValueError("Termination flags must be boolean")
    if not np.allclose(trace["command"], schedule[:, None], rtol=0, atol=1e-6):
        raise ValueError("Executed commands differ from the flat profile")
    # Merge adjacent identical commands, particularly the single 11 s stand.
    boundaries = [
        0,
        *(np.flatnonzero(np.any(np.diff(schedule, axis=0), axis=1)) + 1),
        STEPS,
    ]
    results = []
    for row in range(count):
        ends = np.flatnonzero(trace["terminated"][:, row] | trace["truncated"][:, row])
        length = int(ends[0] + 1) if len(ends) else STEPS
        failures, phases = [], []
        if trace["terminated"][:length, row].any():
            failures.append("native termination")
        if trace["truncated"][:length, row].any():
            failures.append("unexpected timeout")
        position = np.vstack(
            (trace["initial_position_w"][row], trace["position_w"][:length, row])
        ).astype(np.float64)
        quaternion = np.vstack(
            (trace["initial_quaternion_w"][row], trace["quaternion_w"][:length, row])
        ).astype(np.float64)
        finite = all(
            np.isfinite(trace[key][:length, row]).all()
            for key in shapes
            if key not in ("initial_position_w", "initial_quaternion_w")
        )
        finite = (
            finite
            and np.isfinite(position).all()
            and np.isfinite(quaternion).all()
            and np.isfinite(body[:length, row]).all()
        )
        if not finite:
            failures.append("nonfinite first-attempt state")
        elif not np.allclose(
            np.linalg.norm(quaternion, axis=-1), 1.0, atol=1e-3, rtol=0
        ):
            failures.append("invalid quaternion")
        else:
            roll, pitch, heading = _angles(quaternion)
            if (
                max(np.abs(roll).max(), np.abs(pitch).max())
                > Thresholds().flat_attitude_rad
            ):
                failures.append("flat attitude exceeds 15 degrees")
            if (np.abs(position[:, :2] - origins[row, :2]) > 6).any() or (
                np.abs(body[:length, row, :, :2] - origins[row, :2]) > 6
            ).any():
                failures.append("body link leaves flat domain")
            forces = np.linalg.norm(
                trace["base_contact_force_w"][:length, row], axis=-1
            )
            if (
                length >= 5
                and (
                    np.convolve(
                        (forces > 20).astype(int), np.ones(5, dtype=int), mode="valid"
                    )
                    >= 5
                ).any()
            ):
                failures.append("sustained base contact")
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                if end > length:
                    phases.append(dict(start_s=start * PERIOD_S, complete=False))
                    failures.append("incomplete command phase")
                    continue
                command = schedule[start]
                phase = score_command_phase(
                    command,
                    trace["linear_velocity_b"][start:end, row, :2],
                    trace["angular_velocity_b"][start:end, row, 2],
                    trace["angular_velocity_w"][start:end, row, 2],
                    position[start : end + 1, :2],
                    heading[start : end + 1],
                )
                # Full 3 s displacement windows after 1 s; never along a fixed axis.
                stalled = np.linalg.norm(command[:2]) >= 0.2 and any(
                    np.linalg.norm(position[k + 150, :2] - position[k, :2]) < 0.10
                    for k in range(start + 50, end - 150 + 1)
                )
                if stalled:
                    phase["failures"].append("stall")
                    phase["kinematic_passed"] = False
                phase.update(start_s=start * PERIOD_S, complete=True)
                failures.extend(
                    f"phase {start * PERIOD_S:g}s: {reason}"
                    for reason in phase["failures"]
                )
                phases.append(phase)
        results.append(
            dict(
                env_id=row,
                first_attempt_steps=length,
                diagnostic_passed=not failures,
                failures=failures,
                phases=phases,
            )
        )
    return dict(
        version="flat_command_diagnostic_v1",
        group_id=f"flat/{profile}",
        namespace="parkour-lab/flat-diagnostic/v1",
        qualified=False,
        qualification_eligible=False,
        thresholds=asdict(Thresholds()),
        scope="First-attempt kinematics under the recorded task distribution, not the independent noisy nominal/randomized acceptance bank. Map checks use root/body-link centres, not collider extents.",
        passed=sum(row["diagnostic_passed"] for row in results),
        total=count,
        trials=results,
    )
