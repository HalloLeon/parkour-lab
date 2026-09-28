"""CPU-only audit of sampled ROA training telemetry; never starts simulation.

Usage: python -m scripts.analysis.operator_training_credit RUN [RUN ...]
Prints JSON. Samples are selected windows, not independent trials, whole-run
averages, supported obstacle completions or per-terrain gradient attribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_arrays(a):
    """Independently reconstruct the recorded reward/GAE/Gaussian arithmetic."""
    meta = json.loads(str(a["metadata_json"]))
    if meta["schema_version"] != "operator_roa_training_telemetry_v1":
        raise ValueError("Unsupported training telemetry")
    if any(not np.isfinite(v).all() for k, v in a.items() if k != "metadata_json"):
        raise ValueError("Nonfinite telemetry")
    h, steps = meta["history_interval"], meta["rollout_steps"]
    n = a["column_id"].size
    total = h * steps + meta["history_steps"]
    if a["reward_total"].shape != (total, n):
        raise ValueError("Incomplete transition block")
    idx = a["ppo_sample_index"]
    np.testing.assert_array_equal(idx, np.arange(h * steps).reshape(h, steps))
    np.testing.assert_array_equal(
        a["phase"], [0] * (h * steps) + [1] * meta["history_steps"]
    )
    np.testing.assert_array_equal(np.diff(a["native_index"]), 1)
    np.testing.assert_array_equal(np.diff(a["ppo_cycle"]), 1)
    np.testing.assert_array_equal(
        a["cycle"][idx], np.repeat(a["ppo_cycle"][:, None], steps, axis=1)
    )
    np.testing.assert_array_equal(a["cycle"][h * steps :], a["ppo_cycle"][-1])
    np.testing.assert_array_equal(a["ground_height_pre"][~a["ground_valid_pre"]], 0)
    np.testing.assert_array_equal(
        a["bootstrap_time_out"], a["timed_out"] & ~a["terminated"]
    )
    errors = {}

    def close(name, observed, expected, atol=2e-6, rtol=1e-5):
        np.testing.assert_allclose(
            observed, expected, atol=atol, rtol=rtol, err_msg=name
        )
        errors[name] = float(np.max(np.abs(observed.astype(float) - expected)))

    close("native_reward", a["reward_total"], a["reward_contribution"].sum(-1))
    v, reward = a["ppo_value"][..., 0], a["ppo_reward"][..., 0]
    gamma, lam = meta["ppo"]["gamma"], meta["ppo"]["gae_lambda"]
    close(
        "timeout_reward",
        reward,
        a["reward_total"][idx] + gamma * v * a["bootstrap_time_out"][idx],
    )
    done = a["terminated"][idx] | a["timed_out"][idx]
    advantage, returns = np.zeros_like(v[:, 0]), np.empty_like(v)
    for t in reversed(range(steps)):
        next_v = a["ppo_boundary_value"][..., 0] if t == steps - 1 else v[:, t + 1]
        keep = 1 - done[:, t].astype(np.float32)
        delta = reward[:, t] + keep * gamma * next_v - v[:, t]
        advantage = delta + keep * gamma * lam * advantage
        returns[:, t] = advantage + v[:, t]
    close("gae_return", a["ppo_return"][..., 0], returns)
    raw = a["ppo_return"][..., 0] - v
    close("raw_advantage", a["ppo_advantage_raw"][..., 0], raw)
    norm = (raw - raw.mean(axis=(1, 2), keepdims=True)) / (
        raw.std(axis=(1, 2), ddof=1, keepdims=True) + 1e-8
    )
    close(
        "normalized_advantage", a["ppo_advantage_normalized"][..., 0], norm, atol=1e-5
    )
    for when in ("old", "new"):
        mean, std = a[f"ppo_{when}_mean"], a[f"ppo_{when}_std"]
        if (std <= 0).any():
            raise ValueError("Nonpositive Gaussian scale")
        logp = (
            -0.5 * ((a["raw_action"][idx] - mean) / std) ** 2
            - np.log(std)
            - 0.5 * np.log(2 * np.pi)
        ).sum(-1)
        close(f"{when}_log_prob", a[f"ppo_{when}_log_prob"][..., 0], logp, atol=1e-5)
    old, new = a["ppo_old_mean"].astype(float), a["ppo_new_mean"].astype(float)
    os, ns = a["ppo_old_std"].astype(float), a["ppo_new_std"].astype(float)
    kl = (np.log(ns / os) + (os**2 + (old - new) ** 2) / (2 * ns**2) - 0.5).sum(-1)
    close("kl", a["ppo_kl_old_new"], kl, atol=1e-13, rtol=1e-12)
    ratio = np.exp(
        a["ppo_new_log_prob"].astype(float)[..., 0]
        - a["ppo_old_log_prob"].astype(float)[..., 0]
    )
    close("ratio", a["ppo_ratio"], ratio, atol=1e-13, rtol=1e-12)
    np.testing.assert_array_equal(
        a["ppo_ratio_outside_clip"], np.abs(ratio - 1) > meta["ppo"]["clip_param"]
    )
    return errors


def masks(a, geometry):
    """Terrain/command/motion groups; center-ray height is NOT foot support."""
    command, root = a["command_b_pre"], a["root_local_pre"]
    shape = command.shape[:2]
    moving = np.any(command[..., :2] != 0, axis=-1)
    stopped = ~np.any(command != 0, axis=-1)
    speed = np.linalg.norm(a["velocity_b_pre"][..., :2], axis=-1)
    groups = {"all": np.ones(shape, dtype=bool)}
    profiles = json.loads(str(a["metadata_json"]))["profiles"]
    for i, name in enumerate(profiles):
        profile = np.broadcast_to(a["profile_id"] == i, shape)
        groups[name] = profile
        for regime, selected in (
            ("moving_command", moving),
            ("stopped_command", stopped),
            ("pivot_command", ~moving & ~stopped),
        ):
            groups[f"{name}/{regime}"] = profile & selected
        # Preserve the assigned difficulty row, not just pooled terrain means.
        for level in np.unique(a["level_id"]):
            groups[f"{name}/difficulty_row{level}/moving_command"] = (
                profile & (a["level_id"] == level) & moving
            )
    tiles = {(t["row"], t["variant"]): t for t in geometry["tiles"]}
    rise, rough = np.ones(shape[1]), np.zeros(shape[1])
    for j in np.flatnonzero(a["profile_id"] == profiles.index("step_hills")):
        tile = tiles[int(a["level_id"][j]), int(a["column_id"][j])]
        rise[j], rough[j] = tile["riser_height_m"], tile["roughness_absolute_bound_m"]
    height = a["ground_height_pre"]
    level = np.rint(height / rise).astype(int)
    valid = (
        groups["step_hills"]
        & a["ground_valid_pre"]
        & (np.abs(root[..., :2]) < 8).all(-1)
    )
    valid &= (
        (level >= 0) & (level <= 2) & (np.abs(height - level * rise) <= rough + 2e-4)
    )
    for label, selected in (("base", level == 0), ("raised", level > 0)):
        selected = valid & selected & moving
        groups[f"step_hills/{label}/moving_measured"] = selected & (speed > 0.05)
        groups[f"step_hills/{label}/slow_measured"] = selected & (speed <= 0.05)
    groups["step_hills/unclassified_ground"] = groups["step_hills"] & ~valid
    return groups


def summarize(a, selected, phase):
    """Conditional observed means, with counts; empty groups are not successes."""
    phase_mask = np.broadcast_to((a["phase"] == phase)[:, None], selected.shape)
    mask = selected & phase_mask
    count = int(mask.sum())
    if not count:
        return {"samples": 0}
    meta = json.loads(str(a["metadata_json"]))
    command = a["command_b_pre"][mask, :2]
    velocity = a["velocity_b_pre"][mask, :2]
    result = dict(
        samples=count,
        command_speed_mean_m_s=float(np.linalg.norm(command, axis=-1).mean()),
        measured_speed_mean_m_s=float(np.linalg.norm(velocity, axis=-1).mean()),
        xy_tracking_error_mean_m_s=float(
            np.linalg.norm(velocity - command, axis=-1).mean()
        ),
        native_reward_mean=float(a["reward_total"][mask].mean()),
        physical_ends=int(a["terminated"][mask].sum()),
        timeouts=int(a["timed_out"][mask].sum()),
        reward_terms_mean=dict(
            zip(
                meta["reward_names"],
                a["reward_contribution"][mask].astype(float).mean(0).tolist(),
                strict=True,
            )
        ),
    )
    if phase == 0:
        pm = mask[a["ppo_sample_index"]]
        for field in (
            "reward",
            "value",
            "return",
            "advantage_raw",
            "advantage_normalized",
            "kl_old_new",
            "ratio_outside_clip",
        ):
            result[f"ppo_{field}_mean"] = float(a[f"ppo_{field}"][pm].mean())
        advantage = a["ppo_advantage_normalized"][..., 0][pm]
        ratio = a["ppo_ratio"][pm]
        clip = meta["ppo"]["clip_param"]
        result["positive_advantage_fraction"] = float((advantage > 0).mean())
        result["post_update_clipped_surrogate_gain"] = float(
            (
                np.minimum(
                    ratio * advantage, ratio.clip(1 - clip, 1 + clip) * advantage
                )
                - advantage
            ).mean()
        )
        result["action_std_mean"] = float(a["ppo_old_std"][pm].mean())
        result["kl_p95"] = float(np.quantile(a["ppo_kl_old_new"][pm], 0.95))
    return result


def audit_run(run):
    run = Path(run).resolve(strict=True)
    report = json.loads((run / "report.json").read_text())
    protocol = json.loads((run / "training_protocol.json").read_text())
    receipt = json.loads((run.parent / "process_exit.json").read_text())
    if (
        receipt["returncode"] != 0
        or receipt["log_returncode"] != 0
        or receipt["interrupted"]
    ):
        raise ValueError("Training/logging did not finish")
    telemetry = report["training_telemetry"]
    if (
        not telemetry["complete"]
        or telemetry["pending_step"]
        or telemetry["incomplete_block_decisions"]
    ):
        raise ValueError("Incomplete telemetry report")
    for key, value in protocol["training_telemetry"].items():
        if telemetry[key] != value:
            raise ValueError(f"Protocol/report telemetry mismatch: {key}")
    entries = telemetry["files"]
    if [item["end_cycle"] for item in entries] != telemetry["block_end_cycles"]:
        raise ValueError("Missing, repeated or reordered block")
    expected_paths = {
        f"training_telemetry/block_{end:06d}.npz"
        for end in telemetry["block_end_cycles"]
    }
    if {
        str(p.relative_to(run)) for p in (run / "training_telemetry").glob("*.npz")
    } != expected_paths:
        raise ValueError("Unexpected or missing NPZ files")
    if report["cleanup"]["environment"] != "complete" or not all(
        report["evaluation_initial_conditions_match"].values()
    ):
        raise ValueError("Incomplete cleanup or unmatched frozen checks")
    blocks, arrays = [], []
    bindings = {
        name: sha256(run / name)
        for name in (
            "report.json",
            "training_protocol.json",
            "resolved_env.yaml",
            "git/provenance.json",
            "git/parkour_lab.diff",
        )
    }
    bindings["../process_exit.json"] = sha256(run.parent / "process_exit.json")
    for entry in entries:
        relative = f"training_telemetry/block_{entry['end_cycle']:06d}.npz"
        if entry["path"] != relative or sha256(run / relative) != entry["sha256"]:
            raise ValueError("NPZ name or hash mismatch")
        bindings[relative] = entry["sha256"]
        with np.load(run / relative, allow_pickle=False) as loaded:
            a = {key: loaded[key] for key in loaded.files}
        meta = json.loads(str(a["metadata_json"]))
        if any(telemetry[k] != v for k, v in meta.items()):
            raise ValueError("NPZ metadata differs from report")
        errors = audit_arrays(a)
        end = entry["end_cycle"]
        h, steps, hs = (
            meta["history_interval"],
            meta["rollout_steps"],
            meta["history_steps"],
        )
        expected_start = 900 + (end - h) * steps + (end // h - 1) * hs
        np.testing.assert_array_equal(
            a["native_index"],
            np.arange(expected_start, expected_start + h * steps + hs),
        )
        np.testing.assert_array_equal(a["ppo_cycle"], np.arange(end - h + 1, end + 1))
        for key, expected in (
            ("column_id", report["training_exposure"]["column_ids"]),
            ("level_id", report["training_exposure"]["level_ids"]),
        ):
            np.testing.assert_array_equal(a[key], expected)
        np.testing.assert_array_equal(a["profile_id"], a["column_id"] // 4)
        if (
            a["reward_total"].shape != (entry["decisions"], protocol["num_envs"])
            or a["reward_total"].size != entry["transition_rows"]
        ):
            raise ValueError("NPZ counts differ from receipt")
        groups = masks(a, report["native_step_field_geometry"])
        chosen = (
            "all",
            "step_hills/moving_command",
            "step_hills/raised/moving_measured",
        )
        blocks.append(
            dict(
                end_cycle=end,
                arithmetic_max_abs_errors=errors,
                ppo={k: summarize(a, groups[k], 0) for k in chosen},
                history={k: summarize(a, groups[k], 1) for k in chosen},
            )
        )
        arrays.append(a)
    # Concatenation preserves every selected block equally; not a run-wide estimate.
    static = {"column_id", "level_id", "profile_id", "metadata_json"}
    joined = {
        k: (
            arrays[0][k]
            if k in static
            else np.concatenate([a[k] for a in arrays], axis=0)
        )
        for k in arrays[0]
    }
    joined["ppo_sample_index"] = np.concatenate(
        [
            a["ppo_sample_index"] + i * telemetry["decisions_per_block"]
            for i, a in enumerate(arrays)
        ],
        axis=0,
    )
    groups = masks(joined, report["native_step_field_geometry"])
    return dict(
        run=str(run),
        bindings=bindings,
        wall_seconds=receipt["wall_seconds"],
        cumulative_ppo_updates=report["cumulative_ppo_updates"],
        environment_transitions=report["environment_transitions"],
        blocks=blocks,
        pooled_sampled_windows={
            label: {k: summarize(joined, v, phase) for k, v in groups.items()}
            for phase, label in ((0, "ppo"), (1, "history"))
        },
        frozen_checks={
            key: dict(
                physical_ends=report[key]["first_episode"]["terminated"],
                tracking=report[key]["tracking"]["first_episode"]["by_regime"],
            )
            for key in ("evaluation_before", "evaluation_after")
        },
        caveats=[
            "Only selected windows; no whole-run sampling guarantee or independent-trial confidence intervals.",
            "Center-ground ray is not verified foot support or climb completion.",
            "Command, speed, difficulty and survival distributions differ between groups and runs.",
            "Negative average advantage is not evidence that every action in that group is discouraged.",
            "Post-update Gaussian ratios/KL are not minibatch clipping decisions or per-terrain gradients.",
            "Arithmetic/hash audit, not full checkpoint ancestry admission, GPU parity or deployment qualification.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Exclusively create an audit JSON; never overwrite evidence",
    )
    args = parser.parse_args()
    result = {
        "schema": "operator_training_credit_audit_v1",
        "auditor_sha256": sha256(__file__),
        "runs": [audit_run(run) for run in args.runs],
    }
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(encoded + "\n")
        print(args.output)
    else:
        print(encoded)


if __name__ == "__main__":
    main()
