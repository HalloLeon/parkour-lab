"""Portable capture persistence and offline scoring, without a simulator import."""

import hashlib
import json
from pathlib import Path

import numpy as np

from parkour_lab.provenance import file_sha256, write_json

VERSION = "go2_evaluation_capture_v1"
IDENTITY_FIELDS = (
    "policy_state_sha256",
    "training_seed",
    "checkpoint_id",
    "checkpoint_sha256",
)
STATE_FIELDS = {
    "position_w": "root_position_w",
    "quaternion_w": "root_quaternion_w",
    "linear_velocity_b": "root_com_linear_velocity_b",
    "angular_velocity_b": "root_angular_velocity_b",
    "angular_velocity_w": "root_angular_velocity_w",
    "foot_position_w": "foot_position_w",
    "foot_net_forces_w": "foot_contact_forces_w",
    "support_height_w": "support_height_w",
    "support_valid": "support_valid",
}


def reward_identity(recipe):
    encoded = json.dumps(recipe, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _check_receipt(manifest, evidence, case):
    if evidence.get("version") != VERSION or evidence.get("case") != case.as_dict():
        raise ValueError("Case receipt differs from the prescribed manifest")
    if any(evidence.get(key) != manifest[key] for key in IDENTITY_FIELDS):
        raise ValueError("Case policy/checkpoint identity differs from the manifest")
    recipe = evidence.get("realization", {}).get("task", {}).get("reward_recipe")
    if recipe is not None and (
        reward_identity(recipe) != evidence.get("core_reward_recipe_sha256")
        or evidence["core_reward_recipe_sha256"]
        != manifest.get("core_reward_recipe_sha256")
    ):
        raise ValueError("Case core reward recipe differs from the manifest")


def _capture_valid(evidence, result):
    """Admission is separate from computing diagnostics for partial/error traces."""
    count = result["observed_steps"]
    interactions = evidence.get("interactions", {})
    cleanup = evidence.get("cleanup")
    accounted = (
        type(evidence.get("observed_steps")) is int
        and evidence["observed_steps"] == count
        and all(
            type(interactions.get(key)) is int
            for key in ("session", "cumulative", "uncertain")
        )
        and interactions.get("session") == count
        and interactions.get("cumulative") == count
        and interactions.get("uncertain") == 0
    )
    return (
        evidence["status"] == "CAPTURED"
        and isinstance(cleanup, dict)
        and cleanup.get("environment") == "complete"
        and accounted
        and result["core_return"] is not None
        and evidence.get("realization", {}).get("task", {}).get("reward_recipe")
        is not None
        and evidence.get("core_reward_recipe_sha256") is not None
    )


def state_arrays(state):
    """Own the single outgoing native row before any subsequent buffer reuse."""
    return {
        STATE_FIELDS.get(key, key): value[0].detach().cpu().numpy().copy()
        for key, value in state.items()
        if key in STATE_FIELDS or key.startswith("ending_")
    }


def save_trace(folder, rows, initial, origin):
    """Store only physically completed transitions, including an ending once."""
    arrays = {
        **{
            key: (
                np.stack([row[key] for row in rows])
                if rows
                else np.empty((0, *value.shape), dtype=value.dtype)
            )
            for key, value in initial.items()
            if not key.startswith("ending_")
        },
        **{f"initial_{key}": value for key, value in initial.items()},
        "env_origin_w": np.asarray(origin),
    }
    if rows:
        arrays.update({key: np.stack([row[key] for row in rows]) for key in rows[0]})
    else:
        arrays.update(
            decision_time_s=np.empty(0),
            command=np.empty((0, 3)),
            reward=np.empty(0),
            terminated=np.empty(0, dtype=bool),
            truncated=np.empty(0, dtype=bool),
        )
    path = folder / "trace.npz"
    np.savez_compressed(path, **arrays)
    return file_sha256(path)


def score_capture(folder):
    """Recompute from captured measurements, retaining missing prescribed cases."""
    from .cases import case_from_dict
    from .scoring import aggregate_cases, score_case

    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("version") != VERSION:
        raise ValueError("Unsupported evaluation capture version")
    cases = tuple(case_from_dict(item) for item in manifest["cases"])
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("Require distinct prescribed evaluation cases")
    results = {}
    for case in cases:
        path = folder / case.case_id
        receipt = path / "capture.json"
        if not receipt.exists():
            continue
        evidence = json.loads(receipt.read_text())
        _check_receipt(manifest, evidence, case)
        digest = evidence.get("trace_sha256")
        if digest is None:
            continue
        trace = path / "trace.npz"
        if file_sha256(trace) != digest:
            raise ValueError("Trace digest differs from its capture receipt")
        with np.load(trace, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        result = score_case(case, arrays)
        result["capture_status"] = evidence["status"]
        result["capture_error"] = evidence.get("error")
        result["trace_sha256"] = digest
        result["capture_valid"] = _capture_valid(evidence, result)
        result["cleanup"] = evidence.get("cleanup")
        results[case.case_id] = result
    aggregate = aggregate_cases(cases, results.values())
    capture_valid = (
        manifest["status"] == "COMPLETE"
        and manifest.get("final_policy_state_sha256") == manifest["policy_state_sha256"]
        and len(results) == len(cases)
        and all(result["capture_valid"] for result in results.values())
        and aggregate["complete"]
    )
    return {
        "version": VERSION,
        "capture_status": manifest["status"],
        "training_seed": manifest["training_seed"],
        "checkpoint_id": manifest["checkpoint_id"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "policy_state_sha256": manifest["policy_state_sha256"],
        "core_reward_recipe_sha256": manifest.get("core_reward_recipe_sha256"),
        "final_policy_state_sha256": manifest.get("final_policy_state_sha256"),
        "capture_valid": capture_valid,
        "valid_for_comparison": capture_valid and aggregate["full_partition_bank"],
        "cases": [results.get(case.case_id) for case in cases],
        "aggregate": aggregate,
    }


def publish_capture(folder, manifest):
    write_json(folder / "manifest.json", manifest)
    report = score_capture(folder)
    write_json(folder / "report.json", report)
    return report
