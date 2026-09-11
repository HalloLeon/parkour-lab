"""Read saved training settings and verify file identity without loading a policy.

Run this on the machine where the run directory is mounted. Paths in an attached
metrics.json are provenance labels, never files to follow or remote commands.
Only PyYAML is required; Python YAML tags are inspected as inert scalar/container
data with BaseLoader. Checkpoints are hashed, never deserialized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path


UNKNOWN = "UNKNOWN (not saved)"
LIMITATIONS = (
    "Saved settings describe configuration, not proof of executed updates, command exposure, "
    "resume, optimizer state, or effective learning-rate/entropy schedules.",
    "Checkpoint bytes are hashed only; checkpoint-internal provenance/iteration/optimizer "
    "state is UNKNOWN. A matching hash establishes file identity, not training correctness.",
    "Evaluation reward_config is not training evidence and is deliberately ignored. "
    "Missing settings are not replaced with today's code defaults.",
    "Optional provenance sidecars are reported, not independently authenticated. "
    "DR stage and resolved reset ranges are separate settings; stage off can coexist with jitter.",
)

# Values are read from the saved files, not imported from current simulator code.
FIELDS = {
    "env.yaml": {
        "rewards": (
            "rewards.stationary_velocity_tracking.params.pivot_yaw_objective",
            "rewards.stationary_velocity_tracking.params.pivot_yaw_tracking_weight",
            "rewards.stationary_velocity_tracking.params.pivot_yaw_overspeed_weight",
            "rewards.stationary_velocity_tracking.params.pivot_stability_weight",
            "rewards.stationary_velocity_tracking.weight",
            "rewards.stationary_planar_motion.weight",
            "rewards.stationary_planar_motion.params.transition_speed_m_s",
            "rewards.flat_orientation_l2.weight",
            "rewards.stable_orientation_l2.weight",
            "rewards.lin_vel_z_l2.weight",
            "rewards.upright_orientation_l2.weight",
            "rewards.supported_orientation_l2.weight",
            "rewards.supported_vertical_velocity_l2.weight",
            "rewards.action_target_overflow_l2.weight",
            "rewards.action_target_overflow_l2.params.normalization_rad",
            "rewards.action_target_overflow_l2.params.action_term_name",
        ),
        "commands": (
            "commands.intent.command_profile",
            "commands.intent.external_control",
            "commands.intent.pivot_window_probability",
            "commands.intent.pivot_window_range_s",
            "commands.intent.pivot_abs_yaw_rate_range_rad_s",
            "commands.intent.stop_window_probability",
            "commands.intent.resampling_time_range",
            "commands.intent.long_stop_probability",
            "commands.intent.long_stop_window_range_s",
        ),
        "reset_and_dr": (
            "domain_randomization.stage",
            "events.reset_base.params.pose_range",
            "events.reset_base.params.velocity_range",
        ),
        "training_environment": (
            "scene.num_envs",
            "sim.dt",
            "decimation",
            "episode_length_s",
            "actions.joint_pos.max_delay_steps",
        ),
    },
    "agent.yaml": {
        "training": (
            "algorithm.history_rollout_interval",
            "algorithm.learning_rate",
            "algorithm.max_learning_rate",
            "algorithm.schedule",
            "algorithm.entropy_coef",
            "algorithm.entropy_coef_end",
            "algorithm.exploration_warmup_iterations",
            "algorithm.exploration_anneal_iterations",
            "policy.min_noise_std",
            "policy.max_noise_std",
            "policy.use_velocity_estimator",
            "num_steps_per_env",
            "max_iterations",
            "seed",
            "resume",
            "load_run",
            "load_checkpoint",
        ),
    },
}
SIDECARS = (
    "git/provenance.json",
    "params/resume.json",
    "params/warm_start.json",
    "params/curriculum_restart.json",
)


def _mapping_pairs(pairs):
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError(f"Duplicate or non-string mapping key: {key!r}")
        result[key] = value
    return result


def _validate_tree(value, ancestors=(), depth=0):
    """Reject cycles and excessive nesting before formatting inert YAML data."""
    if depth > 64 or id(value) in ancestors:
        raise ValueError("Recursive or excessively nested document")
    if isinstance(value, (dict, list)):
        children = value.values() if isinstance(value, dict) else value
        for child in children:
            _validate_tree(child, (*ancestors, id(value)), depth + 1)


def parse_document(raw: bytes, *, yaml_document: bool) -> dict:
    text = raw.decode("utf-8")
    if yaml_document:
        try:
            import yaml
        except ImportError as error:
            raise ValueError(
                "PyYAML is required (install pyyaml in this Python environment)"
            ) from error

        class ScalarLoader(yaml.BaseLoader):
            def construct_mapping(self, node, deep=False):
                return _mapping_pairs(
                    (
                        self.construct_object(key, deep=deep),
                        self.construct_object(value, deep=deep),
                    )
                    for key, value in node.value
                )

        try:
            document = yaml.load(text, Loader=ScalarLoader)
        except yaml.YAMLError as error:
            raise ValueError(f"Malformed YAML: {error}") from error
    else:

        def reject_constant(value):
            raise ValueError(f"Invalid JSON numeric constant: {value}")

        document = json.loads(
            text, object_pairs_hook=_mapping_pairs, parse_constant=reject_constant
        )
    if not isinstance(document, dict) or not document:
        raise ValueError("Expected a nonempty mapping document")
    _validate_tree(document)
    return document


def read_evidence(path: Path, *, parse: bool = False) -> tuple[dict, dict | None]:
    """Hash actual bytes; optional parsing uses those same bytes, never rereads."""
    record = {"path": str(path), "status": "MISSING", "sha256": None}
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if parse:
                raw = stream.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError("Metadata exceeds the 16 MiB inspection limit")
                digest = hashlib.sha256(raw).hexdigest()
            else:
                digest_state = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest_state.update(block)
                digest = digest_state.hexdigest()
            after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("File changed during inspection")
        record.update(status="READ", sha256=digest)
        document = (
            parse_document(raw, yaml_document=path.suffix in (".yaml", ".yml"))
            if parse
            else None
        )
        return record, document
    except FileNotFoundError:
        return record, None
    except (OSError, ValueError, RecursionError) as error:
        record.update(status="ERROR", error=str(error))
        return record, None


def lookup(document, path):
    value = document
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return UNKNOWN
        value = value[key]
    return value


def reset_profile_hint(env):
    """Label recognized resolved ranges, without claiming a saved CLI argument."""
    pose = lookup(env, "events.reset_base.params.pose_range")
    velocity = lookup(env, "events.reset_base.params.velocity_range")
    if not isinstance(pose, dict) or not isinstance(velocity, dict):
        return UNKNOWN

    def match(mapping, limits):
        if set(mapping) != set(limits):
            return False
        try:
            return all(
                isinstance(mapping[key], list)
                and len(mapping[key]) == 2
                and all(
                    math.isfinite(float(actual))
                    and abs(float(actual) - expected) < 1e-9
                    for actual, expected in zip(mapping[key], (-limit, limit))
                )
                for key, limit in limits.items()
            )
        except (TypeError, ValueError):
            return False

    for label, scale in (("canonical", 0.0), ("jitter", 1.0)):
        if match(
            pose, {"x": 0.025 * scale, "y": 0.025 * scale, "yaw": 0.05 * scale}
        ) and match(
            velocity,
            {key: 0.1 * scale for key in ("x", "y", "z", "roll", "pitch", "yaw")},
        ):
            return f"{label}-equivalent resolved ranges (inferred; CLI profile not recorded)"
    return "custom resolved ranges (CLI profile UNKNOWN)"


def build_report(
    target: Path,
    *,
    checkpoint: Path | None = None,
    metrics: Path | None = None,
    verify: bool = False,
) -> dict:
    target = target.expanduser().resolve()
    if target.suffix == ".pt" or target.is_file():
        if checkpoint is not None:
            raise ValueError(
                "Use either a checkpoint target or --checkpoint with a run directory"
            )
        checkpoint, run = target, target.parent
    else:
        run = target
        if checkpoint is not None:
            checkpoint = checkpoint.expanduser()
            checkpoint = (
                (run / checkpoint).resolve()
                if not checkpoint.is_absolute()
                else checkpoint.resolve()
            )
    strict = verify or metrics is not None
    report = {
        "run": str(run),
        "files": {},
        "settings": {},
        "sidecars": {},
        "errors": [],
        "verification": "NOT_REQUESTED",
        "limitations": list(LIMITATIONS),
    }
    documents = {}
    for name, path in (
        ("env.yaml", run / "params/env.yaml"),
        ("agent.yaml", run / "params/agent.yaml"),
        ("checkpoint", checkpoint),
    ):
        record, document = (
            read_evidence(path, parse=name != "checkpoint")
            if path
            else ({"path": None, "status": "MISSING", "sha256": None}, None)
        )
        report["files"][name] = record
        documents[name] = document
        if record["status"] == "ERROR" or (strict and record["status"] == "MISSING"):
            report["errors"].append(
                f"{name}: {record.get('error', 'MISSING (select an existing actual file)')}"
            )
    for name, sections in FIELDS.items():
        for section, paths in sections.items():
            report["settings"][section] = {
                path: lookup(documents[name], path) for path in paths
            }
    report["settings"]["reset_and_dr"]["reset_profile_hint"] = reset_profile_hint(
        documents["env.yaml"]
    )
    for name in SIDECARS:
        record, document = read_evidence(run / name, parse=True)
        report["sidecars"][name] = {**record, "recorded_metadata": document}
        if record["status"] == "ERROR":
            report["errors"].append(f"{name}: {record['error']}")
    if metrics is not None:
        record, document = read_evidence(metrics.expanduser().resolve(), parse=True)
        report["files"]["metrics"] = record
        if record["status"] != "READ":
            report["errors"].append(f"metrics: {record.get('error', 'MISSING')}")
        for name in ("env.yaml", "agent.yaml", "checkpoint"):
            expected = (
                (document or {}).get("checkpoint_sha256")
                if name == "checkpoint"
                else (
                    ((document or {}).get("training_config") or {}).get(name, {})
                    if isinstance((document or {}).get("training_config"), dict)
                    else {}
                )
            )
            if name != "checkpoint":
                expected = (
                    expected.get("sha256") if isinstance(expected, dict) else None
                )
            actual = report["files"][name]
            actual["expected_sha256"] = expected
            if not isinstance(expected, str) or not re.fullmatch(
                r"[a-fA-F0-9]{64}", expected
            ):
                actual["verification"] = "MISSING_OR_MALFORMED_EXPECTED_HASH"
            elif actual["sha256"] is None:
                actual["verification"] = "UNAVAILABLE"
            else:
                actual["verification"] = (
                    "MATCH" if actual["sha256"] == expected.lower() else "MISMATCH"
                )
            if actual["verification"] != "MATCH":
                report["errors"].append(f"{name}: {actual['verification']}")
    if strict:
        report["verification"] = (
            "FAIL"
            if report["errors"]
            else ("HASHES_MATCH" if metrics else "INPUTS_READABLE")
        )
    return report


def print_report(report):
    print(f"Saved training recipe: {report['run']}")
    for name, record in report["files"].items():
        print(
            f"{name}: {record['status']} {record.get('verification', '')}  sha256={record['sha256'] or 'UNKNOWN'}"
        )
        print(f"  actual path: {record['path'] or 'UNKNOWN (select --checkpoint)'}")
        if record.get("expected_sha256") is not None:
            print(f"  expected sha256: {record['expected_sha256']}")
    for section, values in report["settings"].items():
        print(f"\n{section} (saved configuration only):")
        for name, value in values.items():
            print(
                f"  {name}: {value if isinstance(value, str) else json.dumps(value, sort_keys=True)}"
            )
    print("\nProvenance/resume sidecars:")
    for name, record in report["sidecars"].items():
        print(f"  {name}: {record['status']} sha256={record['sha256'] or 'UNKNOWN'}")
        metadata = record["recorded_metadata"]
        if name == "git/provenance.json" and metadata:
            metadata = {
                key: metadata.get(key, UNKNOWN)
                for key in (
                    "commit_sha",
                    "branch",
                    "dirty",
                    "captured_at_utc",
                    "diff_sha256",
                )
            }
        if metadata:
            print(f"    recorded metadata: {json.dumps(metadata, sort_keys=True)}")
    print(f"\nVerification: {report['verification']}")
    for error in report["errors"]:
        print(f"ERROR: {error}")
    for limitation in report["limitations"]:
        print(f"LIMIT: {limitation}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        type=Path,
        help="Run directory or exact checkpoint .pt path (local/mounted, including HPC paths)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Exact checkpoint path or filename relative to the run; never auto-selects latest",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        help="Verify actual env/agent/checkpoint hashes against this metrics.json; paths may differ",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Require readable env.yaml, agent.yaml and selected checkpoint; missing optional sidecars stay UNKNOWN",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the report as JSON to stdout; does not write files",
    )
    args = parser.parse_args(argv)
    try:
        report = build_report(
            args.target,
            checkpoint=args.checkpoint,
            metrics=args.metrics,
            verify=args.verify,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        print_report(report)
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
