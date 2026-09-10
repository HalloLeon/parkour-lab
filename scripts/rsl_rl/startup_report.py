# SPDX-License-Identifier: BSD-3-Clause
"""Write and compare bounded startup traces without Isaac Sim or PyTorch.

MATCH means initial actor-input/action parity only, not successful traversal or
operator reliability. A trace ending at its step limit is not a full episode.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


OBSERVATION_BLOCKS = (
    "policy",
    "oracle_travel_direction",
    "terrain",
    "dynamics",
    "adaptation_history",
    "velocity_target",
)
MATCHED_METADATA = (
    "checkpoint_sha256",
    "teacher_interface_sha256",
    "policy_mode",
    "seed",
    "reset_profile",
    "terrain_family",
    "geometry_variant",
    "desired_speed_m_s",
    "desired_yaw_rate_rad_s",
    "step_dt_s",
    "num_envs",
    "max_steps",
)
STOP_REASONS = {"step_limit", "episode_terminated", "application_stopped", "error"}


def _finite_number(value: object) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _json_values(value: object, path: str = "report") -> None:
    """Reject nonfinite values anywhere, including optional diagnostic fields."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for key, child in value.items():
            _json_values(child, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _json_values(child, f"{path}[{index}]")
        return
    raise ValueError(f"{path}: expected finite JSON data, got {value!r}")


def _mapping(value: object, path: str) -> dict:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path}: expected a nonempty object")
    return value


def _vector(value: object, path: str, length: int | None = None) -> list:
    if (
        not isinstance(value, list)
        or not value
        or not all(_finite_number(item) for item in value)
        or (length is not None and len(value) != length)
    ):
        suffix = f" of length {length}" if length is not None else ""
        raise ValueError(f"{path}: expected a nonempty finite numeric vector{suffix}")
    return value


def validate_startup_report(
    report: dict, *, allow_empty_interruption: bool = False
) -> None:
    """Validate one trace, including pre-action timing and terminal retention."""
    _mapping(report, "report")
    _json_values(report)
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("unsupported or missing schema_version (expected 1)")
    if report.get("kind") != "startup_diagnostic":
        raise ValueError("kind must be startup_diagnostic")
    meta = _mapping(report.get("metadata"), "metadata")
    for key in MATCHED_METADATA + ("difficulty_level",):
        if key not in meta:
            raise ValueError(f"metadata.{key}: missing")
    for key in ("checkpoint_sha256", "teacher_interface_sha256"):
        if (
            not isinstance(meta[key], str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", meta[key]) is None
        ):
            raise ValueError(f"metadata.{key}: expected a SHA-256 hexadecimal digest")
    for key in (
        "seed",
        "geometry_variant",
        "difficulty_level",
        "max_steps",
        "num_envs",
    ):
        if type(meta[key]) is not int or meta[key] < 0:
            raise ValueError(f"metadata.{key}: expected a nonnegative integer")
    if meta["num_envs"] != 1 or not 1 <= meta["max_steps"] <= 500:
        raise ValueError(
            "startup diagnostics require num_envs=1 and max_steps in [1, 500]"
        )
    for key in ("desired_speed_m_s", "desired_yaw_rate_rad_s", "step_dt_s"):
        if not _finite_number(meta[key]):
            raise ValueError(f"metadata.{key}: expected a finite number")
    if meta["step_dt_s"] <= 0:
        raise ValueError("metadata.step_dt_s: expected a positive interval")
    for key in ("policy_mode", "reset_profile", "terrain_family"):
        if not isinstance(meta[key], str) or not meta[key]:
            raise ValueError(f"metadata.{key}: expected a nonempty string")
    if report.get("stop_reason") not in STOP_REASONS:
        raise ValueError("missing or invalid stop_reason")
    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        if (
            samples == []
            and allow_empty_interruption
            and report["stop_reason"] in {"error", "application_stopped"}
        ):
            return
        raise ValueError("samples: need at least the first pre-action sample")
    if len(samples) > meta["max_steps"]:
        raise ValueError("sample count exceeds max_steps")
    for index, sample in enumerate(samples):
        label = f"samples[{index}]"
        _mapping(sample, label)
        if type(sample.get("step")) is not int or sample["step"] != index:
            raise ValueError(
                f"{label}.step: expected consecutive indices starting at zero"
            )
        for key, expected in (
            ("time_before_s", index * meta["step_dt_s"]),
            ("time_after_s", (index + 1) * meta["step_dt_s"]),
        ):
            actual = sample.get(key)
            if not _finite_number(actual) or not math.isclose(
                actual, expected, rel_tol=1e-7, abs_tol=1e-8
            ):
                raise ValueError(f"{label}.{key}: expected {expected:g}")
        pre = _mapping(sample.get("pre"), f"{label}.pre")
        observations = _mapping(pre.get("observations"), f"{label}.pre.observations")
        for key in OBSERVATION_BLOCKS:
            _vector(observations.get(key), f"{label}.pre.observations.{key}")
            if len(observations[key]) != len(samples[0]["pre"]["observations"][key]):
                raise ValueError(
                    f"{label}.pre.observations.{key}: vector length changed within trace"
                )
        _vector(
            pre.get("estimated_velocity_body_m_s"),
            f"{label}.estimated_velocity_body_m_s",
            3,
        )
        _mapping(pre.get("state"), f"{label}.pre.state")
        _vector(sample.get("policy_action"), f"{label}.policy_action")
        if len(sample["policy_action"]) != len(samples[0]["policy_action"]):
            raise ValueError(
                f"{label}.policy_action: vector length changed within trace"
            )
        post = _mapping(sample.get("post"), f"{label}.post")
        _mapping(post.get("state"), f"{label}.post.state")
        if type(post.get("done")) is not bool:
            raise ValueError(f"{label}.post.done: expected a boolean")
        term = post.get("termination")
        if not isinstance(term, dict) or not all(
            type(flag) is bool for flag in term.values()
        ):
            raise ValueError(
                f"{label}.post.termination: expected boolean termination flags"
            )
        if post["done"] and index != len(samples) - 1:
            raise ValueError(
                "samples continue after episode termination; reset state must not be mixed into the trace"
            )
    final_done = samples[-1]["post"]["done"]
    if report["stop_reason"] == "episode_terminated" and not final_done:
        raise ValueError("episode_terminated requires a retained terminal sample")
    if report["stop_reason"] == "step_limit" and (
        final_done or len(samples) != meta["max_steps"]
    ):
        raise ValueError(
            "step_limit requires max_steps samples and no episode termination"
        )


def write_startup_report(output_dir: str | Path, report: dict) -> Path:
    """Write a validated trace exclusively; never replace an earlier diagnosis.

    The caller owns creating the explicit run directory. Serialization happens
    before opening the file, so invalid/nonfinite data cannot leave a partial
    artifact that looks like a captured trace.
    """
    # Preserve a failure that happened before the first action as evidence, but
    # the comparator deliberately rejects it: no samples is never parity.
    validate_startup_report(report, allow_empty_interruption=True)
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    path = Path(output_dir) / "startup_diagnostics.json"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return path


def _trace_info(report: dict) -> dict:
    final = report["samples"][-1]
    return {
        "difficulty_level": report["metadata"]["difficulty_level"],
        "sample_count": len(report["samples"]),
        "captured_duration_s": final["time_after_s"],
        "stop_reason": report["stop_reason"],
        "terminal_time_s": final["time_after_s"] if final["post"]["done"] else None,
        "termination": [
            key for key, value in final["post"]["termination"].items() if value
        ],
    }


def compare_startup_reports(left: dict, right: dict, atol: float = 1e-5) -> dict:
    """Compare initial actor inputs/actions in a controlled canonical flat/obstacle pair.

    Returns JSON-safe evidence. Invalid artifacts or incompatible configurations
    cannot produce MATCH, even when their vectors happen to be identical.
    """
    result = {
        "valid": False,
        "verdict": "INVALID",
        "atol": atol,
        "errors": [],
        "blocks": [],
        "traces": [],
        "notice": "Startup parity only; not a performance or reliability certificate. Step-limit traces are not complete episodes.",
    }
    if not _finite_number(atol) or atol < 0:
        result["atol"] = None
        result["errors"].append("atol must be finite and nonnegative")
        return result
    for label, report in (("left", left), ("right", right)):
        try:
            validate_startup_report(report)
        except (ValueError, TypeError) as exc:
            result["errors"].append(f"{label}: {exc}")
    if result["errors"]:
        return result
    lm, rm = left["metadata"], right["metadata"]
    result["traces"] = [_trace_info(left), _trace_info(right)]
    for key in MATCHED_METADATA:
        if lm[key] != rm[key]:
            result["errors"].append(
                f"metadata.{key} differs: {lm[key]!r} vs {rm[key]!r}"
            )
    if lm["reset_profile"] != "canonical" or rm["reset_profile"] != "canonical":
        result["errors"].append("startup parity requires canonical resets on both runs")
    if lm["policy_mode"] not in {"history_mean", "privileged_mean"}:
        result["errors"].append(
            "startup parity requires deterministic history_mean or privileged_mean"
        )
    levels = {lm["difficulty_level"], rm["difficulty_level"]}
    if len(levels) != 2 or 0 not in levels:
        result["errors"].append(
            "compare exactly one level-0 and one positive-level trace"
        )
    for key in (
        "command_profile",
        "capture_metadata",
        "kit_args",
        "domain_randomization_stage",
        "environment_physics",
        "action_clip",
        "task",
    ):
        if lm.get(key) != rm.get(key):
            result["errors"].append(
                f"metadata.{key} differs; capture conditions must match"
            )
    if any(report["stop_reason"] == "error" for report in (left, right)):
        result["errors"].append(
            "a capture ended with an error; resolve it before asserting parity"
        )
    if result["errors"]:
        return result
    lp, rp = left["samples"][0]["pre"], right["samples"][0]["pre"]
    # Terrain and command direction enter the actor in both modes. The dynamics
    # latent is privileged-only; history also feeds the shared velocity estimator.
    names = ["policy", "oracle_travel_direction", "terrain", "adaptation_history"]
    if lm["policy_mode"] == "privileged_mean":
        names.append("dynamics")
    pairs = [
        (f"observations.{key}", lp["observations"][key], rp["observations"][key])
        for key in names
    ]
    pairs += [
        (
            "estimated_velocity_body_m_s",
            lp["estimated_velocity_body_m_s"],
            rp["estimated_velocity_body_m_s"],
        ),
        (
            "policy_action",
            left["samples"][0]["policy_action"],
            right["samples"][0]["policy_action"],
        ),
    ]
    for name, lv, rv in pairs:
        if len(lv) != len(rv):
            result["errors"].append(
                f"{name}: vector length differs ({len(lv)} vs {len(rv)})"
            )
            continue
        deltas = [abs(a - b) for a, b in zip(lv, rv)]
        if not all(math.isfinite(delta) for delta in deltas):
            result["errors"].append(f"{name}: numeric overflow comparing vectors")
            continue
        worst = max(range(len(deltas)), key=deltas.__getitem__)
        result["blocks"].append(
            {
                "field": name,
                "size": len(lv),
                "max_abs_delta": deltas[worst],
                "max_delta_index": worst,
                "different_element_count": sum(delta > atol for delta in deltas),
                "match": deltas[worst] <= atol,
            }
        )
    if result["errors"]:
        return result
    result["valid"] = True
    result["verdict"] = (
        "MATCH" if all(block["match"] for block in result["blocks"]) else "DIFFERENT"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "left",
        type=Path,
        help="Exact flat or obstacle startup_diagnostics.json path (one must be level 0).",
    )
    parser.add_argument(
        "right", type=Path, help="Exact other-level startup_diagnostics.json path."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute initial-input/action tolerance (default: 1e-5).",
    )
    args = parser.parse_args()
    try:
        left = json.loads(args.left.read_text(encoding="utf-8"))
        right = json.loads(args.right.read_text(encoding="utf-8"))
        result = compare_startup_reports(left, right, atol=args.atol)
    except (OSError, ValueError) as exc:
        print(f"[STARTUP PARITY] INVALID: {exc}")
        return 2
    print(f"[STARTUP PARITY] {result['verdict']} (atol={args.atol:g})")
    for error in result["errors"]:
        print(f"  - {error}")
    for block in result["blocks"]:
        print(
            f"  {block['field']}: max |delta|={block['max_abs_delta']:.6g}; {block['different_element_count']}/{block['size']} outside tolerance"
        )
    for trace in result["traces"]:
        terminal = (
            f"; terminal at {trace['terminal_time_s']:.3f}s ({', '.join(trace['termination']) or 'unspecified'})"
            if trace["terminal_time_s"] is not None
            else ""
        )
        print(
            f"  L{trace['difficulty_level']}: {trace['sample_count']} steps, {trace['captured_duration_s']:.3f}s, {trace['stop_reason']}{terminal}"
        )
    print(f"  {result['notice']}")
    return 2 if not result["valid"] else (0 if result["verdict"] == "MATCH" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
