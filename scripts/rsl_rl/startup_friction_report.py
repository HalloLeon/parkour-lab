"""Gate a conditional legacy-friction replay, never certify a robot repair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from . import startup_probe_report as probe
    from .startup_centered_report import _outcomes, _same_original_physics
except ImportError:
    import startup_probe_report as probe
    from startup_centered_report import _outcomes, _same_original_physics


def _metadata(trace, mode):
    probe._require(
        "scene_intervention" not in trace["metadata"],
        "Friction probe must be uncentered",
    )
    meta = probe._field(trace["metadata"], "legacy_friction_probe", "trace")
    for key, expected in (
        ("kind", "startup_legacy_joint_friction"),
        ("schema_version", 1),
        ("mode", mode),
    ):
        probe._match(meta.get(key), expected, f"legacy friction {key}")
    probe._match(
        meta.get("action_joint_names"),
        trace["physical_metadata"]["joint_names"],
        "friction action order",
    )
    raw = meta.get("raw_joint_names")
    probe._require(
        isinstance(raw, list)
        and len(raw) == 12
        and set(raw) == set(meta["action_joint_names"]),
        "Friction raw joint map invalid",
    )
    probe._require(
        isinstance(meta.get("reason"), str)
        and isinstance(meta.get("limitations"), list),
        "Friction status explanation missing",
    )
    probe._numeric_shape(
        meta.get("declared_actuator_friction"), (12,), "Declared actuator friction"
    )
    probe._require(
        all(v >= 0 for v in meta["declared_actuator_friction"]),
        "Negative declared actuator friction",
    )
    probe._match(
        meta.get("setter_called"), mode == "zero", "Explicit legacy setter call"
    )
    if meta.get("status") == "UNAVAILABLE":
        probe._require(
            mode == "observe", "Unavailable zeroing is not an applied intervention"
        )
        return meta
    for when in ("before", "after"):
        legacy = meta.get(f"legacy_{when}_action_order")
        new = meta.get(f"new_params_{when}_action_order")
        probe._numeric_shape(legacy, (12,), "legacy coefficient")
        probe._numeric_shape(new, (12, 3), "new friction parameters")
        probe._require(
            all(v >= 0 for v in legacy), "Negative legacy friction coefficient"
        )
        probe._require(
            all(v >= 0 for v in probe._flatten(new)), "Negative new friction parameter"
        )
    probe._match(
        meta["new_params_before_action_order"],
        meta["new_params_after_action_order"],
        "new friction unchanged",
    )
    properties = trace["physical_metadata"]["properties"]
    probe._match(
        meta["legacy_after_action_order"],
        properties.get("joint_legacy_friction_coefficient"),
        "Captured legacy friction readback",
    )
    probe._match(
        meta["new_params_after_action_order"],
        properties.get("joint_friction_static_dynamic_viscous"),
        "Captured new friction readback",
    )
    if mode == "observe":
        probe._match(meta.get("status"), "OBSERVED", "observe status")
        probe._match(
            meta["legacy_before_action_order"],
            meta["legacy_after_action_order"],
            "Observe must not write friction",
        )
    else:
        probe._match(meta.get("status"), "APPLIED", "zero status")
        probe._require(
            any(v > 0 for v in meta["legacy_before_action_order"]),
            "Zeroing must change a nonzero legacy channel",
        )
        probe._match(
            meta["legacy_after_action_order"], [0.0] * 12, "Zero legacy readback"
        )
        probe._match(
            meta["new_params_after_action_order"],
            [[0.0] * 3 for _ in range(12)],
            "Zero trial requires nominal new friction",
        )
        probe._match(
            meta["declared_actuator_friction"],
            [0.0] * 12,
            "Zero trial requires declared nominal friction",
        )
    return meta


def friction_preflight(native, baseline, source, *, reference_sha256):
    probe.validate_failure_trace(source)
    for name, value in (("native", native), ("baseline", baseline)):
        probe._require(
            value["metadata"]["terrain_family"] == "high_step"
            and value["metadata"]["difficulty_level"] == 6,
            f"{name}: requires high_step L6",
        )
        probe._require(
            "scene_intervention" not in value["metadata"], f"{name}: must be uncentered"
        )
        probe._validate_probe(value, source, reference_sha256, name)
    probe._require(
        "legacy_friction_probe" not in baseline["metadata"],
        "Baseline must precede the friction intervention",
    )
    _same_original_physics(native, baseline, "native vs previous control")
    meta = _metadata(native, "observe")
    comparisons = {
        "native_minus_previous_L6": _outcomes(native, baseline),
        "native_minus_original_policy": _outcomes(native, source),
    }
    exact = all(
        field["max_abs_difference"] == 0
        for fields in comparisons.values()
        for field in fields.values()
    )
    if not exact:
        status = "CONTROL_REPRODUCTION_MISMATCH"
    elif meta.get("status") == "UNAVAILABLE":
        status = "LEGACY_CHANNEL_UNAVAILABLE"
    elif not any(meta["legacy_before_action_order"]):
        status = "NO_INTERVENTION_NEEDED"
    elif any(probe._flatten(meta["new_params_before_action_order"])):
        status = "NONZERO_NEW_FRICTION"
    elif any(meta["declared_actuator_friction"]):
        status = "NONZERO_DECLARED_FRICTION"
    else:
        status = "READY_FOR_ZERO_INTERVENTION"
    return {
        "kind": "startup_legacy_friction_preflight",
        "schema_version": 1,
        "evidence_status": status,
        "reference_sha256": reference_sha256,
        "checkpoint_sha256": source["metadata"]["checkpoint_sha256"],
        "native_recorded_outcomes_exact": exact,
        "friction": meta,
        "comparisons": comparisons,
        "limitations": [
            "A readback gate is not a physics-bug verdict or robot acceptance.",
            "Already-zero or unavailable legacy coefficients stop this branch before further simulation.",
        ],
    }


def compare_friction_reports(
    native, flat, obstacle, baseline, source, *, reference_sha256
):
    control = friction_preflight(
        native, baseline, source, reference_sha256=reference_sha256
    )
    probe._require(
        control["evidence_status"] == "READY_FOR_ZERO_INTERVENTION",
        "Native friction preflight did not pass",
    )
    pair = probe.compare_probe_reports(
        flat, obstacle, source, reference_sha256=reference_sha256
    )
    interventions = {}
    for label, value in (("L0", flat), ("L6", obstacle)):
        meta = _metadata(value, "zero")
        probe._match(
            meta["legacy_before_action_order"],
            control["friction"]["legacy_before_action_order"],
            f"{label} original legacy coefficient",
        )
        _same_original_physics(value, native, f"{label} unchanged original physics")
        interventions[label] = meta
    return {
        "kind": "startup_legacy_friction_comparison",
        "schema_version": 1,
        "evidence_status": "VALID_DIAGNOSTIC",
        "reference_sha256": reference_sha256,
        "checkpoint_sha256": source["metadata"]["checkpoint_sha256"],
        "control": control,
        "interventions": interventions,
        "zero_friction_pair": pair,
        "zero_L6_minus_native_L6": _outcomes(obstacle, native),
        "first_action_zero_L6_minus_native_L6": probe._substep_outcomes(
            obstacle, native
        ),
        "limitations": [
            "Only the deprecated joint-friction coefficient was intentionally changed; this is an open-loop diagnostic, not training or policy evaluation.",
            "Reduced divergence after zeroing supports legacy-channel sensitivity, not a complete solver-defect diagnosis or calibrated real-robot friction model.",
            "Do not promote zero friction automatically: normal-policy traversal, stops, both pivots, retained terrain and operator tests remain required.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe_dir", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)

    def read(path):
        return json.loads(path.read_bytes(), object_pairs_hook=probe._unique_object)

    try:
        raw = args.reference.read_bytes()
        source = json.loads(raw, object_pairs_hook=probe._unique_object)
        digest = hashlib.sha256(raw).hexdigest()
        native = read(args.probe_dir / "native_L6/startup_diagnostics.json")
        baseline = read(args.baseline)
        if args.preflight:
            result = friction_preflight(
                native, baseline, source, reference_sha256=digest
            )
        else:
            result = compare_friction_reports(
                native,
                read(args.probe_dir / "zero_L0/startup_diagnostics.json"),
                read(args.probe_dir / "zero_L6/startup_diagnostics.json"),
                baseline,
                source,
                reference_sha256=digest,
            )
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        result = {
            "kind": "startup_legacy_friction_comparison",
            "evidence_status": "INVALID_DIAGNOSTIC",
            "error": str(error),
        }
    print(json.dumps(result, indent=2, allow_nan=False))
    return (
        0
        if result["evidence_status"]
        in ("READY_FOR_ZERO_INTERVENTION", "VALID_DIAGNOSTIC")
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
