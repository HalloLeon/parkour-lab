"""Evidence-bound, one-block v3 continuation of the completed zero-retention run."""

import json
from pathlib import Path

try:
    from . import operator_stop_probe as probe
    from .operator_benchmark_core import load_reference_checkpoint
    from .operator_curriculum import TRANSITION_VERSION, curriculum_manifest
    from .operator_retention import retention_manifest, validate_adam_state
    from .operator_sequences import VERSION, sequence_manifest
except ImportError:
    import operator_stop_probe as probe
    from operator_benchmark_core import load_reference_checkpoint
    from operator_curriculum import TRANSITION_VERSION, curriculum_manifest
    from operator_retention import retention_manifest, validate_adam_state
    from operator_sequences import VERSION, sequence_manifest


# R39's reviewed probe predates the download-path-only validation repair. Accept
# that exact producer, not arbitrary changed code claiming the same version.
REVIEWED_PROBE_SHA256 = (
    "6cc27c1f7aac6300acff3019c620a91e34641a7bfad1e3f955688895f28f1360"
)


def validate_recovery_evidence(directory, baseline):
    directory = directory.resolve(strict=True)
    protocol = probe.read_json(directory / "protocol.json")
    report = probe.read_json(directory / "report.json")
    if (
        report.get("status") != "PROBE_COMPLETE"
        or report.get("policy_acceptance") is not False
        or report.get("protocol") != protocol
        or set(report.get("arms", {})) != set(probe.ARMS)
        or protocol.get("schema_version") != probe.VERSION
        or protocol.get("policy_acceptance") is not False
        or protocol.get("arms")
        != {name: delay * probe.DT for name, delay in probe.ARMS.items()}
        or protocol.get("seed") != 43
        or protocol.get("repetitions") != 10
        or protocol.get("baseline_identity") != baseline["identity"]
    ):
        raise ValueError(
            "Require the complete two-arm matched-prefix recovery experiment"
        )
    names = (
        "operator_stop_probe.py",
        "operator_benchmark.py",
        *probe.STABLE_SOURCES.values(),
    )
    if set(protocol.get("source_sha256", {})) != set(names):
        raise ValueError("Incomplete recovery producer identity")
    for name in names:
        accepted = {probe.file_sha256(Path(__file__).with_name(name))}
        if name == "operator_stop_probe.py":
            accepted.add(REVIEWED_PROBE_SHA256)
        if protocol["source_sha256"][name] not in accepted:
            raise ValueError(f"Unsupported recovery producer: {name}")
    failed = {t["env_id"] for t in baseline["report"]["trials"] if not t["passed"]}
    if not failed:
        raise ValueError("No remaining development failure justifies this repair")
    files = {
        name: probe.file_sha256(directory / name)
        for name in ("protocol.json", "report.json")
    }
    for arm in probe.ARMS:
        path = directory / arm
        measured = probe.read_json(path / "measurement_report.json")
        result = probe.read_json(path / "report.json")
        worker = probe.read_json(path / "worker_status.json")
        if (
            result != report["arms"][arm]
            or any(result.get(k) != v for k, v in measured.items())
            or worker != {"returncode": 0, "timed_out": False}
            or result.get("worker") != worker
            or any(path.glob("*_cleanup_error.json"))
        ):
            raise ValueError(f"Incomplete/inconsistent recovery worker: {arm}")
        probe.validate_probe_output(path, result, baseline, arm)
        trials = result["comparison"]["behavioral_result"]["trials"]
        if {t["env_id"] for t in trials if not t["passed"]} != failed:
            raise ValueError(
                "Recovery outcomes differ from the unresolved-stop hypothesis"
            )
        for t in trials:
            if t["sequence_failures"] or any(
                not failure.startswith("stop:") for failure in t["failures"]
            ):
                raise ValueError(
                    "Sequence-only repair requires isolated stop failures, no sequence violations"
                )
        for name in (
            "report.json",
            "measurement_report.json",
            "worker_status.json",
            "resolved_env.yaml",
            "trace.npz",
            "control_trace.npz",
            "control_interface.json",
            "control_report.json",
            "selection_trace.npz",
        ):
            files[f"{arm}/{name}"] = probe.file_sha256(path / name)
    return {
        "directory": str(directory),
        "sha256": files,
        "finding": "Neither original-reference stop takeover rescues the remaining learner failures; stronger reference anchoring is not supported",
    }


def sequence_resume_preflight(args, agent, saved, baseline):
    """Restore current learning state; change only command-sequence exposure."""
    if (
        args.curriculum != VERSION
        or args.iterations != 200
        or not args.moving_retention
        or args.resume_retention_reference is None
        or args.zero_command_reference_report is None
        or args.skip_check
        or args.refinement_profile != "source"
    ):
        raise ValueError(
            "v3 requires 200 updates, both preserved retention terms, original reference and checks"
        )
    capture = probe.load_baseline(
        args.checkpoint, args.resume_retention_reference, args.baseline_report.parent
    )
    evidence = validate_recovery_evidence(args.reversal_stop_probe, capture)
    source = load_reference_checkpoint(args.checkpoint, agent)
    run = args.checkpoint.parent
    status = probe.read_json(run / "training_status.json")
    handoff = probe.read_json(run / "params/operator_training.json")
    provenance = probe.read_json(run / "source_provenance.json")
    checkpoint_handoff = json.loads(
        json.dumps(source.get("infos", {}).get("handoff"), allow_nan=False)
    )
    reference = capture["reference_identity"]
    loss = handoff.get("moving_retention", {})
    previous_resume = handoff.get("retention_resume", {})
    if (
        status.get("status") != "COMPLETED"
        or status.get("final_sha256") != baseline["sha256"]["checkpoint"]
        or status.get("handoff") != handoff
        or checkpoint_handoff != handoff
        or handoff.get("final_iteration") != source["iter"]
        or source["iter"] != reference["iteration"] + 400
        or handoff.get("source_iteration") != reference["iteration"] + 200
        or handoff.get("first_update") != handoff.get("source_iteration") + 1
        or handoff.get("additional_updates") != 200
        or handoff.get("curriculum")
        != json.loads(json.dumps(curriculum_manifest(TRANSITION_VERSION)))
        or any(
            loss.get(k) != v for k, v in retention_manifest(zero_command=True).items()
        )
        or loss.get("reference_sha256") != reference["sha256"]["checkpoint"]
        or loss.get("reference_iteration") != reference["iteration"]
        or loss.get("check_offsets") != [100, 200]
        or handoff.get("zero_command_reference", {}).get("intervention")
        != loss.get("zero_command_retention")
        or previous_resume.get("adam_steps") != 4000
        or previous_resume.get("cumulative_retention_updates") != 400
        or previous_resume.get("reference_sha256") != reference["sha256"]["checkpoint"]
        or previous_resume.get("checkpoint_sha256") != handoff.get("source_sha256")
        or provenance.get("sha256", {}).get("checkpoint")
        != handoff.get("source_sha256")
        or args.seed != int(agent["seed"])
        or args.seed != int(saved["seed"])
        or args.num_envs != int(saved["scene"]["num_envs"])
        or int(agent["num_steps_per_env"]) != 24
        or int(agent["save_interval"]) != 50
        or int(agent["algorithm"]["num_learning_epochs"]) != 5
        or int(agent["algorithm"]["num_mini_batches"]) != 4
        or agent["algorithm"]["schedule"] != "fixed"
        or float(agent["algorithm"]["learning_rate"]) != 1e-4
    ):
        raise ValueError(
            "v3 requires the completed +400 zero-retention learning state with unchanged seed/config/reference"
        )
    for name in (
        "operator_curriculum.py",
        "operator_command.py",
        "operator_profiles.py",
        "operator_rewards.py",
        "operator_retention.py",
    ):
        if provenance["sha256"].get(name) != probe.file_sha256(
            Path(__file__).with_name(name)
        ):
            raise ValueError(f"Preserved learning implementation changed: {name}")
    validate_adam_state(
        source["optimizer_state_dict"], source["model_state_dict"], 8000
    )
    return {
        "checkpoint_sha256": baseline["sha256"]["checkpoint"],
        "reference_checkpoint": str(args.resume_retention_reference.resolve()),
        "reference_sha256": reference["sha256"]["checkpoint"],
        "reference_iteration": reference["iteration"],
        "adam_steps": 8000,
        "additional_updates": 200,
        "cumulative_retention_updates": 600,
        "optimizer": "exact saved moments, counters, parameter order and options",
        "not_restored": [
            "simulator state",
            "command ages",
            "random generator state",
            "rollout buffers",
        ],
        "curriculum_intervention": sequence_manifest(),
        "recovery_evidence": evidence,
        "evidence_sha256": {
            name: probe.file_sha256(run / name)
            for name in (
                "training_status.json",
                "params/operator_training.json",
                "source_provenance.json",
            )
        },
        "scope": "one 200-update sequence-exposure intervention, both soft losses preserved; simulation/RNG restart, not uninterrupted trajectory equivalence",
    }
