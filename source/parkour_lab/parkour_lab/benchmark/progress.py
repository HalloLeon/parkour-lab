"""Checkpoint-linked task progress; learning state and file formats stay external."""

import json
from copy import deepcopy

from parkour_lab.runtime.accounting import nonnegative_integer

VERSION = "go2_checkpoint_progress_v1"


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Continuation must contain finite JSON data") from error


def _identity(env):
    if getattr(env.cfg, "benchmark_training", None) is not True:
        raise ValueError("Continuation requires a benchmark training environment")
    return dict(
        protocol=_text(env.cfg.benchmark_version, "protocol"),
        task_id=_text(env.cfg.benchmark_task_id, "task ID"),
        seed=nonnegative_integer(env.cfg.seed, "seed"),
        num_envs=env.num_envs,
    )


def _milestones(values, cumulative):
    if type(values) not in (list, tuple):
        raise ValueError("Milestones must be an ordered sequence")
    result = [nonnegative_integer(value, "milestone") for value in values]
    if result != sorted(set(result)) or any(value > cumulative for value in result):
        raise ValueError("Milestones must be unique, ordered and already reached")
    return result


def _curriculum(env):
    terrain = getattr(env.scene, "terrain", None)
    levels = getattr(terrain, "terrain_levels", None)
    types = getattr(terrain, "terrain_types", None)
    if levels is None and types is None:
        return None
    if levels is None or types is None:
        raise ValueError("Incomplete realized terrain assignments")
    return dict(
        levels=levels.detach().cpu().tolist(), types=types.detach().cpu().tolist()
    )


def capture_progress(env, *, run_id, checkpoint_id, configuration_id, milestones=()):
    """Capture at a trainer-owned checkpoint barrier, without advancing the task."""
    identity = _identity(env)
    run_id = _text(run_id, "run ID")
    checkpoint_id = _text(checkpoint_id, "checkpoint ID")
    configuration_id = _text(configuration_id, "configuration ID")
    previous = getattr(env, "_benchmark_progress", None)
    if previous is not None and any(
        previous[name] != value
        for name, value in {
            **identity,
            "run_id": run_id,
            "configuration_id": configuration_id,
        }.items()
    ):
        raise ValueError("Continuation cannot change the run identity")
    completed = _milestones(milestones, env.interactions.cumulative)
    if previous is not None:
        completed = sorted(set(completed + previous["milestones"]))
    command = env.command_manager.get_term("base_velocity")
    record = _json_copy(
        dict(
            version=VERSION,
            **identity,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            configuration_id=configuration_id,
            checkpoint_interactions=env.interactions.cumulative,
            interactions=env.interactions.state_dict(),
            milestones=completed,
            sampler=env.parkour_randomization.state_dict(),
            sampling_report=getattr(command, "sampling_report", None),
            curriculum=_curriculum(env),
            restart=(
                previous["restart"]
                if previous is not None
                else dict(count=0, discarded_interactions=0, reason=None)
            ),
        )
    )
    env._benchmark_progress = deepcopy(record)
    return record


def _validate_record(env, record, checkpoint_id, configuration_id):
    expected = {
        "version",
        "protocol",
        "task_id",
        "seed",
        "num_envs",
        "run_id",
        "checkpoint_id",
        "configuration_id",
        "checkpoint_interactions",
        "interactions",
        "milestones",
        "sampler",
        "sampling_report",
        "curriculum",
        "restart",
    }
    if (
        type(record) is not dict
        or set(record) != expected
        or record["version"] != VERSION
    ):
        raise ValueError("Invalid continuation record version or schema")
    for name, value in {
        **_identity(env),
        "checkpoint_id": checkpoint_id,
        "configuration_id": configuration_id,
    }.items():
        if type(record[name]) is not type(value) or record[name] != value:
            raise ValueError(f"Continuation {name} differs")
    for name in ("run_id", "checkpoint_id", "configuration_id"):
        _text(record[name], name)
    counts = record["interactions"]
    if type(counts) is not dict or set(counts) != {
        "session",
        "cumulative",
        "uncertain",
    }:
        raise ValueError("Invalid continuation accounting schema")
    for name, value in counts.items():
        nonnegative_integer(value, name)
    at_checkpoint = nonnegative_integer(
        record["checkpoint_interactions"], "checkpoint interactions"
    )
    if (
        max(counts["session"], counts["uncertain"], at_checkpoint)
        > counts["cumulative"]
    ):
        raise ValueError("Continuation accounting exceeds its cumulative total")
    _milestones(record["milestones"], counts["cumulative"])
    restart = record["restart"]
    if type(restart) is not dict or set(restart) != {
        "count",
        "discarded_interactions",
        "reason",
    }:
        raise ValueError("Invalid restart accounting schema")
    nonnegative_integer(restart["count"], "restart count")
    if (
        nonnegative_integer(restart["discarded_interactions"], "discarded interactions")
        > counts["cumulative"]
    ):
        raise ValueError("Discarded interactions exceed cumulative spend")
    if restart["reason"] is not None:
        _text(restart["reason"], "restart reason")


def _validate_sampling_report(saved, current):
    if saved is None and current is None:
        return
    if (
        type(saved) is not dict
        or type(current) is not dict
        or set(saved) != set(current)
    ):
        raise ValueError("Command sampling report identity differs")
    for name, value in saved.items():
        if name in ("samples", "stops", "restricted_samples", "pivots"):
            nonnegative_integer(value, name)
        elif type(value) is not type(current[name]) or value != current[name]:
            raise ValueError("Command sampling report definition differs")
    if any(
        saved[name] > saved["samples"]
        for name in ("stops", "restricted_samples", "pivots")
        if name in saved
    ):
        raise ValueError("Command category counts exceed total samples")


def restore_progress(
    env,
    record,
    *,
    checkpoint_id,
    configuration_id,
    charged_interactions=None,
    uncertain_interactions=0,
    reason=None,
):
    """Restore before collection, then explicitly reset fresh episodes.

    charged_interactions is the latest known total, not an amount to add again.
    uncertain_interactions is an additional conservative charge since this record.
    No live trajectory, command packet, learner RNG or replay sequence is restored.
    """
    record = _json_copy(record)
    _validate_record(env, record, checkpoint_id, configuration_id)
    if env.interactions.session:
        raise ValueError("Restore before this environment collects any interactions")
    previous = getattr(env, "_benchmark_progress", None)
    if previous is not None and previous["run_id"] != record["run_id"]:
        raise ValueError("Continuation cannot change the run identity")
    saved = record["interactions"]
    known = (
        saved["cumulative"]
        if charged_interactions is None
        else nonnegative_integer(charged_interactions, "charged interactions")
    )
    floor = max(saved["cumulative"], known)
    extra = nonnegative_integer(uncertain_interactions, "uncertain interactions")
    if (floor > saved["cumulative"] or extra) and reason is None:
        raise ValueError("Rollback or uncertain collection requires an explicit reason")
    reason = _text("restart" if reason is None else reason, "restart reason")
    total, unknown = floor + extra, saved["uncertain"] + extra
    if total < env.interactions.cumulative or unknown < env.interactions.uncertain:
        raise ValueError("Continuation cannot reduce already charged interactions")
    if record["curriculum"] is not None or _curriculum(env) is not None:
        raise NotImplementedError(
            "Curriculum continuation needs the terrain progression binding"
        )
    sampler = env.parkour_randomization
    if (
        type(record["sampler"]) is not dict
        or record["sampler"].get("physical") != sampler.state_dict()["physical"]
    ):
        raise ValueError("Startup physical draws differ from the checkpoint")
    deepcopy(sampler).load_state_dict(record["sampler"])
    command = env.command_manager.get_term("base_velocity")
    current_report = getattr(command, "sampling_report", None)
    report = record["sampling_report"]
    _validate_sampling_report(report, current_report)
    sampler.load_state_dict(record["sampler"])
    env.interactions.restore(total, uncertain=unknown)
    if report is not None:
        command.sampling_report = deepcopy(report)
    record["interactions"] = env.interactions.state_dict()
    record["restart"] = dict(
        count=record["restart"]["count"] + 1,
        discarded_interactions=record["restart"]["discarded_interactions"]
        + floor
        - saved["cumulative"],
        reason=reason,
    )
    env._benchmark_progress = deepcopy(record)
    return record
