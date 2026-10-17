"""Native SB3 archives linked to benchmark progress, never a core checkpoint API.

Load only trusted local checkpoints: SB3 archives, replay and RNG files contain
Python pickle data. File hashes detect accidental corruption, not hostile input.
"""

import json
from pathlib import Path

import stable_baselines3
import torch
from stable_baselines3 import PPO, SAC

from parkour_lab.provenance import dependency_identity, file_sha256, write_json

from .state import (
    capture_model_state,
    capture_rng,
    model_state_sha256,
    replay_state_sha256,
    restore_rng,
    state_sha256,
    validate_model,
)

VERSION = "sb3_go2_smoke_checkpoint_v1"
ALGORITHMS = {"PPO": PPO, "SAC": SAC}
__all__ = (
    "save_checkpoint",
    "load_checkpoint",
    "load_model",
    "capture_model_state",
    "model_state_sha256",
)


def _source_identity():
    return {
        name: dependency_identity(name, distribution)["source_sha256"]
        for name, distribution in (
            ("stable_baselines3", "stable-baselines3"),
            ("parkour_lab", "parkour-lab"),
        )
    }


def save_checkpoint(model, task, folder, *, run_id, milestones=()):
    """Save after completed optimizer work, publishing checkpoint.json last.

    The destination must not exist. An interrupted write leaves an incomplete,
    unloadable directory; previous checkpoint directories remain untouched.
    """
    validate_model(model)
    if (
        model.num_timesteps != task.interactions["cumulative"]
        or task.interactions["uncertain"]
        or model.seed != task.metadata["seed"]
        or model._n_updates < 1
    ):
        raise ValueError("Learner counters, seed or completed updates differ from task")
    if isinstance(model, PPO) and not model.rollout_buffer.full:
        raise ValueError("Save PPO only after a complete rollout and optimizer update")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    model.save(folder / "model.zip")
    checkpoint_hash = file_sha256(folder / "model.zip")
    checkpoint_id = "sb3:" + checkpoint_hash
    if isinstance(model, SAC):
        model.save_replay_buffer(folder / "replay.pkl")
    rng = capture_rng(model.action_space)
    torch.save(rng, folder / "rng.pt")
    progress = task.progress(
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        milestones=milestones,
    )
    write_json(folder / "progress.json", progress)
    receipt = dict(
        version=VERSION,
        library_version=stable_baselines3.__version__,
        source_sha256=_source_identity(),
        algorithm=type(model).__name__,
        training_seed=int(model.seed),
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_hash,
        training_interactions=task.interactions["cumulative"],
        task_identity={
            name: task.metadata[name]
            for name in (
                "protocol",
                "task_id",
                "configuration_id",
                "seed",
            )
        },
        model_state_sha256=model_state_sha256(model),
        rng_state_sha256=state_sha256(rng),
        replay_state_sha256=(
            replay_state_sha256(model) if isinstance(model, SAC) else None
        ),
        fresh_physical_episodes_on_resume=True,
        partial_rollout_restored=False,
        files={path.name: file_sha256(path) for path in sorted(folder.iterdir())},
    )
    write_json(folder / "checkpoint.json", receipt)
    return receipt


def _verified(folder):
    folder = Path(folder)
    receipt = json.loads((folder / "checkpoint.json").read_text())
    if (
        receipt["version"] != VERSION
        or receipt["library_version"] != stable_baselines3.__version__
        or receipt["algorithm"] not in ALGORITHMS
        or receipt["source_sha256"] != _source_identity()
    ):
        raise ValueError("Checkpoint library, version or source differs")
    expected = {"model.zip", "rng.pt", "progress.json"}
    if receipt["algorithm"] == "SAC":
        expected.add("replay.pkl")
    if set(receipt["files"]) != expected:
        raise ValueError("Checkpoint file inventory differs")
    for name, digest in receipt["files"].items():
        if file_sha256(folder / name) != digest:
            raise ValueError(f"Checkpoint hash differs: {name}")
    if (
        receipt["checkpoint_sha256"] != receipt["files"]["model.zip"]
        or receipt["checkpoint_id"] != "sb3:" + receipt["checkpoint_sha256"]
    ):
        raise ValueError("Checkpoint model identity differs")
    return folder, receipt


def load_model(folder, *, device="cpu"):
    """Verify the complete bundle, load inference weights, preserve caller RNG."""
    folder, receipt = _verified(folder)
    before = capture_rng()
    try:
        model = ALGORITHMS[receipt["algorithm"]].load(
            folder / "model.zip",
            device=device,
            force_reset=True,
        )
        # SB3 only applies force_reset when an environment is passed to load.
        # This loader returns an unbound model, so discard archived live frames.
        model._last_obs = None
        model._last_original_obs = None
        if model_state_sha256(model) != receipt["model_state_sha256"]:
            raise ValueError("Restored learner model state differs")
    finally:
        restore_rng(before)
    return model, receipt


def load_checkpoint(folder, task, *, device="cpu"):
    """Restore learner/replay and task progress before reset; restore RNG last."""
    folder, receipt = _verified(folder)
    identity = {name: task.metadata[name] for name in receipt["task_identity"]}
    if identity != receipt["task_identity"]:
        raise ValueError("Checkpoint task protocol, configuration or seed differs")
    model, receipt = load_model(folder, device=device)
    if isinstance(model, SAC):
        model.load_replay_buffer(folder / "replay.pkl")
        if replay_state_sha256(model) != receipt["replay_state_sha256"]:
            raise ValueError("Restored SAC replay state differs")
    progress = json.loads((folder / "progress.json").read_text())
    if (
        progress["checkpoint_id"] != receipt["checkpoint_id"]
        or progress["checkpoint_interactions"] != receipt["training_interactions"]
        or model.num_timesteps != receipt["training_interactions"]
        or model.seed != receipt["training_seed"]
    ):
        raise ValueError("Learner and benchmark progress linkage differs")
    rng = torch.load(folder / "rng.pt", map_location="cpu", weights_only=False)
    if state_sha256(rng) != receipt["rng_state_sha256"]:
        raise ValueError("Saved learner RNG state differs")
    task.restore(progress, checkpoint_id=receipt["checkpoint_id"])
    restore_rng(rng, model.action_space)
    return model, receipt
