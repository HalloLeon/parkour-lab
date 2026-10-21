"""A bounded call to external PPO/SAC learning and their own checkpoint APIs."""

import math
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure

from parkour_lab.runtime.console import is_enabled

from .checkpoint import load_checkpoint
from .environment import Go2VecEnv
from .policy import SB3FramePolicy
from .progress import TrainingProgress
from .snapshots import SnapshotIndex, next_snapshot, validate_schedule
from .state import validate_model

PPO_ROLLOUT_STEPS = 32


def _model(algorithm, env, *, seed, device):
    common = dict(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        device=device,
        learning_rate=3e-4,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=0,
    )
    if algorithm == "ppo":
        return PPO(
            **common,
            n_steps=PPO_ROLLOUT_STEPS,
            batch_size=32,
            n_epochs=2,
            clip_range=0.2,
        )
    if algorithm == "sac":
        return SAC(
            **common,
            buffer_size=20_000,
            batch_size=64,
            learning_starts=max(64, 2 * env.num_envs),
            train_freq=(1, "step"),
            gradient_steps=1,
            optimize_memory_usage=False,
        )
    raise ValueError("Choose ppo or sac")


def _losses(model):
    values = {
        name: float(value)
        for name, value in model.logger.name_to_value.items()
        if name.startswith("train/")
    }
    if not values or not all(math.isfinite(value) for value in values.values()):
        raise RuntimeError("Learning produced missing or nonfinite diagnostics")
    return values


def _recipe(model, algorithm):
    recipe = dict(
        algorithm=algorithm,
        policy="MlpPolicy",
        net_arch=model.policy.net_arch,
        learning_rate=model.learning_rate,
        batch_size=model.batch_size,
        gamma=model.gamma,
    )
    names = (
        (
            "n_steps",
            "n_epochs",
            "gae_lambda",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "normalize_advantage",
            "target_kl",
        )
        if algorithm == "ppo"
        else (
            "buffer_size",
            "learning_starts",
            "gradient_steps",
            "tau",
            "target_update_interval",
            "target_entropy",
            "ent_coef",
            "optimize_memory_usage",
        )
    )
    recipe.update({name: getattr(model, name) for name in names})
    if algorithm == "ppo":
        recipe["clip_range"] = model.clip_range(1.0)
    else:
        recipe["train_freq"] = [model.train_freq.frequency, model.train_freq.unit.value]
    return recipe


def _learn_and_save(
    model,
    task,
    *,
    algorithm,
    remaining,
    target,
    output,
    snapshots,
    interval,
    milestones,
):
    before = SB3FramePolicy(model).state_sha256()
    updates_before = model._n_updates
    formats = ["stdout", "json"] if is_enabled() else ["json"]
    model.set_logger(configure(str(output / "metrics"), formats))
    progress = TrainingProgress(task, algorithm=algorithm, target=target)
    try:
        progress.emit("training_started", model)
        while model.num_timesteps < target:
            point, reasons = next_snapshot(
                model,
                algorithm,
                interval=interval,
                milestones=milestones,
                target=target,
            )
            model.learn(
                total_timesteps=point - model.num_timesteps,
                reset_num_timesteps=False,
                callback=progress,
            )
            counts = task.interactions
            if (
                model.num_timesteps != point
                or counts["cumulative"] != point
                or counts["session"] != point - (target - remaining)
                or counts["uncertain"]
                or ("periodic" in reasons and model._n_updates % interval)
            ):
                raise RuntimeError("Checkpoint barrier differs from real collection")
            if "terminal" in reasons:
                after = SB3FramePolicy(model).state_sha256()
                if model._n_updates <= updates_before or before == after:
                    raise RuntimeError("Learning or parameter change check failed")
            losses = _losses(model)
            model.logger.dump(step=model.num_timesteps)
            receipt = snapshots.save(model, task, reasons)
        progress.emit("training_completed", model)
        return {
            "training_interactions": counts,
            "learner_timesteps": model.num_timesteps,
            "external_update_counter_before": updates_before,
            "external_update_counter_after": model._n_updates,
            "policy_state_before": before,
            "policy_state_after": after,
            "parameters_changed": before != after,
            "last_training_diagnostics": losses,
            "checkpoint": receipt,
            "checkpoint_index": "checkpoint_index.json",
            "training_loop_wall_time_s": snapshots.record["checkpoints"][-1][
                "training_loop_wall_time_s"
            ],
        }
    finally:
        model.logger.close()


def run_training(
    task,
    *,
    algorithm,
    total_interactions,
    output,
    learner_device="cpu",
    resume=None,
    app=None,
    run_id=None,
    save_every_updates=5000,
    milestones=(),
):
    """Complete whole collection blocks; never relabel overshoot as the allowance.

    Constant schedules avoid resetting a decay horizon between short sessions.
    Save after complete external updates, without resetting physical episodes.
    Resume starts fresh episodes; partial PPO rollouts/live Sim state are omitted.
    """
    if algorithm not in ("ppo", "sac"):
        raise ValueError("Choose ppo or sac")
    if type(total_interactions) is not int or total_interactions <= 0:
        raise ValueError("Require a positive total interaction allowance")
    output = Path(output)
    env = Go2VecEnv(task, total_interactions=total_interactions, app=app)
    resumed = None
    if resume is not None:
        model, resumed = load_checkpoint(resume, task, device=learner_device)
        if resumed["algorithm"].lower() != algorithm:
            raise ValueError("Resume must keep the checkpoint's algorithm")
        if run_id is not None and run_id != resumed["run_id"]:
            raise ValueError("Resume must keep the checkpoint's run identity")
        model.set_env(env, force_reset=True)
        run_id = resumed["run_id"]
    else:
        model = _model(
            algorithm, env, seed=task.metadata["seed"], device=learner_device
        )
        run_id = run_id or f"sb3-{algorithm}-{task.metadata['seed']}"
    initial = task.interactions.copy()
    remaining = total_interactions - initial["cumulative"]
    if initial["uncertain"] or model.num_timesteps != initial["cumulative"]:
        raise ValueError("This bounded example requires exact checkpoint accounting")
    validate_model(model)
    milestones = validate_schedule(
        model,
        algorithm,
        initial["cumulative"],
        total_interactions,
        save_every_updates,
        milestones,
    )
    snapshots = SnapshotIndex(
        task,
        output=output,
        run_id=run_id,
        algorithm=algorithm,
        target=total_interactions,
        interval=save_every_updates,
        milestones=milestones,
        resume=resume,
    )
    result = _learn_and_save(
        model,
        task,
        algorithm=algorithm,
        remaining=remaining,
        target=total_interactions,
        output=output,
        snapshots=snapshots,
        interval=save_every_updates,
        milestones=milestones,
    )
    return {
        **result,
        "algorithm": algorithm,
        "training_seed": task.metadata["seed"],
        "learner_recipe": _recipe(model, algorithm),
        "save_every_updates": save_every_updates,
        "requested_milestones": list(milestones),
        "training_information_used": [],
        "actor_inputs": "49D blind proprioception and applied command",
        "critic_inputs": "same 49D blind input; no privileged supervision",
        "observation_normalization": False,
        "reward_normalization": False,
        "initial_interactions": initial,
        "update_counter_meaning": (
            "SB3 PPO epochs" if algorithm == "ppo" else "SB3 SAC gradient steps"
        ),
        "resumed_from": None if resume is None else str(Path(resume).resolve()),
        "learner_checkpoint_restored": resumed is not None,
        "resume_policy": "fresh physical episodes; completed external updates retained",
        "behavior_validated": False,
        "calibration_validated": False,
    }
