"""A bounded call to external PPO/SAC learning and their own checkpoint APIs."""

import math
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure

from .checkpoint import load_checkpoint, save_checkpoint
from .environment import Go2VecEnv
from .policy import SB3FramePolicy

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


def _learn_and_save(model, task, *, remaining, target, output, run_id):
    before = SB3FramePolicy(model).state_sha256()
    updates_before = model._n_updates
    model.set_logger(configure(str(output / "metrics"), ["json"]))
    try:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)
        after = SB3FramePolicy(model).state_sha256()
        counts = task.interactions
        if (
            counts["cumulative"] != target
            or counts["session"] != remaining
            or counts["uncertain"]
            or model.num_timesteps != target
            or model._n_updates <= updates_before
            or before == after
        ):
            raise RuntimeError("Learning, parameter change or interaction check failed")
        losses = _losses(model)
        model.logger.dump(step=model.num_timesteps)
        milestones = tuple(sorted({target - remaining, target} - {0}))
        receipt = save_checkpoint(
            model, task, output / "checkpoint", run_id=run_id, milestones=milestones
        )
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
):
    """Complete whole collection blocks; never relabel overshoot as the allowance.

    Constant schedules avoid resetting a decay horizon between short sessions.
    Each call saves after complete external updates. Physical episodes restart
    on resume; no partial PPO rollout or live simulator state is restored.
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
    quantum = env.num_envs * (model.n_steps if algorithm == "ppo" else 1)
    if remaining <= 0 or remaining % quantum:
        raise ValueError(
            f"Remaining allowance must be a positive multiple of {quantum} transitions"
        )
    if initial["uncertain"] or model.num_timesteps != initial["cumulative"]:
        raise ValueError("This bounded example requires exact checkpoint accounting")
    result = _learn_and_save(
        model,
        task,
        remaining=remaining,
        target=total_interactions,
        output=output,
        run_id=run_id,
    )
    return {
        **result,
        "algorithm": algorithm,
        "training_seed": task.metadata["seed"],
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
