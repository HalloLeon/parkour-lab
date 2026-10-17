"""Native transition witnesses for the bounded learner contract check."""

import hashlib
import json
from contextlib import contextmanager


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _check_noise(before, after):
    import numpy as np

    for original, current in zip(
        before["generators"], after["generators"], strict=True
    ):
        generator = np.random.Generator(np.random.PCG64())
        generator.bit_generator.state = original["observation-noise"]
        generator.uniform(-1.0, 1.0, size=30)
        assert generator.bit_generator.state == current["observation-noise"]


def _buffers(native, ids):
    return {
        "action": native.action_manager.action[ids].detach().clone(),
        "processed_target": native.action_manager.get_term("joint_pos")
        .processed_actions[ids]
        .detach()
        .clone(),
        "joint_target": native.scene["robot"]
        .data.joint_pos_target[ids]
        .detach()
        .clone(),
    }


def _check_delivery(task, raw, transition, witness):
    import torch

    observation, _, terminated, truncated, info = transition
    low = torch.as_tensor(task.single_action_space.low, device=task.device)
    high = torch.as_tensor(task.single_action_space.high, device=task.device)
    delivered = raw.clamp(low, high)
    target = task._env.scene["robot"].data.default_joint_pos + 0.25 * delivered
    ended = terminated | truncated
    live = (~ended).nonzero(as_tuple=False).flatten()
    for name, value in _buffers(task._env, live).items():
        expected = delivered[live] if name == "action" else target[live]
        assert torch.equal(value, expected), f"Live native {name} differs"
    assert torch.equal(observation["actor"][live, 33:45], delivered[live])
    ids = info["final_env_ids"]
    if len(ids):
        assert torch.equal(ids, witness["ids"])
        for name, value in witness["buffers"].items():
            expected = delivered[ids] if name == "action" else target[ids]
            assert torch.equal(value, expected), f"Outgoing native {name} differs"
        assert torch.equal(info["final_observation"]["actor"], witness["actor"])
        assert torch.equal(witness["actor"][:, 33:45], delivered[ids])
    bounds = task._env.scene["robot"].data.soft_joint_pos_limits
    assert ((target >= bounds[..., 0]) & (target <= bounds[..., 1])).all()
    clipped = delivered != raw
    return dict(
        requested_action=raw.detach().clone(),
        delivered_action=delivered.detach().clone(),
        target_rad=target.detach().clone(),
        saturated_joints=int(clipped.sum()),
        saturated_rows=int(clipped.any(dim=1).sum()),
    )


def check_owned(records):
    import torch

    for record in records:
        if record["final_ref"] is not None:
            assert torch.equal(record["final_ref"], record["final_actor"])


@contextmanager
def record_native_steps(task, *, force_timeout=False):
    """Observe real steps; force only the first row's episode age, not its state."""
    import torch

    from parkour_lab.benchmark.observations import actor_frame

    native, records, witness = task._env, [], {}
    original_step, original_reset = task.step, native._reset_idx
    horizon = native.max_episode_length

    def before_reset(ids):
        if native._capturing:
            before = native.parkour_randomization.state_dict()
            clean = native.observation_manager.compute(update_history=False)["proprio"][
                ids
            ].clone()
            witness.update(
                ids=ids.clone(),
                actor=actor_frame(native, clean, final_env_ids=ids),
                clean_actor=clean,
                buffers=_buffers(native, ids),
                episode_age=native.episode_length_buf[ids].clone(),
                terminated=native.reset_terminated[ids].clone(),
                truncated=native.reset_time_outs[ids].clone(),
            )
            assert native.parkour_randomization.state_dict() == before
        original_reset(ids)

    def step(raw):
        witness.clear()
        forced = force_timeout and not records
        if forced:
            native.episode_length_buf[0] = horizon - 1
        before = task.interactions
        noise = native.parkour_randomization.state_dict()
        frames = native._causal_noise_decisions
        transition = original_step(raw)
        observation, reward, terminated, truncated, info = transition
        assert observation["actor"].shape == (task.num_envs, 49)
        assert torch.isfinite(observation["actor"]).all()
        after = task.interactions
        assert all(
            after[key] == before[key] + task.num_envs
            for key in ("session", "cumulative")
        )
        assert after["uncertain"] == before["uncertain"]
        assert info["interactions"] == after
        current_noise = native.parkour_randomization.state_dict()
        _check_noise(noise, current_noise)
        assert native._causal_noise_decisions == frames + 1
        assert native.max_episode_length == horizon
        final = info["final_observation"]
        ids = info["final_env_ids"]
        assert torch.equal(
            ids, (terminated | truncated).nonzero(as_tuple=False).flatten()
        )
        if forced:
            assert ids.tolist() == [0] and truncated[0] and not terminated.any(), (
                f"Expected pure row-0 timeout, got final IDs {ids.tolist()}, "
                f"terminated={terminated.tolist()}, truncated={truncated.tolist()}"
            )
            assert not truncated[1:].any()
            assert native.episode_length_buf[0] == 0
            assert witness["episode_age"].tolist() == [horizon]
            assert witness["truncated"].tolist() == [True]
            assert witness["terminated"].tolist() == [False]
        final_actor = None if final is None else final["actor"]
        record = dict(
            forced_timeout=forced,
            before=before,
            after=after,
            reward=reward.detach().clone(),
            terminated=terminated.clone(),
            truncated=truncated.clone(),
            ids=ids.clone(),
            final_ref=final_actor,
            final_actor=None if final_actor is None else final_actor.clone(),
            next_actor=observation["actor"].clone(),
            outgoing_native=dict(witness),
            sampler_before_sha256=_digest(noise),
            sampler_after_sha256=_digest(current_noise),
            noise_frames_before=frames,
            noise_frames_after=native._causal_noise_decisions,
            **_check_delivery(task, raw, transition, witness),
        )
        records.append(record)
        check_owned(records)
        return transition

    task.step, native._reset_idx = step, before_reset
    try:
        yield records
    finally:
        task.step, native._reset_idx = original_step, original_reset


def transition_receipt(record):
    """Serialize witnesses, omitting only the live reference used for ownership checks."""

    def convert(value):
        if isinstance(value, dict):
            return {
                name: convert(item)
                for name, item in value.items()
                if name != "final_ref"
            }
        return value.detach().cpu().tolist() if hasattr(value, "detach") else value

    return convert(record)


def check_saturation(task, app):
    import torch

    task.reset()
    with record_native_steps(task, force_timeout=True) as records:
        for sign in (1.0, -1.0):
            assert app.is_running()
            raw = torch.full(
                (task.num_envs, 12), sign * 1e6, dtype=torch.float32, device=task.device
            )
            raw[1::2] *= -1
            task.step(raw)
    assert all(record["saturated_joints"] == 12 * task.num_envs for record in records)
    assert all(record["saturated_rows"] == task.num_envs for record in records)
    return dict(
        request_magnitude=1e6,
        coverage="both bounds of every joint, with live and outgoing native readbacks",
        transitions=[transition_receipt(record) for record in records],
    )


def dependency_evidence():
    from parkour_lab.provenance import dependency_identity

    return {
        name: dependency_identity(name, distribution)
        for name, distribution in (
            ("isaaclab", "isaaclab"),
            ("isaaclab_assets", "isaaclab_assets"),
            ("isaaclab_tasks", "isaaclab_tasks"),
            ("isaacsim", "isaacsim"),
            ("stable_baselines3", "stable-baselines3"),
            ("torch", "torch"),
            ("gymnasium", "gymnasium"),
            ("numpy", "numpy"),
        )
    }
