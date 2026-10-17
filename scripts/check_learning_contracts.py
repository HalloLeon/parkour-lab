"""Bounded native timeout, replay and physical joint-target saturation checks."""

import argparse
import sys
import traceback
from pathlib import Path

from parkour_lab.provenance import (
    file_sha256,
    package_source_identity,
    write_json,
    write_run_provenance,
)
from parkour_lab.runtime.session import exit_native_process, finish_session


def _setup(model, steps):
    _, callback = model._setup_learn(total_timesteps=steps * model.n_envs)
    callback.on_training_start({}, {})
    assert model.observation_space.shape == (49,)
    return callback


def _check_ppo(task, app):
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    from examples.sb3.environment import Go2VecEnv
    from scripts._learning_checks import record_native_steps, transition_receipt

    env = Go2VecEnv(task, app=app)
    model = PPO(
        "MlpPolicy",
        env,
        seed=task.metadata["seed"],
        device="cpu",
        n_steps=2,
        batch_size=2 * task.num_envs,
        n_epochs=1,
        gamma=0.9,
        policy_kwargs={"net_arch": [8]},
    )
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.fill_(0.125)
        model.policy.log_std.fill_(-20)
    values = []
    original = model.policy.predict_values

    def value(observation):
        values.append(observation.detach().clone())
        return 2 + 10 * observation[:, 33:34]

    callback = _setup(model, 2)
    model.policy.predict_values = value
    try:
        with record_native_steps(task, force_timeout=True) as records:
            assert model.collect_rollouts(env, callback, model.rollout_buffer, 2)
    finally:
        model.policy.predict_values = original
        callback.on_training_end()
    first = records[0]
    final = first["final_actor"][0].cpu().numpy()
    replacement = first["next_actor"][0].cpu().numpy()
    np.testing.assert_array_equal(values[0].cpu().numpy()[0], final)
    terminal_value = 2 + 10 * float(final[33])
    replacement_value = 2 + 10 * float(replacement[33])
    assert terminal_value != replacement_value
    expected = first["reward"].cpu().numpy().copy()
    expected[0] += model.gamma * terminal_value
    np.testing.assert_allclose(model.rollout_buffer.rewards[0], expected, rtol=1e-6)
    assert model._n_updates == 0 and model.num_timesteps == 2 * task.num_envs
    return dict(
        gamma=model.gamma,
        terminal_value=terminal_value,
        replacement_value=replacement_value,
        expected_bootstrapped_rewards=expected.tolist(),
        actual_rollout_rewards=model.rollout_buffer.rewards[0].tolist(),
        optimizer_updates=model._n_updates,
        learner_observation_shape=list(model.observation_space.shape),
        transitions=[transition_receipt(record) for record in records],
    )


def _check_sac(task, app):
    import numpy as np
    import torch
    from stable_baselines3 import SAC

    from examples.sb3.environment import Go2VecEnv
    from scripts._learning_checks import record_native_steps, transition_receipt

    env = Go2VecEnv(task, app=app)
    model = SAC(
        "MlpPolicy",
        env,
        seed=task.metadata["seed"],
        device="cpu",
        learning_starts=100,
        buffer_size=32,
        train_freq=1,
        policy_kwargs={"net_arch": [8]},
    )
    callback = _setup(model, 1)
    try:
        with record_native_steps(task, force_timeout=True) as records:
            rollout = model.collect_rollouts(
                env,
                callback,
                model.train_freq,
                model.replay_buffer,
                learning_starts=model.learning_starts,
            )
            assert rollout.continue_training
            first = records[0]
            replay = model.replay_buffer
            assert replay.pos == 1
            np.testing.assert_array_equal(
                replay.dones[0],
                first["terminated"].cpu().numpy() | first["truncated"].cpu().numpy(),
            )
            np.testing.assert_array_equal(
                replay.timeouts[0], first["truncated"].cpu().numpy()
            )
            np.testing.assert_array_equal(
                replay.next_observations[0, 0], first["final_actor"][0].cpu().numpy()
            )
            np.testing.assert_array_equal(
                replay.rewards[0], first["reward"].cpu().numpy()
            )
            assert not np.array_equal(
                replay.next_observations[0, 0], first["next_actor"][0].cpu().numpy()
            )
            rng = np.random.get_state()
            try:
                np.random.seed(0)
                sample = replay.sample(128)
            finally:
                np.random.set_state(rng)
            assert not sample.dones.any()
            terminal = torch.as_tensor(replay.next_observations[0, 0])
            assert (sample.next_observations == terminal).all(dim=1).any()
            assert app.is_running()
            saved_next = replay.next_observations[0].copy()
            task.step(torch.zeros(task.num_envs, 12, device=task.device))
            np.testing.assert_array_equal(replay.next_observations[0], saved_next)
    finally:
        callback.on_training_end()
    assert model._n_updates == 0 and model.num_timesteps == task.num_envs
    return dict(
        replay_dones=replay.dones[0].tolist(),
        replay_timeouts=replay.timeouts[0].tolist(),
        replay_terminal_actor=replay.next_observations[0, 0].tolist(),
        sampled_done_values=sample.dones.unique().cpu().tolist(),
        terminal_frame_seen_in_sample=True,
        optimizer_updates=model._n_updates,
        learner_observation_shape=list(model.observation_space.shape),
        transitions=[transition_receipt(record) for record in records],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1077)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 2 <= args.num_envs <= 8 or not 0 <= args.seed < 2**32:
        parser.error("Require 2–8 replicas and a uint32 seed")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="RUNNING",
        behavior_validated=False,
        calibration_validated=False,
        package_sources=package_source_identity(),
        host_sources={
            name: file_sha256(Path(__file__).with_name(name))
            for name in ("check_learning_contracts.py", "_learning_checks.py")
        },
        interpreter=dict(executable=sys.executable, version=sys.version),
        training_information_used=[],
        intervention="First step of each check sets row 0 episode age to horizon minus one; physical state and horizon configuration stay unchanged",
    )

    def publish():
        write_json(output / "report.json", report)

    app = task = None
    code = 2
    try:
        write_run_provenance(output, __file__)
        publish()
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, device=args.device, fast_shutdown=False).app
        import torch
        import yaml

        from examples.sb3 import environment
        from parkour_lab.benchmark import Go2Config, build_config, make
        from scripts._learning_checks import check_saturation, dependency_evidence

        torch.set_num_threads(2)
        report["dependencies"] = dependency_evidence()
        report["example_sources"] = {
            module.__name__: dict(
                path=module.__file__, sha256=file_sha256(module.__file__)
            )
            for module in (sys.modules["examples.sb3"], environment)
        }
        cfg = build_config(
            Go2Config(
                task_id="Go2-Flat-v1",
                num_envs=args.num_envs,
                seed=args.seed,
                device=args.device,
            ),
            training=True,
        )
        (output / "resolved_env.yaml").write_text(
            yaml.dump(cfg.to_dict(), sort_keys=False)
        )
        task = make(cfg)
        report["metadata"] = task.metadata
        for name, check in (
            ("ppo", _check_ppo),
            ("sac", _check_sac),
            ("saturation", check_saturation),
        ):
            result = check(task, app)
            write_json(output / f"{name}.json", result)
            report[name] = dict(passed=True, witness=f"{name}.json")
            publish()
        assert task.interactions == dict(
            session=6 * args.num_envs, cumulative=6 * args.num_envs, uncertain=0
        )
        report.update(
            status="LEARNING_CONTRACTS_PASSED",
            control_steps=6,
            interactions=task.interactions,
            optimizer_updates=0,
            step_roles=dict(learner_collection=3, ownership_witness=1, saturation=2),
            learner_collection_interactions=dict(
                ppo=2 * args.num_envs, sac=args.num_envs
            ),
            transition_counts=task._env.transition_counts,
        )
        code = 0
    except BaseException as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        if task is not None:
            report["interactions"] = task.interactions
        code = finish_session(task, app, report, publish, code)
        if app is not None:
            exit_native_process(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
