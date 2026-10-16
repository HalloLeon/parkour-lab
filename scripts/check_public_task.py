"""External-consumer example and bounded native Go2 task contract check."""

import argparse
import json
import sys
import traceback
from pathlib import Path

from parkour_lab.provenance import (
    dependency_identity,
    package_source_identity,
    write_json,
    write_run_provenance,
)
from parkour_lab.runtime.session import exit_native_process, finish_session


def _check_observation(task, observation, count):
    import torch

    assert set(observation) == ({"actor", "training"} if task.training else {"actor"})
    actor = observation["actor"]
    assert actor.shape == (count, 49) and actor.dtype == torch.float32
    assert torch.isfinite(actor).all()
    if task.training:
        fields = task.metadata["training_information"]["fields"]
        assert set(observation["training"]) == set(fields)
        for name, spec in fields.items():
            value = observation["training"][name]
            assert value.shape == (count, *spec["shape"])
            assert value.dtype == getattr(torch, spec["dtype"])
            assert torch.isfinite(value).all() and not value.requires_grad
    else:
        assert "training_information" not in task.metadata


def _physical_witness(native, ids):
    """Acceptance-host readback only; never supplied to the example controller."""
    import torch
    from isaaclab.utils.math import quat_apply_inverse

    from parkour_lab.control.proprioception import FOOT_NAMES

    data, sensor = native.scene["robot"].data, native.scene["contact_forces"]
    forces = sensor.data.net_forces_w[ids][
        :, [sensor.body_names.index(n) for n in FOOT_NAMES]
    ].clone()
    quaternion = data.root_link_quat_w[ids, None].expand(-1, 4, -1)
    forces = quat_apply_inverse(
        quaternion.reshape(-1, 4), forces.reshape(-1, 3)
    ).reshape(-1, 4, 3)
    valid_force = native.episode_length_buf[ids] > 0
    hits = native.scene["height_scanner"].data.ray_hits_w[ids].clone()
    valid = torch.isfinite(hits).all(-1)
    height = data.root_link_pos_w[ids, 2:3] - 0.3 - hits[..., 2]
    return {
        "root_com_linear_velocity_b": data.root_com_lin_vel_b[ids].clone(),
        "joint_velocity": data.joint_vel[ids].clone(),
        "joint_position_relative_default": (data.joint_pos - data.default_joint_pos)[
            ids
        ].clone(),
        "foot_net_forces_b": torch.where(valid_force[:, None, None], forces, 0.0),
        "foot_force_valid": valid_force[:, None].clone(),
        "terrain_height": torch.where(valid, height, 0.5).clamp(-0.5, 0.5),
        "terrain_valid": valid,
    }


def _values(observation):
    yield "actor", observation["actor"]
    for name, value in observation.get("training", {}).items():
        yield name, value


def _dependency_evidence(output):
    import isaaclab

    identities = {
        module: dependency_identity(module, distribution)
        for module, distribution in (
            ("isaaclab", "isaaclab"),
            ("isaaclab_assets", "isaaclab_assets"),
            ("isaaclab_tasks", "isaaclab_tasks"),
            ("isaacsim", "isaacsim"),
            ("torch", "torch"),
            ("gymnasium", "gymnasium"),
            ("numpy", "numpy"),
            ("yaml", "PyYAML"),
            ("trimesh", "trimesh"),
        )
    }
    if "commit" in identities["isaaclab"]:
        write_run_provenance(output / "sdk", isaaclab.__file__)
    return identities


def _resume_and_reset(task, args, report):
    import torch

    native = task._env  # Private checks remain inside this acceptance host.
    initial = uncertain = 0
    run_id = args.run_id
    if args.resume:
        record = json.loads(args.resume.read_text())
        restored = task.restore(record, checkpoint_id=args.checkpoint_id)
        assert native.parkour_randomization.state_dict() == record["sampler"]
        assert (
            native.command_manager.get_term("base_velocity").sampling_report
            == record["sampling_report"]
        )
        initial, run_id = restored["interactions"]["cumulative"], restored["run_id"]
        uncertain = restored["interactions"]["uncertain"]
        report["restored_task_progress"] = restored
    observation, info = task.reset(seed=args.seed)
    _check_observation(task, observation, task.num_envs)
    assert info["interactions"] == dict(
        session=0, cumulative=initial, uncertain=uncertain
    )
    assert set(native.observation_manager.active_terms) == {"proprio"}
    assert native.capture_training_information is task.training
    if task.training:
        current = _physical_witness(
            native, torch.arange(task.num_envs, device=task.device)
        )
        for name, value in current.items():
            torch.testing.assert_close(observation["training"][name], value)
    else:
        streams = native.parkour_randomization.state_dict()
        observation = task.set_command(
            torch.zeros(task.num_envs, 3, device=task.device, dtype=torch.float32)
        )
        assert native.parkour_randomization.state_dict() == streams
        _check_observation(task, observation, task.num_envs)
    return initial, uncertain, run_id


def _check_transition(task, action, transition, witness, expected_counts):
    import torch

    observation, reward, terminated, truncated, info = transition
    _check_observation(task, observation, task.num_envs)
    assert set(info) == {"final_env_ids", "final_observation", "interactions"}
    assert torch.isfinite(reward).all()
    assert terminated.dtype == truncated.dtype == torch.bool
    ended = terminated | truncated
    ids = ended.nonzero(as_tuple=False).flatten()
    assert torch.equal(ids, info["final_env_ids"])
    assert info["interactions"] == expected_counts
    data = task._env.scene["robot"].data
    targets, bounds = data.joint_pos_target, data.soft_joint_pos_limits
    assert ((targets >= bounds[..., 0]) & (targets <= bounds[..., 1])).all()
    torch.testing.assert_close(
        targets[~ended], (data.default_joint_pos + 0.25 * action)[~ended]
    )
    final = info["final_observation"]
    if len(ids):
        _check_observation(task, final, len(ids))
        assert torch.equal(ids, witness["ids"])
        torch.testing.assert_close(final["actor"][:, 6:9], witness["command"])
        torch.testing.assert_close(final["actor"][:, 33:45], action[ids])
        if task.training:
            for name, value in witness["physical"].items():
                torch.testing.assert_close(
                    final["training"][name], value, atol=1e-5, rtol=1e-5
                )
    else:
        assert final is None


def _check_partial_reset(task):
    import torch

    native = task._env
    survivors, before = native.episode_length_buf[1:].clone(), task.interactions
    observation, _ = task.reset(env_ids=torch.tensor([0], device=task.device))
    _check_observation(task, observation, task.num_envs)
    assert native.episode_length_buf[0] == 0
    assert torch.equal(native.episode_length_buf[1:], survivors)
    assert task.interactions == before


def _save_final(task, transition, witness, output):
    observation, _, _, truncated, info = transition
    assert truncated[0] and task._env.episode_length_buf[0] == 0
    assert not truncated[1:].any()
    final = info["final_observation"]
    write_json(
        output / "outgoing_final.json",
        {name: value.cpu().tolist() for name, value in _values(final)},
    )
    if task.training:
        assert not observation["training"]["foot_force_valid"][0].any()
        write_json(
            output / "physical_witness.json",
            {name: value.cpu().tolist() for name, value in witness["physical"].items()},
        )
    return final, {name: value.clone() for name, value in _values(final)}


def _collect(task, app, output, initial, uncertain):
    import torch

    native, witness = task._env, {}
    original_reset = native._reset_idx

    def before_reset(ids):
        if native._capturing:
            witness.update(
                ids=ids.clone(),
                command=native.command_manager.get_command("base_velocity")[
                    ids
                ].clone(),
            )
            if task.training:
                witness["physical"] = _physical_witness(native, ids)
        original_reset(ids)

    action = torch.zeros(task.num_envs, 12, device=task.device, dtype=torch.float32)
    action[:, 0] = 0.02  # A 5 mrad shift from the default first-joint target.
    assert (
        action >= torch.as_tensor(task.single_action_space.low, device=task.device)
    ).all()
    assert (
        action <= torch.as_tensor(task.single_action_space.high, device=task.device)
    ).all()
    saved = copies = None
    native._reset_idx = before_reset
    try:
        with torch.inference_mode():
            for step in range(32):
                assert app.is_running(), "Application stopped before the contract check"
                witness.clear()
                if step == 2:
                    native.episode_length_buf[0] = native.max_episode_length - 1
                transition = task.step(action)
                counts = dict(
                    session=(step + 1) * task.num_envs,
                    cumulative=initial + (step + 1) * task.num_envs,
                    uncertain=uncertain,
                )
                _check_transition(task, action, transition, witness, counts)
                if saved is not None:
                    assert all(
                        torch.equal(value, copies[name])
                        for name, value in _values(saved)
                    )
                if step == 2:
                    saved, copies = _save_final(task, transition, witness, output)
                if step == 16:
                    _check_partial_reset(task)
    finally:
        native._reset_idx = original_reset


def _run(task, app, args, output, report):
    initial, uncertain, run_id = _resume_and_reset(task, args, report)
    _collect(task, app, output, initial, uncertain)
    if task.training:
        record = task.progress(
            run_id=run_id,
            checkpoint_id=args.checkpoint_id,
            milestones=(task.interactions["cumulative"],),
        )
        write_json(output / "progress.json", record)
        assert json.loads((output / "progress.json").read_text()) == record
    report.update(
        status="TASK_CHECK_PASSED",
        control_steps=32,
        interactions=task.interactions,
        timeout_row=0,
        partial_reset_rows=[0],
        metadata=task.metadata,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("train", "eval"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1077)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume", type=Path, help="Training task-progress JSON; restores no learner"
    )
    parser.add_argument("--checkpoint-id", default="no-learner-contract-fixture")
    parser.add_argument("--run-id", default="public-task-contract")
    args = parser.parse_args()
    if args.num_envs < 2 or not 0 <= args.seed < 2**32:
        parser.error("Require at least two replicas and a uint32 seed")
    if args.resume and args.role != "train":
        parser.error("Task progress is training-only")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="RUNNING",
        role=args.role,
        behavior_validated=False,
        learner_checkpoint_restored=False,
        package_sources=package_source_identity(),
        interpreter=dict(executable=sys.executable, version=sys.version),
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
        import yaml

        from parkour_lab.benchmark import Go2Config, build_config, make
        from parkour_lab.environments.dynamics import dynamics_report

        options = Go2Config(num_envs=args.num_envs, seed=args.seed, device=args.device)
        cfg = build_config(options, training=args.role == "train")
        (output / "resolved_env.yaml").write_text(
            yaml.dump(cfg.to_dict(), sort_keys=False)
        )
        report["dependencies"] = _dependency_evidence(output)
        task = make(cfg)
        report["physical_readback"] = dynamics_report(task._env)
        _run(task, app, args, output, report)
        code = 0
    except BaseException as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
    finally:
        code = finish_session(task, app, report, publish, code)
        if app is not None:
            exit_native_process(code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
