"""Settled nonzero contact and owned outgoing force-frame witnesses."""

from parkour_lab.provenance import write_json

from .geometry import assignments


def _contact_sample(native, ids):
    from isaaclab.utils.math import quat_apply_inverse

    from parkour_lab.control.proprioception import FOOT_NAMES

    sensor, data = native.scene["contact_forces"], native.scene["robot"].data
    forces = sensor.data.net_forces_w[ids][
        :, [sensor.body_names.index(n) for n in FOOT_NAMES]
    ].clone()
    quaternion = data.root_link_quat_w[ids].clone()
    body = quat_apply_inverse(
        quaternion[:, None].expand(-1, 4, -1).reshape(-1, 4), forces.reshape(-1, 3)
    )
    return dict(
        force_w=forces, quaternion_wxyz=quaternion, force_b=body.reshape(-1, 4, 3)
    )


def check_contact(task, app, output):
    import torch

    native, ending, witness = task._env, None, {}
    initial = task.interactions
    flat = [row["env_id"] for row in assignments(native) if row["family"] == "Flat"]
    original_reset = native._reset_idx

    def before_reset(ids):
        if native._capturing and ending is not None and (ids == ending).any():
            witness.update(_contact_sample(native, ids))
            witness["ids"] = ids.clone()
        original_reset(ids)

    native._reset_idx = before_reset
    try:
        action = torch.zeros(task.num_envs, 12, device=task.device)
        for step in range(160):
            assert app.is_running()
            if ending is not None:
                native.episode_length_buf[ending] = native.max_episode_length - 1
            observation, reward, _, truncated, info = task.step(action)
            assert set(observation) == (
                {"actor", "training"} if task.training else {"actor"}
            )
            assert (
                observation["actor"].shape == (20, 49) and torch.isfinite(reward).all()
            )
            assert task.interactions == dict(
                session=initial["session"] + (step + 1) * 20,
                cumulative=initial["cumulative"] + (step + 1) * 20,
                uncertain=initial["uncertain"],
            )
            if ending is not None:
                assert truncated[ending]
                index = int((witness["ids"] == ending).nonzero()[0])
                assert witness["force_w"][index].norm(dim=-1).max() > 1.5
                if task.training:
                    torch.testing.assert_close(
                        info["final_observation"]["training"]["foot_net_forces_b"],
                        witness["force_b"],
                        atol=1e-5,
                        rtol=1e-5,
                    )
                    assert info["final_observation"]["training"]["foot_force_valid"][
                        index
                    ].all()
                    assert not observation["training"]["foot_force_valid"][ending].any()
                write_json(
                    output / "contact_witness.json",
                    {name: value.cpu().tolist() for name, value in witness.items()},
                )
                return dict(
                    control_steps=step + 1,
                    nonzero_contact_env=ending,
                    interactions=task.interactions,
                )
            if step >= 31:
                samples = _contact_sample(
                    native, torch.tensor(flat, device=task.device)
                )
                nonzero = samples["force_w"].norm(dim=-1).amax(-1) > 1.5
                if nonzero.any():
                    ending = flat[int(nonzero.nonzero()[0])]
        raise AssertionError("No settled nonzero Flat contact in 160 control calls")
    finally:
        native._reset_idx = original_reset
