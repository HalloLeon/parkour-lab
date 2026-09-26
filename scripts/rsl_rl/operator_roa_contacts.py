"""Teacher-only net-normal foot forces; no tangential forces or support labels."""

import torch

VERSION = "operator_roa_contact_context_v1"
FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FORCE_SCALE_N = 100.0


def recipe():
    return {
        "version": VERSION,
        "feet": list(FEET),
        "width": 12,
        "source": "contact_forces.data.net_forces_w; summed normal force per named foot, excluding tangential/friction forces",
        "frame": "root link body; world vector rotated by current root_link_quat_w inverse",
        "transform": "tanh(force_body / force_scale_n), foot-major XYZ",
        "force_scale_n": FORCE_SCALE_N,
        "reset": "Explicit zero on reset rows after copying; sensor buffers unchanged",
        "sampling": "Once per delivered pre-action observation; no force history or future samples",
        "scope": "Privileged teacher and detached adaptation labels only; absent from exported causal controller",
    }


class ContactFeatures:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.sensor = env.scene["contact_forces"]
        if any(
            names.count(foot) != 1
            for names in (self.robot.body_names, self.sensor.body_names)
            for foot in FEET
        ):
            raise ValueError(
                "Require each named teacher foot in robot and contact sensor"
            )
        self.ids = [self.sensor.body_names.index(foot) for foot in FEET]
        self.calls = self.reset_rows = 0
        self.nonzero_rows = torch.zeros((), dtype=torch.long, device=env.device)
        self.maximum = torch.zeros((), device=env.device)

    def sample(self, reset):
        from isaaclab.utils.math import quat_apply_inverse

        count = self.env.num_envs
        forces = self.sensor.data.net_forces_w[:, self.ids].detach().clone()
        quat = self.robot.data.root_link_quat_w.detach().clone()
        if (
            forces.shape != (count, 4, 3)
            or quat.shape != (count, 4)
            or reset.shape != (count,)
            or reset.dtype != torch.bool
            or reset.device != forces.device
            or not torch.isfinite(forces).all()
            or not torch.isfinite(quat).all()
            or not torch.allclose(
                quat.norm(dim=1), torch.ones_like(quat[:, 0]), atol=1e-4, rtol=0
            )
        ):
            raise ValueError("Invalid native teacher contact observation")
        body = quat_apply_inverse(
            quat[:, None].expand(-1, 4, -1).reshape(-1, 4), forces.reshape(-1, 3)
        ).reshape(count, 12)
        if not torch.isfinite(body).all():
            raise ValueError("Non-finite body-frame teacher contact force")
        features = torch.tanh(body / FORCE_SCALE_N)
        # A reset marks the sensor outdated; reading .data may fetch stale PhysX
        # terminal forces again. Never supervise the reset frame with those forces.
        features[reset] = 0
        self.calls += 1
        self.reset_rows += int(reset.sum())
        self.nonzero_rows += features.ne(0).any(dim=1).sum()
        self.maximum = torch.maximum(self.maximum, features.abs().max())
        return features

    def report(self):
        return {
            "recipe": recipe(),
            "num_envs": self.env.num_envs,
            "contact_body_ids": self.ids,
            "observation_calls": self.calls,
            "reset_rows_zeroed": self.reset_rows,
            "nonzero_rows": int(self.nonzero_rows),
            "maximum_absolute_feature": float(self.maximum),
        }


def validate_report(report, *, num_envs, steps, full_resets=4):
    """Bind source loading to complete native collection, not only a feature flag."""
    if not isinstance(report, dict):
        raise ValueError("Require a teacher contact observation receipt")
    calls, resets, nonzero = (
        report[key]
        for key in ("observation_calls", "reset_rows_zeroed", "nonzero_rows")
    )
    ids = report["contact_body_ids"]
    if (
        type(full_resets) is not int
        or full_resets < 1
        or report["recipe"] != recipe()
        or type(report["num_envs"]) is not int
        or report["num_envs"] != num_envs
        or any(type(value) is not int for value in (calls, resets, nonzero))
        or calls != steps + full_resets
        or not full_resets * num_envs <= resets <= calls * num_envs
        or not 0 < nonzero <= calls * num_envs - resets
        or not isinstance(ids, list)
        or len(ids) != 4
        or any(type(index) is not int or index < 0 for index in ids)
        or len(set(ids)) != 4
        or type(report["maximum_absolute_feature"]) not in (int, float)
        or not 0 < report["maximum_absolute_feature"] <= 1
    ):
        raise ValueError("Incomplete native teacher contact observation receipt")
