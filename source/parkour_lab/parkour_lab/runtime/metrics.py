"""Bounded, device-side summaries of native rewards and outgoing transitions."""

import torch


REGIMES = ("initial_stop", "later_stop", "pivot", "planar", "arc")
PHYSICAL_METRICS = (
    "planar_error_m_s",
    "yaw_error_rad_s",
    "base_height_m",
    "joint_posture_l2_rad",
    "tilt_rad",
)


def diagnostic_state(env):
    """Compact simulator truth; height is relative to the ray hit beneath the base."""
    data = env.scene["robot"].data
    ground = env.scene["base_height_scanner"].data.ray_hits_w[:, 0, 2]
    gravity = data.projected_gravity_b
    return torch.cat(
        (
            data.root_lin_vel_b,
            data.root_ang_vel_b[:, 2:3],
            (data.root_pos_w[:, 2] - ground)[:, None],
            torch.linalg.vector_norm(
                data.joint_pos - data.default_joint_pos, dim=-1, keepdim=True
            ),
            torch.atan2(
                torch.linalg.vector_norm(gravity[:, :2], dim=-1), -gravity[:, 2]
            )[:, None],
        ),
        dim=1,
    ).detach()


class TransitionMetrics:
    """Aggregate each learning phase separately, without retaining step traces.

    Native term values are already weighted rates. Contributions apply the control
    timestep once. Regimes use the outgoing command, not a newly sampled command;
    a later stop follows any nonzero command in that episode, including a pivot.
    """

    def __init__(self, env):
        self.names = tuple(env.reward_manager.active_terms)
        self.dt = env.step_dt
        self.seen_command = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        self.sums, self.errors = {}, {}

    def reset(self):
        self.seen_command.zero_()

    def add(self, phase, command, reward, rates, state, terminated, truncated):
        if rates.shape != (len(command), len(self.names)) or state.shape != (
            len(command),
            7,
        ):
            raise ValueError("Native reward or diagnostic-state dimensions changed")
        planar, turning = (command[:, :2] != 0).any(-1), command[:, 2] != 0
        regime = self.seen_command.long()
        regime = torch.where(turning, 2, regime)
        regime = torch.where(planar, 3 + turning.long(), regime)
        self.seen_command |= planar | turning
        self.seen_command[terminated | truncated] = False
        values = torch.cat(
            (
                torch.stack(
                    (
                        torch.ones_like(reward),
                        terminated,
                        truncated,
                        terminated & truncated,
                        reward,
                    ),
                    dim=1,
                ),
                torch.linalg.vector_norm(state[:, :2] - command[:, :2], dim=-1)[
                    :, None
                ],
                (state[:, 3] - command[:, 2]).abs()[:, None],
                state[:, 4:],
                rates,
            ),
            dim=1,
        ).to(torch.float64)
        if phase not in self.sums:
            self.sums[phase] = values.new_zeros(len(REGIMES), values.shape[1])
            self.errors[phase] = values.new_zeros(2)
        self.sums[phase].index_add_(0, regime, values)
        difference = (rates.sum(-1) * self.dt - reward).abs()
        tolerance = 1e-6 + 1e-5 * rates.abs().sum(-1) * self.dt
        error = torch.stack((difference.max(), (difference / tolerance).max()))
        self.errors[phase] = torch.maximum(self.errors[phase], error)

    def drain(self):
        """Return one update's summaries; episode command history survives draining."""
        result = {}
        for phase, totals in self.sums.items():
            packed = torch.cat((totals.flatten(), self.errors[phase])).cpu()
            if not torch.isfinite(packed).all() or packed[-1] > 1:
                raise ValueError(
                    "Nonfinite diagnostics or native reward terms do not reconcile"
                )
            rows = packed[:-2].reshape(totals.shape).tolist()
            regimes = {}
            for name, row in zip(REGIMES, rows, strict=True):
                count = int(row[0])
                if not count:
                    continue
                regimes[name] = dict(
                    samples=count,
                    simulated_seconds=count * self.dt,
                    terminated=int(row[1]),
                    truncated=int(row[2]),
                    simultaneous=int(row[3]),
                    reward_contribution_sum=row[4],
                    physical_mean=dict(
                        zip(
                            PHYSICAL_METRICS,
                            (x / count for x in row[5:10]),
                            strict=True,
                        )
                    ),
                    reward_terms={
                        term: dict(
                            weighted_rate_mean=value / count,
                            contribution_sum=value * self.dt,
                        )
                        for term, value in zip(self.names, row[10:], strict=True)
                    },
                )
            result[phase] = dict(
                regimes=regimes, reward_reconciliation_abs_max=float(packed[-2])
            )
        self.sums.clear()
        self.errors.clear()
        return result
