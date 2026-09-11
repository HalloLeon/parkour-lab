"""Tensor-only command-phase accounting; never used to compute policy rewards."""

from __future__ import annotations

import torch

PHASE_NAMES = (
    "translation",
    "restart",
    "stop",
    "pivot_acquisition",
    "pivot_sustained",
    "terminal_hold",
)
PHASE_REWARD_TERMS = (
    "stationary_velocity_tracking",
    "stationary_planar_motion",
    "action_rate_l2",
    "action_target_overflow_l2",
    "joint_torques_l2",
    "ang_vel_xy_l2",
    "flat_orientation_l2",
    "stable_orientation_l2",
    "upright_orientation_l2",
    "supported_orientation_l2",
    "joint_deviation_l2",
    "lin_vel_z_l2",
    "supported_vertical_velocity_l2",
    "feet_slide",
    "feet_stumble",
    "feet_edge",
    "undesired_contact",
    "chassis_contact",
    "physical_failure",
    "base_clearance_below",
)
PHASE_SIGNAL_NAMES = (
    (
        "command_yaw_rate_rad_s",
        "achieved_yaw_rate_rad_s",
        "abs_command_yaw_rate_rad_s",
        "aligned_yaw_rate_rad_s",
        "abs_yaw_error_rad_s",
        "planar_speed_m_s",
    )
    + tuple(f"reward_{name}_rate" for name in PHASE_REWARD_TERMS)
    + (
        "reward_pivot_yaw_rate",
        "reward_pivot_stability_rate",
        "reward_total_rate",
    )
)


class CommandPhaseDiagnostics:
    """Accumulate actual rollout samples, including unfinished episodes.

    Episode resets clear only command clocks. Draining clears only interval
    sums, so neither asynchronous resets nor PPO rollout boundaries bias phase
    exposure. Pivot acquisition/restart cover the first second of a command;
    pivot sign or magnitude changes restart acquisition. Translation at episode
    start is not a restart. Terminal tapering is translation until exact hold.
    """

    def __init__(self, num_envs: int, device: str, step_dt: float) -> None:
        self.step_dt = float(step_dt)
        self._first_second_steps = max(1, round(1.0 / self.step_dt))
        self._mode = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        self._age_steps = torch.zeros_like(self._mode)
        self._restart = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self._last_yaw = torch.zeros(num_envs, device=device)
        self._sums = torch.zeros(
            (len(PHASE_NAMES), 1 + len(PHASE_SIGNAL_NAMES)),
            device=device,
        )

    @torch.no_grad()
    def update(
        self,
        preferred_speed: torch.Tensor,
        target_speed: torch.Tensor,
        target_yaw: torch.Tensor,
        achieved_yaw: torch.Tensor,
        planar_speed: torch.Tensor,
        reward_rates: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Modes: translating=0, intentional stop=1, pivot=2, terminal hold=3.
        mode = torch.zeros_like(self._mode)
        mode = torch.where(preferred_speed.eq(0), 1, mode)
        mode = torch.where(preferred_speed.eq(0) & target_yaw.ne(0), 2, mode)
        mode = torch.where(preferred_speed.gt(0) & target_speed.eq(0), 3, mode)
        changed = mode.ne(self._mode) | (mode.eq(2) & target_yaw.ne(self._last_yaw))
        self._restart.copy_(
            torch.where(
                changed,
                mode.eq(0) & (self._mode.eq(1) | self._mode.eq(2)),
                self._restart,
            )
        )
        self._age_steps.copy_(torch.where(changed, 1, self._age_steps + 1))
        self._mode.copy_(mode)
        self._last_yaw.copy_(target_yaw)
        early = self._age_steps <= self._first_second_steps
        phase = torch.zeros_like(mode)
        phase = torch.where(mode.eq(0) & self._restart & early, 1, phase)
        phase = torch.where(mode.eq(1), 2, phase)
        phase = torch.where(mode.eq(2) & early, 3, phase)
        phase = torch.where(mode.eq(2) & ~early, 4, phase)
        phase = torch.where(mode.eq(3), 5, phase)
        signals = {
            "command_yaw_rate_rad_s": target_yaw,
            "achieved_yaw_rate_rad_s": achieved_yaw,
            "abs_command_yaw_rate_rad_s": target_yaw.abs(),
            # Opposite turns must not cancel in the training tracking metric.
            "aligned_yaw_rate_rad_s": achieved_yaw * target_yaw.sign(),
            "abs_yaw_error_rad_s": (achieved_yaw - target_yaw).abs(),
            "planar_speed_m_s": planar_speed,
            **reward_rates,
        }
        samples = torch.stack(
            [torch.ones_like(target_speed)]
            + [signals[name] for name in PHASE_SIGNAL_NAMES],
            dim=-1,
        ).to(dtype=self._sums.dtype)
        # Fixed-size GPU reduction. No .item(), CPU copy, or per-environment loop.
        self._sums.index_add_(0, phase, samples)
        return {
            "phase_id": phase,
            "phase_time_s": self._age_steps * self.step_dt,
            **signals,
        }

    def reset(self, env_ids: object = slice(None)) -> None:
        """Reset episode clocks without dropping samples collected for logging."""
        self._mode[env_ids] = -1
        self._age_steps[env_ids] = 0
        self._restart[env_ids] = False
        self._last_yaw[env_ids] = 0

    def drain(self) -> dict[str, torch.Tensor]:
        """Return raw sums/counts and their ratios once per training rollout.

        Empty phases have zero count/mean, not evidence of perfect tracking.
        Reward columns are signed weighted rates, not integrated rewards.
        Pivot components decompose stationary tracking; do not add them twice.
        """
        sums = self._sums.clone()
        self._sums.zero_()
        total = sums[:, 0].sum()
        result = {"step_count": total}
        for row, phase in enumerate(PHASE_NAMES):
            count = sums[row, 0]
            result[f"{phase}/step_count"] = count
            result[f"{phase}/fraction"] = count / total.clamp_min(1)
            for col, name in enumerate(PHASE_SIGNAL_NAMES, start=1):
                result[f"{phase}/{name}_sum"] = sums[row, col]
                result[f"{phase}/{name}_mean"] = sums[row, col] / count.clamp_min(1)
        return result
