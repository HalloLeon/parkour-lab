# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deployable travel and pivot commands plus local-heading diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from ._shared.runtime import _all_env_ids
from .navigation import geometry, route

if TYPE_CHECKING:
    from torch import Tensor


INTENT_COMMAND_NAME = "intent"
"""Command-manager term owning requested travel direction, speed, and pivot rate."""

COMMAND_PROFILES = (
    "mixed",
    "translation_only",
    "stop_restart",
    "pivot_restart",
)
"""Supported scripted command schedules."""

# Provisional fixed-evaluation threshold only. No runtime assistance bound is
# enforced until the later residual-producing student is implemented.
PROVISIONAL_ORACLE_RESIDUAL_THRESHOLD_RAD = math.radians(35.0)
EVALUATION_PIVOT_WINDOW_DURATION_S = 2.0
EVALUATION_RESTART_WINDOW_DURATION_S = EVALUATION_PIVOT_WINDOW_DURATION_S

__all__ = [
    "COMMAND_PROFILES",
    "EVALUATION_PIVOT_WINDOW_DURATION_S",
    "EVALUATION_RESTART_WINDOW_DURATION_S",
    "INTENT_COMMAND_NAME",
    "ParkourIntentCommand",
    "ParkourIntentCommandCfg",
    "PROVISIONAL_ORACLE_RESIDUAL_THRESHOLD_RAD",
    "active_motion_time_s",
    "get_preferred_speed",
    "get_requested_travel_direction_yaw_xy",
    "get_target_speed",
    "get_target_yaw_rate",
    "normalize_direction_yaw_xy",
    "wrapped_heading_residual_rad",
]


def normalize_direction_yaw_xy(
    direction_yaw_xy: Tensor,
    *,
    fallback_direction_yaw_xy: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Normalize yaw-frame directions and report which rows were valid."""

    if direction_yaw_xy.ndim != 2 or direction_yaw_xy.shape[-1] != 2:
        raise ValueError(
            f"direction_yaw_xy must have shape [batch, 2], got {tuple(direction_yaw_xy.shape)}."
        )
    finite = torch.all(torch.isfinite(direction_yaw_xy), dim=-1)
    norms = torch.linalg.norm(direction_yaw_xy, dim=-1)
    valid = finite & torch.isfinite(norms) & (norms > 1.0e-6)
    if fallback_direction_yaw_xy is None:
        fallback_direction_yaw_xy = torch.zeros_like(direction_yaw_xy)
        fallback_direction_yaw_xy[:, 0] = 1.0
    if fallback_direction_yaw_xy.shape != direction_yaw_xy.shape:
        raise ValueError("fallback_direction_yaw_xy must match direction_yaw_xy shape.")
    fallback = fallback_direction_yaw_xy / torch.linalg.norm(
        fallback_direction_yaw_xy, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    normalized = direction_yaw_xy / norms.clamp_min(1.0e-6).unsqueeze(-1)
    return torch.where(valid.unsqueeze(-1), normalized, fallback), valid


def wrapped_heading_residual_rad(
    external_direction_yaw_xy: Tensor,
    local_direction_yaw_xy: Tensor,
) -> Tensor:
    """Return signed wrapped local-minus-external yaw in ``[-pi, pi]``."""

    external, _ = normalize_direction_yaw_xy(external_direction_yaw_xy)
    local, _ = normalize_direction_yaw_xy(local_direction_yaw_xy)
    if external.shape != local.shape:
        raise ValueError(
            "External and local heading directions must have identical shape."
        )
    dot = torch.sum(external * local, dim=-1)
    cross = external[:, 0] * local[:, 1] - external[:, 1] * local[:, 0]
    return torch.atan2(cross, dot)


class ParkourIntentCommand(CommandTerm):
    """Own travel direction, speed, yaw rate, and non-translating windows."""

    cfg: ParkourIntentCommandCfg

    def __init__(self, cfg: ParkourIntentCommandCfg, env: ManagerBasedRLEnv) -> None:
        cfg.validate_configuration()
        super().__init__(cfg, env)
        self._command = torch.zeros((self.num_envs, 4), device=self.device)
        self._command[:, 0] = 1.0
        self._external_override = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.active_motion_time_s = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> Tensor:
        """Return ``[requested_forward, requested_left, speed, yaw_rate]``."""

        return self._command

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Clear episode-local command state before the inherited resample."""

        ids = _all_env_ids(self._env, env_ids)
        self.active_motion_time_s[ids] = 0.0
        self._external_override[ids] = False
        self._command[ids, 0] = 1.0
        self._command[ids, 1:] = 0.0
        return super().reset(env_ids=ids)

    def set_external_intent(
        self,
        env_ids: Sequence[int] | Tensor | None,
        requested_travel_direction_yaw_xy: Tensor,
        preferred_speed_m_s: Tensor,
        preferred_yaw_rate_rad_s: Tensor,
        *,
        valid: Tensor | None = None,
    ) -> None:
        """Install one canonical command packet, failing invalid rows closed."""

        ids = _all_env_ids(self._env, env_ids)
        if requested_travel_direction_yaw_xy.shape != (ids.numel(), 2):
            raise ValueError(
                "requested_travel_direction_yaw_xy must contain one two-vector "
                "per selected environment."
            )
        if preferred_speed_m_s.shape != (ids.numel(),):
            raise ValueError(
                "preferred_speed_m_s must contain one value per selected environment."
            )
        if preferred_yaw_rate_rad_s.shape != (ids.numel(),):
            raise ValueError(
                "preferred_yaw_rate_rad_s must contain one value per selected environment."
            )
        if valid is not None and valid.shape != (ids.numel(),):
            raise ValueError(
                "valid must contain one Boolean value per selected environment."
            )
        if valid is not None and valid.dtype != torch.bool:
            raise TypeError("valid must be a Boolean tensor.")

        speed = preferred_speed_m_s.to(device=self.device, dtype=self._command.dtype)
        speed_valid = torch.isfinite(speed) & (speed >= 0.0)
        speed = speed.clamp_max(self.cfg.max_external_speed_m_s)
        speed = torch.where(
            (speed > 0.0) & (speed <= self.cfg.stop_deadband_m_s),
            torch.zeros_like(speed),
            speed,
        )
        yaw_rate = preferred_yaw_rate_rad_s.to(
            device=self.device, dtype=self._command.dtype
        )
        yaw_rate_valid = torch.isfinite(yaw_rate)
        yaw_rate = yaw_rate.clamp(
            min=-self.cfg.max_external_yaw_rate_rad_s,
            max=self.cfg.max_external_yaw_rate_rad_s,
        )
        yaw_rate = torch.where(
            torch.abs(yaw_rate) <= self.cfg.yaw_rate_deadband_rad_s,
            torch.zeros_like(yaw_rate),
            yaw_rate,
        )
        # The first training contract supports either translation or an
        # in-place pivot, never an untrained simultaneous command.
        translating = speed > 0.0
        yaw_rate = torch.where(translating, torch.zeros_like(yaw_rate), yaw_rate)
        previous_requested_direction = self._command[ids, :2]
        direction, direction_valid = normalize_direction_yaw_xy(
            requested_travel_direction_yaw_xy.to(
                device=self.device, dtype=self._command.dtype
            ),
            fallback_direction_yaw_xy=previous_requested_direction,
        )
        adapter_valid = (
            torch.ones_like(direction_valid) if valid is None else valid.to(self.device)
        )
        packet_valid = speed_valid & yaw_rate_valid & adapter_valid
        command_valid = packet_valid & ((~translating) | direction_valid)
        update_direction = command_valid & translating
        self._command[ids, :2] = torch.where(
            update_direction.unsqueeze(-1), direction, previous_requested_direction
        )
        self._command[ids, 2] = torch.where(
            command_valid, speed, torch.zeros_like(speed)
        )
        self._command[ids, 3] = torch.where(
            command_valid, yaw_rate, torch.zeros_like(yaw_rate)
        )
        self._external_override[ids] = True
        self.time_left[ids] = math.inf

    def invalidate(self, env_ids: Sequence[int] | Tensor | None = None) -> None:
        """Turn stale commands into exact-zero requests while preserving direction."""

        ids = _all_env_ids(self._env, env_ids)
        self._command[ids, 2:] = 0.0
        self._external_override[ids] = True
        self.time_left[ids] = math.inf

    def _resample(self, env_ids: Sequence[int]) -> None:
        """Resample without touching global RNG for deterministic profiles."""

        if len(env_ids) == 0:
            return
        fixed_pivot = self.cfg.fixed_yaw_rate_rad_s not in (None, 0.0)
        if getattr(self.cfg, "command_profile", "mixed") == "mixed" and not fixed_pivot:
            super()._resample(env_ids)
            return
        self._resample_command(env_ids)
        self.command_counter[env_ids] += 1

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        """Sample translation, stationary stops, and flat-only pivots."""

        ids = _all_env_ids(self._env, env_ids)
        scripted = ids[~self._external_override[ids]]
        if scripted.numel() == 0:
            return

        first_command = self.command_counter[scripted] == 0
        was_nontranslating = self._command[scripted, 2] == 0.0
        flat = route.active_difficulty_indices(self._env, scripted) == 0
        flat_min, flat_max = self.cfg.flat_speed_range_m_s
        obstacle_min, obstacle_max = self.cfg.obstacle_speed_range_m_s
        min_speed = torch.where(flat, flat_min, obstacle_min)
        max_speed = torch.where(flat, flat_max, obstacle_max)
        force_stop = (min_speed == 0.0) & (max_speed == 0.0)
        profile = getattr(self.cfg, "command_profile", "mixed")
        fixed_yaw_rate = self.cfg.fixed_yaw_rate_rad_s
        fixed_pivot = fixed_yaw_rate not in (None, 0.0)
        deterministic_stop = profile == "stop_restart"
        deterministic_pivot = profile == "pivot_restart" or fixed_pivot
        deterministic_translation = profile == "translation_only"
        if profile == "mixed" and not fixed_pivot:
            draw = torch.rand(scripted.numel(), device=self.device)
            # Pivots may occur on the first command so reset-to-pivot is part
            # of the trained domain. Stops still begin only after translation.
            eligible_stop = (
                (~first_command) & (~was_nontranslating) & flat & (~force_stop)
            )
            eligible_pivot = (
                (first_command | (~was_nontranslating)) & flat & (~force_stop)
            )
            choose_stop = force_stop | (
                eligible_stop & (draw < self.cfg.stop_window_probability)
            )
            choose_pivot = (
                eligible_pivot
                & (draw >= self.cfg.stop_window_probability)
                & (
                    draw
                    < self.cfg.stop_window_probability
                    + self.cfg.pivot_window_probability
                )
            )
        elif deterministic_stop:
            # One prescribed translate-stop-translate trial per episode.
            choose_stop = (self.command_counter[scripted] == 1) & flat & (~force_stop)
            choose_pivot = torch.zeros_like(choose_stop)
        elif deterministic_pivot:
            # One prescribed translate-pivot-translate trial per episode.
            choose_pivot = (self.command_counter[scripted] == 1) & flat & (~force_stop)
            choose_stop = force_stop
        elif deterministic_translation:
            choose_stop = force_stop
            choose_pivot = torch.zeros_like(choose_stop)
        else:  # Guard direct term construction in addition to config validation.
            raise ValueError(f"Unsupported command profile: {profile!r}.")

        moving = ~(choose_stop | choose_pivot)
        moving_ids = scripted[moving]
        moving_flat = flat[moving]
        stopped_ids = scripted[choose_stop]
        pivot_ids = scripted[choose_pivot]

        if moving_ids.numel() > 0:
            moving_min = min_speed[moving]
            moving_max = max_speed[moving]
            if deterministic_translation or deterministic_stop or deterministic_pivot:
                self._command[moving_ids, 2] = moving_min
            else:
                self._command[moving_ids, 2] = moving_min + torch.rand(
                    moving_ids.numel(), device=self.device, dtype=self._command.dtype
                ) * (moving_max - moving_min)
            self._command[moving_ids, 3] = 0.0
            if deterministic_stop or deterministic_pivot:
                self.time_left[moving_ids] = math.inf
                initial_ids = moving_ids[first_command[moving]]
                self.time_left[initial_ids] = EVALUATION_RESTART_WINDOW_DURATION_S
            elif deterministic_translation:
                self.time_left[moving_ids] = math.inf
            else:
                self.time_left[moving_ids[~moving_flat]] = math.inf

        if stopped_ids.numel() > 0:
            self._command[stopped_ids, 2:] = 0.0
            if deterministic_stop:
                self.time_left[stopped_ids] = EVALUATION_RESTART_WINDOW_DURATION_S
            elif not deterministic_translation:
                long_stop = (
                    torch.rand(stopped_ids.numel(), device=self.device)
                    < self.cfg.long_stop_probability
                )
                long_stop_ids = stopped_ids[long_stop]
                if long_stop_ids.numel() > 0:
                    low, high = self.cfg.long_stop_window_range_s
                    self.time_left[long_stop_ids] = low + torch.rand(
                        long_stop_ids.numel(),
                        device=self.device,
                        dtype=self.time_left.dtype,
                    ) * (high - low)

        if pivot_ids.numel() > 0:
            self._command[pivot_ids, 2] = 0.0
            if not deterministic_pivot:
                low, high = self.cfg.pivot_abs_yaw_rate_range_rad_s
                magnitude = low + torch.rand(
                    pivot_ids.numel(),
                    device=self.device,
                    dtype=self._command.dtype,
                ) * (high - low)
                sign = torch.where(
                    torch.rand(pivot_ids.numel(), device=self.device) < 0.5,
                    -torch.ones_like(magnitude),
                    torch.ones_like(magnitude),
                )
                self._command[pivot_ids, 3] = sign * magnitude
            else:
                self._command[pivot_ids, 3] = fixed_yaw_rate
            if not deterministic_pivot:
                low, high = self.cfg.pivot_window_range_s
                self.time_left[pivot_ids] = low + torch.rand(
                    pivot_ids.numel(),
                    device=self.device,
                    dtype=self.time_left.dtype,
                ) * (high - low)
            else:
                self.time_left[pivot_ids] = EVALUATION_RESTART_WINDOW_DURATION_S

        # Preserve the previous bearing throughout an ordinary stop. A fresh
        # zero-speed episode still needs a meaningful requested travel direction.
        direction_ids = torch.cat(
            (moving_ids, scripted[first_command & (choose_stop | choose_pivot)])
        )
        if direction_ids.numel() > 0:
            self._update_scripted_requested_travel_direction(direction_ids)

    def _update_metrics(self) -> None:
        """Accumulate translation and pivot time used by the task timeout."""

        has_completed_step = self._env.episode_length_buf > 0
        commanded_motion = (self._command[:, 2] > 0.0) | (
            torch.abs(self._command[:, 3]) > 0.0
        )
        self.active_motion_time_s += (
            has_completed_step & commanded_motion
        ).float() * float(self._env.step_dt)

    def _update_command(self) -> None:
        """Track the scripted flat heading or obstacle-course final goal."""

        ids = torch.nonzero(
            (~self._external_override) & (self._command[:, 2] > 0.0), as_tuple=False
        ).flatten()
        if ids.numel() > 0:
            self._update_scripted_requested_travel_direction(ids)

    def _update_scripted_requested_travel_direction(self, env_ids: Tensor) -> None:
        """Use the local flat command and broad obstacle-course goal bearings."""

        directions, valid = geometry._final_waypoint_direction_yaw_xy_components(
            self._env
        )
        flat = route.active_difficulty_indices(self._env, env_ids) == 0
        terminal = route.active_waypoint_is_terminal_landing(self._env)[env_ids]
        active_directions = geometry._active_waypoint_direction_yaw_xy(self._env)
        directions[env_ids] = torch.where(
            (flat | terminal).unsqueeze(-1),
            active_directions[env_ids],
            directions[env_ids],
        )
        valid[env_ids] |= flat | terminal
        valid_ids = env_ids[valid[env_ids]]
        if valid_ids.numel() > 0:
            self._command[valid_ids, :2] = directions[valid_ids]


@configclass
class ParkourIntentCommandCfg(CommandTermCfg):
    """Training distribution for deployable travel and pivot commands."""

    class_type: type = ParkourIntentCommand
    command_profile: str = "mixed"
    resampling_time_range: tuple[float, float] = (0.5, 1.5)
    flat_speed_range_m_s: tuple[float, float] = (0.20, 0.70)
    obstacle_speed_range_m_s: tuple[float, float] = (0.45, 0.70)
    stop_deadband_m_s: float = 0.05
    max_external_speed_m_s: float = 0.70
    yaw_rate_deadband_rad_s: float = 0.05
    max_external_yaw_rate_rad_s: float = 0.80
    terminal_slowdown_distance_m: float = 0.60
    terminal_min_approach_speed_m_s: float = 0.20
    stop_window_probability: float = 0.125
    pivot_window_probability: float = 0.05
    # Covers short heading trims through approximately 180-degree turns while
    # remaining within the canonical external yaw-rate bound.
    pivot_abs_yaw_rate_range_rad_s: tuple[float, float] = (0.25, 0.80)
    pivot_window_range_s: tuple[float, float] = (0.75, 4.0)
    long_stop_probability: float = 0.20
    long_stop_window_range_s: tuple[float, float] = (2.0, 3.0)
    # Fixed evaluation may set a signed rate. Training leaves this unset and
    # samples symmetric flat-only pivots from the magnitude range above.
    fixed_yaw_rate_rad_s: float | None = None

    def validate_configuration(self) -> None:
        """Validate command ingress and scripted nontranslation scheduling."""

        if self.command_profile not in COMMAND_PROFILES:
            raise ValueError(
                f"command_profile must be one of {COMMAND_PROFILES}, got {self.command_profile!r}."
            )

        for name in (
            "stop_deadband_m_s",
            "max_external_speed_m_s",
            "yaw_rate_deadband_rad_s",
            "max_external_yaw_rate_rad_s",
            "terminal_slowdown_distance_m",
            "terminal_min_approach_speed_m_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.max_external_speed_m_s <= self.stop_deadband_m_s:
            raise ValueError("max_external_speed_m_s must exceed stop_deadband_m_s.")
        if self.max_external_yaw_rate_rad_s <= self.yaw_rate_deadband_rad_s:
            raise ValueError(
                "max_external_yaw_rate_rad_s must exceed yaw_rate_deadband_rad_s."
            )
        if self.terminal_min_approach_speed_m_s > self.max_external_speed_m_s:
            raise ValueError(
                "terminal_min_approach_speed_m_s cannot exceed max_external_speed_m_s."
            )
        if self.terminal_min_approach_speed_m_s <= self.stop_deadband_m_s:
            raise ValueError(
                "terminal_min_approach_speed_m_s must exceed stop_deadband_m_s."
            )
        for name in ("flat_speed_range_m_s", "obstacle_speed_range_m_s"):
            min_speed, max_speed = _validate_range(
                getattr(self, name), name, non_negative=True
            )
            if (min_speed, max_speed) != (
                0.0,
                0.0,
            ) and min_speed <= self.stop_deadband_m_s:
                raise ValueError(
                    "Moving speeds must lie above stop_deadband_m_s; exact zero is sampled separately."
                )
            if max_speed > self.max_external_speed_m_s:
                raise ValueError(f"{name} cannot exceed max_external_speed_m_s.")
        for name in (
            "resampling_time_range",
            "long_stop_window_range_s",
            "pivot_window_range_s",
        ):
            _validate_range(getattr(self, name), name, non_negative=False)
        pivot_min, pivot_max = _validate_range(
            self.pivot_abs_yaw_rate_range_rad_s,
            "pivot_abs_yaw_rate_range_rad_s",
            non_negative=False,
        )
        if pivot_min <= self.yaw_rate_deadband_rad_s:
            raise ValueError("Pivot yaw rates must lie above yaw_rate_deadband_rad_s.")
        if pivot_max > self.max_external_yaw_rate_rad_s:
            raise ValueError(
                "pivot_abs_yaw_rate_range_rad_s cannot exceed max_external_yaw_rate_rad_s."
            )
        for name in (
            "stop_window_probability",
            "pivot_window_probability",
            "long_stop_probability",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if self.stop_window_probability + self.pivot_window_probability > 1.0:
            raise ValueError(
                "stop_window_probability + pivot_window_probability cannot exceed 1."
            )
        deterministic = self.command_profile != "mixed"
        if deterministic:
            for name in ("flat_speed_range_m_s", "obstacle_speed_range_m_s"):
                min_speed, max_speed = getattr(self, name)
                if min_speed != max_speed or min_speed <= self.stop_deadband_m_s:
                    raise ValueError(
                        f"{self.command_profile!r} requires fixed positive speed ranges."
                    )
        if self.fixed_yaw_rate_rad_s is not None:
            fixed = float(self.fixed_yaw_rate_rad_s)
            if (
                not math.isfinite(fixed)
                or abs(fixed) > self.max_external_yaw_rate_rad_s
            ):
                raise ValueError(
                    "fixed_yaw_rate_rad_s must be finite and within the external yaw-rate limit."
                )
            if 0.0 < abs(fixed) <= self.yaw_rate_deadband_rad_s:
                raise ValueError(
                    "A nonzero fixed yaw rate must lie above yaw_rate_deadband_rad_s."
                )
            if fixed != 0.0 and any(
                speed_range[0] <= self.stop_deadband_m_s
                for speed_range in (
                    self.flat_speed_range_m_s,
                    self.obstacle_speed_range_m_s,
                )
            ):
                raise ValueError(
                    "A fixed nonzero yaw rate requires positive translation ranges for restart trials."
                )
        has_fixed_pivot = self.fixed_yaw_rate_rad_s not in (None, 0.0)
        if self.command_profile == "pivot_restart" and not has_fixed_pivot:
            raise ValueError("'pivot_restart' requires a nonzero fixed_yaw_rate_rad_s.")
        if (
            self.command_profile in ("translation_only", "stop_restart")
            and has_fixed_pivot
        ):
            raise ValueError(
                f"{self.command_profile!r} does not permit fixed_yaw_rate_rad_s."
            )


def active_motion_time_s(env: ManagerBasedRLEnv) -> Tensor:
    """Return accumulated translation-or-pivot time for the current episode."""

    term = cast(ParkourIntentCommand, env.command_manager.get_term(INTENT_COMMAND_NAME))
    return term.active_motion_time_s


def get_requested_travel_direction_yaw_xy(env: ManagerBasedRLEnv) -> Tensor:
    """Return the last valid requested travel direction in yaw coordinates."""

    return env.command_manager.get_command(INTENT_COMMAND_NAME)[:, :2]


def get_preferred_speed(env: ManagerBasedRLEnv) -> Tensor:
    """Return the unmodified speed stored in the external intent packet."""

    return env.command_manager.get_command(INTENT_COMMAND_NAME)[:, 2]


def get_target_speed(env: ManagerBasedRLEnv) -> Tensor:
    """Return the route-conditioned translational target seen by the motor.

    A terminal landing tapers the preferred speed toward its root-reach circle,
    retains a small approach speed outside the circle, and requests an exact
    stop inside it. Other route phases preserve the intent packet unchanged.
    """

    preferred_speed = get_preferred_speed(env)
    # ObservationManager evaluates terms once for shape inference before the
    # first curriculum reset creates route state. At that point there is no
    # route context to condition on, so preserve the raw command semantics.
    if not route.has_active_routes(env):
        return preferred_speed

    terminal_landing = route.active_waypoint_is_terminal_landing(env)
    term = cast(ParkourIntentCommand, env.command_manager.get_term(INTENT_COMMAND_NAME))
    distance = geometry._active_waypoint_distance_xy(env).to(
        device=preferred_speed.device,
        dtype=preferred_speed.dtype,
    )
    reach_radius = route.active_waypoint_root_reach_radii(env).to(
        device=preferred_speed.device,
        dtype=preferred_speed.dtype,
    )
    approach_scale = (
        (distance - reach_radius) / term.cfg.terminal_slowdown_distance_m
    ).clamp(0.0, 1.0)
    creep_speed = preferred_speed.clamp_max(term.cfg.terminal_min_approach_speed_m_s)
    approach_speed = torch.maximum(preferred_speed * approach_scale, creep_speed)
    terminal_speed = torch.where(
        distance <= reach_radius,
        torch.zeros_like(preferred_speed),
        approach_speed,
    )
    return torch.where(terminal_landing, terminal_speed, preferred_speed)


def get_target_yaw_rate(env: ManagerBasedRLEnv) -> Tensor:
    """Return the signed in-place yaw-rate command in radians per second."""

    return env.command_manager.get_command(INTENT_COMMAND_NAME)[:, 3]


def _validate_range(
    values: tuple[float, float],
    name: str,
    *,
    non_negative: bool,
) -> tuple[float, float]:
    try:
        low, high = (float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain exactly two finite values.") from error
    lower_bound_invalid = low < 0.0 if non_negative else low <= 0.0
    if (
        not math.isfinite(low)
        or not math.isfinite(high)
        or lower_bound_invalid
        or high < low
    ):
        qualifier = "non-negative" if non_negative else "positive"
        raise ValueError(
            f"{name} must contain finite {qualifier} bounds in ascending order."
        )
    return low, high
