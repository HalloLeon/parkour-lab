"""Bounded, single-episode startup capture; no simulator imports.

The estimator hook observes the *actual* inference call. Post-step state must be
supplied by the reward-time snapshot, because a vector wrapper auto-resets done
environments before returning from ``step``.
"""

from __future__ import annotations

import math
from contextlib import nullcontext


OBSERVATION_GROUPS = (
    "policy",
    "oracle_travel_direction",
    "terrain",
    "dynamics",
    "adaptation_history",
    "velocity_target",
)
MAX_STARTUP_STEPS = 500


def validate_startup_arguments(args) -> None:
    """Reject conflicting modes before launching the simulator."""
    replay = getattr(args, "startup_action_replay", None)
    centered = getattr(args, "startup_center_scene", False)
    if centered and (
        not args.startup_diagnostics
        or not replay
        or args.terrain_family != "high_step"
        or args.difficulty_level not in (0, 6)
        or args.geometry_variant != 0
    ):
        raise ValueError(
            "--startup_center_scene is only for high_step L0/L6 variant 0 action replay; "
            "it changes the physical scene and is not a policy evaluation."
        )
    if not args.startup_diagnostics:
        if (
            args.startup_steps is not None
            or args.startup_output_dir is not None
            or replay is not None
        ):
            raise ValueError(
                "--startup_steps/--startup_output_dir/--startup_action_replay require --startup_diagnostics."
            )
        return
    if args.startup_steps is None:
        args.startup_steps = 10 if replay is not None else 100
    if not 1 <= args.startup_steps <= MAX_STARTUP_STEPS:
        raise ValueError(f"--startup_steps must be between 1 and {MAX_STARTUP_STEPS}.")
    if replay is not None and (
        not replay
        or args.startup_steps > 10
        or args.command_profile != "translation_only"
        or args.policy_mode != "history_mean"
    ):
        raise ValueError(
            "--startup_action_replay requires an explicit source trace, at most 10 steps, "
            "history_mean and translation_only; it is not a policy evaluation."
        )
    if (
        args.num_envs != 1
        or args.eval_episodes != 1
        or args.reset_profile not in ("canonical", "jitter")
        or args.policy_mode not in ("history_mean", "privileged_mean")
        or args.command_profile
        not in ("translation_only", "stop_restart", "pivot_restart")
        or args.terrain_family is None
        or args.difficulty_level is None
        or args.geometry_variant is None
        or args.seed is None
        or args.desired_speed is None
        or not math.isfinite(args.desired_speed)
        or args.desired_speed <= 0
        or (
            args.command_profile == "pivot_restart"
            and (
                args.desired_yaw_rate is None
                or not math.isfinite(args.desired_yaw_rate)
                or args.desired_yaw_rate == 0.0
            )
        )
        or (
            args.command_profile != "pivot_restart"
            and args.desired_yaw_rate not in (None, 0.0)
        )
        or args.action_noise_std is not None
        or args.action_noise_seed is not None
        or args.teleop
        or args.video
        or args.telemetry
        or args.screen
        or args.all_courses
        or args._course_manifest is not None
        or args.real_time
    ):
        raise ValueError(
            "--startup_diagnostics requires --num_envs=1 --eval_episodes=1 "
            "--reset_profile=canonical or jitter, a deterministic mean policy, explicit family/level/variant/seed, "
            "positive speed and an explicit command profile (pivot_restart needs finite nonzero yaw; "
            "translation_only/stop_restart need zero yaw); no video, screen, telemetry, "
            "teleop, matrix, real-time or action noise."
        )


def _single_environment(value):
    """Copy one batch row to JSON data; never retain mutable simulator views."""
    if value.ndim < 1 or value.shape[0] != 1:
        raise ValueError(
            "Startup capture requires tensors with exactly one environment."
        )
    return value.detach()[0].cpu().tolist()


def collect_startup_diagnostics(
    env,
    policy,
    *,
    max_steps,
    snapshot_state,
    latest_step,
    termination_outcomes,
    is_running,
    report,
    action_probe=None,
) -> None:
    """Fill a report in place, retaining partial samples if an exception occurs.

    Callers own serialization and environment lifetime. No second inference,
    observation recomputation, reset or RNG draw is made. The optional explicit
    action probe replaces executed actions with a recorded prefix; current
    policy outputs remain separately recorded and this is not policy evaluation.
    """
    import torch

    if env.num_envs != 1 or not 1 <= max_steps <= MAX_STARTUP_STEPS:
        raise ValueError(
            "Startup capture requires one environment and a bounded step count."
        )
    owner = getattr(policy, "__self__", None)
    estimator = getattr(getattr(owner, "actor", None), "velocity_estimator", None)
    if estimator is None:
        raise ValueError(
            "Startup capture requires the velocity-estimator teacher's bound mean policy."
        )
    estimates = []

    def observe_estimator(_module, _inputs, output):
        estimates.append(output.detach().clone())
        # Returning None preserves the module output exactly.

    handle = estimator.register_forward_hook(observe_estimator)
    report["samples"] = []
    report["stop_reason"] = "step_limit"
    try:
        with torch.inference_mode():
            obs = env.get_observations()
            for step in range(max_steps):
                if not is_running():
                    report["stop_reason"] = "application_stopped"
                    break
                # Serialize pre-state before step() can overwrite shared buffers.
                pre = {
                    "observations": {
                        name: _single_environment(obs[name])
                        for name in OBSERVATION_GROUPS
                    },
                    "state": {
                        name: _single_environment(value)
                        for name, value in snapshot_state(env.unwrapped).items()
                    },
                }
                estimates.clear()
                actions = policy(obs)
                if len(estimates) != 1 or estimates[0].shape != (1, 3):
                    raise ValueError(
                        "Expected exactly one body-velocity estimate of shape (1, 3) per action."
                    )
                pre["estimated_velocity_body_m_s"] = _single_environment(estimates[0])
                action_values = _single_environment(actions)
                if action_probe is not None:
                    actions = action_probe.action(step, actions, pre)
                    replayed_values = _single_environment(actions)
                with (
                    nullcontext()
                    if action_probe is None
                    else action_probe.capture_physics(env.unwrapped, step)
                ):
                    obs, _, dones, _ = env.step(actions)
                transition = latest_step(env.unwrapped)
                if transition is None or transition.startup is None:
                    raise RuntimeError(
                        "Missing post-physics, pre-reset startup snapshot."
                    )
                done_mask = dones.to(dtype=torch.bool)
                done = bool(done_mask[0].item())
                report["samples"].append(
                    {
                        "step": step,
                        "time_before_s": step * env.unwrapped.step_dt,
                        "time_after_s": (step + 1) * env.unwrapped.step_dt,
                        "pre": pre,
                        "policy_action": action_values,
                        "post": {
                            "state": {
                                name: _single_environment(value)
                                for name, value in transition.startup.items()
                            },
                            "done": done,
                            "termination": {
                                name: bool(_single_environment(value))
                                for name, value in termination_outcomes(
                                    env.unwrapped, done_mask
                                ).items()
                            },
                        },
                    }
                )
                if action_probe is not None:
                    sample = report["samples"][-1]
                    sample["replayed_action"] = replayed_values
                    action_probe.validate_post(step, sample["post"]["state"])
                if done:
                    report["stop_reason"] = "episode_terminated"
                    break
    except Exception as error:
        report["stop_reason"] = "error"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        handle.remove()
