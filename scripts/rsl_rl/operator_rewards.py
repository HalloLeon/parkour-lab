"""Command-gated tracking and soft stance objectives for operator training.

These rewards use native commands and robot state, not inference-time feedback.
Privileged reward inputs do not enter the causal actor's observations. There
is no world-position/heading anchor, timer, phase label, action override or
persistent reward state. The separate exposure observer only counts samples.
A soft joint-posture prior is not a rigid stance, and zero
velocity is NOT absolute position/heading restoration.
"""

import math
from dataclasses import fields, is_dataclass

import torch


STUMBLE_FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
STUMBLE_PARAMS = {
    "command_name": "base_velocity",
    "command_xy_threshold": 0.1,
    "lateral_to_vertical_force_ratio": 4.0,
    "min_force": 1.0,
}
STUMBLE_COUNTS = (
    "terminal_excluded_rows",
    "translation_rows",
    "step_translation_rows",
    "stumble_rows",
    "step_stumble_rows",
)
STUMBLE_EXPOSURE_SCOPE = (
    "Pre-action commands with post-return net-normal-force history; reset rows excluded. "
    "Descriptive exposure, not exact pre-reset reward accounting or traversal qualification."
)


def stumble_objective(weight):
    if type(weight) not in (int, float) or weight not in (0.0, -0.5):
        raise ValueError("Require explicit zero or -0.5 translation-stumble weight")
    return {
        "version": "operator_translation_stumble_v1",
        "term": "feet_stumble",
        "function": "scripts.rsl_rl.operator_rewards:feet_stumble_translation",
        "weight": float(weight),
        "registered": True,
        "contributes": weight != 0,
        "params": dict(STUMBLE_PARAMS),
        "sensor": {
            "name": "contact_forces",
            "body_names": list(STUMBLE_FEET),
            "history_length": 3,
        },
        "scope": "Stateless commanded-XY gate; native normal-force history predicate, not friction force. Native reward weight and control dt applied once.",
    }


def feet_stumble_translation(
    env,
    sensor_cfg,
    command_name="base_velocity",
    command_xy_threshold=0.1,
    lateral_to_vertical_force_ratio=4.0,
    min_force=1.0,
):
    from parkour_lab.tasks.manager_based.parkour_lab.mdp.reward_terms.limb import (
        feet_stumble,
    )

    moving = (
        torch.linalg.vector_norm(
            env.command_manager.get_command(command_name)[:, :2], dim=1
        )
        > command_xy_threshold
    )
    return (
        feet_stumble(env, lateral_to_vertical_force_ratio, min_force, sensor_cfg)
        * moving
    )


def configure_stumble(cfg, weight):
    """Declare the term so native configclass.copy retains every original field."""
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    from isaaclab.utils import configclass

    stumble_objective(weight)
    original = cfg.rewards
    if not is_dataclass(original) or hasattr(original, "feet_stumble"):
        raise ValueError("Require native rewards without an existing feet_stumble term")
    declared = {item.name for item in fields(original) if item.init}
    if any(name not in declared for name in vars(original) if not name.startswith("_")):
        raise ValueError("Cannot discard undeclared reward fields")

    @configclass
    class StumbleRewards(type(original)):
        feet_stumble: RewardTermCfg = RewardTermCfg(
            func=feet_stumble_translation,
            weight=float(weight),
            params={
                **STUMBLE_PARAMS,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=list(STUMBLE_FEET), preserve_order=True
                ),
            },
        )

    cfg.rewards = StumbleRewards(**{name: getattr(original, name) for name in declared})


def verify_stumble_objective(env, weight):
    expected = stumble_objective(weight)
    if "feet_stumble" not in env.reward_manager.active_terms:
        raise ValueError("Native feet_stumble objective is not registered")
    native = env.reward_manager.get_term_cfg("feet_stumble")
    for term in (env.cfg.rewards.feet_stumble, native):
        if not isinstance(term.params, dict):
            raise ValueError("Invalid native stumble parameters")
        sensor = term.params.get("sensor_cfg")
        if (
            term.func is not feet_stumble_translation
            or type(term.weight) not in (int, float)
            or term.weight != weight
            or set(term.params) != {*STUMBLE_PARAMS, "sensor_cfg"}
            or any(
                type(term.params[k]) is not type(v) or term.params[k] != v
                for k, v in STUMBLE_PARAMS.items()
            )
            or getattr(sensor, "name", None) != "contact_forces"
            or getattr(sensor, "body_names", None) != list(STUMBLE_FEET)
            or getattr(sensor, "preserve_order", None) is not True
        ):
            raise ValueError(
                "Native translation-stumble objective differs from selected arm"
            )
    sensor = native.params["sensor_cfg"]
    names = env.scene[sensor.name].body_names
    ids = (
        list(range(len(names)))[sensor.body_ids]
        if isinstance(sensor.body_ids, slice)
        else sensor.body_ids
    )
    if any(names.count(name) != 1 for name in STUMBLE_FEET) or ids != [
        names.index(name) for name in STUMBLE_FEET
    ]:
        raise ValueError("Native stumble sensor did not resolve the four named feet")
    contact = env.scene[sensor.name]
    if (
        type(contact.cfg.history_length) is not int
        or contact.cfg.history_length != 3
        or contact.data.net_forces_w_history.shape != (env.num_envs, 3, len(names), 3)
    ):
        raise ValueError("Require the inherited three-sample native force history")
    return expected


class StumbleExposure:
    """Post-step observer only; reward evaluation never creates persistent state."""

    def __init__(self, env):
        term = env.reward_manager.get_term_cfg("feet_stumble")
        verify_stumble_objective(env, term.weight)
        self.env, self.sensor, self.steps = env, term.params["sensor_cfg"], 0
        columns = env.scene.terrain.terrain_types
        if (
            columns.shape != (env.num_envs,)
            or columns.dtype not in (torch.int32, torch.int64)
            or not bool(((columns >= 0) & (columns < 20)).all())
        ):
            raise ValueError(
                "Stumble exposure requires the native 20-column environment"
            )
        self.step_rows = (columns >= 12) & (columns < 16)
        self.counts = torch.zeros(
            len(STUMBLE_COUNTS), dtype=torch.long, device=env.device
        )

    def sample(self, done, command):
        from parkour_lab.tasks.manager_based.parkour_lab.mdp.reward_terms.limb import (
            feet_stumble,
        )

        if (
            not isinstance(done, torch.Tensor)
            or not isinstance(command, torch.Tensor)
            or done.shape != (self.env.num_envs,)
            or done.dtype != torch.bool
            or done.device != self.counts.device
            or command.shape != (self.env.num_envs, 3)
            or command.device != self.counts.device
            or not torch.is_floating_point(command)
            or not bool(torch.isfinite(command).all())
        ):
            raise ValueError("Require native done mask and finite pre-action commands")
        force = self.env.scene[self.sensor.name].data.net_forces_w_history
        if (
            force.shape
            != (
                self.env.num_envs,
                3,
                len(self.env.scene[self.sensor.name].body_names),
                3,
            )
            or not torch.is_floating_point(force)
            or not bool(torch.isfinite(force[~done]).all())
        ):
            raise ValueError("Require finite nonreset native force history")
        moving = (torch.linalg.vector_norm(command[:, :2], dim=1) > 0.1) & ~done
        stumble = feet_stumble(self.env, 4.0, 1.0, self.sensor).bool() & moving
        self.counts += torch.stack(
            (
                done.sum(),
                moving.sum(),
                (moving & self.step_rows).sum(),
                stumble.sum(),
                (stumble & self.step_rows).sum(),
            )
        )
        self.steps += 1

    def report(self):
        return {
            "version": "operator_translation_stumble_exposure_v1",
            "scope": STUMBLE_EXPOSURE_SCOPE,
            "control_steps": self.steps,
            "num_envs": self.env.num_envs,
            "counts": dict(
                zip(STUMBLE_COUNTS, self.counts.cpu().tolist(), strict=True)
            ),
        }


def validate_stumble_exposure(report, *, num_envs, steps):
    if type(num_envs) is not int or num_envs < 1 or type(steps) is not int or steps < 0:
        raise ValueError("Require integer exposure dimensions")
    try:
        counts = report["counts"]
        if (
            set(report) != {"version", "scope", "control_steps", "num_envs", "counts"}
            or report["version"] != "operator_translation_stumble_exposure_v1"
            or report["scope"] != STUMBLE_EXPOSURE_SCOPE
            or type(report["control_steps"]) is not int
            or report["control_steps"] != steps
            or type(report["num_envs"]) is not int
            or report["num_envs"] != num_envs
            or type(counts) is not dict
            or set(counts) != set(STUMBLE_COUNTS)
            or any(type(v) is not int or v < 0 for v in counts.values())
        ):
            raise ValueError("Invalid stumble exposure receipt")
        ended, moving, step_moving, stumble, step_stumble = (
            counts[name] for name in STUMBLE_COUNTS
        )
        if not (
            ended <= num_envs * steps
            and moving <= num_envs * steps - ended
            and step_moving <= moving
            and stumble <= moving
            and step_stumble <= min(step_moving, stumble)
            and stumble - step_stumble <= moving - step_moving
        ):
            raise ValueError("Inconsistent stumble exposure populations")
    except (TypeError, KeyError) as error:
        raise ValueError("Malformed stumble exposure receipt") from error
    return report


def joint_posture_stopped(env, command_name: str):
    """Soft default-pose cost for exact zero twist, not a joint-target override.

    Use the L2 norm (not its square) in native radians. Moving commands,
    including pure pivots and arbitrarily small commands, pay no new cost.
    No speed gate lets the policy escape the cost by moving during a stop.
    The caller supplies the negative weight; RewardManager integrates dt.
    """
    command = env.command_manager.get_command(command_name)
    data = env.scene["robot"].data
    deviation = torch.linalg.vector_norm(
        data.joint_pos - data.default_joint_pos, dim=-1
    )
    return torch.where(torch.all(command == 0, dim=-1), deviation, 0.0)


def teacher_physical_failure(
    env,
    termination_names=(
        "base_contact",
        "course_chassis",
        "course_fall",
        "course_off_route",
        "persistent_tilt",
    ),
):
    """One training impulse per physical failure, even if several gates fire."""
    failure = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for name in termination_names:
        failure |= env.termination_manager.get_term(name)
    return failure.float() / env.step_dt


def _kernels(error_squared, std, stationary_std, precision_fraction):
    if not (
        math.isfinite(std)
        and math.isfinite(stationary_std)
        and 0 < stationary_std < std
        and math.isfinite(precision_fraction)
        and 0 < precision_fraction < 1
    ):
        raise ValueError("Use 0 < stationary_std < std and 0 < precision_fraction < 1")
    broad = torch.exp(-error_squared / std**2)
    fine = torch.exp(-error_squared / stationary_std**2)
    # Keep the original peak and a broad acquisition component. Do not multiply
    # the two kernels: that would suppress all large-error learning signal.
    return broad, (1 - precision_fraction) * broad + precision_fraction * fine


def track_lin_vel_xy_stationary(
    env,
    command_name: str,
    std: float,
    stationary_std: float,
    precision_fraction: float,
    full_stop_only: bool = False,
    pivot_only: bool = False,
    root_link_velocity: bool = False,
):
    """Blend fine planar tracking only when commanded planar velocity is zero.

    Both yaw signs use identical gating. Nonzero planar commands, even small
    ones, retain the exact broad reward. With full_stop_only, pure pivots also
    retain the broad reward. With pivot_only, exact zero twist retains it
    instead. These restrictions are mutually exclusive; defaults preserve
    historical recipes. Training command modes emit exact zeros.
    root_link_velocity selects the native root-link origin for ALL commands;
    the historical default uses root-body COM velocity. Neither is whole-robot
    COM velocity. Only the reference point changes, not the body-frame axes.
    """
    if full_stop_only and pivot_only:
        raise ValueError("full_stop_only and pivot_only are mutually exclusive")
    command = env.command_manager.get_command(command_name)
    data = env.scene["robot"].data
    velocity = (
        data.root_link_lin_vel_b if root_link_velocity else data.root_lin_vel_b
    )[:, :2]
    error_squared = torch.sum(torch.square(command[:, :2] - velocity), dim=1)
    broad, fine = _kernels(error_squared, std, stationary_std, precision_fraction)
    stopped = torch.all((command if full_stop_only else command[:, :2]) == 0, dim=1)
    if pivot_only:
        stopped &= command[:, 2] != 0
    return torch.where(stopped, fine, broad)


def track_ang_vel_z_stopped(
    env,
    command_name: str,
    std: float,
    stationary_std: float,
    precision_fraction: float,
    include_pivots: bool = False,
    full_stop_std: float | None = None,
):
    """Blend fine yaw tracking at zero twist, optionally including pure pivots.

    The default preserves historical stop-only recipes. With include_pivots,
    both yaw signs track their commanded rate, not zero. Every nonzero planar
    command retains the exact broad reward, including tiny commands and arcs.
    full_stop_std optionally changes only the fine width at exact zero twist;
    the mixture fraction and pure-pivot kernel stay unchanged.
    This stateless rate objective is not absolute heading restoration.
    """
    command = env.command_manager.get_command(command_name)
    yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2]
    error_squared = torch.square(command[:, 2] - yaw_rate)
    broad, fine = _kernels(error_squared, std, stationary_std, precision_fraction)
    if full_stop_std is not None:
        _, stop_fine = _kernels(error_squared, std, full_stop_std, precision_fraction)
        fine = torch.where(torch.all(command == 0, dim=1), stop_fine, fine)
    stationary = torch.all((command[:, :2] if include_pivots else command) == 0, dim=1)
    return torch.where(stationary, fine, broad)
