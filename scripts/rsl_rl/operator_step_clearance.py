"""Training-only low foot-LINK height cost, not sole clearance or swing control."""

from copy import deepcopy
from dataclasses import fields, is_dataclass
import math

VERSION = "operator_step_link_height_v1"
TERM_NAME = "step_link_height"
FEET = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
WEIGHT = -0.25
GAP_EDGES = (-0.05, 0.0, 0.025, 0.05, 0.075, 0.10, 0.12, 0.15, 0.20)
THRESHOLDS = (0.08, 0.10, 0.12)
COUNT_NAMES = (
    "translation_step_samples",
    "physical_terminal_samples",
    "invalid_input_samples",
    "eligible_environment_samples",
    "ray_valid_foot_samples",
    "ray_missed_foot_samples",
    "eligible_valid_foot_samples",
    "all_feet_airborne_samples",
)


def recipe():
    return {
        "version": VERSION,
        "term": TERM_NAME,
        "weight": WEIGHT,
        "target_link_height_m": 0.12,
        "speed_tanh_multiplier_s_per_m": 2.0,
        "command_xy_threshold_m_s": 0.1,
        "columns": [12, 13, 14, 15],
        "formula": "mean4(clip((0.12-signed_link_to_mesh_z)/0.12,0,1)^2*tanh(2*link_world_xy_speed))",
        "contact_gate": False,
        "physical_terminal_cost": 0.0,
        "invalid_input_or_ray_cost": 0.0,
        "ray_start_min_above_env_origin_m": 1.0,
        "ray_max_distance_m": 5.0,
        "gap_histogram_edges_m": list(GAP_EDGES),
        "gap_thresholds_m": list(THRESHOLDS),
        "long_air_threshold_s": 0.5,
        "scope": "Translation on step columns only; signed vertical LINK-to-terrain-mesh distance, NOT sole clearance or foot support. Zero-speed feet incur zero cost; not a direct wall-escape incentive.",
    }


def configure(cfg):
    """Declare the new term so native configclass.copy() retains it."""
    from isaaclab.managers import RewardTermCfg
    from isaaclab.utils import configclass

    original = cfg.rewards
    if not is_dataclass(original) or hasattr(original, TERM_NAME):
        raise ValueError("Require an unmodified native reward config without this term")
    declared = {item.name for item in fields(original) if item.init}
    if any(name not in declared for name in vars(original) if not name.startswith("_")):
        raise ValueError("Cannot discard undeclared reward fields")

    @configclass
    class ClearanceRewards(type(original)):
        step_link_height: RewardTermCfg = RewardTermCfg(func=reward, weight=WEIGHT)

    cfg.rewards = ClearanceRewards(
        **{name: getattr(original, name) for name in declared}
    )


def reward(env):
    state = getattr(env, "_operator_step_clearance", None)
    if state is None:
        raise RuntimeError("Foot-link height cost was not installed/admitted")
    return state.compute()


def install(env):
    if hasattr(env, "_operator_step_clearance"):
        raise ValueError("Do not overwrite foot-link height state")
    term = env.reward_manager.get_term_cfg(TERM_NAME)
    if term.func is not reward or term.weight != WEIGHT or term.params != {}:
        raise ValueError("Native RewardManager did not admit the exact foot-link term")
    state = ClearanceState(env)
    _, _, _, _, _, finite, ray_valid, _ = state.read()
    if not bool(finite.all()) or not bool(ray_valid.all()):
        raise ValueError(
            "Foot-link cost startup requires finite inputs and all ground rays"
        )
    env._operator_step_clearance = state
    return state


class ClearanceState:
    def __init__(self, env):
        import torch

        self.env, self.active, self.steps, self.last_clock = env, False, 0, None
        self.robot, self.contacts = env.scene["robot"], env.scene["contact_forces"]
        names = (self.robot.body_names, self.contacts.body_names)
        if any(collection.count(name) != 1 for collection in names for name in FEET):
            raise ValueError(
                "Require four unambiguous named feet in each native sensor"
            )
        self.feet, self.contact_feet = (
            [collection.index(name) for name in FEET] for collection in names
        )
        if not self.contacts.cfg.track_air_time:
            raise ValueError("Native contact air timers must be enabled")
        self.contact_threshold = float(self.contacts.cfg.force_threshold)
        if not math.isfinite(self.contact_threshold) or self.contact_threshold <= 0:
            raise ValueError("Invalid native contact threshold")
        self.columns = env.scene.terrain.terrain_types.clone()
        self.origins = env.scene.env_origins.clone()
        if (
            self.columns.shape != (env.num_envs,)
            or self.origins.shape != (env.num_envs, 3)
            or not bool(torch.isfinite(self.origins).all())
            or not bool(((self.columns >= 0) & (self.columns < 20)).all())
            or not math.isclose(env.step_dt, 0.02, rel_tol=0, abs_tol=1e-12)
        ):
            raise ValueError("Require the admitted 20-column/50Hz environment")
        self.step_mask = (self.columns >= 12) & (self.columns < 16)
        sensor = env.scene["base_height_scanner"]
        paths = sensor.cfg.mesh_prim_paths
        if paths != ["/World/ground"] or paths[0] not in sensor.meshes:
            raise ValueError("Require the initialized terrain-only RayCaster mesh")
        self.mesh = sensor.meshes[paths[0]]
        self.binding = {
            "feet": list(FEET),
            "articulation_ids": self.feet,
            "contact_sensor_ids": self.contact_feet,
            "mesh_path": paths[0],
            "columns": self.columns.cpu().tolist(),
            "dt_s": env.step_dt,
            "contact_force_threshold_n": self.contact_threshold,
        }
        self.counts = torch.zeros(len(COUNT_NAMES), dtype=torch.long, device=env.device)
        self.hist = torch.zeros(
            (4, len(GAP_EDGES) + 1), dtype=torch.long, device=env.device
        )
        self.exceeds = torch.zeros(
            (4, len(THRESHOLDS)), dtype=torch.long, device=env.device
        )
        self.contact_hist = torch.zeros(5, dtype=torch.long, device=env.device)
        self.loaded_hist = torch.zeros_like(self.contact_hist)
        self.long_air = torch.zeros(4, dtype=torch.long, device=env.device)
        self.low_static = torch.zeros_like(self.long_air)
        self.rate_sum = torch.zeros((), dtype=torch.float64, device=env.device)
        self.edges = self.origins.new_tensor(GAP_EDGES)
        self.thresholds = self.origins.new_tensor(THRESHOLDS)

    def read(self):
        """Read the current native post-physics state; no actor observation writes."""
        import torch
        from isaaclab.utils.warp import raycast_mesh

        env, data = self.env, self.robot.data
        position = data.body_link_pos_w[:, self.feet]
        velocity = data.body_link_lin_vel_w[:, self.feet]
        force = self.contacts.data.net_forces_w[:, self.contact_feet]
        air = self.contacts.data.current_air_time[:, self.contact_feet]
        command = env.command_manager.get_command("base_velocity")
        if (
            any(
                value.shape != (env.num_envs, 4, 3)
                for value in (position, velocity, force)
            )
            or air.shape != (env.num_envs, 4)
            or command.shape != (env.num_envs, 3)
        ):
            raise ValueError("Invalid native foot/command sample shape")
        finite = (
            torch.isfinite(command).all(1)
            & torch.isfinite(air).all(1)
            & (air >= 0).all(1)
        )
        for value in (position, velocity, force):
            finite &= torch.isfinite(value).all(dim=(1, 2))
        starts = torch.where(
            finite[:, None, None], position, self.origins[:, None, :]
        ).clone()
        starts[..., 2] = torch.maximum(starts[..., 2], self.origins[:, None, 2] + 1.0)
        direction = torch.zeros_like(starts)
        direction[..., 2] = -1
        hit, _, _, face = raycast_mesh(
            starts, direction, self.mesh, max_dist=5.0, return_face_id=True
        )
        if hit.shape != position.shape or face.shape != air.shape:
            raise ValueError("Invalid native foot-ray result shape")
        ray_valid = torch.isfinite(hit).all(2) & (face >= 0)
        gap = position[..., 2] - hit[..., 2]
        speed = torch.linalg.vector_norm(velocity[..., :2], dim=2)
        return gap, speed, force, air, command, finite, ray_valid, position

    def compute(self):
        import torch

        env = self.env
        if not self.active:
            return torch.zeros(env.num_envs, device=env.device)
        clock = env.common_step_counter
        if clock == self.last_clock:
            raise RuntimeError("Foot-link reward sampled twice for one native step")
        gap, speed, force, air, command, finite, ray_valid, _ = self.read()
        terminal = env.reset_terminated
        if terminal.shape != (env.num_envs,) or terminal.dtype != torch.bool:
            raise ValueError("Require the native physical termination mask")
        gate = self.step_mask & (torch.linalg.vector_norm(command[:, :2], dim=1) > 0.1)
        eligible = gate & ~terminal & finite
        valid = eligible[:, None] & ray_valid
        deficit = torch.clamp((0.12 - gap) / 0.12, 0, 1).square()
        per_foot = torch.where(valid, deficit * torch.tanh(2 * speed), 0.0)
        cost = per_foot.mean(1)
        contacts = torch.linalg.vector_norm(force, dim=2) > self.contact_threshold
        loaded = force[..., 2] > self.contact_threshold
        self.counts += torch.stack(
            (
                gate.sum(),
                terminal.sum(),
                (~finite).sum(),
                eligible.sum(),
                ray_valid.sum(),
                (~ray_valid).sum(),
                valid.sum(),
                (eligible & ~contacts.any(1)).sum(),
            )
        )
        bins = torch.bucketize(torch.nan_to_num(gap), self.edges)
        self.hist.scatter_add_(1, bins.T, valid.T.long())
        self.exceeds += ((gap[..., None] >= self.thresholds) & valid[..., None]).sum(0)
        self.contact_hist.scatter_add_(0, contacts.sum(1), eligible.long())
        self.loaded_hist.scatter_add_(0, loaded.sum(1), eligible.long())
        self.long_air += ((air > 0.5) & valid).sum(0)
        self.low_static += ((speed <= 0.01) & (gap < 0.12) & valid).sum(0)
        self.rate_sum += cost.sum(dtype=torch.float64)
        self.steps += 1
        self.last_clock = clock
        return cost

    def report(self):
        import torch

        if not torch.equal(self.columns, self.env.scene.terrain.terrain_types):
            raise ValueError("Foot-link reward terrain assignment changed")
        total = float(self.rate_sum)
        return {
            "recipe": recipe(),
            "binding": deepcopy(self.binding),
            "control_steps": self.steps,
            "environment_samples": self.steps * self.env.num_envs,
            **dict(zip(COUNT_NAMES, self.counts.cpu().tolist())),
            "gap_histogram_by_foot": self.hist.cpu().tolist(),
            "gap_exceedances_by_foot": self.exceeds.cpu().tolist(),
            "contact_count_histogram": self.contact_hist.cpu().tolist(),
            "upward_loaded_count_histogram": self.loaded_hist.cpu().tolist(),
            "long_air_samples_by_foot": self.long_air.cpu().tolist(),
            "low_gap_zero_speed_samples_by_foot": self.low_static.cpu().tolist(),
            "raw_rate_sum": total,
            "weighted_rate_sum": WEIGHT * total,
            "weighted_reward_integral": WEIGHT * total * self.env.step_dt,
        }


def validate_training_report(report, *, num_envs, steps):
    """Accounting admission only, never a clearance or locomotion success gate."""
    import numpy as np

    try:
        if (
            type(num_envs) is not int
            or num_envs <= 0
            or type(steps) is not int
            or steps <= 0
            or type(report["control_steps"]) is not int
            or type(report["environment_samples"]) is not int
            or report["recipe"] != recipe()
            or report["control_steps"] != steps
            or report["environment_samples"] != num_envs * steps
        ):
            raise ValueError("Foot-link reward budget/recipe mismatch")
        binding = report["binding"]
        if (
            binding["feet"] != list(FEET)
            or len(binding["columns"]) != num_envs
            or binding["dt_s"] != 0.02
            or any(type(x) is not int or not 0 <= x < 20 for x in binding["columns"])
            or any(
                len(binding[key]) != 4
                or len(set(binding[key])) != 4
                or any(type(x) is not int or x < 0 for x in binding[key])
                for key in ("articulation_ids", "contact_sensor_ids")
            )
            or binding["mesh_path"] != "/World/ground"
            or not math.isfinite(binding["contact_force_threshold_n"])
            or binding["contact_force_threshold_n"] <= 0
        ):
            raise ValueError("Invalid foot-link reward binding")
        if any(type(report[key]) is not int or report[key] < 0 for key in COUNT_NAMES):
            raise ValueError("Invalid foot-link reward counters")
        arrays = {}
        for name, shape in (
            ("gap_histogram_by_foot", (4, 10)),
            ("gap_exceedances_by_foot", (4, 3)),
            ("contact_count_histogram", (5,)),
            ("upward_loaded_count_histogram", (5,)),
            ("long_air_samples_by_foot", (4,)),
            ("low_gap_zero_speed_samples_by_foot", (4,)),
        ):
            value = np.asarray(report[name])
            if (
                value.shape != shape
                or not np.issubdtype(value.dtype, np.integer)
                or (value < 0).any()
            ):
                raise ValueError("Invalid foot-link reward histogram")
            arrays[name] = value
        eligible = report["eligible_environment_samples"]
        per_foot = arrays["gap_histogram_by_foot"].sum(1)
        exceeds = arrays["gap_exceedances_by_foot"]
        n = num_envs * steps
        if (
            report["ray_valid_foot_samples"] + report["ray_missed_foot_samples"]
            != 4 * n
            or not 0
            <= eligible
            <= report["translation_step_samples"]
            <= sum(12 <= x < 16 for x in binding["columns"]) * steps
            or max(report["physical_terminal_samples"], report["invalid_input_samples"])
            > n
            or arrays["gap_histogram_by_foot"].sum()
            != report["eligible_valid_foot_samples"]
            or report["eligible_valid_foot_samples"] > 4 * eligible
            or (per_foot > eligible).any()
            or (exceeds > per_foot[:, None]).any()
            or (np.diff(exceeds, axis=1) > 0).any()
            or any(
                (arrays[key] > per_foot).any()
                for key in (
                    "long_air_samples_by_foot",
                    "low_gap_zero_speed_samples_by_foot",
                )
            )
            or any(
                arrays[key].sum() != eligible
                for key in ("contact_count_histogram", "upward_loaded_count_histogram")
            )
            or report["all_feet_airborne_samples"]
            != arrays["contact_count_histogram"][0]
            or not math.isfinite(report["raw_rate_sum"])
            or not 0 <= report["raw_rate_sum"] <= eligible + 1e-6
            or report["weighted_rate_sum"] != WEIGHT * report["raw_rate_sum"]
            or report["weighted_reward_integral"]
            != WEIGHT * report["raw_rate_sum"] * 0.02
        ):
            raise ValueError("Inconsistent foot-link reward accounting")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed foot-link reward report") from error
    return deepcopy(report)
