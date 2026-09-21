"""Bounded command-curriculum refinement of a stock Go2 reference checkpoint.

No external Isaac Lab training script is needed. Actor, critic and action noise
are restored exactly. Command sampling and versioned reward/entropy profiles
are explicit; Adam starts fresh unless an evidence-bound retention resume is
requested. --procedural-config-check validates the replacement operator-only
configuration without constructing an environment, collecting transitions or
learning. Its simulator scan is a privileged fixture, not a deployed sensor.
--procedural-train learns a fresh proprioceptive GRU on supported easy terrain;
the positional checkpoint supplies only validated physical configuration, never
policy weights. The old four-family recipe is retired; archived readers remain.
--procedural-refine-checkpoint starts a separate recurrent refinement or matched
control run from learned weights and scalar LR, never from old optimizer state.
--procedural-resume-checkpoint continues the same recipe with saved Adam and
cumulative update numbers in a new run; simulator, command and GRU state reset.
--restart-coverage explicitly starts a low-speed restart sampling stage while
retaining that optimizer state. --pivot-planar-precision instead changes only
the pure-pivot planar precision mixture. Plain resume inherits either stage.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import importlib.metadata
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback

import yaml

try:
    from .operator_benchmark import reference_config, supervise, write_json
    from .operator_benchmark_core import (
        DT,
        OBSERVATION_TERMS,
        PROPRIO_EVALUATION_SEEDS,
        file_sha256,
        load_reference_checkpoint,
        read_yaml_data,
    )
    from .operator_curriculum import (
        OperatorExposureWrapper,
        VERSIONS,
        VERSION,
        curriculum_manifest,
    )
    from .operator_profiles import (
        PROFILES,
        apply_reward_profile,
        profile_manifest,
        select_profile,
        source_profile,
    )
    from .run_provenance import write_run_provenance
    from .operator_retention import (
        install_moving_retention,
        retention_manifest,
        restore_adam_state,
        validate_adam_state,
    )
    from .operator_sequences import (
        VERSION as SEQUENCE_VERSION,
        sequence_manifest,
        arrival_hold_manifest,
        restart_coverage_manifest,
        SequenceExposureWrapper,
    )
    from .operator_sequence_resume import sequence_resume_preflight
except ImportError:
    from operator_benchmark import reference_config, supervise, write_json
    from operator_benchmark_core import (
        DT,
        OBSERVATION_TERMS,
        PROPRIO_EVALUATION_SEEDS,
        file_sha256,
        load_reference_checkpoint,
        read_yaml_data,
    )
    from operator_curriculum import (
        OperatorExposureWrapper,
        VERSIONS,
        VERSION,
        curriculum_manifest,
    )
    from operator_profiles import (
        PROFILES,
        apply_reward_profile,
        profile_manifest,
        select_profile,
        source_profile,
    )
    from run_provenance import write_run_provenance
    from operator_retention import (
        install_moving_retention,
        retention_manifest,
        restore_adam_state,
        validate_adam_state,
    )
    from operator_sequences import (
        VERSION as SEQUENCE_VERSION,
        sequence_manifest,
        arrival_hold_manifest,
        restart_coverage_manifest,
        SequenceExposureWrapper,
    )
    from operator_sequence_resume import sequence_resume_preflight


def training_curriculum_manifest(version):
    return (
        sequence_manifest()
        if version == SEQUENCE_VERSION
        else curriculum_manifest(version)
    )


def retention_check_offsets(args):
    """Predeclare saved checkpoints; evaluation does not truncate learning."""
    offsets = getattr(args, "check_offsets", None)
    if not args.moving_retention:
        if offsets is not None:
            raise ValueError("--check-offsets requires --moving-retention")
        return []
    if args.curriculum == SEQUENCE_VERSION:
        if not 50 <= args.iterations <= 3000 or args.iterations % 50:
            raise ValueError("v3 requires 50-aligned iterations between 50 and 3000")
    elif args.iterations != 200:
        raise ValueError("The legacy v2 retention protocol requires 200 updates")
    if offsets is None:
        offsets = [100, 200] if args.iterations == 200 else [args.iterations]
    if (
        not 1 <= len(offsets) <= 3
        or offsets != sorted(set(offsets))
        or any(offset <= 0 or offset % 50 for offset in offsets)
        or offsets[-1] != args.iterations
    ):
        raise ValueError(
            "Predeclare one to three increasing, unique, positive 50-aligned check offsets, ending at --iterations"
        )
    if args.curriculum != SEQUENCE_VERSION and offsets != [100, 200]:
        raise ValueError("The legacy v2 retention checks remain +100/+200")
    return offsets


def training_configs(saved, agent, args):
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg import (
        UnitreeGo2FlatPPORunnerCfg,
    )

    try:
        from .operator_command import OperatorTransitionCommand, OperatorVelocityCommand
    except ImportError:
        from operator_command import OperatorTransitionCommand, OperatorVelocityCommand

    source = source_profile(saved, agent)
    selected = select_profile(
        saved, agent, getattr(args, "refinement_profile", "source")
    )
    cfg = reference_config(saved)
    runner_cfg = UnitreeGo2FlatPPORunnerCfg().to_dict()
    known_algorithm = yaml.load(
        yaml.dump(runner_cfg["algorithm"]), Loader=yaml.BaseLoader
    )
    known_algorithm["entropy_coef"] = str(source.entropy_coef)
    # Never execute class/function names from an archived YAML through RSL eval.
    # Only the named profile's entropy and the explicit optimizer protocol differ.
    for key in known_algorithm.keys() | agent["algorithm"].keys():
        if key not in ("learning_rate", "schedule") and agent["algorithm"].get(
            key
        ) != known_algorithm.get(key):
            raise ValueError(f"Unsupported source algorithm.{key}")
    apply_reward_profile(cfg, selected)
    command = cfg.commands.base_velocity
    version = getattr(args, "curriculum", VERSION)
    training_curriculum_manifest(version)
    command.class_type = (
        OperatorVelocityCommand if version == VERSION else OperatorTransitionCommand
    )
    if version == SEQUENCE_VERSION:
        try:
            from .operator_sequence_command import OperatorReversalSequenceCommand
        except ImportError:
            from operator_sequence_command import OperatorReversalSequenceCommand
        command.class_type = OperatorReversalSequenceCommand
    command.heading_command = False
    command.rel_heading_envs = 0.0
    command.rel_standing_envs = 0.0
    command.ranges.heading = None
    command.ranges.lin_vel_x = (-0.3, 0.7)
    command.ranges.lin_vel_y = (-0.2, 0.2)
    command.ranges.ang_vel_z = (-0.8, 0.8)
    command.resampling_time_range = (2.0, 12.0)
    command.debug_vis = False
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device
    # Retain observation corruption, 20-s episodes, all reward weights, action
    # scaling, mass/friction/reset events, collision geometry and terminations.
    runner_cfg.update(
        seed=args.seed,
        device=args.device,
        max_iterations=args.iterations,
        experiment_name="go2_operator_refinement",
        run_name=version,
        logger="tensorboard",
        save_interval=50,
        resume=False,
        obs_groups={"policy": ["policy"], "critic": ["policy"]},
    )
    runner_cfg["algorithm"].update(
        learning_rate=1.0e-4, schedule="fixed", entropy_coef=selected.entropy_coef
    )
    return cfg, runner_cfg


def restore_reference(runner, data):
    """Exact actor/critic/std handoff, fresh optimizer, unambiguous next update."""
    import torch

    if runner.alg.optimizer.state:
        raise ValueError("Refinement requires a newly constructed, empty optimizer")
    runner.alg.policy.load_state_dict(data["model_state_dict"], strict=True)
    actual = runner.alg.policy.state_dict()
    if any(
        not torch.equal(actual[key].detach().cpu(), value)
        for key, value in data["model_state_dict"].items()
    ):
        raise RuntimeError("Initial actor/critic/std differ from the source checkpoint")
    runner.current_learning_iteration = data["iter"] + 1
    return {
        "source_iteration": data["iter"],
        "first_update": runner.current_learning_iteration,
        "restored": ["actor", "critic", "action_standard_deviation"],
        "initial_state_verified_exact": True,
        "optimizer": "fresh Adam; source optimizer deliberately not restored",
        "learning_rate": runner.alg.learning_rate,
        "schedule": runner.alg.schedule,
    }


TERRAIN_VERSION = "go2_operator_terrain_teacher_v1"
TERRAIN_READINESS_STEPS = 1000
TERRAIN_TRAINING_VERSION = "go2_operator_terrain_progressive_v1"
TERRAIN_TASK_CRITIC_VERSION = "go2_operator_terrain_progressive_task_critic_v2"
TERRAIN_CRITIC_CONTEXT = {
    "group": "critic_context",
    "fields": [
        "operator_role",
        "active_waypoint_distance_xy * 0.25",
        "route_cursor_phase",
        "safe_route_progress_phase",
    ],
    "course_fields_zero_on_operator_rows": True,
    "actor_access": False,
    "initialization": "zero additive value projection; preserve existing tensors and RNG",
}
TERRAIN_CHECK_UPDATES = (500, 1500, 3000)
TERRAIN_COURSE_REWARDS = (
    "waypoint_velocity_tracking",
    "waypoint_heading_alignment",
    "stationary_velocity_tracking",
    "route_cross_track_excess",
    "completed_course",
    "intermediate_milestone",
    "base_clearance_below",
    "action_rate_l2",
    "ang_vel_xy_l2",
    "flat_orientation_l2",
    "stable_orientation_l2",
    "joint_deviation_l2",
    "joint_torques_l2",
)
TERRAIN_ARTIFACTS = (
    "trace.npz",
    "terrain_readiness.pt",
    "params/interface.json",
    "params/env.yaml",
    "params/agent.yaml",
)


def validate_terrain_scan(scan, *, num_envs):
    """Check the delivered production scan, whose heights are normalized, not metres."""
    import torch

    if scan.shape != (num_envs, 264):
        raise ValueError(f"Expected delivered terrain scan [{num_envs}, 264]")
    if not torch.isfinite(scan).all():
        raise ValueError("Nonfinite delivered terrain scan")
    heights, valid = scan.split(132, dim=-1)
    # _terrain_height_components clips in metres, then divides by that bound.
    # A metric clip of 0.5 m therefore still produces heights in [-1, 1].
    if (heights.abs() > 1).any():
        raise ValueError("Delivered normalized terrain heights must be in [-1, 1]")
    if not ((valid == 0) | (valid == 1)).all():
        raise ValueError("Delivered terrain validity bits must be exactly 0 or 1")
    if ((valid == 0) & (heights != 1)).any():
        raise ValueError("Missing terrain rays must use normalized height +1")


def report_terrain_readiness_error(output, error, progress):
    """Expose and persist the cause before Kit shutdown can exit Python."""
    stack = traceback.format_exc()
    print(
        f"Terrain readiness ERROR: {error}\nProgress: {progress}\n{stack}",
        file=sys.stderr,
        flush=True,
    )
    write_json(
        output / "training_status.json",
        {
            "status": "ERROR",
            "error": str(error),
            "traceback": stack,
            "progress": progress,
        },
    )


def validate_terrain_critic_context(context, *, num_envs):
    """Validate the explicit four-value training-only schema at construction."""
    import torch

    if (
        context.shape != (num_envs, 4)
        or context.dtype != torch.float32
        or not torch.isfinite(context).all()
        or not ((context[:, 0] == 0) | (context[:, 0] == 1)).all()
        or (context[:, 1:] < 0).any()
        or (context[:, 2:] > 1).any()
        or torch.count_nonzero(context[context[:, 0] == 1, 1:])
    ):
        raise ValueError("Invalid four-value critic task context")


def terrain_training_version(critic_context):
    return TERRAIN_TASK_CRITIC_VERSION if critic_context else TERRAIN_TRAINING_VERSION


def build_terrain_policy(observations, source_state, *, critic_context=False):
    """Explicit stock-to-terrain warm start; never load legacy RMA or Adam."""
    import torch
    from rsl_rl.modules import ActorCritic
    from parkour_lab.learning.distillation.teacher.model import StockTerrainInput

    if (
        observations["policy"].ndim != 2
        or observations["policy"].shape[-1] != 48
        or observations["terrain"].shape != (observations["policy"].shape[0], 264)
    ):
        raise ValueError(
            "Terrain teacher requires separate 48-D policy and 264-D scan groups"
        )
    if any(
        v.dtype != torch.float32 or not torch.isfinite(v).all()
        for v in source_state.values()
    ):
        raise ValueError("Warm start requires finite float32 reference tensors")
    if critic_context:
        if "critic_context" not in observations.keys():
            raise ValueError(
                "The task-context teacher requires a critic_context observation group"
            )
        validate_terrain_critic_context(
            observations["critic_context"], num_envs=len(observations["policy"])
        )
    reference = ActorCritic(
        observations,
        {"policy": ["policy"], "critic": ["policy"]},
        12,
        actor_hidden_dims=[128] * 3,
        critic_hidden_dims=[128] * 3,
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
    )
    reference.load_state_dict(source_state, strict=True)
    if (reference.std <= 0).any():
        raise ValueError("Warm-start action noise must be positive")
    policy = copy.deepcopy(reference)
    policy.obs_groups = {
        "policy": ["policy", "terrain"],
        "critic": ["policy", "terrain"],
    }
    policy.actor[0] = StockTerrainInput(policy.actor[0])
    policy.critic[0] = StockTerrainInput(
        policy.critic[0], critic_task_dim=4 if critic_context else 0
    )
    if critic_context:
        policy.obs_groups["critic"].append("critic_context")
    device = observations["policy"].device
    return policy.to(device), reference.to(device).eval().requires_grad_(False)


PROCEDURAL_TERRAIN_VERSION = "go2_operator_procedural_terrain_v2"
# Official release wheels; post1 is also the version in our recorded GPU runs.
# Source-checkout extension versions (e.g. 0.54.2) are a separate identity and
# are not admitted by this wheel-only check. Do not use a broad 2.3.* match.
PROCEDURAL_ISAACLAB_DISTRIBUTIONS = ("2.3.2", "2.3.2.post1")
PROCEDURAL_EASY_DIFFICULTY = (0.05, 0.15)
PROPRIO_ACQUISITION_VERSION = "operator_proprio_acquisition_v3"
PROPRIO_STOP_PRECISION_VERSION = "operator_proprio_stop_precision_v1"
PROPRIO_TERRAIN_EXPOSURE_VERSION = "operator_proprio_terrain_exposure_v1"
PROPRIO_ARRIVAL_HOLD_VERSION = "operator_proprio_arrival_hold_v1"
PROPRIO_STANCE_VERSION = "operator_proprio_stance_v1"
PROPRIO_PIVOT_PRECISION_VERSION = "operator_proprio_pivot_precision_v1"
PROPRIO_STATIONARY_YAW_VERSION = "operator_proprio_stationary_yaw_v1"
PROPRIO_STOP_YAW_VERSION = "operator_proprio_stop_yaw_v1"
PROPRIO_LINK_ORIGIN_VERSION = "operator_proprio_link_origin_v1"
PROPRIO_HIGHER_TERRAIN_VERSION = "operator_proprio_higher_terrain_v1"
PROPRIO_LINK_ORIGIN_VERSIONS = (
    PROPRIO_LINK_ORIGIN_VERSION,
    PROPRIO_HIGHER_TERRAIN_VERSION,
)
PROPRIO_STOP_YAW_VERSIONS = (PROPRIO_STOP_YAW_VERSION, *PROPRIO_LINK_ORIGIN_VERSIONS)
PROPRIO_STATIONARY_YAW_VERSIONS = (
    PROPRIO_STATIONARY_YAW_VERSION,
    *PROPRIO_STOP_YAW_VERSIONS,
)
PROPRIO_PIVOT_VERSIONS = (
    PROPRIO_PIVOT_PRECISION_VERSION,
    *PROPRIO_STATIONARY_YAW_VERSIONS,
)
PROPRIO_STANCE_VERSIONS = (PROPRIO_STANCE_VERSION, *PROPRIO_PIVOT_VERSIONS)
PROPRIO_ARRIVAL_VERSIONS = (PROPRIO_ARRIVAL_HOLD_VERSION, *PROPRIO_STANCE_VERSIONS)
PROPRIO_MIXED_TERRAIN_VERSIONS = (
    PROPRIO_TERRAIN_EXPOSURE_VERSION,
    *PROPRIO_ARRIVAL_VERSIONS,
)
PROPRIO_REFINEMENT_VERSIONS = {
    "stock": PROPRIO_ACQUISITION_VERSION,
    "terrain_exposure": PROPRIO_TERRAIN_EXPOSURE_VERSION,
    "arrival_hold": PROPRIO_ARRIVAL_HOLD_VERSION,
    "stance": PROPRIO_STANCE_VERSION,
    "pivot_precision": PROPRIO_PIVOT_PRECISION_VERSION,
    "stationary_yaw": PROPRIO_STATIONARY_YAW_VERSION,
    "stop_yaw": PROPRIO_STOP_YAW_VERSION,
    "link_origin": PROPRIO_LINK_ORIGIN_VERSION,
    "higher_terrain": PROPRIO_HIGHER_TERRAIN_VERSION,
    "stop_precision": PROPRIO_STOP_PRECISION_VERSION,
}
PROPRIO_JOINT_LIMIT_VERSION = "operator_proprio_acquisition_v2"
PROPRIO_LEGACY_ACQUISITION_VERSION = "operator_proprio_acquisition_v1"
PROPRIO_ACQUISITION_VERSIONS = (
    PROPRIO_LEGACY_ACQUISITION_VERSION,
    PROPRIO_JOINT_LIMIT_VERSION,
    PROPRIO_ACQUISITION_VERSION,
)
PROPRIO_TRAINING_VERSIONS = (
    *PROPRIO_ACQUISITION_VERSIONS,
    PROPRIO_STOP_PRECISION_VERSION,
    *PROPRIO_MIXED_TERRAIN_VERSIONS,
)
PROPRIO_UPRIGHT_VERSIONS = (
    PROPRIO_ACQUISITION_VERSION,
    PROPRIO_STOP_PRECISION_VERSION,
    *PROPRIO_MIXED_TERRAIN_VERSIONS,
)
# Shared by launch preflight and archived-checkpoint validation. A stance
# restart is the unchanged-objective control, not an uninterrupted resume.
PROPRIO_WARM_START_SOURCES = {
    PROPRIO_ACQUISITION_VERSION: (PROPRIO_ACQUISITION_VERSION,),
    PROPRIO_STOP_PRECISION_VERSION: (PROPRIO_ACQUISITION_VERSION,),
    PROPRIO_TERRAIN_EXPOSURE_VERSION: (PROPRIO_ACQUISITION_VERSION,),
    PROPRIO_ARRIVAL_HOLD_VERSION: (PROPRIO_TERRAIN_EXPOSURE_VERSION,),
    PROPRIO_STANCE_VERSION: (PROPRIO_TERRAIN_EXPOSURE_VERSION, PROPRIO_STANCE_VERSION),
    PROPRIO_PIVOT_PRECISION_VERSION: (PROPRIO_STANCE_VERSION,),
    PROPRIO_STATIONARY_YAW_VERSION: (PROPRIO_STANCE_VERSION,),
    PROPRIO_STOP_YAW_VERSION: (PROPRIO_STANCE_VERSION, PROPRIO_STOP_YAW_VERSION),
    PROPRIO_LINK_ORIGIN_VERSION: (
        PROPRIO_STOP_YAW_VERSION,
        PROPRIO_LINK_ORIGIN_VERSION,
    ),
    PROPRIO_HIGHER_TERRAIN_VERSION: (PROPRIO_LINK_ORIGIN_VERSION,),
}
PROPRIO_TERRAIN_EXPOSURE_CHANGE = {
    "difficulty_range": [0.05, 0.35],
    "num_rows": 3,
    "row_bands": [[0.05, 0.15], [0.15, 0.25], [0.25, 0.35]],
    "assignment": "native uniform initial row sampling, fixed for the run; realized row/profile counts are recorded, not assumed balanced",
    "retention": "20 percent plane columns in every row; lowest row retains the acquisition amplitude band, not the identical old meshes",
    "scope": "one static mixed-amplitude exposure stage; no adaptive promotion, reward, command, episode-duration, policy or motor change",
}
PROPRIO_HIGHER_TERRAIN_CHANGE = {
    **PROPRIO_TERRAIN_EXPOSURE_CHANGE,
    "difficulty_range": [0.05, 0.55],
    "num_rows": 5,
    "row_bands": [[0.05, 0.15], [0.15, 0.25], [0.25, 0.35], [0.35, 0.45], [0.45, 0.55]],
    "retention": "20 percent plane columns in every row; three of five row bands retain earlier amplitudes, not identical meshes or equal per-band transition budgets",
}
PROPRIO_STOP_PRECISION_CHANGE = {
    "gate": "exact zero body twist only; moving and pure-pivot rewards unchanged",
    "broad_std": 0.5,
    "planar_std_m_s": 0.05,
    "yaw_std_rad_s": 0.1,
    "precision_fraction": 1.0 / 3.0,
    "kernel": "(1-f)*exp(-error_squared/broad_std**2) + f*exp(-error_squared/fine_std**2)",
    "scope": "reward-only refinement; no pose anchor, command override, entropy or terrain change",
}
PROPRIO_STANCE_CHANGE = {
    "term": "joint_posture_stopped",
    "function": "operator_rewards:joint_posture_stopped",
    "weight": -0.1,
    "gate": "exact zero body twist only; no measured-speed gate",
    "cost": "L2 norm of all 12 measured joint positions minus native default positions, radians; not squared",
    "integration": "native RewardManager weight * step_dt exactly once",
    "scope": "soft posture prior, not a rigid stance or world-pose anchor; arrival sampler, moving/pivot rewards, actor and motors unchanged",
}
PROPRIO_PIVOT_PRECISION_CHANGE = {
    "term": "track_lin_vel_xy_exp",
    "function": "operator_rewards:track_lin_vel_xy_stationary",
    "gate": "exact zero planar command and nonzero yaw command, either sign; no measured-speed gate",
    "broad_std_m_s": 0.5,
    "planar_std_m_s": 0.05,
    "precision_fraction": 0.1,
    "weight": 1.5,
    "kernel": "0.9*exp(-planar_error_squared/0.5**2) + 0.1*exp(-planar_error_squared/0.05**2)",
    "integration": "native RewardManager weight * step_dt exactly once",
    "scope": "pure-pivot planar reward only; exact-zero and translating rewards, yaw objective, stance cost, commands, terrain, actor and motors unchanged; no pose anchor or inference assistance",
}
PIVOT_PLANAR_PRECISION_CHANGE = {
    "term": "track_lin_vel_xy_exp",
    "function": "operator_rewards:track_lin_vel_xy_stationary",
    "gate": "exact zero planar command and nonzero yaw command, either sign; no measured-speed gate",
    "from_precision_fraction": 0.1,
    "to_precision_fraction": 0.3,
    "broad_std_m_s": 0.5,
    "planar_std_m_s": 0.05,
    "weight": 1.5,
    "root_link_velocity": True,
    "kernel": "0.7*exp(-planar_error_squared/0.5**2) + 0.3*exp(-planar_error_squared/0.05**2)",
    "scope": "one pure-pivot planar mixture delta; exact-stop/translating rewards, yaw/stance objectives, sampler, terrain, actor, motor and evaluation gates unchanged; no pose anchor or inference assistance",
}
PROPRIO_STATIONARY_YAW_CHANGE = {
    "term": "track_ang_vel_z_exp",
    "function": "operator_rewards:track_ang_vel_z_stopped",
    "gate": "exact zero planar command; full stops and pure pivots of either sign; no measured-speed gate",
    "broad_std_rad_s": 0.5,
    "yaw_std_rad_s": 0.1,
    "precision_fraction": 0.1,
    "weight": 0.75,
    "kernel": "0.9*exp(-yaw_error_squared/0.5**2) + 0.1*exp(-yaw_error_squared/0.1**2)",
    "integration": "native RewardManager weight * step_dt exactly once",
    "scope": "angular delta on top of pivot_precision; keep its planar term and stance cost; translating/arc rewards, commands, terrain, actor and motors unchanged; no heading anchor or inference assistance",
}
PROPRIO_STOP_YAW_CHANGE = {
    "term": "track_ang_vel_z_exp",
    "function": "operator_rewards:track_ang_vel_z_stopped",
    "gate": "exact zero body twist only; no measured-speed gate",
    "from_yaw_std_rad_s": 0.1,
    "to_yaw_std_rad_s": 0.05,
    "broad_std_rad_s": 0.5,
    "precision_fraction": 0.1,
    "weight": 0.75,
    "kernel": "0.9*exp(-yaw_error_squared/0.5**2) + 0.1*exp(-yaw_error_squared/0.05**2)",
    "integration": "native RewardManager weight * step_dt exactly once",
    "scope": "one full-stop angular width delta versus stationary_yaw; pure-pivot and translating kernels, stance cost, commands, terrain, actor and motors unchanged; no heading anchor or inference assistance",
}
PROPRIO_LINK_ORIGIN_CHANGE = {
    "term": "track_lin_vel_xy_exp",
    "parameter": "root_link_velocity",
    "from": "root_lin_vel_b (root-body COM, expressed in link axes)",
    "to": "root_link_lin_vel_b (root-link origin, expressed in link axes)",
    "gate": "all commands, both yaw signs; no phase or measured-speed gate",
    "identity": "v_link_b = v_com_b - cross(omega_b, body_com_pos_b[:, 0])",
    "scope": "reference-point delta only versus stop_yaw; same kernels, weights, yaw/posture objectives, actor/critic observations, commands, motors and historical evaluation gates; not a world-pose anchor or whole-robot COM controller",
}
PROPRIO_REWARD_CHANGE = {
    "term": "dof_pos_limits",
    "function": "isaaclab.envs.mdp.rewards:joint_pos_limits",
    "from_weight": 0.0,
    "to_weight": -10.0,
    "soft_joint_pos_limit_factor": 0.9,
}
PROPRIO_POSTURE_CHANGE = {
    "reward": {"term": "flat_orientation_l2", "from_weight": 0.0, "to_weight": -2.5},
    "termination": {
        "term": "procedural_physical_failure",
        "max_tilt_rad": math.pi / 4,
        "time_out": False,
        "bootstrap": "physical failures take precedence over simultaneous time limits",
        "criterion": "total body tilt from world upright; projected_gravity_b.z > -cos(max_tilt_rad)",
    },
    "scope": "fixed easy difficulty 0.05–0.15 only; task repair, not a single-factor ablation or terrain acceptance",
}
PROCEDURAL_ROLLOUT_STEPS = 200
PROCEDURAL_SCAN_INTERFACE = {
    "version": "privileged_centered_height_valid_132_v1",
    "height_count": 132,
    "validity_count": 132,
    "grid_size_m": [1.65, 1.50],
    "grid_resolution_m": 0.15,
    "grid_offset_m": [0.0, 0.0, 20.0],
    "ordering": "xy",
    "ray_alignment": "yaw",
    "vertical_offset_m": 0.30,
    "clip_m": 0.50,
    "deployment_claim": "privileged teacher only; causal student sensing unresolved",
    "checkpoint_compatibility": "stock 48-D warm start only; no legacy terrain resume",
}


def _proprio_posture_change(version):
    if version not in PROPRIO_UPRIGHT_VERSIONS:
        return None
    result = copy.deepcopy(PROPRIO_POSTURE_CHANGE)
    if version in PROPRIO_MIXED_TERRAIN_VERSIONS:
        result["scope"] = (
            "retain the v3 posture objective and physical failure guard during bounded terrain exposure; not terrain acceptance"
        )
    return result


def _proprio_terrain_exposure(version):
    """Immutable stage geometry, shared by reconstruction, validation and launch."""
    return (
        PROPRIO_HIGHER_TERRAIN_CHANGE
        if version == PROPRIO_HIGHER_TERRAIN_VERSION
        else PROPRIO_TERRAIN_EXPOSURE_CHANGE
    )


def _configure_recurrent_terrain(cfg, difficulty_range, *, num_rows=1):
    """Set static layout and its derived floor bound together, before construction."""
    from parkour_lab.tasks.manager_based.parkour_lab.mdp.terrain.operator_terrain import (
        ENVELOPES,
    )

    generator = cfg.scene.terrain.terrain_generator
    generator.difficulty_range = tuple(difficulty_range)
    generator.num_rows = num_rows
    cfg.scene.terrain.max_init_terrain_level = num_rows - 1
    cfg.terminations.procedural_physical_failure.params["minimum_surface_z_m"] = (
        -max(height for height, _ in ENVELOPES.values()) * difficulty_range[1]
    )


def _procedural_environment_configs(saved, agent, args):
    """Build shared operator terrain and stock physics without choosing a policy.

    This is a configuration/preflight contract, not a training or acceptance
    claim. A fixed easy row deliberately has no adaptive progression until
    command-integrated traversal criteria and deployed sensing are validated.
    """
    from isaaclab.managers import (
        ObservationGroupCfg,
        ObservationTermCfg,
        SceneEntityCfg,
        TerminationTermCfg,
    )
    from isaaclab.sensors import RayCasterCfg, patterns
    from parkour_lab.tasks.manager_based.parkour_lab import mdp
    from parkour_lab.tasks.manager_based.parkour_lab.mdp.terrain.operator_terrain import (
        BORDER_WIDTH,
        ENVELOPES,
        make_operator_terrain_generator,
    )

    try:
        from .operator_command import (
            ProceduralTerrainCommand,
            procedural_physical_failure,
            procedural_workspace,
        )
    except ImportError:
        from operator_command import (
            ProceduralTerrainCommand,
            procedural_physical_failure,
            procedural_workspace,
        )

    # Reconstruct/validate the stock physical checkpoint contract first.
    # This task explicitly chooses v3 for its level-terrain training draws.
    procedural_args = copy.copy(args)
    procedural_args.curriculum = SEQUENCE_VERSION
    cfg, runner_cfg = training_configs(saved, agent, procedural_args)
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = make_operator_terrain_generator(
        seed=args.seed,
        num_rows=1,
        difficulty_range=PROCEDURAL_EASY_DIFFICULTY,
        # Native generator flag means deterministic profile columns here;
        # one row plus no manager curriculum means NO level promotion.
        curriculum=True,
    )
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None
    cfg.commands.base_velocity.class_type = ProceduralTerrainCommand
    # The reset support pad extends one metre in both axes. The smaller root
    # jitter leaves room for the complete Go2 support footprint at any yaw.
    cfg.events.reset_base.params["pose_range"] = {
        "x": (-0.2, 0.2),
        "y": (-0.2, 0.2),
        "yaw": (-math.pi, math.pi),
    }
    # Match the pinned Isaac Lab Go2 rough baseline for the two explicitly
    # flat-specific rewards. All motors, actions, other rewards, noise and
    # startup physical randomization still come from the validated reference.
    cfg.rewards.flat_orientation_l2.weight = 0.0
    cfg.rewards.feet_air_time.weight = 0.01

    scan = PROCEDURAL_SCAN_INTERFACE
    cfg.scene.height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=tuple(scan["grid_offset_m"])),
        ray_alignment=scan["ray_alignment"],
        pattern_cfg=patterns.GridPatternCfg(
            resolution=scan["grid_resolution_m"],
            size=tuple(scan["grid_size_m"]),
            direction=(0.0, 0.0, -1.0),
            ordering=scan["ordering"],
        ),
        mesh_prim_paths=[cfg.scene.terrain.prim_path],
        max_distance=25.0,
        update_period=DT,
        debug_vis=False,
    )
    cfg.scene.base_height_scanner = cfg.scene.height_scanner.replace(
        pattern_cfg=patterns.GridPatternCfg(
            resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)
        )
    )
    terrain_observations = ObservationGroupCfg(
        enable_corruption=False, concatenate_terms=True
    )
    terrain_observations.height_scan = ObservationTermCfg(
        func=mdp.terrain_height_scan,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "obs_cfg": mdp.config.HeightScanObservationCfg(
                num_rays=scan["height_count"],
                vertical_offset=scan["vertical_offset_m"],
                clip=scan["clip_m"],
            ),
        },
    )
    cfg.observations.terrain = terrain_observations
    cfg.terminations.procedural_physical_failure = TerminationTermCfg(
        func=procedural_physical_failure,
        params={
            "minimum_m": 0.12,
            "minimum_surface_z_m": -max(height for height, _ in ENVELOPES.values())
            * PROCEDURAL_EASY_DIFFICULTY[1],
            "fall_margin_m": 0.5,
        },
    )
    # Install last so existing base contact / terrain-relative physical
    # failure cannot become a bootstrapped workspace timeout.
    cfg.terminations.procedural_workspace = TerminationTermCfg(
        func=procedural_workspace,
        params={"margin_m": BORDER_WIDTH + 0.25},
        time_out=True,
    )
    return cfg, runner_cfg


def procedural_terrain_configs(saved, agent, args):
    """Privileged stock-to-terrain fixture for configuration and motor diagnostics."""
    cfg, runner_cfg = _procedural_environment_configs(saved, agent, args)
    runner_cfg.update(
        run_name=PROCEDURAL_TERRAIN_VERSION,
        obs_groups={"policy": ["policy", "terrain"], "critic": ["policy", "terrain"]},
        terrain_warm_start_builder="operator_train.build_terrain_policy",
        interface_version=PROCEDURAL_TERRAIN_VERSION,
    )
    return cfg, runner_cfg


def proprioceptive_procedural_configs(
    saved, agent, args, *, acquisition_version=PROPRIO_ACQUISITION_VERSION
):
    """Causal GRU recipe on versioned, static supported terrain.

    The source configuration binds the physical motor only. This policy starts
    from fresh weights unless a separately validated recurrent source is supplied;
    neither stock8500 nor the terrain teacher can initialize this actor.
    Terrain promotion and held-out behavior qualification remain separate work.
    """
    try:
        from .operator_student_bridge import (
            RECURRENT_OPERATOR_VERSION,
            recurrent_policy_config,
        )
    except ImportError:
        from operator_student_bridge import (
            RECURRENT_OPERATOR_VERSION,
            recurrent_policy_config,
        )

    if acquisition_version not in PROPRIO_TRAINING_VERSIONS:
        raise ValueError("Unsupported proprioceptive acquisition version")
    cfg, runner_cfg = _procedural_environment_configs(saved, agent, args)
    # Do not inherit the source8500 fine stationary kernels into a fresh actor.
    # Keep the rough orientation/air-time adjustments and physical motor.
    apply_reward_profile(cfg, PROFILES["stock"])
    # One-term fresh-learning ablation, not target clipping or a motor change.
    # Preserve the zero-weight recipe when reconstructing archived v1 runs.
    if acquisition_version != PROPRIO_LEGACY_ACQUISITION_VERSION:
        if cfg.scene.robot.soft_joint_pos_limit_factor != 0.9:
            raise ValueError(
                "The joint-limit ablation requires the stock 0.9 soft factor"
            )
        cfg.rewards.dof_pos_limits.weight = PROPRIO_REWARD_CHANGE["to_weight"]
    if acquisition_version in PROPRIO_UPRIGHT_VERSIONS:
        generator = cfg.scene.terrain.terrain_generator
        if (
            tuple(generator.difficulty_range) != (0.05, 0.15)
            or generator.num_rows != 1
            or cfg.curriculum.terrain_levels is not None
        ):
            raise ValueError(
                "The posture repair requires fixed easy acquisition terrain"
            )
        # The pinned Go2 flat baseline supplies this smooth posture objective.
        # Keep failure in the existing physical term, BEFORE workspace timeout.
        cfg.rewards.flat_orientation_l2.weight = PROPRIO_POSTURE_CHANGE["reward"][
            "to_weight"
        ]
        failure = cfg.terminations.procedural_physical_failure
        failure.params["max_tilt_rad"] = PROPRIO_POSTURE_CHANGE["termination"][
            "max_tilt_rad"
        ]
        failure.time_out = False
    if acquisition_version in PROPRIO_MIXED_TERRAIN_VERSIONS:
        terrain = _proprio_terrain_exposure(acquisition_version)
        _configure_recurrent_terrain(
            cfg,
            terrain["difficulty_range"],
            num_rows=terrain["num_rows"],
        )
    if acquisition_version in PROPRIO_ARRIVAL_VERSIONS:
        if cfg.episode_length_s != 20.0:
            raise ValueError(
                "Arrival/hold training requires the unchanged 20-second horizon"
            )
        try:
            from .operator_command import ProceduralArrivalHoldCommand
        except ImportError:
            from operator_command import ProceduralArrivalHoldCommand
        cfg.commands.base_velocity.class_type = ProceduralArrivalHoldCommand
    if acquisition_version in PROPRIO_STANCE_VERSIONS:
        from isaaclab.managers import RewardTermCfg

        try:
            from .operator_rewards import joint_posture_stopped
        except ImportError:
            from operator_rewards import joint_posture_stopped
        cfg.rewards.joint_posture_stopped = RewardTermCfg(
            func=joint_posture_stopped,
            weight=PROPRIO_STANCE_CHANGE["weight"],
            params={"command_name": "base_velocity"},
        )
    if acquisition_version in PROPRIO_PIVOT_VERSIONS:
        try:
            from .operator_rewards import track_lin_vel_xy_stationary
        except ImportError:
            from operator_rewards import track_lin_vel_xy_stationary
        change = PROPRIO_PIVOT_PRECISION_CHANGE
        term = cfg.rewards.track_lin_vel_xy_exp
        if (
            term.weight != change["weight"]
            or term.params["std"] != change["broad_std_m_s"]
        ):
            raise ValueError(
                "Pivot precision requires the unchanged stock tracking scale"
            )
        term.func = track_lin_vel_xy_stationary
        term.params.update(
            stationary_std=change["planar_std_m_s"],
            precision_fraction=change["precision_fraction"],
            pivot_only=True,
        )
    if acquisition_version in PROPRIO_STATIONARY_YAW_VERSIONS:
        try:
            from .operator_rewards import track_ang_vel_z_stopped
        except ImportError:
            from operator_rewards import track_ang_vel_z_stopped
        change = PROPRIO_STATIONARY_YAW_CHANGE
        term = cfg.rewards.track_ang_vel_z_exp
        if (
            term.weight != change["weight"]
            or term.params["std"] != change["broad_std_rad_s"]
        ):
            raise ValueError(
                "Stationary yaw requires the unchanged stock tracking scale"
            )
        term.func = track_ang_vel_z_stopped
        term.params.update(
            stationary_std=change["yaw_std_rad_s"],
            precision_fraction=change["precision_fraction"],
            include_pivots=True,
        )
        if acquisition_version in PROPRIO_STOP_YAW_VERSIONS:
            term.params["full_stop_std"] = PROPRIO_STOP_YAW_CHANGE["to_yaw_std_rad_s"]
    if acquisition_version in PROPRIO_LINK_ORIGIN_VERSIONS:
        cfg.rewards.track_lin_vel_xy_exp.params["root_link_velocity"] = True
    if acquisition_version == PROPRIO_STOP_PRECISION_VERSION:
        try:
            from .operator_rewards import (
                track_lin_vel_xy_stationary,
                track_ang_vel_z_stopped,
            )
        except ImportError:
            from operator_rewards import (
                track_lin_vel_xy_stationary,
                track_ang_vel_z_stopped,
            )
        change = PROPRIO_STOP_PRECISION_CHANGE
        for name, function, width in (
            ("track_lin_vel_xy_exp", track_lin_vel_xy_stationary, "planar_std_m_s"),
            ("track_ang_vel_z_exp", track_ang_vel_z_stopped, "yaw_std_rad_s"),
        ):
            term = getattr(cfg.rewards, name)
            term.func = function
            term.params.update(
                std=change["broad_std"],
                stationary_std=change[width],
                precision_fraction=change["precision_fraction"],
            )
        cfg.rewards.track_lin_vel_xy_exp.params["full_stop_only"] = True
    # Copy the noisy sensor group BEFORE making the privileged critic noiseless.
    # Removing this term at the manager boundary avoids passing oracle velocity
    # into actor normalization, recurrent state or inference preprocessing.
    cfg.observations.proprio = copy.deepcopy(cfg.observations.policy)
    cfg.observations.proprio.base_lin_vel = None
    cfg.observations.policy.enable_corruption = False
    runner_cfg.update(
        run_name=RECURRENT_OPERATOR_VERSION,
        interface_version=RECURRENT_OPERATOR_VERSION,
        save_interval=getattr(args, "save_interval", 50),
        policy=recurrent_policy_config(),
        obs_groups={"policy": ["proprio"], "critic": ["policy", "terrain"]},
        resume=False,
    )
    # Fresh acquisition uses the pinned Isaac Lab Go2 PPO recipe, not the
    # reference actor's 1e-4 fixed-rate refinement protocol. No auxiliary loss.
    runner_cfg["num_steps_per_env"] = 24
    runner_cfg["algorithm"].update(
        learning_rate=1.0e-3,
        schedule="adaptive",
        entropy_coef=0.01,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    return cfg, runner_cfg


def terrain_readiness_configs(saved, agent, args):
    """Reuse the stock motor and production geometry in one bounded fixture."""
    try:
        from .operator_benchmark import configure_terrain, make_recorder_cfg
        from .operator_command import (
            TerrainReadinessCommand,
            initialize_readiness_levels,
            readiness_course_success,
            readiness_course_off_route,
        )
    except ImportError:
        from operator_benchmark import configure_terrain, make_recorder_cfg
        from operator_command import (
            TerrainReadinessCommand,
            initialize_readiness_levels,
            readiness_course_success,
            readiness_course_off_route,
        )
    from parkour_lab.tasks.manager_based.parkour_lab.parkour_lab_env_cfg import (
        ParkourLabEnvCfg,
    )

    cfg, runner_cfg = training_configs(saved, agent, args)
    # This checked overlay supplies contacts and ordered stable-finish gates;
    # replace its single-family layout with the production training matrix.
    configure_terrain(cfg, "gap", 1)
    courses = ParkourLabEnvCfg()
    cfg.scene.terrain.terrain_generator = courses.scene.ground.terrain_generator
    cfg.scene.terrain.terrain_generator.seed = args.seed
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.scene.height_scanner = courses.scene.height_scanner
    cfg.scene.height_scanner.mesh_prim_paths = [cfg.scene.terrain.prim_path]
    cfg.scene.height_scanner.update_period = DT
    cfg.observations.terrain = courses.observations.terrain
    cfg.events.initialize_terrain_levels = courses.events.initialize_terrain_levels
    cfg.events.initialize_terrain_levels.func = initialize_readiness_levels
    cfg.events.initialize_terrain_levels.params.pop("initial_level_override")
    cfg.events.reset_routes = courses.events.reset_routes
    cfg.commands.base_velocity.class_type = TerrainReadinessCommand
    # Free body-twist operator episodes must not terminate for disregarding
    # route guidance. Physical base/head/below-course failures remain active.
    cfg.terminations.course_success.func = readiness_course_success
    cfg.terminations.course_off_route.func = readiness_course_off_route
    cfg.recorders = make_recorder_cfg()
    runner_cfg["obs_groups"] = {
        "policy": ["policy", "terrain"],
        "critic": ["policy", "terrain"],
    }
    runner_cfg["num_steps_per_env"] = TERRAIN_READINESS_STEPS
    runner_cfg["terrain_warm_start_builder"] = "operator_train.build_terrain_policy"
    runner_cfg["interface_version"] = TERRAIN_VERSION
    # Source rewards are a PPO plumbing fixture, not the final obstacle
    # objective. No retention loss, promotion schedule or convergence claim.
    return cfg, runner_cfg


def terrain_source_identity(args):
    """Bind parent and worker to exactly the same source, inputs and runtime."""
    root = Path(__file__).resolve().parents[2]
    producers = sorted(
        [
            *root.joinpath("scripts/rsl_rl").glob("*.py"),
            *root.joinpath("source/parkour_lab/parkour_lab").rglob("*.py"),
        ]
    )
    return {
        "checkpoint": file_sha256(args.checkpoint),
        "agent": file_sha256(args.checkpoint.parent / "params/agent.yaml"),
        "environment": file_sha256(args.checkpoint.parent / "params/env.yaml"),
        "mesh_flat_report": file_sha256(args.mesh_flat_report),
        "producers": {str(p.relative_to(root)): file_sha256(p) for p in producers},
        "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "rsl-rl-lib")
        },
    }


def terrain_execution_identity(args):
    """Keep diagnostic consumers separate from immutable checkpoint producers."""
    identity = terrain_source_identity(args)
    if getattr(args, "terrain_diagnostics", None) is None:
        return identity
    return {
        "runtime": identity,
        "audit": file_sha256(
            Path(__file__).resolve().parents[1] / "analysis/operator_terrain_audit.py"
        ),
    }


def validate_terrain_capture_config(output, baseline, training_run):
    """Only native reward observation may differ from the archived evaluation."""

    def read(path):
        # Script and -m entry points name the same in-repo callables differently.
        def normalize(value):
            if isinstance(value, dict):
                return {k: normalize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [normalize(v) for v in value]
            if isinstance(value, str) and value.startswith("scripts.rsl_rl."):
                return value.removeprefix("scripts.rsl_rl.")
            return value

        return normalize(read_yaml_data(path / "params/env.yaml"))

    current, archived, training = read(output), read(baseline), read(training_run)
    rewards = current.pop("rewards")
    if archived.pop("rewards") != {} or current != archived:
        raise ValueError("Diagnostic evaluation config differs beyond reward capture")
    expected = training["rewards"]
    expected["physical_failure"]["params"] = {
        "termination_names": [
            "base_contact",
            "course_chassis",
            "course_fall",
            "course_off_route",
        ]
    }
    if rewards != expected:
        raise ValueError(
            "Diagnostic rewards differ from the archived training objective"
        )


def terrain_update_evidence(policy, algorithm, losses):
    """Require actual finite learning in both new terrain branches."""
    import torch

    if not losses or any(not math.isfinite(float(value)) for value in losses.values()):
        raise ValueError("Nonfinite or missing terrain-readiness PPO losses")
    if (
        any(not torch.isfinite(p).all() for p in policy.parameters())
        or (policy.std <= 0).any()
    ):
        raise ValueError("Nonfinite policy or invalid action noise after PPO")
    expected_steps = algorithm.num_learning_epochs * algorithm.num_mini_batches
    if set(algorithm.optimizer.state) != set(policy.parameters()) or any(
        int(s["step"].item()) != expected_steps
        or not torch.isfinite(s["exp_avg"]).all()
        or not torch.isfinite(s["exp_avg_sq"]).all()
        for s in algorithm.optimizer.state.values()
    ):
        raise ValueError(
            "Readiness requires one fresh, fully finite Adam update sequence"
        )
    branches = {}
    for name in ("actor", "critic"):
        branch = getattr(policy, name)[0]
        encoder_gradients = [p.grad for p in branch.encoder.parameters()]
        gradient = branch.projection.weight.grad
        if (
            gradient is None
            or not torch.isfinite(gradient).all()
            or not torch.count_nonzero(gradient)
            or not torch.count_nonzero(branch.projection.weight)
            or any(g is None or not torch.isfinite(g).all() for g in encoder_gradients)
            or not any(torch.count_nonzero(g) for g in encoder_gradients)
        ):
            raise ValueError(
                f"No finite learning path through the {name} terrain encoder"
            )
        branches[name] = {
            "projection_max_abs": float(branch.projection.weight.detach().abs().max()),
            "encoder_gradient_norm": float(
                sum(g.square().sum() for g in encoder_gradients).sqrt()
            ),
        }
    return {
        "ppo_updates": 1,
        "adam_steps": expected_steps,
        "losses": losses,
        "terrain_branches": branches,
    }


def validate_readiness_report(path, identity, *, critic_context=False):
    """Consume completed engineering evidence, never its weights as a trained teacher."""
    import numpy as np
    import torch

    try:
        from .operator_benchmark import validate_motor_trace
    except ImportError:
        from operator_benchmark import validate_motor_trace

    report = json.loads(path.read_text())
    prior = report.get("source_identity", {})
    if (
        report.get("status") != "READINESS_PASS"
        or report.get("interface_version") != TERRAIN_VERSION
        or report.get("worker") != {"returncode": 0, "timed_out": False}
        or report.get("environment_transitions") != 80000
        or report.get("exact_action_and_value_parity_comparisons") != 80000
        or report.get("update", {}).get("adam_steps") != 20
        or report.get("update", {}).get("ppo_updates") != 1
        or report.get("behavior_validated") is not False
        or report.get("promoted") is not False
        or any(
            prior.get(k) != identity[k]
            for k in ("checkpoint", "agent", "environment", "mesh_flat_report")
        )
        or list(path.parent.glob("*_cleanup_error.json"))
    ):
        raise ValueError("A complete source-bound READINESS_PASS is required")
    protocol = json.loads((path.parent / "readiness_protocol.json").read_text())
    if protocol.get("source_identity") != prior:
        raise ValueError("Readiness protocol/report identity mismatch")
    # These are the explicit training-adapter changes. Geometry, motor, scan,
    # route, physical scoring and every other producer must remain unchanged.
    changed = {
        f"scripts/rsl_rl/{name}.py"
        for name in (
            "operator_train",
            "operator_command",
            "operator_curriculum",
            "operator_rewards",
            "operator_retention",
        )
    } | {
        "source/parkour_lab/parkour_lab/tasks/manager_based/parkour_lab/mdp/commands.py",
        "source/parkour_lab/parkour_lab/tasks/manager_based/parkour_lab/mdp/curriculums/curriculums.py",
    }
    if critic_context:
        # Explicit v2-only additive critic branch; default stock/terrain math
        # remains unchanged and is covered by exact parity/RNG regression tests.
        changed.add(
            "source/parkour_lab/parkour_lab/learning/distillation/teacher/model.py"
        )
    producers = prior.get("producers", {})
    if set(producers) != set(identity["producers"]) or any(
        value != identity["producers"][name]
        for name, value in producers.items()
        if name not in changed
    ):
        raise ValueError(
            "Readiness producer drift outside the declared training adapters"
        )
    if set(report.get("sha256", {})) != set(TERRAIN_ARTIFACTS) or any(
        file_sha256(path.parent / name) != report["sha256"][name]
        for name in TERRAIN_ARTIFACTS
    ):
        raise ValueError("Readiness artifacts are missing or changed")
    with np.load(path.parent / "trace.npz", allow_pickle=False) as archive:
        trace = {
            k: archive[k]
            for k in (
                "action",
                "joint_target",
                "default_joint_position",
                "observation",
                "command",
                "terminated",
                "time_out",
            )
        }
        if trace["action"].shape != (1000, 80, 12):
            raise ValueError("Incomplete readiness rollout")
        validate_motor_trace(trace)
        scan = torch.from_numpy(archive["terrain_observation"].reshape(-1, 264))
        validate_terrain_scan(scan, num_envs=80000)
    data = torch.load(
        path.parent / "terrain_readiness.pt", map_location="cpu", weights_only=True
    )
    state = data["model_state_dict"]
    adam = data["optimizer_state_dict"]["state"]
    if (
        data.get("readiness_only") is not True
        or data.get("learning_updates") != 1
        or len(adam) != len(state)
        or any(not torch.isfinite(v).all() for v in state.values())
        or any(
            s["step"].item() != 20
            or not torch.isfinite(s["exp_avg"]).all()
            or not torch.isfinite(s["exp_avg_sq"]).all()
            for s in adam.values()
        )
        or any(
            not torch.count_nonzero(state[f"{branch}.0.projection.weight"])
            for branch in ("actor", "critic")
        )
    ):
        raise ValueError("Readiness checkpoint lacks finite learning evidence")
    return {
        "report_sha256": file_sha256(path),
        "artifacts": report["sha256"],
        "source_identity": prior,
        "use": "prerequisite only; fresh zero-fusion source warm start",
    }


def terrain_teacher_configs(saved, agent, args, *, evaluation=False):
    """One opt-in objective on the verified motor/scan and native course matrix."""
    from isaaclab.managers import (
        ObservationGroupCfg,
        ObservationTermCfg,
        RewardTermCfg,
        TerminationTermCfg,
        RecorderManagerBaseCfg,
    )
    from parkour_lab.tasks.manager_based.parkour_lab.parkour_lab_env_cfg import (
        ParkourLabEnvCfg,
    )

    try:
        from . import operator_command as binding
        from .operator_rewards import teacher_physical_failure
    except ImportError:
        import operator_command as binding
        from operator_rewards import teacher_physical_failure

    cfg, runner_cfg = terrain_readiness_configs(saved, agent, args)
    courses = ParkourLabEnvCfg()
    cfg.terrain_teacher_evaluation = evaluation
    cfg.scene.num_envs = 160 if evaluation else args.num_envs
    cfg.seed = 43 if evaluation else args.seed
    cfg.scene.terrain.terrain_generator.seed = cfg.seed
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.events.initialize_terrain_levels.func = binding.initialize_teacher_levels
    cfg.commands.base_velocity.class_type = binding.TerrainTeacherCommand
    cfg.body_twist_command_name = "base_velocity"
    cfg.parkour_termination_names = {
        "chassis_contact": "course_chassis",
        "fell_below_course": "course_fall",
        "off_route": "course_off_route",
        "success": "course_success",
        "wall_time_out": None,
    }
    cfg.parkour_extra_failure_terms = ("base_contact", "persistent_tilt")
    cfg.terminations.course_success.func = binding.teacher_course_success
    cfg.terminations.course_off_route.func = binding.teacher_course_off_route
    cfg.terminations.persistent_tilt = TerminationTermCfg(
        func=binding.PersistentTilt, params={"angle_deg": 70.0, "duration_s": 0.5}
    )
    cfg.terminations.operator_workspace = TerminationTermCfg(
        func=binding.teacher_operator_workspace, params={"margin_m": 0.5}, time_out=True
    )
    rewards = {}
    for prefix, operator, terms in (
        ("operator", True, vars(cfg.rewards).items()),
        (
            "course",
            False,
            ((name, getattr(courses.rewards, name)) for name in TERRAIN_COURSE_REWARDS),
        ),
    ):
        for name, term in terms:
            if isinstance(term, RewardTermCfg) and term.weight:
                rewards[f"{prefix}_{name}"] = RewardTermCfg(
                    func=binding.RoleReward,
                    weight=term.weight,
                    params={"term": copy.deepcopy(term), "operator": operator},
                )
    rewards["physical_failure"] = RewardTermCfg(
        func=teacher_physical_failure, weight=-10.0
    )
    cfg.rewards = rewards
    cfg.curriculum = {
        "terrain_levels": copy.deepcopy(courses.curriculum.terrain_levels)
    }
    cfg.curriculum["terrain_levels"].func = binding.TerrainTeacherCurriculum
    mask = ObservationGroupCfg(concatenate_terms=True, enable_corruption=False)
    mask.role = ObservationTermCfg(func=binding.terrain_operator_mask)
    cfg.observations.operator_mask = mask
    critic_context = getattr(args, "terrain_critic_context", False)
    if critic_context:
        task = ObservationGroupCfg(concatenate_terms=True, enable_corruption=False)
        task.context = ObservationTermCfg(func=binding.terrain_critic_context)
        cfg.observations.critic_context = task
        runner_cfg["obs_groups"]["critic"] = ["policy", "terrain", "critic_context"]
    if evaluation:
        # Same physical scoring as stock benchmark, including no extra tilt gate.
        cfg.curriculum = None
        cfg.terminations.persistent_tilt = None
        cfg.terminations.operator_workspace = None
        if getattr(args, "terrain_diagnostics", None) is not None:
            # Observe the native objective without restoring a training-only
            # reset. The unavailable tilt impulse is explicitly not measured.
            cfg.rewards["physical_failure"].params = {
                "termination_names": (
                    "base_contact",
                    "course_chassis",
                    "course_fall",
                    "course_off_route",
                )
            }
        else:
            cfg.rewards = {}
        cfg.observations.policy.enable_corruption = False
        cfg.episode_length_s = 20.02
    else:
        cfg.recorders = RecorderManagerBaseCfg()
    runner_cfg.update(
        num_steps_per_env=24,
        max_iterations=args.iterations,
        experiment_name="go2_operator_terrain_teacher",
        run_name=terrain_training_version(critic_context),
    )
    return cfg, runner_cfg


def make_terrain_runner(env, runner_cfg, output, source_state, protocol):
    """Native RSL rollout/logging/checkpoint loop with explicit construction only."""
    import torch
    from rsl_rl.algorithms import PPO
    from rsl_rl.runners import OnPolicyRunner

    try:
        from .operator_retention import MovingRetentionUpdate
    except ImportError:
        from operator_retention import MovingRetentionUpdate

    version = protocol["version"]
    if version not in (TERRAIN_TRAINING_VERSION, TERRAIN_TASK_CRITIC_VERSION):
        raise ValueError("Unknown terrain-training protocol version")
    critic_context = version == TERRAIN_TASK_CRITIC_VERSION

    class TerrainRunner(OnPolicyRunner):
        def _construct_algorithm(self, obs):
            if self.is_distributed or self.alg_cfg.get("class_name") != "PPO":
                raise ValueError(
                    "Terrain training supports pinned single-device PPO only"
                )
            policy, reference = build_terrain_policy(
                obs, source_state, critic_context=critic_context
            )
            if runner_cfg["obs_groups"] != policy.obs_groups:
                raise ValueError(
                    "Terrain runner observation groups differ from the declared model"
                )
            if critic_context and not torch.equal(
                obs["critic_context"][:, :1], obs["operator_mask"]
            ):
                raise ValueError("Critic task role and retention role differ")
            with torch.no_grad():
                if not torch.equal(
                    policy.act_inference(obs), reference.act_inference(obs)
                ) or not torch.equal(policy.evaluate(obs), reference.evaluate(obs)):
                    raise ValueError(
                        "Initial terrain teacher action/value parity failed"
                    )
            algorithm_cfg = dict(self.alg_cfg)
            algorithm_cfg.pop("class_name")
            alg = PPO(policy, device=self.device, **algorithm_cfg)
            alg.init_storage("rl", env.num_envs, self.num_steps_per_env, obs, [12])
            self.retention_update = MovingRetentionUpdate(
                alg, reference_state=source_state, terrain_operator=True
            )
            alg.update = self.retention_update
            return alg

        def save(self, path, infos=None):
            updates = self.retention_update.updates
            if updates != self.current_learning_iteration:
                raise ValueError("Terrain checkpoint iteration/update mismatch")
            validate_teacher_adam(
                self.alg.policy.state_dict(),
                self.alg.optimizer.state_dict(),
                updates * 20,
            )
            curriculum = env.unwrapped.terrain_teacher_curriculum
            exposure = env.exposure.report()
            exposure.update(
                background_sampler_version=exposure["version"],
                version=f"{version}:operator_commands",
                sequence_sampler=sequence_manifest(),
                scope="Executed operator-lane commands only; v3 chains with v2 background sampling",
                workspace_censored_episodes=int(env.workspace_censored.item()),
            )
            metadata = {
                "version": version,
                "interface_version": TERRAIN_VERSION,
                "source_identity": protocol["source_identity"],
                "learning_updates": updates,
                "environment_transitions": updates
                * env.num_envs
                * self.num_steps_per_env,
                "readiness_only": False,
                "behavior_validated": False,
            }
            if critic_context:
                metadata["critic_context"] = copy.deepcopy(TERRAIN_CRITIC_CONTEXT)
            super().save(
                path,
                {
                    "terrain_training": metadata,
                    "curriculum": curriculum.state_dict(),
                    "operator_exposure": exposure,
                },
            )
            write_json(output / "command_exposure.json", exposure)
            write_json(
                output / "course_outcomes.json",
                {
                    "axis_order": [
                        "physical_column (family*10+variant)",
                        "attempted_level",
                        "outcome",
                    ],
                    "outcome_order": curriculum.outcome_names,
                    "counts": curriculum.outcomes.cpu().tolist(),
                    "scope": "Completed training episodes only; course lanes only; not heldout mastery",
                },
            )
            write_json(output / "training_progress.json", metadata)

    runner = TerrainRunner(
        env, copy.deepcopy(runner_cfg), str(output), runner_cfg["device"]
    )
    # Terrain update numbering is separate from the stock source's iteration8500.
    runner.current_learning_iteration = 1
    return runner


def validate_teacher_adam(state, optimizer, expected_steps):
    """Finite complete native Adam, with every parameter updated as declared."""
    import torch

    params = [p for group in optimizer["param_groups"] for p in group["params"]]
    states = optimizer["state"]
    if (
        len(params) != len(set(params))
        or len(params) != len(state)
        or set(states) != set(params)
        or any(not torch.isfinite(value).all() for value in state.values())
        or (state["std"] <= 0).any()
        or any(
            s["step"].item() != expected_steps
            or not torch.isfinite(s["exp_avg"]).all()
            or not torch.isfinite(s["exp_avg_sq"]).all()
            for s in states.values()
        )
    ):
        raise ValueError("Invalid finite teacher weights/Adam/update accounting")


def load_terrain_teacher(
    path, observations, source_state, identity, *, expected_version=None
):
    """Strict versioned loader; actor remains 312-D and only v2 has value context."""
    import torch

    data = torch.load(path, map_location="cpu", weights_only=True)
    metadata = data.get("infos", {}).get("terrain_training", {})
    updates = metadata.get("learning_updates")
    version = metadata.get("version")
    if (
        version not in (TERRAIN_TRAINING_VERSION, TERRAIN_TASK_CRITIC_VERSION)
        or (expected_version is not None and version != expected_version)
        or metadata.get("critic_context")
        != (TERRAIN_CRITIC_CONTEXT if version == TERRAIN_TASK_CRITIC_VERSION else None)
        or metadata.get("interface_version") != TERRAIN_VERSION
        or metadata.get("source_identity") != identity
        or metadata.get("readiness_only") is not False
        or type(updates) is not int
        or updates < 1
        or data.get("iter") != updates
    ):
        raise ValueError("Not a source-bound learned terrain-teacher checkpoint")
    policy, _ = build_terrain_policy(
        observations,
        source_state,
        critic_context=version == TERRAIN_TASK_CRITIC_VERSION,
    )
    policy.load_state_dict(data["model_state_dict"], strict=True)
    validate_teacher_adam(
        data["model_state_dict"], data["optimizer_state_dict"], updates * 20
    )
    return policy.eval(), metadata


def evaluate_terrain_teacher(raw, policy, output, *, diagnostic_baseline=None):
    """Replay unchanged physical scorers on the actual terrain-conditioned policy."""
    import numpy as np
    import torch
    from parkour_lab.tasks.manager_based.parkour_lab.parkour_lab_env_cfg import (
        ParkourLabEnvCfg,
    )

    try:
        from .operator_benchmark import (
            course_command,
            command_observation,
            validate_motor_trace,
        )
        from .operator_benchmark_core import (
            command_schedule,
            score_trace,
            score_course_trace,
        )
    except ImportError:
        from operator_benchmark import (
            course_command,
            command_observation,
            validate_motor_trace,
        )
        from operator_benchmark_core import (
            command_schedule,
            score_trace,
            score_course_trace,
        )

    labels, commands = command_schedule(4)
    schedule = torch.as_tensor(commands, device=raw.device)
    operator_ids = torch.arange(0, 160, 4, device=raw.device)
    initial = {
        "initial_position": raw.scene["robot"].data.root_pos_w.cpu().numpy().copy(),
        "initial_quaternion": raw.scene["robot"].data.root_quat_w.cpu().numpy().copy(),
        "env_origins": raw.scene.env_origins.cpu().numpy().copy(),
        "terrain_levels": raw.scene.terrain.terrain_levels.cpu().numpy().copy(),
        "terrain_columns": raw.scene.terrain.terrain_types.cpu().numpy().copy(),
    }
    if not np.array_equal(
        initial["terrain_levels"], np.tile([0, 1, 3, 6], 40)
    ) or not np.array_equal(initial["terrain_columns"], np.repeat([0, 10, 20, 30], 40)):
        raise ValueError("Unexpected teacher evaluation family/level assignment")
    capture = raw.operator_capture
    capture.enabled = capture.motor_parity = True
    capture.course = {"teacher_development": True}
    diagnostic = None
    if diagnostic_baseline is not None:
        try:
            from .operator_control_trace import OperatorControlTrace
        except ImportError:
            from operator_control_trace import OperatorControlTrace
        diagnostic = OperatorControlTrace(raw, native_rewards=True)
        capture.control_trace = diagnostic
    finished = torch.zeros(160, dtype=torch.bool, device=raw.device)
    scans = []
    with torch.inference_mode():
        for step in range(1000):
            desired = course_command(
                raw, {"course": {"target_speed": 0.55}}, step, finished
            )
            desired[operator_ids] = schedule[step]
            observation = command_observation(raw, desired)
            terrain = raw.observation_manager.compute_group("terrain")
            validate_terrain_scan(terrain, num_envs=160)
            obs = {"policy": observation, "terrain": terrain}
            capture.observation = observation.clone()
            scans.append(terrain.cpu().numpy().copy())
            action = policy.act_inference(obs)
            if not torch.isfinite(action).all():
                raise ValueError(f"Nonfinite learned teacher action at step {step}")
            if diagnostic is not None:
                diagnostic.before_step(observation, action)
            _, _, terminated, truncated, _ = raw.step(action)
            finished |= terminated | truncated
            if step % 250 == 0:
                print(f"Teacher development evaluation: {step}/1000", flush=True)
    capture.enabled = False
    trace = capture.finish()
    trace.update(initial, terrain_observation=np.stack(scans))
    if diagnostic is not None:
        control = diagnostic.finish()
        # Reuse the same artifact; don't duplicate delivered frames/actions.
        if not np.array_equal(
            control.pop("action"), trace["action"]
        ) or not np.array_equal(control.pop("observation_pre"), trace["observation"]):
            raise ValueError(
                "Native control capture is not paired with the delivered action"
            )
        trace.update(control)
    np.savez_compressed(output / "trace.npz", **trace)
    motor = validate_motor_trace(trace)

    def sliced(ids):
        return {
            k: value[ids] if k in initial else value[:, ids]
            for k, value in trace.items()
        }

    operator = score_trace(sliced(np.arange(0, 160, 4)), labels)
    native = ParkourLabEnvCfg()
    layout = native.events.reset_routes.params["terrain_layout"]
    cfg = native.parkour_curriculum
    results, metadata = {}, {}
    for family in range(4):
        for level in ((1,) if diagnostic is not None else (1, 3, 6)):
            ids = np.flatnonzero(
                (initial["terrain_columns"] == family * 10)
                & (initial["terrain_levels"] == level)
            )
            key = f"{cfg.families[family].name}_L{level}"
            course = {
                "family_index": family,
                "difficulty_index": level,
                "geometry_variant_index": 0,
                "course": cfg.course(family, level, 0).metadata(),
                "env_origins": initial["env_origins"][ids].tolist(),
                "contact_body_names": list(raw.scene["contact_forces"].body_names),
                "require_stable_finish": True,
            }
            metadata[key] = course
            results[key] = score_course_trace(sliced(ids), course)
    measurement = {
        "operator": operator,
        "courses": results,
        "motor": motor,
        "course_metadata": metadata,
        "family_by_column": list(layout.family_index_by_column),
        "scope": "Seed 43 teacher development; canonical variant 0; not held-out student/handoff/exit evidence",
    }
    status = "MEASURED"
    if diagnostic is not None:
        # This additional consumer is hashed in terrain_execution_identity.
        from scripts.analysis.operator_terrain_audit import summarize_native_capture

        measurement["native_control"] = summarize_native_capture(
            trace, diagnostic.metadata, diagnostic_baseline, measurement
        )
        if not measurement["native_control"]["reproduction"]["matched"]:
            status = "REPRODUCTION_DIVERGED"
    write_json(output / "measurement_report.json", measurement)
    return {
        "status": status,
        "operator_status": operator["status"],
        "course_statuses": {key: value["status"] for key, value in results.items()},
        "sha256": {
            name: file_sha256(output / name)
            for name in ("trace.npz", "measurement_report.json")
        },
        "promoted": False,
        "exit_gate_passed": False,
    }


def run_terrain_teacher(args, output, saved, agent, protocol):
    """Supervised native training or fixed-checkpoint evaluation; no new PPO loop."""
    app = env = None
    progress = {"stage": "source_validation"}
    try:
        diagnostic_run = getattr(args, "terrain_diagnostics", None)
        expected_identity = protocol.get(
            "diagnostic_consumer_identity", protocol["source_identity"]
        )
        if (
            terrain_execution_identity(args) != expected_identity
            or protocol["version"]
            != terrain_training_version(args.terrain_critic_context)
            or (
                diagnostic_run is None
                and file_sha256(args.readiness_report)
                != protocol["readiness"]["report_sha256"]
            )
        ):
            raise ValueError(
                "Terrain training source/prerequisite changed after preflight"
            )
        for package in ("isaaclab", "isaacsim", "rsl-rl-lib", "torch"):
            if importlib.metadata.version(package) != protocol["mesh_flat"][
                "packages"
            ].get(package):
                raise ValueError(f"Runtime {package} differs from verified physics")
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, livestream=0, device=args.device).app
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        try:
            from .operator_benchmark import mesh_identity
        except ImportError:
            from operator_benchmark import mesh_identity

        evaluation = args.terrain_evaluate_checkpoint is not None
        cfg, runner_cfg = terrain_teacher_configs(
            saved, agent, args, evaluation=evaluation
        )
        params = output / "params"
        params.mkdir(exist_ok=True)
        # Serialize before managers replace function classes with instances.
        (params / "env.yaml").write_text(yaml.dump(cfg.to_dict(), sort_keys=False))
        (params / "agent.yaml").write_text(yaml.dump(runner_cfg, sort_keys=False))
        baseline = None
        if diagnostic_run is not None:
            if not evaluation:
                raise ValueError("Native diagnostics cannot train")
            baseline = (
                diagnostic_run / "teacher_check" / args.terrain_evaluate_checkpoint.stem
            )
            validate_terrain_capture_config(output, baseline, diagnostic_run)
        interface = {
            "version": TERRAIN_VERSION,
            "training_version": protocol["version"],
            "inputs": ["policy:48", "terrain:264"],
            "critic_inputs": ["policy:48", "terrain:264"]
            + (["critic_context:4"] if args.terrain_critic_context else []),
            "critic_context": (
                TERRAIN_CRITIC_CONTEXT if args.terrain_critic_context else None
            ),
            "retention_metadata_not_policy_input": "operator_mask:1",
            "actions": "12 raw actions; default_joint_position + 0.25*action; no clipping",
            "source_identity": protocol["source_identity"],
            "scope": "Privileged teacher, not a causal student",
        }
        write_json(params / "interface.json", interface)
        progress["stage"] = "environment_setup"
        raw = ManagerBasedRLEnv(cfg=cfg)
        env = raw
        wrapped = RslRlVecEnvWrapper(raw, clip_actions=None)
        env = wrapped
        if (
            abs(raw.step_dt - DT) > 1e-9
            or tuple(raw.observation_manager.active_terms["policy"])
            != OBSERVATION_TERMS
            or raw.action_manager.total_action_dim != 12
            or list(raw.scene["robot"].joint_names)
            != protocol["mesh_flat"]["joint_names"]
        ):
            raise ValueError(
                "Terrain training changed the verified motor/observation contract"
            )
        geometry = mesh_identity(raw)
        obs = wrapped.get_observations()
        validate_terrain_scan(obs["terrain"], num_envs=raw.num_envs)
        source = load_reference_checkpoint(args.checkpoint, agent)["model_state_dict"]
        if evaluation:
            progress["stage"] = "teacher_evaluation"
            if (
                file_sha256(args.terrain_evaluate_checkpoint)
                != protocol["evaluation_checkpoint_sha256"]
            ):
                raise ValueError("Evaluation checkpoint changed after selection")
            policy, metadata = load_terrain_teacher(
                args.terrain_evaluate_checkpoint,
                obs,
                source,
                protocol["source_identity"],
                expected_version=protocol["version"],
            )
            result = evaluate_terrain_teacher(
                raw, policy, output, diagnostic_baseline=baseline
            )
            result.update(
                checkpoint_sha256=file_sha256(args.terrain_evaluate_checkpoint),
                learning_updates=metadata["learning_updates"],
            )
        else:
            progress["stage"] = "teacher_learning"
            env = OperatorExposureWrapper(wrapped, operator_only=True)
            if torch.count_nonzero(raw.scene.terrain.terrain_levels) or set(
                raw.scene.terrain.terrain_types.tolist()
            ) != set(range(40)):
                raise ValueError(
                    "Teacher acquisition must start all 40 production columns at L0"
                )
            runner = make_terrain_runner(env, runner_cfg, output, source, protocol)
            runner.learn(
                num_learning_iterations=args.iterations, init_at_random_ep_len=False
            )
            if runner.retention_update.updates != args.iterations:
                raise ValueError("Incomplete declared terrain-learning budget")
            result = {
                "status": "TRAINED",
                "learning_updates": args.iterations,
                "environment_transitions": args.num_envs * 24 * args.iterations,
                "checkpoints": {
                    str(i): file_sha256(output / f"model_{i}.pt")
                    for i in TERRAIN_CHECK_UPDATES
                },
                "promoted": False,
                "exit_gate_passed": False,
            }
        if terrain_execution_identity(args) != expected_identity:
            raise ValueError("Terrain training sources changed during execution")
        result.update(source_identity=protocol["source_identity"], mesh=geometry)
        if diagnostic_run is not None:
            result["diagnostic_consumer_identity"] = expected_identity
        write_json(output / "training_status.json", result)
    except Exception as error:
        report_terrain_readiness_error(output, error, progress)
    finally:
        for name, resource in (("environment", env), ("application", app)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    write_json(
                        output / f"{name}_cleanup_error.json", {"error": str(error)}
                    )


def run_training(args, output, agent, saved):
    resume = getattr(args, "retention_resume", None)
    zero_reference = getattr(args, "zero_command_reference", None)
    version = getattr(args, "curriculum", VERSION)
    reference = None
    if resume is not None:
        reference_path = args.resume_retention_reference
        reference = load_reference_checkpoint(
            reference_path, read_yaml_data(reference_path.parent / "params/agent.yaml")
        )
        if file_sha256(reference_path) != resume["reference_sha256"]:
            raise ValueError("Retention reference changed after preflight")
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, livestream=0, device=args.device)
    app = launcher.app
    env = None
    runner = None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        cfg, runner_cfg = training_configs(saved, agent, args)
        raw_env = ManagerBasedRLEnv(cfg=cfg)
        env = raw_env  # Keep ownership even if wrapper construction fails.
        if abs(raw_env.step_dt - DT) > 1e-9:
            raise ValueError("Reference refinement requires the 50-Hz action interface")
        if (
            tuple(raw_env.observation_manager.active_terms["policy"])
            != OBSERVATION_TERMS
        ):
            raise ValueError("Runtime observation order differs from the reference")
        if raw_env.action_manager.total_action_dim != 12:
            raise ValueError("Runtime action dimension differs from the reference")
        exposure_wrapper = (
            SequenceExposureWrapper
            if version == SEQUENCE_VERSION
            else OperatorExposureWrapper
        )
        env = exposure_wrapper(RslRlVecEnvWrapper(raw_env, clip_actions=None))
        params = output / "params"
        params.mkdir(exist_ok=True)
        (params / "env.yaml").write_text(yaml.dump(cfg.to_dict(), sort_keys=False))
        (params / "agent.yaml").write_text(yaml.dump(runner_cfg, sort_keys=False))
        source = load_reference_checkpoint(args.checkpoint, agent)
        if (
            resume is not None
            and file_sha256(args.checkpoint) != resume["checkpoint_sha256"]
        ):
            raise ValueError("Resume checkpoint changed after preflight")

        class RefinementRunner(OnPolicyRunner):
            def save(self, path, infos=None):
                exposure = env.exposure.report()
                super().save(path, {"handoff": handoff, "command_exposure": exposure})
                write_json(output / "command_exposure.json", exposure)

        runner = RefinementRunner(
            env, copy.deepcopy(runner_cfg), str(output), args.device
        )
        handoff = restore_reference(runner, source)
        retention = None
        if getattr(args, "moving_retention", False):
            # Freeze the exactly restored source once. Never re-anchor at an
            # intermediate checkpoint; keep a single uninterrupted Adam run.
            retention = install_moving_retention(
                runner.alg,
                reference_state=(
                    None if reference is None else reference["model_state_dict"]
                ),
                zero_command=zero_reference is not None,
            )
            if resume is not None:
                restore_adam_state(runner.alg, source, resume["adam_steps"])
                handoff["optimizer"] = (
                    "restored Adam moments, counters and options exactly"
                )
                handoff["restored"].append("optimizer_state")
                handoff["retention_resume"] = resume
        refinement = profile_manifest(
            saved, agent, getattr(args, "refinement_profile", "source")
        )
        if resume is not None:
            refinement["interpretation"] = (
                "Source reward/entropy profile preserved; Adam moments/counters/options "
                "restored exactly. Simulator and RNG restart. "
                + (
                    "Both retention terms preserved; randomized reversal/hold/restart exposure is the sole intervention. "
                    if version == SEQUENCE_VERSION
                    else (
                        "A separate zero-command retention loss is the new learning intervention. "
                        if zero_reference is not None
                        else "Retention objective preserved. "
                    )
                )
                + "Judge unchanged physical gates, not reward or action similarity alone."
            )
        handoff.update(
            source_checkpoint=str(args.checkpoint),
            source_sha256=file_sha256(args.checkpoint),
            additional_updates=args.iterations,
            final_iteration=source["iter"] + args.iterations,
            environment_transitions=args.num_envs
            * runner.num_steps_per_env
            * args.iterations,
            curriculum=training_curriculum_manifest(
                getattr(args, "curriculum", VERSION)
            ),
            refinement_profile=refinement,
            command_metric_warning=(
                "Stock error_vel_xy/error_vel_yaw accumulators divide by max command duration: "
                "12 s here versus 4 s in the original reference. Identical physical errors "
                "can log 3x smaller values. Compare unchanged physical benchmark errors instead."
            ),
            unchanged=[
                "network",
                "observations",
                "actions",
                "physics",
                "resets",
                "terminations",
                "reward_weights",
            ],
        )
        if not refinement["reward_parameters_changed"]:
            handoff["unchanged"].append("rewards")
        elif refinement["reward_functions_changed"]:
            handoff["unchanged"].append("rewards_except_versioned_velocity_tracking")
        else:
            handoff["unchanged"].append("reward_parameters_except_yaw_tracking_std")
        if retention is not None:
            handoff["moving_retention"] = retention_manifest(
                zero_command=zero_reference is not None
            )
            handoff["moving_retention"]["reference_sha256"] = (
                handoff["source_sha256"]
                if resume is None
                else resume["reference_sha256"]
            )
            handoff["moving_retention"]["reference_iteration"] = (
                source["iter"] if reference is None else reference["iter"]
            )
            handoff["moving_retention"]["check_offsets"] = getattr(
                args, "check_offsets", [100, 200]
            )
            handoff["moving_retention"][
                "check_timing"
            ] = f"after the uninterrupted {args.iterations}-update worker exits"
        if zero_reference is not None:
            handoff["zero_command_reference"] = zero_reference
        write_json(params / "operator_training.json", handoff)
        # The pre-update checkpoint includes the fresh OR exactly resumed Adam.
        # Never overwrite or renumber the source run's model files.
        torch.save(
            {
                "model_state_dict": runner.alg.policy.state_dict(),
                "optimizer_state_dict": runner.alg.optimizer.state_dict(),
                "iter": source["iter"],
                "infos": {"handoff": handoff, "pre_update": True},
            },
            output / f"model_{source['iter']}.pt",
        )
        print(
            f"Restored actor/critic/std exactly from iteration {source['iter']}; "
            f"training updates {handoff['first_update']}–{handoff['final_iteration']}, "
            f"{handoff['optimizer']}; fixed 1e-4, streaming OFF.",
            flush=True,
        )
        # Normal reset ages preserve long standing windows from the first rollout.
        print(handoff["command_metric_warning"], flush=True)
        print(
            f"Refinement profile: {refinement['source']['name']} → "
            f"{refinement['selected']['name']}; "
            f"yaw reward std={refinement['selected']['yaw_tracking_std']}, "
            f"entropy coefficient={refinement['selected']['entropy_coef']}, "
            f"stationary precision={refinement['selected']['stationary_precision']}. "
            "Acceptance thresholds unchanged.",
            flush=True,
        )
        if zero_reference is not None and version != SEQUENCE_VERSION:
            print(
                "New zero-command retention intervention: original frozen reference, "
                "separate moving/zero loss means; pure pivots excluded. "
                "Not an unchanged-objective continuation or a safety guarantee.",
                flush=True,
            )
        runner.learn(
            num_learning_iterations=args.iterations, init_at_random_ep_len=False
        )
        final = output / f"model_{handoff['final_iteration']}.pt"
        checked = load_reference_checkpoint(
            final, read_yaml_data(params / "agent.yaml")
        )
        if checked["iter"] != handoff["final_iteration"]:
            raise RuntimeError("Training did not reach the requested final checkpoint")
        if resume is not None:
            validate_adam_state(
                checked["optimizer_state_dict"],
                checked["model_state_dict"],
                resume["adam_steps"]
                + args.iterations
                * runner.alg.num_learning_epochs
                * runner.alg.num_mini_batches,
            )
        if (
            env.exposure.report()["environment_transitions"]
            != handoff["environment_transitions"]
        ):
            raise RuntimeError("Executed training budget differs from requested budget")
        if retention is not None and retention.updates != args.iterations:
            raise RuntimeError("Moving retention was not applied to every PPO update")
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        write_json(
            output / "training_status.json",
            {
                "status": "COMPLETED",
                "handoff": handoff,
                "final_checkpoint": str(final),
                "final_sha256": file_sha256(final),
                "behavior_validated": False,
            },
        )
    except Exception as error:
        write_json(
            output / "training_status.json",
            {
                "status": "ERROR",
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
    finally:
        # Kit can hard-exit Python with exit(0). Publish evidence before close;
        # only the independent parent owns success/failure exit semantics.
        for name, resource in (("environment", env), ("application", app)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    write_json(
                        output / f"{name}_cleanup_error.json",
                        {
                            "error": str(error),
                            "traceback": traceback.format_exc(),
                        },
                    )


def run_final_check(checkpoint, output, device, *, audit_student_interface=False):
    """Require both the benchmark exit code and its checkpoint-bound artifact."""
    result = {"status": "ERROR", "output_parent": str(output), "seed": 43}
    try:
        check = subprocess.run(
            [
                sys.executable,
                "-u",
                str(Path(__file__).with_name("operator_benchmark.py")),
                str(checkpoint),
                "--output-parent",
                str(output),
                "--device",
                device,
                *(["--audit-student-interface"] if audit_student_interface else []),
            ],
            check=False,
            timeout=600,
        )
        reports = list(output.glob("operator_screen_*/report.json"))
        if len(reports) != 1 or check.returncode not in (0, 1):
            raise ValueError("Benchmark failed or did not produce exactly one report")
        measured = json.loads(reports[0].read_text())
        if not isinstance(measured, dict):
            raise ValueError("Benchmark report must be an object")
        expected_status = "PASS" if check.returncode == 0 else "FAIL"
        if (
            measured.get("status") != expected_status
            or measured.get("seed") != 43
            or measured.get("provenance", {}).get("sha256", {}).get("checkpoint")
            != file_sha256(checkpoint)
        ):
            raise ValueError("Benchmark result/seed/checkpoint identity mismatch")
        result.update(status=expected_status, report=str(reports[0]))
        return result, check.returncode
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
        return result, 2


def retention_preflight(args, agent, saved):
    """Bind a bounded repair to replayed development evidence, before Kit starts."""
    try:
        from .operator_checkpoint_screen import replay_report
    except ImportError:
        from operator_checkpoint_screen import replay_report
    retention_check_offsets(args)
    if (
        args.curriculum not in ("operator_transitions_v2", SEQUENCE_VERSION)
        or args.refinement_profile != "source"
        or source_profile(saved, agent).name != "stationary_twist_v1"
        or args.skip_check
        or args.baseline_report is None
        or importlib.metadata.version("rsl-rl-lib") != "3.1.2"
    ):
        raise ValueError(
            "Moving retention requires source stationary_twist_v1, v2/v3 commands, RSL 3.1.2, a baseline report and checks enabled"
        )
    baseline = replay_report(args.baseline_report.resolve(strict=True), args.checkpoint)
    phases = baseline["phase_summary"]["phases"]
    if (
        baseline["passed"] < 90
        or any(p["sequence_failed"] for p in phases.values())
        or any(
            phases[name]["kinematic_passed"] != phases[name]["expected"]
            for name in ("forward", "restart")
        )
        or baseline["iteration"] % 50
    ):
        raise ValueError(
            "Repair source must have >=90/100 passes, complete forward/restart retention, no sequence violation, and a 50-aligned checkpoint"
        )
    return baseline


def retention_resume_preflight(args, agent, saved, baseline):
    """Permit one 200-update continuation of a completed initial retention run.

    Physics/rollout state is not checkpointed. This restores the learning state,
    not the exact random stream or trajectory of an uninterrupted 400-update run.
    """
    source = load_reference_checkpoint(args.checkpoint, agent)
    reference_path = args.resume_retention_reference.resolve(strict=True)
    reference = load_reference_checkpoint(
        reference_path, read_yaml_data(reference_path.parent / "params/agent.yaml")
    )
    reference_hash = file_sha256(reference_path)
    run = args.checkpoint.parent
    training_status = json.loads((run / "training_status.json").read_text())
    handoff = json.loads((run / "params/operator_training.json").read_text())
    provenance = json.loads((run / "source_provenance.json").read_text())
    loss = handoff.get("moving_retention", {})
    # Torch preserves tuples; JSON sidecars encode those same tuples as lists.
    checkpoint_handoff = json.loads(
        json.dumps(source.get("infos", {}).get("handoff"), allow_nan=False)
    )
    curriculum = json.loads(json.dumps(curriculum_manifest("operator_transitions_v2")))
    if (
        training_status.get("status") != "COMPLETED"
        or training_status.get("final_sha256") != baseline["sha256"]["checkpoint"]
        or training_status.get("handoff") != handoff
        or checkpoint_handoff != handoff
        or source["iter"] != handoff.get("final_iteration")
        or source["iter"] != reference["iter"] + 200
        or handoff.get("additional_updates") != 200
        or handoff.get("source_iteration") != reference["iter"]
        or handoff.get("first_update") != reference["iter"] + 1
        or handoff.get("source_sha256") != reference_hash
        or handoff.get("retention_resume") is not None
        or handoff.get("zero_command_reference") is not None
        or "zero_command_retention" in loss
        or handoff.get("curriculum") != curriculum
        or any(loss.get(k) != v for k, v in retention_manifest().items())
        or loss.get("reference_sha256") != reference_hash
        or loss.get("reference_iteration") != reference["iter"]
        or loss.get("check_offsets") != [100, 200]
        or provenance.get("sha256", {}).get("checkpoint") != reference_hash
        or baseline["packages"].get("rsl-rl-lib") != "3.1.2"
    ):
        raise ValueError(
            "Resume requires the completed initial 200-update moving_anchor_v1 run and its exact original reference"
        )
    if (
        args.seed != int(agent["seed"])
        or args.seed != int(saved["seed"])
        or args.num_envs != int(saved["scene"]["num_envs"])
        or int(agent["num_steps_per_env"]) != 24
        or int(agent["save_interval"]) != 50
        or agent["algorithm"]["schedule"] != "fixed"
        or float(agent["algorithm"]["learning_rate"]) != 1e-4
    ):
        raise ValueError(
            "Retention resume must preserve training seed, environment count and optimizer/rollout settings"
        )
    for name in (
        "operator_curriculum.py",
        "operator_command.py",
        "operator_profiles.py",
        "operator_rewards.py",
    ):
        if provenance["sha256"].get(name) != file_sha256(
            Path(__file__).with_name(name)
        ):
            raise ValueError(
                f"Learning implementation changed since retention training: {name}"
            )
    steps = (
        200
        * int(agent["algorithm"]["num_learning_epochs"])
        * int(agent["algorithm"]["num_mini_batches"])
    )
    validate_adam_state(
        source["optimizer_state_dict"], source["model_state_dict"], steps
    )
    return {
        "checkpoint_sha256": baseline["sha256"]["checkpoint"],
        "reference_checkpoint": str(reference_path),
        "reference_sha256": reference_hash,
        "reference_iteration": reference["iter"],
        "adam_steps": steps,
        "additional_updates": 200,
        "cumulative_retention_updates": 400,
        "optimizer": "exact saved moments, counters, parameter order and options",
        "not_restored": [
            "simulator state",
            "command ages",
            "random generator state",
            "rollout buffers",
        ],
        "scope": "One bounded learning-state continuation; environment restarts, not bitwise uninterrupted-run equivalence",
        "evidence_sha256": {
            name: file_sha256(run / name)
            for name in (
                "training_status.json",
                "params/operator_training.json",
                "source_provenance.json",
            )
        },
    }


def zero_command_retention_preflight(args, baseline, resume):
    """Bind the zero-twist term to verified holds of the original actor.

    The v2 alternative adds this term at the initial 200-update endpoint; the
    evidence-bound v3 continuation preserves it. Neither branch lets reference
    actions control the simulator.
    """
    try:
        from .operator_checkpoint_screen import replay_report
    except ImportError:
        from operator_checkpoint_screen import replay_report
    if resume is None:
        raise ValueError(
            "Zero-command retention requires the original-reference retention resume"
        )
    measured = replay_report(
        args.zero_command_reference_report.resolve(strict=True),
        args.resume_retention_reference.resolve(strict=True),
    )
    phases = measured["phase_summary"]["phases"]
    if (
        measured["sha256"]["checkpoint"] != resume["reference_sha256"]
        or measured["iteration"] != resume["reference_iteration"]
        or measured["packages"] != baseline["packages"]
        or any(p["sequence_failed"] for p in phases.values())
        or any(
            phases[name]["expected"] != expected
            or phases[name]["complete"] != expected
            or phases[name]["kinematic_passed"] != expected
            for name, expected in (("stand", 10), ("initial_stand", 90), ("stop", 90))
        )
    ):
        raise ValueError(
            "Zero-command reference must be the exact original actor with complete stand/stop passes, no sequence violation and matching runtime"
        )
    return {
        "reference_screen": measured,
        "intervention": retention_manifest(zero_command=True)["zero_command_retention"],
        "scope": (
            "existing zero-command loss retained from the completed zero-retention endpoint; only command-sequence exposure changes; simulation restarts and seed43 development selection are not independent confirmation"
            if getattr(args, "curriculum", VERSION) == SEQUENCE_VERSION
            else "paired alternative from the initial retention endpoint; same learning state, seed and budget, only the zero-command loss is added; simulation restarts and seed43 development selection are not independent confirmation"
        ),
    }


def retention_decision(measured, baseline):
    """Do not rescue a physical failure with an audit or a pooled score."""
    if measured["status"] == "PASS":
        return "DEVELOPMENT_PASS"
    phases = measured["phase_summary"]["phases"]
    if (
        measured["passed"] < baseline["passed"]
        or any(p["sequence_failed"] for p in phases.values())
        or any(
            phases[name]["kinematic_passed"] != phases[name]["expected"]
            for name in ("forward", "restart")
        )
    ):
        return "REGRESSED_CANDIDATE"
    return "NO_DEVELOPMENT_PASS"


def run_retention_checks(checkpoints, output, device, baseline):
    """Screen predeclared checkpoints after the worker has released the GPU."""
    try:
        from .operator_checkpoint_screen import replay_report
    except ImportError:
        from operator_checkpoint_screen import replay_report
    result = {
        "status": "ERROR",
        "promoted": False,
        "baseline": baseline,
        "candidates": [],
        "scope": "The entire predeclared training budget already completed; only evaluation stops early. Selection seed43, not held-out confirmation, convergence proof, RMA or hardware acceptance.",
    }
    try:
        output.mkdir(parents=True, exist_ok=True)
        for checkpoint in checkpoints:
            check, code = run_final_check(
                checkpoint,
                output / checkpoint.stem,
                device,
                audit_student_interface=True,
            )
            if code not in (0, 1):
                raise RuntimeError(f"Retention screen execution failed: {check}")
            measured = replay_report(Path(check["report"]), checkpoint)
            audit = measured["student_interface_audit"]
            if not isinstance(audit, dict) or any(
                audit.get(k) != v
                for k, v in {
                    "status": "ORACLE_PARITY_PASS",
                    "control_steps": 1000,
                    "action_comparisons": 100000,
                    "exact_action_equality": True,
                    "student_status": "UNTRAINED_NOT_RUN",
                }.items()
            ):
                raise ValueError("Missing or incomplete shadow oracle audit")
            if measured["packages"] != baseline["packages"]:
                raise ValueError("Runtime packages changed from the repair baseline")
            measured["repair_classification"] = retention_decision(measured, baseline)
            result["candidates"].append(measured)
            result["status"] = (
                "DEVELOPMENT_PASS"
                if measured["status"] == "PASS"
                else "NO_DEVELOPMENT_PASS"
            )
            write_json(output / "report.json", result)
            if result["status"] == "DEVELOPMENT_PASS":
                break
        return result, 0 if result["status"] == "DEVELOPMENT_PASS" else 1
    except Exception as error:
        result.update(status="ERROR", error=str(error), error_type=type(error).__name__)
        return result, 2


def validate_procedural_rollout(result, output, num_envs):
    """Replay the saved motor/target evidence before accepting a worker receipt."""
    import numpy as np
    from parkour_lab.learning.controller import VERSION as controller_version

    try:
        from .operator_benchmark import validate_motor_trace
        from .operator_student_bridge import CONTROLLER_ROLLOUT_COMMANDS
    except ImportError:
        from operator_benchmark import validate_motor_trace
        from operator_student_bridge import CONTROLLER_ROLLOUT_COMMANDS

    hashes = result["sha256"]
    if set(hashes) != {"trace.npz", "resolved_env.yaml"} or any(
        file_sha256(output / name) != digest for name, digest in hashes.items()
    ):
        raise ValueError("Rollout artifact hashes differ")
    controller = result["controller"]
    expected = {
        "status": "SIM_ADAPTER_PARITY_PASS",
        "control_steps": PROCEDURAL_ROLLOUT_STEPS,
        "action_comparisons": PROCEDURAL_ROLLOUT_STEPS * num_envs,
        "substep_target_comparisons": PROCEDURAL_ROLLOUT_STEPS * num_envs * 4 * 12,
        "forced_timeout_step": PROCEDURAL_ROLLOUT_STEPS // 2 - 1,
        "forced_timeout_count": num_envs // 2,
        "terrain_fixture_steps": PROCEDURAL_ROLLOUT_STEPS,
        "terrain_actor_access": False,
        "exit_allowed": False,
        "learning_updates": 0,
        "phase_steps": 40,
        "command_phases": [list(phase) for phase in CONTROLLER_ROLLOUT_COMMANDS],
        "pivot_env_ids": list(range(num_envs // 2)),
        "pivot_phase_steps": [40, 120],
        "exact_action_and_target_equality": True,
        "source_sha256": result["source_sha256"],
        "environment_transitions": PROCEDURAL_ROLLOUT_STEPS * num_envs,
    }
    if any(controller.get(key) != value for key, value in expected.items()):
        raise ValueError("Incomplete native controller evidence")
    for name, key in (
        ("controller_manifest", "interface_sha256"),
        ("motor_binding", "motor_binding_sha256"),
    ):
        digest = hashlib.sha256(
            json.dumps(controller[name], sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        if digest != controller[key]:
            raise ValueError("Native controller/motor identity differs")
    manifest = controller["controller_manifest"]
    if (
        manifest.get("version") != controller_version
        or manifest.get("name") != "stock_operator_oracle"
        or manifest.get("preprocessing_version") != "stock_48_unscaled_v1"
        or manifest.get("period_s") != DT
        or manifest.get("sensors", {}).get("oracle_base_lin_vel", {}).get("privileged")
        is not True
        or controller.get("policy_inputs")
        != "stock48 with simulator velocity; explicit oracle only"
        or manifest["artifact_sha256"] != result["source_sha256"]
        or manifest["joint_names"] != controller["motor_binding"]["joint_names"]
        or manifest["actuator_profile"]
        != "native_motor_sha256:" + controller["motor_binding_sha256"]
    ):
        raise ValueError("Controller identity is not bound to this checkpoint/motor")
    with np.load(output / "trace.npz", allow_pickle=False) as archive:
        trace = dict(archive)
    shape = (PROCEDURAL_ROLLOUT_STEPS, num_envs)
    if (
        trace["terminated"].shape != shape
        or trace["time_out"].shape != shape
        or trace["terminated"].dtype != np.bool_
        or trace["time_out"].dtype != np.bool_
        or trace["forced_timeout"].dtype != np.bool_
        or trace["adapter_target"].shape != (*shape, 12)
        or trace["joint_target_substeps"].shape != (*shape, 4, 12)
        or trace["terrain_observation"].shape != (*shape, 264)
        or any(not np.isfinite(value).all() for value in trace.values())
        or not np.isin(trace["terrain_observation"][:, :, 132:], (0, 1)).all()
        or not np.array_equal(trace["adapter_target"], trace["joint_target"])
        or not np.array_equal(
            trace["joint_target_substeps"],
            np.repeat(trace["adapter_target"][:, :, None], 4, axis=2),
        )
        or validate_motor_trace(trace) != controller["motor_interface"]
    ):
        raise ValueError("Native motor/adapter trace replay failed")
    forced = expected["forced_timeout_step"]
    injected = np.zeros(shape, dtype=np.bool_)
    injected[forced, : num_envs // 2] = True
    schedule = np.repeat(
        np.asarray(CONTROLLER_ROLLOUT_COMMANDS, dtype=np.float32), 40, axis=0
    )
    schedule = np.broadcast_to(schedule[:, None], (*shape, 3)).copy()
    schedule[40:120, : num_envs // 2, 0] = 0
    reset = trace["terminated"] | trace["time_out"]
    if (
        not trace["time_out"][forced, : num_envs // 2].all()
        or (trace["terminated"][forced] | trace["time_out"][forced]).all()
        or not np.array_equal(trace["forced_timeout"], injected)
        or not np.array_equal(trace["command"], schedule)
        or not np.array_equal(
            trace["decision_time_s"], np.arange(PROCEDURAL_ROLLOUT_STEPS) * DT
        )
        or controller["physical_failure_count"] != int(trace["terminated"].sum())
        or controller["other_timeout_count"]
        != int((trace["time_out"] & ~injected).sum())
        or controller["partial_reset_steps"]
        != int((reset.any(axis=1) & ~reset.all(axis=1)).sum())
    ):
        raise ValueError("Recorded commands, outcomes or partial auto-reset differ")


def procedural_config_main(args, parser):
    """Configuration or bounded adapter integration, never a learning launch."""
    rollout = args.procedural_rollout_check
    success_status = (
        "SIM_ADAPTER_PARITY_PASS" if rollout else "CONFIG_VALIDATED_NOT_SIMULATED"
    )
    report_filename = "measurement_report.json" if rollout else "config_report.json"
    if (
        args.terrain_train
        or args.terrain_readiness
        or args.terrain_critic_context
        or args.moving_retention
        or args.skip_check
        or args.curriculum != VERSION
        or args.refinement_profile != "source"
        or args.iterations != 300  # Reject an attempted learning budget in this mode.
        or args.seed != 42
        or args.num_envs < 20
        or args.num_envs % 20
        or (rollout and (args.num_envs > 80 or args.validate_only))
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or (args.validate_only and args.worker_output is not None)
        or any(
            getattr(args, name) is not None
            for name in (
                "terrain_evaluate_checkpoint",
                "mesh_flat_report",
                "readiness_report",
                "check_offsets",
                "baseline_report",
                "resume_retention_reference",
                "zero_command_reference_report",
                "reversal_stop_probe",
            )
        )
    ):
        parser.error(
            "Procedural checks require --num-envs a positive multiple of 20 "
            "and --seed 42; rollout allows 20-80 environments and no --validate-only. "
            "No learning, legacy terrain, refinement or resume flags"
        )
    try:
        checkpoint = args.checkpoint.resolve(strict=True)
        saved = read_yaml_data(checkpoint.parent / "params/env.yaml")
        agent = read_yaml_data(checkpoint.parent / "params/agent.yaml")
        load_reference_checkpoint(checkpoint, agent)
        select_profile(saved, agent, "source")
        if importlib.metadata.version("rsl-rl-lib") != "3.1.2":
            raise ValueError("This integration targets the pinned RSL-RL 3.1.2")
    except Exception as error:
        parser.error(f"Procedural source preflight failed: {error}")
    if args.validate_only:
        try:
            try:
                from .operator_student_bridge import controller_preflight
            except ImportError:
                from operator_student_bridge import controller_preflight
            print(
                json.dumps(controller_preflight(checkpoint), indent=2, allow_nan=False)
            )
        except Exception as error:
            parser.error(f"Controller boundary preflight failed: {error}")
        print(
            "Checkpoint architecture, controller adapter and reward profile validated. Native physical/"
            "algorithm configuration, physics, learning and behavior remain UNRUN; "
            "no files written."
        )
        return 0
    if args.worker_output is None:
        source_hash = file_sha256(checkpoint)
        # Config-only receipts stay temporary. Physics evidence must survive for
        # independent inspection, including a failed or interrupted rollout.
        with ExitStack() as cleanup:
            if rollout:
                args.output_parent.mkdir(parents=True, exist_ok=True)
                folder = tempfile.mkdtemp(
                    prefix="operator_procedural_rollout_", dir=args.output_parent
                )
                args.procedural_output = Path(folder)
                print(f"Adapter integration (no learning): {folder}", flush=True)
            else:
                folder = cleanup.enter_context(
                    tempfile.TemporaryDirectory(prefix="operator_config_check_")
                )
            result = supervise(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    str(checkpoint),
                    (
                        "--procedural-rollout-check"
                        if rollout
                        else "--procedural-config-check"
                    ),
                    "--num-envs",
                    str(args.num_envs),
                    "--device",
                    args.device,
                    "--worker-output",
                    folder,
                ],
                Path(folder),
                timeout_s=args.timeout,
                report_filename=report_filename,
                valid_statuses=(success_status, "ERROR"),
            )
        if result.get("status") == success_status and (
            not isinstance(result.get("packages"), dict)
            or result["packages"].get("isaaclab")
            not in PROCEDURAL_ISAACLAB_DISTRIBUTIONS
            or any(
                result.get(key) != value
                for key, value in {
                    "version": PROCEDURAL_TERRAIN_VERSION,
                    "source_sha256": source_hash,
                    "num_envs": args.num_envs,
                    "environment_transitions": (
                        PROCEDURAL_ROLLOUT_STEPS * args.num_envs if rollout else 0
                    ),
                    "learning_updates": 0,
                    "exit_allowed": False,
                }.items()
            )
        ):
            result = {
                "status": "ERROR",
                "error": "Invalid procedural receipt",
                "measurement_result": result,
            }
        if rollout and result.get("status") == success_status:
            try:
                validate_procedural_rollout(result, Path(folder), args.num_envs)
            except Exception as error:
                result = {
                    "status": "ERROR",
                    "error": str(error),
                    "measurement_result": result,
                }
        if file_sha256(checkpoint) != source_hash:
            result = {"status": "ERROR", "error": "Source checkpoint changed"}
        if rollout:
            result["exit_allowed"] = False
            write_json(Path(folder) / "report.json", result)
            print(f"Report: {Path(folder) / 'report.json'}", flush=True)
            summary = {
                key: result[key]
                for key in (
                    "status",
                    "error",
                    "environment_transitions",
                    "learning_updates",
                    "exit_allowed",
                )
                if key in result
            }
            summary["controller"] = {
                key: result.get("controller", {}).get(key)
                for key in (
                    "action_comparisons",
                    "substep_target_comparisons",
                    "physical_failure_count",
                    "forced_timeout_count",
                    "other_timeout_count",
                )
            }
            print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
        else:
            print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 0 if result["status"] == success_status else 2
    app = None
    env = None
    packages = {}
    try:
        source_hash = file_sha256(checkpoint)
        if rollout:
            write_run_provenance(args.worker_output, __file__)
        packages["isaaclab"] = importlib.metadata.version("isaaclab")
        if packages["isaaclab"] not in PROCEDURAL_ISAACLAB_DISTRIBUTIONS:
            raise ValueError(
                "Native configuration check requires an Isaac Lab release wheel "
                f"with distribution version in {PROCEDURAL_ISAACLAB_DISTRIBUTIONS}; "
                f"detected isaaclab=={packages['isaaclab']} using {sys.executable}"
            )
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, livestream=0, device=args.device).app
        from parkour_lab.tasks.manager_based.parkour_lab.mdp.terrain.operator_terrain import (
            terrain_envelope,
        )

        cfg, runner_cfg = procedural_terrain_configs(saved, agent, args)
        if rollout:
            from isaaclab.envs import ManagerBasedRLEnv

            try:
                from .operator_benchmark import make_recorder_cfg
                from .operator_benchmark_core import load_reference_actor
                from .operator_student_bridge import run_controller_rollout
            except ImportError:
                from operator_benchmark import make_recorder_cfg
                from operator_benchmark_core import load_reference_actor
                from operator_student_bridge import run_controller_rollout
            # Independent semantic samples can be compared to native observation
            # terms without drawing a second noise realization. Evaluation only.
            cfg.observations.policy.enable_corruption = False
            cfg.recorders = make_recorder_cfg()
        cfg.validate()
        result = {
            "status": success_status,
            "version": PROCEDURAL_TERRAIN_VERSION,
            "packages": packages,
            "source_sha256": source_hash,
            "num_envs": cfg.scene.num_envs,
            "acquisition_difficulty": list(PROCEDURAL_EASY_DIFFICULTY),
            "geometry": terrain_envelope(),
            "scan": PROCEDURAL_SCAN_INTERFACE,
            "policy_groups": runner_cfg["obs_groups"],
            "environment_transitions": 0,
            "learning_updates": 0,
            "exit_allowed": False,
        }
        if rollout:
            import numpy as np

            (args.worker_output / "resolved_env.yaml").write_text(
                yaml.dump(cfg.to_dict(), sort_keys=False)
            )
            actor, _ = load_reference_actor(checkpoint, agent)
            env = ManagerBasedRLEnv(cfg=cfg)
            env.reset(seed=args.seed)
            controller, trace = run_controller_rollout(
                env,
                actor.to(env.device),
                source_hash,
                is_running=app.is_running,
                steps=PROCEDURAL_ROLLOUT_STEPS,
            )
            np.savez_compressed(args.worker_output / "trace.npz", **trace)
            result.update(
                controller=controller,
                environment_transitions=PROCEDURAL_ROLLOUT_STEPS * args.num_envs,
                evaluation_changes=[
                    "observation corruption disabled",
                    "fixed body-twist schedule",
                    "one injected partial timeout",
                ],
                sha256={
                    name: file_sha256(args.worker_output / name)
                    for name in ("trace.npz", "resolved_env.yaml")
                },
            )
            if file_sha256(checkpoint) != source_hash:
                raise ValueError("Checkpoint changed during native rollout")
        write_json(
            args.worker_output / report_filename,
            result,
        )
        return 0
    except Exception as error:
        failure = {
            "status": "ERROR",
            "error": str(error),
            "error_type": type(error).__name__,
            "traceback": traceback.format_exc(),
            "packages": packages,
            "exit_allowed": False,
        }
        capture = getattr(env, "operator_capture", None)
        if capture is not None and capture.samples:
            try:
                import numpy as np

                np.savez_compressed(
                    args.worker_output / "trace.npz", **capture.finish()
                )
                failure["partial_control_steps"] = len(capture.samples)
                failure["trace_sha256"] = file_sha256(args.worker_output / "trace.npz")
            except Exception as capture_error:
                failure["partial_trace_error"] = str(capture_error)
        write_json(
            args.worker_output / report_filename,
            failure,
        )
        return 2
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                write_json(
                    args.worker_output / "environment_cleanup_error.json",
                    {"error": str(error)},
                )
        if app is not None:
            try:
                app.close()
            except Exception as error:
                write_json(
                    args.worker_output / "application_cleanup_error.json",
                    {"error": str(error)},
                )


def recurrent_training_identity(checkpoint):
    """Bind executable inputs only; docs and tests need not exist on the server."""
    root = Path(__file__).resolve().parents[2]
    sources = sorted((root / "scripts/rsl_rl").glob("*.py")) + sorted(
        (root / "source/parkour_lab/parkour_lab").rglob("*.py")
    )
    return {
        "physical_reference": {
            "checkpoint": file_sha256(checkpoint),
            **{
                name: file_sha256(checkpoint.parent / "params" / name)
                for name in ("agent.yaml", "env.yaml")
            },
        },
        "runtime": {str(p.relative_to(root)): file_sha256(p) for p in sources},
    }


def _canonical_runtime_config(value):
    """Compare data only; script and module names denote the same local binding."""
    value = yaml.load(yaml.dump(value), Loader=yaml.BaseLoader)

    def normalize(item):
        if isinstance(item, dict):
            return {key: normalize(value) for key, value in item.items()}
        if isinstance(item, list):
            return [normalize(value) for value in item]
        if isinstance(item, str) and item.startswith("scripts.rsl_rl."):
            return item.removeprefix("scripts.rsl_rl.")
        return item

    return normalize(value)


def recurrent_evaluation_files(checkpoint):
    return {
        "checkpoint": file_sha256(checkpoint),
        **{
            name: file_sha256(checkpoint.parent / name)
            for name in (
                "training_protocol.json",
                "params/env.yaml",
                "params/agent.yaml",
            )
        },
    }


def _requires_root_point_check(protocol):
    return protocol["version"] in PROPRIO_LINK_ORIGIN_VERSIONS or (
        protocol["version"] == PROPRIO_STOP_YAW_VERSION
        and (protocol.get("warm_start") or {}).get("version")
        == PROPRIO_STOP_YAW_VERSION
    )


def _check_root_point_receipt(receipt, num_envs):
    if not isinstance(receipt, dict) or any(
        receipt.get(key) != value
        for key, value in {
            "version": "isaaclab_2_3_2_root_reference_points_v1",
            "status": "PASS",
            "samples": 24 * num_envs,
            "checked_rollout_steps": 24,
            "sample_timing": "pre_action_first_rollout",
            "diagnostic_only": True,
            "root_position_reference": "LINK",
            "root_linear_velocity_reference": "COM",
            "body_com_offset_reference": "LINK_BODY",
            "absolute_tolerance_m": 2e-5,
            "absolute_tolerance_m_s": 2e-5,
            "minimum_rotational_leverage_m_s": 1e-5,
        }.items()
    ):
        raise ValueError("Missing or incomplete pre-update root-point audit")
    values = [
        receipt.get(key)
        for key in (
            "position_identity_abs_max_m",
            "position_identity_effective_tolerance_max_m",
            "velocity_identity_abs_max_m_s",
            "rotational_leverage_max_m_s",
        )
    ]
    bounds = [receipt.get(f"body_com_offset_{side}_m") for side in ("min", "max")]
    if any(
        type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values
    ) or any(
        not isinstance(bound, list)
        or len(bound) != 3
        or any(type(v) not in (float, int) or not math.isfinite(v) for v in bound)
        for bound in bounds
    ):
        raise ValueError("Invalid root-point residuals, leverage or COM offset bounds")
    position, tolerance, velocity, leverage = values
    # The fixed procedural workspace is only hundreds of metres across. This
    # permits float32 world-origin roundoff, not arbitrary declared tolerances.
    if not (
        position <= tolerance
        and 2e-5 <= tolerance <= 1e-3
        and velocity <= 2e-5
        and leverage > 1e-5
        and all(lo <= hi for lo, hi in zip(*bounds))
    ):
        raise ValueError("Failed pre-update root-point identity or rotational leverage")


def _validate_optimizer_stage(protocol, key, manifest_key, manifest):
    """Bind a single optional stage to its first immutable optimizer source."""
    change = protocol.get(key)
    if change is None:
        return
    if not isinstance(change, dict):
        raise ValueError(f"Invalid {key} stage")
    resumed = protocol.get("resume_from") or {}
    if not isinstance(resumed, dict) or not isinstance(resumed.get("sources"), dict):
        raise ValueError(f"{key} requires a bound optimizer source")
    start = change.get("started_at_learning_updates")
    digest = change.get("source_checkpoint_sha256")
    if (
        set(change)
        != {manifest_key, "started_at_learning_updates", "source_checkpoint_sha256"}
        or change[manifest_key] != manifest
        or protocol.get("version") != PROPRIO_HIGHER_TERRAIN_VERSION
        or all(
            protocol.get(name) is not None
            for name in ("restart_coverage_change", "pivot_planar_precision_change")
        )
        or type(start) is not int
        or type(resumed.get("learning_updates")) is not int
        or not 0 < start <= resumed.get("learning_updates", -1)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
        or (
            start == resumed.get("learning_updates")
            and digest != resumed.get("sources", {}).get("checkpoint")
        )
    ):
        raise ValueError(f"Invalid {key} stage or source binding")


def validate_restart_coverage_change(protocol):
    _validate_optimizer_stage(
        protocol, "restart_coverage_change", "sampling", restart_coverage_manifest()
    )


def validate_pivot_planar_precision_change(protocol):
    _validate_optimizer_stage(
        protocol,
        "pivot_planar_precision_change",
        "reward",
        PIVOT_PLANAR_PRECISION_CHANGE,
    )


def check_pivot_planar_precision_reward(term, precision_fraction):
    """Fail closed on every kernel/gate/reference parameter, not just the blend."""
    try:
        from .operator_rewards import track_lin_vel_xy_stationary
    except ImportError:
        from operator_rewards import track_lin_vel_xy_stationary
    change = PIVOT_PLANAR_PRECISION_CHANGE
    if (
        term.func is not track_lin_vel_xy_stationary
        or term.weight != change["weight"]
        or term.params
        != {
            "command_name": "base_velocity",
            "std": change["broad_std_m_s"],
            "stationary_std": change["planar_std_m_s"],
            "precision_fraction": precision_fraction,
            "pivot_only": True,
            "root_link_velocity": True,
        }
    ):
        raise ValueError(
            "Pivot planar precision requires the unchanged root-link pure-pivot reward"
        )


def apply_pivot_planar_precision(cfg):
    """After complete source reconstruction, change exactly one reward scalar."""
    term = cfg.rewards.track_lin_vel_xy_exp
    change = PIVOT_PLANAR_PRECISION_CHANGE
    check_pivot_planar_precision_reward(term, change["from_precision_fraction"])
    term.params["precision_fraction"] = change["to_precision_fraction"]


def apply_restart_coverage(cfg):
    """Change only the training command class, after source reconstruction."""
    try:
        from .operator_command import (
            ProceduralArrivalHoldCommand,
            ProceduralRestartCoverageCommand,
        )
    except ImportError:
        from operator_command import (
            ProceduralArrivalHoldCommand,
            ProceduralRestartCoverageCommand,
        )
    if cfg.commands.base_velocity.class_type is not ProceduralArrivalHoldCommand:
        raise ValueError("Restart coverage requires the original arrival/hold sampler")
    cfg.commands.base_velocity.class_type = ProceduralRestartCoverageCommand


def recurrent_evaluation_source(checkpoint, physical_identity):
    """Read an immutable learned snapshot without requiring its training to finish."""
    try:
        from .operator_student_bridge import load_recurrent_checkpoint
    except ImportError:
        from operator_student_bridge import load_recurrent_checkpoint

    files = recurrent_evaluation_files(checkpoint)
    policy, metadata, digest = load_recurrent_checkpoint(checkpoint, device="cpu")
    protocol = json.loads((checkpoint.parent / "training_protocol.json").read_text())
    validate_restart_coverage_change(protocol)
    validate_pivot_planar_precision_change(protocol)
    recipe = metadata["recipe"]
    exposure = protocol["version"] in PROPRIO_MIXED_TERRAIN_VERSIONS
    terrain = _proprio_terrain_exposure(protocol["version"])
    save_interval = protocol.get("save_interval", 50)
    archived_agent = read_yaml_data(checkpoint.parent / "params/agent.yaml")
    if (
        digest != files["checkpoint"]
        or type(save_interval) is not int
        or save_interval < 1
        or type(recipe.get("save_interval")) is not int
        or save_interval != recipe["save_interval"]
        or protocol["version"] not in PROPRIO_TRAINING_VERSIONS
        or protocol.get("reward_change")
        != (
            PROPRIO_REWARD_CHANGE
            if protocol["version"] != PROPRIO_LEGACY_ACQUISITION_VERSION
            else None
        )
        or protocol.get("posture_change")
        != _proprio_posture_change(protocol["version"])
        or protocol.get("stop_precision_change")
        != (
            PROPRIO_STOP_PRECISION_CHANGE
            if protocol["version"] == PROPRIO_STOP_PRECISION_VERSION
            else None
        )
        or protocol.get("warm_start") != metadata.get("warm_start")
        or protocol.get("resume_from") != metadata.get("resume_from")
        or protocol.get("restart_coverage_change")
        != metadata.get("restart_coverage_change")
        or protocol.get("pivot_planar_precision_change")
        != metadata.get("pivot_planar_precision_change")
        or protocol.get("terrain_exposure_change") != (terrain if exposure else None)
        or protocol.get("arrival_hold_change")
        != (
            arrival_hold_manifest()
            if protocol["version"] in PROPRIO_ARRIVAL_VERSIONS
            else None
        )
        or protocol.get("stance_change")
        != (
            PROPRIO_STANCE_CHANGE
            if protocol["version"] in PROPRIO_STANCE_VERSIONS
            else None
        )
        or protocol.get("pivot_precision_change")
        != (
            PROPRIO_PIVOT_PRECISION_CHANGE
            if protocol["version"] in PROPRIO_PIVOT_VERSIONS
            else None
        )
        or protocol.get("stationary_yaw_change")
        != (
            PROPRIO_STATIONARY_YAW_CHANGE
            if protocol["version"] in PROPRIO_STATIONARY_YAW_VERSIONS
            else None
        )
        or protocol.get("stop_yaw_change")
        != (
            PROPRIO_STOP_YAW_CHANGE
            if protocol["version"] in PROPRIO_STOP_YAW_VERSIONS
            else None
        )
        or protocol.get("link_origin_change")
        != (
            PROPRIO_LINK_ORIGIN_CHANGE
            if protocol["version"] in PROPRIO_LINK_ORIGIN_VERSIONS
            else None
        )
        or protocol["policy_version"] != metadata["policy_version"]
        or protocol["source_identity"]["physical_reference"] != physical_identity
        or protocol["terrain"] != "operator_procedural_surface_v2"
        or protocol["difficulty_range"]
        != (
            terrain["difficulty_range"]
            if exposure
            else list(PROCEDURAL_EASY_DIFFICULTY)
        )
        or protocol["adaptive_terrain_promotion"] is not False
        or protocol["gaps"] is not False
        or protocol["stage"]
        != ("fixed_mixed_terrain_exposure" if exposure else "fixed_easy_acquisition")
        or type(protocol["seed"]) is not int
        or protocol["seed"] < 0
        # Preserve historical training admissibility; new reservations are
        # enforced prospectively by the CLI, not retroactively on archives.
        or protocol["seed"] in (43, 44, 45)
        or type(protocol["num_envs"]) is not int
        or not 80 <= protocol["num_envs"] <= 5120
        or protocol["num_envs"] % 20
        or type(protocol["learning_updates"]) is not int
        or protocol["learning_updates"] < 1
        or protocol["seed"] != recipe["seed"]
        or protocol["learning_updates"] != recipe["max_iterations"]
        or not 1 <= metadata["learning_updates"] <= protocol["learning_updates"]
        or metadata["environment_transitions"]
        != metadata["learning_updates"] * 24 * protocol["num_envs"]
        or _canonical_runtime_config(recipe)
        != _canonical_runtime_config(archived_agent)
    ):
        raise ValueError(
            "Learned checkpoint, training archive or physical reference differs"
        )
    warm_start = protocol.get("warm_start")
    if _requires_root_point_check(protocol):
        _check_root_point_receipt(
            metadata.get("root_point_check"), protocol["num_envs"]
        )
    if (
        protocol["version"]
        in (
            PROPRIO_STOP_PRECISION_VERSION,
            *PROPRIO_MIXED_TERRAIN_VERSIONS,
        )
        and warm_start is None
    ):
        raise ValueError("Refinement requires a bound recurrent warm start")
    if warm_start is not None:
        if (
            protocol["version"] not in PROPRIO_UPRIGHT_VERSIONS
            or warm_start["version"]
            not in PROPRIO_WARM_START_SOURCES.get(protocol["version"], ())
            or type(warm_start["learning_updates"]) is not int
            or warm_start["learning_updates"] < 1
            or type(warm_start["optimizer_steps"]) is not int
            or warm_start["optimizer_steps"] != 20 * warm_start["learning_updates"]
            or set(warm_start["sources"]) != set(files)
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", v)
                for v in warm_start["sources"].values()
            )
            or not re.fullmatch(r"[0-9a-f]{64}", warm_start["actor_sha256"])
            or warm_start["joint_names"]
            != metadata["controller_manifest"]["joint_names"]
            or warm_start["default_position_rad"]
            != metadata["controller_manifest"]["configuration"]["default_position_rad"]
            or not math.isfinite(warm_start["learning_rate"])
            or not 0 < warm_start["learning_rate"] <= 0.01
            or warm_start["learning_rate"] != recipe["algorithm"]["learning_rate"]
        ):
            raise ValueError("Invalid recurrent warm-start lineage or learning rate")
    resumed = protocol.get("resume_from")
    if resumed is not None:
        from parkour_lab.learning.recurrent_operator import RECURRENT_RESUME_MODE

        start = resumed["learning_updates"]
        if (
            resumed["mode"] != RECURRENT_RESUME_MODE
            or resumed["version"] != protocol["version"]
            or type(start) is not int
            or not 0 < start < metadata["learning_updates"]
            or resumed["optimizer_steps"] != 20 * start
            or type(protocol["session_learning_updates"]) is not int
            or protocol["session_learning_updates"]
            != protocol["learning_updates"] - start
            or metadata["session_learning_updates"]
            != metadata["learning_updates"] - start
            or set(resumed["sources"]) != set(files)
            or any(
                not re.fullmatch(r"[0-9a-f]{64}", v)
                for v in resumed["sources"].values()
            )
            or not re.fullmatch(r"[0-9a-f]{64}", resumed["actor_sha256"])
            or resumed["joint_names"] != metadata["controller_manifest"]["joint_names"]
            or resumed["default_position_rad"]
            != metadata["controller_manifest"]["configuration"]["default_position_rad"]
            or not math.isfinite(resumed["learning_rate"])
            or not 0 < resumed["learning_rate"] <= 0.01
        ):
            raise ValueError("Invalid recurrent resume lineage or session budget")
    if recurrent_evaluation_files(checkpoint) != files:
        raise ValueError("Evaluation source changed during preflight")
    return policy, metadata, protocol, files


def recurrent_source_configs(
    saved, agent, args, training_protocol, metadata, checkpoint
):
    """Reconstruct the complete source recipe before evaluation or refinement."""
    original = copy.copy(args)
    original.num_envs = training_protocol["num_envs"]
    original.seed = training_protocol["seed"]
    original.iterations = training_protocol["learning_updates"]
    original.save_interval = metadata["recipe"]["save_interval"]
    original.device = metadata["recipe"]["device"]
    cfg, runner = proprioceptive_procedural_configs(
        saved, agent, original, acquisition_version=training_protocol["version"]
    )
    if training_protocol.get("restart_coverage_change") is not None:
        apply_restart_coverage(cfg)
    if training_protocol.get("pivot_planar_precision_change") is not None:
        apply_pivot_planar_precision(cfg)
    if training_protocol.get("warm_start") is not None:
        runner["algorithm"]["learning_rate"] = training_protocol["warm_start"][
            "learning_rate"
        ]
    archived = checkpoint.parent / "params"
    if _canonical_runtime_config(cfg.to_dict()) != _canonical_runtime_config(
        read_yaml_data(archived / "env.yaml")
    ) or _canonical_runtime_config(runner) != _canonical_runtime_config(
        read_yaml_data(archived / "agent.yaml")
    ):
        raise ValueError(
            "Reconstructed training configuration differs from learned archive"
        )
    return cfg, runner


def recurrent_evaluation_configs(saved, agent, args, training_protocol, metadata):
    """Validate the full training recipe first, then apply only evaluation overrides."""
    cfg, runner = recurrent_source_configs(
        saved,
        agent,
        args,
        training_protocol,
        metadata,
        args.procedural_evaluate_checkpoint,
    )
    try:
        from .operator_benchmark import make_recorder_cfg
        from .operator_student_bridge import recurrent_evaluation_protocol
    except ImportError:
        from operator_benchmark import make_recorder_cfg
        from operator_student_bridge import recurrent_evaluation_protocol
    tape = recurrent_evaluation_protocol(
        difficulty_range=getattr(args, "evaluation_difficulty", None),
        long_stops=getattr(args, "evaluation_long_stops", False),
        command_coverage=getattr(args, "evaluation_command_coverage", False),
        negative_pivot_first=getattr(args, "evaluation_negative_pivot_first", False),
        root_point_diagnostics=getattr(args, "evaluation_difficulty", None) is not None,
        reward_capture=getattr(args, "evaluation_reward_capture", False),
        out_and_back=getattr(args, "evaluation_out_and_back", False),
        seed=args.seed,
    )
    cfg.seed = cfg.scene.terrain.terrain_generator.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.observations.proprio.enable_corruption = False
    cfg.episode_length_s = tape["steps"] * tape["period_s"]
    if training_protocol["version"] in PROPRIO_ARRIVAL_VERSIONS:
        # Reconstruct/validate the training recipe above, but preserve the
        # existing evaluator's command-reset RNG and externally owned tape.
        try:
            from .operator_command import ProceduralTerrainCommand
        except ImportError:
            from operator_command import ProceduralTerrainCommand
        cfg.commands.base_velocity.class_type = ProceduralTerrainCommand
    # Only after the complete training archive has matched. Every checkpoint
    # uses the same one-row evaluation fixture, even after multi-row training.
    _configure_recurrent_terrain(
        cfg, getattr(args, "evaluation_difficulty", None) or PROCEDURAL_EASY_DIFFICULTY
    )
    cfg.recorders = make_recorder_cfg(procedural=True)
    runner.update(seed=args.seed, device=args.device, max_iterations=0, resume=False)
    return cfg, runner


def recurrent_training_main(args, parser):
    """Shared supervised acquisition/evaluation lifecycle; never implicit promotion."""
    evaluation = args.procedural_evaluate_checkpoint is not None
    refinement = args.procedural_refine_checkpoint is not None
    resuming = args.procedural_resume_checkpoint is not None
    learned_source = (
        args.procedural_evaluate_checkpoint
        or args.procedural_refine_checkpoint
        or args.procedural_resume_checkpoint
    )
    warm_start, resume_from, optimizer_state = None, None, None
    start_updates = 0
    if resuming:
        try:
            defaults = json.loads(
                (learned_source.parent / "training_protocol.json").read_text()
            )
            for name in ("num_envs", "seed", "save_interval"):
                value = defaults.get(name, 50 if name == "save_interval" else None)
                if getattr(args, name) is None:
                    setattr(args, name, value)
                elif name != "save_interval" and getattr(args, name) != value:
                    raise ValueError(f"Resume must preserve archived {name}={value}")
                if type(getattr(args, name)) is not int:
                    raise ValueError(f"Resume {name} must be an integer")
            if args.save_interval < 1:
                raise ValueError("Resume save interval must be positive")
        except Exception as error:
            parser.error(f"Invalid resume configuration: {error}")
    invalid_budget = (
        (
            (args.iterations, args.num_envs) != (0, 80)
            or type(args.seed) is not int
            or args.seed not in PROPRIO_EVALUATION_SEEDS
        )
        if evaluation
        else (
            args.iterations < 1
            or not 80 <= args.num_envs <= 5120
            or args.num_envs % 20
            or args.seed < 0
            or args.seed in PROPRIO_EVALUATION_SEEDS
        )
    )
    if (
        invalid_budget
        or (evaluation and args.timeout is None)
        or (
            args.timeout is not None
            and (not math.isfinite(args.timeout) or args.timeout <= 0)
        )
        or args.curriculum != VERSION
        or args.refinement_profile != "source"
        or args.skip_check
        or args.moving_retention
        or args.terrain_train
        or args.terrain_readiness
        or args.terrain_critic_context
        or any(
            getattr(args, name) is not None
            for name in (
                "terrain_evaluate_checkpoint",
                "mesh_flat_report",
                "readiness_report",
                "check_offsets",
                "baseline_report",
                "resume_retention_reference",
                "zero_command_reference_report",
                "reversal_stop_probe",
            )
        )
        or (args.validate_only and args.worker_output is not None)
    ):
        parser.error(
            "Procedural acquisition requires positive updates, 80–5120 environments "
            "in multiples of 20 and a training seed outside 43–48; evaluation requires "
            "exactly 0 updates, 80 environments and a seed in 43–48. An explicit timeout must "
            "be finite and positive; omit it for acquisition without a time limit. "
            "No legacy training, retention, skip-check or retention-resume flags."
        )
    try:
        from .operator_student_bridge import RECURRENT_OPERATOR_VERSION
    except ImportError:
        from operator_student_bridge import RECURRENT_OPERATOR_VERSION
    try:
        checkpoint = args.checkpoint.resolve(strict=True)
        agent = read_yaml_data(checkpoint.parent / "params/agent.yaml")
        saved = read_yaml_data(checkpoint.parent / "params/env.yaml")
        load_reference_checkpoint(checkpoint, agent)
        select_profile(saved, agent, "source")
        if importlib.metadata.version("rsl-rl-lib") != "3.1.2":
            raise ValueError("Recurrent acquisition requires RSL-RL 3.1.2")
        identity = recurrent_training_identity(checkpoint)
        if learned_source is not None:
            learned_source = learned_source.resolve(strict=True)
            if evaluation:
                args.procedural_evaluate_checkpoint = learned_source
            elif refinement:
                args.procedural_refine_checkpoint = learned_source
            else:
                args.procedural_resume_checkpoint = learned_source
            target = args.worker_output or args.output_parent
            if target.resolve().is_relative_to(learned_source.parent):
                raise ValueError(
                    "Evaluation/refinement/resume output must be outside the immutable training run"
                )
            policy, metadata, archived_protocol, evaluation_files = (
                recurrent_evaluation_source(
                    learned_source, identity["physical_reference"]
                )
            )
            if (args.restart_coverage or args.pivot_planar_precision) and (
                archived_protocol["version"] != PROPRIO_HIGHER_TERRAIN_VERSION
                or archived_protocol.get("restart_coverage_change") is not None
                or archived_protocol.get("pivot_planar_precision_change") is not None
            ):
                raise ValueError(
                    "An optimizer stage starts once from an unchanged higher_terrain "
                    "checkpoint; omit the stage flag to resume its archived settings"
                )
            if evaluation and args.seed == archived_protocol["seed"]:
                raise ValueError(
                    "Evaluation seed must differ from the archived training seed"
                )
            if (
                args.evaluation_difficulty is not None
                and archived_protocol["version"] not in PROPRIO_UPRIGHT_VERSIONS
            ):
                raise ValueError(
                    "Terrain probes require a v3 upright-posture checkpoint"
                )
            if refinement or resuming:
                import torch

                if refinement:
                    required_sources = PROPRIO_WARM_START_SOURCES[
                        PROPRIO_REFINEMENT_VERSIONS[args.procedural_refinement]
                    ]
                    if archived_protocol["version"] not in required_sources:
                        raise ValueError(
                            f"This refinement requires a recurrent source in {required_sources}"
                        )
                # Refinement carries only the live LR; resume also restores Adam below.
                # Resetting either to the recipe's initial LR can be a large jump.
                source = torch.load(
                    learned_source, map_location="cpu", weights_only=True
                )
                groups = source["optimizer_state_dict"]["param_groups"]
                adam = source["optimizer_state_dict"]["state"]
                parameter_ids = list(range(len(list(policy.parameters()))))
                source_steps = metadata["learning_updates"] * 20
                if (
                    len(groups) != 1
                    or groups[0]["params"] != parameter_ids
                    or tuple(groups[0]["betas"]) != (0.9, 0.999)
                    or groups[0]["eps"] != 1e-8
                    or groups[0]["weight_decay"] != 0
                    or groups[0]["amsgrad"] is not False
                    or groups[0].get("maximize", False) is not False
                    or groups[0].get("decoupled_weight_decay", False) is not False
                    or type(groups[0]["lr"]) is not float
                    or not math.isfinite(groups[0]["lr"])
                    or not 0 < groups[0]["lr"] <= 0.01
                    or set(adam) != set(parameter_ids)
                    or any(
                        state["step"].item() != source_steps for state in adam.values()
                    )
                    or not torch.isfinite(policy.log_std.exp()).all()
                    or torch.any(policy.log_std.exp() <= 0)
                ):
                    raise ValueError(
                        "Invalid source Adam groups, update count, learning rate or action std"
                    )
                binding = {
                    "sources": evaluation_files,
                    "version": archived_protocol["version"],
                    "learning_updates": metadata["learning_updates"],
                    "learning_rate": groups[0]["lr"],
                    "optimizer_steps": source_steps,
                    "actor_sha256": metadata["controller_manifest"]["artifact_sha256"],
                    "joint_names": metadata["controller_manifest"]["joint_names"],
                    "default_position_rad": metadata["controller_manifest"][
                        "configuration"
                    ]["default_position_rad"],
                }
                if resuming:
                    from parkour_lab.learning.recurrent_operator import (
                        RECURRENT_RESUME_MODE,
                        validate_recurrent_optimizer,
                    )

                    if (args.seed, args.num_envs) != (
                        archived_protocol["seed"],
                        archived_protocol["num_envs"],
                    ):
                        raise ValueError(
                            "Resume seed/environment count changed during preflight"
                        )
                    optimizer_state = source["optimizer_state_dict"]
                    validate_recurrent_optimizer(policy, optimizer_state, source_steps)
                    resume_from = {"mode": RECURRENT_RESUME_MODE, **binding}
                    start_updates = metadata["learning_updates"]
                    warm_start = archived_protocol.get("warm_start")
                else:
                    warm_start = binding
                if recurrent_evaluation_files(learned_source) != evaluation_files:
                    raise ValueError("Training source changed during preflight")
    except Exception as error:
        parser.error(f"Invalid physical reference or runtime: {error}")
    protocol = {
        "version": PROPRIO_ACQUISITION_VERSION,
        "policy_version": RECURRENT_OPERATOR_VERSION,
        "source_identity": identity,
        "initialization": "fresh actor, critic and Adam; physical reference weights NOT loaded",
        "seed": args.seed,
        "num_envs": args.num_envs,
        "learning_updates": args.iterations,
        "rollout_steps_per_update": 24,
        "simulated_seconds_per_environment": args.iterations * 24 * DT,
        "terrain": "operator_procedural_surface_v2",
        "difficulty_range": list(PROCEDURAL_EASY_DIFFICULTY),
        "adaptive_terrain_promotion": False,
        "gaps": False,
        "stage": "fixed_easy_acquisition",
        "ppo": {"learning_rate": 1e-3, "schedule": "adaptive", "entropy_coef": 0.01},
        "reward_profile": "stock broad tracking kernels; flat_orientation_l2=-2.5, feet_air_time=0.01, dof_pos_limits=-10; no stationary precision or action-retention loss",
        "reward_change": copy.deepcopy(PROPRIO_REWARD_CHANGE),
        "posture_change": copy.deepcopy(PROPRIO_POSTURE_CHANGE),
        "initial_action_std": 0.5,
        "metrics": "per-profile command tracking, measured moving/nonflat exposure, physical failures and timeouts; training data, not held-out success rates",
        "save_interval": args.save_interval,
        "checkpoint_selection": f"save every {args.save_interval} completed updates and final; no automatic selection or promotion",
        "scope": "Acquire a causal gait before progression; short budgets are integration only. No terrain exit acceptance or deployment claim.",
        "exit_allowed": False,
    }
    if refinement:
        version = PROPRIO_REFINEMENT_VERSIONS[args.procedural_refinement]
        precision = version == PROPRIO_STOP_PRECISION_VERSION
        arrival = version in PROPRIO_ARRIVAL_VERSIONS
        exposure = version in PROPRIO_MIXED_TERRAIN_VERSIONS
        protocol.update(
            version=version,
            warm_start=warm_start,
            initialization="exact actor, critic and action std from recurrent checkpoint; retain scalar LR; fresh Adam moments, recurrent/episode state and local update counters; NOT uninterrupted resume",
            initial_action_std="copied exactly from source checkpoint",
            reward_profile=(
                "full-zero precision blend; other v3 rewards unchanged"
                if precision
                else protocol["reward_profile"]
            ),
            scope="Matched warm-start refinement: compare stock and stop_precision at equal additional transitions/seed; changed rewards are not comparable returns; no automatic selection or acceptance",
        )
        protocol["ppo"]["learning_rate"] = warm_start["learning_rate"]
        if precision:
            protocol["stop_precision_change"] = copy.deepcopy(
                PROPRIO_STOP_PRECISION_CHANGE
            )
        if exposure:
            terrain = _proprio_terrain_exposure(version)
            protocol.update(
                stage="fixed_mixed_terrain_exposure",
                difficulty_range=list(terrain["difficulty_range"]),
                terrain_exposure_change=copy.deepcopy(terrain),
                posture_change=_proprio_posture_change(
                    PROPRIO_TERRAIN_EXPOSURE_VERSION
                ),
                scope="Terrain-exposure experiment from a bound v3 checkpoint; compare against equal-budget stock refinement and the source on the same easy/stress screens; no automatic selection or acceptance",
            )
        if arrival:
            protocol.update(
                arrival_hold_change=arrival_hold_manifest(),
                scope="Arrival/long-hold/restart candidate from a bound terrain-exposure checkpoint; fresh Adam is NOT a matched continuation. Compare with the frozen source; no sampler-only causal claim, automatic selection or acceptance",
            )
        if version in PROPRIO_STANCE_VERSIONS:
            protocol.update(
                stance_change=copy.deepcopy(PROPRIO_STANCE_CHANGE),
                reward_profile=protocol["reward_profile"]
                + "; exact-zero joint-posture L2 norm cost, weight=-0.1",
                scope="Single stance-cost candidate with the unchanged arrival sampler. Compare equal-update arrival_hold and stance restarts from the same exposure checkpoint, seed and scalar LR; no automatic selection or acceptance",
            )
        if warm_start["version"] == PROPRIO_STANCE_VERSION:
            protocol["scope"] = (
                "Matched stance versus pivot_precision restarts from the same stance checkpoint, seed, saved scalar LR and additional transitions. Fresh Adam in both arms; frozen source is a regression reference, not the matched control. Predeclare one endpoint; require joint yaw/translation/stop/physical-health retention, not improved return alone. Development only; no automatic selection or acceptance."
            )
        if version in PROPRIO_PIVOT_VERSIONS:
            protocol["pivot_precision_change"] = copy.deepcopy(
                PROPRIO_PIVOT_PRECISION_CHANGE
            )
        if version == PROPRIO_PIVOT_PRECISION_VERSION:
            protocol[
                "reward_profile"
            ] += "; pure-pivot planar precision blend, fraction=0.1, std=0.05 m/s; yaw reward unchanged"
        if version in PROPRIO_STATIONARY_YAW_VERSIONS:
            protocol.update(
                stationary_yaw_change=copy.deepcopy(PROPRIO_STATIONARY_YAW_CHANGE),
                reward_profile="stance cost and pure-pivot planar precision retained; stop/pure-pivot angular precision fraction=0.1, std=0.1 rad/s; translating kernels unchanged; flat_orientation_l2=-2.5, feet_air_time=0.01, dof_pos_limits=-10; no action-retention loss",
                scope="One angular-reward delta versus pivot_precision from the same stance checkpoint, seed, saved scalar LR and additional transitions; fresh Adam in both. An archived comparator is valid only after configuration/source compatibility checks. Sequential single-seed development ablation, not independent replication or held-out acceptance. Predeclare one endpoint; require joint stop/yaw/translation/physical-health retention; retain the frozen stance source if neither improves. No automatic selection or extension.",
            )
        if version in PROPRIO_STOP_YAW_VERSIONS:
            protocol.update(
                stop_yaw_change=copy.deepcopy(PROPRIO_STOP_YAW_CHANGE),
                reward_profile="stationary_yaw recipe with only full-stop fine angular std changed from 0.1 to 0.05 rad/s; fraction=0.1 and broad std=0.5 retained; pivot/translation/stance rewards unchanged",
                scope="One full-stop angular width delta versus stationary_yaw from the same stance checkpoint, seed, saved scalar LR and additional transitions; fresh Adam in both. Reuse an archived equal-budget comparator only after source/configuration checks. Require all stops, easy and flat-command retention, no lost pivot passes or moving/physical regressions. Sequential single-seed development only; no automatic extension, checkpoint selection or acceptance.",
            )
        if version in PROPRIO_LINK_ORIGIN_VERSIONS:
            protocol["link_origin_change"] = copy.deepcopy(PROPRIO_LINK_ORIGIN_CHANGE)
            protocol[
                "reward_profile"
            ] += "; all planar tracking uses root-link origin velocity, not root-body COM velocity"
        if warm_start["version"] == PROPRIO_STOP_YAW_VERSION:
            protocol["scope"] = (
                "Matched stop_yaw versus link_origin restarts from the same stop_yaw checkpoint, seed, saved scalar LR and additional transitions; fresh Adam in both. Verify native point identities during the first rollout before any optimizer update. Compare the predeclared endpoint using unchanged canonical command/position gates and additive link diagnostics. Historical velocity scores remain root-body COM based. No automatic extension, selection or acceptance."
            )
        if warm_start["version"] == PROPRIO_LINK_ORIGIN_VERSION:
            protocol["scope"] = (
                "Matched link_origin versus higher_terrain restarts from the same link-origin checkpoint, seed, live scalar LR and additional transitions; fresh Adam in both. Only static terrain exposure changes: three rows through0.35 versus five through0.55, retaining lower bands. Commands, rewards, motor, actor and gates unchanged. Compare the predeclared endpoint on easy, prior-hard and higher bands; retain all source failures. No adaptive promotion, automatic extension, checkpoint selection or acceptance."
            )
    if resuming:
        protocol = copy.deepcopy(archived_protocol)
        protocol.update(
            source_identity=identity,
            resume_from=resume_from,
            initialization="exact saved model/std, Adam moments/steps and live adaptive LR; fresh simulator/command/GRU/RNG state; NOT bitwise uninterrupted training",
            initial_action_std="copied exactly from source checkpoint",
            learning_updates=start_updates + args.iterations,
            session_learning_updates=args.iterations,
            simulated_seconds_per_environment=(start_updates + args.iterations)
            * 24
            * DT,
            save_interval=args.save_interval,
            checkpoint_selection=f"save every {args.save_interval} cumulative completed updates and final; no automatic selection or promotion",
            metrics="current session only; cumulative learning/control/transition counters are reported separately from session exposure and resets",
            scope="Optimizer continuation of the unchanged archived recipe in a separate immutable-source run; --iterations adds updates. No behavioral acceptance or automatic extension decision.",
        )
        if args.restart_coverage:
            protocol["restart_coverage_change"] = {
                "sampling": restart_coverage_manifest(),
                "started_at_learning_updates": start_updates,
                "source_checkpoint_sha256": resume_from["sources"]["checkpoint"],
            }
            protocol["scope"] = (
                "Explicit training command-distribution stage: enrich low-speed "
                "post-hold restarts, retain full speed support and all other sampling. "
                "Preserve model/std, Adam/live LR and cumulative counters; simulator, "
                "command and GRU state reset. Terrain/rewards/motor/policy/gates unchanged. "
                "Candidate-only triage, not sampler-only causal attribution. No automatic "
                "extension, checkpoint selection or acceptance."
            )
        if args.pivot_planar_precision:
            protocol["pivot_planar_precision_change"] = {
                "reward": copy.deepcopy(PIVOT_PLANAR_PRECISION_CHANGE),
                "started_at_learning_updates": start_updates,
                "source_checkpoint_sha256": resume_from["sources"]["checkpoint"],
            }
            protocol["scope"] = (
                "Explicit pure-pivot planar reward stage: precision fraction 0.1 to 0.3; "
                "same broad/fine widths, weight and root-link reference. Preserve model/std, "
                "Adam/live LR and cumulative counters; simulator, command and GRU reset. "
                "Stop/translating rewards, yaw/stance objectives, sampler, terrain, motor, "
                "policy and gates unchanged. Compare one predeclared endpoint with an "
                "equal-budget unchanged resume from the same source/seed; frozen-source "
                "scores are regression references, not a matched control. Development only; "
                "no automatic extension, checkpoint selection or acceptance."
            )
        validate_restart_coverage_change(protocol)
        validate_pivot_planar_precision_change(protocol)
    final_updates = start_updates + args.iterations
    if evaluation:
        try:
            from .operator_student_bridge import recurrent_evaluation_protocol
        except ImportError:
            from operator_student_bridge import recurrent_evaluation_protocol
        evaluation_tape = recurrent_evaluation_protocol(
            difficulty_range=args.evaluation_difficulty,
            long_stops=args.evaluation_long_stops,
            command_coverage=args.evaluation_command_coverage,
            negative_pivot_first=args.evaluation_negative_pivot_first,
            root_point_diagnostics=args.evaluation_difficulty is not None,
            reward_capture=getattr(args, "evaluation_reward_capture", False),
            out_and_back=getattr(args, "evaluation_out_and_back", False),
            seed=args.seed,
        )
        protocol = {
            **evaluation_tape,
            "source_identity": identity,
            "training_producer_identity": archived_protocol["source_identity"],
            "evaluation_sources": evaluation_files,
            "checkpoint_learning_updates": metadata["learning_updates"],
            "policy_version": RECURRENT_OPERATOR_VERSION,
            "learning_updates": 0,
            "configuration_check": "reconstruct and compare full archived training config before evaluation-only overrides in native worker",
            "exit_allowed": False,
        }
    protocol_name = (
        "evaluation_protocol.json" if evaluation else "training_protocol.json"
    )
    receipt_name = "measurement_report.json" if evaluation else "training_status.json"
    success = (
        "DEVELOPMENT_EVALUATED_NOT_ACCEPTED"
        if evaluation
        else "ACQUISITION_COMPLETE_NOT_ACCEPTED"
    )
    artifact_names = {
        "params/env.yaml",
        "params/agent.yaml",
        "trace.npz" if evaluation else f"model_{final_updates}.pt",
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "SOURCE_VALIDATED_NOT_SIMULATED",
                    "protocol": protocol,
                    **(
                        {"native_configuration_comparison": "UNRUN"}
                        if learned_source is not None
                        else {}
                    ),
                },
                indent=2,
            )
        )
        return 0
    if args.worker_output is None:
        args.output_parent.mkdir(parents=True, exist_ok=True)
        if evaluation:
            name, label = "screen", "Frozen recurrent evaluation"
        elif args.restart_coverage:
            name, label = "restart_coverage", "Restart-coverage stage"
        elif args.pivot_planar_precision:
            name, label = "pivot_planar_precision", "Pivot-planar-precision stage"
        elif resuming:
            name, label = "resume", "Recurrent resume"
        elif refinement:
            name, label = args.procedural_refinement, "Recurrent refinement"
        else:
            name, label = "", "Fresh recurrent acquisition"
        output = Path(
            tempfile.mkdtemp(
                prefix=f"operator_proprio_{name + '_' if name else ''}",
                dir=args.output_parent,
            )
        ).resolve()
        args.procedural_output = output
        write_run_provenance(output, __file__)
        write_json(output / protocol_name, protocol)
        print(f"{label}: {output}", flush=True)
        result = supervise(
            [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                str(checkpoint),
                *(
                    [
                        "--procedural-evaluate-checkpoint",
                        str(args.procedural_evaluate_checkpoint),
                        *(
                            [
                                "--evaluation-difficulty",
                                *map(str, args.evaluation_difficulty),
                            ]
                            if args.evaluation_difficulty is not None
                            else []
                        ),
                        *(
                            ["--evaluation-long-stops"]
                            if args.evaluation_long_stops
                            else []
                        ),
                        *(
                            ["--evaluation-command-coverage"]
                            if args.evaluation_command_coverage
                            else []
                        ),
                        *(
                            ["--evaluation-negative-pivot-first"]
                            if args.evaluation_negative_pivot_first
                            else []
                        ),
                        *(
                            ["--evaluation-reward-capture"]
                            if getattr(args, "evaluation_reward_capture", False)
                            else []
                        ),
                        *(
                            ["--evaluation-out-and-back"]
                            if getattr(args, "evaluation_out_and_back", False)
                            else []
                        ),
                    ]
                    if evaluation
                    else [
                        *(
                            [
                                "--procedural-refine-checkpoint",
                                str(learned_source),
                                "--procedural-refinement",
                                args.procedural_refinement,
                            ]
                            if refinement
                            else (
                                ["--procedural-resume-checkpoint", str(learned_source)]
                                if resuming
                                else ["--procedural-train"]
                            )
                        ),
                        "--save-interval",
                        str(args.save_interval),
                        *(["--restart-coverage"] if args.restart_coverage else []),
                        *(
                            ["--pivot-planar-precision"]
                            if args.pivot_planar_precision
                            else []
                        ),
                    ]
                ),
                "--iterations",
                str(args.iterations),
                "--num-envs",
                str(args.num_envs),
                "--seed",
                str(args.seed),
                "--device",
                args.device,
                "--worker-output",
                str(output),
            ],
            output,
            timeout_s=args.timeout,
            report_filename=receipt_name,
            valid_statuses=(success, "ERROR"),
        )
        try:
            if identity != recurrent_training_identity(checkpoint):
                raise ValueError(
                    "Physical reference or runtime changed during training"
                )
            if (
                learned_source is not None
                and recurrent_evaluation_files(learned_source) != evaluation_files
            ):
                raise ValueError("Frozen evaluation source changed during execution")
            if result.get("status") == success:
                if not evaluation and _requires_root_point_check(protocol):
                    _check_root_point_receipt(
                        result.get("root_point_check"), args.num_envs
                    )
                expected = {
                    "policy_version": RECURRENT_OPERATOR_VERSION,
                    "learning_updates": final_updates,
                    "environment_transitions": (
                        evaluation_tape["steps"] * evaluation_tape["num_envs"]
                        if evaluation
                        else final_updates * 24 * args.num_envs
                    ),
                    "protocol_sha256": file_sha256(output / protocol_name),
                    "exit_allowed": False,
                }
                if evaluation:
                    expected.update(
                        protocol=evaluation_tape,
                        control_steps=evaluation_tape["steps"],
                        checkpoint_sha256=evaluation_files["checkpoint"],
                        checkpoint_learning_updates=metadata["learning_updates"],
                        evaluation_sources=evaluation_files,
                    )
                    if len(result.get("trials", [])) != 80:
                        raise ValueError(
                            "Require all 80 first-attempt evaluation trials"
                        )
                elif resuming:
                    expected.update(
                        resume_from=resume_from,
                        control_steps=final_updates * 24,
                        optimizer_steps=final_updates * 20,
                        session_learning_updates=args.iterations,
                        session_control_steps=args.iterations * 24,
                        session_environment_transitions=args.iterations
                        * 24
                        * args.num_envs,
                    )
                if any(result.get(k) != v for k, v in expected.items()):
                    raise ValueError("Incomplete recurrent training receipt")
                artifacts = result["sha256"]
                if set(artifacts) != artifact_names or any(
                    file_sha256(output / name) != digest
                    for name, digest in artifacts.items()
                ):
                    raise ValueError("Recurrent checkpoint/configuration hashes differ")
                if evaluation:
                    import numpy as np

                    try:
                        from .operator_student_bridge import (
                            summarize_recurrent_evaluation,
                        )
                    except ImportError:
                        from operator_student_bridge import (
                            summarize_recurrent_evaluation,
                        )
                    with np.load(output / "trace.npz", allow_pickle=False) as archive:
                        summary = summarize_recurrent_evaluation(
                            dict(archive), evaluation_tape
                        )
                    if any(result.get(key) != value for key, value in summary.items()):
                        raise ValueError(
                            "Evaluation measurements differ from raw first-attempt trace"
                        )
                if not evaluation:
                    # A matching file hash alone does not make a usable checkpoint.
                    import torch

                    learned = torch.load(
                        output / f"model_{final_updates}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    info = learned["infos"]
                    metadata = info["recurrent_training"]
                    if (
                        _requires_root_point_check(protocol)
                        and metadata.get("root_point_check")
                        != result["root_point_check"]
                    ):
                        raise ValueError(
                            "Checkpoint root-point audit differs from receipt"
                        )
                    weights = learned["model_state_dict"]
                    adam = learned["optimizer_state_dict"]["state"]
                    if (
                        learned["iter"] != final_updates - 1
                        or info["learning_updates"] != final_updates
                        or metadata.get("warm_start") != warm_start
                        or result.get("warm_start") != warm_start
                        or metadata.get("resume_from") != resume_from
                        or metadata.get("restart_coverage_change")
                        != protocol.get("restart_coverage_change")
                        or result.get("restart_coverage_change")
                        != protocol.get("restart_coverage_change")
                        or metadata.get("pivot_planar_precision_change")
                        != protocol.get("pivot_planar_precision_change")
                        or result.get("pivot_planar_precision_change")
                        != protocol.get("pivot_planar_precision_change")
                        or (
                            resuming
                            and any(
                                metadata.get(key) != expected[key]
                                for key in (
                                    "control_steps",
                                    "session_learning_updates",
                                    "session_control_steps",
                                    "session_environment_transitions",
                                )
                            )
                        )
                        or any(
                            metadata.get(k) != expected[k]
                            for k in (
                                "policy_version",
                                "learning_updates",
                                "environment_transitions",
                            )
                        )
                        or not weights
                        or any(not torch.isfinite(v).all() for v in weights.values())
                        or not adam
                        or any(
                            state["step"].item() != 20 * final_updates
                            or not torch.isfinite(state["exp_avg"]).all()
                            or not torch.isfinite(state["exp_avg_sq"]).all()
                            for state in adam.values()
                        )
                    ):
                        raise ValueError(
                            "Invalid saved recurrent weights, Adam state or update counts"
                        )
        except Exception as error:
            result = {
                "status": "ERROR",
                "error": str(error),
                "measurement_result": result,
            }
        result["exit_allowed"] = False
        write_json(output / "report.json", result)
        print(f"{result['status']}: {output / 'report.json'}", flush=True)
        if result["status"] == "ERROR":
            print(json.dumps(result, indent=2), flush=True)
        return 0 if result["status"] == success else 2

    output, app, env = args.worker_output, None, None
    try:
        if json.loads((output / protocol_name).read_text()) != protocol:
            raise ValueError("Training inputs differ from the predeclared protocol")
        detected = importlib.metadata.version("isaaclab")
        if detected not in PROCEDURAL_ISAACLAB_DISTRIBUTIONS:
            raise ValueError(
                f"Require Isaac Lab {PROCEDURAL_ISAACLAB_DISTRIBUTIONS}; found {detected}"
            )
        from isaaclab.app import AppLauncher

        app = AppLauncher(headless=True, livestream=0, device=args.device).app
        from isaaclab.envs import ManagerBasedRLEnv

        try:
            from .operator_student_bridge import run_recurrent_training
        except ImportError:
            from operator_student_bridge import run_recurrent_training
        if evaluation:
            cfg, runner_cfg = recurrent_evaluation_configs(
                saved, agent, args, archived_protocol, metadata
            )
        elif resuming:
            cfg, runner_cfg = recurrent_source_configs(
                saved, agent, args, archived_protocol, metadata, learned_source
            )
            if args.restart_coverage:
                apply_restart_coverage(cfg)
            if args.pivot_planar_precision:
                apply_pivot_planar_precision(cfg)
            cfg.sim.device = args.device
            runner_cfg.update(
                device=args.device,
                max_iterations=final_updates,
                save_interval=args.save_interval,
            )
        else:
            if refinement:
                recurrent_source_configs(
                    saved, agent, args, archived_protocol, metadata, learned_source
                )
            cfg, runner_cfg = proprioceptive_procedural_configs(
                saved, agent, args, acquisition_version=protocol["version"]
            )
            if refinement:
                runner_cfg["algorithm"]["learning_rate"] = warm_start["learning_rate"]
        cfg.validate()
        params = output / "params"
        params.mkdir()
        (params / "env.yaml").write_text(yaml.dump(cfg.to_dict(), sort_keys=False))
        (params / "agent.yaml").write_text(yaml.dump(runner_cfg, sort_keys=False))
        env = ManagerBasedRLEnv(cfg=cfg)
        if evaluation:
            import numpy as np

            try:
                from .operator_student_bridge import evaluate_recurrent_operator
            except ImportError:
                from operator_student_bridge import evaluate_recurrent_operator
            result, trace = evaluate_recurrent_operator(
                env,
                policy.to(args.device),
                evaluation_files["checkpoint"],
                metadata,
                is_running=app.is_running,
                difficulty_range=args.evaluation_difficulty,
                long_stops=args.evaluation_long_stops,
                command_coverage=args.evaluation_command_coverage,
                negative_pivot_first=args.evaluation_negative_pivot_first,
                reward_capture=getattr(args, "evaluation_reward_capture", False),
                out_and_back=getattr(args, "evaluation_out_and_back", False),
                seed=args.seed,
            )
            np.savez_compressed(output / "trace.npz", **trace)
            if (
                recurrent_evaluation_files(args.procedural_evaluate_checkpoint)
                != evaluation_files
            ):
                raise ValueError("Frozen evaluation source changed during execution")
            result["evaluation_sources"] = evaluation_files
        else:
            result = run_recurrent_training(
                env,
                runner_cfg,
                output,
                is_running=app.is_running,
                iterations=args.iterations,
                **(
                    {"root_point_check": True}
                    if _requires_root_point_check(protocol)
                    else {}
                ),
                **(
                    {"initial_policy": policy, "warm_start": warm_start}
                    if refinement or resuming
                    else {}
                ),
                **(
                    {"resume_from": resume_from, "optimizer_state": optimizer_state}
                    if resuming
                    else {}
                ),
                **(
                    {"restart_coverage_change": protocol["restart_coverage_change"]}
                    if protocol.get("restart_coverage_change") is not None
                    else {}
                ),
                **(
                    {
                        "pivot_planar_precision_change": protocol[
                            "pivot_planar_precision_change"
                        ]
                    }
                    if protocol.get("pivot_planar_precision_change") is not None
                    else {}
                ),
                **(
                    {
                        "terrain_rows": _proprio_terrain_exposure(protocol["version"])[
                            "num_rows"
                        ],
                        "terrain_difficulty": tuple(
                            _proprio_terrain_exposure(protocol["version"])[
                                "difficulty_range"
                            ]
                        ),
                    }
                    if protocol["version"] in PROPRIO_MIXED_TERRAIN_VERSIONS
                    else {}
                ),
            )
            if (refinement or resuming) and recurrent_evaluation_files(
                learned_source
            ) != evaluation_files:
                raise ValueError("Training source changed during execution")
        if identity != recurrent_training_identity(checkpoint):
            raise ValueError("Physical reference or runtime changed during training")
        result.update(
            status=success,
            policy_version=RECURRENT_OPERATOR_VERSION,
            protocol_sha256=file_sha256(output / protocol_name),
            packages={"isaaclab": detected, "rsl-rl-lib": "3.1.2"},
            sha256={name: file_sha256(output / name) for name in artifact_names},
            exit_allowed=False,
        )
        write_json(output / receipt_name, result)
        return 0
    except Exception as error:
        failure = {
            "status": "ERROR",
            "error": str(error),
            "traceback": traceback.format_exc(),
            "exit_allowed": False,
        }
        capture = getattr(env, "operator_capture", None)
        if evaluation and capture is not None and capture.samples:
            try:
                import numpy as np

                np.savez_compressed(output / "partial_trace.npz", **capture.finish())
                failure["partial_trace"] = {
                    "sha256": file_sha256(output / "partial_trace.npz"),
                    "scope": "diagnostic partial capture only; not complete evaluation evidence",
                }
            except Exception as capture_error:
                failure["partial_trace_error"] = str(capture_error)
        write_json(output / receipt_name, failure)
        # Kit cleanup can block or exit Python; expose the error before closing it.
        print(
            f"ERROR: {error} (details: {output / receipt_name})",
            file=sys.stderr,
            flush=True,
        )
        return 2
    finally:
        for name, resource in (("environment", env), ("application", app)):
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    write_json(
                        output / f"{name}_cleanup_error.json", {"error": str(error)}
                    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    procedural = parser.add_mutually_exclusive_group()
    procedural.add_argument(
        "--procedural-evaluate-checkpoint",
        type=Path,
        help="Evaluate one frozen causal checkpoint on the fixed 80-trial tape; --seed 43 (default), 44 or 45 selects development layouts, not exit acceptance; no learning or optimizer resume",
    )
    procedural.add_argument(
        "--procedural-refine-checkpoint",
        type=Path,
        help="Warm-start from an immutable recurrent checkpoint with exact model/std and saved scalar LR, fresh Adam and recurrent state; not resume. stop_yaw sources support matched stop_yaw/link_origin restarts",
    )
    procedural.add_argument(
        "--procedural-resume-checkpoint",
        type=Path,
        help="Continue the archived recurrent recipe with exact model/std, Adam and live LR in a new run. --iterations adds updates; checkpoint names stay cumulative. Inherit seed, environment count and save interval; simulator/commands/GRU/RNG reset, not bitwise continuation",
    )
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument(
        "--restart-coverage",
        action="store_true",
        help="Explicit sampling-stage change with --procedural-resume-checkpoint from higher_terrain: enrich low-speed post-hold restarts, retaining Adam/live LR and all other settings. Omit on subsequent resumes; the stage is inherited",
    )
    stages.add_argument(
        "--pivot-planar-precision",
        action="store_true",
        help="Explicit one-factor reward stage with --procedural-resume-checkpoint from unchanged higher_terrain: pure-pivot planar precision fraction 0.1 to 0.3, preserving widths, weight, saved Adam/live LR and every other setting. Omit on subsequent resumes; the stage is inherited",
    )
    parser.add_argument(
        "--procedural-refinement",
        choices=tuple(PROPRIO_REFINEMENT_VERSIONS),
        help="With --procedural-refine-checkpoint: stock restarts v3; terrain_exposure adds fixed difficulty bands; arrival_hold adds arrival/hold/restart sampling; stance adds zero-command posture cost; pivot_precision adds planar pivot precision; stationary_yaw adds angular precision; stop_yaw narrows full-stop angular width or restarts stop_yaw unchanged; link_origin changes planar reference point from stop_yaw or restarts link_origin unchanged; higher_terrain extends link_origin to five fixed bands through0.55; stop_precision is historical",
    )
    procedural.add_argument(
        "--procedural-train",
        action="store_true",
        help="Train a fresh proprioceptive GRU with the v3 joint-limit and upright-posture objective on fixed easy supported terrain; reference binds physics only; no resume or exit acceptance",
    )
    parser.add_argument(
        "--evaluation-difficulty",
        type=float,
        nargs=2,
        metavar=("LOW", "HIGH"),
        help="Frozen v3 evaluation only: change terrain amplitude within [0,1] and capture joint/actuator/contact diagnostics; omitted preserves the original easy screen. Does not change training or enable a curriculum",
    )
    parser.add_argument(
        "--evaluation-long-stops",
        action="store_true",
        help="With --evaluation-difficulty: extend the two post-motion stops from 2 to 10 seconds (36-second frozen probe); keep the approach, native diagnostics and short-stop deadline unchanged",
    )
    parser.add_argument(
        "--evaluation-command-coverage",
        action="store_true",
        help="With --evaluation-difficulty, instead of --evaluation-long-stops: fixed 63-second frozen screen with longer reverse/pivots, flat slow/lateral commands and prospective development control limits; no acceptance",
    )
    parser.add_argument(
        "--evaluation-out-and-back",
        action="store_true",
        help="With explicit difficulty and no other tape/reward options: frozen 30.28-second flat/rough return diagnostic with balanced initial headings/turn signs and passive foot/mesh capture; no steering, learning or acceptance",
    )
    parser.add_argument(
        "--evaluation-reward-capture",
        action="store_true",
        help="Frozen evaluation with explicit difficulty only: record native reward contributions in the existing trace; no reward, policy or scoring changes",
    )
    parser.add_argument(
        "--evaluation-negative-pivot-first",
        action="store_true",
        help="With --evaluation-command-coverage: swap only the two 6-second pivots; retain the first 25 seconds, suffix and thresholds. Separate frozen order/sign diagnostic, not canonical coverage or acceptance",
    )
    procedural.add_argument(
        "--procedural-config-check",
        action="store_true",
        help="Check the easy operator-only native configuration; no environment, rollout or learning. Add --validate-only for checkpoint/source and controller-adapter CPU validation",
    )
    procedural.add_argument(
        "--procedural-rollout-check",
        action="store_true",
        help="Run 200 control steps with native physics through the stock oracle adapter; 20-80 environments, no learning or terrain acceptance",
    )
    parser.add_argument(
        "--terrain-train",
        action="store_true",
        help="Retired four-family acquisition mode (rejected); use --procedural-config-check",
    )
    parser.add_argument(
        "--readiness-report",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--terrain-critic-context",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--terrain-evaluate-checkpoint", type=Path, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--terrain-readiness",
        action="store_true",
        help="Retired four-family readiness mode (rejected); use --procedural-config-check",
    )
    parser.add_argument(
        "--mesh-flat-report",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--refinement-profile",
        choices=("source", *PROFILES),
        default="source",
        help="Preserve the source profile, or explicitly select a versioned reward/entropy objective",
    )
    parser.add_argument(
        "--curriculum",
        choices=(*VERSIONS, SEQUENCE_VERSION),
        default=VERSION,
        help="Versioned commands; v3 requires evidence-bound reversal-stop continuation",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Additional PPO updates (default: procedural training/refinement/resume 1000, no upper cap; legacy refinement 300; frozen evaluation 0)",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--save-interval",
        type=int,
        help="Procedural checkpoint interval in cumulative completed PPO updates (default: 50; resume inherits source); the final checkpoint is always saved",
    )
    parser.add_argument(
        "--moving-retention",
        action="store_true",
        help="Source-mean retention; v2 keeps its 200-update protocol, evidence-bound v3 permits up to 3000",
    )
    parser.add_argument(
        "--check-offsets",
        nargs="+",
        type=int,
        help="One to three saved-checkpoint offsets after the source; positive, increasing and 50-aligned, including the final update. Defaults to +100/+200 for 200 updates, otherwise final only. All checks run after training exits",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Checkpoint-bound development report for --moving-retention",
    )
    parser.add_argument(
        "--resume-retention-reference",
        type=Path,
        help="Resume completed retention checkpoint AND Adam, keeping this original frozen reference (requires --moving-retention)",
    )
    parser.add_argument(
        "--zero-command-reference-report",
        type=Path,
        help="Opt in to separately normalized zero-twist retention using this original-reference stand/stop-pass screen; requires --resume-retention-reference",
    )
    parser.add_argument(
        "--reversal-stop-probe",
        type=Path,
        help="Complete two-arm negative recovery evidence for the single v3 sequence-exposure intervention",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Source-only CPU preflight for sequence v3 and procedural modes; no output or simulator launch; native configuration comparison remains unrun",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("logs/rsl_rl/go2_operator_refinement"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional positive finite worker time limit in seconds (default: no limit for procedural training/refinement/resume; 3600 for other modes)",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Training smoke only; do not claim behavioral acceptance",
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    evaluation = args.procedural_evaluate_checkpoint is not None
    refinement = args.procedural_refine_checkpoint is not None
    resuming = args.procedural_resume_checkpoint is not None
    learning = args.procedural_train or refinement or resuming
    if args.restart_coverage and not resuming:
        parser.error("--restart-coverage requires --procedural-resume-checkpoint")
    if args.pivot_planar_precision and not resuming:
        parser.error("--pivot-planar-precision requires --procedural-resume-checkpoint")
    if args.procedural_refinement is not None and not refinement:
        parser.error("--procedural-refinement requires --procedural-refine-checkpoint")
    if refinement and args.procedural_refinement is None:
        args.procedural_refinement = "stock"
    if (
        args.evaluation_difficulty is not None
        or args.evaluation_long_stops
        or args.evaluation_command_coverage
        or args.evaluation_negative_pivot_first
        or args.evaluation_reward_capture
        or args.evaluation_out_and_back
    ):
        if not evaluation:
            parser.error("Evaluation options require --procedural-evaluate-checkpoint")
        try:
            try:
                from .operator_student_bridge import recurrent_evaluation_protocol
            except ImportError:
                from operator_student_bridge import recurrent_evaluation_protocol
            recurrent_evaluation_protocol(
                difficulty_range=args.evaluation_difficulty,
                long_stops=args.evaluation_long_stops,
                command_coverage=args.evaluation_command_coverage,
                negative_pivot_first=args.evaluation_negative_pivot_first,
                reward_capture=args.evaluation_reward_capture,
                out_and_back=args.evaluation_out_and_back,
                seed=args.seed if args.seed is not None else 43,
            )
        except ValueError as error:
            parser.error(str(error))
    if args.save_interval is not None and (not learning or args.save_interval < 1):
        parser.error(
            "--save-interval requires procedural training/refinement/resume and a positive integer"
        )
    if learning and not resuming and args.save_interval is None:
        args.save_interval = 50
    if args.timeout is None and not learning:
        args.timeout = 3600
    if args.iterations is None:
        args.iterations = 0 if evaluation else (1000 if learning else 300)
    if args.num_envs is None and not resuming:
        args.num_envs = 80 if evaluation else (1280 if learning else 4096)
    if args.seed is None and not resuming:
        args.seed = 43 if evaluation else 42
    if (
        learning
        or evaluation
        or args.procedural_config_check
        or args.procedural_rollout_check
    ):
        try:
            return (
                recurrent_training_main(args, parser)
                if learning or evaluation
                else procedural_config_main(args, parser)
            )
        except Exception as error:
            result = {"status": "ERROR", "error": str(error), "exit_allowed": False}
            output = getattr(args, "procedural_output", None)
            if output is not None:
                result["output"] = str(output)
                try:
                    write_json(output / "report.json", result)
                except Exception as publication_error:
                    result["report_write_error"] = str(publication_error)
            print(json.dumps(result, indent=2, allow_nan=False), flush=True)
            return 2
    if (
        args.terrain_train
        or args.terrain_readiness
        or args.terrain_critic_context
        or args.readiness_report is not None
        or args.terrain_evaluate_checkpoint is not None
        or args.mesh_flat_report is not None
    ):
        parser.error(
            "Four-family terrain training/readiness is retired. Use "
            "--procedural-config-check --num-envs 80 for the replacement configuration; "
            "historical artifact replay remains in scripts.analysis.operator_terrain_audit."
        )
    if (args.curriculum == SEQUENCE_VERSION) != (args.reversal_stop_probe is not None):
        parser.error("v3 and --reversal-stop-probe must be selected together")
    if args.reversal_stop_probe is not None and (
        not args.moving_retention
        or args.resume_retention_reference is None
        or args.zero_command_reference_report is None
    ):
        parser.error("v3 must preserve both retention terms and the original reference")
    if args.validate_only and (
        args.reversal_stop_probe is None or args.worker_output is not None
    ):
        parser.error("--validate-only is reserved for v3 parent preflight")
    if (
        args.iterations < 1
        or args.num_envs < 4
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
    ):
        parser.error("Use positive iterations/timeout and at least four environments")
    if args.seed < 0:
        parser.error("Training seed must be a nonnegative integer")
    if args.seed in PROPRIO_EVALUATION_SEEDS:
        parser.error(
            "Seeds 43–48 are reserved for evaluation; choose a different training seed"
        )
    try:
        args.check_offsets = retention_check_offsets(args)
    except ValueError as error:
        parser.error(str(error))
    try:
        args.checkpoint = args.checkpoint.resolve(strict=True)
        agent_path, env_path = (
            args.checkpoint.parent / "params" / name
            for name in ("agent.yaml", "env.yaml")
        )
        agent, saved = read_yaml_data(agent_path), read_yaml_data(env_path)
    except Exception as error:
        parser.error(f"Invalid source files ({type(error).__name__}): {error}")
    if (
        args.baseline_report is not None or args.resume_retention_reference is not None
    ) and not args.moving_retention:
        parser.error(
            "--baseline-report/--resume-retention-reference require --moving-retention"
        )
    if args.zero_command_reference_report is not None and (
        not args.moving_retention or args.resume_retention_reference is None
    ):
        parser.error(
            "--zero-command-reference-report requires --moving-retention and --resume-retention-reference"
        )
    baseline = None
    args.retention_resume = None
    args.zero_command_reference = None
    if args.moving_retention:
        try:
            baseline = retention_preflight(args, agent, saved)
            if args.reversal_stop_probe is not None:
                args.retention_resume = sequence_resume_preflight(
                    args, agent, saved, baseline
                )
            elif args.resume_retention_reference is not None:
                args.retention_resume = retention_resume_preflight(
                    args, agent, saved, baseline
                )
            if args.zero_command_reference_report is not None:
                args.zero_command_reference = zero_command_retention_preflight(
                    args, baseline, args.retention_resume
                )
        except Exception as error:
            parser.error(str(error))
    if args.validate_only:
        print(
            f"Validated v3 source, Adam8000, original reference, both losses and two negative recovery arms; {args.iterations} updates, check offsets {args.check_offsets}, final Adam step {8000 + 20 * args.iterations}. Training remains UNRUN."
        )
        return 0
    if args.worker_output is not None:
        try:
            run_training(args, args.worker_output, agent, saved)
        except Exception as error:
            write_json(
                args.worker_output / "training_status.json",
                {
                    "status": "ERROR",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        return 0

    try:
        select_profile(saved, agent, args.refinement_profile)
        source = load_reference_checkpoint(args.checkpoint, agent)
    except Exception as error:
        parser.error(str(error))  # Reject before Kit/worker/output creation.
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(prefix="operator_refine_", dir=args.output_parent)
    ).resolve()
    print(f"Operator refinement (headless, streaming off): {output}", flush=True)
    write_run_provenance(output, Path(__file__))
    if baseline is not None:
        write_json(
            output / "retention_protocol.json",
            {
                "baseline": baseline,
                "loss": retention_manifest(
                    zero_command=args.zero_command_reference is not None
                ),
                "zero_command_reference": args.zero_command_reference,
                "training_updates": args.iterations,
                "check_offsets": args.check_offsets,
                "seed": args.seed,
                "evaluation_seed": 43,
                "optimizer": (
                    "exact restored Adam; fixed original reference"
                    if args.retention_resume
                    else "one fresh Adam at the initial source; uninterrupted through all 200 updates"
                ),
                "retention_resume": args.retention_resume,
                "curriculum": training_curriculum_manifest(args.curriculum),
                "evaluation_timing": "after training worker exit; never concurrent Kit workers",
                "stop_screening": [
                    "complete unchanged physical PASS",
                    "execution/integrity error",
                ],
                "regressed_candidate": "reject individually, but still screen the remaining predeclared, already-trained candidates; learning can be non-monotonic",
                "not_promoted": True,
            },
        )
    provenance = {
        "source_checkpoint": str(args.checkpoint),
        "sha256": {
            "checkpoint": file_sha256(args.checkpoint),
            "agent.yaml": file_sha256(agent_path),
            "env.yaml": file_sha256(env_path),
        },
        "packages": {},
    }
    for name in (
        "operator_train.py",
        "operator_curriculum.py",
        "operator_command.py",
        "operator_profiles.py",
        "operator_rewards.py",
        "operator_retention.py",
        "operator_benchmark.py",
        "operator_benchmark_core.py",
        *(
            (
                "operator_sequences.py",
                "operator_sequence_command.py",
                "operator_sequence_resume.py",
                "operator_stop_probe.py",
            )
            if args.curriculum == SEQUENCE_VERSION
            else ()
        ),
    ):
        provenance["sha256"][name] = file_sha256(Path(__file__).with_name(name))
    for package in ("isaaclab", "isaacsim", "rsl-rl-lib", "torch"):
        try:
            provenance["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            provenance["packages"][package] = "unknown"
    write_json(output / "source_provenance.json", provenance)
    if args.retention_resume is not None and provenance["packages"] != {
        name: baseline["packages"].get(name) for name in provenance["packages"]
    }:
        write_json(
            output / "report.json",
            {
                "status": "ERROR",
                "error": "Runtime packages changed since the retention baseline",
            },
        )
        print(
            f"ERROR: runtime packages changed; see {output / 'report.json'}", flush=True
        )
        return 2
    report = supervise(
        [
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            str(args.checkpoint),
            "--worker-output",
            str(output),
            "--iterations",
            str(args.iterations),
            "--num-envs",
            str(args.num_envs),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--curriculum",
            args.curriculum,
            "--refinement-profile",
            args.refinement_profile,
            *(
                [
                    "--moving-retention",
                    "--baseline-report",
                    str(args.baseline_report.resolve()),
                ]
                if args.moving_retention
                else []
            ),
            *(
                [
                    "--resume-retention-reference",
                    str(args.resume_retention_reference.resolve()),
                ]
                if args.retention_resume is not None
                else []
            ),
            *(
                [
                    "--zero-command-reference-report",
                    str(args.zero_command_reference_report.resolve()),
                ]
                if args.zero_command_reference is not None
                else []
            ),
            *(
                ["--reversal-stop-probe", str(args.reversal_stop_probe.resolve())]
                if args.reversal_stop_probe is not None
                else []
            ),
            *(
                ["--check-offsets", *(str(offset) for offset in args.check_offsets)]
                if args.moving_retention
                else []
            ),
        ],
        output,
        timeout_s=args.timeout,
        report_filename="training_status.json",
        valid_statuses=("COMPLETED", "ERROR"),
    )
    final = output / f"model_{source['iter'] + args.iterations}.pt"
    if report["status"] == "COMPLETED" and (
        report.get("final_checkpoint") != str(final) or not final.is_file()
    ):
        report = {
            "status": "ERROR",
            "error": "Worker did not save the expected final checkpoint",
            "training_result": report,
        }
    write_json(output / "report.json", report)
    if report["status"] == "ERROR":
        print(report.get("traceback", report.get("error")), flush=True)
        return 2
    if baseline is not None:
        if (
            report.get("handoff", {}).get("source_sha256")
            != baseline["sha256"]["checkpoint"]
            or (
                args.retention_resume is not None
                and report.get("handoff", {}).get("retention_resume")
                != args.retention_resume
            )
            or (
                args.curriculum == SEQUENCE_VERSION
                and (
                    report.get("handoff", {}).get("curriculum") != sequence_manifest()
                    or report.get("handoff", {}).get("additional_updates")
                    != args.iterations
                    or report.get("handoff", {})
                    .get("moving_retention", {})
                    .get("check_offsets")
                    != args.check_offsets
                )
            )
            or (
                args.zero_command_reference is not None
                and (
                    report.get("handoff", {}).get("zero_command_reference")
                    != args.zero_command_reference
                    or report.get("handoff", {})
                    .get("moving_retention", {})
                    .get("zero_command_retention")
                    != args.zero_command_reference["intervention"]
                )
            )
        ):
            report.update(
                status="ERROR", error="Training source differs from retention baseline"
            )
            write_json(output / "report.json", report)
            return 2
        checkpoints = [
            output / f"model_{source['iter'] + offset}.pt"
            for offset in args.check_offsets
        ]
        print(
            f"Training worker finished all {args.iterations} updates. Screening only predeclared offsets {args.check_offsets}.",
            flush=True,
        )
        report["operator_check"], code = run_retention_checks(
            checkpoints, output / "operator_check", args.device, baseline
        )
        try:
            (output / "operator_check").mkdir(parents=True, exist_ok=True)
            write_json(output / "operator_check/report.json", report["operator_check"])
            write_json(output / "report.json", report)
        except OSError as error:
            # A launch may fail before it creates a screen directory. Preserve
            # ERROR/exit2 even when nested evidence itself cannot be published.
            code = 2
            report["operator_check"].update(
                status="ERROR", publication_error=str(error)
            )
            try:
                write_json(output / "report.json", report)
            except OSError as parent_error:
                print(
                    f"Could not publish retention report: {parent_error}",
                    file=sys.stderr,
                )
        print(
            f"{report['operator_check']['status']}: {output / 'report.json'}",
            flush=True,
        )
        return code
    print(f"Training completed: {final}", flush=True)
    if args.skip_check:
        print("NOT behaviorally checked (--skip-check).", flush=True)
        return 0
    print("Running the unchanged 100-trial operator screen, seed 43.", flush=True)
    report["operator_check"], exit_code = run_final_check(
        final, output / "operator_check", args.device
    )
    write_json(output / "report.json", report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
