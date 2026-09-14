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
        SequenceExposureWrapper,
    )
    from .operator_sequence_resume import sequence_resume_preflight
except ImportError:
    from operator_benchmark import reference_config, supervise, write_json
    from operator_benchmark_core import (
        DT,
        OBSERVATION_TERMS,
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
PROPRIO_ACQUISITION_VERSION = "operator_proprio_acquisition_v2"
PROPRIO_LEGACY_ACQUISITION_VERSION = "operator_proprio_acquisition_v1"
PROPRIO_REWARD_CHANGE = {
    "term": "dof_pos_limits",
    "function": "isaaclab.envs.mdp.rewards:joint_pos_limits",
    "from_weight": 0.0,
    "to_weight": -10.0,
    "soft_joint_pos_limit_factor": 0.9,
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
    """Fresh causal GRU recipe on the fixed easy supported acquisition stage.

    The source configuration binds the physical motor only. This policy starts
    from fresh weights; neither stock8500 nor the terrain teacher can be resumed.
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

    if acquisition_version not in (
        PROPRIO_ACQUISITION_VERSION,
        PROPRIO_LEGACY_ACQUISITION_VERSION,
    ):
        raise ValueError("Unsupported proprioceptive acquisition version")
    cfg, runner_cfg = _procedural_environment_configs(saved, agent, args)
    # Do not inherit the source8500 fine stationary kernels into a fresh actor.
    # Keep the rough orientation/air-time adjustments and physical motor.
    apply_reward_profile(cfg, PROFILES["stock"])
    # One-term fresh-learning ablation, not target clipping or a motor change.
    # Preserve the zero-weight recipe when reconstructing archived v1 runs.
    if acquisition_version == PROPRIO_ACQUISITION_VERSION:
        if cfg.scene.robot.soft_joint_pos_limit_factor != 0.9:
            raise ValueError(
                "The joint-limit ablation requires the stock 0.9 soft factor"
            )
        cfg.rewards.dof_pos_limits.weight = PROPRIO_REWARD_CHANGE["to_weight"]
    # Copy the noisy sensor group BEFORE making the privileged critic noiseless.
    # Removing this term at the manager boundary avoids passing oracle velocity
    # into actor normalization, recurrent state or inference preprocessing.
    cfg.observations.proprio = copy.deepcopy(cfg.observations.policy)
    cfg.observations.proprio.base_lin_vel = None
    cfg.observations.policy.enable_corruption = False
    runner_cfg.update(
        run_name=RECURRENT_OPERATOR_VERSION,
        interface_version=RECURRENT_OPERATOR_VERSION,
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


def recurrent_evaluation_source(checkpoint, physical_identity):
    """Read an immutable learned snapshot without requiring its training to finish."""
    try:
        from .operator_student_bridge import load_recurrent_checkpoint
    except ImportError:
        from operator_student_bridge import load_recurrent_checkpoint

    files = recurrent_evaluation_files(checkpoint)
    policy, metadata, digest = load_recurrent_checkpoint(checkpoint, device="cpu")
    protocol = json.loads((checkpoint.parent / "training_protocol.json").read_text())
    recipe = metadata["recipe"]
    archived_agent = read_yaml_data(checkpoint.parent / "params/agent.yaml")
    if (
        digest != files["checkpoint"]
        or protocol["version"]
        not in (PROPRIO_LEGACY_ACQUISITION_VERSION, PROPRIO_ACQUISITION_VERSION)
        or protocol.get("reward_change")
        != (
            PROPRIO_REWARD_CHANGE
            if protocol["version"] == PROPRIO_ACQUISITION_VERSION
            else None
        )
        or protocol["policy_version"] != metadata["policy_version"]
        or protocol["source_identity"]["physical_reference"] != physical_identity
        or protocol["terrain"] != "operator_procedural_surface_v2"
        or protocol["difficulty_range"] != list(PROCEDURAL_EASY_DIFFICULTY)
        or protocol["adaptive_terrain_promotion"] is not False
        or protocol["gaps"] is not False
        or protocol["stage"] != "fixed_easy_acquisition"
        or type(protocol["seed"]) is not int
        or protocol["seed"] < 0
        or protocol["seed"] in (43, 44, 45)
        or type(protocol["num_envs"]) is not int
        or not 80 <= protocol["num_envs"] <= 5120
        or protocol["num_envs"] % 20
        or type(protocol["learning_updates"]) is not int
        or not 1 <= protocol["learning_updates"] <= 3000
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
    if recurrent_evaluation_files(checkpoint) != files:
        raise ValueError("Evaluation source changed during preflight")
    return policy, metadata, protocol, files


def recurrent_evaluation_configs(saved, agent, args, training_protocol, metadata):
    """Validate the full training recipe first, then apply only evaluation overrides."""
    original = copy.copy(args)
    original.num_envs = training_protocol["num_envs"]
    original.seed = training_protocol["seed"]
    original.iterations = training_protocol["learning_updates"]
    original.device = metadata["recipe"]["device"]
    cfg, runner = proprioceptive_procedural_configs(
        saved, agent, original, acquisition_version=training_protocol["version"]
    )
    archived = args.procedural_evaluate_checkpoint.parent / "params"
    if _canonical_runtime_config(cfg.to_dict()) != _canonical_runtime_config(
        read_yaml_data(archived / "env.yaml")
    ) or _canonical_runtime_config(runner) != _canonical_runtime_config(
        read_yaml_data(archived / "agent.yaml")
    ):
        raise ValueError(
            "Reconstructed training configuration differs from learned archive"
        )
    try:
        from .operator_benchmark import make_recorder_cfg
    except ImportError:
        from operator_benchmark import make_recorder_cfg
    cfg.seed = cfg.scene.terrain.terrain_generator.seed = args.seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.observations.proprio.enable_corruption = False
    cfg.recorders = make_recorder_cfg(procedural=True)
    runner.update(seed=args.seed, device=args.device, max_iterations=0, resume=False)
    return cfg, runner


def recurrent_training_main(args, parser):
    """Shared supervised acquisition/evaluation lifecycle; never implicit promotion."""
    evaluation = args.procedural_evaluate_checkpoint is not None
    invalid_budget = (
        (args.iterations, args.num_envs, args.seed) != (0, 80, 43)
        if evaluation
        else (
            not 1 <= args.iterations <= 3000
            or not 80 <= args.num_envs <= 5120
            or args.num_envs % 20
            or args.seed < 0
            or args.seed in (43, 44, 45)
        )
    )
    if (
        invalid_budget
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
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
            "Procedural acquisition requires 1–3000 updates, 80–5120 environments "
            "in multiples of 20 and a training seed outside 43–45; evaluation requires "
            "exactly 0 updates, 80 environments and seed 43. Use a positive timeout. "
            "No legacy training, refinement, retention, skip-check or resume flags."
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
        if evaluation:
            args.procedural_evaluate_checkpoint = (
                args.procedural_evaluate_checkpoint.resolve(strict=True)
            )
            target = args.worker_output or args.output_parent
            if target.resolve().is_relative_to(
                args.procedural_evaluate_checkpoint.parent
            ):
                raise ValueError(
                    "Evaluation output must be outside the immutable training run"
                )
            policy, metadata, archived_protocol, evaluation_files = (
                recurrent_evaluation_source(
                    args.procedural_evaluate_checkpoint, identity["physical_reference"]
                )
            )
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
        "reward_profile": "stock broad tracking kernels; flat_orientation_l2=0, feet_air_time=0.01, dof_pos_limits=-10; no other reward changes, stationary precision or action-retention loss",
        "reward_change": copy.deepcopy(PROPRIO_REWARD_CHANGE),
        "initial_action_std": 0.5,
        "metrics": "per-profile command tracking, measured moving/nonflat exposure, physical failures and timeouts; training data, not held-out success rates",
        "checkpoint_selection": "save every 50 completed updates and final; no automatic selection or promotion",
        "scope": "Acquire a causal gait before progression; short budgets are integration only. No terrain exit acceptance or deployment claim.",
        "exit_allowed": False,
    }
    if evaluation:
        try:
            from .operator_student_bridge import recurrent_evaluation_protocol
        except ImportError:
            from operator_student_bridge import recurrent_evaluation_protocol
        protocol = {
            **recurrent_evaluation_protocol(),
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
        "trace.npz" if evaluation else f"model_{args.iterations}.pt",
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "SOURCE_VALIDATED_NOT_SIMULATED",
                    "protocol": protocol,
                    **(
                        {"native_configuration_comparison": "UNRUN"}
                        if evaluation
                        else {}
                    ),
                },
                indent=2,
            )
        )
        return 0
    if args.worker_output is None:
        args.output_parent.mkdir(parents=True, exist_ok=True)
        output = Path(
            tempfile.mkdtemp(
                prefix=(
                    "operator_proprio_screen_" if evaluation else "operator_proprio_"
                ),
                dir=args.output_parent,
            )
        ).resolve()
        args.procedural_output = output
        write_run_provenance(output, __file__)
        write_json(output / protocol_name, protocol)
        print(
            f"{'Frozen recurrent evaluation' if evaluation else 'Fresh recurrent acquisition'}: {output}",
            flush=True,
        )
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
                    ]
                    if evaluation
                    else ["--procedural-train"]
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
                evaluation
                and recurrent_evaluation_files(args.procedural_evaluate_checkpoint)
                != evaluation_files
            ):
                raise ValueError("Frozen evaluation source changed during execution")
            if result.get("status") == success:
                expected = {
                    "policy_version": RECURRENT_OPERATOR_VERSION,
                    "learning_updates": args.iterations,
                    "environment_transitions": (
                        80000 if evaluation else args.iterations * 24 * args.num_envs
                    ),
                    "protocol_sha256": file_sha256(output / protocol_name),
                    "exit_allowed": False,
                }
                if evaluation:
                    expected.update(
                        control_steps=1000,
                        checkpoint_sha256=evaluation_files["checkpoint"],
                        checkpoint_learning_updates=metadata["learning_updates"],
                        evaluation_sources=evaluation_files,
                    )
                    if len(result.get("trials", [])) != 80:
                        raise ValueError(
                            "Require all 80 first-attempt evaluation trials"
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
                            dict(archive), recurrent_evaluation_protocol()
                        )
                    if any(result.get(key) != value for key, value in summary.items()):
                        raise ValueError(
                            "Evaluation measurements differ from raw first-attempt trace"
                        )
                if not evaluation:
                    # A matching file hash alone does not make a usable checkpoint.
                    import torch

                    learned = torch.load(
                        output / f"model_{args.iterations}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    info = learned["infos"]
                    metadata = info["recurrent_training"]
                    weights = learned["model_state_dict"]
                    adam = learned["optimizer_state_dict"]["state"]
                    if (
                        learned["iter"] != args.iterations - 1
                        or info["learning_updates"] != args.iterations
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
                            state["step"].item() != 20 * args.iterations
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
        else:
            cfg, runner_cfg = proprioceptive_procedural_configs(saved, agent, args)
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
            )
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
        help="Evaluate one frozen causal checkpoint on a fixed 80-trial seed-43 tape; no learning, optimizer resume or exit acceptance",
    )
    procedural.add_argument(
        "--procedural-train",
        action="store_true",
        help="Train a fresh proprioceptive GRU with the v2 soft-joint-limit objective on fixed easy supported terrain; reference binds physics only; no resume or exit acceptance",
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
        help="PPO updates (default: procedural acquisition 1000; refinement 300)",
    )
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
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
        help="v3, procedural-config or procedural-train source-only CPU preflight; no output or simulator launch",
    )
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("logs/rsl_rl/go2_operator_refinement"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="Training worker time limit, seconds",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Training smoke only; do not claim behavioral acceptance",
    )
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    evaluation = args.procedural_evaluate_checkpoint is not None
    if args.iterations is None:
        args.iterations = 0 if evaluation else (1000 if args.procedural_train else 300)
    if args.num_envs is None:
        args.num_envs = 80 if evaluation else (1280 if args.procedural_train else 4096)
    if args.seed is None:
        args.seed = 43 if evaluation else 42
    if (
        args.procedural_train
        or evaluation
        or args.procedural_config_check
        or args.procedural_rollout_check
    ):
        try:
            return (
                recurrent_training_main(args, parser)
                if args.procedural_train or evaluation
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
    if args.seed in (43, 44, 45):
        parser.error(
            "Seeds 43–45 are reserved for evaluation; choose a different training seed"
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
