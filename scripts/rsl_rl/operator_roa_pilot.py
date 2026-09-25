"""Native ROA mechanism pilot and bounded incremental-learning comparison.

Alternate privileged PPO with reverse latent regularization and history-owned
adaptation blocks, sharing one motor. Both paths use causal estimated velocity.
No oracle fallback, controller promotion, live input or hardware I/O.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time
import traceback

VERSION = "operator_roa_learning_pilot_v3"
ENVIRONMENT_VERSION = "operator_roa_environment_learning_v1"


@dataclass(frozen=True)
class EnvironmentStage:
    version: str
    source_stage: str
    source_updates: int
    updates: int
    status: str
    result_stage: str
    geometry_version: str | None = None
    support_resets: bool = False
    step_clearance: bool = False
    contact_conditioned: bool = False


ENVIRONMENT_STAGES = {
    "hills": EnvironmentStage(
        ENVIRONMENT_VERSION,
        "estimator_refinement",
        1000,
        1000,
        "ROA_ENVIRONMENT_LEARNING_COMPLETED_NOT_QUALIFIED",
        "environment_learning",
    ),
    "step_fields": EnvironmentStage(
        "operator_roa_step_field_learning_v1",
        "environment_learning",
        2000,
        500,
        "ROA_STEP_FIELD_LEARNING_COMPLETED_NOT_QUALIFIED",
        "step_field_learning",
        geometry_version="operator_step_field_v1",
    ),
    "step_support": EnvironmentStage(
        "operator_roa_step_support_learning_v1",
        "step_field_learning",
        2500,
        500,
        "ROA_STEP_SUPPORT_LEARNING_COMPLETED_NOT_QUALIFIED",
        "step_support_learning",
        geometry_version="operator_step_field_v1",
        support_resets=True,
    ),
    "step_bootstrap": EnvironmentStage(
        "operator_roa_step_bootstrap_learning_v1",
        "step_field_learning",
        2500,
        500,
        "ROA_STEP_BOOTSTRAP_LEARNING_COMPLETED_NOT_QUALIFIED",
        "step_bootstrap_learning",
        geometry_version="operator_step_field_bootstrap_v1",
        support_resets=True,
    ),
    "step_clearance": EnvironmentStage(
        "operator_roa_step_clearance_learning_v1",
        "step_field_learning",
        2500,
        500,
        "ROA_STEP_CLEARANCE_LEARNING_COMPLETED_NOT_QUALIFIED",
        "step_clearance_learning",
        geometry_version="operator_step_field_bootstrap_v1",
        support_resets=True,
        step_clearance=True,
    ),
    "contact_teacher": EnvironmentStage(
        "operator_roa_contact_teacher_learning_v1",
        "step_field_learning",
        2500,
        500,
        "ROA_CONTACT_TEACHER_LEARNING_COMPLETED_NOT_QUALIFIED",
        "contact_teacher_learning",
        geometry_version="operator_step_field_bootstrap_v1",
        support_resets=True,
        contact_conditioned=True,
    ),
}
REGULARIZATION_COEFFICIENTS = (0.1, 0.55, 1.0)
ROLLOUT_STEPS = 24
HISTORY_STEPS = 64
ADAPTATION_EPOCHS = 4
ADAPTATION_BATCHES = 4
FROZEN_STEPS = 32
LEARNING_UPDATES = 100
HISTORY_INTERVAL = 20
EVALUATION_SEED = 1042
DYNAMICS_NAMES = (
    "base_mass_relative_to_nominal_minus_one",
    "base_local_com_x_m",
    "base_local_com_y_m",
    "base_local_com_z_m",
    "mean_robot_shape_static_friction",
    "mean_robot_shape_dynamic_friction",
    "mean_robot_shape_restitution",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--version", action="version", version=f"{VERSION}\n{Path(__file__).resolve()}"
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="Validated stock 48-D checkpoint; NOT a GRU checkpoint",
    )
    parser.add_argument("--num-envs", type=int, choices=(80, 160, 320), default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument(
        "--learning-checkpoint",
        type=Path,
        help="Completed v2 pilot.pt: initialize a bounded learning experiment; fresh optimizers",
    )
    sources.add_argument(
        "--environment-checkpoint",
        type=Path,
        help="Completed source for the selected bounded environment stage; fresh optimizers",
    )
    parser.add_argument(
        "--environment-layout",
        choices=tuple(ENVIRONMENT_STAGES),
        help="hills: refined source; step_fields: hills source; other step stages: step-field source",
    )
    parser.add_argument(
        "--regularization",
        choices=("ramp", "off"),
        help="Additional regularization only: zero for 20 updates then ramp to 0.1, or remain zero",
    )
    parser.add_argument("--learning-updates", type=int, choices=(100, 500, 1000))
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("logs/rsl_rl/go2_operator_refinement"),
    )
    args = parser.parse_args(argv)
    if args.seed < 0 or args.cpu_threads < 1:
        parser.error("seed must be nonnegative and cpu-threads positive")
    learning = args.learning_checkpoint or args.environment_checkpoint
    if args.environment_layout is not None and args.environment_checkpoint is None:
        parser.error("--environment-layout requires --environment-checkpoint")
    args.environment_layout = args.environment_layout or "hills"
    stage = ENVIRONMENT_STAGES[args.environment_layout]
    if (learning is None) != (args.regularization is None):
        parser.error("A learning checkpoint and --regularization must be used together")
    if args.learning_updates is not None and learning is None:
        parser.error("--learning-updates requires a learning checkpoint")
    if learning is not None and args.learning_updates is None:
        args.learning_updates = (
            stage.updates if args.environment_checkpoint else LEARNING_UPDATES
        )
    if args.environment_checkpoint and (
        args.regularization != "off" or args.learning_updates != stage.updates
    ):
        parser.error(
            f"{args.environment_layout} requires --regularization off and {stage.updates} new updates"
        )
    return args


def learning_coefficients(regularization, updates=LEARNING_UPDATES):
    """A predeclared short experiment, not the paper's convergence schedule."""
    if regularization not in ("ramp", "off") or type(updates) is not int or updates < 1:
        raise ValueError("Unknown additional-regularization experiment")
    return tuple(
        0.1 * min(1, max(0, update - 20) / 80) if regularization == "ramp" else 0.0
        for update in range(1, updates + 1)
    )


def load_learning_source(path, physical_reference):
    """Admit only completed v2 mechanism evidence, never a deployment artifact."""
    import torch
    from parkour_lab.learning.motor_contract import validate_motor_contract
    from .operator_train import file_sha256

    path = path.resolve(strict=True)
    files = (path, path.parent / "training_protocol.json", path.parent / "report.json")
    identity = {str(item): file_sha256(item) for item in files}
    protocol, report = (json.loads(item.read_text()) for item in files[1:])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    motor = report["motor_delivery"]
    if (
        protocol["version"] != "operator_roa_learning_pilot_v2"
        or checkpoint["version"] != protocol["version"]
        or protocol["source_identity"]["physical_reference"] != physical_reference
        or report.get("session_status", report["status"])
        != "ROA_LEARNING_PILOT_COMPLETED_NOT_QUALIFIED"
        or report["status"]
        not in (
            "ROA_LEARNING_PILOT_COMPLETED_NOT_QUALIFIED",
            "SESSION_COMPLETED_CLEANUP_PENDING",
        )
        or report["checkpoint_sha256"] != identity[str(path)]
        or checkpoint["readiness_only"] is not True
        or checkpoint["deployment_allowed"] is not False
        or checkpoint["completed_cycles"] != 3
        or tuple(checkpoint["regularization_coefficients"])
        != REGULARIZATION_COEFFICIENTS
        or report["ppo_updates_completed"] != 3
        or report["adaptation_optimizer_steps"] != 48
        or report["exit_allowed"] is not False
        or report["behavior_validated"] is not False
        or motor["faulted"] is not False
        or motor["pending_delivery"] is not False
        or any(
            motor[name] != 296
            for name in (
                "encoded_steps",
                "verified_delivery_steps",
                "native_step_returns",
            )
        )
        or report["environment_transitions"] != 296 * protocol["num_envs"]
    ):
        raise ValueError("Require a completed, source-bound v2 ROA pilot")
    validate_motor_contract(checkpoint["motor_contract"], checkpoint["motor_manifest"])
    if any(file_sha256(item) != identity[str(item)] for item in files):
        raise ValueError("Learning source changed while loading")
    return checkpoint, {
        "files": identity,
        "policy_state_sha256": report["policy_state_sha256"],
        "optimizer": "fresh PPO and adaptation Adam; weights-only warm start, NOT exact resume",
        "comparison": "Effect of ADDITIONAL regularization from a common already-ROA-initialized policy; not ROA versus no ROA",
    }


def load_environment_source(path, physical_reference, seed, *, layout="hills"):
    """Admit only the selected stage's completed, source-linked predecessor."""
    from .operator_roa_checkpoint import load_completed_checkpoint

    policy, contract, _, receipt = load_completed_checkpoint(path)
    stage = ENVIRONMENT_STAGES[layout]
    if (
        receipt.get("stage") != stage.source_stage
        or receipt["physical_reference"] != physical_reference
        or receipt["learning_updates"] != stage.source_updates
        or receipt["training_seed"] == seed
    ):
        raise ValueError(
            f"Require completed {stage.source_stage} with {stage.source_updates} selected-lineage updates and a fresh seed"
        )
    return {"policy_state": policy.state_dict(), "motor_contract": contract}, receipt


def validate_events(cfg, *, support_resets=False):
    """Only this reconstructed startup-persistent physics recipe can be cached."""
    expected = {
        "physics_material": ("startup", "randomize_rigid_body_material"),
        "add_base_mass": ("startup", "randomize_rigid_body_mass"),
        "base_external_force_torque": ("reset", "apply_external_force_torque"),
        "reset_base": ("reset", "reset_root_state_uniform"),
        "reset_robot_joints": ("reset", "reset_joints_by_scale"),
    }
    active = {
        name: term
        for name, term in vars(cfg.events).items()
        if term is not None and not name.startswith("_")
    }
    if set(active) != set(expected):
        raise ValueError(
            "ROA pilot requires exactly the reviewed persistent-physics events"
        )
    for name, (mode, function) in expected.items():
        term = active[name]
        if support_resets and name == "reset_base":
            from .operator_step_support import reset_root_state

            if term.mode != mode or term.func is not reset_root_state:
                raise ValueError("Unreviewed support-patch reset event")
            continue
        if (
            term.mode != mode
            or term.func.__name__ != function
            or term.func.__module__ != "isaaclab.envs.mdp.events"
        ):
            raise ValueError(f"Unreviewed ROA event: {name}")


def read_dynamics(env):
    """Actual startup properties, not sampled requests or effective contact friction."""
    import torch

    robot = env.scene["robot"]
    base = robot.body_names.index("base")
    mass = robot.root_physx_view.get_masses()[:, base : base + 1].to(env.device)
    nominal = robot.data.default_mass[:, base : base + 1].to(env.device)
    com = robot.root_physx_view.get_coms()[:, base, :3].to(env.device)
    material = (
        robot.root_physx_view.get_material_properties().to(env.device).mean(dim=1)
    )
    if (nominal <= 0).any() or (mass <= 0).any():
        raise ValueError("Invalid native mass")
    result = torch.cat((mass / nominal - 1.0, com, material), dim=1).float()
    if result.shape != (env.num_envs, 7) or not torch.isfinite(result).all():
        raise ValueError("Invalid measured dynamics vector")
    return result.detach().clone()


class PilotEnvironment:
    """One native observation delivery, one verified motor delivery per control tick."""

    def __init__(self, env, app, *, contact_conditioned=False):
        import torch
        from parkour_lab.learning.operator_roa import CausalHistory
        from parkour_lab.learning.motor_contract import make_motor_contract
        from .operator_motor_bridge import (
            NativeJointTargetBridge,
            _runtime_motor_binding,
            NATIVE_RAW_ACTION_MEANING,
        )

        self.env, self.app = env, app
        self.history = CausalHistory()
        binding, digest = _runtime_motor_binding(env)
        self.manifest = {
            "joint_names": binding["joint_names"],
            "period_s": 0.02,
            "configuration": {"default_position_rad": binding["default_position_rad"]},
            "actuator_profile": "native_motor_sha256:" + digest,
            "raw_action_meaning": NATIVE_RAW_ACTION_MEANING,
        }
        self.motor_contract = make_motor_contract(
            binding, self.manifest["actuator_profile"]
        )
        self.bridge = NativeJointTargetBridge(
            env, self.motor_contract, self.manifest, preserve_native_raw=True
        )
        self.dynamics = read_dynamics(env)
        self.contacts = None
        if contact_conditioned:
            from .operator_roa_contacts import ContactFeatures

            self.contacts = ContactFeatures(env)
        self.previous = torch.zeros((env.num_envs, 12), device=env.device)
        self.resets = self.partial_reset_steps = self.steps = 0
        self.stage_counts = {}
        if env.step_dt != 0.02 or env.physics_dt != 0.005 or env.cfg.decimation != 4:
            raise ValueError(
                "ROA pilot must retain native 50Hz control / 200Hz physics"
            )

    def observations(self, native, reset, command=None):
        import torch
        from tensordict import TensorDict

        frame, clean = native["proprio"], native["policy"]
        if command is not None:
            # The frozen screen owns just the command component. Keep the one
            # native noisy sensor draw and push history exactly once per tick.
            term = self.env.command_manager.get_term("base_velocity")
            desired = frame.new_tensor(command)
            if (
                desired.shape != (3,)
                or not torch.isfinite(desired).all()
                or not torch.equal(frame[:, 6:9], term.vel_command_b)
                or not torch.equal(clean[:, 9:12], term.vel_command_b)
            ):
                raise ValueError(
                    "Invalid command or native command observation binding"
                )
            desired = desired.expand(self.env.num_envs, 3)
            term.time_left.fill_(float("inf"))
            term.is_standing_env.fill_(False)
            term.is_heading_env.fill_(False)
            term.vel_command_b.copy_(desired)
            frame, clean = frame.clone(), clean.clone()
            frame[:, 6:9], clean[:, 9:12] = desired, desired
        expected = self.previous.clone()
        expected[reset] = 0
        command = self.env.command_manager.get_command("base_velocity")
        if (
            frame.shape != (self.env.num_envs, 45)
            or clean.shape != (self.env.num_envs, 48)
            or not torch.equal(frame[:, -12:], expected)
            or not torch.equal(frame[:, 6:9], command)
            or not torch.equal(
                clean[:, :3], self.env.scene["robot"].data.root_lin_vel_b
            )
        ):
            raise ValueError(
                "Pre-action sensor, command, COM-velocity or previous-action alignment failed"
            )
        # Teacher and student share noisy proprioception; only the declared
        # teacher extension sees contacts. The scan remains critic-only.
        context = {"contacts": self.contacts.sample(reset)} if self.contacts else {}
        return TensorDict(
            {
                "policy": frame.clone(),
                "history": self.history.push(frame, reset).flatten(1),
                "critic_state": clean.clone(),
                "terrain": native["terrain"].clone(),
                "dynamics": self.dynamics.clone(),
                **context,
            },
            batch_size=[self.env.num_envs],
        )

    def reset(self, *, seed=None, command=None):
        import torch

        native, _ = self.env.reset(**({"seed": seed} if seed is not None else {}))
        if not torch.equal(read_dynamics(self.env), self.dynamics):
            raise ValueError("Persistent dynamics changed on reset")
        self.previous.zero_()
        reset = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        return self.observations(native, reset, command), reset

    def step(self, raw, stage, *, next_command=None):
        import torch
        from parkour_lab.learning.controller import JointTargets

        if not self.app.is_running():
            raise RuntimeError("Simulation application stopped during learning pilot")
        raw = raw.detach().clone()
        targets = self.bridge.default + 0.25 * raw
        delivered = self.bridge.encode(
            JointTargets(self.bridge.joint_names, targets, raw)
        )
        with torch.no_grad():
            native, reward, terminated, timed_out, extras = self.env.step(delivered)
        self.bridge.verify_delivery(terminated, timed_out)
        done = terminated | timed_out
        self.previous = delivered.clone()
        self.steps += 1
        self.resets += int(done.sum())
        self.partial_reset_steps += int(done.any() and not done.all())
        counts = self.stage_counts.setdefault(
            stage, {"control_steps": 0, "terminated_rows": 0, "timeout_rows": 0}
        )
        counts["control_steps"] += 1
        counts["terminated_rows"] += int(terminated.sum())
        counts["timeout_rows"] += int(timed_out.sum())
        return (
            self.observations(native, done, next_command),
            reward,
            done,
            {**extras, "time_outs": timed_out & ~terminated},
        )


def run_pilot(
    env,
    app,
    source_state,
    runner_cfg,
    output,
    report,
    publish,
    *,
    learning=None,
    environment=False,
    layout="hills",
):
    started = time.perf_counter()
    stage = ENVIRONMENT_STAGES[layout] if environment else None
    host = PilotEnvironment(
        env, app, contact_conditioned=bool(stage and stage.contact_conditioned)
    )
    report["stage_counts"] = host.stage_counts
    report["dynamics_std"] = host.dynamics.std(dim=0, unbiased=False).cpu().tolist()
    try:
        _learn_pilot(
            host,
            source_state,
            runner_cfg,
            output,
            report,
            publish,
            learning=learning,
            environment=environment,
            layout=layout,
        )
    finally:
        set_training_mechanisms(env, active=False)
        if host.contacts is not None:
            report["teacher_contact_observations"] = host.contacts.report()
        motor_progress = host.bridge.progress()
        report.update(
            runner_wall_seconds=time.perf_counter() - started,
            environment_transitions=motor_progress["native_step_returns"]
            * env.num_envs,
            resets=host.resets,
            partial_reset_steps=host.partial_reset_steps,
            motor_delivery=motor_progress,
        )


def set_training_mechanisms(env, *, active):
    """Keep experimental resets/rewards out of frozen screens and cleanup."""
    for name in ("_operator_step_support", "_operator_step_clearance"):
        state = getattr(env, name, None)
        if state is not None:
            state.active = active


def _adapt_history(
    host, policy, optimizer, obs, record, report, *, require_change, observe=None
):
    """Collect with fixed causal weights, then fit only the history estimator."""
    import torch
    from parkour_lab.learning.operator_roa import (
        FRAME_DIM,
        HISTORY_LENGTH,
        set_phase,
        state_sha256,
    )

    actor, env = policy.actor, host.env

    def frames(observations):
        return observations["history"].reshape(-1, HISTORY_LENGTH, FRAME_DIM)

    def privileged_hashes():
        return {
            name: state_sha256(module)
            for name, module in (
                ("motor", actor.motor),
                ("encoder", actor.encoder),
                ("critic", policy.critic),
            )
        }

    # Collect the whole history-owned block with fixed weights. Fit only
    # afterwards, so no action can incorporate its own privileged label.
    set_phase(policy, "frozen")
    fixed_hashes = privileged_hashes()
    fixed_std = policy.std.detach().clone()
    fixed_estimator = state_sha256(actor.estimator)
    samples = []
    with torch.no_grad():
        for _ in range(HISTORY_STEPS):
            if observe is not None:
                observe()
            samples.append(
                (
                    frames(obs).clone(),
                    actor.privileged_input(obs).clone(),
                    obs["critic_state"][:, :3].clone(),
                )
            )
            action = actor.history_action(obs["policy"], frames(obs))
            obs, _, _, _ = host.step(action, "history_adaptation")
    record["estimator_unchanged_during_history_collection"] = (
        state_sha256(actor.estimator) == fixed_estimator
    )
    if not record["estimator_unchanged_during_history_collection"]:
        raise RuntimeError("History estimator changed before block collection ended")
    histories, privilege, velocities = (
        torch.cat(values, dim=0) for values in zip(*samples)
    )
    del samples
    with torch.no_grad():
        before = actor.adaptation_losses(histories, privilege, velocities)
    record["adaptation_before"] = {name: float(value) for name, value in before.items()}
    set_phase(policy, "history")
    for _ in range(ADAPTATION_EPOCHS):
        for indices in torch.randperm(len(histories), device=env.device).chunk(
            ADAPTATION_BATCHES
        ):
            optimizer.zero_grad(set_to_none=True)
            losses = actor.adaptation_losses(
                histories[indices], privilege[indices], velocities[indices]
            )
            loss = losses["latent"] + losses["velocity"]
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite ROA adaptation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                actor.estimator.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            report["adaptation_optimizer_steps"] += 1
    with torch.no_grad():
        after = actor.adaptation_losses(histories, privilege, velocities)
    record["adaptation_after"] = {name: float(value) for name, value in after.items()}
    record["privileged_modules_unchanged_during_adaptation"] = (
        fixed_hashes == privileged_hashes() and torch.equal(fixed_std, policy.std)
    )
    record["estimator_changed_during_adaptation"] = (
        state_sha256(actor.estimator) != fixed_estimator
    )
    if not record["privileged_modules_unchanged_during_adaptation"] or (
        require_change and not record["estimator_changed_during_adaptation"]
    ):
        raise RuntimeError(
            "ROA adaptation violated optimizer ownership or did not learn"
        )
    if (
        not all(torch.isfinite(p).all() for p in policy.parameters())
        or (policy.std <= 0).any()
    ):
        raise RuntimeError("Invalid ROA parameters")
    return obs


def _learn_pilot(
    host,
    source_state,
    runner_cfg,
    output,
    report,
    publish,
    *,
    learning=None,
    environment=False,
    layout="hills",
):
    import torch
    from parkour_lab.learning.operator_roa import (
        build_policy,
        HISTORY_LENGTH,
        FRAME_DIM,
        state_sha256,
        branch_diagnostics,
        set_phase,
    )
    from parkour_lab.learning.operator_roa_training import ROAPPO

    env = host.env
    stage = ENVIRONMENT_STAGES[layout] if environment else None
    if environment and learning is None:
        raise ValueError("Environment learning requires a validated warm start")
    evaluation_seed = learning[3] + 1000 if environment else EVALUATION_SEED
    exposure = None
    support = getattr(env, "_operator_step_support", None)
    if (support is not None) != bool(stage and stage.support_resets):
        raise ValueError("Support-reset admission must match the selected stage")
    clearance = getattr(env, "_operator_step_clearance", None)
    if (clearance is not None) != bool(stage and stage.step_clearance):
        raise ValueError("Foot-clearance admission must match the selected stage")
    set_training_mechanisms(env, active=False)
    obs, _ = host.reset()
    policy, reference = build_policy(obs, source_state)
    actor = policy.actor

    def frames(observations):
        return observations["history"].reshape(-1, HISTORY_LENGTH, FRAME_DIM)

    with torch.no_grad():
        velocity = actor.estimate(frames(obs))[:, :3]
        stock_input = torch.cat((velocity, obs["policy"]), dim=1)
        if not torch.equal(policy.act_inference(obs), reference.actor(stock_input)):
            raise RuntimeError("ROA initial motor arithmetic compatibility failed")
        if not torch.equal(policy.evaluate(obs), reference.evaluate(obs)):
            raise RuntimeError("ROA initial critic compatibility failed")
    del reference
    report.update(
        initial_motor_arithmetic_parity=True,
        parity_scope="Identical estimated-velocity inputs; NOT equivalence to the true-velocity stock controller",
        motor_verification=host.bridge.motor_verification,
    )
    coefficients = REGULARIZATION_COEFFICIENTS
    history_interval = 1
    if learning is not None:
        from .operator_roa_evaluation import evaluate_history
        from parkour_lab.learning.motor_contract import binding_sha256

        checkpoint, metadata, regularization, seed, updates = learning
        if updates % HISTORY_INTERVAL:
            raise ValueError("Learning budget must end on a complete adaptation block")
        if binding_sha256(checkpoint["motor_contract"]["binding"]) != binding_sha256(
            host.motor_contract["binding"]
        ):
            raise ValueError("Learning checkpoint motor differs from native motor")
        policy.load_state_dict(checkpoint["policy_state"], strict=True)
        if (
            state_sha256(policy) != metadata["policy_state_sha256"]
            or not all(torch.isfinite(p).all() for p in policy.parameters())
            or (policy.std <= 0).any()
        ):
            raise ValueError(
                "Learning checkpoint state is invalid or differs from report"
            )
        if stage and stage.contact_conditioned:
            with torch.no_grad():
                before_teacher = actor.encode(obs["dynamics"]).clone()
                before_action = actor.history_action(obs["policy"], frames(obs)).clone()
                before_privileged_action = policy.act_inference(obs).clone()
                actor.enable_contact_conditioning()
                policy.obs_groups["policy"] = actor.privileged_obs_groups
                if not (
                    torch.equal(
                        before_teacher, actor.encode(actor.privileged_input(obs))
                    )
                    and torch.equal(
                        before_action, actor.history_action(obs["policy"], frames(obs))
                    )
                    and torch.equal(before_privileged_action, policy.act_inference(obs))
                ):
                    raise RuntimeError(
                        "Contact teacher initialization changed source outputs"
                    )
            report["contact_initialization"] = {
                "source_policy_state_sha256": metadata["policy_state_sha256"],
                "initialized_policy_state_sha256": state_sha256(policy),
                "teacher_latent_exact": True,
                "causal_action_exact": True,
                "privileged_action_exact": True,
                "new_projection_zero": True,
            }
        report["evaluation_before"] = evaluate_history(
            host, policy, seed=evaluation_seed
        )
        publish()
        # Reset the training RNG and physical episodes after the diagnostic.
        # This is a fresh seeded run, not simulator/RNG continuation of the pilot.
        set_training_mechanisms(env, active=True)
        obs, _ = host.reset(seed=seed)
        policy.train()
        coefficients = learning_coefficients(regularization, updates)
        history_interval = HISTORY_INTERVAL
        if environment:
            from .operator_roa_adapt import TerrainExposure

            exposure = TerrainExposure(env, geometry_version=stage.geometry_version)
            report.update(
                inherited_ppo_updates=metadata["learning_updates"],
                optimizer_initialization="Fresh PPO and history Adam; full-policy warm start, NOT exact resume",
                update_counting_scope="Counts from the selected v3 experiment onward; excludes the ancestor's three-update mechanism pilot",
            )
    options = copy.deepcopy(runner_cfg["algorithm"])
    options.pop("class_name")
    options.update(
        learning_rate=1e-4,
        schedule="fixed",
        desired_kl=None,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    report["ppo_options"] = options
    algorithm = ROAPPO(policy, device=env.device, **options)
    algorithm.init_storage("rl", env.num_envs, ROLLOUT_STEPS, obs, [12])
    optimizer = torch.optim.Adam(
        actor.estimator.parameters(), lr=1e-4 if environment else 1e-3
    )
    report["cycles"] = []
    report["ppo_updates_completed"] = 0
    report["adaptation_optimizer_steps"] = 0

    def observe():
        if exposure is not None:
            exposure.sample()
        if support is not None:
            support.sample()

    def publish_exposure():
        if exposure is not None:
            report["training_exposure"] = exposure.report()
        if support is not None:
            report["training_support_resets"] = support.report()
        if clearance is not None:
            report["training_step_clearance"] = clearance.report()
        if host.contacts is not None:
            report["teacher_contact_observations"] = host.contacts.report()

    def save_checkpoint(update):
        from .operator_train import file_sha256

        path = output / ("pilot.pt" if learning is None else f"learning_{update}.pt")
        pending = path.with_suffix(".pt.pending")
        torch.save(
            {
                "version": stage.version if stage else VERSION,
                "readiness_only": True,
                "deployment_allowed": False,
                "policy_state": policy.state_dict(),
                "ppo_optimizer": algorithm.optimizer.state_dict(),
                "adaptation_optimizer": optimizer.state_dict(),
                "motor_contract": host.motor_contract,
                "motor_manifest": host.manifest,
                "completed_cycles": update,
                "regularization_coefficients": coefficients[:update],
                "history_interval": history_interval,
                "learning_source": None if learning is None else metadata,
            },
            pending,
        )
        pending.replace(path)
        receipt = {
            "path": path.name,
            "sha256": file_sha256(path),
            "ppo_updates": update,
        }
        report.setdefault("checkpoints", []).append(receipt)

    for cycle, coefficient in enumerate(coefficients, 1):
        record = {"cycle": cycle, "regularization_coefficient": coefficient}
        report["cycles"].append(record)
        set_phase(policy, "privileged")
        estimator_before = state_sha256(actor.estimator)
        encoder_before = state_sha256(actor.encoder)
        algorithm.regularization_coef = coefficient
        with torch.no_grad():
            for _ in range(ROLLOUT_STEPS):
                observe()
                action = algorithm.act(obs)
                obs, reward, done, extras = host.step(action, "privileged_ppo")
                algorithm.process_env_step(obs, reward, done.long(), extras)
            algorithm.compute_returns(obs)
        record["ppo_losses"] = algorithm.update()
        report["ppo_updates_completed"] += 1
        if learning is None and (
            min(
                record["ppo_losses"]["encoder_gradient_l2_max"],
                record["ppo_losses"]["projection_gradient_l2_max"],
            )
            <= 0
        ):
            raise RuntimeError("ROA encoder or motor-injection branch had no gradient")
        record["estimator_unchanged_during_ppo"] = (
            state_sha256(actor.estimator) == estimator_before
        )
        record["encoder_changed_during_ppo"] = (
            state_sha256(actor.encoder) != encoder_before
        )
        record["latent_branch"] = branch_diagnostics(actor, obs)
        if not record["estimator_unchanged_during_ppo"]:
            raise RuntimeError("ROA PPO changed the frozen history/velocity estimator")
        if learning is None and not record["encoder_changed_during_ppo"]:
            raise RuntimeError("ROA privileged encoder did not update")
        if learning is None and (
            record["latent_branch"]["latent_batch_std_max"] <= 0
            or record["latent_branch"]["latent_permutation_action_abs_max"] <= 0
        ):
            raise RuntimeError("ROA latent branch is constant or unused")
        print(
            f"ROA update {cycle}/{len(coefficients)}: privileged PPO complete; lambda={coefficient}",
            flush=True,
        )
        publish_exposure()
        publish()
        if cycle % history_interval:
            continue

        obs = _adapt_history(
            host,
            policy,
            optimizer,
            obs,
            record,
            report,
            require_change=learning is None,
            observe=observe,
        )
        print(
            f"ROA update {cycle}/{len(coefficients)}: history adaptation complete",
            flush=True,
        )
        checkpoint_interval = len(coefficients) if environment else 100
        if learning is not None and (
            cycle % checkpoint_interval == 0 or cycle == len(coefficients)
        ):
            save_checkpoint(cycle)
        publish_exposure()
        publish()

    set_training_mechanisms(env, active=False)
    policy.eval()
    set_phase(policy, "frozen")
    frozen_hash = state_sha256(policy)
    if learning is None:
        obs, _ = host.reset()
        with torch.no_grad():
            for _ in range(FROZEN_STEPS):
                action = actor.history_action(obs["policy"], frames(obs))
                obs, _, _, _ = host.step(action, "frozen_history_integration")
    else:
        report["evaluation_after"] = evaluate_history(
            host, policy, seed=evaluation_seed
        )
        report["evaluation_initial_conditions_match"] = {
            name: (
                report["evaluation_before"].get(name)
                == report["evaluation_after"].get(name)
                if report["evaluation_before"].get(name) is not None
                and report["evaluation_after"].get(name) is not None
                else None
            )
            for name in (
                "initial_observation_sha256",
                "initial_dynamics_sha256",
                "initial_root_state_sha256",
                "terrain_assignment",
                "command_tape_sha256",
            )
        }
    if state_sha256(policy) != frozen_hash or not torch.equal(
        read_dynamics(env), host.dynamics
    ):
        raise RuntimeError("Frozen policy or persistent dynamics changed")
    if learning is None:
        save_checkpoint(len(coefficients))
    report.update(
        status=(
            stage.status
            if stage
            else (
                "ROA_LEARNING_PILOT_COMPLETED_NOT_QUALIFIED"
                if learning is None
                else "ROA_INCREMENTAL_EXPERIMENT_COMPLETED_NOT_QUALIFIED"
            )
        ),
        policy_state_sha256=frozen_hash,
        frozen_rollout_scope="Same training scene/dynamics; no heldout terrain or deployment qualification",
    )
    if environment:
        report["cumulative_ppo_updates"] = metadata["learning_updates"] + len(
            coefficients
        )


def finish_session(env, app, report, publish, code):
    """Close in order; preserve the last durable receipt if Kit exits in close()."""
    resources = (("environment", env), ("application", app))
    report["session_status"] = report["status"]
    if code == 0:
        report["status"] = "SESSION_COMPLETED_CLEANUP_PENDING"
    report["cleanup"] = {
        name: "pending" if resource is not None else "not_created"
        for name, resource in resources
    }

    def save():
        nonlocal code
        try:
            publish()
        except Exception as error:
            code = 2
            report.update(status="ERROR", report_write_error=str(error))
            try:
                print(f"Could not publish ROA report: {error}", flush=True)
            except OSError:
                pass  # A broken logging pipe must not prevent resource cleanup.

    save()
    for name, resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
            report["cleanup"][name] = "complete"
        except Exception as error:
            report["cleanup"][name] = f"ERROR: {error}"
            code = 2
        save()
    report["status"] = report["session_status"] if code == 0 else "ERROR"
    save()
    return code


def main(argv=None):
    args = parse_args(argv)
    environment = args.environment_checkpoint is not None
    stage = ENVIRONMENT_STAGES[args.environment_layout] if environment else None
    version = stage.version if stage else VERSION
    print(f"ROA entry point: {Path(__file__).resolve()} ({version})", flush=True)
    from .operator_play import configure_live_execution, verify_live_execution

    execution = configure_live_execution(args.cpu_threads)
    from . import operator_train as training

    args.reference = args.reference.resolve(strict=True)
    if args.output_parent.resolve().is_relative_to(args.reference.parent):
        raise ValueError("Output must be outside the immutable source run")
    if importlib.metadata.version("rsl-rl-lib") != "3.1.2":
        raise ValueError("ROA pilot requires pinned RSL-RL 3.1.2")
    identity = training.recurrent_training_identity(args.reference)
    agent = training.read_yaml_data(args.reference.parent / "params/agent.yaml")
    saved = training.read_yaml_data(args.reference.parent / "params/env.yaml")
    source = training.load_reference_checkpoint(args.reference, agent)
    if training.recurrent_training_identity(args.reference) != identity:
        raise ValueError("Source or runtime changed during preflight")
    learning = None
    learning_path = args.environment_checkpoint or args.learning_checkpoint
    if learning_path is not None:
        learning_path = learning_path.resolve(strict=True)
        if environment:
            checkpoint, metadata = load_environment_source(
                learning_path,
                identity["physical_reference"],
                args.seed,
                layout=args.environment_layout,
            )
        else:
            checkpoint, metadata = load_learning_source(
                learning_path, identity["physical_reference"]
            )
        if any(
            args.output_parent.resolve().is_relative_to(Path(path).parent)
            for path in metadata["files"]
        ):
            raise ValueError("Output must be outside every immutable source run")
        learning = (
            checkpoint,
            metadata,
            args.regularization,
            args.seed,
            args.learning_updates,
        )
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(prefix="operator_roa_pilot_", dir=args.output_parent)
    ).resolve()
    print(f"ROA learning pilot: {output}", flush=True)
    protocol = {
        "version": version,
        "source_identity": identity,
        "seed": args.seed,
        "stock_motor_source": str(args.reference),
        "num_envs": args.num_envs,
        "cycles": len(REGULARIZATION_COEFFICIENTS),
        "regularization_coefficients": REGULARIZATION_COEFFICIENTS,
        "rollout_steps_per_update": ROLLOUT_STEPS,
        "history_steps_per_cycle": HISTORY_STEPS,
        "frozen_history_steps": FROZEN_STEPS,
        "dynamics": list(DYNAMICS_NAMES),
        "dynamics_sampling": "post-startup; persistent physics checked on phase resets and completion",
        "velocity": "root_lin_vel_b: root COM linear velocity expressed in root body frame, m/s",
        "history": "25 delivered noisy 45-D frames, oldest to current t; 0.48s span at 50Hz; reset rows repeat first frame",
        "supervision": "pre-action COM velocity labels and detached privileged dynamics latent; neither velocity labels nor scans enter actors",
        "model": {
            "version": "operator_roa_pilot_v1",
            "frame_dim": 45,
            "history_frames": 25,
            "dynamics_dim": 7,
            "latent_dim": 8,
            "velocity_dim": 3,
            "motor": "shared stock48 MLP128x3 ELU; dynamics7->64->32->8 tanh, zero-init8->128 additive projection",
            "history_encoder": "flatten25x45->128->64->11; shared detached estimated velocity3 for BOTH actor paths, tanh latent8",
        },
        "adaptation_optimizer": {
            "class_name": "Adam",
            "learning_rate": 1e-3,
            "gradient_norm_limit": 1.0,
            "loss": "mean per-row unsquared L2(phi - stopgrad(mu)) + supervised velocity MSE",
            "epochs_per_history_block": ADAPTATION_EPOCHS,
            "minibatches_per_epoch": ADAPTATION_BATCHES,
        },
        "adaptation_collection": "64 fixed-weight history-owned steps, then estimator-only fitting; no PPO on these actions",
        "privileged_update": "PPO + lambda * mean per-row unsquared L2(mu - stopgrad(phi)); estimator frozen",
        "schedule_scope": "accelerated 3-cycle mechanism check; NOT paper H=20 / long-run lambda schedule",
        "method_source": "https://proceedings.mlr.press/v205/fu23a/fu23a-supp.pdf",
        "reference_implementation": "MarkFzp/Deep-Whole-Body-Control@8159e4ed8695b2d3f62a40d2ab8d88205ac5021a",
        "actor_scan": False,
        "critic_scan": True,
        "normalization": "none; retain stock observation scales",
        "terrain": "existing five-profile easy 0.05–0.15 acquisition recipe; no gaps; NOT challenging-terrain qualification",
        "reward": "existing proprio acquisition v3; native weighted rates multiplied by control dt once",
        "randomization": "unchanged source startup base mass; fixed source materials/COM; not broad sim-to-real randomization",
        "learning_scope": "mechanism readiness only; new unqualified teacher and student; not a continuation of LINK20k",
        "exit_allowed": False,
    }
    if learning is not None:
        from .operator_roa_evaluation import (
            COMMAND_TAPE,
            COMMAND_TAPE_SHA256,
            EVALUATION_STEPS,
        )

        protocol.update(
            learning_source=metadata,
            cycles=args.learning_updates,
            regularization_coefficients=learning_coefficients(
                args.regularization, args.learning_updates
            ),
            history_interval=HISTORY_INTERVAL,
            history_blocks=args.learning_updates // HISTORY_INTERVAL,
            checkpoint_interval=100,
            frozen_history_steps=2 * EVALUATION_STEPS,
            evaluation={
                "seed": EVALUATION_SEED,
                "steps_each": EVALUATION_STEPS,
                "command_tape": COMMAND_TAPE,
                "command_tape_sha256": COMMAND_TAPE_SHA256,
                "scope": "Seeded before/after causal screen in the SAME training geometry and dynamics; not heldout terrain",
                "training_rng": "Native reset(seed=training_seed) after the pre-screen; fresh episode/history",
            },
            planned_environment_transitions=args.num_envs
            * (
                args.learning_updates * ROLLOUT_STEPS
                + args.learning_updates // HISTORY_INTERVAL * HISTORY_STEPS
                + 2 * EVALUATION_STEPS
            ),
            adaptation_collection="64 fixed-weight history-owned steps after each 20 PPO updates; then estimator-only fitting",
            schedule_scope="Bounded incremental experiment; warmup20 then lambda ramp to0.1 at100 and hold versus0 control, NOT original paper schedule or convergence",
            learning_scope=(
                "Joint hill/rough-environment motor acquisition from estimator-refined policy; NOT true vertical-step training or qualification"
                if environment
                else metadata["comparison"]
            ),
        )
    if environment:
        from .operator_roa_adapt import DIFFICULTY

        protocol.update(
            environment_checkpoint=str(learning_path),
            terrain_rows=3,
            difficulty_range=DIFFICULTY,
            checkpoint_interval=args.learning_updates,
            terrain="Existing five-profile free-command environments, three static rows; slanted step_hills are NOT vertical stairs; no courses, waypoints or success resets",
            schedule_scope="1000 new PPO updates, H block every20; zero ADDITIONAL regularization from the already-ROA-initialized source; fresh optimizers, not resume",
        )
        protocol["evaluation"]["seed"] = args.seed + 1000
        protocol["adaptation_optimizer"]["learning_rate"] = 1e-4
        if stage.geometry_version:
            from . import operator_step_field as step_field

            protocol.update(
                environment_layout=args.environment_layout,
                step_field_geometry=step_field.envelope(stage.geometry_version),
                terrain="Free mixed environments; only step_hills columns12–15 replaced by versioned rough vertical-step fields",
                learning_scope="Joint true-step field acquisition; no route, waypoint or success reset; not qualification",
                schedule_scope="500 new PPO updates, H every20; zero additional regularization; fresh optimizers, not resume",
            )
        if stage.support_resets:
            from . import operator_step_support as step_support

            protocol.update(
                support_reset_recipe=step_support.recipe(),
                learning_scope="Mixed support-start acquisition in the declared step-field recipe; fixed checks retain center starts; not qualification",
            )
        if stage.step_clearance:
            from . import operator_step_clearance as step_clearance

            protocol.update(
                step_clearance_recipe=step_clearance.recipe(),
                reward="Unchanged proprio acquisition v3 plus training-only step-column low-foot-link-height cost; native dt once",
                learning_scope="Controlled low-foot-lift cost on the bootstrap/support recipe; same 2500-update source; not qualification",
            )
        if stage.contact_conditioned:
            from .operator_roa_contacts import recipe

            protocol["teacher_contacts"] = recipe()
            protocol["model"].update(
                version="operator_roa_contact_pilot_v1",
                contact_dim=12,
                motor="Unchanged shared motor/history estimator; original dynamics7 GEMM plus zero-initialized contact12->64 projection before teacher ELU",
            )
            protocol["supervision"] = (
                "Current body-COM velocity labels and detached dynamics/contact teacher latent; no contacts or privileged state enter the causal actor"
            )
            protocol["learning_scope"] = (
                "Matched contact-teacher information ablation; unchanged bootstrap geometry, rewards, lambda-off schedule and 2500-update source; not qualification"
            )
    report = {
        "status": "RUNNING_NOT_QUALIFIED",
        "exit_allowed": False,
        "behavior_validated": False,
        "sim_to_real": "UNRUN",
    }

    def publish():
        training.write_json(output / "report.json", report)

    app = env = None
    code = 2
    try:
        training.write_run_provenance(output, __file__)
        training.write_json(output / "training_protocol.json", protocol)
        if learning is not None:
            from .operator_roa_checkpoint import verify_source_files

            verify_source_files(metadata)
        if training.recurrent_training_identity(args.reference) != identity:
            raise ValueError("Source or runtime changed before simulator launch")
        if (
            importlib.metadata.version("isaaclab")
            not in training.PROCEDURAL_ISAACLAB_DISTRIBUTIONS
        ):
            raise ValueError("Unsupported Isaac Lab version")
        from isaaclab.app import AppLauncher

        app = AppLauncher(
            headless=True, device=args.device, kit_args=execution["kit_args"]
        ).app
        report["execution"] = verify_live_execution(execution)
        from isaaclab.envs import ManagerBasedRLEnv
        import yaml

        args.iterations = protocol["cycles"]
        cfg, runner_cfg = training.proprioceptive_procedural_configs(saved, agent, args)
        if environment:
            training._configure_recurrent_terrain(cfg, DIFFICULTY, num_rows=3)
        if stage and stage.geometry_version:
            step_field.configure(cfg, version=stage.geometry_version)
        if stage and stage.support_resets:
            step_support.configure(cfg)
        if stage and stage.step_clearance:
            step_clearance.configure(cfg)
        validate_events(cfg, support_resets=bool(stage and stage.support_resets))
        cfg.validate()
        (output / "resolved_env.yaml").write_text(
            yaml.dump(cfg.to_dict(), sort_keys=False)
        )
        env = ManagerBasedRLEnv(cfg=cfg)
        if stage and stage.geometry_version:
            report["native_step_field_geometry"] = step_field.verify_native_geometry(
                env
            )
            publish()
        if stage and stage.support_resets:
            report["native_support_patches"] = step_support.install(
                env, report["native_step_field_geometry"]
            )
            publish()
        if stage and stage.step_clearance:
            step_clearance.install(env)
        run_pilot(
            env,
            app,
            source["model_state_dict"],
            runner_cfg,
            output,
            report,
            publish,
            learning=learning,
            environment=environment,
            layout=args.environment_layout,
        )
        if training.recurrent_training_identity(args.reference) != identity:
            raise RuntimeError("Source or runtime changed during learning pilot")
        if learning is not None and any(
            training.file_sha256(Path(path)) != digest
            for path, digest in metadata["files"].items()
        ):
            raise RuntimeError("Learning source changed during experiment")
        report["checkpoint_sha256"] = training.file_sha256(
            output
            / (
                "pilot.pt"
                if learning is None
                else f"learning_{args.learning_updates}.pt"
            )
        )
        code = 0
    except Exception as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
        traceback.print_exc()
    finally:
        try:
            set_training_mechanisms(env, active=False)
            print(f"ROA pilot report: {output / 'report.json'}", flush=True)
        finally:
            code = finish_session(env, app, report, publish, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
