"""Bounded causal ROA estimator refinement in free-command terrain environments.

No PPO, privileged action substitution, waypoint steering or course completion.
The motor, privileged encoder, critic and action noise stay exactly frozen.
This additional history-owned experiment is not a qualification or exact resume.
"""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import tempfile
import time
import traceback

VERSION = "operator_roa_estimator_refinement_v1"
SCHEDULE = {
    "history_block_steps": 64,
    "blocks": 75,
    "estimator_optimizer_steps": 1200,
    "new_ppo_updates": 0,
    "learning_rate": 1e-4,
}
DIFFICULTY = (0.15, 0.55)
NATIVE_STEPS = 6600  # 4800 collection + two 900-step frozen command checks.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reference", type=Path, help="Original stock48 physical checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Completed final v3 ROA checkpoint; fresh estimator Adam",
    )
    parser.add_argument("--seed", type=int, default=1045)
    parser.add_argument("--num-envs", type=int, choices=(80, 160, 320), default=80)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--output-parent",
        type=Path,
        default=Path("logs/rsl_rl/go2_operator_refinement"),
    )
    args = parser.parse_args(argv)
    if args.seed < 0 or args.cpu_threads < 1:
        parser.error("Require nonnegative seed and positive threads")
    return args


class TerrainExposure:
    """Pre-action training-state counts; no contact/support or success inference."""

    def __init__(self, env, *, step_fields=False):
        import torch
        from parkour_lab.tasks.manager_based.parkour_lab.mdp.terrain.operator_terrain import (
            PROFILE_BY_COLUMN,
        )

        self.env = env
        self.step_fields = step_fields
        terrain = env.scene.terrain
        self.columns = terrain.terrain_types.clone()
        self.levels = terrain.terrain_levels.clone()
        if (
            self.columns.shape != (env.num_envs,)
            or self.levels.shape != self.columns.shape
            or (self.columns < 0).any()
            or (self.columns >= 20).any()
            or (self.levels < 0).any()
            or (self.levels >= 3).any()
        ):
            raise ValueError("Unexpected static three-row terrain assignment")
        self.profiles = list(dict.fromkeys(PROFILE_BY_COLUMN))
        profile_ids = self.columns.new_tensor(
            [self.profiles.index(p) for p in PROFILE_BY_COLUMN]
        )
        self.groups = profile_ids[self.columns] * 3 + self.levels
        self.counts = torch.zeros((15, 6), dtype=torch.int64, device=env.device)
        self.steps = 0

    def sample(self):
        import torch

        env = self.env
        data = env.scene["robot"].data
        local = data.root_pos_w - env.scene.env_origins
        hits = env.scene["base_height_scanner"].data.ray_hits_w
        if hits.shape != (env.num_envs, 1, 3):
            raise ValueError("Exposure requires the native single center ray")
        finite = torch.isfinite(hits[:, 0]).all(1) & torch.isfinite(local).all(1)
        moving = torch.linalg.vector_norm(data.root_lin_vel_b[:, :2], dim=1) > 0.05
        command = env.command_manager.get_command("base_velocity")
        # Geometry taper is exactly zero in these pads, bands and borders.
        outside_band = local[:, 1].abs() > 0.6
        if self.step_fields:
            outside_band |= (self.columns >= 12) & (self.columns < 16)
        off_flat = (local[:, :2].abs().amax(1) > 1.0) & outside_band
        off_flat &= (local[:, :2].abs() < 7.0).all(1) & finite
        nonzero_height = (hits[:, 0, 2] - env.scene.env_origins[:, 2]).abs() > 0.001
        values = torch.stack(
            (
                torch.ones_like(moving),
                command[:, :2].abs().any(1),
                moving,
                moving & off_flat,
                moving & off_flat & nonzero_height,
                ~finite,
            ),
            dim=1,
        )
        self.counts.index_add_(0, self.groups, values.long())
        self.steps += 1

    def report(self):
        import torch

        if not (
            torch.equal(self.columns, self.env.scene.terrain.terrain_types)
            and torch.equal(self.levels, self.env.scene.terrain.terrain_levels)
        ):
            raise RuntimeError("Static terrain assignments changed during refinement")
        names = (
            "samples",
            "translation_command_samples",
            "moving_samples",
            "moving_outside_flat_regions_samples",
            "moving_outside_flat_regions_nonzero_surface_samples",
            "invalid_ray_or_root_samples",
        )
        counts = self.counts.cpu().tolist()
        result = {
            "control_steps": self.steps,
            "column_ids": self.columns.cpu().tolist(),
            "level_ids": self.levels.cpu().tolist(),
            "groups": [
                dict(
                    profile=profile,
                    level=level,
                    **dict(zip(names, counts[i * 3 + level])),
                )
                for i, profile in enumerate(self.profiles)
                for level in range(3)
            ],
            "scope": "All pre-action training states, including reset and later episodes; measured COM XY speed >0.05m/s, center-ray height magnitude >1mm. Root location only, NOT foot support, course completion or qualification.",
        }
        if self.step_fields:
            from .operator_step_field import VERSION

            result["geometry_overrides"] = {"step_hills": VERSION}
        return result


def refine(host, policy, source, output, report, publish, *, seed, source_checkpoint):
    """Reuse the existing H-phase optimizer/collector; never construct PPO."""
    import torch
    from parkour_lab.learning.operator_roa import state_sha256, set_phase
    from .operator_roa_pilot import _adapt_history, HISTORY_STEPS
    from .operator_roa_evaluation import evaluate_history
    from .operator_roa_checkpoint import verify_source_files

    if HISTORY_STEPS != SCHEDULE["history_block_steps"]:
        raise ValueError("History collector budget changed")
    policy.to(host.env.device)
    if state_sha256(policy) != source["policy_state_sha256"]:
        raise ValueError("Source policy changed before refinement")
    original = {
        name: value.detach().clone() for name, value in policy.state_dict().items()
    }
    report.update(
        ppo_updates_completed=0,
        adaptation_optimizer_steps=0,
        completed_blocks=0,
        blocks=[],
        motor_verification=host.bridge.motor_verification,
    )
    report["evaluations_before"] = evaluate_history(host, policy, seed=seed + 1000)
    publish()
    # Native reset restores stochastic command timers; no scripted-command
    # override or hidden-history injection survives the frozen check.
    obs, _ = host.reset(seed=seed)
    exposure = TerrainExposure(host.env)
    optimizer = torch.optim.Adam(
        policy.actor.estimator.parameters(), lr=SCHEDULE["learning_rate"]
    )
    for block in range(1, SCHEDULE["blocks"] + 1):
        record = {"block": block}
        report["blocks"].append(record)
        obs = _adapt_history(
            host,
            policy,
            optimizer,
            obs,
            record,
            report,
            require_change=True,
            observe=exposure.sample,
        )
        report["completed_blocks"] = block
        if block % 5 == 0:
            report["training_exposure"] = exposure.report()
            print(
                f"ROA estimator block {block}/{SCHEDULE['blocks']}; no PPO", flush=True
            )
            publish()
    report["evaluations_after"] = evaluate_history(host, policy, seed=seed + 1000)
    state = policy.state_dict()
    report["fixed_modules_unchanged"] = all(
        torch.equal(value, state[name])
        for name, value in original.items()
        if not name.startswith("actor.estimator.")
    )
    report["estimator_changed"] = any(
        not torch.equal(value, state[name])
        for name, value in original.items()
        if name.startswith("actor.estimator.")
    )
    if not report["fixed_modules_unchanged"] or not report["estimator_changed"]:
        raise RuntimeError("Estimator-only ownership or learning check failed")
    if (
        host.steps != NATIVE_STEPS
        or report["adaptation_optimizer_steps"] != SCHEDULE["estimator_optimizer_steps"]
    ):
        raise RuntimeError("Incomplete estimator refinement budget")
    verify_source_files(source)
    set_phase(policy, "frozen")
    pending = output / "adapted.pt.pending"
    torch.save(
        {
            "version": VERSION,
            "readiness_only": True,
            "deployment_allowed": False,
            "policy_state": state,
            "adaptation_optimizer": optimizer.state_dict(),
            "motor_contract": host.motor_contract,
            "motor_manifest": host.manifest,
            "source_checkpoint": str(source_checkpoint),
            "source": source,
            "completed_blocks": report["completed_blocks"],
        },
        pending,
    )
    pending.replace(output / "adapted.pt")
    report["policy_state_sha256"] = state_sha256(policy)


def main(argv=None):
    args = parse_args(argv)
    from .operator_play import configure_live_execution, verify_live_execution

    execution = configure_live_execution(args.cpu_threads)
    from . import operator_train as training
    from .operator_roa_checkpoint import load_completed_checkpoint, verify_source_files
    from .operator_roa_pilot import PilotEnvironment, validate_events, finish_session
    from parkour_lab.learning.motor_contract import binding_sha256

    args.reference = args.reference.resolve(strict=True)
    args.checkpoint = args.checkpoint.resolve(strict=True)
    if any(
        args.output_parent.resolve().is_relative_to(path.parent)
        for path in (args.reference, args.checkpoint)
    ):
        raise ValueError("Output must be outside immutable source runs")
    identity = training.recurrent_training_identity(args.reference)
    agent = training.read_yaml_data(args.reference.parent / "params/agent.yaml")
    saved = training.read_yaml_data(args.reference.parent / "params/env.yaml")
    training.load_reference_checkpoint(args.reference, agent)
    policy, contract, _, source = load_completed_checkpoint(
        args.checkpoint, allow_refinement=False
    )
    if (
        source["physical_reference"] != identity["physical_reference"]
        or args.seed == source["training_seed"]
    ):
        raise ValueError("Require matching physical source and a new environment seed")
    verify_source_files(source)
    if training.recurrent_training_identity(args.reference) != identity:
        raise ValueError("Physical source or runtime changed during preflight")
    args.output_parent.mkdir(parents=True, exist_ok=True)
    output = Path(
        tempfile.mkdtemp(prefix="operator_roa_adapt_", dir=args.output_parent)
    ).resolve()
    protocol = {
        "version": VERSION,
        "source_identity": identity,
        "source_checkpoint": str(args.checkpoint),
        "source": source,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "schedule": SCHEDULE,
        "evaluation_reset_seed": args.seed + 1000,
        "difficulty_range": list(DIFFICULTY),
        "terrain_rows": 3,
        "planned_environment_transitions": NATIVE_STEPS * args.num_envs,
        "collection": "Persistent causal history and native free body-twist sampling across 75 blocks; native physical episode resets only. Random reset yaw; no routes, waypoint rewards, success resets or adaptive level promotion.",
        "supervision": "Unchanged ROA unsquared latent alignment to frozen privileged encoder plus pre-action true COM velocity MSE. Labels never substitute deployed inputs; fresh estimator Adam, clip1, four epochs/four minibatches per64-step block.",
        "evaluation": "Frozen before/after 900-step command checks in the SAME procedural training geometry; diagnostic retention, not held-out or course qualification.",
        "scope": "Whole freely commanded procedural environments at three static difficulty rows. Existing step_hills has slanted risers, NOT true high-step/stair coverage. No new locomotion-policy updates or sim-to-real validation.",
        "exit_allowed": False,
    }
    training.write_json(output / "training_protocol.json", protocol)
    report = {
        "status": "SOURCE_VALIDATED_NOT_SIMULATED",
        "exit_allowed": False,
        "behavior_validated": False,
        "sim_to_real": "UNRUN",
    }

    def publish():
        training.write_json(output / "report.json", report)

    publish()
    print(f"ROA estimator refinement: {output}", flush=True)
    if args.validate_only:
        return 0
    app = env = host = None
    code = 2
    started = time.monotonic()
    try:
        training.write_run_provenance(output, __file__)
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

        args.iterations = 0
        cfg, _ = training.proprioceptive_procedural_configs(saved, agent, args)
        training._configure_recurrent_terrain(cfg, DIFFICULTY, num_rows=3)
        cfg.seed = cfg.scene.terrain.terrain_generator.seed = args.seed
        validate_events(cfg)
        cfg.validate()
        (output / "resolved_env.yaml").write_text(
            yaml.dump(cfg.to_dict(), sort_keys=False)
        )
        env = ManagerBasedRLEnv(cfg=cfg)
        host = PilotEnvironment(env, app)
        if binding_sha256(contract["binding"]) != binding_sha256(
            host.motor_contract["binding"]
        ):
            raise ValueError("Refinement source differs from native motor")
        report["status"] = "RUNNING_NOT_QUALIFIED"
        refine(
            host,
            policy,
            source,
            output,
            report,
            publish,
            seed=args.seed,
            source_checkpoint=args.checkpoint,
        )
        if training.recurrent_training_identity(args.reference) != identity:
            raise RuntimeError("Physical source or runtime changed during refinement")
        report["checkpoint_sha256"] = training.file_sha256(output / "adapted.pt")
        report["status"] = "ROA_ESTIMATOR_REFINEMENT_COMPLETED_NOT_QUALIFIED"
        code = 0
    except Exception as error:
        report.update(
            status="ERROR", error=str(error), traceback=traceback.format_exc()
        )
        traceback.print_exc()
    finally:
        report["wall_seconds"] = time.monotonic() - started
        try:
            if host is not None:
                motor = host.bridge.progress()
                report.update(
                    motor_delivery=motor,
                    environment_transitions=motor["native_step_returns"] * env.num_envs,
                    verified_control_steps=host.steps,
                    stage_counts=host.stage_counts,
                )
            print(f"ROA refinement report: {output / 'report.json'}", flush=True)
        except Exception as error:
            code = 2
            report.update(status="ERROR", finalization_error=str(error))
        finally:
            code = finish_session(env, app, report, publish, code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
