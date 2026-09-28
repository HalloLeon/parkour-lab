"""Read-only action/reference and terminal-safe actuator/contact diagnostics.

This captures the unchanged stock benchmark, not a new controller or training
recipe. Reference outputs are counterfactual actions on learner-visited states;
they are never executed and do not demonstrate counterfactual recovery.
"""

from __future__ import annotations

import json
import copy
import math
from pathlib import Path

import numpy as np
import torch

try:
    from .operator_benchmark_core import (
        DT,
        file_sha256,
        load_reference_actor,
        read_yaml_data,
    )
except ImportError:
    from operator_benchmark_core import (
        DT,
        file_sha256,
        load_reference_actor,
        read_yaml_data,
    )


VERSION = "operator_control_trace_v1"


def controller_record(
    session,
    *,
    requested_command,
    requested_at_s,
    delivered_position_rad,
    delivery_time_s,
    command_source,
    safety_events,
):
    """JSON-ready snapshot for the new controller boundary, not legacy NPZ v1.

    Enable session capture explicitly (CPU copies are not a real-time logger).
    Persist the session manifest once alongside these records. Delivered targets
    are host-reported command delivery, NOT measured motion/torque. None means
    not delivered/unknown, never inferred from a successful policy computation.
    Replay requested BODY COMMANDS with live sensing for future closed-loop
    comparison; this function neither replays motor actions nor controls hardware.
    """
    from parkour_lab.learning.controller import finite_tensor

    if session.record is None:
        raise ValueError(
            "No captured validated controller input; capture must be enabled"
        )
    record = copy.deepcopy(session.record)
    batch = len(record["applied_command"])
    finite_tensor(requested_command, (batch, 3))
    if (
        not math.isfinite(requested_at_s)
        or not 0 <= requested_at_s <= record["applied_at_s"]
        or not isinstance(command_source, str)
        or not command_source
        or not isinstance(safety_events, (tuple, list))
        or any(not isinstance(event, str) or not event for event in safety_events)
    ):
        raise ValueError("Invalid command source, timestamp or safety events")
    requested = requested_command.detach().cpu().tolist()
    if requested != record["applied_command"] and not safety_events:
        raise ValueError("Changed applied command requires an explicit event/reason")
    if (delivered_position_rad is None) != (delivery_time_s is None):
        raise ValueError("Delivery target and time must both be known or both absent")
    delivered = None
    if delivered_position_rad is not None:
        if record["status"] == "FAULT":
            raise ValueError("Faulted inference cannot claim target delivery")
        finite_tensor(delivered_position_rad, (batch, len(record["joint_names"])))
        if not math.isfinite(delivery_time_s) or delivery_time_s < record["time_s"]:
            raise ValueError("Invalid delivery time")
        delivered = delivered_position_rad.detach().cpu().tolist()
        if delivered != record["requested_position_rad"] and not safety_events:
            raise ValueError(
                "Changed delivered target requires an explicit event/reason"
            )
    record.update(
        version="operator_controller_record_v1",
        controller_artifact_sha256=session.manifest["artifact_sha256"],
        requested_command=requested,
        requested_at_s=requested_at_s,
        command_source=command_source,
        safety_events=list(safety_events),
        delivered_position_rad=delivered,
        delivery_time_s=delivery_time_s,
        delivery_evidence=(
            "NOT_REPORTED"
            if delivered is None
            else "HOST_REPORTED_TARGET_NOT_MEASURED_MOTION"
        ),
    )
    return record


def load_diagnostic_reference(checkpoint: Path, reference: Path):
    """Require matching saved physical/observation contracts; preserve CPU RNG."""
    checkpoint, reference = (
        checkpoint.resolve(strict=True),
        reference.resolve(strict=True),
    )
    environment_hash = file_sha256(checkpoint.parent / "params/env.yaml")
    if file_sha256(reference.parent / "params/env.yaml") != environment_hash:
        raise ValueError("Diagnostic reference must have the same saved env.yaml")
    agent_path = reference.parent / "params/agent.yaml"
    # Actor construction initializes layers before loading weights. Diagnostics
    # must not consume the RNG sequence used by environment initialization.
    with torch.random.fork_rng(devices=[]):
        actor, iteration = load_reference_actor(reference, read_yaml_data(agent_path))
    return actor, {
        "checkpoint": str(reference),
        "iteration": iteration,
        "sha256": {
            "checkpoint": file_sha256(reference),
            "env.yaml": environment_hash,
            "agent.yaml": file_sha256(agent_path),
        },
    }


def _snapshot(values):
    return {
        name: value.detach().to("cpu", copy=True).numpy()
        for name, value in values.items()
    }


def native_reward_snapshot(env, names, weights):
    """Read cached pre-reset rewards; the pinned manager retains them across reset.

    Call after reward computation, before the next step. Never re-execute terms
    using the now-reset state or resampled command.
    """
    names, weights = tuple(names), tuple(weights)
    if (
        env.cfg.decimation != 4
        or not math.isclose(env.step_dt, DT, rel_tol=0, abs_tol=1e-9)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or len(weights) != len(names)
        or not np.isfinite(weights).all()
    ):
        raise ValueError("Invalid native reward timing or term specification")
    manager = env.reward_manager
    if (
        tuple(manager.active_terms) != names
        or tuple(float(manager.get_term_cfg(name).weight) for name in names) != weights
    ):
        raise ValueError("Native reward order or weights changed during capture")
    # The pinned manager stores weighted rates, not per-step contributions.
    contribution = manager._step_reward * env.step_dt
    total = env.reward_buf
    if (
        contribution.shape != (env.num_envs, len(names))
        or total.shape != (env.num_envs,)
        or not torch.isfinite(contribution).all()
        or not torch.isfinite(total).all()
    ):
        raise ValueError("Invalid native reward matrix or total")
    if not torch.allclose(contribution.sum(-1), total, atol=2e-6, rtol=1e-5):
        raise ValueError("Native reward decomposition does not sum to reward_buf")
    return {"reward_contribution": contribution, "reward_total": total}


class ROATrainingTelemetry:
    """Opt-in sampled credit proxies, not per-terrain gradients or qualification.

    Copy only selected complete PPO/H blocks to CPU. No reward calls, random
    draws, actions, optimizer writes or persistent changes to policy caches.
    Common rows are pre-action states paired with native cached transition
    rewards. PPO rows join them by index; H rewards never enter the PPO buffer.
    """

    @staticmethod
    def recipe(updates, history_interval, rollout_steps, history_steps):
        if (
            any(
                type(x) is not int or x < 1
                for x in (updates, history_interval, rollout_steps, history_steps)
            )
            or history_interval not in (5, 20)
            or updates % history_interval
        ):
            raise ValueError("Telemetry requires complete H5/H20 training blocks")
        ends = sorted({history_interval, updates, *range(100, updates + 1, 100)})
        return {
            "schema_version": "operator_roa_training_telemetry_v1",
            "updates": updates,
            "history_interval": history_interval,
            "rollout_steps": rollout_steps,
            "history_steps": history_steps,
            "block_end_cycles": ends,
            "decisions_per_block": history_interval * rollout_steps + history_steps,
            "sampling": "Complete first/final H blocks and blocks ending every100 PPO updates; deterministic, not random sampling or whole-run means",
            "phase_ids": {"ppo": 0, "history_collection": 1},
            "native_timing": "Pre-action state/command/action; cached post-physics pre-reset native rewards, read after step returns; frozen checks excluded",
            "timeout_rule": "RSL3.1.2 native_reward + gamma * stored pre-action value * (timed_out & ~terminated); not terminal-state V",
            "ppo_axes": "update, rollout_step, environment, optional feature; ppo_sample_index joins common transition rows",
            "policy_probe": "After all PPO Adam steps, before storage clear and H collection; old actions under old/new diagonal Gaussians, not minibatch-average KL",
            "ratio_outside_clip": "abs(exp(new_log_prob - old_log_prob) -1) > clip_param; indicator, not whether the signed PPO surrogate chose its clipped branch",
            "advantage": "Raw = returns - stored values; normalized globally over the complete rollout using sample std (ddof1) +1e-8",
            "ground": "Center-ray height relative to tile origin; invalid hits encoded0 with explicit mask; NOT foot support",
            "scope": "Sampled learning-credit proxies only; no gradient attribution, causal claim, promotion or deployment evidence",
        }

    def __init__(
        self,
        env,
        output,
        exposure,
        *,
        updates,
        history_interval,
        rollout_steps,
        history_steps,
    ):
        self.env = env
        self.metadata = self.recipe(
            updates, history_interval, rollout_steps, history_steps
        )
        self.exposure = exposure
        self.names = tuple(env.reward_manager.active_terms)
        self.weights = tuple(
            float(env.reward_manager.get_term_cfg(n).weight) for n in self.names
        )
        self.metadata.update(
            reward_names=list(self.names),
            reward_weights=list(self.weights),
            reward_units="Native weighted rates * control dt exactly once",
            step_dt_s=env.step_dt,
            profiles=list(exposure.profiles),
            geometry_version=exposure.geometry_version,
        )
        self.output = Path(output) / "training_telemetry"
        self.output.mkdir()  # A new run owns a new directory; never overwrite evidence.
        self.cycle = 0
        self.active = self.complete = False
        self.pending = None
        self.samples, self.rollouts, self.files = [], [], []
        self.last_native_index = None

    def begin_cycle(self, cycle):
        if (
            self.pending is not None
            or cycle != self.cycle + 1
            or cycle > self.metadata["updates"]
        ):
            raise ValueError(
                "Telemetry cycles must be consecutive with no pending step"
            )
        if self.active and self.cycle % self.metadata["history_interval"] == 0:
            raise ValueError("Previous telemetry block was not finished")
        self.cycle = cycle
        interval = self.metadata["history_interval"]
        end = ((cycle - 1) // interval + 1) * interval
        self.active = end in self.metadata["block_end_cycles"]

    @torch.no_grad()
    def before_step(self, native_index, stage, raw):
        if not self.active:
            return
        if self.pending is not None or stage not in (
            "privileged_ppo",
            "causal_ppo",
            "history_adaptation",
        ):
            raise ValueError("Telemetry requires exactly one pre-action training frame")
        phase = int(stage == "history_adaptation")
        steps = self.metadata["rollout_steps"]
        interval = self.metadata["history_interval"]
        expected = ((self.cycle - 1) % interval) * steps
        if phase:
            valid_order = self.cycle % interval == 0 and len(self.rollouts) == interval
            valid_order &= all(
                r.get("post_update_complete", False) for r in self.rollouts
            )
            valid_order &= (
                interval * steps
                <= len(self.samples)
                < self.metadata["decisions_per_block"]
            )
        else:
            valid_order = len(self.rollouts) == (self.cycle - 1) % interval
            valid_order &= expected <= len(self.samples) < expected + steps
        if not valid_order or (
            self.last_native_index is not None
            and native_index != self.last_native_index + 1
        ):
            raise ValueError(
                "Telemetry phase/step order or native decision index changed"
            )
        data, scene = self.env.scene["robot"].data, self.env.scene
        if not (
            torch.equal(scene.terrain.terrain_types, self.exposure.columns)
            and torch.equal(scene.terrain.terrain_levels, self.exposure.levels)
        ):
            raise ValueError("Telemetry terrain assignments changed")
        hits = scene["base_height_scanner"].data.ray_hits_w
        if hits.shape != (self.env.num_envs, 1, 3):
            raise ValueError("Telemetry requires a single native center-ground ray")
        valid = torch.isfinite(hits[:, 0]).all(-1)
        values = {
            "command_b_pre": self.env.command_manager.get_command("base_velocity"),
            "raw_action": raw,
            "root_local_pre": data.root_pos_w - scene.env_origins,
            "velocity_b_pre": data.root_lin_vel_b,
            "ground_height_pre": torch.where(
                valid, hits[:, 0, 2] - scene.env_origins[:, 2], 0
            ),
            "ground_valid_pre": valid,
        }
        shapes = {
            "command_b_pre": (self.env.num_envs, 3),
            "raw_action": (self.env.num_envs, 12),
            "root_local_pre": (self.env.num_envs, 3),
            "velocity_b_pre": (self.env.num_envs, 3),
            "ground_height_pre": (self.env.num_envs,),
            "ground_valid_pre": (self.env.num_envs,),
        }
        if any(
            v.shape != shapes[k] or not torch.isfinite(v).all()
            for k, v in values.items()
        ):
            raise ValueError("Invalid pre-action telemetry state")
        self.pending = dict(
            _snapshot(values), cycle=self.cycle, phase=phase, native_index=native_index
        )

    @torch.no_grad()
    def after_step(self, reward, terminated, timed_out):
        if not self.active:
            return
        if self.pending is None:
            raise ValueError("Telemetry has no matching pre-action frame")
        snapshot = native_reward_snapshot(self.env, self.names, self.weights)
        if not torch.equal(reward, snapshot["reward_total"]) or any(
            v.shape != (self.env.num_envs,) or v.dtype != torch.bool
            for v in (terminated, timed_out)
        ):
            raise ValueError(
                "Telemetry reward/termination differs from native transition"
            )
        self.pending.update(
            _snapshot(
                dict(
                    snapshot,
                    terminated=terminated,
                    timed_out=timed_out,
                    bootstrap_time_out=timed_out & ~terminated,
                )
            )
        )
        self.samples.append(self.pending)
        self.last_native_index = self.pending["native_index"]
        self.pending = None

    @torch.no_grad()
    def capture_rollout(self, algorithm, last_obs):
        """After compute_returns, before PPO mutates parameters or clears storage."""
        if not self.active:
            return
        storage = algorithm.storage
        steps = self.metadata["rollout_steps"]
        slot = (self.cycle - 1) % self.metadata["history_interval"]
        if (
            self.pending is not None
            or len(self.rollouts) != slot
            or len(self.samples) != (slot + 1) * steps
            or storage.step != steps
            or algorithm.normalize_advantage_per_mini_batch
        ):
            raise ValueError(
                "Telemetry requires a complete globally-normalized PPO rollout"
            )
        rows = self.samples[-steps:]
        for field, stored in (
            ("raw_action", storage.actions),
            ("command_b_pre", storage.observations["policy"][..., 6:9]),
        ):
            if not np.array_equal(
                np.stack([r[field] for r in rows]), stored.detach().cpu().numpy()
            ):
                raise ValueError("PPO storage does not match captured pre-action rows")
        native = np.stack([r["reward_total"] for r in rows])
        timeout = np.stack([r["bootstrap_time_out"] for r in rows])
        done = np.stack([r["terminated"] | r["timed_out"] for r in rows])
        snapshot = _snapshot(
            {
                "ppo_reward": storage.rewards,
                "ppo_value": storage.values,
                "ppo_return": storage.returns,
                "ppo_advantage_raw": storage.returns - storage.values,
                "ppo_advantage_normalized": storage.advantages,
                "ppo_old_log_prob": storage.actions_log_prob,
                "ppo_old_mean": storage.mu,
                "ppo_old_std": storage.sigma,
                "ppo_boundary_value": algorithm.policy.evaluate(last_obs),
            }
        )
        expected = native + algorithm.gamma * snapshot["ppo_value"][..., 0] * timeout
        if not np.array_equal(
            done, storage.dones[..., 0].cpu().numpy().astype(bool)
        ) or not np.allclose(
            expected, snapshot["ppo_reward"][..., 0], atol=2e-6, rtol=1e-5
        ):
            raise ValueError(
                "PPO reward/timeout ownership differs from native transition"
            )
        if any(not np.isfinite(v).all() for v in snapshot.values()):
            raise ValueError("Nonfinite PPO telemetry")
        parameters = dict(
            gamma=algorithm.gamma,
            gae_lambda=algorithm.lam,
            clip_param=algorithm.clip_param,
            num_learning_epochs=algorithm.num_learning_epochs,
            num_mini_batches=algorithm.num_mini_batches,
            route=algorithm.phase,
        )
        if "ppo" in self.metadata and self.metadata["ppo"] != parameters:
            raise ValueError("PPO telemetry recipe changed")
        self.metadata["ppo"] = parameters
        self.rollouts.append(
            dict(
                snapshot,
                ppo_cycle=self.cycle,
                ppo_sample_index=np.arange(slot * steps, (slot + 1) * steps),
            )
        )

    @torch.no_grad()
    def after_update(self, algorithm):
        if not self.active:
            return
        if (
            not self.rollouts
            or self.rollouts[-1]["ppo_cycle"] != self.cycle
            or self.rollouts[-1].get("post_update_complete", False)
            or algorithm.storage.step != self.metadata["rollout_steps"]
        ):
            raise ValueError(
                "Telemetry policy probe must precede storage clear exactly once"
            )
        policy, storage = algorithm.policy, algorithm.storage
        distribution = policy.distribution
        probes = []
        try:
            for step in range(storage.step):
                # Same batch size as collection; do not call act() or sample().
                policy._update_distribution(
                    policy.get_actor_obs(storage.observations[step])
                )
                probes.append(
                    _snapshot(
                        {
                            "ppo_new_mean": policy.action_mean,
                            "ppo_new_std": policy.action_std,
                            "ppo_new_log_prob": policy.get_actions_log_prob(
                                storage.actions[step]
                            ).unsqueeze(-1),
                        }
                    )
                )
        finally:
            policy.distribution = distribution
        values = {key: np.stack([row[key] for row in probes]) for key in probes[0]}
        old = self.rollouts[-1]
        old_mean, old_std, new_mean, new_std = (
            a.astype(np.float64)
            for a in (
                old["ppo_old_mean"],
                old["ppo_old_std"],
                values["ppo_new_mean"],
                values["ppo_new_std"],
            )
        )
        values["ppo_kl_old_new"] = (
            np.log(new_std / old_std)
            + (old_std**2 + (old_mean - new_mean) ** 2) / (2 * new_std**2)
            - 0.5
        ).sum(-1)
        values["ppo_ratio"] = np.exp(
            values["ppo_new_log_prob"].astype(np.float64)[..., 0]
            - old["ppo_old_log_prob"].astype(np.float64)[..., 0]
        )
        values["ppo_ratio_outside_clip"] = (
            np.abs(values["ppo_ratio"] - 1) > algorithm.clip_param
        )
        if any(not np.isfinite(v).all() for v in values.values()):
            raise ValueError("Nonfinite post-update policy probe")
        self.rollouts[-1].update(values, post_update_complete=True)

    def finish_block(self):
        if not self.active:
            return
        if (
            self.cycle not in self.metadata["block_end_cycles"]
            or self.pending is not None
            or len(self.samples) != self.metadata["decisions_per_block"]
            or len(self.rollouts) != self.metadata["history_interval"]
            or not all(r.get("post_update_complete", False) for r in self.rollouts)
        ):
            raise ValueError("Refuse to publish an incomplete telemetry block")
        arrays = {k: np.stack([r[k] for r in self.samples]) for k in self.samples[0]}
        arrays.update(
            {
                k: np.stack([r[k] for r in self.rollouts])
                for k in self.rollouts[0]
                if k != "post_update_complete"
            }
        )
        arrays.update(
            _snapshot(
                {
                    "column_id": self.exposure.columns,
                    "level_id": self.exposure.levels,
                    "profile_id": self.exposure.groups // 3,
                }
            )
        )
        arrays["metadata_json"] = np.asarray(json.dumps(self.metadata, sort_keys=True))
        path = self.output / f"block_{self.cycle:06d}.npz"
        # Exclusive creation and hash receipt: a partial file is never a completed block.
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        self.files.append(
            dict(
                path=str(path.relative_to(self.output.parent)),
                sha256=file_sha256(path),
                end_cycle=self.cycle,
                decisions=len(self.samples),
                transition_rows=len(self.samples) * self.env.num_envs,
            )
        )
        self.samples.clear()
        self.rollouts.clear()
        self.last_native_index = None
        self.active = False

    def finish(self):
        if (
            self.cycle != self.metadata["updates"]
            or self.active
            or self.pending is not None
            or [f["end_cycle"] for f in self.files] != self.metadata["block_end_cycles"]
        ):
            raise ValueError("Telemetry did not complete every declared block")
        self.complete = True

    def report(self):
        return dict(
            self.metadata,
            complete=self.complete,
            files=list(self.files),
            incomplete_block_decisions=len(self.samples),
            pending_step=self.pending is not None,
        )


class OperatorControlTrace:
    """Pair the actual pre-action frame with the post-physics/pre-reset state."""

    def __init__(self, env, reference=None, *, native_rewards=False, env_ids=None):
        if reference is not None and (
            reference.training or any(p.requires_grad for p in reference.parameters())
        ):
            raise ValueError("Diagnostic reference must be frozen and in eval mode")
        self.env, self.reference = env, reference
        self.pending = None
        self.samples = []
        self.native_rewards = native_rewards
        self.env_ids = slice(None) if env_ids is None else env_ids
        self.substeps = []
        robot, sensor = env.scene["robot"], env.scene["contact_forces"]
        joints, bodies, contacts = (
            list(robot.joint_names),
            list(robot.body_names),
            list(sensor.body_names),
        )
        feet = [name for name in bodies if name.endswith("_foot")]
        if (
            len(joints) != 12
            or len(set(joints)) != 12
            or len(set(bodies)) != len(bodies)
            or len(set(contacts)) != len(contacts)
            or len(feet) != 4
            or not set(feet).issubset(contacts)
        ):
            raise ValueError(
                "Require twelve joints and four uniquely resolved Go2 feet"
            )
        self.foot_ids = [bodies.index(name) for name in feet]
        self.metadata = {
            "schema_version": VERSION,
            "step_dt_s": DT,
            "joint_names": joints,
            "foot_names": feet,
            "contact_body_names": contacts,
            "observation_pre": "exact delivered 48-D frame; command at 9:12, previous raw action at 36:48",
            "action": "original actor's raw mean action, never replaced",
            "reference_action": "frozen reference on that same frame; shadow only",
            "pre_fields": "before env.step and action processing",
            "post_fields": "record_post_step: after physics, before any auto-reset",
            "joint_units": "position/target rad; velocity rad/s; actuator torque N m",
            "contact_normal_force_w": "N; net NORMAL forces only, not tangential friction or full contact wrench",
            "foot_velocity_w": "m/s; body linear velocity in world frame, not a contact-point slip measurement",
            "sampling": "50 Hz; torque/contact fields are last-substep snapshots, not peaks over all physics substeps",
            "scope": "diagnostic only; no action blending, reference rollout, training, or acceptance relaxation",
        }
        if reference is None:
            self.metadata.pop("reference_action")
        if native_rewards:
            if env.cfg.decimation != 4 or abs(env.step_dt - DT) > 1e-9:
                raise ValueError("Native capture requires the stock four-substep motor")
            manager = env.reward_manager
            names = list(manager.active_terms)
            weights = [float(manager.get_term_cfg(n).weight) for n in names]
            if (
                not names
                or len(set(names)) != len(names)
                or not np.isfinite(weights).all()
            ):
                raise ValueError("Invalid native reward terms")
            self.metadata.update(
                schema_version="operator_native_control_trace_v1",
                env_ids=(
                    list(range(env.num_envs)) if env_ids is None else env_ids.tolist()
                ),
                body_names=bodies,
                reward_names=names,
                reward_weights=weights,
                reward_contribution="Native weighted rate times step_dt, computed once by RewardManager; sum checked against reward_buf",
                omitted_training_objective="persistent_tilt impulse (gate disabled in evaluation), PPO entropy/value/retention losses; these are evaluation returns, not training returns",
                sampling="50 Hz post-step/pre-reset joint/contact/body state; four chronological 200 Hz computed/applied actuator torque commands (N m), not measured hardware torques",
                height_scan_post="Unclipped world hit z (m); nonfinite rays retained, not flat-support evidence",
                body_position_post="Body origins in world meters; NOT contact-point locations",
            )

    def snapshot(self, values):
        return _snapshot({name: value[self.env_ids] for name, value in values.items()})

    def before_step(self, observation, action):
        if self.pending is not None:
            raise RuntimeError("Previous diagnostic action lacks a pre-reset capture")
        if (
            observation.ndim != 2
            or observation.shape[1] != 48
            or action.shape != (len(observation), 12)
            or not all(torch.isfinite(x).all() for x in (observation, action))
        ):
            raise ValueError("Invalid diagnostic observation or action")
        robot = self.env.scene["robot"].data
        self.pending = self.snapshot(
            {
                "observation_pre": observation,
                "action": action,
                "joint_position_pre": robot.joint_pos,
                "joint_velocity_pre": robot.joint_vel,
            }
        )
        if self.reference is not None:
            with torch.inference_mode():
                reference_action = self.reference(observation)
            if (
                reference_action.shape != action.shape
                or not torch.isfinite(reference_action).all()
            ):
                self.pending = None
                raise ValueError("Invalid diagnostic reference action")
            self.pending.update(self.snapshot({"reference_action": reference_action}))
        self.substeps = []

    def after_substep(self):
        if self.native_rewards:
            if self.pending is None or len(self.substeps) >= 4:
                raise RuntimeError("Unpaired or extra actuator substep")
            # Native hook precedes scene.update: only explicit actuator commands
            # are current here, NOT cached post-physics body/joint/contact state.
            robot = self.env.scene["robot"].data
            self.substeps.append(
                self.snapshot(
                    {
                        "computed_torque_substeps": robot.computed_torque,
                        "applied_torque_substeps": robot.applied_torque,
                    }
                )
            )

    def after_step(self):
        if self.pending is None:
            raise RuntimeError("Post-physics diagnostics lack the delivered action")
        robot = self.env.scene["robot"].data
        sample = {
            **self.pending,
            **self.snapshot(
                {
                    "joint_position_post": robot.joint_pos,
                    "joint_velocity_post": robot.joint_vel,
                    "joint_position_target": robot.joint_pos_target,
                    "computed_torque": robot.computed_torque,
                    "applied_torque": robot.applied_torque,
                    "contact_normal_force_w": self.env.scene[
                        "contact_forces"
                    ].data.net_forces_w,
                    "foot_velocity_w": robot.body_lin_vel_w[:, self.foot_ids],
                }
            ),
        }
        if self.native_rewards:
            if len(self.substeps) != 4:
                raise RuntimeError("Missing actuator substeps")
            sample.update(
                self.snapshot(
                    {
                        **native_reward_snapshot(
                            self.env,
                            self.metadata["reward_names"],
                            self.metadata["reward_weights"],
                        ),
                        "body_position_post": robot.body_pos_w,
                        "height_scan_post": self.env.scene[
                            "height_scanner"
                        ].data.ray_hits_w[..., 2],
                    }
                )
            )
            for name in self.substeps[0]:
                sample[name] = np.stack([s[name] for s in self.substeps], axis=1)
            if any(
                not np.isfinite(v).all()
                for k, v in sample.items()
                if k != "height_scan_post"
            ):
                raise ValueError("Nonfinite native control data")
        self.samples.append(sample)
        self.pending = None

    def finish(self):
        if self.pending is not None or not self.samples:
            raise RuntimeError("Incomplete control diagnostic capture")
        return {
            key: np.stack([s[key] for s in self.samples]) for key in self.samples[0]
        }


def summarize_control_trace(control, trace, labels, metadata):
    """Validate pairing and summarize zero-command windows without bridging resets.

    Keep per-second bins to distinguish initial braking from later drift. Full
    arrays remain the source of truth; no causal diagnosis is assigned here.
    """
    steps, count, width = trace["command"].shape
    if width != 3 or count != len(labels) or steps == 0:
        raise ValueError("Invalid physical trace layout")
    shapes = {
        name: (steps, count, 12)
        for name in (
            "action",
            "reference_action",
            "joint_position_pre",
            "joint_velocity_pre",
            "joint_position_post",
            "joint_velocity_post",
            "joint_position_target",
            "computed_torque",
            "applied_torque",
        )
    }
    shapes.update(
        observation_pre=(steps, count, 48),
        contact_normal_force_w=(steps, count, len(metadata["contact_body_names"]), 3),
        foot_velocity_w=(steps, count, 4, 3),
    )
    if set(control) != set(shapes) or any(
        control[key].shape != shape or not np.isfinite(control[key]).all()
        for key, shape in shapes.items()
    ):
        raise ValueError("Missing, malformed or nonfinite control diagnostic arrays")
    if not np.array_equal(control["observation_pre"][:, :, 9:12], trace["command"]):
        raise ValueError("Diagnostic frame/physical command mismatch")
    done = trace["terminated"] | trace["time_out"]
    if done.shape != (steps, count) or done.dtype != np.bool_:
        raise ValueError("Invalid physical reset masks")
    continuing = ~done[:-1]
    if not np.array_equal(
        control["observation_pre"][1:, :, 36:48][continuing],
        control["action"][:-1][continuing],
    ):
        raise ValueError("Diagnostic previous-action timing mismatch")
    if not np.array_equal(
        control["joint_position_pre"][1:][continuing],
        control["joint_position_post"][:-1][continuing],
    ):
        raise ValueError("Diagnostic pre/post joint-state timing mismatch")
    foot_contacts = [
        metadata["contact_body_names"].index(name) for name in metadata["foot_names"]
    ]
    windows = []
    for env_id, label in enumerate(labels):
        terminal = np.flatnonzero(done[:, env_id])
        end = int(terminal[0]) + 1 if len(terminal) else steps
        zero = (trace["command"][:end, env_id] == 0).all(axis=1)
        boundaries = np.diff(np.r_[False, zero, False].astype(int))
        for start, stop in zip(
            np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
        ):
            bins = []
            for left in range(int(start), int(stop), round(1 / DT)):
                right = min(left + round(1 / DT), int(stop))
                part = slice(left, right)
                delta = (
                    control["action"][part, env_id]
                    - control["reference_action"][part, env_id]
                )
                tracking = (
                    control["joint_position_target"][part, env_id]
                    - control["joint_position_post"][part, env_id]
                )
                clipping = (
                    control["computed_torque"][part, env_id]
                    - control["applied_torque"][part, env_id]
                )
                bins.append(
                    {
                        "start_s": (left - int(start)) * DT,
                        "duration_s": (right - left) * DT,
                        "mean_body_yaw_rad_s": float(
                            trace["angular_velocity_b"][part, env_id, 2].mean()
                        ),
                        "mean_world_up_yaw_rad_s": float(
                            trace["angular_velocity_w"][part, env_id, 2].mean()
                        ),
                        "reference_action_rms_raw_per_joint": np.sqrt(
                            np.mean(delta**2, axis=0)
                        ).tolist(),
                        "joint_target_error_rms_rad_per_joint": np.sqrt(
                            np.mean(tracking**2, axis=0)
                        ).tolist(),
                        "torque_clipping_gap_max_nm_per_joint": np.abs(clipping)
                        .max(axis=0)
                        .tolist(),
                        "foot_mean_normal_force_z_n": control["contact_normal_force_w"][
                            part, env_id
                        ][:, foot_contacts, 2]
                        .mean(axis=0)
                        .tolist(),
                        "foot_max_speed_xy_m_s": np.linalg.norm(
                            control["foot_velocity_w"][part, env_id, :, :2], axis=-1
                        )
                        .max(axis=0)
                        .tolist(),
                    }
                )
            windows.append(
                {
                    "env_id": env_id,
                    "profile": label,
                    "start_s": int(start) * DT,
                    "duration_s": int(stop - start) * DT,
                    "ends_in_physical_reset": bool(done[int(stop) - 1, env_id]),
                    "bins": bins,
                }
            )
    return {
        "schema_version": VERSION,
        "status": "CAPTURE_VALID",
        "control_steps": steps,
        "metadata": metadata,
        "zero_command_windows": windows,
        "scope": "paired diagnostics, NOT a behavioral pass or proof of root cause; exclude all frames after the first physical reset per trial",
    }


def validate_control_artifacts(output, report, reference, learner_sha256):
    """Fail closed if a nominally successful worker omitted the requested capture."""
    diagnostic = report.get("control_diagnostics", {})
    summary = json.loads((output / "control_report.json").read_text())
    interface = json.loads((output / "control_interface.json").read_text())
    if not all(isinstance(value, dict) for value in (diagnostic, summary, interface)):
        raise ValueError("Diagnostic metadata must be JSON objects")
    expected_hashes = {
        "learner_checkpoint": learner_sha256,
        "physical_trace": file_sha256(output / "trace.npz"),
        "control_trace": file_sha256(output / "control_trace.npz"),
        "interface": file_sha256(output / "control_interface.json"),
    }
    if (
        diagnostic.get("status") != "CAPTURE_VALID"
        or diagnostic.get("sha256") != file_sha256(output / "control_report.json")
        or diagnostic.get("reference") != reference
        or summary.get("status") != "CAPTURE_VALID"
        or summary.get("schema_version") != VERSION
        or summary.get("reference") != reference
        or summary.get("sha256") != expected_hashes
        or interface.get("reference") != reference
        or interface.get("learner_sha256") != learner_sha256
    ):
        raise ValueError("Diagnostic artifact/reference identity mismatch")
