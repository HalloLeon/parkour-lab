# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a restricted student from a frozen privileged teacher online."""

# Launch Isaac Sim before importing modules that depend on it.

import argparse
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(
    description="Run student-driven online action distillation."
)
parser.add_argument(
    "--teacher_checkpoint",
    type=str,
    required=True,
    help="Path to the exact privileged-teacher checkpoint used for labels.",
)
parser.add_argument(
    "--student_checkpoint",
    type=str,
    default=None,
    help="Student checkpoint to resume. Its frozen teacher identity must match.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of parallel student-driven environments.",
)
parser.add_argument(
    "--task", type=str, default="Parkour-Lab-v0", help="Adaptive training task name."
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Privileged PPO teacher configuration entry point.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="Environment and student initialization seed.",
)
parser.add_argument(
    "--max_iterations",
    type=int,
    default=300,
    help="Number of collect-and-update iterations.",
)
parser.add_argument(
    "--steps_per_iteration",
    type=int,
    default=24,
    help="Student rollout steps per environment.",
)
parser.add_argument(
    "--learning_epochs",
    type=int,
    default=4,
    help="Supervised passes over each online rollout.",
)
parser.add_argument(
    "--num_mini_batches", type=int, default=4, help="Mini-batches per supervised epoch."
)
parser.add_argument(
    "--learning_rate", type=float, default=1.0e-3, help="Student Adam learning rate."
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="Adam weight decay applied to the student parameters.",
)
parser.add_argument(
    "--max_grad_norm", type=float, default=1.0, help="Maximum student gradient norm."
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=50,
    help="Iterations between student checkpoints.",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="Output directory. Defaults to a timestamped directory below logs/distillation/parkour_lab.",
)
parser.add_argument(
    "--allow_zero_exteroception",
    action="store_true",
    default=False,
    help=(
        "Allow the current information-free exteroception stub. Such a run only validates the "
        "distillation pipeline and cannot train a terrain-aware student."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

for argument_name in (
    "max_iterations",
    "steps_per_iteration",
    "learning_epochs",
    "num_mini_batches",
    "save_interval",
):
    if getattr(args_cli, argument_name) <= 0:
        parser.error(f"--{argument_name} must be positive.")
if args_cli.learning_rate <= 0.0:
    parser.error("--learning_rate must be positive.")
if args_cli.weight_decay < 0.0:
    parser.error("--weight_decay must be non-negative.")
if args_cli.max_grad_norm <= 0.0:
    parser.error("--max_grad_norm must be positive.")
if args_cli.num_envs is not None and args_cli.num_envs <= 0:
    parser.error("--num_envs must be positive.")

# ``hydra_task_config`` reads the global ``sys.argv`` when the decorated
# ``main`` is called later. Leave it only the script name and unparsed Hydra
# overrides, excluding options already consumed by argparse and AppLauncher.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# The remaining imports require the running simulation application.

import json
import os
from collections.abc import Callable
from datetime import datetime

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import parkour_lab.tasks  # noqa: F401
import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from parkour_lab.learning.distillation.contracts import (
    TEACHER_OBSERVATION_GROUPS,
    TeacherCheckpoint,
    assert_teacher_interface_matches,
    build_teacher_interface,
    interface_sha256,
    load_teacher_checkpoint,
    write_json,
)
from parkour_lab.learning.distillation.student import (
    STUDENT_OBSERVATION_GROUPS,
    StudentModelCfg,
    StudentPolicy,
    compute_distillation_losses,
)
from parkour_lab.tasks.manager_based.parkour_lab.mdp.observations import (
    student_exteroception_stub,
)
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

STUDENT_CHECKPOINT_VERSION = 2
"""Serialization version of student checkpoints written by this script."""


# Capture the task ID and agent entry-point name now. The returned wrapper later
# loads their registered configuration defaults, lets Hydra consume the retained
# ``sys.argv`` overrides, and calls this function as ``main(env_cfg, agent_cfg)``.
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Run student-driven online distillation from one exact teacher checkpoint."""

    teacher_checkpoint = load_teacher_checkpoint(args_cli.teacher_checkpoint)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    else:
        env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    else:
        # Keep simulation, teacher, and student tensors on one device.
        agent_cfg.device = env_cfg.sim.device

    synchronize_curriculum = getattr(env_cfg, "synchronize_curriculum_config", None)
    if callable(synchronize_curriculum):
        synchronize_curriculum()

    log_dir = (
        os.path.abspath(os.path.expanduser(args_cli.log_dir))
        if args_cli.log_dir is not None
        else os.path.abspath(
            os.path.join(
                "logs",
                "distillation",
                "parkour_lab",
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
        )
    )
    os.makedirs(log_dir, exist_ok=False)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        observations = env.get_observations()
        runtime_teacher_interface = build_teacher_interface(
            env.unwrapped, observations, agent_cfg
        )
        assert_teacher_interface_matches(
            teacher_checkpoint.teacher_interface,
            runtime_teacher_interface,
            context="Online-distillation runtime",
        )
        required_groups = (
            *TEACHER_OBSERVATION_GROUPS,
            *STUDENT_OBSERVATION_GROUPS,
            "heading_target",
        )
        information_dimensions = {
            group_name: int(observations[group_name].shape[-1])
            for group_name in required_groups
        }

        exteroception_cfg = env.unwrapped.cfg.observations.student_exteroception
        uses_exteroception_stub = any(
            getattr(exteroception_cfg, term_name).func is student_exteroception_stub
            for term_name in env.unwrapped.observation_manager.active_terms[
                "student_exteroception"
            ]
        )
        if uses_exteroception_stub and not args_cli.allow_zero_exteroception:
            raise RuntimeError(
                "Student exteroception is the constant-zero tutorial stub. "
                "Use --allow_zero_exteroception only for a pipeline smoke test."
            )

        training_scope = (
            "zero_exteroception_pipeline_smoke_test"
            if uses_exteroception_stub
            else "perceptive_student_distillation"
        )
        action_dim = env.unwrapped.action_manager.total_action_dim
        teacher_dim = sum(
            information_dimensions[name] for name in TEACHER_OBSERVATION_GROUPS
        )
        student_dim = sum(
            information_dimensions[name] for name in STUDENT_OBSERVATION_GROUPS
        )

        print(
            "[DISTILL] Frozen teacher: "
            f"{teacher_checkpoint.checkpoint_path} "
            f"({teacher_checkpoint.checkpoint_sha256[:12]})"
        )
        print(
            "[DISTILL] Teacher actor groups: policy -> terrain "
            f"({information_dimensions['policy']} + {information_dimensions['terrain']} = "
            f"{teacher_dim})"
        )
        print(
            "[DISTILL] Student input groups: student_policy -> student_exteroception "
            f"({information_dimensions['student_policy']} + "
            f"{information_dimensions['student_exteroception']} = {student_dim})"
        )
        print(
            "[DISTILL] Oracle heading is supervision only; physics is driven only by the student."
        )
        if uses_exteroception_stub:
            print(
                "[DISTILL] WARNING: Student exteroception is the constant-zero tutorial stub. "
                "This run is a pipeline smoke test, not terrain-aware student training."
            )

        teacher_policy = _load_teacher_policy(
            env,
            agent_cfg,
            checkpoint_path=teacher_checkpoint.checkpoint_path,
        )

        student_cfg = StudentModelCfg(
            state_dim=observations["student_policy"].shape[-1],
            exteroception_dim=observations["student_exteroception"].shape[-1],
            action_dim=action_dim,
        )

        # Create the student neural network from the resolved input and action
        # dimensions, then place its parameters on the simulation device.
        student = StudentPolicy(student_cfg).to(env.device)

        # Adam updates only the student parameters. ``weight_decay`` adds an
        # optional penalty for large weights; its zero default disables it.
        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=args_cli.learning_rate,
            weight_decay=args_cli.weight_decay,
        )
        start_iteration = 0
        if args_cli.student_checkpoint is not None:
            start_iteration = _load_student_checkpoint(
                args_cli.student_checkpoint,
                student=student,
                optimizer=optimizer,
                teacher_checkpoint=teacher_checkpoint,
            )
            if start_iteration >= args_cli.max_iterations:
                raise ValueError(
                    "--max_iterations must exceed the resumed student checkpoint iteration "
                    f"({start_iteration})."
                )

        write_json(
            os.path.join(log_dir, "teacher_checkpoint.json"),
            teacher_checkpoint.to_dict(),
        )
        write_json(
            os.path.join(log_dir, "distillation_config.json"),
            {
                "student_checkpoint_version": STUDENT_CHECKPOINT_VERSION,
                "runtime_group_dimensions": information_dimensions,
                "student_observation_group_order": list(STUDENT_OBSERVATION_GROUPS),
                "student_observation_dimension": student_dim,
                "student_model": student_cfg.to_dict(),
                "optimizer": {
                    "name": "Adam",
                    "learning_rate": args_cli.learning_rate,
                    "weight_decay": args_cli.weight_decay,
                    "max_grad_norm": args_cli.max_grad_norm,
                },
                "rollout": {
                    "driver": "student",
                    "steps_per_iteration": args_cli.steps_per_iteration,
                    "learning_epochs": args_cli.learning_epochs,
                    "num_mini_batches": args_cli.num_mini_batches,
                },
                "training_scope": training_scope,
                "teacher_runtime_interface_sha256": interface_sha256(
                    runtime_teacher_interface
                ),
            },
        )
        dump_yaml(os.path.join(log_dir, "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "teacher_agent.yaml"), agent_cfg)

        _run_training(
            env,
            observations,
            teacher_policy,
            student=student,
            optimizer=optimizer,
            start_iteration=start_iteration,
            log_dir=log_dir,
            teacher_checkpoint=teacher_checkpoint,
            training_scope=training_scope,
        )
    finally:
        env.close()


def _collect_rollout(
    env: RslRlVecEnvWrapper,
    observations: TensorDict,
    student: StudentPolicy,
    teacher_policy: Callable[[TensorDict], torch.Tensor],
    steps: int,
) -> tuple[
    TensorDict,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    dict[str, float | int],
]:
    """Collect labels and actions from states visited by the student."""

    student.eval()
    rollout_state: list[torch.Tensor] = []
    rollout_exteroception: list[torch.Tensor] = []
    rollout_heading_target: list[torch.Tensor] = []
    rollout_teacher_action: list[torch.Tensor] = []
    reward_sum = 0.0
    done_count = 0
    action_l2_sum = 0.0
    action_l2_max = 0.0
    action_l2_count = 0

    # Both labels and student actions are computed from the same
    # student-visited state. Only the student action advances physics.
    for _ in range(steps):
        with torch.inference_mode():
            student_state = observations["student_policy"]
            exteroception = observations["student_exteroception"]
            heading_target = observations["heading_target"]
            teacher_action = teacher_policy(observations)

            # Produce the deterministic motor command from only the student's
            # restricted state and exteroception. This action, rather than the
            # teacher label, is used to advance the environment below.
            student_action = student.act_inference(student_state, exteroception)
            if not torch.isfinite(student_action).all():
                raise RuntimeError("The student produced a non-finite action.")
            action_l2 = torch.linalg.norm(student_action - teacher_action, dim=-1)

        # Clone observation-backed tensors because the simulator may reuse its
        # buffers when the environment advances to the next state.
        rollout_state.append(student_state.clone())
        rollout_exteroception.append(exteroception.clone())
        rollout_heading_target.append(heading_target.clone())
        rollout_teacher_action.append(teacher_action.clone())
        action_l2_sum += float(action_l2.sum().item())
        action_l2_max = max(action_l2_max, float(action_l2.max().item()))
        action_l2_count += int(action_l2.numel())

        observations, rewards, dones, _ = env.step(student_action)
        reward_sum += float(rewards.mean().item())
        done_count += int(torch.count_nonzero(dones).item())

    batches = (
        torch.cat(rollout_state, dim=0),
        torch.cat(rollout_exteroception, dim=0),
        torch.cat(rollout_heading_target, dim=0),
        torch.cat(rollout_teacher_action, dim=0),
    )
    for batch_name, batch in zip(
        ("student state", "student exteroception", "heading target", "teacher action"),
        batches,
    ):
        if not torch.isfinite(batch).all():
            raise RuntimeError(
                f"The collected {batch_name} batch contains non-finite values."
            )

    return (
        observations,
        batches,
        {
            "mean_step_reward": reward_sum / steps,
            "completed_episodes": done_count,
            "rollout_action_l2_mean": action_l2_sum / action_l2_count,
            "rollout_action_l2_max": action_l2_max,
        },
    )


def _load_student_checkpoint(
    path: str,
    *,
    student: StudentPolicy,
    optimizer: torch.optim.Optimizer,
    teacher_checkpoint: TeacherCheckpoint,
) -> int:
    """Resume when model dimensions and the exact teacher contract match."""

    # Expand paths beginning with ``~`` and make the result independent of the
    # process's working directory before opening the checkpoint.
    checkpoint_path = os.path.abspath(os.path.expanduser(path))

    # Load tensors directly onto the student's device. ``weights_only=True``
    # restricts deserialization to tensors and basic data structures instead
    # of allowing arbitrary Python objects from the checkpoint to execute.
    checkpoint = torch.load(
        # File containing the dictionary previously written by ``torch.save``.
        checkpoint_path,
        # Place every restored tensor on the same device as the student. The
        # first parameter is representative because the whole model was moved
        # to one device when it was created.
        map_location=next(student.parameters()).device,
        # Use PyTorch's restricted unpickler instead of reconstructing
        # arbitrary Python objects stored in the file.
        weights_only=True,
    )

    # Reject formats whose stored fields may have different meanings.
    if checkpoint.get("checkpoint_version") != STUDENT_CHECKPOINT_VERSION:
        raise ValueError(
            "Student checkpoint uses an unsupported serialization version."
        )

    # The saved tensors can only be restored into the same student network
    # architecture and loss configuration.
    if checkpoint.get("student_config") != student.cfg.to_dict():
        raise ValueError(
            "Student checkpoint uses a different student model configuration."
        )

    # Continuing with labels from different teacher weights would silently
    # change the supervised target midway through student training. The
    # interface hash also prevents reinterpreting the same weights with changed
    # observation, action, terrain, or timing semantics.
    saved_teacher = checkpoint["teacher_checkpoint"]
    if (
        saved_teacher["checkpoint_sha256"] != teacher_checkpoint.checkpoint_sha256
        or saved_teacher["teacher_interface_sha256"]
        != teacher_checkpoint.teacher_interface_sha256
    ):
        raise ValueError(
            "Student checkpoint was trained against different teacher weights or interface."
        )

    # Restore both the learned student weights and Adam's running state, then
    # continue with the next training iteration after the saved one.
    student.load_state_dict(checkpoint["student_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["iteration"])


def _load_teacher_policy(
    env: RslRlVecEnvWrapper,
    agent_cfg: RslRlBaseRunnerCfg,
    *,
    checkpoint_path: str,
) -> Callable[[TensorDict], torch.Tensor]:
    """Load the PPO teacher and return its deterministic inference callable."""

    # Construct the original PPO actor-critic so checkpoint loading and actor
    # preprocessing exactly match teacher training.
    teacher_runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=env.device,
    )
    teacher_runner.load(checkpoint_path, load_optimizer=False)

    # RSL-RL switches the policy to evaluation mode and returns its
    # deterministic ``act_inference`` method.
    return teacher_runner.get_inference_policy(device=env.device)


def _run_training(
    env: RslRlVecEnvWrapper,
    observations: TensorDict,
    teacher_policy: Callable[[TensorDict], torch.Tensor],
    *,
    student: StudentPolicy,
    optimizer: torch.optim.Optimizer,
    start_iteration: int,
    log_dir: str,
    teacher_checkpoint: TeacherCheckpoint,
    training_scope: str,
) -> None:
    """Alternate student-driven collection with supervised updates."""

    metrics_path = os.path.join(log_dir, "metrics.jsonl")
    checkpoint_manifest = teacher_checkpoint.to_dict()
    for iteration in range(start_iteration, args_cli.max_iterations):
        observations, batches, rollout_metrics = _collect_rollout(
            env,
            observations,
            student,
            teacher_policy,
            args_cli.steps_per_iteration,
        )
        losses = _update_student(
            student,
            optimizer,
            batches=batches,
            learning_epochs=args_cli.learning_epochs,
            num_mini_batches=args_cli.num_mini_batches,
            max_grad_norm=args_cli.max_grad_norm,
        )
        metrics = {
            "iteration": iteration + 1,
            "samples": batches[0].shape[0],
            **rollout_metrics,
            **losses,
        }
        with open(metrics_path, "a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            f"[DISTILL] {iteration + 1}/{args_cli.max_iterations} "
            f"motor={losses['motor_huber']:.6f} "
            f"heading={losses['heading_error_deg']:.2f} deg "
            f"action_l2={metrics['rollout_action_l2_mean']:.4f} "
            f"reward={metrics['mean_step_reward']:.4f}"
        )

        if (iteration + 1) % args_cli.save_interval == 0:
            _save_student_checkpoint(
                os.path.join(log_dir, f"student_{iteration + 1:06d}.pt"),
                student=student,
                optimizer=optimizer,
                iteration=iteration + 1,
                teacher_checkpoint=checkpoint_manifest,
                training_scope=training_scope,
            )

    _save_student_checkpoint(
        os.path.join(log_dir, "student_final.pt"),
        student=student,
        optimizer=optimizer,
        iteration=args_cli.max_iterations,
        teacher_checkpoint=checkpoint_manifest,
        training_scope=training_scope,
    )


def _save_student_checkpoint(
    path: str,
    *,
    student: StudentPolicy,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    teacher_checkpoint: dict[str, object],
    training_scope: str,
) -> None:
    """Save student weights without duplicating the frozen teacher weights."""

    torch.save(
        {
            "checkpoint_version": STUDENT_CHECKPOINT_VERSION,
            "iteration": iteration,
            "student_config": student.cfg.to_dict(),
            "student_state_dict": student.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "teacher_checkpoint": teacher_checkpoint,
            "training_scope": training_scope,
        },
        path,
    )


def _update_student(
    student: StudentPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    batches: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    learning_epochs: int,
    num_mini_batches: int,
    max_grad_norm: float,
) -> dict[str, float]:
    """Optimize the student on labels from its newly visited states."""

    # Enable training behavior for modules that distinguish between training
    # and evaluation, then read the number of collected state-label samples.
    student.train()
    num_samples = batches[0].shape[0]

    # Sum diagnostics over every optimizer update. They are divided by the
    # update count before being returned for iteration-level logging.
    accumulated = {
        "total": 0.0,
        "motor_huber": 0.0,
        "heading_direction": 0.0,
        "heading_norm": 0.0,
        "heading_error_deg": 0.0,
    }
    updates = 0

    # Revisit the freshly collected rollout for the configured number of
    # supervised learning epochs.
    for _ in range(learning_epochs):
        # Randomize sample order independently in each epoch so mini-batches do
        # not preserve rollout-time or parallel-environment ordering.
        permutation = torch.randperm(num_samples, device=batches[0].device)

        # Split the shuffled indices into approximately equal mini-batches.
        # Capping the number of chunks prevents empty batches when there are
        # fewer samples than requested mini-batches.
        for indices in torch.chunk(permutation, min(num_mini_batches, num_samples)):
            # Select aligned student states, exteroception, heading targets,
            # and teacher-action labels, then evaluate all supervised losses.
            losses = compute_distillation_losses(
                student,
                batches[0][indices],
                batches[1][indices],
                batches[2][indices],
                batches[3][indices],
            )

            # Clear gradients from the preceding mini-batch, backpropagate the
            # combined loss, bound unusually large gradients, and update only
            # the student parameters managed by Adam.
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer.step()

            # Detach scalar diagnostics from autograd and move their values
            # into ordinary Python accumulators used only for reporting.
            for name in accumulated:
                accumulated[name] += float(losses[name].detach().item())
            updates += 1

    # Report the mean value of each diagnostic across all mini-batch updates.
    return {name: value / updates for name, value in accumulated.items()}


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
