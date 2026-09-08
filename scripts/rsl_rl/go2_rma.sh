#!/usr/bin/env bash
# Focused Go2 RMA workflow. Run from the repository root in the Isaac Lab env.
set -euo pipefail

PARKOUR_MODE="${1:?Usage: bash scripts/rsl_rl/go2_rma.sh train|recover|check-step|check|teleop CHECKPOINT [extra arguments]}"
PARKOUR_CHECKPOINT="${2:?Pass an explicit checkpoint path}"
shift 2
if [[ ! -f "$PARKOUR_CHECKPOINT" ]]; then
  echo "Checkpoint not found: $PARKOUR_CHECKPOINT" >&2
  exit 1
fi

PARKOUR_COMMON=(
  --task=Parkour-Lab-v0
  --seed=42
  '--kit_args=--/physics/collisionApproximateCylinders=true'
)

case "$PARKOUR_MODE" in
  train)
    python scripts/rsl_rl/train.py "${PARKOUR_COMMON[@]}" \
      --headless --livestream=0 \
      --warm_start_velocity="$PARKOUR_CHECKPOINT" \
      --run_name=go2_rma_velocity --max_iterations=500 --num_envs=4096 \
      --domain_randomization_stage=off --reset_profile=jitter \
      --logger=tensorboard \
      env.commands.intent.pivot_window_probability=0.30 \
      'env.commands.intent.long_stop_window_range_s=[2.0,4.0]' \
      env.rewards.stationary_velocity_tracking.params.pivot_yaw_tracking_weight=2.0 \
      agent.algorithm.history_rollout_interval=4 \
      agent.algorithm.learning_rate=0.0001 \
      agent.algorithm.max_learning_rate=0.0001 \
      agent.algorithm.entropy_coef=0.003 \
      "$@"
    ;;
  recover)
    # Continue a v21 checkpoint, including optimizer/update counters. Only the
    # high-step mastery frontier/evidence restarts; other families keep theirs.
    python scripts/rsl_rl/train.py "${PARKOUR_COMMON[@]}" \
      --headless --livestream=0 --resume --checkpoint="$PARKOUR_CHECKPOINT" \
      --reset_curriculum_family=high_step --reset_curriculum_level=1 \
      --run_name=go2_rma_high_step_recovery --max_iterations=300 --num_envs=4096 \
      --domain_randomization_stage=off --reset_profile=jitter \
      --logger=tensorboard \
      env.commands.intent.pivot_window_probability=0.30 \
      'env.commands.intent.long_stop_window_range_s=[2.0,4.0]' \
      env.rewards.stationary_velocity_tracking.params.pivot_yaw_tracking_weight=2.0 \
      agent.algorithm.history_rollout_interval=4 \
      agent.algorithm.learning_rate=0.0001 \
      agent.algorithm.max_learning_rate=0.0001 \
      agent.algorithm.entropy_coef=0.003 \
      "$@"
    ;;
  check-step)
    # Start with three complete easy high-step episodes, not a full sweep.
    # Append --difficulty_level=6 only after the easy approach/traversal passes.
    python scripts/rsl_rl/play.py "${PARKOUR_COMMON[@]}" \
      --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT" \
      --num_envs=1 --eval_episodes=3 --reset_profile=jitter \
      --policy_mode=history_mean --geometry_variant=0 \
      --terrain_family=high_step --difficulty_level=1 --desired_speed=0.55 \
      --command_profile=translation_only --desired_yaw_rate=0 "$@"
    ;;
  check)
    # 21 complete episodes total: three per case, no video, one environment.
    # Do not use first-completed episodes from many async environments: that
    # can overrepresent short failures in a small screening run.
    # Batch evaluation needs no GUI or streaming client.
    PARKOUR_EVAL=("${PARKOUR_COMMON[@]}" --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT"
      --num_envs=1 --eval_episodes=3 --reset_profile=jitter
      --policy_mode=history_mean --geometry_variant=0)
    python scripts/rsl_rl/play.py "${PARKOUR_EVAL[@]}" \
      --terrain_family=tilted_ramps --difficulty_level=0 --desired_speed=0.55 \
      --command_profile=stop_restart --desired_yaw_rate=0 --telemetry "$@"
    for PARKOUR_YAW in -0.5 0.5; do
      python scripts/rsl_rl/play.py "${PARKOUR_EVAL[@]}" \
        --terrain_family=tilted_ramps --difficulty_level=0 --desired_speed=0.55 \
        --command_profile=pivot_restart --desired_yaw_rate="$PARKOUR_YAW" --telemetry "$@"
    done
    for PARKOUR_FAMILY in high_step gap hurdle tilted_ramps; do
      case "$PARKOUR_FAMILY" in
        high_step) PARKOUR_SPEED=0.55 ;;
        hurdle) PARKOUR_SPEED=0.65 ;;
        *) PARKOUR_SPEED=0.60 ;;
      esac
      python scripts/rsl_rl/play.py "${PARKOUR_EVAL[@]}" \
        --terrain_family="$PARKOUR_FAMILY" --difficulty_level=6 \
        --desired_speed="$PARKOUR_SPEED" --desired_yaw_rate=0 \
        --command_profile=translation_only "$@"
    done
    ;;
  teleop)
    python scripts/rsl_rl/play.py "${PARKOUR_COMMON[@]}" \
      --livestream=2 --checkpoint="$PARKOUR_CHECKPOINT" --teleop \
      --terrain_family=tilted_ramps --difficulty_level=0 "$@"
    ;;
  *)
    echo "Unknown mode: $PARKOUR_MODE (use train, recover, check-step, check or teleop)." >&2
    exit 2
    ;;
esac
