#!/usr/bin/env bash
# Focused Go2 RMA workflow. Run from the repository root in the Isaac Lab env.
set -euo pipefail

PARKOUR_MODE="${1:?Usage: bash scripts/rsl_rl/go2_rma.sh train|recover|resume|repair|diagnose|startup|check-repair|check-step|check-control|check|teleop CHECKPOINT [extra arguments]}"
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

# Isolated reward ablations, plus an explicitly combined control repair. Resolved
# weights/parameters are archived in env.yaml and evaluation_reward_config.
PARKOUR_DEFAULT_PROFILE=baseline
if [[ "$PARKOUR_MODE" == repair ]]; then
  PARKOUR_DEFAULT_PROFILE=control
fi
PARKOUR_REWARD_PROFILE="${PARKOUR_REWARD_PROFILE:-$PARKOUR_DEFAULT_PROFILE}"
case "$PARKOUR_REWARD_PROFILE" in
  baseline) ;; # Existing objective, with the signed-progress bug corrected.
  stationary)
    PARKOUR_COMMON+=(env.rewards.stationary_planar_motion.weight=-0.5)
    ;;
  control)
    PARKOUR_COMMON+=(
      env.rewards.stationary_velocity_tracking.params.pivot_yaw_objective=signed_progress
      env.rewards.stationary_velocity_tracking.params.pivot_yaw_overspeed_weight=1.0
      env.rewards.stationary_planar_motion.weight=-0.5)
    ;;
  support)
    PARKOUR_COMMON+=(
      env.rewards.flat_orientation_l2.weight=0.0
      env.rewards.stable_orientation_l2.weight=0.0
      env.rewards.upright_orientation_l2.weight=-0.25
      env.rewards.supported_orientation_l2.weight=-1.0)
    ;;
  vertical)
    PARKOUR_COMMON+=(
      env.rewards.lin_vel_z_l2.weight=0.0
      env.rewards.supported_vertical_velocity_l2.weight=-0.5)
    ;;
  corridor)
    PARKOUR_COMMON+=(
      env.rewards.route_cross_track_excess.params.normalize_by_margin=true
      env.rewards.route_cross_track_excess.weight=-0.25)
    ;;
  failure)
    PARKOUR_COMMON+=(
      env.rewards.chassis_contact.weight=0.0
      env.rewards.off_route_failure.weight=0.0
      env.rewards.physical_failure.weight=-10.0)
    ;;
  *)
    echo "Unknown PARKOUR_REWARD_PROFILE: $PARKOUR_REWARD_PROFILE (use baseline, control, stationary, support, vertical, corridor or failure)." >&2
    exit 2
    ;;
esac

case "$PARKOUR_MODE" in
  diagnose)
    # Two bounded first-episode traces, not another training run or success
    # sweep. Preserve the failed screen's checkpoint, jitter and mean actions.
    # An optional exact metrics path verifies archived configuration hashes.
    if (( $# )); then
      echo "diagnose has fixed capture settings and accepts no extra arguments. Use PARKOUR_DIAGNOSTIC_METRICS for an exact existing metrics.json." >&2
      exit 2
    fi
    PARKOUR_CHECKPOINT_DIR="$(dirname -- "$PARKOUR_CHECKPOINT")"
    PARKOUR_DIAGNOSTIC_DIR="$(mktemp -d "$PARKOUR_CHECKPOINT_DIR/failure_diagnostics_XXXXXX")"
    echo "Failure diagnostics: $PARKOUR_DIAGNOSTIC_DIR"
    PARKOUR_RECIPE_ARGS=(--verify --json)
    if [[ -n "${PARKOUR_DIAGNOSTIC_METRICS:-}" ]]; then
      PARKOUR_RECIPE_ARGS+=(--metrics="$PARKOUR_DIAGNOSTIC_METRICS")
    fi
    python scripts/rsl_rl/training_recipe_report.py "$PARKOUR_CHECKPOINT" \
      "${PARKOUR_RECIPE_ARGS[@]}" > "$PARKOUR_DIAGNOSTIC_DIR/training_recipe.json"
    PARKOUR_CAPTURE=("${PARKOUR_COMMON[@]}" --headless --livestream=0
      --checkpoint="$PARKOUR_CHECKPOINT" --startup_diagnostics --no-screen
      --num_envs=1 --eval_episodes=1 --reset_profile=jitter
      --policy_mode=history_mean --geometry_variant=0 --desired_speed=0.55)
    python scripts/rsl_rl/play.py "${PARKOUR_CAPTURE[@]}" \
      --terrain_family=high_step --difficulty_level=6 --desired_yaw_rate=0 \
      --command_profile=translation_only --startup_steps=100 \
      --startup_output_dir="$PARKOUR_DIAGNOSTIC_DIR/high_step_L6"
    python scripts/rsl_rl/play.py "${PARKOUR_CAPTURE[@]}" \
      --terrain_family=tilted_ramps --difficulty_level=0 --desired_yaw_rate=0.5 \
      --command_profile=pivot_restart --startup_steps=250 \
      --startup_output_dir="$PARKOUR_DIAGNOSTIC_DIR/pivot_positive"
    for PARKOUR_CASE in high_step_L6 pivot_positive; do
      python scripts/rsl_rl/failure_trace_report.py \
        "$PARKOUR_DIAGNOSTIC_DIR/$PARKOUR_CASE/startup_diagnostics.json" --json \
        > "$PARKOUR_DIAGNOSTIC_DIR/$PARKOUR_CASE/trace_summary.json"
    done
    echo "Captured at most 350 control steps total; these are failure traces, not an acceptance PASS."
    echo "Inspect training_recipe.json and both trace_summary.json files under $PARKOUR_DIAGNOSTIC_DIR"
    ;;
  repair)
    if [[ "$PARKOUR_REWARD_PROFILE" != control ]]; then
      echo "repair requires PARKOUR_REWARD_PROFILE=control (unset it to use the repair default)." >&2
      exit 2
    fi
    # Bounded v21 continuation, not a new architecture/warm start. Keep saved
    # optimizer, curriculum frontiers, entropy and exploration schedules. Jitter
    # covers the failing startup distribution; longer, more frequent pivots and
    # on-policy history rollouts cover sustained operator control. This is a
    # combined repair, not a causal single-variable ablation.
    python scripts/rsl_rl/train.py "${PARKOUR_COMMON[@]}" \
      --resume --checkpoint="$PARKOUR_CHECKPOINT" \
      --run_name=go2_rma_control_repair --max_iterations=100 --num_envs=4096 \
      --domain_randomization_stage=off --reset_profile=jitter --logger=tensorboard \
      env.commands.intent.pivot_window_probability=0.30 \
      'env.commands.intent.pivot_window_range_s=[2.0,4.0]' \
      agent.algorithm.history_rollout_interval=4 \
      agent.algorithm.learning_rate=0.0001 \
      agent.algorithm.max_learning_rate=0.0001 \
      "$@" --headless --livestream=0
    ;;
  check-repair)
    # Four fixed history-mean cases, three complete episodes each. Collect all
    # behavioral failures, then fail the aggregate gate. No GUI/video startup.
    for PARKOUR_ARG in "$@"; do
      case "$PARKOUR_ARG" in
        --video|--video=*|--video_length*|--teleop*|--startup*|--enable_cameras*|--domain_randomization_stage*)
          echo "check-repair is a fixed metrics-only screen; incompatible argument: $PARKOUR_ARG" >&2
          exit 2
          ;;
      esac
    done
    PARKOUR_CHECKPOINT_DIR="$(dirname -- "$PARKOUR_CHECKPOINT")"
    PARKOUR_REPAIR_DIR="$(mktemp -d "$PARKOUR_CHECKPOINT_DIR/control_screen_XXXXXX")"
    PARKOUR_EVAL_SEED="${PARKOUR_EVAL_SEED:-42}"
    echo "Control screen artifacts: $PARKOUR_REPAIR_DIR"
    PARKOUR_EVAL=("${PARKOUR_COMMON[@]}"
      --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT"
      --num_envs=1 --eval_episodes=3 --reset_profile=jitter --no-screen
      --policy_mode=history_mean --geometry_variant=0 --seed="$PARKOUR_EVAL_SEED"
      --video_output_dir="$PARKOUR_REPAIR_DIR")
    python scripts/rsl_rl/play.py "$@" "${PARKOUR_EVAL[@]}" \
      --terrain_family=high_step --difficulty_level=6 --desired_speed=0.55 \
      --command_profile=translation_only --desired_yaw_rate=0
    python scripts/rsl_rl/play.py "$@" "${PARKOUR_EVAL[@]}" \
      --terrain_family=tilted_ramps --difficulty_level=0 --desired_speed=0.55 \
      --command_profile=stop_restart --desired_yaw_rate=0 --telemetry
    for PARKOUR_YAW in -0.5 0.5; do
      python scripts/rsl_rl/play.py "$@" "${PARKOUR_EVAL[@]}" \
        --terrain_family=tilted_ramps --difficulty_level=0 --desired_speed=0.55 \
        --command_profile=pivot_restart --desired_yaw_rate="$PARKOUR_YAW" --telemetry
    done
    python scripts/rsl_rl/control_repair_report.py "$PARKOUR_REPAIR_DIR" \
      --checkpoint="$PARKOUR_CHECKPOINT" --seed="$PARKOUR_EVAL_SEED"
    ;;
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
  recover|resume)
    # Continue a v21 checkpoint, including optimizer/update counters. Recovery
    # restarts only high-step frontier/evidence; resume retains all frontiers.
    PARKOUR_RESTART=(--resume --checkpoint="$PARKOUR_CHECKPOINT")
    PARKOUR_RUN_NAME="go2_rma_resume_${PARKOUR_REWARD_PROFILE}"
    if [[ "$PARKOUR_MODE" == recover ]]; then
      PARKOUR_RESTART+=(--reset_curriculum_family=high_step --reset_curriculum_level=1)
      PARKOUR_RUN_NAME="go2_rma_high_step_recovery_${PARKOUR_REWARD_PROFILE}"
    fi
    python scripts/rsl_rl/train.py "${PARKOUR_COMMON[@]}" \
      --headless --livestream=0 \
      "${PARKOUR_RESTART[@]}" \
      --run_name="$PARKOUR_RUN_NAME" --max_iterations=300 --num_envs=4096 \
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
  startup)
    # Compare the first causal inputs on a canonical flat tile and an easy
    # step. Fresh processes avoid sharing an Isaac stage between geometries.
    # This is a bounded diagnostic, not an episode-success evaluation.
    PARKOUR_CHECKPOINT_DIR="$(dirname -- "$PARKOUR_CHECKPOINT")"
    PARKOUR_STARTUP_DIR="$(mktemp -d "$PARKOUR_CHECKPOINT_DIR/startup_diagnostics_XXXXXX")"
    echo "Startup diagnostics: $PARKOUR_STARTUP_DIR"
    for PARKOUR_LEVEL in 0 1; do
      # The step budget may be overridden; play.py validates its hard bound.
      # Keep the paired-case settings last so extra arguments cannot silently
      # turn this comparison into different policies, resets, or checkpoints.
      python scripts/rsl_rl/play.py --startup_steps=100 "$@" \
        "${PARKOUR_COMMON[@]}" --headless --livestream=0 \
        --checkpoint="$PARKOUR_CHECKPOINT" --startup_diagnostics \
        --num_envs=1 --eval_episodes=1 --reset_profile=canonical --no-screen \
        --policy_mode=history_mean --geometry_variant=0 \
        --terrain_family=high_step --difficulty_level="$PARKOUR_LEVEL" \
        --desired_speed=0.55 --desired_yaw_rate=0 \
        --command_profile=translation_only \
        --startup_output_dir="$PARKOUR_STARTUP_DIR/level_$PARKOUR_LEVEL"
    done
    python scripts/rsl_rl/startup_report.py \
      "$PARKOUR_STARTUP_DIR/level_0/startup_diagnostics.json" \
      "$PARKOUR_STARTUP_DIR/level_1/startup_diagnostics.json"
    ;;
  check-step)
    # Start with three complete easy high-step episodes, not a full sweep.
    # Append --difficulty_level=6 only after the easy approach/traversal passes.
    python scripts/rsl_rl/play.py "${PARKOUR_COMMON[@]}" \
      --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT" \
      --num_envs=1 --eval_episodes=3 --reset_profile=jitter --screen \
      --policy_mode=history_mean --geometry_variant=0 \
      --terrain_family=high_step --difficulty_level=1 --desired_speed=0.55 \
      --command_profile=translation_only --desired_yaw_rate=0 "$@"
    ;;
  check|check-control)
    # Up to 21 episodes for check, nine for check-control: three per case.
    # Do not use first-completed episodes from many async environments: that
    # can overrepresent short failures in a small screening run.
    # Batch evaluation needs no GUI or streaming client.
    PARKOUR_EVAL=("${PARKOUR_COMMON[@]}" --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT"
      --num_envs=1 --eval_episodes=3 --reset_profile=jitter --screen
      --policy_mode=history_mean --geometry_variant=0)
    python scripts/rsl_rl/play.py "${PARKOUR_EVAL[@]}" \
      --terrain_family=tilted_ramps --difficulty_level=0 --desired_speed=0.55 \
      --command_profile=stop_restart --desired_yaw_rate=0 --telemetry "$@"
    for PARKOUR_YAW in -0.5 0.5; do
      python scripts/rsl_rl/play.py "${PARKOUR_EVAL[@]}" \
        --terrain_family=tilted_ramps --difficulty_level=0 --desired_speed=0.55 \
        --command_profile=pivot_restart --desired_yaw_rate="$PARKOUR_YAW" --telemetry "$@"
    done
    # Nine episodes at most; a failed case exits immediately via set -e.
    if [[ "$PARKOUR_MODE" == check-control ]]; then
      exit 0
    fi
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
    echo "Unknown mode: $PARKOUR_MODE (use train, recover, resume, repair, diagnose, startup, check-repair, check-step, check-control, check or teleop)." >&2
    exit 2
    ;;
esac
