#!/usr/bin/env bash
# Focused Go2 RMA workflow. Run from the repository root in the Isaac Lab env.
set -euo pipefail

PARKOUR_MODE="${1:?Usage: bash scripts/rsl_rl/go2_rma.sh train|recover|resume|repair|diagnose|probe-startup|probe-centered|probe-friction|probe-solver|probe-collision|check-centered|startup|check-repair|check-step|check-control|check|check-operator-reference|refine-operator|teleop CHECKPOINT [extra arguments]}"
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
  refine-operator)
    if [[ "$PARKOUR_REWARD_PROFILE" != baseline ]]; then
      echo "refine-operator preserves stock rewards; parkour reward profiles do not apply." >&2
      exit 2
    fi
    # Installed stock task, no legacy collision/streaming overrides or external scripts.
    python scripts/rsl_rl/operator_train.py "$PARKOUR_CHECKPOINT" "$@"
    ;;
  check-operator-reference)
    if [[ "$PARKOUR_REWARD_PROFILE" != baseline ]]; then
      echo "check-operator-reference uses the stock task, not parkour reward profiles." >&2
      exit 2
    fi
    # No PARKOUR_COMMON: this is a different, explicitly validated checkpoint
    # interface. The runner forces headless=True and livestream=0 internally.
    python scripts/rsl_rl/operator_benchmark.py "$PARKOUR_CHECKPOINT" "$@"
    ;;
  probe-collision)
    if (( $# != 2 )) || [[ ! -f "$1" ]] || \
       [[ ! -f "$2/level_0/startup_diagnostics.json" ]] || \
       [[ ! -f "$2/level_6/startup_diagnostics.json" ]]; then
      echo "Usage: go2_rma.sh probe-collision CHECKPOINT SOURCE_STARTUP_JSON ORIGINAL_ACTION_PROBE_DIR (no extra overrides)" >&2
      exit 2
    fi
    if [[ "$PARKOUR_REWARD_PROFILE" != baseline ]]; then
      echo "probe-collision requires PARKOUR_REWARD_PROFILE=baseline." >&2
      exit 2
    fi
    PARKOUR_PROBE_SOURCE="$1"
    PARKOUR_ORIGINAL_PROBE="$2"
    PARKOUR_CHECKPOINT_DIR="$(dirname -- "$PARKOUR_CHECKPOINT")"
    PARKOUR_PROBE_DIR="$(mktemp -d "$PARKOUR_CHECKPOINT_DIR/collision_probe_XXXXXX")"
    echo "One-action ground-collision bisection (not locomotion): $PARKOUR_PROBE_DIR"
    PARKOUR_COLLISION_REPORT=(scripts/rsl_rl/startup_collision_report.py "$PARKOUR_PROBE_DIR"
      --reference="$PARKOUR_PROBE_SOURCE" --original-probe="$PARKOUR_ORIGINAL_PROBE")
    for PARKOUR_CASE in native_L0 native_L6 ground_off_L0 ground_off_L6; do
      PARKOUR_COLLISION_MODE=native
      PARKOUR_LEVEL=0
      if [[ "$PARKOUR_CASE" == ground_off_* ]]; then PARKOUR_COLLISION_MODE=ground_off; fi
      if [[ "$PARKOUR_CASE" == *_L6 ]]; then PARKOUR_LEVEL=6; fi
      python scripts/rsl_rl/play.py "${PARKOUR_COMMON[@]}" \
        --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT" \
        --startup_diagnostics --startup_steps=1 \
        --startup_action_replay="$PARKOUR_PROBE_SOURCE" \
        --startup_ground_collision="$PARKOUR_COLLISION_MODE" \
        --startup_output_dir="$PARKOUR_PROBE_DIR/$PARKOUR_CASE" \
        --num_envs=1 --eval_episodes=1 --no-screen --reset_profile=jitter \
        --policy_mode=history_mean --terrain_family=high_step \
        --difficulty_level="$PARKOUR_LEVEL" --geometry_variant=0 \
        --desired_speed=0.55 --desired_yaw_rate=0 --command_profile=translation_only
      if [[ "$PARKOUR_CASE" == native_L6 ]]; then
        if ! python "${PARKOUR_COLLISION_REPORT[@]}" --preflight > "$PARKOUR_PROBE_DIR/preflight_report.json"; then
          echo "Native reproduction failed; no ground-off runs launched. See $PARKOUR_PROBE_DIR/preflight_report.json" >&2
          exit 2
        fi
      fi
    done
    python "${PARKOUR_COLLISION_REPORT[@]}" > "$PARKOUR_PROBE_DIR/comparison.json"
    echo "Saved $PARKOUR_PROBE_DIR/comparison.json (neither outcome certifies robot operation)."
    ;;
  check-centered)
    if (( $# )); then
      echo "check-centered has fixed paired evaluation settings and accepts no extra overrides." >&2
      exit 2
    fi
    PARKOUR_CHECKPOINT_DIR="$(dirname -- "$PARKOUR_CHECKPOINT")"
    PARKOUR_PAIR_DIR="$(mktemp -d "$PARKOUR_CHECKPOINT_DIR/scene_feedback_XXXXXX")"
    echo "Native versus centered normal-policy evaluation: $PARKOUR_PAIR_DIR"
    for PARKOUR_PLACEMENT in native centered; do
      # Apply the behavioral screens only after BOTH runs. Native falling is
      # an outcome, whereas a simulator/configuration failure stops immediately.
      python scripts/rsl_rl/play.py "${PARKOUR_COMMON[@]}" \
        --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT" \
        --evaluation_scene_probe="$PARKOUR_PLACEMENT" \
        --no-screen --num_envs=1 --eval_episodes=3 --reset_profile=jitter \
        --policy_mode=history_mean --terrain_family=high_step \
        --difficulty_level=6 --geometry_variant=0 \
        --desired_speed=0.55 --desired_yaw_rate=0 --command_profile=translation_only \
        --video_output_dir="$PARKOUR_PAIR_DIR/${PARKOUR_PLACEMENT}_L6"
      if [[ "$PARKOUR_PLACEMENT" == native ]]; then
        if ! python scripts/rsl_rl/scene_feedback_report.py "$PARKOUR_PAIR_DIR" \
          --native-only > "$PARKOUR_PAIR_DIR/native_report.json"; then
          echo "Native evidence is incomplete or invalid; centered run not launched. See $PARKOUR_PAIR_DIR/native_report.json" >&2
          exit 2
        fi
      fi
    done
    PARKOUR_PAIR_EXIT=0
    python scripts/rsl_rl/scene_feedback_report.py "$PARKOUR_PAIR_DIR" \
      > "$PARKOUR_PAIR_DIR/comparison.json" || PARKOUR_PAIR_EXIT=$?
    echo "Paired feedback report: $PARKOUR_PAIR_DIR/comparison.json (exit $PARKOUR_PAIR_EXIT; not operator acceptance)."
    exit "$PARKOUR_PAIR_EXIT"
    ;;
  probe-solver)
    if (( $# != 3 )) || [[ ! -f "$1" ]] || [[ ! -f "$2" ]] || \
       [[ ! -f "$3/level_0/startup_diagnostics.json" ]] || \
       [[ ! -f "$3/level_6/startup_diagnostics.json" ]]; then
      echo "Usage: go2_rma.sh probe-solver CHECKPOINT SOURCE_STARTUP_JSON FRICTION_NATIVE_JSON ORIGINAL_ACTION_PROBE_DIR (no extra overrides)" >&2
      exit 2
    fi
    PARKOUR_PROBE_SOURCE="$1"
    PARKOUR_PROBE_BASELINE="$2"
    PARKOUR_ORIGINAL_PROBE="$3"
    PARKOUR_CHECKPOINT_DIR="$(dirname -- "$PARKOUR_CHECKPOINT")"
    PARKOUR_PROBE_DIR="$(mktemp -d "$PARKOUR_CHECKPOINT_DIR/solver_probe_XXXXXX")"
    echo "Conditional solver experiment: $PARKOUR_PROBE_DIR"
    PARKOUR_SOLVER_REPORT=(scripts/rsl_rl/startup_solver_report.py "$PARKOUR_PROBE_DIR"
      --reference="$PARKOUR_PROBE_SOURCE" --baseline="$PARKOUR_PROBE_BASELINE"
      --original-probe="$PARKOUR_ORIGINAL_PROBE")
    for PARKOUR_CASE in native_L6 pgs_L0 pgs_L6; do
      PARKOUR_SOLVER=pgs
      PARKOUR_LEVEL=6
      if [[ "$PARKOUR_CASE" == native_L6 ]]; then PARKOUR_SOLVER=tgs; fi
      if [[ "$PARKOUR_CASE" == pgs_L0 ]]; then PARKOUR_LEVEL=0; fi
      python scripts/rsl_rl/play.py "${PARKOUR_COMMON[@]}" \
        --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT" \
        --startup_diagnostics --startup_steps=10 \
        --startup_action_replay="$PARKOUR_PROBE_SOURCE" \
        --startup_solver_probe="$PARKOUR_SOLVER" \
        --startup_output_dir="$PARKOUR_PROBE_DIR/$PARKOUR_CASE" \
        --num_envs=1 --eval_episodes=1 --no-screen --reset_profile=jitter \
        --policy_mode=history_mean --terrain_family=high_step \
        --difficulty_level="$PARKOUR_LEVEL" --geometry_variant=0 \
        --desired_speed=0.55 --desired_yaw_rate=0 --command_profile=translation_only
      if [[ "$PARKOUR_CASE" == native_L6 ]]; then
        if ! python "${PARKOUR_SOLVER_REPORT[@]}" --preflight > "$PARKOUR_PROBE_DIR/preflight_report.json"; then
          echo "Solver preflight stopped; no PGS runs launched. See $PARKOUR_PROBE_DIR/preflight_report.json" >&2
          exit 2
        fi
      fi
    done
    if ! python "${PARKOUR_SOLVER_REPORT[@]}" > "$PARKOUR_PROBE_DIR/solver_report.json"; then
      echo "Solver replay gate stopped; no policy runs launched. Inspect $PARKOUR_PROBE_DIR/solver_report.json (partial improvement is not an invalid experiment)." >&2
      exit 2
    fi
    # The same explicit PGS selection/readback also applies to normal feedback.
    # No replay, centering, gains, limit or training changes in these episodes.
    python scripts/rsl_rl/play.py "${PARKOUR_COMMON[@]}" \
      --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT" \
      --evaluation_solver_probe=pgs --screen --num_envs=1 --eval_episodes=3 \
      --reset_profile=jitter --policy_mode=history_mean \
      --terrain_family=high_step --difficulty_level=6 --geometry_variant=0 \
      --desired_speed=0.55 --desired_yaw_rate=0 --command_profile=translation_only \
      --video_output_dir="$PARKOUR_PROBE_DIR/policy_L6"
    echo "PGS high-step screen completed; this is not full control/operator acceptance. Artifacts: $PARKOUR_PROBE_DIR"
    ;;
  probe-friction)
    if (( $# != 2 )) || [[ ! -f "$1" ]] || [[ ! -f "$2" ]]; then
      echo "Usage: go2_rma.sh probe-friction CHECKPOINT SOURCE_STARTUP_JSON PREVIOUS_UNCENTERED_L6_JSON (no extra overrides)" >&2
      exit 2
    fi
    PARKOUR_PROBE_SOURCE="$1"
    PARKOUR_PROBE_BASELINE="$2"
    PARKOUR_CHECKPOINT_DIR="$(dirname -- "$PARKOUR_CHECKPOINT")"
    PARKOUR_PROBE_DIR="$(mktemp -d "$PARKOUR_CHECKPOINT_DIR/friction_probe_XXXXXX")"
    echo "Conditional legacy-friction probe: $PARKOUR_PROBE_DIR"
    for PARKOUR_CASE in native_L6 zero_L0 zero_L6; do
      PARKOUR_FRICTION_MODE=zero
      PARKOUR_LEVEL=6
      if [[ "$PARKOUR_CASE" == native_L6 ]]; then PARKOUR_FRICTION_MODE=observe; fi
      if [[ "$PARKOUR_CASE" == zero_L0 ]]; then PARKOUR_LEVEL=0; fi
      python scripts/rsl_rl/play.py "${PARKOUR_COMMON[@]}" \
        --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT" \
        --startup_diagnostics --startup_steps=10 \
        --startup_action_replay="$PARKOUR_PROBE_SOURCE" \
        --startup_legacy_friction="$PARKOUR_FRICTION_MODE" \
        --startup_output_dir="$PARKOUR_PROBE_DIR/$PARKOUR_CASE" \
        --num_envs=1 --eval_episodes=1 --no-screen --reset_profile=jitter \
        --policy_mode=history_mean --terrain_family=high_step \
        --difficulty_level="$PARKOUR_LEVEL" --geometry_variant=0 \
        --desired_speed=0.55 --desired_yaw_rate=0 --command_profile=translation_only
      if [[ "$PARKOUR_CASE" == native_L6 ]]; then
        if ! python scripts/rsl_rl/startup_friction_report.py "$PARKOUR_PROBE_DIR" \
          --baseline="$PARKOUR_PROBE_BASELINE" --reference="$PARKOUR_PROBE_SOURCE" \
          --preflight > "$PARKOUR_PROBE_DIR/preflight_report.json"; then
          echo "Friction preflight stopped this branch; no zeroing runs launched. See $PARKOUR_PROBE_DIR/preflight_report.json" >&2
          exit 2
        fi
      fi
    done
    python scripts/rsl_rl/startup_friction_report.py "$PARKOUR_PROBE_DIR" \
      --baseline="$PARKOUR_PROBE_BASELINE" --reference="$PARKOUR_PROBE_SOURCE" \
      > "$PARKOUR_PROBE_DIR/friction_report.json"
    echo "Saved $PARKOUR_PROBE_DIR/friction_report.json (diagnostic only; no production physics changed)."
    ;;
  probe-centered)
    if (( $# != 2 )) || [[ ! -f "$1" ]] || \
       [[ ! -f "$2/level_0/startup_diagnostics.json" ]] || \
       [[ ! -f "$2/level_6/startup_diagnostics.json" ]]; then
      echo "Usage: go2_rma.sh probe-centered CHECKPOINT SOURCE_STARTUP_JSON ORIGINAL_PROBE_DIR (no extra overrides)" >&2
      exit 2
    fi
    PARKOUR_PROBE_SOURCE="$1"
    PARKOUR_ORIGINAL_PROBE="$2"
    PARKOUR_CHECKPOINT_DIR="$(dirname -- "$PARKOUR_CHECKPOINT")"
    PARKOUR_PROBE_DIR="$(mktemp -d "$PARKOUR_CHECKPOINT_DIR/centered_probe_XXXXXX")"
    echo "Scene-placement probe: $PARKOUR_PROBE_DIR"
    # Keep a fresh uncentered L6 control: new getters/runtime must reproduce
    # before differences can be attributed to the scene intervention.
    for PARKOUR_CASE in uncentered_L6 centered_L0 centered_L6; do
      PARKOUR_CENTER_ARGS=()
      PARKOUR_LEVEL=6
      if [[ "$PARKOUR_CASE" == centered_* ]]; then
        PARKOUR_CENTER_ARGS+=(--startup_center_scene)
      fi
      if [[ "$PARKOUR_CASE" == centered_L0 ]]; then PARKOUR_LEVEL=0; fi
      python scripts/rsl_rl/play.py "${PARKOUR_COMMON[@]}" \
        --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT" \
        --startup_diagnostics --startup_steps=10 \
        --startup_action_replay="$PARKOUR_PROBE_SOURCE" ${PARKOUR_CENTER_ARGS[@]+"${PARKOUR_CENTER_ARGS[@]}"} \
        --startup_output_dir="$PARKOUR_PROBE_DIR/$PARKOUR_CASE" \
        --num_envs=1 --eval_episodes=1 --no-screen --reset_profile=jitter \
        --policy_mode=history_mean --terrain_family=high_step \
        --difficulty_level="$PARKOUR_LEVEL" --geometry_variant=0 \
        --desired_speed=0.55 --desired_yaw_rate=0 --command_profile=translation_only
    done
    python scripts/rsl_rl/startup_centered_report.py \
      "$PARKOUR_PROBE_DIR" --original-probe="$PARKOUR_ORIGINAL_PROBE" \
      --reference="$PARKOUR_PROBE_SOURCE" > "$PARKOUR_PROBE_DIR/centered_report.json"
    echo "Saved $PARKOUR_PROBE_DIR/centered_report.json (diagnostic evidence, not robot acceptance)."
    ;;
  probe-startup)
    # Counterfactual dynamics probe, not a new policy rollout/acceptance screen.
    # Remove the small first-action difference by replaying one explicit source
    # prefix through the normal mapping, delay, safe clamp and actuator pipeline.
    if (( $# != 1 )) || [[ ! -f "$1" ]]; then
      echo "Usage: go2_rma.sh probe-startup CHECKPOINT SOURCE_STARTUP_DIAGNOSTICS_JSON (no extra overrides)" >&2
      exit 2
    fi
    PARKOUR_PROBE_SOURCE="$1"
    PARKOUR_CHECKPOINT_DIR="$(dirname -- "$PARKOUR_CHECKPOINT")"
    PARKOUR_PROBE_DIR="$(mktemp -d "$PARKOUR_CHECKPOINT_DIR/action_probe_XXXXXX")"
    echo "Matched-action probe: $PARKOUR_PROBE_DIR"
    for PARKOUR_LEVEL in 0 6; do
      python scripts/rsl_rl/play.py "${PARKOUR_COMMON[@]}" \
        --headless --livestream=0 --checkpoint="$PARKOUR_CHECKPOINT" \
        --startup_diagnostics --startup_steps=10 \
        --startup_action_replay="$PARKOUR_PROBE_SOURCE" \
        --startup_output_dir="$PARKOUR_PROBE_DIR/level_$PARKOUR_LEVEL" \
        --num_envs=1 --eval_episodes=1 --no-screen --reset_profile=jitter \
        --policy_mode=history_mean --terrain_family=high_step \
        --difficulty_level="$PARKOUR_LEVEL" --geometry_variant=0 \
        --desired_speed=0.55 --desired_yaw_rate=0 --command_profile=translation_only
    done
    python scripts/rsl_rl/startup_probe_report.py \
      "$PARKOUR_PROBE_DIR/level_0/startup_diagnostics.json" \
      "$PARKOUR_PROBE_DIR/level_6/startup_diagnostics.json" \
      --reference="$PARKOUR_PROBE_SOURCE" --json > "$PARKOUR_PROBE_DIR/probe_report.json"
    echo "Saved $PARKOUR_PROBE_DIR/probe_report.json (valid probe does not mean the robot passes)."
    ;;
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
    echo "Unknown mode: $PARKOUR_MODE (use train, recover, resume, repair, diagnose, probe-startup, probe-centered, probe-friction, probe-solver, probe-collision, startup, check-repair, check-step, check-control, check or teleop)." >&2
    exit 2
    ;;
esac
