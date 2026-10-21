#!/usr/bin/env bash
# About four hours of training at the recorded throughput; no time cutoff.
# Finite forward-entry diagnostic; invoke from the repository root.
set -uo pipefail
PLAB_PY=${PLAB_PY:-python}
[[ -f examples/sb3/__main__.py ]] || { echo "Run from the repository root." >&2; exit 1; }
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/forward_entry_XXXXXX) || exit 1
trap 'status=$?; printf "{\"returncode\": %d}\n" "$status" > "$RUN/process_exit.json"; echo "Results: $PWD/$RUN"' EXIT
"$PLAB_PY" -m examples.sb3.entry_report declare "$RUN" || exit "$?"
run_stage() {
    local name=$1
    shift
    local log="$RUN/${name//\//_}"
    echo "Running $name; log: $log.log"
    "$PLAB_PY" -u -m examples.sb3 "$@" --device cuda:0 --learner-device cpu \
        --output "$RUN/$name" > "$log.log" 2>&1
    local code=$?
    printf '{"returncode": %d}\n' "$code" > "${log}_process_exit.json"
    echo "Finished $name: returncode=$code"
    return "$code"
}
failed=0
for algorithm in ppo sac; do
    for seed in 1078 2078; do
        prefix="$algorithm/seed_$seed"
        run_stage "$prefix/train" train --algorithm "$algorithm" \
            --task Go2-Mixed-v1 --num-envs 20 --seed "$seed" \
            --command-lesson forward-stand --run-id "${RUN##*/}:$algorithm:$seed" \
            --total-interactions 3200000 --milestones 800000 1600000 3200000 || failed=1
        for point in 1600000 3200000; do
            checkpoint="$RUN/$prefix/train/checkpoints/step_$point"
            [[ $point == 3200000 ]] && checkpoint="$RUN/$prefix/train/checkpoint"
            [[ -r "$checkpoint/checkpoint.json" ]] || { failed=1; continue; }
            run_stage "$prefix/evaluation_$point" evaluate --suite flat-entry \
                --checkpoint "$checkpoint" || failed=1
        done
    done
done
"$PLAB_PY" -m examples.sb3.entry_report report "$RUN" > "$RUN/summary.log" 2>&1 || failed=1
cat "$RUN/summary.log"
exit "$failed"
