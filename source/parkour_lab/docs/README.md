# Parkour Lab

An Isaac Lab / Unitree Go2 locomotion research package with an optional ROA-like
teacher and causal history student. This page covers installation, package
responsibilities and verification. The repository [README](../../../README.md)
contains the command, control and diagnostic reference. Command diagnostics are
not a standardized benchmark; successful execution does not qualify a robot.

## Installation

Use Python 3.10 or 3.11. Install Isaac Lab 2.3.2 / Isaac Sim 5.1 separately in
the interpreter configured for native execution; the package does not select a
simulator or CUDA distribution. The optional `roa` extra uses RSL-RL 3.1.2.
From the repository root:

```bash
python -m pip install -e source/parkour_lab
python -m parkour_lab --help
```

To include the bundled learner, use
`python -m pip install -e 'source/parkour_lab[roa]'` instead. Package imports and
CLI help do not launch Isaac Lab. Select the native device explicitly and record
the SDK revision and local changes actually imported; a version label alone does
not establish compatibility.

The CLI provides `train`, `export`, `bound-actions`, `evaluate`, `play` and
`analyze`. See [bounded action delivery](../../../README.md#bounded-action-delivery)
for explicit learner revisions and
[reward diagnostics](../../../README.md#reward-configuration-and-diagnostics)
for recorded task metrics. `train --diagnose-training` captures extreme-action
transitions, subsequent PPO rollouts and exceptions without changing learning.
These tensor captures omit optimizer and simulator state and cannot resume a run.

## Architecture

The environment, inference controller and learner are separate responsibilities.

| Owner | Responsibility |
| --- | --- |
| `config.py` | Validated task and training settings. |
| `artifacts.py` | Data-only learning/controller artifacts and motor contracts. |
| `provenance.py` | Package hashes, source-owned dependency identity and Git evidence. |
| `control/` | Causal sensors, controller/reset interface, command sources/tapes and motor contracts. |
| `environments/` | Native task configuration, observations, dynamics, rewards and reset semantics. Isaac Lab owns physics, managers and step/reset order. |
| `environments/commands/` | Native command terms, training command sampling and fixed command sequences. |
| `environments/terrains/` | Procedural and connected terrain geometry, traversal fixtures and roughness helpers. |
| `runtime/` | Native sensor/motor binding, transition accounting, training host and ordered environment/application cleanup. |
| `evaluation/runner.py` | Frozen live playback and recording. |
| `evaluation/terrain_analysis.py` | Saved connected-attempt validation and analysis without live execution or learner imports. |
| `evaluation/flat.py`, `terrain.py` | Development-case definitions and physical trace scoring. |
| `methods/` | Optional backend discovery, ROA-like optimization, inference implementation and learning payloads. |
| `experiment.py` | Training orchestration and learner checkpoint continuation. |
| `tests/` | Behavioral contracts, package boundaries, CPU learning and analytic geometry checks. |
| `scripts/smoke_native.py` | Bounded native construction, stepping, final-observation, reset and shutdown checks. |

`control.Controller` and `runtime.NativeControllerSession` accept a loaded causal
controller independently of a learner. The CLI artifact workflow uses
`MethodConfig` and a method backend to load `.plab` files; it is not a
method-independent benchmark API. Saved terrain analysis is implemented in
`terrain_analysis.py`; `runner.analyze_terrain_attempt` re-exports it for import
compatibility.

Native stepping preserves separate termination/truncation flags and owned outgoing
final-observation rows before automatic reset. Learner adapters own optimization,
rollout/replay storage and learning snapshots. Shared control and terrain code do
not select a network architecture. ROA inference does not import PPO; numerical
and native imports remain lazy where unnecessary.

## Verification

### CPU checks

From the repository root in a CPU Python environment:

```bash
python -m pip install -e 'source/parkour_lab[roa]'
python -m pip install black==25.9.0 isort==6.1.0 flake8==7.1.1
python -m unittest discover -s tests -t . -v
black --workers 1 --check source/parkour_lab tests scripts
isort --check-only source/parkour_lab tests scripts
flake8 source/parkour_lab tests scripts
python -m parkour_lab --help
python scripts/smoke_native.py --help
```

Formatting tools match `.pre-commit-config.yaml`. Formatting uses 88 columns;
long strings/comments remain permitted, with a narrow lint exception for protocol
ellipsis bodies. See [tests/README.md](../../../tests/README.md) for component
commands and coverage limits. CPU doubles and skipped SDK-source checks do not
establish native compatibility.

### Distribution checks

Build a source archive, rebuild its wheel and install outside the checkout in a
fresh environment. The installation below resolves the core dependencies; the
native SDK remains separate.

```bash
bash <<'BASH'
set -euo pipefail
python -m pip install build
BUILD=$(mktemp -d)
python -m build --sdist --outdir "$BUILD/dist" source/parkour_lab
SDIST=("$BUILD"/dist/*.tar.gz)
tar -xzf "${SDIST[0]}" -C "$BUILD"
PACKAGE=("$BUILD"/parkour_lab-*)
python -m build --wheel --outdir "$BUILD/wheel" "${PACKAGE[0]}"
python -m venv "$BUILD/install"
"$BUILD/install/bin/python" -m pip install "$BUILD"/wheel/*.whl
cd "$BUILD"
unset PYTHONPATH
"$BUILD/install/bin/python" -I -m parkour_lab --help
"$BUILD/install/bin/python" -I -c 'import parkour_lab; from importlib.metadata import requires; print(parkour_lab.__file__); print(requires("parkour_lab"))'
echo "Build and installation: $BUILD"
BASH
```

Check dependency metadata, installed backend discovery and archive contents as
well as imports. The source archive must include `config/extension.toml` and this
README, and exclude private planning files. An editable installation alone does
not verify those contents. The CPU suite covers backend discovery, artifact round
trips and data-only payload loading.

### Native checks

Use the configured Isaac Lab interpreter from the repository root. Activate it or
export `PLAB_PY` with its Python executable path. Each fresh process creates one
task, runs 32 zero-action control steps, forces one row to timeout, checks owned
final observations, resets one row and closes the environment before the
application. Connected terrain has one replica, so its explicit reset is a full
single-replica reset. Zero actions check plumbing, not locomotion ability.

```bash
bash <<'BASH'
set -uo pipefail
PLAB_PY=${PLAB_PY:-python}
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/native_smoke_XXXXXX) || exit 1
for terrain in flat rough procedural steps traversal connected; do
    "$PLAB_PY" -u scripts/smoke_native.py --terrain "$terrain" --device cuda:0 \
        --output "$RUN/$terrain" > "$RUN/$terrain.log" 2>&1
    code=$?
    printf '{"returncode": %d}\n' "$code" > "$RUN/${terrain}_process_exit.json"
    if (( code != 0 )); then
        echo "Failed: $terrain. Results: $PWD/$RUN"
        exit "$code"
    fi
done
echo "Results: $PWD/$RUN"
BASH
```

Every process must exit zero and report `SMOKE_PASSED` with both cleanup fields
`complete` in `report.json`. Review `sdk/git/` and dependency records for actual
native source identity. An incomplete close receipt, missing identity or failed
assertion leaves native validation incomplete. Without SDK Git metadata, resolve
the selected source revision/diff separately; source hashes alone do not provide
that history.

Successful smoke checks do not establish traversal, robot stability, benchmark
scores or equivalence between simulators. Commands not yet run on the server are
not native evidence. Run relevant checks after task, controller, scoring or
continuation changes and retain complete failed attempts when assessing behavior.
