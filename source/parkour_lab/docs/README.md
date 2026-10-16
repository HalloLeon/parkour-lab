# Parkour Lab

A learner-independent Isaac Lab / Unitree Go2 task package with blind controller
inputs and a separate training-information contract. This page covers the public
task API, installation and verification. The repository [README](../../../README.md)
contains the retained ROA-like learner and artifact-command reference. Its command
diagnostics are not a standardized benchmark; successful execution does not
qualify a robot.

## Installation

The portable package supports Python 3.10 and 3.11. Its
[dependency requirements](../setup.py) allow Gymnasium 1.x, PyTorch >=2.5 and
NumPy <2. The selected SDK and learner may impose tighter requirements.

For native execution, install Isaac Lab and Isaac Sim separately and use an
interpreter supported by that stack. The package does not select a simulator,
CUDA distribution or SDK installation layout. The optional `roa` extra uses
RSL-RL 3.1.2, separately from the SDK's optional learner extras.

From the repository root, in the configured interpreter:

```bash
python -m pip install -e source/parkour_lab
python -m parkour_lab --help
```

To include the bundled learner, use
`python -m pip install -e 'source/parkour_lab[roa]'` instead. Package imports and
CLI help do not launch Isaac Lab. Select the native device explicitly. The native
checks below establish compatibility only for their tested task and scope;
matching version labels alone do not establish it.

The CLI provides `train`, `export`, `bound-actions`, `evaluate`, `play` and
`analyze`. See [bounded action delivery](../../../README.md#bounded-action-delivery)
for explicit learner revisions and
[reward diagnostics](../../../README.md#reward-configuration-and-diagnostics)
for recorded task metrics. `train --diagnose-training` captures extreme-action
transitions, subsequent PPO rollouts and exceptions without changing learning.
These tensor captures omit optimizer and simulator state and cannot resume a run.

## Public task API

`Go2-Flat-v1` is implemented. `Go2-Uneven-v1`, `Go2-Slopes-v1`, `Go2-Stairs-v1`,
`Go2-Hills-v1` and `Go2-Mixed-v1` are declared definitions that currently raise
`NotImplementedError`. They are not executable terrain coverage or a calibrated
benchmark suite.

After starting `AppLauncher`, an independent consuming project can use:

```python
import torch
from parkour_lab.benchmark import Go2Config, build_config, make

options = Go2Config(num_envs=4, seed=42, device="cuda:0")
native_cfg = build_config(options, training=True)
task = make(native_cfg)
try:
    observation, info = task.reset(seed=options.seed)
    action = torch.zeros(task.num_envs, 12, device=task.device)
    observation, reward, terminated, truncated, info = task.step(action)
finally:
    task.close()
```

The caller owns the application and closes it after the task. The complete runnable
[consumer/check script](../../../scripts/check_public_task.py) handles ordered
environment/application cleanup and standalone native process exit. No
backend registration, RSL-RL, learner implementation or `.plab` file is needed.

The common task uses 200 Hz physics, 50 Hz actions, a 20 s episode horizon,
the native Go2 rough-task reward and stock physical termination rules. Training
commands come from the common sampler. Flat has no terrain curriculum; continuation
currently rejects non-null curriculum state. With `training=False`, call
`task.set_command(command)` after reset with an `(N, 3)` body-twist tensor
`[vx, vy, yaw_rate]` in m/s, m/s and rad/s. It returns the actor observation with
that command applied, retaining the existing sensor-noise draw and episode memory.

Actions have shape `(N, 12)` in the published motor joint order. They are encoded,
unitless raw coordinates, not radians. The native target is
`default_position + 0.25 * clamped_action`; clamping uses the encoded bounds in
`single_action_space` to respect the robot's 90% soft joint range. Previous-action
sensor values use the delivered encoding. `observation_space`/`action_space`
describe the batch; `single_observation_space`/`single_action_space` describe one
robot. Tensor conversion, history, replay and optimizers belong to the consumer.

### Actor and training information

Every observation has `actor`, a float32 `(N, 49)` causal frame: angular velocity,
projected gravity, applied command, relative joint positions, joint velocities,
previous delivered action and four binary foot contacts in FR, FL, RR, RL order.
This is blind with idealized contact sensing; it is not strictly IMU/joint-only.

Training additionally exposes `training`, a dictionary with the following fixed
`go2_training_physics_v1` fields. Shapes exclude the batch; all values are float32
except the Boolean validity fields. All methods have the same available packet;
declare the subset used by a critic, teacher or training supervision. Inference
and its history, estimator or recurrent/world-model state use causal inputs only.

| Field | Shape | Units / meaning |
| --- | --- | --- |
| `root_com_linear_velocity_b` | `(3,)` | m/s; root rigid-body COM velocity in root-link axes, not whole-robot COM |
| `root_angular_velocity_b` | `(3,)` | rad/s; root-link axes |
| `projected_gravity_b` | `(3,)` | Unit gravity direction in root-link axes |
| `joint_position_relative_default` | `(12,)` | rad |
| `joint_velocity` | `(12,)` | rad/s |
| `foot_net_forces_b` | `(4, 3)` | N; summed normal forces in root-link axes, FR/FL/RR/RL, excluding tangential forces |
| `foot_force_valid` | `(1,)` | Boolean; false with zero forces before the first post-reset physics sample |
| `base_mass` | `(1,)` | kg; realized base-link mass |
| `base_com_offset_b` | `(3,)` | m; COM offset in base-link axes |
| `material_friction` | `(2,)` | Robot static/dynamic coefficients; not effective contact-pair friction |
| `joint_stiffness` | `(12,)` | N·m/rad; realized explicit actuator gains |
| `joint_damping` | `(12,)` | N·m·s/rad; realized explicit actuator gains |
| `motor_strength` | `(12,)` | Effort multiplier before the motor limiter |
| `terrain_height` | `(132,)` | m; `clip(root_link_z - 0.3 - hit_z, -0.5, 0.5)` |
| `terrain_valid` | `(132,)` | Boolean; all coordinates of the ray hit are finite |

The 352 values are clean, detached copies at the current control boundary. Contact
forces use the latest completed physics sample. Verified startup mass/material/COM
and motor parameters remain fixed across episode resets. The terrain scan uses
12 x positions from −0.825 to +0.825 m and 11 y positions from −0.75 to +0.75 m,
both at 0.15 m spacing, with x varying fastest. It follows root yaw, with x forward
and y left; rays point vertically downward from a 20 m offset, with 25 m maximum
range. Missing hits use +0.5 m with validity false; finite clipped hits remain valid.
`task.metadata["training_information"]` provides exact field order, joint/foot
names, coordinates and interpretation. The packet contains no global pose, case
IDs, seeds, future commands, fault schedule or environment handles. Evaluation
observations and metadata omit the training packet/schema entirely.

### Endings, accounting and continuation

`step` returns the Gym-style five-tuple. Same-step automatic reset supplies the
next-action reset observation for ending rows. `info` contains only
`final_env_ids`, `final_observation` and `interactions`. Final observations are
owned compact outgoing rows, including separately aligned training GT when
enabled. They cannot be replaced by next-episode GT or successful reset episodes.
Both ending flags remain available; a physical termination takes precedence over
a simultaneous timeout when deciding value bootstrapping. Process final states
with outgoing recurrent memory, then reset memory only for ending rows.

GT never samples actor noise. Causal final observations preview their rows' next
noise draw without advancing the ordinary observation stream; they remain separate
from clean GT. Reading or ignoring final observations does not change the sensor
stream used for subsequent actions.

`interactions` reports session, cumulative and uncertain robot transitions. A
completed step with 160 replicas charges 160 interactions, including discarded
or warm-up collection. Explicit episode reset does not refund the total. The
counter is not a crash-proof journal. `reset(seed=...)` only checks the fixed
construction seed; it does not reseed the learner's global Torch RNG or rewind
the task's private sampler streams.

The consuming trainer owns its full checkpoints, including optimizer, replay,
normalization, memory/RNG and method-specific state. At its checkpoint barrier,
save `task.progress(run_id=..., checkpoint_id=..., milestones=...)` alongside the
learner checkpoint it identifies. In a new process, rebuild the identical task,
restore the learner separately and call `task.restore(record, checkpoint_id=...)`
before the first explicit reset. Task progress restores accounting and shared
sampler progress, then starts fresh physical episodes; it is not an exact
interrupted-trajectory snapshot. To retain later known or conservatively charged
spend, `charged_interactions` means the latest known **total**, while
`uncertain_interactions` is an **additional** conservative charge; supply an
explicit `reason`. Treat recreation as a fresh actor-memory/replay-sequence boundary.
Rollback must not erase collection. The consumer owns its save cadence;
the task does not write learner checkpoints or enforce an optimizer schedule.

### Reward customization

Only rewards may be changed in the native configuration; other protocol sections
remain fixed. Modify a native term's weight/parameters, set weight zero to disable
its computation, or add a native `RewardTermCfg` with an ordinary callable:

```python
from isaaclab.managers import RewardTermCfg

def speed_squared(env):
    return env.scene["robot"].data.root_com_lin_vel_b.square().sum(-1)

native_cfg.rewards.track_ang_vel_z_exp.weight = 1.0
native_cfg.rewards.extra_speed = RewardTermCfg(func=speed_squared, weight=-0.01)
```

Apply changes before `make(native_cfg)`. This example defines a custom objective,
not a recommended baseline. Native rewards sum `weight * term * control_dt`;
metadata identifies the realized recipe. Custom objectives must be identified
separately from a common-reward method comparison.

## Architecture

The environment, inference controller and learner are separate responsibilities.

| Owner | Responsibility |
| --- | --- |
| `benchmark/` | Learner-independent task configuration/facade, role-specific observations, fixed GT schema, spaces and checkpoint-linked progress. |
| `config.py` | Validated task and training settings. |
| `artifacts.py` | Data-only learning/controller artifacts and motor contracts. |
| `provenance.py` | Package hashes, source-owned dependency identity and Git evidence. |
| `control/` | Causal sensors, controller/reset interface, command tapes and motor contracts. |
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
| `scripts/check_public_task.py` | Runnable public-task consumer and bounded training/evaluation/continuation contract check. |

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
python scripts/check_public_task.py --help
```

Formatting tools match `.pre-commit-config.yaml`. Formatting uses 88 columns;
long strings/comments remain permitted, with a narrow lint exception for protocol
ellipsis bodies. See [tests/README.md](../../../tests/README.md) for component
commands and coverage limits. CPU doubles and skipped SDK-source checks do not
establish native compatibility.

The portable public-task contracts are checked with Gymnasium 1.0.0, 1.2.0 and
1.2.1. This covers the declared spaces, batched observations and interface behavior;
it does not establish native SDK compatibility for every allowed Gymnasium version.
The selected SDK's dependency requirements still apply.

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

These checks have been run with Isaac Lab 2.3.2.post1 and Isaac Sim 5.1. Run them
when using a different stack to establish compatibility for the supported task.

Use the configured Isaac Lab interpreter from the repository root. Activate it or
export `PLAB_PY` with its Python executable path. Each invocation below launches a
fresh process, creates the public Flat task, runs 32 near-default action steps,
forces row zero to timeout and explicitly resets row zero. It checks physical
action delivery, role-specific observations, owned final rows and exact
interaction accounting. Training additionally compares selected GT with direct
native readbacks captured before reset. Those private host checks never supply
native handles or truth to the example controller.

```bash
bash <<'BASH'
set -uo pipefail
PLAB_PY=${PLAB_PY:-python}
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/public_task_XXXXXX) || exit 1
trap 'status=$?; printf "{\"returncode\": %d}\n" "$status" > "$RUN/process_exit.json"; echo "Results: $PWD/$RUN"' EXIT
run_check() {
    local name=$1
    shift
    "$PLAB_PY" -u scripts/check_public_task.py --device cuda:0 --num-envs 4 \
        --seed 1077 --output "$RUN/$name" "$@" > "$RUN/$name.log" 2>&1
    code=$?
    printf '{"returncode": %d}\n' "$code" > "$RUN/${name}_process_exit.json"
    return "$code"
}
run_check train --role train || exit "$?"
run_check eval --role eval || exit "$?"
run_check resume --role train --resume "$RUN/train/progress.json" \
    --checkpoint-id no-learner-contract-fixture || exit "$?"
BASH
```

The first training and evaluation processes each charge 128 interactions. The
fresh resumed process restores 128 and finishes at cumulative 256, with session
128. Its opaque checkpoint token is a contract-test fixture: **no learner model,
weights or optimizer are loaded or restored**. Real training must link its progress
record to an actual full learner checkpoint.

Every process must exit zero and report `TASK_CHECK_PASSED` with both cleanup fields
`complete` in `report.json`. Retain actual dependency versions and imported-source
hashes with the results. For reproduction, keep installation artifacts and their
hashes, or source checkout revisions/diffs in `sdk/git/`. A packaged SDK does not
require Git metadata. An incomplete close receipt, missing dependency/source
identity or failed assertion leaves native validation incomplete.

The retained terrain plumbing helper remains available as
`scripts/smoke_native.py --terrain flat --device cuda:0 --output NEW_DIRECTORY`
(also rough, procedural, steps, traversal and connected); it reports `SMOKE_PASSED`
and checks the retained command workflow separately from the public task API.

Successful smoke checks do not establish traversal, robot stability, benchmark
scores or equivalence between simulators. Commands not yet run on the server are
not native evidence. Run relevant checks after task, controller, scoring or
continuation changes and retain complete failed attempts when assessing behavior.
