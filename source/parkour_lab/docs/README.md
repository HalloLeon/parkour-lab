# Parkour Lab

A learner-independent Isaac Lab / Unitree Go2 task package with blind controller
inputs and a separate training-information contract. This page covers the public
task API, frozen evaluation, installation and verification. The consumer owns its
learning code and checkpoint format. Successful execution alone does not establish
locomotion quality or a calibrated benchmark ranking.

## Installation

The portable package supports Python 3.10 and 3.11. Its
[dependency requirements](../setup.py) allow Gymnasium 1.x, PyTorch >=2.5 and
NumPy <2. The selected SDK and learner may impose tighter requirements.

For native execution, install Isaac Lab and Isaac Sim separately and use an
interpreter supported by that stack. The package does not select a simulator,
CUDA distribution or SDK installation layout. External learners install their own
training dependencies.

From the repository root, in the configured interpreter:

```bash
python -m pip install -e source/parkour_lab
python -m parkour_lab --help
```

Package imports and CLI help do not launch Isaac Lab. Select the native device
explicitly. The CLI scores saved evaluation captures; it does not load policies
or train models. Native checks establish compatibility only for their tested
scope, not from matching version labels alone.

## Public task API

The public tasks are `Go2-Flat-v1`, `Go2-Uneven-v1`, `Go2-Slopes-v1`,
`Go2-Stairs-v1`, `Go2-Hills-v1` and `Go2-Mixed-v1`. Mixed allocates equal shares
to the five families. The terrain recipe below is a development candidate;
construction and CPU geometry checks do not establish native calibration or
learned locomotion.

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
commands come from the common episode sampler. Flat has no terrain curriculum;
other families keep their assigned type while advancing within its level ladder.
With `training=False`, call
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

### Terrains and difficulty

Select the task and terrain settings before native construction:

```python
from parkour_lab.benchmark import Go2Config, TerrainConfig, build_config

training_cfg = build_config(Go2Config(task_id="Go2-Mixed-v1"), training=True)
test_cfg = build_config(
    Go2Config(task_id="Go2-Stairs-v1", terrain=TerrainConfig(fixed_level=3)),
    training=False,
)
```

`go2-terrain-development-v1` uses 12 × 12 m tiles. Mixed has 20 ordered columns:
four Flat, four Uneven, four Slopes, four Stairs and four Hills. Slopes and Stairs
split their columns equally between ordinary and inverted pyramids. Pyramid sign
describes the surface; it does not establish that a robot ascended or descended.
Family-specific tasks use the same physical recipe. Flat is explicitly present
and stays level zero; a non-flat family's level zero is already non-flat.

The eight candidate levels use these requested native-generator parameters:

| Level | Uneven noise amplitude, m | Uneven sample spacing, m | Slope rise/run | Stair riser, m | Hill amplitude parameter, m |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.0050 | 0.8000 | 0.0350 | 0.02 | 0.02 |
| 1 | 0.0114 | 0.7143 | 0.0729 | 0.04 | 0.06 |
| 2 | 0.0179 | 0.6286 | 0.1107 | 0.06 | 0.10 |
| 3 | 0.0243 | 0.5429 | 0.1486 | 0.08 | 0.14 |
| 4 | 0.0307 | 0.4571 | 0.1864 | 0.10 | 0.18 |
| 5 | 0.0371 | 0.3714 | 0.2243 | 0.12 | 0.22 |
| 6 | 0.0436 | 0.2857 | 0.2621 | 0.14 | 0.26 |
| 7 | 0.0500 | 0.2000 | 0.3000 | 0.16 | 0.30 |

Heightfields use 0.1 m horizontal cells and 0.005 m height increments. Uneven
uses the native random-uniform heightfield with cubic interpolation; its realized
height can exceed the requested noise amplitude. Slopes and Stairs have 2 m
central platforms; stairs use 0.3 m treads and a 0.3 m border. Hills uses two waves
along each axis, with approximately 5.95 m wavelength; the native amplitude
parameter contributes half its value on each axis. Its resulting peak-to-peak
height is approximately twice the table value. No extra roughness is added to
Slopes, Stairs or Hills. Reports retain exact unrounded parameters.

Training initializes non-flat replicas at levels 0–1 by default, configurable
through `TerrainConfig(initial_level_range=(low, high))`. A shared 20 s episode
packet selects translation with probability 0.5, or standing, left/right pivot
and left/right arc with probability 0.1 each. Only an unchanged translation
episode is eligible for progression. Net XY displacement greater than half the
tile length promotes one level; displacement below half the full-horizon travel
implied by the translational command demotes one level, unless promoted. Other
episodes hold. The bottom clips at zero and the top holds at seven. This is a
distance heuristic, without tracking or obstacle-clearance gates.

Evaluation disables runtime progression and generates one row at an explicit
fixed physical level: easy 0, medium 3 or hard 7; the default is 3. Its native
row index is zero for all three severities. All eight levels are allowed during
training, so these are in-distribution severities, even if a particular learner
never reaches them. Held-out composition and out-of-distribution cases require
separate definitions. A root crossing a tile's ±6 m XY
boundary truncates the episode. Feet can extend beyond that boundary, and
neighboring tiles or central platforms must be disclosed when interpreting
surface exposure. Remaining on a spawn platform does not demonstrate locomotion
on the surrounding family. Geometry/family assignment is host metadata and task
progress, not an extra actor input or a training GT label.

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

Configuration identity includes serialized native callable module paths, so a
source relocation can change it even when numeric settings agree. Existing
progress records require their matching source configuration; restoration does
not silently migrate them.

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

## Frozen evaluation

Start `AppLauncher` in the consuming program, load the policy using its own
checkpoint format, then call the common evaluator:

```python
from parkour_lab.evaluation import case_bank, evaluate

def evaluate_loaded_policy(policy, app, output):
    return evaluate(
        policy,
        case_bank("validation", attempts=1),
        output,
        device="cuda:0",
        training_seed="training-seed-42",
        checkpoint_id="my-endpoint-checkpoint",
        app=app,
    )
```

A frame policy implements `reset(mask)`, `act(observation)` and
`state_sha256()`. It receives only the shared float32 `(N, 49)` causal frame;
`act` returns float32 `(N, 12)` encoded action requests. It owns feature
normalization and causal memory. The state fingerprint must include learned
parameters, buffers and preprocessing statistics, excluding evolving episode
memory. Freeze them before evaluation. The evaluator checks that fingerprint,
resets memory at episode boundaries and bounds targets through the same motor
transform used by the public task. An existing named `Controller` is also
accepted when its motor contract declares those same joint limits; named targets
outside them are rejected. Neither interface loads files or receives simulator
handles.

The evaluator owns a dedicated simulation stage and closes each case environment.
Call it with an application that has no active native environment; it creates a
fresh stage and discards old ray-mesh caches between cases. Native SDK
construction seeds process RNGs. A consuming trainer should evaluate in a separate
process, or preserve and restore its own learner RNG state. The benchmark does not
snapshot that state. The caller retains application and checkpoint ownership. `checkpoint_id` is an
opaque caller label; optional `checkpoint_sha256` identifies its actual file.
`repository_path` optionally adds Git evidence for an explicit checkout. Imported
package-source hashes are always recorded; an installed wheel need not live in
Git. Output directories are new captures and are never reused for another run.

The versioned development bank contains six profiles: stand, forward at
`(0.35, 0, 0)`, left/right pivots at `(0, 0, ±0.5)` and left/right arcs at
`(0.35, 0, ±0.3)`. Commands use body-frame m/s, m/s and rad/s. Every attempt
lasts at most 20 s: stand 0–2 s, named motion 2–14 s and stop 14–20 s. The stand
profile remains zero throughout. Report fixed onset 2–3 s, settled motion 3–14 s,
braking 14–15 s and stop/hold 15–20 s windows. Transients remain in the total
native core reward return.

`case_bank("train" | "validation" | "test", attempts=...)` gives disjoint
semantic seed identities. One attempt per cell produces 114 cases: one Flat
geometry plus easy/medium/hard Uneven, Slopes, Stairs and Hills; Slopes and Stairs
include both pyramid signs. Each geometry cell receives every profile. Flat has
no severity ladder. Geometry/start/dynamics/noise seeds are shared across its
profiles and are independent of execution order. Cases execute sequentially with
one replica; training remains batched. A pyramid sign is geometry, while recorded
support changes describe actual ascent or descent.

Scores retain original core return, planar and body-Z yaw RMSE, signed yaw rate,
wrong-direction fraction, unwanted pivot translation, stopping residual motion
and drift. World-heading diagnostics are separate from body-Z angular rate on
slopes. Every attempt ends at its first native termination or truncation; no
replacement reset episode contributes. Terminal reward is included once. Falls,
tile-boundary censoring, horizon survival and phase coverage remain distinct.
Missing phases are unavailable, never zero error. Missing cases prevent a complete
comparison rather than disappearing from its denominators.

Aggregate weights give equal shares to families, then their physical levels,
subtypes, profiles and attempts. Keep family and direction tables visible, and
summarize independent training seeds separately. A completed subset is labeled
separately from the full bank. This development recipe does not establish final
calibration or a ranking from an untrained fixture.

Captures include outgoing postphysics states, commands, per-step rewards,
termination causes, foot forces/support rays, physical realizations, motor/noise
receipts and source identities. These are evaluator-only measurements. Surface
exposure uses contact-supported height changes; a family label or a pivot on the
central platform does not demonstrate non-flat locomotion. No curriculum levels
advance and no training-information packet reaches inference.

The authoritative scorer is the same pure function used by native playback and
the offline command. Recompute from a saved capture without launching Sim or
loading a policy:

```bash
python -m parkour_lab cases --partition validation --attempts 1
python -m parkour_lab score CAPTURE_DIRECTORY
```

`score_capture` verifies capture consistency and retains incomplete/failed cases.
`capture_valid` identifies a consistently completed prescribed capture;
`valid_for_comparison` additionally requires the full partition bank. A six-case
engineering subset can have valid captures while remaining ineligible as a
full-bank comparison. Check these fields and coverage before interpreting aggregates.

## Architecture

The environment, inference controller and learner are separate responsibilities.

| Owner | Responsibility |
| --- | --- |
| `benchmark/` | Public learner-independent task API: configuration/facade, actor/GT contracts, spaces and checkpoint-linked progress. |
| `provenance.py` | Package hashes, source-owned dependency identity and Git evidence. |
| `control/` | Causal sensors, caller-loaded policy/controller adapters, command tapes and motor contracts. |
| `environments/` | Native task configuration, observations, dynamics, rewards and reset semantics. Isaac Lab owns physics, managers and step/reset order. |
| `environments/configuration.py` | Shared native Go2 scene, sensors, action and physical reset configuration. |
| `environments/commands/episode.py` | Shared training episode sampler and prescribed curriculum eligibility. |
| `environments/terrains/` | Native Flat/Uneven/Slopes/Stairs/Hills definitions, geometry bindings and per-replica curriculum. |
| `runtime/` | Native sensor/motor binding, transition accounting and ordered environment/application cleanup. |
| `evaluation/runner.py`, `environment.py`, `capture.py` | Frozen playback, dedicated case environments and portable outgoing-transition capture. |
| `evaluation/cases.py`, `scoring/` | Fixed case identities and authoritative physical scores/aggregates. |
| `tests/` | Behavioral contracts, package boundaries, analytic scores and geometry checks. |
| `scripts/check_public_task.py` | Runnable public-task consumer and bounded training/evaluation/continuation contract check. |
| `scripts/check_terrain.py`, `scripts/_terrain_checks/` | Bounded native geometry, contact, curriculum and recreation checks. |
| `scripts/check_evaluation.py` | Caller-format policy save/load, frozen native evaluation and independent offline scoring checks. |

`control.Controller` and `runtime.NativeControllerSession` accept a loaded causal
controller independently of a learner. `control.policy.PolicyController` adapts
the common 49D policy interface to named sensors and bounded joint targets.
The evaluation host supplies sensing and motor binding; a policy never receives
an environment handle.

Native stepping preserves separate termination/truncation flags and owned outgoing
final rows before automatic reset. Consuming trainers own optimization,
rollout/replay storage, inference memory and checkpoint loading. Shared control
and terrain code do not select a network architecture.

## Verification

### CPU checks

From the repository root in a CPU Python environment:

```bash
python -m pip install -e source/parkour_lab
python -m pip install scipy black==25.9.0 isort==6.1.0 flake8==7.1.1
python -m unittest discover -s tests -t . -v
black --workers 1 --check source/parkour_lab tests scripts
isort --check-only source/parkour_lab tests scripts
flake8 source/parkour_lab tests scripts
python -m parkour_lab --help
python scripts/check_public_task.py --help
python scripts/check_terrain.py --help
python scripts/check_evaluation.py --help
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
SciPy supports the native heightfield source checks; native execution obtains it
through the SDK. It is not an additional portable runtime requirement.

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

Check dependency metadata and archive contents as well as imports. The source archive must include `config/extension.toml` and this
README, and exclude private planning files. An editable installation alone does
not verify those contents. The CPU suite checks that portable scoring and public
task imports do not require learning libraries or a native simulator.

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

Successful checks do not establish robot stability, benchmark
scores or equivalence between simulators. Commands not yet run on the server are
not native evidence. Run relevant checks after task, controller, scoring or
continuation changes and retain complete failed attempts when assessing behavior.

### Terrain and curriculum checks

This command runs three fixed-severity Mixed views and training/recreation in
five fresh native processes, with 20 replicas covering every terrain column.
The geometry probe reads the actual USD collider and raycasts an interior grid
from −5.4 to +5.4 m in both axes. It saves surface arrays, initial root/foot link
clearances, height/roughness measurements, signed pyramid profiles and edge hits.
These measurements describe the sampled extent; they do not certify every
triangle or a robot's ability to traverse it.

The contact search uses near-default actions for at most 160 control calls,
followed by one tile-edge fixture step. The check
requires a nonzero contact sample on a Flat replica after at least 32 calls,
then captures its outgoing force frame before a forced timeout. Training compares
that frame with the separately aligned final GT. A controlled tile-edge pose
checks truncation and reset, followed by a list-based partial reset that preserves
other replicas, assignments and interaction totals. The training host also writes
declared poses and command packets to exercise promotion, demotion, holds,
changed packets and bottom/top cases through the actual curriculum manager. These administrative
fixtures measure rules, not policy performance; only real steps charge interactions.

```bash
bash <<'BASH'
set -uo pipefail
PLAB_PY=${PLAB_PY:-python}
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/terrain_check_XXXXXX) || exit 1
trap 'status=$?; printf "{\"returncode\": %d}\n" "$status" > "$RUN/process_exit.json"; echo "Results: $PWD/$RUN"' EXIT
run_check() {
    local name=$1
    shift
    "$PLAB_PY" -u scripts/check_terrain.py --device cuda:0 --seed 1077 \
        --output "$RUN/$name" "$@" > "$RUN/$name.log" 2>&1
    code=$?
    printf '{"returncode": %d}\n' "$code" > "$RUN/${name}_process_exit.json"
    return "$code"
}
for level in 0 3 7; do
    run_check "level_$level" --role eval --level "$level" || exit "$?"
done
run_check train --role train || exit "$?"
run_check resume --role train --resume "$RUN/train/progress.json" || exit "$?"
BASH
```

Each process must exit zero, report `TERRAIN_CHECK_PASSED` and close both resources.
The fresh resumed task must recover levels, stable columns, sampler/report and
cumulative interactions before resetting fresh episodes. The checkpoint token
is a fixture; no learner checkpoint is restored. Inspect `report.json`,
`surface_readback.npz`, `contact_witness.json` and the training `progress.json`
alongside source/dependency receipts. The script keeps `calibration_validated`
and `behavior_validated` false: returned native measurements and an explicit
review are still needed before accepting candidate geometry or learning difficulty.


### Loaded-policy evaluation checks

This command checks a caller-format policy save/load, the fixed evaluation path
and a fresh task-progress restart in three native processes. The evaluation
fixture is a tiny untrained policy with frozen normalization; it is not a trained
locomotion baseline. Its six-case engineering subset spans all five terrain
families, all six profiles and physical levels 0, 3 and 7. It is explicitly not
the full validation bank or a balanced method-comparison result.

```bash
bash <<'BASH'
set -uo pipefail
PLAB_PY=${PLAB_PY:-python}
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/evaluation_check_XXXXXX) || exit 1
trap 'status=$?; printf "{\"returncode\": %d}\n" "$status" > "$RUN/process_exit.json"; echo "Results: $PWD/$RUN"' EXIT
run_check() {
    local name=$1 script=$2
    shift 2
    "$PLAB_PY" -u "$script" --device cuda:0 --output "$RUN/$name" "$@" \
        > "$RUN/$name.log" 2>&1
    code=$?
    printf '{"returncode": %d}\n' "$code" > "$RUN/${name}_process_exit.json"
    return "$code"
}
run_check train scripts/check_public_task.py --role train --num-envs 4 --seed 1077 || exit "$?"
run_check resume scripts/check_public_task.py --role train --num-envs 4 --seed 1077 \
    --resume "$RUN/train/progress.json" --checkpoint-id no-learner-contract-fixture || exit "$?"
run_check evaluation scripts/check_evaluation.py || exit "$?"
BASH
```

The task processes must report `TASK_CHECK_PASSED` and the evaluation process
`EVALUATION_CHECK_PASSED`, all with zero process status and complete application
cleanup. The evaluator closes every native case environment; its case receipts
must each show complete cleanup. The outer evaluation script owns only the app.
It checks exact command timing, one sensor-noise frame per policy action,
fixed physical levels, absent training GT and unchanged policy parameters and
normalization. At least one non-standing case must complete all 1,000 steps;
`phase_boundary_witness.json` retains commanded and outgoing physical samples
around 2, 3, 14 and 15 s and the final step. If none reaches the horizon, the check
fails rather than claiming phase coverage. Its fresh offline subprocess blocks native and learning-library
imports, invokes the actual scoring CLI and requires identical JSON results.
Retain the entire returned directory, including failures and dependency/source
receipts. Native success still requires review of those measurements.

Add `--full-bank` to a separate `scripts/check_evaluation.py` invocation with a
new output directory to capture all 114 validation cases. Both modes stop each
attempt at its first ending; the full bank can therefore take up to 38 simulated
minutes, plus environment-construction overhead. The fixture's performance is
not a benchmark baseline. Training-progress recreation restores task counters
and samplers only; genuine learner checkpoint continuation belongs to the
consuming trainer.
