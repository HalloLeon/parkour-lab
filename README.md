# Parkour Lab

An Isaac Lab / Unitree Go2 locomotion research framework with ROA-like
teacher/student learning by default and interfaces for future learning methods. One workflow
covers training, continuation, export and command playback. **Successful training or playback
does not establish robot qualification.**

## Setup and commands

Use the interpreter already configured for Isaac Lab and RSL-RL:

```bash
python -m pip install -e source/parkour_lab
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
python -m parkour_lab --help
```

The development environment used Isaac Lab 2.3.2.post1, Isaac Sim 5.1, and RSL-RL
3.1.2. CPU tests are not simulator validation. There is no GPU discovery, leasing or
exact-wheel admission gate. Select the device explicitly.

| Command | Purpose |
| --- | --- |
| `python -m parkour_lab train` | Train the configured method; optionally continue a current learning snapshot |
| `python -m parkour_lab export CHECKPOINT ACTOR` | Export only the method's causal inference state |
| `python -m parkour_lab evaluate ACTOR` | Frozen headless command playback and diagnostic tracking |
| `python -m parkour_lab play ACTOR` | The same playback with a visible simulator, one environment by default |
| `python -m parkour_lab analyze RUN` | Print a run's result; `--flat-bank` aggregates development groups; `--terrain` scores one saved connected-world attempt |

For training and playback, `--config FILE` accepts explicit settings, not an old
experiment manifest.
Unknown fields are errors. CLI overrides include `--device`, `--num-envs`,
`--seed`, and training `--updates`. For example:

```json
{
  "task": {"terrain": "procedural", "num_envs": 160, "seed": 42, "device": "cuda:0"},
  "method": {"name": "roa", "options": {"history_interval": 5, "regularization_coef": 0.1}},
  "updates": 1000,
  "save_interval": 100
}
```

Task defaults live in `config.py`; method defaults/validation belong to their
backend. Every run records the resolved configuration. Terrain choices are flat,
rough, procedural, steps, traversal and connected. These
are development fixtures, **not the approved mixed-world qualification bank**.
Traversal and connected are evaluation-only. Current step fixtures use their measured three-row
range; procedural settings expose difficulty range and one, three or five rows.

Training produces `checkpoint_NNNNNN.plab` and streaming `metrics.jsonl`.
It also saves `initial_learning_state.payload` before the first update, after any
checkpoint restoration. This is the selected backend's model/optimizer payload,
not an executable model or a simulator/RNG snapshot. The report records its path,
update count and SHA-256. Matching payload hashes establish identical bytes;
different device/library encodings can require comparing decoded state values.
`train --checkpoint FILE --updates N` restores method-owned learning state and
performs N additional collection/update cycles. It starts a fresh simulator,
episode memory and seeded RNG; it is not exact interrupted-rollout resumption.
Method settings must match. Reports distinguish transitions, elapsed training time
and method-specific gradient/adaptation counts; equal cycles are not equal compute.
When an artifact is supplied, a task-only configuration overrides its task fields
while retaining its method/options. Explicit conflicting method settings are rejected;
the report and dependency identity always describe the method actually executed.
All commands create separate run directories under `--output-parent`.
Checkpoint and actor outputs refuse to overwrite existing files.

### Progressive rough-terrain training

`task.terrain="rough"` uses Isaac Lab's Go2 rough terrain: stairs in both
directions, slopes in both directions, boxes and uneven ground. It adds plane
columns for 20% flat practice. Defaults are ten difficulty rows over `(0, 1)` and
initial row zero. The recorded native training run actually began on row one,
consistent with curriculum processing during the initial reset; that initialization
mismatch remains to be corrected. Playback starts on row zero. The upstream
distance curriculum moves robots between levels. Stairs, boxes and slopes grow harder with row difficulty;
the stock uneven-ground noise stays fixed. Flat practice provides training
coverage; the separate flat bank still measures retention.

The actor keeps its 49 causal inputs and history. Terrain heights and simulator
linear velocity remain training-only information. The mode reuses the bounded
dynamics, sensor noise, Go2 motor law and 50 Hz control. Rewards retain the shared
recipe and any explicit `task.rewards` overrides; selecting rough terrain does
not silently replace the objective with upstream reward weights.

Training commands change every four seconds. Moderate terrain uses uniform
`vx ∈ [-0.2, 0.5]`, `vy ∈ [-0.2, 0.2]`, `wz ∈ [-0.5, 0.5]`, with 10% exact
stops. Rows that can exceed an 8 cm step or a 10° carrier incline use forward-only
`vx ∈ [0.2, 0.5]`. The step bound includes differences between neighboring box
heights; the slope bound includes the pyramid shape and height rounding. Each
row uses its upper difficulty bound, so boundary rows can receive the narrower
command set early. Uneven ground retains turns and stops: its local roughness is
distinct from the underlying carrier incline. These training distributions use
the assigned tile, rather than measurements of the robot's current surface. External playback
commands bypass this sampler.

`task.rough_pivot_fraction` optionally replaces part of the moderate-terrain
mixed draws with exact turns in place. It defaults to `0.0`, preserving the
original distribution. At `0.2`, eligible draws are 10% stops, 20% pure pivots
and 70% mixed twists; pivots use both yaw signs with magnitude 0.3–0.5 rad/s.
Restricted forward-only assignments and the four-second interval are unchanged.
This setting is used only by the rough training sampler. Its `pivots` count is
included in the existing command-sampling report; external playback remains
independent of the training mixture.

Training starts from fresh initialization or this project's own checkpoint
lineage. Never initialize or continue training from another project's checkpoint;
external checkpoints may be used for frozen evaluation references only.

When transferring our own flat checkpoint, supply the terrain settings explicitly;
unspecified fields retain the checkpoint's configuration:

```json
{"task": {"terrain": "rough", "num_rows": 10, "difficulty_range": [0.0, 1.0]}}
```

Compatible continuation retains model weights, optimizers and update counters.
The upstream curriculum uses travel distance and the final command as a progress
proxy; it does not certify tracking or traversal, particularly during stops and
turns. `metrics.jsonl` records the number of environments at each terrain level;
the final report includes command sampling counts. Playback freezes level changes.
The native `fresh_flat_rough_oLzXNt` comparison completed 20k fresh flat updates
and 20k rough updates. First-episode rough physical failures fell from 63/160 to
5/160, but flat forward passes fell from 98/100 to 40/100 and pivot-left from
84/100 to 0/100. Matched posture penalties of −0.1 and −0.2 reduce rough physical
failures to zero and improve flat forward passes to 66/100 and 76/100, but pivots
pass only 0/100 and 4/100. Posture and terrain progress improve while turning in
place remains unresolved. Explicit pivot practice is the next unrun comparison;
the recipe is not a validated default. This training layout remains separate
from connected-world qualification.

### Simulation timing candidate

`task.physics_hz` defaults to `200`; the only alternative is the approved `400`
candidate. Both use 50 Hz commands/actions and the same 25-frame causal history.
Physics uses four/eight substeps, contact sensors update each substep, and native
collision-history lookback remains 10 ms (three/five samples). Actor contact flags
still use only the latest sample. Rendering and height scans remain at 50 Hz.

Artifacts bind their exact physics timing. A config override cannot run a 200 Hz
actor or resume its training at 400 Hz. To evaluate unchanged weights at the new
rate, explicitly export a **new, unvalidated candidate** from a current checkpoint:

```bash
python -m parkour_lab export CHECKPOINT.plab NEW_ACTOR.plab --physics-hz 400
```

Omit the flag to retain the checkpoint rate. Save the printed receipt: it records
the source checkpoint, source/target timing and motor identities, and new actor
hash. The source checkpoint is never modified. Export preserves inference weights,
not closed-loop behavior or qualification. This setting does not configure a
hardware servo or change the 50 Hz policy interface; changed simulated dynamics
and contact inputs can nevertheless affect behavior and eventual sim-to-real
transfer. The retained 200 Hz flat anchor remains the comparison baseline. No
terrain training or production adoption is implied by candidate export.

### Reward configuration and diagnostics

`task.rewards` overrides only named weights and the numeric parameters below.
Omitted terms keep their defaults; weight `0` disables computation of a term.
Positive rewards cannot become penalties or vice versa. Unknown names, unsupported
parameters, booleans and nonfinite numbers are rejected before simulation starts.

| Term | Default weight | Meaning before weighting |
| --- | ---: | --- |
| `track_lin_vel_xy_exp` | 1.5 | Exponential body-frame root-COM xy tracking |
| `track_ang_vel_z_exp` | 0.75 | Exponential body-z yaw tracking |
| `lin_vel_z_l2` | −2 | Squared vertical COM velocity |
| `ang_vel_xy_l2` | −0.05 | Squared roll/pitch angular velocity |
| `dof_torques_l2` | −0.0002 | Sum of squared applied joint torques |
| `dof_acc_l2` | −2.5e−7 | Sum of squared joint accelerations |
| `action_rate_l2` | −0.01 | Squared change in raw joint actions |
| `feet_air_time` | 0.01 | First-contact air time minus threshold; only when commanded xy speed >0.1 m/s |
| `flat_orientation_l2` | −2.5 | Squared projected-gravity xy components |
| `dof_pos_limits` | −10 | Joint excursion beyond soft limits |
| `joint_posture` | 0 | Optional unsquared joint deviation from the default pose |

The new posture term follows [Unitree's Go2 reward](https://github.com/unitreerobotics/unitree_rl_lab/blob/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py).
Its multiplier is `stand_still_scale` (default 5, minimum 1) only when the complete
`[vx, vy, yaw_rate]` command is zero and measured planar COM speed is at most
`velocity_threshold` (default 0.3 m/s, minimum 0). A pure pivot is not a stop.
`feet_air_time.params.threshold` defaults to 0.5 s and must be nonnegative;
short swings can contribute negatively even with a positive weight.

`task.linear_tracking_std` (m/s) and `task.angular_tracking_std` (rad/s) are the
only tracking-width settings; both default to 0.5. Their kernels are
`exp(-squared_error / std**2)`. Do not put `std` under reward overrides.
For example, this task-only file changes two weights and disables one penalty:

```json
{
  "task": {
    "rewards": {
      "feet_air_time": {"weight": 0.25},
      "joint_posture": {"weight": -0.7, "params": {"stand_still_scale": 5}},
      "action_rate_l2": {"weight": 0}
    }
  }
}
```

This is a configuration example, not a recommended trained baseline. When loading
an artifact, omitting `task.rewards` retains its overrides; supplying it replaces
the entire override map. Use `"rewards": {}` to restore defaults. Other task fields
and the learner remain unchanged. Weights and tracking widths change the training
objective, not motor limits, commands, observations or evaluation thresholds.

Every native locomotion report includes the actual manager's `reward_recipe`:
weights, functions, formulas, parameters and control timestep. Native rewards are the
signed sum of `weight * term * dt`, without clipping. `metrics.jsonl` adds
`task_metrics`, grouped by learning phase and outgoing command regime. Initial
stops are distinguished from stops after any movement command in the same episode;
reset starts that history again. Counts and simulated seconds sum across robot
instances, not wall-clock time. Each term records a weighted rate mean and its
dt-weighted contribution sum; contributions reconcile with the native reward.
Physical means use post-physics, pre-reset truth for xy/yaw tracking, height above
the surface beneath the base, joint-posture deviation and tilt. Termination and
timeout counts remain separate, including overlaps. PPO and student-history
metrics are not mixed; optimizer/auxiliary losses remain method-owned. Evaluation
reports the same diagnostics separately from its first-attempt behavioral scores.

Compare frozen causal behavior, not total reward across different objectives.
Training diagnostics do not establish a passing locomotion policy.

Evaluation/play accept either `--tape FILE` or `--command VX VY WZ --steps N`.
The latter includes a final one-second stop (all stop when N <= 50).
Successful playback records the commands actually delivered in `commands.json`;
replay preserves their sequence, not stochastic physics. Physical keyboard,
joystick and Unitree transport belong in the deployment application.

### Flat command diagnostics

`evaluate ACTOR --profile NAME` runs a fixed 11 s profile: 2 s stop, 6 s motion,
3 s stop. `stand` is scored as one uninterrupted stop. Names are `stand`,
`forward-02`, `forward-05`, `reverse-02`, `left-02`, `right-02`, `pivot-left`,
`pivot-right`, `arc-left`, and `arc-right`. The actor's resolved task must be flat,
with an episode longer than 11 s. Profiles cannot be combined with `--tape`,
`--command` or `--steps`; they use the same causal inference/motor path as playback.

`report.json` contains per-row, per-phase `flat_diagnostic` results. Tracking
uses 0.4 s blocks, including the acquisition block ending at 1 s; good episode
averages cannot hide a failed command. Both body-z and world-up yaw, initial/final
stops, posture, stationary excursion and drift are checked. Only the original
attempt counts: a native termination/timeout remains a failure after auto-reset.

All evaluation also writes `motion.npz`: post-physics, **pre-reset** root pose,
root-body COM velocity, body/world angular velocity, body-link positions and base
contact force, with initial pose anchors, environment origins, commands and native
ending flags. It is evaluator-only truth. Existing `tracking.npz` keeps its
pre-action tracking/contact semantics. Motion capture is disabled during training.

For ROA, `tracking.npz` also records `estimated_linear_velocity_b` and
`root_com_linear_velocity_b`: aligned body-frame COM velocity estimate and truth
in m/s, shaped `(control, environment, 3)` with components `(vx, vy, vz)`.
The estimate is the one used to choose that control's action; truth is sampled
before the physics step and is never supplied to the actor. The existing
`root_com_velocity` field remains `(vx, vy, yaw_rate)`.

`--profile` is a diagnostic using the configured batch and seed. For a frozen
development group, use `--bank-profile NAME` instead. It fixes 100 independent
attempt IDs: 50 nominal and 50 randomized, with stratified world headings in each
subset. Do not supply `--seed` or `--num-envs`. Every group records `manifest.json`
with full SHA256-derived PCG64 stream identities, starts and dynamics. Geometry,
start, dynamics, command and observation-noise streams are separate from learning.

Run each of the ten groups once with the same actor and code, placing the outputs
under one directory, then run `python -m parkour_lab analyze DIRECTORY --flat-bank`.
Analysis checks saved evidence hashes, canonical manifests, command tapes, sensor
noise and motion before recomputing scores. Each group needs ≥90/100 overall and
≥45/50 in each stratum. Missing groups remain nonpasses; duplicate groups are
rejected rather than selecting the best retry. Later reset episodes cannot replace
a failed first attempt. This is **development evidence, not held-out qualification**.
The retained 200 Hz anchor and its same-weight 400 Hz export pass the development
flat bank; terrain acceptance remains open.

Domain checks use root/body-link centres in a local ±6 m square, not collision-volume
extents. Reports always retain `qualified=false` and `qualification_eligible=false`.

### Connected source geometry

The simulator-free `environments.worlds.build_world` API constructs a single
connected diagnostic mesh containing stairs, a perpendicular ramp, a rounded hill
and rough connecting ground. One assigned encounter supplies prospective start,
landing and stop regions for the fixed 30 s tape; these are geometry annotations,
not policy goals or a frozen evaluation bank.

```bash
python - <<'PY'
import json
from parkour_lab.environments.worlds import build_world

mesh = build_world("ramp", 20, coarse_seed=17, fine_seed=23)
print(json.dumps(mesh["metadata"], indent=2, allow_nan=False))
PY
```

Carrier profiles and roughness generation are shared with the local fixtures.
Roughness is applied after assembly; start/stop pads and actual riser edges are
the only taper regions. The returned arrays retain carrier/layer witnesses,
support faces and regional caps. Metadata measures each full-height feature,
source slopes, roughness and nominal start/yaw-envelope margins. Infeasible
geometry raises before use, without changing seeds or limits.

Vertices use local coordinates. `world_yaw` records a rigid column-vector
local-to-world transform and rotates the annotated start pose; it does not bake
rotation into the mesh. The nominal motion envelope is a construction check,
not a guarantee about robot trajectories or collision-volume clearance. Local
float32 diagnostics are not native USD/PhysX evidence. Native checks are separate
server commands supplied when needed, not permanent diagnostic-only runtime
modules. Existing terrain paths remain unchanged.
The world builder reuses analytical common roughness scaling at 25°, quietly
accepting small angle rounding. Height caps and layer-RMS floors remain strict;
failures do not redraw noise. No exact floating-point search is needed.

At tread-range boundaries, a flight may need a common float32 coordinate lattice
to keep its realized treads in range. Metadata records requested and realized
entry/tread dimensions and any correction; already legal coordinates are retained.
Converted spacing, feature dimensions, topology and roughness witnesses are
checked separately. These checks still do not establish native collider behavior.

### Connected-world robot playback

`evaluate` and `play` also accept `task.terrain="connected"` with a `task.world`
object containing the keyword arguments to `build_world`. This initial integration
supports **one robot**, `num_rows=1` and `difficulty_range=[1,1]`; training is
rejected. Specify those fields explicitly in a task-only JSON when overriding an
actor's saved flat configuration. Use the existing `--config` and `--tape` options;
no separate diagnostic command is installed.

The scene imports one connected mesh, preserves its local float32 points and rigid
world yaw, and uses the measured flat start-pad height plus stock root clearance.
Start jitter rotates with the assigned heading. Bounded motors, dynamics and
causal sensor noise use the same path as flat playback; the actor must retain its
exact physics-rate binding. No plane, steering or terrain information is added to
the actor. Single-world operation avoids the stock ray caster's first-mesh limit.

Connected playback records `world.npz` and source/native identity in `report.json`.
`motion.npz` aligns named foot positions, net normal forces and terrain rays at
each completed control step, preserving outgoing terminal state before reset.
`tracking.npz` remains pre-action. Forces are the latest physics sample, not a
substep contact history; foot link origins are not collision surfaces.
Both ray sensors use lazy updates with period zero: physics invalidates the cache,
and the first control-boundary read refreshes it. Repeated reads reuse that sample;
this avoids accumulated float32 timestamp drift skipping nominal 20 ms updates.
Playback stops at the first termination or truncation, saves the consumed partial
tape, and reports `FIRST_ATTEMPT_ENDED_NOT_QUALIFIED`. Clean completion also remains
unqualified. This is operational integration, not a terrain bank or training release.

The first native run, `robot_world_PR8px3`, settled on the elevated pad but ended
at 11.12 s with 45.79° tilt after accelerating down the ramp. Shutdown completed;
the failed first attempt remains preserved. It also exposed 78 stale ray frames
after 8 s, motivating the lazy-update correction above. The terminal ray was fresh,
so this sampling defect does not excuse the tilt failure. The same-input
`ray_refresh_W0XdIg` verification now has fresh rays at all 556 control samples,
including all 156 after 8 s. Non-ray motion and all tracking arrays are identical
to the original: the robot still terminates at 11.12 s. This verifies sampling on
that trace, not full-horizon terrain capability; terrain learning is not released.

Score saved connected-world evidence without starting the simulator:

```bash
python -m parkour_lab analyze RUN --terrain
```

This read-only analysis checks artifact hashes, fixed-policy/delivery/cleanup
receipts and first-attempt consistency, then scores approach-side entry, crossing,
tracking, falls, stalls, map bounds and the final stop. A pass requires the full
prescribed 30 s terrain tape; short technical checks remain explicit nonpasses.
Crossing uses base/foot-link centres, not reconstructed contacts. The result always
remains unqualified: independently assigned layout/dynamics banks and retention
are separate requirements. `--terrain` and `--flat-bank` are mutually exclusive.
Inspect `diagnostic_passed` and `failures`: a successful analysis process can
correctly report a failed robot attempt.

### Prospective terrain development assignments

`environments.randomization.terrain_development_assignments()` returns the
requested cases for all 20 terrain groups, with 100 fixed IDs per group. Passing
a canonical group ID, such as `ramp/down/10deg`, returns just that group's same
100 assignments. There is no candidate seed or qualification-namespace option.

Each 50-attempt dynamics subset has balanced target dimensions and stratified
world yaw; stair treads use independently shuffled 50-bin Latin hypercubes. Six
full-SHA seed mappings separate geometry, coarse/fine roughness, starts, dynamics
and observation noise. A common local placement offset of up to 0.25 m varies
feature centres; it does not independently rearrange the surrounding structures.
Roughness seeds retain all 256 bits, rather than using the task's uint32 run seed.

These are **prospective assignments, not a validated bank**. Generating them does
not build meshes, apply dynamics or run the policy. One assignment can now be
selected explicitly through the existing connected evaluator:

```bash
python -m parkour_lab evaluate ACTOR.plab \
  --terrain-group ramp/down/10deg --attempt-index 50 --device cuda:0
```

Run this only for an unstarted development ID with the source and actor frozen.
The selector fixes the canonical world, one robot, global attempt ID, dynamics
stratum and independent full-entropy streams. IDs 0–49 are nominal; 50–99 are
randomized. The run seed cannot substitute for the assigned stream seeds. The
actor's physics rate remains exact; selecting terrain does not rebind it.

The command tape is always 2 seconds stopped, 25 seconds body-forward at 0.35 m/s,
then 3 seconds stopped, at 50 Hz. Custom commands/tapes/lengths, seeds, batch counts,
flat profiles and conflicting world/dynamics settings are rejected. Configuration
records derive the world/stratum and round-trip, but execution also requires the
explicit evaluate selectors. Training and play cannot execute development IDs.
There are no turn commands in this tape: one obstacle is tested at a time. Rotating
the world/start heading does not command a turn. Pivot and arc profiles cover
turning separately in the flat bank; an observed heading change with zero yaw
command is tracking drift.

The evaluator archives the selected assignment, actual source/native mesh identity,
supported first reset, physical readback and consumed causal noise. It stops at
the first native ending and retains the partial tape; a reset supplies no retry.
`analyze RUN_DIRECTORY --terrain` checks the canonical binding, saved physical
receipts and noise before returning the assigned group/index alongside kinematic
results. This does not qualify a complete bank or certify cooked contacts.

The first assigned native case, `ramp/down/10deg` index 50, has a valid capture and
binding, but terminates at 10.82 s with 45.03° tilt before clearing the ramp or
reaching the final stop. Clean process exit means the capture completed, not that
the robot passed. Broader native coverage, aggregation and retention remain open.
Keep the generated assignment file and its source identity when preparing a bank;
do not replace cases after observing robot outcomes.

## Ownership and boundaries

| Package/module | Responsibility |
| --- | --- |
| `config.py`, `__main__.py`, `experiment.py` | Typed settings, public commands, lifecycle and training orchestration |
| `environments/` | Explicit native configuration, terrain, observations, command sampling and physical termination |
| `methods/roa/` | ROA observation adapter, teacher/student model, PPO/history-fitting schedule/state, numerical serialization and causal controller |
| `methods/base.py`, `methods/__init__.py` | Small backend/learner contracts and named plugin loading |
| `methods/models.py` | Torch neural components used by ROA; not required by other frameworks |
| `control/` | Portable command tapes, sensor/action contracts and controller session semantics |
| `runtime/` | Shared training ticks, native sensor/motor binding and cleanup |
| `artifacts.py`, `provenance.py` | Method-neutral artifact envelope and source/dependency/run receipts |
| `evaluation/` | Frozen command playback and diagnostic metrics |
| `tests/` | Component-organized regression, contract, geometry and CPU integration checks; see [tests/README.md](tests/README.md) |

There is no legacy task registration, gap curriculum, waypoint navigation, old
teacher pipeline, experiment-specific executable or compatibility alias.
Package imports do not launch Isaac Lab. ROA inference does not import PPO.

`environments/runtime.py` retains separate termination/truncation flags and owned
pre-reset observation rows. Final samples retain the outgoing command: capture
precedes reset, command resampling and interval events. They never replace the
next-action reset observations. Native observation terms must be stateless;
history, filters and stateful preprocessing belong in the learner adapter.
Flat native groups remain clean: noise is applied once at the shared causal-input
boundary. Terminal-only reads do not consume that stream. Other diagnostic terrains
retain isolated native observation noise; they are not yet acceptance-aligned tasks.

ROA uses the final clean state/terrain only for timeout value bootstrapping;
this does not add privileged student inputs or push causal history twice.
True termination takes precedence over a simultaneous timeout. This intentionally
replaces RSL-RL's previous-state timeout estimate with the final-state estimate;
it is a learning correction, not a claim of identical historical training.
Reports count captured final rows and metrics count timeout bootstraps per run.

## Method extension boundary

ROA is the only implemented learner. The shared host provides observations, reset
flags, owned terminal samples and motor delivery; the adapter owns learning.
Native `proprio` is causal; the native group named `policy` contains true velocity
and is privileged. An adapter must not expose privileged observations to its student.

Each `.plab` file is a ZIP with JSON metadata and a checksum-bound numerical
payload owned by the method. ROA uses Torch weights-only loading, not pickled
Python model objects. Hashes detect changes, not trust
or authenticity. This current format replaces ROA-only `.pt` snapshots.

An external adapter implements `MethodBackend` in `methods/base.py` and registers
an installed `parkour_lab.methods` entry point, for example in its `pyproject.toml`:

```toml
[project.entry-points."parkour_lab.methods"]
my_method = "my_adapter.backend"
```

When adding a method, pin its upstream repository as a Git submodule, install it and its adapter, and
select `method.name = "my_method"`. Plugin names cannot shadow built-ins. Installing
a plugin explicitly trusts its code; checkpoint metadata cannot import arbitrary
Python paths. Dependencies must be available before native startup; receipts record
the imported version, path, source hash and Git commit when present.

The learner owns `advance()`, its schedule, memory/replay, gradients and snapshot
state. The backend owns option validation, creation, numerical serialization,
causal export and the controller; the host never inspects a Torch optimizer or
PPO rollout. `Controller.state_sha256()` fingerprints trained state, including
learned preprocessing but excluding per-episode recurrent memory.

For a future DreamFLEX adapter, its upstream teacher/student and adaptation
schedule stay in the method, and its causal inference graph implements Controller.
For Dreamer-like training, the adapter converts native tensors to its framework,
owns sequence replay/world-model/imagination updates and persists those states.
It uses final observations and reset flags to prevent replay sequences crossing
episodes; its controller owns recurrent inference/reset state. These responsibilities
are designed to avoid PPO hooks or edits to shared tasks, commands, motor delivery
or evaluation. No DreamFLEX/Dreamer implementation or behavioral equivalence is
claimed here. Contract tests and architecture review are not empirical proof of
multi-learner interoperability; a real second learner is not required for acceptance.

## Breaking change and evidence

Old checkpoints, experiment manifests, baseline admission chains and script
commands are intentionally unsupported. No converter is provided.
Keep archived runs unchanged and use their original checkout for historical
reproduction. The current workflow starts fresh or loads artifacts it produced.

Source hashes are evidence, not a demand that every run use identical source.
Commit new executable files before server runs: the Git receipt includes tracked
changes but does not archive untracked file contents. No test/documentation files
need to be synced to run the package commands.

The current `.plab` format is `parkour_lab_method_v3`, with a nominal motor identity
independent of bounded realized gains/strength. Earlier artifacts, including 49-D
v2 files, are unsupported. Start fresh; do not convert archived weights or mix
older runs into matched reward comparisons.

## Default ROA recipe

The causal frame has 49 values: angular velocity, projected gravity, applied
command, relative joint positions, joint velocities, previous raw action, then
four binary foot contacts in **FR, FL, RR, RL** order. Each flag is
`norm(net_forces_w[foot]) > 1.5 N`, using the latest completed physics sample
(200 Hz default, 400 Hz candidate)
at the 50 Hz action boundary. This is a native net-normal-force vector, not a
six-axis wrench or the sum of individual contact magnitudes. Contact flags have
no noise/filter/delay and remain zero until the first completed post-reset step.
Reset history repeats that cleared frame; surviving rows keep their own history.
The deployment controller uses the same detector. Evaluation's `tracking.npz`
also records ordered force vectors, flags and reset masks for auditing; raw
forces are diagnostics, not student inputs.

| Setting | Go2 default and rationale |
| --- | --- |
| Actor/history | 49 values, 25 frames including the current frame (0.48 s span); longer context retained for the velocity estimator |
| Learning cycle | 24 privileged PPO steps; every fifth update adds 64 causal steps and 4×4 supervised minibatches; frequent fitting of the evolving teacher |
| Optimizers | PPO Adam `2e-4`, five epochs/four minibatches; supervised Adam `1e-3` |
| Regularization | `0.1 * clip((completed_updates - 3000) / 7000, 0, 1)`; warms up before aligning the teacher to the student |
| Latents/supervision | Eight-dimensional dynamics latent, separately directed unsquared L2 alignment, plus supervised three-dimensional COM-velocity estimation |
| Networks | Motor ELU MLP 128/128/128 with additive latent projection; estimator 128/64; asymmetric terrain-conditioned critic |
| Exploration | Learned action standard deviation initialized at 1.0; entropy coefficient 0.01, no minimum-std clamp |
| Control | Stock Go2 motors; `q_target = default_q + 0.25 * raw_action`, no action clipping; 50 Hz control, 200 Hz default / 400 Hz candidate physics |

The regularization endpoints follow the **non-resume branch**, not the enabled
resume branch, of the [pinned author configuration](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py).
`regularization_start_update` and `regularization_end_update` are configurable;
both zero selects a constant coefficient. Continuation derives the coefficient
from the restored total update count, so it does not restart the ramp.

This is an **ROA-like Go2 port, not an exact reproduction**. The
[ROA supplement](https://proceedings.mlr.press/v205/fu23a/fu23a-supp.pdf) uses ten
history frames, interval 20 and a different 0→1 ramp. This port instead retains
additional history-collection blocks, explicit velocity supervision, a default-on
12-D privileged force extension (`contact_conditioned`), an asymmetric terrain
critic, and Go2-specific networks, rewards, noise, resets and action scaling.
Both actor routes use estimated rather than true COM velocity. Student inputs
remain causal; teacher force labels are separate from the four binary contacts.
No observation normalization is learned. The entropy/noise settings also differ
from upstream. Equal update numbers do not imply equal samples or training effort.

Flat training draws independent uniform body twists every exactly 200 control
ticks (4 s): vx ∈ [−0.2,0.5] m/s, vy ∈ [−0.2,0.2] m/s and yaw ∈ [−0.5,0.5] rad/s.
Ten percent of draws become exact stops; no heading controller changes the packet.

Flat training uses randomized dynamics. Per robot, added base mass is U[−1,3] kg
with the stock mass-scaled inertia rule; static friction is U[0.6,1.0] and dynamic
friction is 0.75 times static. Ground coefficients are 1/1 with multiply combination.
Independent per-joint strength, Kp and Kd factors are U[0.9,1.1], drawn once and
fixed across resets. Strength scales generated torque **before unchanged physical
torque/speed limits**. Reports verify native mass/material/inertia/gain readbacks
and retain the nominal motor contract separately from realized values.

Both flat evaluation strata use bounded starts and causal sensor noise. Starts
have local xy ±0.05 m, yaw ±0.03 rad around the prescribed heading, joints ±0.05 rad,
zero velocities and level attitude. Noise amplitudes are angular velocity ±0.2 rad/s,
gravity ±0.05, joint position ±0.01 rad and joint velocity ±1.5 rad/s. Commands,
previous actions and binary contacts stay uncorrupted. `initial_inputs.npz` records
training's first raw/delivered frame; evaluation's `tracking.npz` records noise draws
and the first raw/delivered sensor sample. Replaying the same manifest resets these
streams, not the physics engine's numerical nondeterminism.

For ordinary flat diagnostics, `task.dynamics` can select `nominal` or `randomized`;
the development bank always supplies its fixed 50/50 split. These corrections
currently apply only to flat tasks. Rewards and learner settings are unchanged.

## Acceptance

Project targets are stairs with realized risers 4/8/12/16 cm, ramps/hills at
10/15/20 degrees, and multiscale unevenness on structures and connecting ground.
One causal policy must cross the assigned obstacle, track commands and stop
stably. Each of 20 terrain and 10 flat groups has 100 first attempts, balanced between
nominal and randomized dynamics; every group needs ≥90/100 overall and ≥45/50
in each subset. Later candidates must also retain flat performance: at most 5
percentage points pass loss and 0.02 m/s planar RMSE increase per profile.
No gaps, steep backward traversal or sim-to-real qualification is required.
These are ROA-informed project targets, not official ROA benchmark claims.

The causal flat anchor is accepted; connected terrain contacts, traversal and
complete-bank coverage remain unresolved. Terrain learning and held-out
qualification are not cleared. Diagnostic tracking, exports and local tests
cannot replace the empirical gates or independent overall score of at least 18/20.
Detailed requirements and research history are maintained locally under ignored
`.agent/`; they are not needed to run the package.
