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
    Go2Config(task_id="Go2-Stairs-v1", terrain=TerrainConfig(fixed_level=8)),
    training=False,
)
```

`go2-terrain-development-v2` uses 12 × 12 m tiles. Mixed has 20 ordered columns:
four Flat, four Uneven, four Slopes, four Stairs and four Hills. Slopes and Stairs
split their columns equally between ordinary and inverted pyramids. Pyramid sign
describes the surface; recorded motion and support establish ascent or descent.
Family-specific tasks use the same physical recipe.

All replicas start at level 0. This is a true plane in every family, explicitly
reported as **flat entry**, without implying nonflat capability. Flat stays flat;
other families progress through these 16 candidate levels:

| Level | Uneven amplitude, mm | Slope rise/run | Stair riser, cm | Hill native amplitude, cm |
| --- | --- | --- | --- | --- |
| 0 | 0 | 0.000 | 0 | 0 |
| 1 | 1 | 0.005 | 0.5 | 1 |
| 2 | 2 | 0.010 | 1 | 2 |
| 3 | 3 | 0.015 | 1.5 | 3 |
| 4 | 5 | 0.020 | 2 | 4 |
| 5 | 7 | 0.030 | 3 | 5 |
| 6 | 10 | 0.040 | 4 | 6 |
| 7 | 12 | 0.060 | 5 | 8 |
| 8 | 15 | 0.080 | 6 | 10 |
| 9 | 20 | 0.100 | 7 | 12 |
| 10 | 25 | 0.120 | 8 | 14 |
| 11 | 30 | 0.150 | 9 | 16 |
| 12 | 35 | 0.180 | 10 | 19 |
| 13 | 40 | 0.220 | 12 | 22 |
| 14 | 45 | 0.260 | 14 | 26 |
| 15 | 50 | 0.300 | 16 | 30 |

Heightfields use 0.1 m horizontal cells and 0.001 m height increments; Uneven's
noise step is also 0.001 m. Its sample spacing stays 0.8 m through level 4, then
decreases linearly to 0.2 m at level 15. Native cubic interpolation may exceed the
requested amplitude. Slopes and Stairs retain 2 m central platforms; stairs have
0.3 m treads and a 0.3 m border. Hills retain two waves per axis, approximately
5.95 m wavelength, and half the listed amplitude parameter per axis. Combined
ideal peak-to-peak height is approximately twice that parameter. No added
roughness appears on ID Slopes, Stairs or Hills. Receipts keep unrounded values.

The shared command sampler is unchanged: one held 20 s episode packet selects
translation with probability 0.5, or stand, left/right pivot and left/right arc
with probability 0.1 each. The `go2_family_block_curriculum_v2` teacher maintains
one target per nonflat family and assesses non-overlapping blocks of 20 unchanged
translation episodes at that target, pooled across the family's replicas.

An eligible episode succeeds when it has no physical fall, ends at the horizon
or a legitimate tile boundary, supplies at least 8 s of measurements, tracks
planar speed with RMSE at most 30% of command magnitude after the first second,
tracks yaw with RMSE at most 0.15 rad/s, and achieves at least 70% of requested
travel projected onto the initial commanded world direction. At nonzero levels,
it must additionally supply at least 1 s of supported contact on the relevant
terrain: outside the 1.3 m platform buffer for Slopes/Stairs, or changed/local
support height above 0.5 mm for Uneven/Hills. Support requires a named foot force
above 1.5 N, a valid surface ray and absolute foot-center clearance at most 0.08 m.
Level 0 is exempt from the nonflat-contact requirement. Early physical falls and
short otherwise eligible endings count as failures.

At least 16 successes out of 20 promotes one level; eight or fewer demotes one;
9–15 holds. Every assessed block clears. The bottom clips at 0 and the top holds
at 15. Apply target changes only as replicas reset; running episodes retain
their assigned geometry and obsolete-target outcomes cannot fill a newer block.
Stand, pivot, arc, changed-command and no-step resets hold progression. These
thresholds are provisional teacher settings, not evaluation success gates.

Task progress preserves family targets, assessment blocks/history and physical
assignments. A fresh-process restart retains completed teacher state while
clearing partial physical episodes; periodic saves do not reset episodes.
Teacher-only simulator measurements do not enter actor or method GT observations.

Evaluation disables progression and generates one row at an explicit physical
anchor: 0 (flat entry), 4, 8, 12 or 15. The default fixed level is 8; its native
row index is zero at every severity. All levels are permitted during training,
so an unreached level is still ID. A root crossing a tile's ±6 m XY boundary
truncates the episode; feet may already extend beyond it. Report actual exposure:
a spawn platform does not establish traversal of its surrounding terrain.

The changed quantization and teacher have new recipe identities. Existing v1
captures and campaigns remain scoreable as v1; they are not reinterpreted using
this table. Old task/learner checkpoints cannot resume under the new curriculum.
Geometry, progression and empirical difficulty still require native calibration.

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
semantic seed identities. One attempt per cell produces 186 cases: one Flat
geometry plus levels 0/4/8/12/15 of Uneven, Slopes, Stairs and Hills; Slopes and Stairs
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

### Unseen terrain and actuator robustness

`robustness_bank("validation" | "test", attempts=...)` extends the core bank.
One attempt produces **588 cases**, all evaluated with the same frozen policy:

| Block | Cases | Condition and control |
| --- | --- | --- |
| Core ID | 186 | Base families, all profiles and five physical anchors, including flat entry. |
| Unseen compositions | 150 | Uneven Slopes, Stairs and Hills, with every matching base case in the core bank. |
| Terrain OOD | 180 | Each non-flat family's parameter exceeds the entire allowed training range; matched nominal base cases remain available. |
| Timed drive loss | 72 | Flat and level-8 Uneven, forward motion/stopping, all 12 joints, factors 1, 0.5 and 0. Factor 1 is the matched healthy control for each joint. |

```python
from parkour_lab.evaluation import evaluate, robustness_bank

report = evaluate(policy, robustness_bank("validation"), output, app=app)
```

Compositions add seeded native Uneven heights to the same supporting mesh.
Stair faces are refined before displacement, preserving their nominal risers and
supporting treads. Both pyramid signs are included. Component parameters remain
inside their respective training ranges. The layer is zero within the central
2 x 2 m region, blends to full strength over the next 0.5 m and tapers over the
outer 0.3 m boundary. Base origins and central starting support stay unchanged.
Level-0 composition is an intentional plane with no added roughness.
These are declared start provisions; remaining there is not evidence of traversing
the composed surface. Combined local grades and step transitions can exceed those
of either component and must be assessed from realized geometry.

Terrain OOD uses excluded parameters at anchors 0/4/8/12/15. Its anchor 0
remains nonflat; it is not the ID flat entry.

| Parameter | Training maximum | OOD 0 | OOD 4 | OOD 8 | OOD 12 | OOD 15 |
| --- | --- | --- | --- | --- | --- | --- |
| Uneven noise amplitude, m | 0.05 | 0.06 | 0.0706667 | 0.0813333 | 0.092 | 0.1 |
| Slope rise/run | 0.3 | 0.35 | 0.376667 | 0.403333 | 0.43 | 0.45 |
| Stair riser, m | 0.16 | 0.18 | 0.190667 | 0.201333 | 0.212 | 0.22 |
| Hill native amplitude, m | 0.3 | 0.35 | 0.376667 | 0.403333 | 0.43 | 0.45 |

Reports retain unrounded values and the excluded training bounds. Uneven's
interpolation can overshoot its requested amplitude. These are explicit parameter
extrapolations, not calibrated recoverability levels. A fresh seed within allowed
support is unseen ID, not OOD. Deterministic base slopes/stairs/hills repeat their
primitive geometry; held-out seeds vary starts, dynamics and noise. This bank does
not yet define dynamics OOD, jams or sensor faults.

A timed fault multiplies **final native limited drive torque** by 0.5 or 0 at one
named joint on every physics substep. It leaves targets, PD gains, native limits
and startup randomization unchanged. Zero drive is a passive joint, not a jam or
a zero-position command. Actual delivered effort also supplies the torque reward.
The seeded onset falls on a control boundary between 6 and 8 seconds, shared by
all joints/factors for the same base draw. The first physics interval starting at
onset applies the loss; it persists until that row resets. The next attempt starts
healthy and follows its own schedule. No fault label or memory reset is supplied
to the policy at onset. Healthy controls use the same schedule with factor 1.

For an explicit evaluation task, configure before construction:

```python
from parkour_lab.benchmark import FaultConfig, Go2Config, TerrainConfig, build_config

composed = build_config(
    Go2Config(task_id="Go2-Stairs-v1",
              terrain=TerrainConfig(fixed_level=8, condition="composition")),
    training=False,
)
faulted = build_config(
    Go2Config(fault=FaultConfig("FR_thigh_joint", factor=0.5, onset_s=7.0)),
    training=False,
)
```

Core training rejects compositions, excluded terrain ranges and timed faults.
Keep these conditions out of tuning and checkpoint selection when reporting
unseen generalization. Using validation outcomes for tuning makes those outcomes
development evidence; reserve a fresh final bank and disclose what was learned.

The overall/family/profile scores remain **core ID only**. Each robustness block
has separate weights and family summaries. Paired results include return and
survival-duration changes against the prescribed control. Fault results add
pre/post-onset tracking, post-onset coverage and time to failure. Failures before
onset remain in the prescribed denominator, with unavailable post-onset tracking;
missing controls yield unavailable paired deltas. Post-onset return deltas are
explicitly conditional on both attempts reaching onset and their observed
durations. Healthy seconds do not become a recovery score.

Fault traces record `fault_active`, `drive_factor`, `limited_effort` and
`applied_effort`. Effort/factor columns use leg-major FL, FR, RL, RR, each with
hip, thigh, calf; this evaluator order is independent of native motor order.
The offline scorer checks the prescribed joint/onset/factor and post-limit
arithmetic, plus the realized condition and common core reward. Every missing or
failed attempt remains visible. Full-bank admission requires all 588 prescriptions;
adding a partial robustness subset cannot masquerade as a complete comparison.

Inspect the prescriptions or score an existing capture offline:

```bash
python -m parkour_lab cases --suite robustness --partition validation --output cases.json
python -m parkour_lab score CAPTURE_DIRECTORY --output offline_score.json
```

## Development protocol and comparisons

The development recipe is fixed by its versioned definitions and the realized
metadata recorded with each task/capture. These identify a reproducible software
contract; they do not establish a calibrated final benchmark or a method ranking.

| Component | Fixed development rule |
| --- | --- |
| Task | `go2-blind-v1`: Go2, 200 Hz physics, 50 Hz control, 20-second episodes and native physical termination. Tiled tasks have explicit tile-boundary truncation; standalone Flat uses an unbounded plane. |
| Training terrain | `go2-terrain-development-v2`: the five-family mixture, 16 physical levels and family-block curriculum described above. Use a multiple of 20 replicas for exact replication of the 20 columns, equal family shares and balanced slope/stair signs. |
| Commands | `go2_episode_twist_v1`: 50% translation, 10% stand and 10% each left/right pivot and arc; one command per episode. |
| Information and actions | Common 49D blind actor frame, declared use of the shared 352-value training-only packet, and the same bounded 12D action/motor contract. No training packet reaches evaluation. |
| Objective | The realized native Go2 rough reward, identified by its recipe hash. Overrides define a separately identified custom objective. |
| Evaluation | `go2_fixed_cases_development_v2`: disjoint train/validation/test seed identities; the 186-case bank and fixed command phases described above. The same endpoint policy is evaluated across every cell. |
| Scores | `go2_fixed_scores_development_v1`: equal families, then equal physical levels, subtypes, profiles and attempts. Core ID remains separate from each `go2_robustness_development_v2` block. Preserve failures, unavailable phases and exposure labels. |
| Experience | Aggregate real robot transitions, including warm-up and discarded collection. Consumer projects declare and enforce the same allowance across compared methods; optimizer or imagination steps are separate compute statistics. |
| Continuation | A consumer-owned full learner checkpoint linked to task progress; cumulative experience and run identity survive recreation into fresh physical episodes. |

Keep the resolved configuration, package/example sources, dependency identities,
reward hash, motor/dynamics readbacks and checkpoint/progress linkage with results.
Version labels alone do not prove that two runs used identical definitions or
physical realizations. A changed source/configuration identity needs matching
evidence; old checkpoints are not silently migrated.

Before a method comparison, declare training seeds, replica count, interaction
allowance, tuning allowance, training-information use and checkpoint-selection
rule. Compare common-budget endpoints without choosing a different checkpoint
for each family. Validation may guide development; final-test outcomes may not
select the recipe or winning checkpoints. Report per-family/command failures and
tracking alongside return, and summarize independent training seeds separately.
Choose an allowance compatible with every integration's complete collection
blocks. The bounded example rejects an incompatible remainder instead of hiding
extra collection or rounding a run's recorded spend down to the allowance.

There is no calibrated default training budget or guaranteed mastery threshold.
The short PPO/SAC runs below exercise integration, not competitive baselines.
Physical severity calibration, learnability, final comparison budgets and
statistical ranking remain to be validated before making those claims. A full
capture with `valid_for_comparison=true` establishes bank completeness and capture
consistency; it does not supply that empirical validation. A poorly performing
policy is still a valid test of failure accounting and the public evaluator.

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
| `environments/terrains/` | Native Flat/Uneven/Slopes/Stairs/Hills definitions, geometry bindings and shared family targets. |
| `runtime/` | Native sensor/motor binding, transition accounting and ordered environment/application cleanup. |
| `evaluation/runner.py`, `environment.py`, `capture.py` | Frozen playback, dedicated case environments and portable outgoing-transition capture. |
| `evaluation/cases.py`, `scoring/` | Fixed case identities and authoritative physical scores/aggregates. |
| `evaluation/campaign.py` | Offline planned-run completeness and uncertainty across independent training seeds. |
| `tests/` | Behavioral contracts, package boundaries, analytic scores and geometry checks. |
| `scripts/check_public_task.py` | Runnable public-task consumer and bounded training/evaluation/continuation contract check. |
| `scripts/check_terrain.py`, `scripts/_terrain_checks/` | Bounded native geometry, contact, curriculum and recreation checks. |
| `scripts/check_evaluation.py` | Caller-format policy save/load, frozen native evaluation and independent offline scoring checks. |
| `scripts/check_learning_contracts.py`, `scripts/_learning_checks.py` | Bounded native timeout/bootstrap, replay ownership and physical action-saturation witnesses for the optional learner bridge. |

`control.Controller` and `runtime.NativeControllerSession` accept a loaded causal
controller independently of a learner. `control.policy.PolicyController` adapts
the common 49D policy interface to named sensors and bounded joint targets.
The evaluation host supplies sensing and motor binding; a policy never receives
an environment handle.

Native stepping preserves separate termination/truncation flags and owned outgoing
final rows before automatic reset. Consuming trainers own optimization,
rollout/replay storage, inference memory and checkpoint loading. Shared control
and terrain code do not select a network architecture.

## Optional Stable-Baselines3 integration

The repository example in [`examples/sb3/`](../../../examples/sb3/) calls real
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) PPO and SAC.
It is outside the installed benchmark package and adds no core learning
dependency. The small NumPy `VecEnv` bridge transfers one native batch at a time;
it is an integration reference, not a GPU-throughput baseline. Both actor and
critic use the same blind 49D input; training GT is available but unused. Rewards,
terrain, commands, action bounds and curriculum retain the public task recipe.
Observation/reward normalization is disabled and schedules are constant.

The optional requirements select SB3 2.7/2.8, tested here with 2.8.0. That line
supports the reference stack's PyTorch 2.7;
[SB3 2.9 requires PyTorch >=2.8](https://github.com/DLR-RM/stable-baselines3/releases/tag/v2.9.0).
These example requirements do not constrain the benchmark core or other external
learners. Check the selected native stack before upgrading its Torch distribution.

Install into the configured Isaac Lab interpreter, from the repository root:

```bash
"${PLAB_PY:-python}" -m pip install -r examples/sb3/requirements.txt
"${PLAB_PY:-python}" -m examples.sb3 --help
```

The following bounded command runs each method in three fresh processes: collect
and update through 2,560 total interactions, resume through 5,120, then evaluate
the saved policy on the six engineering cases. Twenty Mixed replicas include all
five families from the first collection. PPO uses complete 32-step rollouts;
an incompatible remaining allowance is rejected before collection. SAC's random
warm-up counts toward the same physical allowance. Learner counters must match
the native cumulative counter exactly, with no uncertain collection.

```bash
bash <<'BASH'
set -uo pipefail
PLAB_PY=${PLAB_PY:-python}
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/learner_check_XXXXXX) || exit 1
trap 'status=$?; printf "{\"returncode\": %d}\n" "$status" > "$RUN/process_exit.json"; echo "Results: $PWD/$RUN"' EXIT
run_stage() {
    local name=$1
    shift
    local log="$RUN/${name//\//_}"
    "$PLAB_PY" -u -m examples.sb3 "$@" --device cuda:0 --learner-device cpu \
        --output "$RUN/$name" > "$log.log" 2>&1
    local code=$?
    printf '{"returncode": %d}\n' "$code" > "${log}_process_exit.json"
    return "$code"
}
for algorithm in ppo sac; do
    run_stage "$algorithm/train" train --algorithm "$algorithm" \
        --total-interactions 2560 || exit "$?"
    run_stage "$algorithm/resume" train --algorithm "$algorithm" \
        --resume "$RUN/$algorithm/train/checkpoint" \
        --total-interactions 5120 || exit "$?"
    run_stage "$algorithm/evaluation" evaluate \
        --checkpoint "$RUN/$algorithm/resume/checkpoint" || exit "$?"
done
BASH
```

Training reports must say `LEARNER_CHECK_PASSED`, show changed parameters and
increased external update counters, and close both native resources. The resumed
report must have `learner_checkpoint_restored=true`, session 2,560 and cumulative
5,120 interactions. Evaluation must say `LEARNER_EVALUATION_PASSED` with complete
case/app cleanup and saved-score parity. Tiny runs establish collection, learning,
checkpoint restoration and scoring; they do not establish successful locomotion.
Use a new output folder and add `--full-bank` to `evaluate` for all 186 validation
cases. Keep the complete returned directory, including failed attempts.

Each completed phase saves SB3's model/optimizer archive, SAC replay when used,
global/action-space RNG and checkpoint-linked task progress. Files and learner
state are verified before continued learning; the original SB3 version and source
must match. Inference uses SB3's deterministic prediction, including its own
action unscaling. Timeout bootstrapping uses owned outgoing observations, while
physical termination takes precedence. Resumption starts fresh physical episodes;
no partial PPO rollout or live simulator state is restored. Full saves occur every
5,000 declared updates and at normal completion. PPO counts epochs; SAC counts
gradient steps. `--milestones` additionally saves at absolute interaction totals;
all targets must align with complete collection blocks. Saving continues the live
physical episodes. `checkpoint_index.json` links each completed bundle, counter,
save reason and session training-loop time. Intermediate bundles are under
`checkpoints/step_<interactions>/`; the terminal bundle remains `checkpoint/`.
An interrupted save cannot replace an earlier complete bundle. The short default
run stays below the periodic interval and saves only its terminal checkpoint.

This example resumes clean checkpoints with exact accounting. It is not a crash
recovery service. Error reports retain known interaction totals; collection after
an older checkpoint still counts. Do not restart a failed run from that older
checkpoint and erase its additional spend. Load only trusted SB3 checkpoint
bundles, which contain Python serialization.

### Longer continuation and full validation

After both short checks finish at 5,120 interactions, continue each saved learner
to **25,600 total interactions** and evaluate its endpoint on all **186 validation
cases**. Set `PLAB_INPUT` to the complete previous results directory, containing
`ppo/resume/checkpoint/` and `sac/resume/checkpoint/`. Run from the repository root
with the same native interpreter, dependencies and Python sources as those
checkpoints. No package upgrade is needed for this continuation.

```bash
export PLAB_INPUT=/absolute/path/to/completed_learner_results
export PLAB_PY=/absolute/path/to/native/python
bash <<'BASH'
set -uo pipefail
: "${PLAB_INPUT:?Set PLAB_INPUT to the previous results directory}"
PLAB_PY=${PLAB_PY:-python}
BASE=$(cd "$PLAB_INPUT" && pwd) || exit 1
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
for algorithm in ppo sac; do
    CHECKPOINT="$BASE/$algorithm/resume/checkpoint/checkpoint.json"
    [[ -r "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
done
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/learner_continuation_XXXXXX) || exit 1
trap 'status=$?; printf "{\"returncode\": %d}\n" "$status" > "$RUN/process_exit.json"; echo "Results: $PWD/$RUN"' EXIT
run_stage() {
    local name=$1
    shift
    local log="$RUN/${name//\//_}"
    "$PLAB_PY" -u -m examples.sb3 "$@" --device cuda:0 --learner-device cpu \
        --output "$RUN/$name" > "$log.log" 2>&1
    local code=$?
    printf '{"returncode": %d}\n' "$code" > "${log}_process_exit.json"
    return "$code"
}
for algorithm in ppo sac; do
    run_stage "$algorithm/resume" train --algorithm "$algorithm" \
        --task Go2-Mixed-v1 --num-envs 20 --seed 1077 \
        --resume "$BASE/$algorithm/resume/checkpoint" \
        --total-interactions 25600 || exit "$?"
done
for algorithm in ppo sac; do
    run_stage "$algorithm/evaluation" evaluate --full-bank \
        --checkpoint "$RUN/$algorithm/resume/checkpoint" || exit "$?"
done
BASH
```

This adds 20,480 interactions per method: 1,024 native batch steps, or 20.48
simulated seconds of collection per replica in the new process. PPO completes
32 additional rollouts. The original run identity, learner settings, reward and
task recipe are retained; physical episodes restart. Surviving episodes can now
reach the 20-second training timeout. Falls can prevent that coverage, so inspect
the returned episode data before claiming timeout validation. Both terminal
checkpoints are saved before the longer full-bank evaluations begin.

Each resumed report must start at cumulative 5,120 and finish with session 20,480,
cumulative 25,600 and zero uncertain interactions. The final checkpoint must link
the same run identity and record changed policy parameters. SAC's 20,000-transition
replay now wraps and evicts its oldest data normally; preserving the entire old
buffer prefix is no longer expected. Both methods remain below 5,000 external
updates, so the terminal checkpoint is sufficient for this bounded run.

Full validation must retain every prescribed case, including early physical
failures and unavailable later phases. Require `full_bank=true`,
`capture_valid=true`, `valid_for_comparison=true` and saved-score parity. Full-bank
coverage qualifies the capture; this small, single-seed run still does not
establish learnability, final score calibration or a method ranking. Keep the
complete output directory for review.

The optional CPU suite exercises genuine PPO/SAC updates, outgoing-state replay
semantics, checkpoint/RNG equality and fresh-process continuation on a synthetic
task. It does not launch Isaac Sim:

```bash
python -m unittest discover -s examples/sb3/tests -t . -v
```

### Bounded development pilot

This pilot measures learning trends under the existing integration recipes.
Train three fresh seeds per method on the five-family Mixed task, retaining full
checkpoints at 256,000, 512,000 and 1,024,000 cumulative real transitions. The six
runs collect **6,144,000 training transitions** in total. Evaluate all 186 fixed
validation cases at each checkpoint in separate native processes; evaluation
interactions are recorded separately. Keep the seed list, recipe, milestones and
bank fixed for the entire pilot. The command preserves failures and produces an
offline completeness report even when a run fails.

Run from the repository root with the configured native interpreter. This uses
the same small PPO/SAC architectures, update ratios and SAC replay capacity as the
integration checks above. These settings are disclosed integration recipes. Negligible commanded movement
would leave entry learnability unvalidated and require further common-task
calibration before method ranking. Inspect return,
tracking in both turn directions, stopping, falls and actual nonflat support
exposure together. Survival alone cannot establish commanded locomotion.

This first v2 pilot checks flat entry and early progression. Twenty eligible
translation outcomes are needed for each family assessment; the finite pilot
does not require reaching level 15. Inspect family blocks/history, command
response and physical exposure before extending the budget.

```bash
export PLAB_PY=/absolute/path/to/native/python
bash <<'BASH'
set -uo pipefail
PLAB_PY=${PLAB_PY:-python}
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/calibration_pilot_XXXXXX) || exit 1
export RUN
trap 'status=$?; printf "{\"returncode\": %d}\n" "$status" > "$RUN/process_exit.json"; echo "Results: $PWD/$RUN"' EXIT
# Declare the complete grid before any collection; absent artifacts stay planned.
"$PLAB_PY" - <<'PY' || exit "$?"
import hashlib, json, os
from pathlib import Path
from examples.sb3.__main__ import _example_sources
from parkour_lab.provenance import dependency_identity
root = Path(os.environ['RUN'])
library = dependency_identity('stable_baselines3', 'stable-baselines3')
library = {k: library[k] for k in ('version', 'source_sha256')}
sources = dict(example_sources=_example_sources(), stable_baselines3=library)
source = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
plan = dict(version='go2_campaign_v1', purpose='development', partition='validation',
            suite='core', case_version='v2', seeds=[1078, 2078, 3078], milestones=[256000, 512000, 1024000], methods=[])
for algorithm in ('ppo', 'sac'):
    method = dict(method_id=algorithm, source_id=source, recipe_id=f'{algorithm}:{source}',
                  ground_truth_usage=[], runs=[])
    for seed in plan['seeds']:
        prefix = f'{algorithm}/seed_{seed}'
        run = dict(training_seed=seed, run_id=f'{root.name}:{algorithm}:{seed}',
                   configuration_id='unavailable', training_report=f'{prefix}/train/report.json', endpoints=[])
        for point in plan['milestones']:
            checkpoint = 'checkpoint' if point == plan['milestones'][-1] else f'checkpoints/step_{point}'
            run['endpoints'].append(dict(interactions=point,
                capture=f'{prefix}/evaluation_{point}/capture',
                progress=f'{prefix}/train/{checkpoint}/progress.json', resources=dict(wall_time_s=0)))
        method['runs'].append(run)
    plan['methods'].append(method)
(root/'campaign.json').write_text(json.dumps(plan, indent=2)+'\n')
PY
run_stage() {
    local name=$1
    shift
    local log="$RUN/${name//\//_}"
    "$PLAB_PY" -u -m examples.sb3 "$@" --device cuda:0 --learner-device cpu \
        --output "$RUN/$name" > "$log.log" 2>&1
    local code=$?
    printf '{"returncode": %d}\n' "$code" > "${log}_process_exit.json"
    return "$code"
}
failed=0
for algorithm in ppo sac; do
    for seed in 1078 2078 3078; do
        prefix="$algorithm/seed_$seed"
        run_stage "$prefix/train" train --algorithm "$algorithm" \
            --task Go2-Mixed-v1 --num-envs 20 --seed "$seed" \
            --run-id "${RUN##*/}:$algorithm:$seed" --total-interactions 1024000 \
            --milestones 256000 512000 1024000 || failed=1
        for point in 256000 512000 1024000; do
            checkpoint="$RUN/$prefix/train/checkpoints/step_$point"
            [[ $point == 1024000 ]] && checkpoint="$RUN/$prefix/train/checkpoint"
            [[ -r "$checkpoint/checkpoint.json" ]] || { failed=1; continue; }
            run_stage "$prefix/evaluation_$point" evaluate --full-bank \
                --checkpoint "$checkpoint" || failed=1
        done
    done
done
# Fill observed configuration identities and cumulative fresh-run timing.
# Original reports/indexes remain separate, including any later failed collection.
"$PLAB_PY" - <<'PY' || exit "$?"
import hashlib, json, os
from pathlib import Path
root = Path(os.environ['RUN'])
plan = json.loads((root/'campaign.json').read_text())
def read(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
for method in plan['methods']:
    for run in method['runs']:
        report = read(root/run['training_report'])
        index = read((root/run['training_report']).with_name('checkpoint_index.json'))
        entries = {v['training_interactions']: v for v in index.get('checkpoints', [])}
        library = report.get('dependencies', {}).get('stable_baselines3', {})
        sources = dict(example_sources=report.get('example_sources'),
                       stable_baselines3={k: library.get(k) for k in ('version', 'source_sha256')})
        source = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        for endpoint in run['endpoints']:
            progress = read(root/endpoint['progress'])
            if progress:
                run['configuration_id'] = progress['configuration_id']
            endpoint['resources'] = dict(report.get('resources', {}),
                wall_time_s=entries.get(endpoint['interactions'], {}).get('training_loop_wall_time_s', 0),
                timing_scope='cumulative fresh training loop; full process timing in training_report')
            if source != method['source_id']:
                endpoint.update(status='failed', reason='Training source receipt missing or differs from the declared method')
            elif report.get('status') != 'LEARNER_CHECK_PASSED':
                endpoint.update(status='failed', reason='Training process failed; retain its report and all collected spend')
            elif report.get('training_information_used') != method['ground_truth_usage']:
                endpoint.update(status='failed', reason='Training information use differs from the declaration')
(root/'campaign.json').write_text(json.dumps(plan, indent=2)+'\n')
PY
"$PLAB_PY" -m parkour_lab report "$RUN/campaign.json" \
    --output "$RUN/campaign_report.json" || exit "$?"
exit "$failed"
BASH
```

Recompute the report on a CPU machine with the installed core package:

```bash
python -m parkour_lab report /absolute/path/to/pilot/campaign.json --output /tmp/pilot_report.json
```

The manifest is learner independent: each method declares source/recipe identity
and the training-packet GT fields it uses; each seed declares one run identity,
task configuration and checkpoint-linked progress/capture paths. Paths resolve
relative to the manifest. `resources.wall_time_s` is a cumulative external
declaration at that checkpoint, never a sum of milestone timings. Other external
learners can write this manifest with their own checkpoint format. The reporter
loads only task progress and saved traces, rechecks native/source/reward identities
and complete banks, and computes sample SD/SE across independent training seeds.
Missing or invalid endpoints block a complete result; physical falls retain
their recorded returns. Generating a report succeeds even for incomplete data;
inspect its completeness and readiness fields.

For a clean continuation, keep the same seed/task/replica count and sources, pass
`--resume <full-checkpoint-folder>`, and set `--total-interactions` to the original
absolute budget, with only future `--milestones`. New-session loop times must be
added to prior session times when declaring cumulative resources. A resumed
checkpoint remains part of its original seed run. The example refuses uncertain
or discarded collection that no longer equals the saved learner counter; retain
the failure and its spend rather than treating a retry as free data.

These development results can inform shared protocol changes, which need a new
declared pilot. They do not freeze difficulty, validate the provisional 100M
allowance or rank methods. Keep final-test cases unopened during calibration.
Final comparisons require a frozen protocol, at least five fresh seeds per method,
common-budget endpoints and the complete ID/OOD/fault banks with seed uncertainty.

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
python -m scripts.check_learning_contracts --help
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

The reference native stack is Isaac Lab 2.3.2.post1 and Isaac Sim 5.1. Each check
establishes only the contracts it actually exercises. Retain and inspect its
complete output before accepting compatibility, including when using another stack.

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

This command runs five fixed-severity Mixed views and training/recreation in
seven fresh native processes, with 20 replicas covering every terrain column.
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
other replicas, assignments and interaction totals. The training host also supplies
controlled teacher measurements to exercise family-block promotion, demotion,
holds, changed packets and bottom/top cases through the curriculum manager. These administrative
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
for level in 0 4 8 12 15; do
    run_check "level_$level" --role eval --level "$level" || exit "$?"
done
run_check train --role train || exit "$?"
run_check resume --role train --resume "$RUN/train/progress.json" || exit "$?"
BASH
```

Each process must exit zero, report `TERRAIN_CHECK_PASSED` and close both resources.
The fresh resumed task must recover levels, family targets/blocks/history, stable columns, sampler/report
and cumulative interactions before resetting fresh episodes. The checkpoint token
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
families, all six profiles and physical levels 0, 8 and 15. It is explicitly not
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
new output directory to capture all 186 validation cases. Both modes stop each
attempt at its first ending; the full bank can therefore take up to 62 simulated
minutes, plus environment-construction overhead. The fixture's performance is
not a benchmark baseline. Training-progress recreation restores task counters
and samplers only; genuine learner checkpoint continuation belongs to the
consuming trainer.

### Timeout and action-saturation checks

This bounded check uses the optional SB3 installation and the real Flat task.
It collects two PPO steps and one SAC step without optimizing either model,
then checks stored-frame ownership and both action bounds. The host sets row
zero's episode age to one step before the normal horizon for the first step of
each check. It changes neither the physical state nor the configured horizon.
The PPO critic uses an explicit analytic value of the previous delivered action
so that outgoing and reset-frame bootstrap targets are observably different.
These are controlled contract fixtures, not trained policies or stability tests.

Run from the repository root using the configured native interpreter:

```bash
bash <<'BASH'
set -uo pipefail
PLAB_PY=${PLAB_PY:-python}
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/learning_contracts_XXXXXX) || exit 1
trap 'status=$?; printf "{\"returncode\": %d}\n" "$status" > "$RUN/process_exit.json"; echo "Results: $PWD/$RUN"' EXIT
"$PLAB_PY" -u -m scripts.check_learning_contracts \
    --device cuda:0 --num-envs 4 --seed 1077 --output "$RUN/native" \
    > "$RUN/native.log" 2>&1
code=$?
printf '{"returncode": %d}\n' "$code" > "$RUN/native_process_exit.json"
exit "$code"
BASH
```

With four replicas the report must show `LEARNING_CONTRACTS_PASSED`, six control
steps, session/cumulative 24 interactions, zero uncertain interactions and zero
optimizer updates. Both environment/application cleanup fields must be `complete`.
`ppo.json` records raw native rewards, pure timeout flags, outgoing/reset frames
and the analytic bootstrap comparison. `sac.json` records owned outgoing replay
frames and timeout-masked sampled dones. `saturation.json` records both extreme
request signs for every joint, the delivered action, native processed/joint
targets and saturation counts, including outgoing rows before automatic reset.
The witnesses also check that final-frame reads do not consume extra actor-noise
draws and that subsequent steps do not mutate saved frames.

Keep the entire results directory, including source/dependency identities and
`resolved_env.yaml`. Local CPU tests cannot replace this native witness. This
command is independent of model checkpoints and does not require more reference
training. The optional CPU suite includes its collection/assertion guards;
successful execution establishes these contracts only, not benchmark calibration.

### Composed-surface and drive-loss checks

Use the configured native interpreter from the repository root. This check needs
no learner or trained checkpoint. Each geometry invocation compares matched ID,
composed and excluded-range surfaces, saving collider ray grids, starts and
support measurements. A controlled placement outside the central region checks
actual composed-surface foot contact with at most 32 control calls. It is a
collision/support fixture, not learned traversal. Each motor invocation checks
native compute at both torque signs, saturation and overspeed, then six real
control calls with two replicas, onset, persistence and one-row reset. Its 0.04 s
onset is a deliberate short fixture; the evaluation bank uses 6–8 s.

```bash
bash <<'BASH'
set -uo pipefail
PLAB_PY=${PLAB_PY:-python}
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/robustness_check_XXXXXX) || exit 1
trap 'status=$?; printf "{\"returncode\": %d}\n" "$status" > "$RUN/process_exit.json"; echo "Results: $PWD/$RUN"' EXIT
run_check() {
    local name=$1
    shift
    "$PLAB_PY" -u -m scripts.check_robustness "$@" --device cuda:0 \
        --seed 1077 --output "$RUN/$name" > "$RUN/$name.log" 2>&1
    local code=$?
    printf '{"returncode": %d}\n' "$code" > "$RUN/${name}_process_exit.json"
    return "$code"
}
for family in Slopes Stairs Hills; do
    for level in 0 4 8 12 15; do
        run_check "${family}_${level}" --check geometry --family "$family" \
            --level "$level" || exit "$?"
    done
done
for factor in 1 0.5 0; do
    run_check "drive_${factor}" --check motor --factor "$factor" \
        --joint FR_thigh_joint || exit "$?"
done
BASH
```

Every child report must say `ROBUSTNESS_CHECK_PASSED`, with complete environment
and application cleanup, exact interaction totals and zero uncertain collection.
Geometry receipts separate zero-step ray/start checks from real contact calls.
Motor receipts separate synthetic native-compute inputs from actual physics
intervals; extra same-counter writes do not count as physics. Retain the full
returned directory, including failed outcomes, raw witnesses, configurations,
source hashes and dependency identities. The CPU suite and these commands do not
establish native acceptance until their actual outputs have been reviewed.

### Full frozen-policy robustness collection

An independent project passes its already-loaded policy to `evaluate` with
`robustness_bank`, as above. The optional SB3 example can reuse a trusted existing
endpoint without collecting more training data. Inference verifies the entire
saved bundle and exact SB3 implementation, while allowing the evaluator package
to change; the original training sources remain in the checkpoint receipt and
current evaluator sources in the capture. Full training resume still requires
the original package sources and task identity. This is not checkpoint migration
or a claim of unchanged physical behavior between arbitrary package revisions.

Set `PLAB_CHECKPOINT` to the existing full checkpoint folder, then run:

```bash
bash <<'BASH'
set -uo pipefail
: "${PLAB_CHECKPOINT:?Set PLAB_CHECKPOINT to the endpoint checkpoint folder}"
PLAB_PY=${PLAB_PY:-python}
[[ -r "$PLAB_CHECKPOINT/checkpoint.json" ]] || exit 1
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs/parkour_lab || exit 1
RUN=$(mktemp -d logs/parkour_lab/robustness_evaluation_XXXXXX) || exit 1
trap 'status=$?; printf "{\"returncode\": %d}\n" "$status" > "$RUN/process_exit.json"; echo "Results: $PWD/$RUN"' EXIT
"$PLAB_PY" -u -m examples.sb3 evaluate --suite robustness \
    --checkpoint "$PLAB_CHECKPOINT" --device cuda:0 --learner-device cpu \
    --output "$RUN/evaluation" > "$RUN/evaluation.log" 2>&1
code=$?
printf '{"returncode": %d}\n' "$code" > "$RUN/evaluation_process_exit.json"
[[ "$code" -eq 0 ]] || exit "$code"
"$PLAB_PY" -m parkour_lab score "$RUN/evaluation/capture" \
    --output "$RUN/offline_score.json" || exit "$?"
BASH
```

The example requires a complete valid capture and identical recomputed offline
scores. Inspect all block summaries, missing attempts, pre-onset failures, paired
controls and exposure. A full 588-case bank has at most 196 simulated minutes,
plus construction overhead; early endings shorten collection. A poor recovery
score is a valid result. It does not justify excluding cases or tuning the
policy against this bank. Final robustness/difficulty calibration remains open.
