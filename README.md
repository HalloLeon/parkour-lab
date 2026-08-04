# Parkour Lab

Parkour Lab is an Isaac Lab reinforcement-learning environment for training a
Unitree A1 to finish waypoint routes across progressively harder obstacles.
Training uses a balanced obstacle-family by difficulty terrain matrix;
evaluation freezes one matrix cell so policy changes can be compared and
recorded under the same conditions.

## Setup

Install Isaac Lab, then install this extension with the Python interpreter from
the same environment:

```bash
python -m pip install -e source/parkour_lab
```

Parkour Lab targets Isaac Lab 2.3.0 and its RSL-RL 3.0.1 integration. Install
the learning framework through Isaac Lab before installing this extension:

```bash
cd PATH_TO_ISAACLAB
./isaaclab.sh --install rsl_rl
```

List the registered environments to verify the installation:

```bash
python scripts/list_envs.py
```

The two task IDs serve different purposes:

- `Parkour-Lab-v0` is the vectorized training task with the adaptive
  curriculum enabled.
- `Parkour-Lab-Play-v0` is the smaller, fixed-family/fixed-difficulty task for
  evaluation and video.

## Train

Run PPO training with RSL-RL:

```bash
python scripts/rsl_rl/train.py \
  --task=Parkour-Lab-v0 \
  --headless
```

Training can periodically record a qualitative progress clip:

```bash
python scripts/rsl_rl/train.py \
  --task=Parkour-Lab-v0 \
  --headless \
  --video \
  --video_length=500 \
  --video_interval=10000
```

Runs are written beneath `logs/rsl_rl/parkour_lab/<run>/`. This includes policy
checkpoints (`model_*.pt`), the resolved environment and agent configurations in
`params/`, TensorBoard data, and optional clips in `videos/train/`.

The default teacher budget is 1,000 PPO iterations with 48 control steps per
environment and update, 20-second episodes, a discount factor of 0.995, and
checkpoints every 100 iterations. The longer rollout and horizon preserve more
approach-to-landing credit than the former short smoke-test settings.
`--max_iterations` can still override the budget for diagnostics. Action noise
starts at 0.8, is bounded to 0.10–0.80, and has no permanent entropy bonus, so
PPO can reduce exploration once the deterministic motor acquires its gait.

New checkpoints store the per-environment curriculum frontier, rolling evidence,
and demotion grace alongside RSL-RL's learner state. `--resume` restores that
memory and starts fresh episodes sampled from the restored frontier. Checkpoints
created before this state was added remain compatible and conservatively use the
configured bootstrap rows. Fixed `Parkour-Lab-Play-v0` evaluation remains pinned
to its requested matrix cell and is not changed by this training behavior.

### Staged domain randomization

Domain randomization is deliberately disabled by default so the teacher first
learns the nominal task. Resume a successful nominal run with the narrow stage,
then use the wide stage only after narrow training remains stable:

```bash
python scripts/rsl_rl/train.py \
  --task=Parkour-Lab-v0 \
  --resume \
  --load_run=<nominal-run> \
  --domain_randomization_stage=narrow \
  --headless
```

Replace `narrow` with `wide` for the final stage. The selected profile perturbs
contact friction and restitution, trunk payload and center of mass, actuator
gains, initial pose and velocity, external pushes, control-step action latency,
and delayed/noisy proprioception. The selected stage and configured ranges are
stored in `params/env.yaml`. Action and observation widths do not change between
stages. Fixed evaluation always selects `off`, removing randomization, noise,
pushes, and delays before constructing the environment.

The same option is available in `distill.py`. If it is supplied together with
the advanced Hydra override `env.domain_randomization.stage=...`, the explicit
CLI option takes precedence.

## Declarative course configuration

Each curriculum matrix cell is a reusable course description rather than a
special case in the runtime. Terrain row 0 gives every family cohort the same
straight obstacle-free route at 0.55 m/s. Rows 1 through 5 contain the five
obstacle difficulties. Equal column blocks retain their future gap, high-step,
hurdle, or tilted-ramp family even on the flat row, so each environment advances
from flat ground into one stable family without resampling its obstacle type.
Each course records ordered terrain-local waypoints, named mesh structures and
their factory arguments, planar support polygons, target speed and clearance,
and explicit difficulty metadata. Obstacle rows keep each family's target speed
fixed and use a common 0.24 m minimum clearance, preventing harder geometry from
being confounded with a faster command or stricter clearance requirement.
Factory arguments are passed directly to each structure factory. A support
region either refers to the generated base ground or names the structure whose
surface it describes. Base-ground regions are authoritative physical patches:
the terrain generator emits one collision box for each such rectangle, so an
uncovered interval is a real hole in both collision and raycast geometry. Named
regions annotate separately generated structures without duplicating them.
Terrain generation still iterates configured structures generically; it does
not branch on a level number or obstacle family.

The high-step family uses 1.6 m deep elevated platforms with vertical,
ground-mounted front faces. Each final waypoint lies inside the annotated top
surface, so completing the course means climbing and landing on the platform.
The hurdle family uses 0.18 m deep barriers spanning the full 4.0 m tile width,
blocking every lateral path around an exposed end while the robot remains
inside its assigned tile. Base-ground support remains continuous beneath and
beyond each barrier, its top is not declared traversable, and its waypoint is
on ground after the rear face. Heights increase within each family while
remaining below the 0.5 m maximum.

Gap widths increase from 0.10 m to 0.50 m. Approach and landing supports stop
at opposite gap lips, and the ordered route directs the robot onto the landing
side before the final waypoint. The approach waypoint remains an unrewarded
directional guide; the supported post-gap landing is explicitly marked as a
physical milestone.

The tilted-ramp family uses an acquisition ladder instead of introducing the
complete compound obstacle at once. Its first obstacle row contains one wide,
straight, gently banked slab. Later rows add a contiguous second slab, then an
inter-ramp gap, then yaw and lateral offset. Only the hardest row narrows the
slabs and combines opposite 12-degree banks with yaws of 10 and 32 degrees.
Each slab's local X axis is its travel direction; signed roll banks its surface
across the width and yaw rotates its travel direction in terrain-local XY.
Ordered waypoints align the approach, mark supported ramp entries and exits,
redirect along the next slab, and finally target the ground landing.

Every support surface records an ordered planar XYZ polygon. Horizontal
rectangles still generate the base-ground collision patches, while each banked
ramp uses the exact corners of its collision slab's top face. The edge penalty
selects the resulting 3D boundary segments for each environment's current level
and counts only feet that are both within 0.05 m of an exposed edge and in recent
contact. Swing feet passing over a lip are therefore not penalized, and different
terrain levels can be evaluated in one vectorized batch without a rasterized
height-field mask.

Each environment owns an active waypoint index and course-progress state. A
reset selects waypoint zero from the route belonging to that environment's
obstacle family and difficulty. A route-control target advances immediately
when the robot enters its 0.20 m XY radius or genuinely crosses its route-normal
plane inside the same lateral corridor. Physical milestones instead require
radius entry and recent foot contact on their explicitly named support polygon.
Both cases immediately retarget the marker, oracle heading, critic distance,
and directional reward without teaching the robot to dwell at control markers.

Only explicitly marked physical milestones split a one-shot `+2` shaping budget
across the course. The flat bootstrap's two intermediate ground targets each
receive `+1`; obstacle approach and alignment guides do not pay, and adding
control markers cannot increase the available budget. Intermediate waypoints
still do not end an episode or count as curriculum success. The final waypoint
ignores plane crossing and pays `+10` only after the robot reaches its radius
with a foot on the named support, sufficient base clearance, low vertical speed
and tilt, and no trunk contact. Event rewards are divided by the control
timestep before Isaac Lab's reward integration, making these configured amounts
the exact per-event bonuses. The terminating trunk-contact penalty uses the same
rule for an exact `-10`; a trunk crash takes precedence if crash and success
would otherwise occur on the same step.

Maximum progress is projected onto the active route segment only inside its
lateral corridor, remains monotonic within an episode, and does not increase
during trunk contact. Demotion therefore compares comparable route fractions
instead of commanded speed times episode duration. Three successes in the last
five frontier attempts promote one row. Two stalled failures in the last three
eligible frontier attempts, each below 60% normalized route progress, demote one
row. The first harder attempt is protected from demotion, and 25% of later
episodes replay the immediate predecessor without altering frontier evidence.
Promotion changes only future course sampling and pays no additional reward
because completion already receives `+10`.

`ParkourCurriculumState` is the complete mutable curriculum-buffer inventory:
per-environment frontier levels, promotion histories, demotion histories, and
post-promotion grace counters. Static thresholds remain in
`ParkourCurriculumCfg`; the episode row selected by frontier/replay sampling
remains authoritative in `TerrainImporter.terrain_levels`.

Dense acquisition scales positive forward speed up to the active course target,
then exponentially suppresses overspeed and lateral velocity. Standing still
and moving backward receive zero. The positive heading kernel is gated by the
same forward-speed fraction, so facing the target while stationary earns
nothing. Both terms are suppressed on the exact retarget step so a marker jump
is not mistaken for robot motion. Flat terrain disables feet-air-time shaping
and obstacle-only leg, edge, and stumble penalties; small air-time shaping and
the contact penalties activate after promotion. Low-clearance error remains
normalized to `[0, 1]` before its squared penalty.

The teacher-interface manifest is version 13. Version 4 introduced complete
declarative terrain courses because physical support segmentation changes the
privileged ray values seen by the teacher. Version 5 replaces horizontal-only
support metadata with ordered planar XYZ boundaries, making the banked ramp
surfaces and their safety edges part of the frozen checkpoint interface.
Version 6 freezes the complete obstacle-family by difficulty matrix. Version 7
separates training-only noise, delay, and corruption switches from the
deterministic checkpoint inference interface. Version 8 records the modular
privileged scan encoder, fixed terrain latent, transferable motor actor, and
their checkpoint paths and input ordering. Version 9 adds actual privileged
dynamics, the ten-step deployable history, both adaptation encoders, and the
required 20-D motor adaptation input. Version 10 adds the shared flat bootstrap
row and shifts the five family-specific obstacle difficulties to rows 1–5.
Version 11 records physical-milestone annotations, immediate intermediate
radius-or-plane transitions, and the former final-only proximity dwell. Version
12 removes that dwell and standardizes navigation names around active and final
waypoints. Version 13 records fixed per-term observation scaling, named support
targets with contact-gated physical milestones, stable crash-free completion,
and the revised flat and tilted-ramp curriculum geometry.

## Phase 1 observation architecture

Phase 1 trains an asymmetric, privileged parkour teacher. The runtime roles are:

| Role | RSL-RL observation mapping | Inputs |
|---|---|---|
| Teacher actor | `policy + heading_target + terrain + dynamics` | 43-D deployable state, 2-D oracle heading, 264-D privileged scan, and 31-D actual dynamics (340 total) |
| Privileged critic | `policy + heading_target + terrain + dynamics + critic_privileged` | Everything seen by the teacher, plus 11-D simulator-only value and route-phase information (351 total) |
| Restricted student | `policy + student_exteroception + adaptation_history` | The same 43-D deployable state, a temporary 32-D terrain latent, and ten-step 430-D deployable history (505 observed values; its motor also receives predicted 2-D heading and 20-D adaptation latents) |

The shared `policy` group is base angular velocity, projected gravity, desired
speed, relative joint position and velocity, and the previous action. Fixed
per-term scales keep angular velocity and joint velocity numerically controlled
without checkpoint-dependent running normalization. The teacher then receives
the yaw-aligned oracle heading, simulator ray-cast height scan, and actual
randomized dynamics. Heights are clipped to ±0.50 m and normalized to `[-1, 1]`,
followed by a binary hit-validity mask. Missing hits use normalized height `+1`
and mask `0`, so a future gap cannot be confused with a valid surface.

Exact active-waypoint distance, scaled base linear velocity and clearance, a
two-component normalized route cursor/progress phase, and simulator-derived
foot contacts are critic-only. In particular, foot contacts are not classified
as deployable until equivalent hardware sensing is defined.

The ray grid uses explicit `xy` flattening: longitudinal X changes fastest,
then lateral Y. It has 12 longitudinal samples from -0.45 m behind the trunk to
1.20 m ahead and 11 lateral samples from -0.75 m to 0.75 m. A flattened height
index is `lateral_index * 12 + longitudinal_index`; the validity mask uses the
same index mapping. Isaac Lab preserves the configured term order, while the
RSL-RL `obs_groups` configuration defines which groups are concatenated for
the policy and critic. Each run stores the resolved environment and agent
configuration in `params/`; use those files with the corresponding checkpoint.

The current terrain and every configured structure are baked into the static
generated `/World/Ground` mesh, so the standard static ray caster is correct.
If future obstacles become separate objects that move during resets, the sensor
must not keep using this static target. Either bake the obstacles into generated
terrain or upgrade to an Isaac Lab release that provides transform-aware
multi-mesh ray casting.

The teacher and student deliberately reuse one `policy` group, preventing two
copies of the deployable state order from drifting apart. The student excludes
the oracle heading, exact active-waypoint distance, simulator ray hits, contacts and
clearance, curriculum-level and obstacle-family identifiers, and configured
obstacle dimensions. Its 32-D exteroception group is all zeros in this tutorial;
it represents the fixed-width output contract of a later depth encoder, not a
visual representation.

Teacher and future student share one action contract: 12 Unitree A1 joint-position
offsets, scale `0.25`, interpreted relative to default joint positions at the
same 50 Hz control rate. Observation asymmetry therefore does not alter the
low-level controller or action interface.

`learning/distillation/architecture.py` fixes the shared transferable motor
input order as deployable state, two-component heading, 32-D terrain latent,
and a required 20-D adaptation latent. Teacher-specific PyTorch composition
lives in `teacher/model.py`, while `teacher/rsl_rl.py` contains the RSL-RL
`ActorCritic` and regularized-PPO adapters. The student remains in `student.py`
because it does not currently require a package of its own.

The Phase-1 actor compresses the 264-D privileged height-and-validity scan to
the fixed 32-D terrain latent. It separately compresses 31 actual randomized
dynamics values—trunk mass and center of mass, mean contact material, and
per-joint stiffness and damping ratios—to the 20-D adaptation latent. The
motor therefore always receives the same 97 values: 43-D deployable state,
2-D heading, 32-D terrain latent, and 20-D adaptation latent.

Isaac Lab also maintains a ten-step, term-major flattened history of the 43-D
deployable state, including previous actions. Full regularized online adaptation
uses two directed losses: `actor.history_encoder` learns the detached latent
from `actor.dynamics_encoder`, while the reverse stop-gradient loss regularizes
the privileged latent so that it remains reproducible from deployable history.
After a 200-iteration warmup, the reverse coefficient ramps from 0 to 0.1 over
300 iterations. Every twentieth rollout is collected and optimized through the
history path rather than privileged dynamics, exposing it to its own state
distribution. The checkpoint additionally keeps the ROA schedule position, scan
encoder under `actor.terrain_encoder`, and shared motor under `actor.motor`.
During deployment, the student copies the history encoder and motor, and
estimates adaptation solely from that deployable history; the privileged
dynamics vector is not an input to `StudentPolicy`.

Three controlled RSL-RL entry points support observation ablations without
changing PPO settings, hidden layers, rewards, curricula, or actions:

```text
rsl_rl_baseline_cfg_entry_point              actor: policy + oracle heading + dynamics
rsl_rl_privileged_critic_cfg_entry_point     actor: policy + oracle heading + dynamics; critic also sees terrain
rsl_rl_cfg_entry_point                       actor and critic see policy + oracle heading + terrain + dynamics
```

Select one with `--agent=<entry-point>` and use that same entry point for
training and playback.

## Evaluate and record video

Evaluate a checkpoint across the complete fixed matrix with one command.
Families are `gap`, `high_step`, `hurdle`, and `tilted_ramps`; difficulty
levels are zero-based from `0` (shared flat bootstrap) through `5` (hardest).

```bash
python scripts/rsl_rl/play.py \
  --task=Parkour-Lab-Play-v0 \
  --checkpoint=/absolute/path/to/model_1000.pt \
  --all_courses \
  --eval_episodes=20 \
  --headless
```

For a shorter diagnostic or video, select one fixed cell instead:

```bash
python scripts/rsl_rl/play.py \
  --task=Parkour-Lab-Play-v0 \
  --checkpoint=/absolute/path/to/model_1000.pt \
  --terrain_family=gap \
  --difficulty_level=1 \
  --eval_episodes=20 \
  --headless
```

Then record one representative full-episode clip with the same checkpoint,
level, and seed:

```bash
python scripts/rsl_rl/play.py \
  --task=Parkour-Lab-Play-v0 \
  --checkpoint=/absolute/path/to/model_1000.pt \
  --terrain_family=gap \
  --difficulty_level=1 \
  --eval_episodes=1 \
  --headless \
  --video
```

`--all_courses` creates all 24 independent reports; it cannot be combined with
the two single-cell selectors. Evaluation reports success, maximum course
progress, trunk contact, timeout, return, episode length, forward speed,
overspeed, vertical-velocity RMS, and all-feet-airborne fraction for each
selected matrix cell. It writes
`metrics.json` plus the optional MP4 beneath
`<run>/evaluation/<checkpoint>-<hash>/family_<family>/level_<n>/seed_<seed>/`, separated
into `metrics/episodes_<n>/` and `video/episodes_<n>-steps_<length>/`.
Each invocation gets a timestamped `run_*` leaf so before/after results are not
overwritten. Use `--video_output_dir` to choose another artifact root. Omit
`--video` for faster numerical evaluation.

## Online student-driven distillation

The distillation pipeline accepts the exact teacher checkpoint directly.
Teacher training writes the compact `params/teacher_interface.json` manifest
beside its checkpoints. Distillation hashes the requested checkpoint, validates
that manifest, and reconstructs the runtime interface before loading the
policy. The manifest covers only checkpoint-facing semantics: actor observation
order and dimensions, normalization, terrain preprocessing, action order and
scaling, and control timing. It deliberately excludes critic details, unused
observation groups, framework versions, and source-code hashes so unrelated
extensions and behavior-preserving refactors do not invalidate a teacher.

Use `play.py` independently to compare promising checkpoints under identical
fixed evaluation conditions. After choosing one from those results, pass its
checkpoint path directly to distillation:

```bash
python scripts/rsl_rl/distill.py \
  --task=Parkour-Lab-v0 \
  --teacher_checkpoint=/absolute/path/to/model_1000.pt \
  --allow_zero_exteroception \
  --max_iterations=2 \
  --steps_per_iteration=2 \
  --headless
```

At this stage, `--allow_zero_exteroception` is an explicit acknowledgement
that the run is only a short pipeline smoke test. Without that option,
`distill.py` refuses to train from the information-free placeholder. A
constant-zero feature cannot describe obstacles, so checkpoints from this
stage must not be selected as terrain-aware or deployable students.

The distinct information sets are:

| Information set | Shape | Contents and status |
|---|---:|---|
| Teacher actor observations | Runtime-derived; currently `[N, 340]` | Shared 43-D `policy`, 2-D oracle `heading_target`, 264-D privileged `terrain`, and 31-D privileged `dynamics` |
| Student observations | Runtime-derived; currently `[N, 505]` | Shared 43-D `policy`, temporary 32-D zero terrain latent, and 430-D deployable `adaptation_history`; no privileged dynamics |
| Deployable adaptation history | `[N, 430]` | Ten current-and-past samples of deployable proprioception, command state, and previous actions; flattened term-major from oldest to newest |
| Privileged dynamics target | `[N, 31]` | Actual randomized physical properties used only to train the teacher's privileged encoder and supervise the history encoder |
| Oracle heading target | `[N, 2]` | Yaw-aligned direction to the active course waypoint, `[forward, left] = [cos(Δψ), sin(Δψ)]`; teacher input and student supervision, never a student motor input |
| Teacher motor-action target | `[N, 12]` | Frozen teacher's deterministic action-distribution mean in resolved A1 joint order; supervision only |

The instantiated group dimensions and exact student group order are stored in
`distillation_config.json`. Frames, units, normalization, and deployment status
are documented here and beside the corresponding observation definitions,
avoiding a second hard-coded runtime description. The heading head predicts a
two-vector from restricted student information, normalizes it to a unit
direction, and appends that predicted direction—not the oracle target—to the
motor MLP. This is continuous across the `-pi/+pi` boundary. At an exactly
reached active waypoint, where direction is undefined, the target
deterministically falls back to body-forward until the waypoint transition is
applied.

Every online transition follows one ownership rule: construct teacher and
student inputs from the same current state, obtain the frozen teacher mean as a
label, obtain the student action, store the pair, and step physics only with the
student action. Consequently, `last_action` is the previously executed student
action and training data comes from student-visited states. The initial losses
are action Smooth L1/Huber (`1.0`), heading cosine direction (`0.2`), and raw
heading-vector unit-norm regularization (`0.01`). The teacher is in evaluation
mode, label generation runs under inference mode, and only student parameters
enter the optimizer.

This stage does not render cameras, encode depth, or claim that the zero feature
is deployable perception. It establishes and tests the information barrier and
student-driven training semantics. The terrain-latent contract is fixed at 32
values and stored with the student model; both the privileged scan encoder and
future depth encoder must produce that width. A new student starts from the
exact `actor.motor` weights in the selected teacher checkpoint before online
updates begin. Teacher and student still emit the same 12 action values, which
use the same scale, default-position offset, controller, and 50 Hz rate.

Runs are stored beneath `logs/distillation/parkour_lab/`. Each run records the
teacher checkpoint identity, runtime group dimensions and student group order,
resolved environment and teacher configuration, JSONL losses, pre-update
teacher/student action L2 disagreement, and student-only checkpoints. The run
configuration labels the current mode as
`zero_exteroception_pipeline_smoke_test`. A resumed student is accepted when
its serialization version, model configuration, exact teacher checkpoint, and
teacher interface match. The runtime teacher interface is validated separately
before the checkpoint is loaded.

## Evaluation best practice

Treat success rate and failure outcomes over multiple episodes as the primary
comparison; use video to understand *why* behavior changed. For fair before/after
comparisons:

- evaluate each promising checkpoint on all 24 family/difficulty cells;
- keep the seed, number of episodes, and environment count unchanged;
- do not enable the adaptive training curriculum during evaluation;
- compare the same metrics before selecting representative clips;
- record short clips after the numerical run, since rendering reduces throughput.

Training videos are useful for monitoring, but they are not a stable benchmark:
the training task changes difficulty as the policy succeeds or fails.

## Simulator-free checks

The difficulty mapping has dependency-free unit tests, so it can be checked
without launching Isaac Sim:

```bash
python -m unittest discover -s tests -v
python -m compileall -q source/parkour_lab/parkour_lab scripts
```

Use the zero-action and random-action scripts as full simulator smoke tests:

```bash
python scripts/zero_agent.py --task=Parkour-Lab-v0 --num_envs=4 --headless
python scripts/random_agent.py --task=Parkour-Lab-v0 --num_envs=4 --headless
```

If Isaac Lab is not installed in the active Python environment, replace
`python` with the Isaac Lab launcher, for example
`PATH_TO_ISAACLAB/isaaclab.sh -p`.
