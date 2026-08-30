# Parkour Lab

Parkour Lab is an Isaac Lab reinforcement-learning environment for training a
Unitree Go2 to finish waypoint routes across progressively harder obstacles.
Training uses a balanced obstacle-family by difficulty terrain matrix with
small deterministic geometry variants inside each cell; evaluation freezes the
canonical variant of one matrix cell so policy changes can be compared and
recorded under the same conditions.

## Setup

Install Isaac Lab, then install this extension with the Python interpreter from
the same environment:

```bash
python -m pip install -e source/parkour_lab
```

Parkour Lab uses the following tested server stack as one compatibility unit:

| Package | Version |
| --- | --- |
| Isaac Lab | `2.3.2.post1` |
| Isaac Sim | `5.1.0.0` |
| PyTorch | `2.7.0+cu128` |
| TorchVision | `0.22.0+cu128` |
| RSL-RL | `3.1.2` |

Training and evaluation validate these exact versions before starting Isaac
Sim. Install RSL-RL through the matching Isaac Lab checkout before installing
this extension:

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
Training also archives Git provenance. It refuses untracked executable files
under `scripts/` or `source/`, because RSL-RL's Git diff cannot preserve their
contents; stage or commit new source before starting a run.

To retain the complete terminal report while still watching it live, run Python
unbuffered and copy stdout and stderr with `tee`:

```bash
python -u scripts/rsl_rl/train.py \
  --task=Parkour-Lab-v0 \
  --headless 2>&1 | tee training.log
```

Each completed-episode batch prints behavior-neutral diagnostics under
`Curriculum/training_diagnostics/`. In addition to pooled per-foot values,
per-episode distributions report zero-touchdown feet, minimum contact and load
fractions, maximum uninterrupted non-contact time, and absolute left-right and
front-rear imbalance. This prevents opposite tripod modes in different
environments from averaging into an apparently balanced gait. Per-leg action
magnitude and rate, default-pose deviation, tracking error, applied torque,
torque clipping, and velocity-limit occupancy separate a learned tucked
command from actuator or joint-limit problems. Task metrics report signed
forward speed, speed error,
lateral motion, and retreat;
body metrics report tilt, vertical motion, clearance, and missing base-ray
hits. Episode progress is reported for every reset, while
`Curriculum/terrain_levels/family/*` reports attempt coverage, mutually
exclusive failure causes, frontier success, stalls, and both geometric and
waypoint progress separately for each obstacle family. It also separates the
mean speed command of successful and failed frontier attempts, exposing
speed-dependent feasibility failures. The single
`Episode_Reward/training_diagnostics` entry is intentionally always zero: the
sampler never changes the reward or policy objective.

The terminal also prints one JSON diagnostic-context line containing the seed,
timing, speed range, reward weights, PPO and symmetry settings, randomization
stage, curriculum dimensions, and resolved robot, sensor, foot, and joint
orders. Keep `training.log` together with the run's `params/env.yaml` and
`params/agent.yaml` when comparing experiments.

The default teacher budget is 1,000 PPO iterations with 48 control steps per
environment and update, a 25-second commanded-motion budget, a 32-second hard
wall-clock cap, a discount factor of 0.995, and checkpoints every 100
iterations. The longer rollout and horizon preserve more approach-to-landing
credit than the former short smoke-test settings.
`--max_iterations` can still override the budget for diagnostics. Action noise
starts at 0.8 and is bounded to 0.10–0.80. A 0.01 entropy coefficient retains
exploration through initial locomotion acquisition while allowing the bounded
distribution to settle around a deterministic gait.

New checkpoints store the per-environment curriculum frontier, rolling evidence,
and demotion grace alongside RSL-RL's learner state. `--resume` restores that
memory only when checkpoint and runtime terrain/curriculum provenance match,
then starts fresh episodes sampled from the restored frontier. Changed geometry
still permits weight/optimizer fine-tuning but restarts curriculum memory from
the configured bootstrap rows. A same-terrain checkpoint without curriculum
state is rejected instead of silently changing resume semantics. Fixed
`Parkour-Lab-Play-v0` evaluation remains pinned to its requested matrix cell.
After promotion grace, training keeps 75% of eligible environments at their
frontier, replays its immediate predecessor in 15%, and anchors the shared flat
bootstrap in 10%. At frontier one, both replay choices resolve to level zero.
Once a family reaches the final row, the same combined 25% replay budget is
rebalanced to keep 15% of all eligible episodes on the shared flat row. The
remaining 10% is distributed uniformly across acquired lower obstacle rows,
preserving 75% final-frontier exposure while continually rehearsing ordinary
locomotion.

### Staged domain randomization

Domain randomization is deliberately disabled by default so the teacher first
learns the nominal task. Resume a successful nominal run with the narrow stage,
then use the wide stage only after narrow training remains stable:

```bash
python scripts/rsl_rl/train.py \
  --task=Parkour-Lab-v0 \
  --resume \
  --checkpoint=/absolute/path/to/model_1000.pt \
  --domain_randomization_stage=narrow \
  --headless
```

`--checkpoint` has the same complete-path meaning in `train.py` and `play.py`.
Alternatively, use `--load_run=<pattern>` for automatic lookup using the
runner's configured checkpoint pattern; the two selectors cannot be combined.

Replace `narrow` with `wide` for the final stage. The selected profile perturbs
contact friction and restitution, base payload and center of mass, actuator
gains, initial pose and velocity, external pushes, control-step action latency,
and delayed/noisy proprioception. The selected stage and configured ranges are
stored in `params/env.yaml`. Action and observation widths do not change between
stages. Fixed evaluation always selects `off`, removing randomization, noise,
pushes, and delays before constructing the environment.

## Declarative course configuration

Each curriculum matrix cell is a reusable course description rather than a
special case in the runtime. Terrain row 0 is obstacle-free: variants 0–5 use
the canonical straight route, variants 6/7 use an exact mirrored pair of
constant-heading rotated-straight routes, and variants 8/9 use an exact mirrored
pair of broad gentle turns. Rows 1 through 6 contain the six obstacle
difficulties. Equal column blocks retain their future
gap, high-step, hurdle, or tilted-ramp family even on the flat row, so each
environment advances from flat ground into one stable family without
resampling its obstacle type. Each family block is further divided into ten
deterministic variants: five zero-mean severity values, each represented by an
adjacent left-right pair.
Variant zero is the unchanged nominal ladder. High-step and tilted-ramp pairs
contain exact reflected courses; centered gap and hurdle geometries are their
own reflections. Nonzero severity values perturb only bounded obstacle geometry
by at most 5%. Every variant is a complete prebuilt course, so its mesh, route,
support polygons, and edge geometry remain exact and synchronized. Isaac Lab's
unretained random fraction within a terrain row is deliberately not used.
Default builders use the shared normalized row scalar
`s = row / 6`; simple obstacle dimensions interpolate from that scalar while
the proven tilted-ramp stages remain explicit normalized keyframes. Each course
records ordered terrain-local waypoints, named mesh structures and their
factory arguments, planar support polygons, nominal speed metadata, clearance,
and explicit difficulty metadata. Moving commands are sampled independently
from 0.20 to 0.70 m/s on the flat row and 0.45 to 0.70 m/s on obstacle rows.
Only the flat row receives exact-zero stop/wait/restart windows. Obstacle rows
use a common 0.27 m minimum clearance. Forward reward saturates at the sampled command while a
waypoint-local 1.5x ceiling permits the policy to acquire extra traversal speed
without rewarding it for doing so.
Factory arguments are passed directly to each structure factory. A support
region either refers to the generated base ground or names the structure whose
surface it describes. Base-ground regions are authoritative physical patches:
the terrain generator emits one collision box for each such rectangle, so an
uncovered interval is a real hole in both collision and raycast geometry. Named
regions annotate separately generated structures without duplicating them.
Terrain generation still iterates configured structures generically; it does
not branch on a level number or obstacle family.

The high-step family uses 1.6 m deep elevated platforms with vertical,
ground-mounted front faces. A supported center-top waypoint credits the climb
and landing, followed by a final target near the rear of the annotated top
surface so completion also requires stable platform traversal.
The hurdle family uses 0.18 m deep barriers spanning the full 4.0 m tile width,
blocking every lateral path around an exposed end while the robot remains
inside its assigned tile. Base-ground support remains continuous beneath and
beyond each barrier, its top is not declared traversable, and a supported
post-hurdle landing milestone precedes a later ground exit. Heights increase
within each family while remaining below the 0.5 m maximum.

Gap widths increase from 0.10 m to 0.50 m. Approach and landing supports stop
at opposite gap lips, and the ordered route directs the robot onto the landing
side before the final waypoint. The approach waypoint remains an unrewarded
directional guide; the supported post-gap landing is explicitly marked as a
physical milestone.

The tilted-ramp family uses an acquisition ladder instead of introducing the
complete compound obstacle at once. Its first obstacle row contains one wide,
straight, gently banked slab. Later rows add a contiguous second slab, then an
inter-ramp gap, then yaw and lateral offset. A dedicated bridge row interpolates
the support width, banks, redirection, and spacing before the hardest row
combines opposite 12-degree banks with yaws of 10 and 32 degrees.
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
when the robot enters its 0.20 m XY radius. Physical milestones additionally
require recent foot contact on their explicitly named support polygon.
Both cases immediately retarget the marker, oracle travel direction, critic distance,
and directional reward without teaching the robot to dwell at control markers.

Only explicitly marked physical milestones split a one-shot `+2` shaping budget
across the course. The flat bootstrap's two intermediate ground targets each
receive `+1`; obstacle approach and alignment guides do not pay, and adding
control markers cannot increase the available budget. Intermediate waypoints
still do not end an episode or count as curriculum success. The final waypoint
pays `+4` only after the robot reaches its radius with a foot on the named
support, sufficient base clearance, low vertical speed and tilt, and no chassis
contact. Event rewards are divided by the control
timestep before Isaac Lab's reward integration, making these configured amounts
the exact per-event bonuses. The terminating chassis-contact penalty uses the
same rule for an exact `-10`; a base or head crash takes precedence if crash
and success would otherwise occur on the same step. Falling more than 0.5 m
below the environment-local course is also a terminal failure, which ends
unrecoverable gap falls promptly without confusing elevated terrain with world
height.

Maximum progress is projected onto the active route segment only inside its
lateral corridor, remains monotonic within an episode, and does not increase
during chassis contact. It remains a dense diagnostic. Demotion instead uses
the fraction of intermediate route waypoints already passed, independently of
the milestone reward budget. This discrete signal changes only when the route
cursor advances, while reward tuning cannot silently change demotion evidence.
Three successes in the last five frontier attempts promote one row.
Two stalled failures in the last three eligible frontier attempts, each below
60% waypoint progress, demote one row. The first harder attempt is protected
from demotion. Later episodes use a 10% flat anchor and 15%
immediate-predecessor replay below the ceiling; at the ceiling their combined
budget reserves 15% for the shared flat row and distributes 10% over acquired
lower obstacle rows. Replay never alters frontier evidence.
Promotion changes only future course sampling and pays no additional reward
because completion already receives an explicit `+4` event.

`ParkourCurriculumState` is the complete mutable curriculum-buffer inventory:
per-environment frontier levels, promotion histories, demotion histories, and
post-promotion grace counters. Static thresholds remain in
`ParkourCurriculumCfg`; the episode row selected by frontier/replay sampling
remains authoritative in `TerrainImporter.terrain_levels`.

Dense velocity shaping uses signed waypoint-directed speed on every terrain,
while suppressing lateral motion. Standing still earns zero task reward and
retreating is penalized. Forward reward saturates at the episode command; the
flat speed ceiling equals that command, while the waypoint-local obstacle
ceiling still permits faster takeoff without paying extra forward reward.
Heading guidance uses the same signed progress gate, preventing a zero-net
fore-aft oscillation from accumulating alignment credit. Reward samples on the
exact retarget step are masked so a marker jump is not mistaken for robot
motion. No air-time, cross-foot timing, equal-load, or periodic-gait term
prescribes contact timing during obstacle traversal. Mild world-up orientation
and all-joint default-pose penalties instead make a persistent lean or folded
leg costly without specifying which feet must be in contact. Undesired hip,
thigh, and calf contacts remain recoverable penalties; base and Go2 head
contacts are terminal. Edge, slide, and stumble penalties use the same
semantics on every terrain row. Low-clearance error remains normalized to
`[0, 1]` before its squared penalty; a missing ray over an intentional gap is
tracked separately and does not masquerade as zero physical clearance. Every
default course uses a 0.27 m Go2 base-clearance floor; this is a lower bound
rather than an exact height-tracking target.

The teacher-interface manifest is version 18. Version 4 introduced complete
declarative terrain courses because physical support segmentation changes the
privileged ray values seen by the teacher. Version 5 replaces horizontal-only
support metadata with ordered planar XYZ boundaries, making the banked ramp
surfaces and their safety edges part of the recorded training provenance.
Version 6 records the complete obstacle-family by difficulty matrix. Version 7
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
and the revised flat and tilted-ramp curriculum geometry. Version 14 freezes
the robot model and exact source asset path, intentionally rejecting older A1
checkpoints despite their compatible action dimensions. Version 15 records the
Go2 chassis-contact contract, where base and head impacts are fatal while limb
contacts remain recoverable. Version 16 moves preferred speed into the intent
command, adds exact-zero stop and signed in-place yaw-rate semantics, and
exposes the separate 2-D requested travel direction. The appended yaw-rate scalar
changes the deployable state/history and motor widths. Version 17 adds the
resolved per-joint processed-target safety clamp. Version 18 derives adaptation
history from the exact delivered policy stream in frame-major order and records
the tested runtime stack. Version 19 names the privileged local input
`oracle_travel_direction` and keeps the requested travel direction solely in
the command manager until a deployable student consumes it. Version 19
checkpoints load strictly; earlier interface versions are not imported or
resumed.

The complete manifest and its hash remain the exact training provenance.
Playback compares a projected inference contract that excludes
command-source, training-provenance, and terrain-curriculum metadata: robot
asset, actor observations, scanner, network, action, waypoint protocol, and
timing mismatches remain fatal, while a changed terrain domain emits an
out-of-distribution warning without making compatible weights unloadable.

## Phase 1 observation architecture

Phase 1 trains an asymmetric, privileged parkour teacher. The runtime roles are:

| Role | RSL-RL observation mapping | Inputs |
|---|---|---|
| Teacher actor | `policy + oracle_travel_direction + terrain + dynamics` | 44-D deployable state, 2-D local travel direction, 264-D privileged scan, and 31-D actual dynamics (341 total) |
| Privileged critic | `policy + oracle_travel_direction + terrain + dynamics + critic_privileged` | Everything seen by the teacher, plus 11-D simulator-only value and route-phase information (352 total) |

The shared `policy` group is base angular velocity, projected gravity, desired
speed, relative joint position and velocity, the previous action, and finally
the signed desired yaw rate. Fixed
per-term scales keep angular velocity and joint velocity numerically controlled
without checkpoint-dependent running normalization. The teacher then receives
the yaw-aligned oracle travel direction, simulator ray-cast height scan, and actual
randomized dynamics. Heights are clipped to ±0.50 m and normalized to `[-1, 1]`,
followed within the same observation term by a binary hit-validity mask.
Missing hits use normalized height `+1` and mask `0`, so a future gap cannot be
confused with a valid surface while the ray data is preprocessed only once.

Exact active-waypoint distance, scaled base linear velocity and clearance, a
two-component normalized route cursor/progress phase, and simulator-derived
foot contacts are critic-only. In particular, foot contacts are not classified
as deployable until equivalent hardware sensing is defined. Go2's low-level
`LowState` likewise does not provide base linear velocity, so simulator truth
is excluded from the actor unless a deployment-equivalent estimator is added.

The ray grid uses explicit `xy` flattening: longitudinal X changes fastest,
then lateral Y. It has 12 longitudinal samples from -0.45 m behind the base to
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

The motor action contract is 12 Unitree Go2 joint-position offsets with scale
`0.25`, interpreted relative to default joint positions at 50 Hz. Observation
asymmetry therefore does not alter the low-level controller or action interface.

Teacher PPO updates append an exact sagittal left-right reflection of every
collected transition. The transform reflects proprioception and actions by
resolved Go2 joint name, flips the travel direction and both terrain-scan
channels, and also
reflects the adaptation history, randomized dynamics, and critic-only state.
Adaptive KL and entropy statistics continue to use only the collected samples.
This is data augmentation rather than a gait-phase objective: mirror loss is
disabled, and inference still evaluates one unmodified observation at a time.

`learning/distillation/architecture.py` fixes the shared transferable motor
input order as deployable state, two-component local travel direction, 32-D
terrain latent, and a required 20-D adaptation latent. Teacher-specific PyTorch composition
lives in `teacher/model.py`, while `teacher/rsl_rl.py` contains the RSL-RL
`ActorCritic` and regularized-PPO adapters. The perception student is deferred
until its real depth encoder and training loop are implemented.

The Phase-1 actor compresses the 264-D privileged height-and-validity scan to
the fixed 32-D terrain latent. It separately compresses 31 actual randomized
dynamics values—base mass and center of mass, mean contact material, and
per-joint stiffness and damping ratios—to the 20-D adaptation latent. The
motor therefore always receives the same 98 values: 44-D deployable state,
2-D local travel direction, 32-D terrain latent, and 20-D adaptation latent.

The RSL-RL wrapper maintains a ten-step, frame-major flattened history of the
exact delivered 44-D policy packets, including previous actions. A reset row is
filled with its first packet and then receives one packet per control step, so
the current observation and newest history frame cannot receive independent
noise or delay. Full regularized online adaptation
uses two directed losses: `actor.history_encoder` learns the detached latent
from `actor.dynamics_encoder`, while the reverse stop-gradient loss regularizes
the privileged latent so that it remains reproducible from deployable history.
After a 200-iteration warmup, the reverse coefficient ramps from 0 to 0.1 over
300 iterations. Every twentieth rollout is collected and optimized through the
history path rather than privileged dynamics, exposing it to its own state
distribution. The checkpoint additionally keeps the ROA schedule position, scan
encoder under `actor.terrain_encoder`, and shared motor under `actor.motor`.
The history-policy evaluation path estimates adaptation solely from deployable
history; the privileged dynamics vector is not one of its runtime inputs.

Three controlled RSL-RL entry points support observation ablations without
changing PPO settings, hidden layers, rewards, curricula, or actions:

```text
rsl_rl_baseline_cfg_entry_point              actor: policy + oracle travel direction + dynamics
rsl_rl_privileged_critic_cfg_entry_point     actor: policy + oracle travel direction + dynamics; critic also sees terrain
rsl_rl_cfg_entry_point                       actor and critic see policy + oracle travel direction + terrain + dynamics
```

Select one with `--agent=<entry-point>` and use that same entry point for
training and playback.

## Evaluate and record video

Evaluate a checkpoint across the complete fixed matrix with one command.
Families are `gap`, `high_step`, `hurdle`, and `tilted_ramps`; difficulty
levels are zero-based from `0` (shared flat bootstrap) through `6` (hardest).

```bash
python scripts/rsl_rl/play.py \
  --task=Parkour-Lab-Play-v0 \
  --checkpoint=/absolute/path/to/model_1000.pt \
  --all_courses \
  --desired_speed=0.55 \
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
  --desired_speed=0.55 \
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
  --desired_speed=0.55 \
  --eval_episodes=1 \
  --headless \
  --video
```

`--geometry_variant=0..9` fixes the within-family geometry; it defaults to the
canonical variant 0 and can also be combined with `--all_courses` to evaluate
one complete 28-cell variant slice. At difficulty 0, variants 6/7 select the
mirrored constant-heading cohort and variants 8/9 select the mirrored
gentle-turn cohort. `--desired_speed` fixes the scalar command for every reset.
When omitted, the selected course's nominal speed is retained.
On level 0, `--desired_yaw_rate=<signed rad/s>` selects deterministic
translate→two-second-pivot→translate pulses; `--desired_speed` is the positive
restart speed, or the course nominal when omitted. Ordinary training and
evaluation sample symmetric `0.25-0.80 rad/s`, `0.75-4.0 s` pivot windows on
only 5% of eligible flat command transitions; obstacle rows never receive
them.
Run the same matrix at 0.45, 0.55, and 0.70 m/s to measure command conditioning
separately from terrain difficulty. `--all_courses` creates all 28 independent
reports, starting a fresh Isaac Sim
process for every cell so simulation state cannot leak between courses. The
expected application restarts are printed as sweep progress. This option cannot
be combined with the two single-cell selectors. Evaluation reports success,
maximum course progress, chassis contact, below-course falls, timeout, return,
episode length, forward speed, overspeed, vertical-velocity RMS, and
all-feet-airborne fraction for each selected matrix cell. Command diagnostics
separate moving speed error, stopped planar speed/yaw rate, mean/p95
movement-direction error, first-threshold-crossing stop settling, maximum XY
excursion during the following two seconds, restart outcomes, and active versus
wall-only timeouts. Pivot samples are excluded from stop denominators and
separately report planar speed, mean/p95 yaw-rate error, wrong-way fraction,
exposure, maximum XY excursion, and eventual course success after a
pivot-to-translation restart. The unclamped oracle-residual report retains
signed ranges,
absolute p50/p95/p99/p99.9/max tails, the provisional 35-degree exceedance rate,
success/failure splits, and waypoint-transition samples. Reports also retain
mean per-episode p50/p95 and the global maximum root cross-track distance for
successful episodes, measured against the finite configured waypoint polyline,
plus the mean per-episode fraction beyond the configured `0.30 m` soft
half-width and raw values at waypoint transitions. The dedicated off-route rate
reports hard-width violations. During training, moving excess beyond the soft
width receives a small squared cost; crossing the `0.50 m` hard width is an
explicit off-route failure that cannot advance route state. Neither distance
nor width is an actor observation. Evaluation samples after physics and reward
computation but before Isaac Lab auto-resets completed rows, so the terminal
state is included and the following reset state is excluded.
Evaluation writes
`metrics.json` plus the optional MP4 beneath
`<run>/evaluation/<checkpoint>-<hash>/family_<family>/level_<n>/variant_<v>/speed_<m_s>/yaw_rate_<rad_s>/seed_<seed>/`, separated
into `metrics/episodes_<n>/` and `video/episodes_<n>-steps_<length>/`.
Each invocation gets a timestamped `run_*` leaf so before/after results are not
overwritten. Use `--video_output_dir` to choose another artifact root. Omit
`--video` for faster numerical evaluation.

## Perception student status

The perception student and its online distillation entry point are intentionally
deferred. They will be added with the real recurrent depth encoder and
student-driven data collection stage; the repository does not expose a
zero-information placeholder policy or a smoke-test checkpoint format.

## Evaluation best practice

Treat success rate and failure outcomes over multiple episodes as the primary
comparison; use video to understand *why* behavior changed. For fair before/after
comparisons:

- evaluate each promising checkpoint on all 28 family/difficulty cells;
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
