# Parkour Lab

Parkour Lab is an Isaac Lab reinforcement-learning environment for training a
Unitree A1 to reach a goal across progressively harder obstacles. Training uses
an adaptive four-level terrain curriculum; evaluation freezes one level so that
policy changes can be compared under the same conditions and recorded on video.

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
- `Parkour-Lab-Play-v0` is the smaller, fixed-difficulty task for
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

## Phase 1 observation architecture

Phase 1 trains an asymmetric, privileged parkour teacher. The runtime roles are:

| Role | RSL-RL observation mapping | Inputs |
|---|---|---|
| Teacher actor | `policy + terrain` | 50-D teacher state/task group plus 132 normalized terrain heights and their 132-D validity mask (314 total) |
| Privileged critic | `policy + terrain + critic_privileged` | Everything seen by the teacher, plus exact base linear velocity and base clearance (318 total) |
| Restricted student | `student_policy + student_exteroception` | 47-D restricted proprioceptive/task state plus a temporary 64-D zero exteroception boundary (111 total) |

The teacher's `policy` group is base angular velocity, projected gravity, exact
body-frame goal direction and distance, desired speed, relative joint position
and velocity, previous action, and foot-contact state. The teacher appends the
simulator ray-cast height scan, clipped to ±0.50 m and normalized to `[-1, 1]`,
followed by a binary hit-validity mask. Missing hits use normalized height `+1`
and mask `0`, so a future gap cannot be confused with a valid surface. The
critic appends only exact base linear velocity and clearance to the complete
teacher input. The restricted student reuses the deployable proprioceptive,
speed-command, action-history, and contact subset, but not the exact goal terms.

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

The restricted student groups are independent of the RSL-RL PPO routes, so
adding them does not change the teacher checkpoint input. The student excludes
the exact goal direction and distance, simulator ray hits, exact base clearance,
curriculum-level and obstacle-family identifiers, and configured obstacle
dimensions. Its 64-D exteroception group is all zeros in this tutorial; it is a
replaceable interface for a later depth encoder, not a visual representation.

Teacher and future student share one action contract: 12 Unitree A1 joint-position
offsets, scale `0.25`, interpreted relative to default joint positions at the
same 50 Hz control rate. Observation asymmetry therefore does not alter the
low-level controller or action interface.

Three controlled RSL-RL entry points support observation ablations without
changing PPO settings, hidden layers, rewards, curricula, or actions:

```text
rsl_rl_baseline_cfg_entry_point              actor: policy
rsl_rl_privileged_critic_cfg_entry_point     actor: policy; critic also sees terrain
rsl_rl_cfg_entry_point                       actor and critic both see terrain
```

Select one with `--agent=<entry-point>` and use that same entry point for
training and playback.

## Evaluate and record video

Evaluate a checkpoint on one frozen difficulty at a time. Difficulty levels are
zero-based: `0` is easiest and `3` is hardest.

```bash
python scripts/rsl_rl/play.py \
  --task=Parkour-Lab-Play-v0 \
  --checkpoint=/absolute/path/to/model_150.pt \
  --difficulty_level=0 \
  --eval_episodes=20 \
  --headless
```

Then record one representative full-episode clip with the same checkpoint,
level, and seed:

```bash
python scripts/rsl_rl/play.py \
  --task=Parkour-Lab-Play-v0 \
  --checkpoint=/absolute/path/to/model_150.pt \
  --difficulty_level=0 \
  --eval_episodes=1 \
  --headless \
  --video
```

Repeat the command with `--difficulty_level=1`, `2`, and `3` for a complete
comparison. Evaluation reports episode outcomes for the selected level and
writes `metrics.json` plus the optional MP4 beneath
`<run>/evaluation/<checkpoint>-<hash>/level_<n>/seed_<seed>/`, separated
into `metrics/episodes_<n>/` and `video/episodes_<n>-steps_<length>/`.
Each invocation gets a timestamped `run_*` leaf so before/after results are not
overwritten. Use `--video_output_dir` to choose another artifact root. Omit
`--video` for faster numerical evaluation.

## Online student-driven distillation

The distillation pipeline accepts a teacher only through one or more completed
fixed-evaluation `metrics.json` files. Teacher training writes the compact
`params/teacher_interface.json` manifest; fixed evaluation verifies it and
records the checkpoint's absolute path and SHA-256. Distillation rechecks the
checkpoint bytes and interface before loading the policy. The manifest covers
only checkpoint-facing semantics: actor observation order and dimensions,
normalization, terrain preprocessing, action order and scaling, and control
timing. It deliberately excludes critic details, unused observation groups,
framework versions, and source-code hashes so unrelated extensions and
behavior-preserving refactors do not invalidate a teacher.

After evaluating the selected teacher on fixed levels, start online
distillation with the resulting metric files:

```bash
python scripts/rsl_rl/distill.py \
  --task=Parkour-Lab-v0 \
  --teacher_evaluation_metrics \
    /absolute/path/to/level_0/metrics.json \
    /absolute/path/to/level_1/metrics.json \
    /absolute/path/to/level_2/metrics.json \
    /absolute/path/to/level_3/metrics.json \
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

The four distinct information sets are:

| Information set | Shape | Contents and status |
|---|---:|---|
| Teacher observations | Runtime-derived; currently `[N, 314]` | `policy` followed by privileged `terrain`; label generation only |
| Student policy observations | Runtime-derived; currently `[N, 111]` | `student_policy` followed by the temporary zero-valued exteroceptive feature; student input |
| Oracle heading target | `[N, 2]` | Yaw-aligned body-frame `[forward, left] = [cos(Δψ), sin(Δψ)]`; supervision only, never a student input |
| Teacher motor-action target | `[N, 12]` | Frozen teacher's deterministic action-distribution mean in resolved A1 joint order; supervision only |

The instantiated group dimensions and exact student group order are stored in
`distillation_config.json`. Frames, units, normalization, and deployment status
are documented here and beside the corresponding observation definitions,
avoiding a second hard-coded runtime description. The heading head predicts a
two-vector from restricted student information, normalizes it to a unit
direction, and appends that predicted direction—not the oracle target—to the
motor MLP. This is continuous across the `-pi/+pi` boundary. At an exactly
reached waypoint, where direction is undefined, the target deterministically
falls back to body-forward.

Every online transition follows one ownership rule: construct teacher and
student inputs from the same current state, obtain the frozen teacher mean as a
label, obtain the student action, store the pair, and step physics only with the
student action. Consequently, `last_action` is the previously executed student
action and training data comes from student-visited states. The initial losses
are action Smooth L1/Huber (`1.0`), heading cosine direction (`0.2`), and raw
heading-vector unit-norm regularization (`0.01`). The teacher is in evaluation
mode, all of its parameters have gradients disabled, and only student
parameters enter the optimizer.

This stage does not render cameras, encode depth, or claim that the zero feature
is deployable perception. It establishes and tests the information barrier and
student-driven training semantics. The current placeholder is 64-D only as a
provisional configuration value; its actual width is derived at runtime and
stored with the student model, so the future depth architecture can choose a
different width deliberately. Before fully student-driven perceptive training,
the next stage should warm-start the otherwise zero-output motor student from
teacher-labeled data and then switch to student-visited online rollouts. Teacher
and student still emit the same 12 action values, which use the same scale,
default-position offset, controller, and 50 Hz rate.

Runs are stored beneath `logs/distillation/parkour_lab/`. Each run records the
teacher selection, runtime group dimensions and student group order, resolved
environment and teacher configuration, JSONL losses, pre-update teacher/student
action L2 disagreement, and student-only checkpoints. The run configuration
labels the current mode as `zero_exteroception_pipeline_smoke_test`. A resumed
student is accepted when its serialization version, model configuration, and
exact teacher checkpoint match. The runtime teacher interface is validated
separately before the checkpoint is loaded.

## Evaluation best practice

Treat success rate and failure outcomes over multiple episodes as the primary
comparison; use video to understand *why* behavior changed. For fair before/after
comparisons:

- evaluate every checkpoint on all four fixed levels;
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
