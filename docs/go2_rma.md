# Go2 RMA locomotion and operator control

This stage targets **simulation locomotion, obstacle traversal, stop/restart and
in-place yaw control**. It does not implement DreamFLEX, capability adaptation,
a world model, or hardware deployment.

## Architecture

The shared motor receives current proprioception, a travel command, a terrain
latent, a 20-D RMA dynamics latent, and a separate 3-D body-velocity estimate.
The velocity head reads the last ten delivered proprioceptive frames (50 Hz).
Simulator linear velocity is an auxiliary MSE label only. Both teacher and
history-policy rollouts/inference use the **estimated** velocity; PPO cannot
backpropagate through that estimate. The estimator therefore stays a physical
state estimate rather than becoming another unconstrained policy latent.

Teacher and deployable-history paths still share one motor and terrain encoder.
History-only dynamics inference never reads the privileged dynamics group.
Terrain sensing is still the simulator height scan: this is an RMA-style **sim
baseline**, not a camera-to-real-robot deployment stack. The separation of explicit
state estimation and implicit dynamics adaptation follows the
[Extreme Parkour authors' architecture](https://github.com/chengxuxin/extreme-parkour).

Human commands take precedence over waypoint direction and terminal speed
assistance. Scripted training/evaluation retains its existing waypoint guidance.
The first operator mode supports forward, stop and pivot, not reverse, lateral
walking or simultaneous translation-plus-yaw. Turn on stable ground, then move
forward toward the next obstacle. The existing course boundary and fall
terminations remain active.

## Train once, then run a short screen

Run from the repository root in your existing Isaac Lab environment. All three
helper modes (`train`, `check`, `teleop`) now pass **`--livestream=2`** and omit
`--headless`, so the GUI is available through the **Isaac Sim WebRTC Streaming
Client**. This assumes a reachable local/private network (for example, your
existing VPN setup). Connect the client to your GPU server once Isaac Sim has
loaded; no desktop window on the server is required.

Isaac Lab deliberately sets the server's internal headless flag when streaming.
An AppLauncher message saying headless was enabled is therefore expected, not
a sign that the streamed GUI is disabled. See the
[Isaac Lab streaming options](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.app.html#environment-variables).
For direct Python commands, use `--livestream=2` instead of `--headless`.
If your working setup requires public-network mode, append `--livestream=1` to
the helper command and supply your server's `PUBLIC_IP` environment variable.

Stage new source scripts if they are not yet tracked (the trainer rejects
untracked source to protect run provenance):

```bash
git add scripts/rsl_rl/teleoperation.py scripts/rsl_rl/go2_rma.sh

PARKOUR_BASELINE="logs/rsl_rl/parkour_lab/2026-09-07_11-38-26_pivot_frequency_ablation/model_11795.pt"
bash scripts/rsl_rl/go2_rma.sh train "$PARKOUR_BASELINE"
```

This is a **v20-to-v21 warm start, not `--resume`**. All old motor weights and
encoders are retained; the three new input columns start at zero. Initial actions
are consequently preserved up to floating-point roundoff. The velocity estimator,
optimizer and update schedules start fresh. Compatible curriculum state is
restored. The source checkpoint and manifest are left untouched and their identity
is recorded in the new run. Unrelated robot/action/input changes are rejected.

The initial run is 500 updates with 4,096 environments, initial-state jitter,
nominal physics, the established pivot frequency/yaw weight, and history rollouts
every fourth update. This is a first training budget, **not a promise that 500
updates solve the task**. There is no reward sweep. Cylinder approximation stays
enabled in training, checks and operation.

Set the path to the newly printed run directory (fresh numbering normally ends
at `model_499.pt`):

```bash
PARKOUR_RMA="logs/rsl_rl/parkour_lab/REPLACE_WITH_NEW_go2_rma_velocity_RUN/model_499.pt"
bash scripts/rsl_rl/go2_rma.sh check "$PARKOUR_RMA"
```

The screen runs **21 complete episodes total**: three stop/restart, three per yaw
sign, and three for each of the four obstacle families at level 6. No video and
no 100-episode sweep. Only flat command tests record detailed telemetry. Results
go under the new checkpoint's evaluation directory. This catches gross failures;
three episodes do not establish statistical reliability.
The seven cases run in separate simulator processes, so the streaming client
may need to reconnect between cases. Streaming adds rendering overhead; it does
not increase the episode count. Run only one streaming simulator at a time.

Evaluate deterministic `history_mean`, not sampled actions, as the operating
policy. Inspect `Loss/velocity_estimation_mse` alongside the actual task results;
a lower estimator loss alone is not success. A useful initial acceptance target
is no falls/failed restarts, command settling within one second, settled yaw error
at most 0.1 rad/s at a 0.5 rad/s request, and no loss of obstacle completion. These
are engineering screening targets, not proven guarantees. If high-step still
fails, keep it classified as unsupported rather than extending a blind sweep.

## Human-operated simulation

After the screen, start on level 0:

```bash
bash scripts/rsl_rl/go2_rma.sh teleop "$PARKOUR_RMA"
```

Connect the Streaming Client and click inside its Isaac Sim viewport before
using the controls. The helper already enables streaming; no `--headless` or
`--enable_cameras` flag is needed:

- **R:** reset and arm; begins with zero motion and cleared key state.
- Hold **left Shift + Up:** forward, ramped up to 0.55 m/s.
- Hold **left Shift + Left/Right:** in-place yaw, up to ±0.5 rad/s.
- Release **Shift**, press **Down**, or press conflicting arrows: zero motion request.
- **X:** latched stop; **R** is required to rearm.
- **Escape:** end the simulation session.

Focus loss reported by Kit, a wall-clock producer stall over 0.25 s, pausing the simulation, or
an episode termination disarms motion and requires R. The command manager also
has a 0.25-s simulation-time packet watchdog. New commands reach the next action
without advancing or resampling proprioceptive history. A simulation-only guard
checks actual planar speed before allowing a pivot. These safeguards request a
stop; they are not a hardware emergency stop or a guarantee against momentum,
slipping, or a learned policy falling.
Client focus loss and network disconnects are not guaranteed to reach Kit as
focus/key-release events. The server watchdog does not measure network health.
Do not treat closing the streaming client as a stop: request zero motion and
press X before disconnecting. Verify key release/focus behavior on flat ground
with your client before attempting obstacles.

First test releasing Shift while walking, both pivot signs, focus loss, X, and
rearming after a reset. Then try a passed obstacle course, starting below the
maximum level:

```bash
bash scripts/rsl_rl/go2_rma.sh teleop "$PARKOUR_RMA" \
  --terrain_family=gap --difficulty_level=3 --teleop_speed=0.60
```

Operator reliability remains **unverified until these simulation checks pass**.
Use fresh seeds and representative operator sequences before treating a checkpoint
as the working baseline; do not infer reliability from a few scripted successes.

## Implementation verification

The focused CPU suite passed 88 tests, with 22 simulator-dependent tests skipped.
It covers both PPO actor paths, estimator gradients, migration, command ownership,
history refresh and operator interlocks. Loading the supplied `model_11795.pt`
through the migration produced identical actions in both teacher and history
modes for the tested observation batch. Lint, compilation and shell syntax checks
passed. The broader suite is not clean: separate waypoint-fixture import errors
and a gait-acquisition reward expectation failure remain outside this change.
No GPU training or live Isaac Sim keyboard session was run on this machine.
