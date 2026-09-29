Changelog
---------

Unreleased
~~~~~~~~~~

Fixed
^^^^^

* Added frozen 500/3000 native reward/actuator capture to the existing terrain
  audit/evaluator/recorder. Contributions are checked against native reward sums;
  four-substep efforts and terminal state share the existing trace artifact.
  Original training identity, stock physics and physical gates remain unchanged.
  Reproduction or publication failures fail closed. Consolidated duplicate
  operator documentation; no new runtime modules or learning intervention.
* Added opt-in ``--terrain-critic-context`` to the existing terrain trainer:
  operator role plus native waypoint-distance/route-phase state enter only the
  value network. A zero-initialized additive projection preserves the 312-D actor,
  all previous parameter initialization, initial actor/value/std and RNG state.
  Versioned checkpoints reject incompatible schemas; native PPO uses fresh
  next/reset context. Detached operator/course value-MSE logs do not change the
  loss. This is a controlled representational repair, not a proven cure for
  failed traversal. The matched 3000-update run uses fresh 8500 initialization and
  was GPU-unrun at implementation; actual-artifact preflight and 1007 CPU tests
  (35 skipped) passed. The subsequent partial and completed v2 GPU results are recorded below.
  No new runtime files or reward, motor, curriculum or acceptance-gate changes.
* Corrected the terrain-readiness scan guard after the supplied GPU run failed:
  production heights are normalized to [-1, 1], not bounded by the 0.5-metre
  clipping constant. Missing rays retain height +1 and validity 0. Added explicit
  unit metadata and before-shutdown traceback/stage/step diagnostics, without
  changing the motor, warm-start equality or behavioral gates. A production-scan
  regression reproduces the old failure; 239 CPU tests and actual-8500 preflight
  passed at review 45. The subsequent supplied readiness run now passes as
  recorded below; obstacle acceptance stays false.

Added
^^^^^

* Audited the completed v2 ``operator_terrain_teacher_50sjo_pa`` experiment:
  operator passes 35/40, 38/40, 24/40 and gap-L1 passes 0/10, 3/10, 8/10 at
  500/1500/3000; all other course groups remain 0/10. Raw scores, all 100 producer
  hashes, checkpoint/sidecar identities, 480,000 mean actions and 5.76 million
  stock joint targets verify. Independent review 49/75 scores 14/20 for partial
  measured learning, not tools; joint acceptance and distillation remain closed.
  Added a read-only CPU audit under ``scripts/analysis/`` with first-reset-safe
  selected cost reconstruction, strict source/artifact checks and CLI regressions.
  It does not recover full training rewards or measured torques. Recorded deeper
  pinned Extreme Parkour, Robot Parkour Learning Go2 and framework comparisons;
  closed unchanged context-only continuation and specified the remaining bounded
  objective/actuator question. Training/evaluation producers, physical gates and
  motor behavior are unchanged; no new simulator run or learning update launched.
* Audited the additional v2 model-1000 development check from
  ``operator_terrain_teacher_50sjo_pa``: operator 21/40 and all twelve L1/L3/L6
  obstacle groups 0/10. Verified all 100 producer hashes, checkpoint/Adam,
  raw scores, 160,000 mean actions and 1.92 million exact joint targets.
  Checkpoint-1350 training diagnostics retain 25% operator exposure but still
  show no L1 successes for step, hurdle or ramps. Review 48/75 remains 13/20.
  Recorded phase-specific yaw failures, waypoint-versus-completion distinctions,
  missing live handoff/student evidence and pinned upstream curriculum/budget
  comparisons. Added a read-only status command for the existing run; preserve
  its declared 3000-update budget and scheduled checks. Focused CPU verification
  ran 72 tests successfully (one skipped). No runtime changes or
  extra simulator probes; the supplied export does not establish live run state.
* Raised the cumulative independent-critic limit from 50 to 75 per user request.
  Review 47 remains 13/20; no review is consumed and the 19/20 plus joint physical
  exit requirements are unchanged. Recorded mandatory commit messages after file
  edits and concrete established-project comparisons in main-agent and critic
  work, including source versions, transferable practices and limitations.
  This workflow-only update changes neither runtime nor the next GPU command.
* Independently audited the complete ``operator_terrain_teacher_koldt2g7`` run:
  3000 updates / 230.4 million transitions; operator passes 0/40, 3/40, 0/40 at
  500/1500/3000. Only final gap-L1 has any obstacle passes (3/10); all other
  L1 and all L3/L6 groups fail 0/10. Raw scoring, 480,000 checkpoint actions,
  5.76 million exact joint targets and artifact/source hashes verify the result.
  L1 exposure exists; later stalling is not mastery. Review 47/50 scores 13/20,
  below the prior 14 because of negative learned-behavior evidence. The next
  experiment isolates critic task context while preserving all physical gates;
  operator preservation and every family's L1 traversal are co-primary outcomes.
* Verified the supplied ``operator_terrain_readiness_hqqx7dme`` GPU result:
  80,000 transitions, exact initial actor/value and 960,000 joint-target
  comparisons, valid scans, clean shutdown and one finite PPO update with both
  terrain encoders learning. The trace is pre-update and all 40 first L1 course
  attempts fail on chassis contact; this is readiness, not obstacle capability.
* Added opt-in ``operator_train.py --terrain-train`` in existing runtime files:
  3000 native PPO updates, 3200 environments and 230.4 million transitions from
  fresh zero-fusion 8500 initialization. Fixed operator lanes retain source
  behavior while course lanes progress from L0 through the production curriculum.
  Explicit body-twist and termination adapters preserve moving yaw and isolate
  role-specific rewards/retention. Training-only extreme-tilt termination and a
  neutral flat-workspace timeout have reset-safe physical-failure precedence.
  Native checkpoints preserve update/Adam, command-exposure and course-outcome
  accounting. Actual 312-input teachers at 500/1500/3000 receive unchanged
  operator and per-family L1/L3/L6 measurements; invalid evidence stops checks,
  measured behavioral failures do not. At review 46, CPU preflight and 309 focused
  tests passed while the GPU run was unrun; that historical score was 14/20.
  The subsequent failed GPU result and review 47 are recorded above. Same-policy
  held-out/student/L6/handoff/release evidence still blocks exit.
* The existing operator trainer now provides opt-in ``--terrain-readiness``:
  exact stock actor/critic/noise initialization with zero-initialized terrain
  connections, production scan preprocessing, fixed paired L0/L1 coverage and
  one fresh-Adam PPO update. Raw motor/scan capture, source/artifact binding,
  finite encoder-gradient checks and fail-closed worker shutdown distinguish
  engineering readiness from behavior acceptance. At implementation, 236 CPU
  tests and actual-8500 CPU preflight passed while the simulator check was unrun.
  The subsequent GPU result is recorded above. This adds no new runtime files and claims no convergence,
  progressive curriculum, operator retention, student or live handoff. Critic
  review 44/50 then remained 14/20; the mandatory joint L6 exit gate remains closed.
* The bounded frozen-8500 L0/L1 batch is now raw-replay verified: L0 completes
  10/10 with supported terminal dwell; all four L1 families fail 0/10. Thirty-eight
  failures involve lower-head contact; two gap trials finish tipped on their sides
  and already fail as incomplete. Recorded ray heights agree with L1 geometry and
  motor parity remains intact. The documentation closes further frozen level
  sweeps and specifies the stock-to-terrain training/interface migration still
  required. Runtime, gates and historical results are unchanged; no new training
  path is claimed. This is a main-agent evidence update, not a new independent
  review; the last completed critic review remains 44/50, scored 14/20, exit closed.
* Raw replay of frozen 8500 verifies mesh-flat 100/100 and failed L6 transfer
  on all four families (0/10 each), with intact stock motor parity and matching
  course-source identities. Failures occur at obstacle encounters, before
  terminal completion. The updated protocol keeps sparse waypoints and unchanged
  acceptance gates, bounds the next check to one L0 control then four L1 probes,
  and records the incompatible stock/RMA command paths and untested operator
  handoff. No runtime changes, new files or training updates were made. Critic
  review 44/50 remains 14/20, below the 19/20 threshold with joint gates unmet.
* Frozen 8500 confirmation raw-replays to 100/100 on seed 44 and 97/100 on
  seed 45: one lateral-tracking and two post-lateral stop failures prevent
  promotion. The existing operator benchmark now supports opt-in collidable
  mesh-flat and production course evaluation, retaining the stock motor and
  requiring a verified mesh control before all four level-6 families. Ordered
  first attempts, raw motor/support evidence and stable terminal dwell are
  checked; legacy defaults remain unchanged. No new runtime files or training
  updates were added. CPU verification passes 941 tests with 34 skipped, but
  simulator traversal was unrun at review 43. Critic review 43/50 scores 14/20; the
  required 19/20 and joint behavioral exit gates remain unmet.
* The independent robotics critic now uses a 20-point scale with a minimum
  exit score of 19, effective from review 43. The cumulative 50-review limit
  and mandatory joint operator/student/level-6 obstacle evidence are unchanged.
  Completed scores through review 42 retain their original 16-point scale and
  14-point target; changing the protocol does not consume a review or open exit.
* The completed 3000-update stock-operator run reaches a raw-replayed 100/100
  development pass at 8500, establishing the baseline for seed-44/45 confirmation
  without further flat training or new runtime modules. Critic review 42/50
  scores 13/16 with target unmet. Its exit gate now explicitly requires actual
  level-6 tilted-ramp/high-step/gap/hurdle traversal and retained operator control
  on the same evaluated policy/interface. At that review the stock-to-course
  adapter was unimplemented and causal-history behavior unvalidated.
* V3 training horizon and predeclared saved-checkpoint offsets are configurable
  in the existing trainer/preflight, replacing the universal 200-update limit
  without adding runtime modules. Budget, optimizer accounting and parent/worker
  metadata stay consistent; legacy v2 checks and all physical gates are preserved.
  The short v3 run repairs reversal stops but transfers failures, including a
  fall at 5700. Independent review 41/50 supports one longer unchanged-objective
  experiment from preserved 5500/Adam8000, not promotion of 5700; 12/16 remains
  below target. Post-worker screening is not compute early stopping.
* Opt-in ``operator_reversal_sequences_v3`` tests randomized reversal/hold/restart
  exposure on 25% of physical-reset episodes, retaining v2 sampling elsewhere.
  Evidence-bound preflight requires the completed 5500/Adam8000 learning state,
  original 5100, both unchanged soft retention losses and the raw negative
  two-arm recovery experiment. One 200-update worker checks only 5600/5700 after
  exit; entered/completed/censored/active chains and actual frames are recorded
  by yaw direction. No policy phase input, inference assistance, reward, physics
  or gate change is introduced. The supplied recovery arms remain 98/100 with
  the same two post-reversal stop failures, so stronger anchoring is unsupported.
  Downloaded probe replay now accepts relocated reference paths only with exact
  hashes and consistently recorded identities. The v3 provenance correctly
  labels the zero-command loss as retained, not newly added. Independent review
  40/50 kept the outcome estimate at 12/16 pending the bounded GPU trial and
  student/obstacle evidence; completed schedules alone are not behavior passes.
* A separate two-arm ``operator_stop_probe.py`` diagnoses whether original 5100
  can stop states reached by frozen 5500, selecting reference actions only at
  post-motion stop onset or after one second of learner braking, then returning
  to learner-controlled restart. The supplied shadow capture exactly reproduces
  the archived 98/100 result but does not establish recovery. Identity-bound raw
  replay and exact pre-intervention matching cover every trial, with no
  initial-stand substitution, action blending, learning or physical-gate changes. Both
  arms retain complete stop/restart outcomes; mixed-controller success is never
  policy acceptance. Coordination/publication failures return ERROR/exit 2 and
  incomplete progress cannot be marked complete. Independent robotics review
  39/50 verified remediation and all 16 probe tests; 910 CPU tests pass with 34
  skipped. That review's outcome estimate remained 12/16 pending the two-arm GPU results,
  standalone learner acceptance, trained student and obstacle evidence.
* Opt-in ``--diagnostic-reference`` stock-operator screening captures exact
  delivered observations/actions and a frozen reference's shadow outputs,
  plus pre-reset joint, target, actuator and contact snapshots. One-second
  zero-command summaries preserve late stop drift and never bridge physical
  resets. Reference loading preserves CPU RNG; identities and artifacts are
  checked without changing commands, actions, scoring or training. The paired
  zero-retention endpoint reaches 98/100 without physical terminations in its
  development screen, but two post-reversal stops still fail. Diagnostics are
  not a behavioral pass, counterfactual recovery test or obstacle integration.
  Invalid learner/reference paths, configurations and tensors now fail CPU
  preflight with exit 2 before output creation or simulator launch, preserving
  exit 1 for measured behavioral failure. Independent robotics review 38/50
  verified this repair; the full-objective estimate remains 12/16 pending new
  GPU, student and obstacle evidence.
* Opt-in ``zero_command_anchor_v1`` adds a separately normalized zero-twist
  mean-action retention term to the bounded original-reference resume. The
  original actor's standing/stopping screen is raw-replayed and identity-bound
  before training; moving loss, pure-pivot freedom, actor/Adam handoff, rewards
  and physical acceptance gates are preserved. Independent exposure, MSE and
  weighted losses make the intervention observable. The paired 200-update branch
  starts from the same 5300 state as the measured control, whose new physical
  regressions prevent promotion despite a 95/100 endpoint score. This soft
  training-only reference is not a recovery controller or proven safety repair.
* Evidence-bound retention continuation restores the learner's actor, critic,
  noise and complete Adam moments/counters/options while retaining the exact
  original frozen reference. One additional 200-update run screens only its
  predeclared +100/+200 checkpoints after the worker exits. Strict source,
  metadata, optimizer and runtime checks reject silent resets or re-anchoring;
  simulator/command/RNG restarts are explicitly distinguished from uninterrupted
  simulation. The objective and physical acceptance gates remain unchanged.
* Opt-in ``moving_anchor_v1`` operator refinement from replay-verified source
  evidence: a frozen source mean-action loss on learner-visited moving commands,
  with unchanged rewards, motor, physics and physical acceptance gates. The
  bounded protocol uses one uninterrupted 200-update worker and checks +100/+200
  only after it exits, retaining original source/optimizer semantics. Behavioral
  failures do not suppress the already-trained second candidate; a full pass or
  execution error stops screening. Oracle audits, provenance and fail-closed
  error publication remain mandatory. This is a soft retention hypothesis,
  not a qualified student, controller merger or demonstrated physical repair.
* Phase-local operator diagnostics separate acquisition, subsequent tracking,
  stationary excursion and heading failures without weakening any trajectory
  acceptance gate. A bounded saved-checkpoint screen replays the archived
  endpoint and tests a predeclared small set, with checkpoint/config/trace
  identity checks, matching runtime versions and explicit FAIL/ERROR handling.
* A versioned stock-operator teacher/history interface with a frozen motor,
  causal per-environment reset handling, explicit velocity supervision and an
  oracle-parity audit path. Saved-checkpoint screens shadow-check delivered
  observations, previous actions and resets while the original actor remains
  in control. The estimator remains untrained and is not used for simulator
  control; this is not full RMA or obstacle acceptance.
* An opt-in ``stationary_twist_v1`` refinement after the matched low-entropy
  branch localized its remaining failures to pivot excursion and post-arc
  heading drift. Existing tracking terms blend broad acquisition and fine
  command-gated stationary precision, with unchanged weights/peaks, moving
  rewards and pivot yaw rewards. No pose anchors or inference feedback are
  introduced. Known-function reconstruction, exact parameter validation,
  source-preserving continuation and reward-code hashes retain provenance.
  Physical improvement remains to be measured with unchanged acceptance gates.
* An opt-in ``low_entropy_v1`` operator control profile retaining stock rewards
  at the same low entropy as the yaw-precision experiment. Profile recognition
  now validates the complete reward-width/entropy pair, and manifests separate
  reward changes from entropy changes. Source tensors, full environment/PPO
  contracts and physical acceptance gates remain protected. The documented
  matched continuation starts from preserved model 3599; the precision model's
  improved yaw tracking does not clear its pivot drift and standing regressions.
* A versioned ``yaw_precision_v1`` operator refinement after the longer v2 run
  improved the unchanged screen to 88/100 and reverse to 10/10. It narrows the
  yaw-tracking kernel and lowers entropy pressure while restoring source
  actor/critic/noise tensors exactly. Source-preserving defaults, explicit
  objective provenance, full environment-contract checks and unchanged physical
  acceptance gates prevent silent reward or evaluation drift. Robot improvement
  from this new objective is not yet established.
* An opt-in ``operator_transitions_v2`` command curriculum targeting reverse and
  live braking while retaining original long-hold coverage. Physical resets
  never inherit a prior command's transition, and v1 remains reproducible.
  Refinement CLI, checkpoint manifests and executed exposure identify the selected
  version; stock rewards, policy/physics contracts and acceptance gates are unchanged.
* A bounded stock-Go2 operator command refinement with exact actor/critic/noise
  checkpoint handoff, explicit stand/pivot/translation modes and long holds,
  executed command exposure accounting, and automatic unchanged operator screening.
  Runs headless without streaming or an external Isaac Lab training script;
  preserves network, observations, physics, actions, resets and rewards.
  Fresh fixed-rate Adam and update numbering are recorded explicitly.
* A finite, headless stock-Go2 operator reference benchmark with ten explicit
  command profiles, pre-reset physical-state capture, sustained tracking and
  braking gates, strict checkpoint/configuration validation, and CPU tests.
  This does not implement or certify a combined operator/obstacle RMA policy.
* Adaptive, outcome-based terrain difficulty progression.
* Declarative multi-waypoint course definitions with obstacle families,
  structures, support regions, and explicit difficulty metadata.
* Vectorized active-waypoint progression with per-environment route cursor,
  monotonic safe progress, and final-only course-completion semantics.
* Physically segmented gap courses up to 0.50 m whose missing base-ground
  intervals are visible to collision and privileged terrain rays.
* Distinct high-step and hurdle course families with parameterized box
  dimensions and positions, elevated-platform versus continuous-ground support
  semantics, and deterministic geometry tests.
* A redirected two-ramp course with configurable cross-slope, yaw, spacing,
  landing geometry, and ordered oracle-travel-direction transitions.
* A vectorized, contact-gated foot-edge penalty derived from exact metric 3D
  support boundaries, including inclined and rotated ramp surfaces.
* Fixed-level evaluation with metrics and reproducible video recording.
* Simulator-free tests for the discrete difficulty mapping.
* Additive deployable-core, privileged-terrain, and critic-only observation
  groups with an explicit missing-ray validity mask.
* A modular Phase-1 RSL-RL teacher whose checkpoint keeps the privileged scan
  encoder and directly transferable motor actor as separately identifiable
  submodules.
* Counterfactual terrain diagnostics for measuring policy sensitivity.
* Explicit ``--domain_randomization_stage`` selection for teacher training.
* A shared flat bootstrap row ahead of every obstacle family's six difficulty
  rows.
* Timestep-independent physical-milestone and course-completion reward events.
* A progressive tilted-ramp acquisition ladder that introduces a second slab,
  gap, steering offset, and strong opposing banks in separate rows.
* A compact direction/speed/yaw-rate command contract with exact-zero stops,
  flat-only in-place pivots, and fail-closed invalidation.
* A finite-route training envelope with a moving-only soft excess cost, named
  hard off-route failures, and episode-balanced fixed-evaluation telemetry.
* Separate ``0.20-0.70 m/s`` flat and ``0.45-0.70 m/s`` obstacle command
  distributions while retaining flat-only exact-zero stop windows.
* Mirrored constant-heading and gentle-turn flat variants that train non-forward
  local headings without adding a joystick simulator or planner.
* Fixed geometry-variant evaluation and unclamped signed oracle-residual tail
  diagnostics for reference/oracle compatibility checks.
* Mean/p95 movement-direction tracking plus stop settling, maximum two-second
  drift excursion, and restart diagnostics for teacher qualification.
* Signed pivot-rate acquisition with zero-translation tracking, bounded
  ``0.75-4.0 s`` flat windows, deterministic fixed-rate pulse/restart trials,
  and dedicated error, drift, exposure, and wrong-way diagnostics.

Changed
^^^^^^^

* Aligned physical terrain rows one-to-one with logical curriculum levels.
* Migrated the simulated robot and policy interface from Unitree A1 to Unitree
  Go2, including base-body selectors and dynamics labels.
* Raised and centralized the default minimum base clearance at 0.27 m for the
  Go2, discouraging an unnecessarily crouched energy-minimizing gait.
* Renamed the Gym task IDs from the template prefix to ``Parkour-Lab-v0`` and
  ``Parkour-Lab-Play-v0``.
* Split reward implementations into waypoint, safety, limb, and root-motion
  modules behind the existing public rewards API.
* Changed waypoint-directed velocity shaping to clamp the signed normalized
  world-frame projection to ``[-1, 1]`` and suppress velocity/heading shaping
  on the exact retarget step.
* Added exact left-right transition augmentation to teacher PPO updates and
  paired high-step and tilted-ramp terrain variants by handedness, without
  imposing a mirror loss or periodic gait target.
* Replaced the teacher's generic flattened actor MLP with the shared modular
  motor architecture while retaining Isaac Lab 2.3's RSL-RL critic and PPO.
* Consolidated privileged heights and validity into one 264-value observation
  term, preserving their flattened order while eliminating duplicate ray-scan
  preprocessing.
* Kept the complete teacher manifest and hash as exact training provenance while
  excluding command-source and training-provenance metadata from hard inference
  compatibility. Playback warns on changed terrain domains while retaining
  strict observation, scanner, network, action, waypoint, and timing checks.
* Renamed the teacher's active-waypoint input to ``oracle_travel_direction`` and
  removed the unused requested-direction observation group; requested travel
  intent remains owned by the command manager until a student consumes it.
* Restored adaptive curriculum memory on PPO resume only when checkpoint and
  runtime terrain/curriculum provenance match; geometry-changed fine-tuning
  restarts from the configured bootstrap rows.
* Reduced playback arguments to supported evaluation, diagnostics, export, and
  checkpoint-selection options.
* Split playback orchestration into focused checkpoint, environment, rollout,
  diagnostics, and reporting helpers.
* Versioned the teacher terrain contract to include complete course geometry.
* Changed curriculum promotion to require three successes in the last five
  frontier attempts, and demotion to require two stalled failures in the last
  three eligible attempts below 60% normalized waypoint progress.
* Derived demotion progress from the route cursor over all intermediate
  waypoints, independently of milestone reward annotations.
* Added one protected post-promotion attempt, 10% shared-flat anchor replay,
  and 15% immediate-predecessor replay without letting replay outcomes alter
  the mastery frontier.
* Kept 15% shared-flat replay at the curriculum ceiling and distributed the
  remaining 10% replay budget over acquired obstacle rows, preventing ordinary
  locomotion exposure from shrinking as the frontier is mastered.
* Replaced the non-negative flat speed kernel with signed waypoint progress on
  every terrain, preventing zero-net fore-aft oscillation while retaining the
  phase-local obstacle speed ceiling.
* Consolidated navigation state under one typed runtime, grouped all mutable
  curriculum buffers under ``ParkourCurriculumState``, removed duplicated
  family/command/support tensors, and conditioned compact curriculum health
  logging on frontier attempts.
* Simplified route-control transitions to one current-root proximity radius.
* Restricted intermediate shaping to explicitly supported physical milestones,
  reduced its fixed per-course budget to ``+2``, and removed the redundant
  curriculum-promotion reward.
* Extended high-step and hurdle routes with supported landing milestones and
  later exits, eliminating their first-obstacle curriculum demotion dead zone.
* Removed the redundant active flat-overspeed, no-feet-contact, hip-deviation,
  touchdown air-time, and cross-foot timing terms so the parkour task does not
  prescribe a flat-terrain gait.
* Added mild world-up orientation and all-joint default-pose penalties to make
  persistent static leaning and folded-leg reward exploits costly without
  prescribing a contact sequence.
* Halved roll/pitch angular-velocity regularization to ``-0.025`` so deliberate
  obstacle motion remains affordable.
* Removed inactive experimental foot-motion, root-chatter, and obsolete waypoint
  rewards while retaining and activating the established joint-deviation
  regularizer.
* Replaced flat/obstacle-specific limb-reward wrappers with terrain-independent
  leg-contact, edge, slide, and stumble penalties.
* Reduced the exact course-completion bonus from ``+10`` to ``+4`` so discounted
  early completion no longer overwhelms dense target-speed tracking; retained
  the separate ``+2`` physical-milestone budget.
* Required physical milestones and final completion to establish recent foot
  contact on their named support polygon; final completion additionally
  requires stable vertical motion and tilt, safe clearance, and no chassis crash.
* Prevented off-corridor motion and chassis-contact steps from increasing safe
  route progress, and gave crashes precedence over simultaneous success.
* Normalized the low-clearance penalty, strengthened heading guidance, applied
  fixed scales to raw physical-unit observations, and exposed normalized route
  cursor/progress phase to the privileged critic.
* Made the terminating chassis-contact penalty timestep-independent with an exact
  ``-10`` value.
* Gave every obstacle-family cohort the same straight 0.55 m/s bootstrap route;
  held family speed and minimum clearance fixed across obstacle rows so geometry
  is the primary difficulty variable.
* Widened each hurdle across the complete 4.0 m tile, removing the in-tile path
  around either exposed end.
* Increased the default PPO budget to 1,000 iterations and 48 control steps per
  environment, used ``gamma=0.995``, and retained checkpoints every 100
  iterations.
* Bounded teacher action noise and retained a ``0.01`` entropy coefficient so
  exploration persists through initial locomotion acquisition.
* Stored adaptive terrain frontiers, rolling evidence, and demotion grace in PPO
  checkpoints so same-terrain resumed training retains curriculum progress and
  rejects checkpoints missing that state.
* Versioned the teacher interface to include milestone annotations and the
  split intermediate/final waypoint-transition semantics.
* Versioned the teacher interface to 13 for fixed observation scaling,
  named-support/contact-gated physical waypoints, stable crash-free completion,
  and the revised curriculum matrix.
* Versioned the teacher interface to 14 and froze the robot model and source
  asset so shape-compatible A1 checkpoints cannot be loaded as Go2 policies.
* Versioned the teacher interface to 15, treating Go2 base and head impacts as
  fatal chassis contacts while extending recoverable-contact penalties to hips.
* Removed final-waypoint dwell so all support, stability, clearance, and
  crash-free completion gates finish the course immediately once satisfied,
  and removed the associated timer buffer.
* Derived milestone reward events from the existing cursor-change state instead
  of maintaining a second per-step event buffer.
* Standardized route, marker, observation, and reward naming on active and
  final waypoints.
* Updated project metadata and documentation for Parkour Lab.

Fixed
^^^^^

* Distinguished a missed base-height ray from genuine zero clearance so
  intentional gap flight is not charged the maximum low-clearance penalty,
  while final completion still requires a valid supporting-surface hit.
* Detected near-horizontal foot stumbles from total contact force instead of
  incorrectly requiring a large vertical-force component.
* Reported foot participation per completed episode and in fixed evaluation so
  mirrored tripod modes cannot disappear through population averaging.
* Normalized Hydra's dictionary representation of nested curriculum levels
  before terrain generation, validation, resets, and evaluation.
* Bounded PPO actions and exploration noise, rejected non-finite optimization
  steps, and capped the adaptive learning rate to prevent rare action outliers
  from corrupting the critic and adaptation history.
* Kept resumed training on its configured curriculum bootstrap instead of
  silently replaying every terrain row.
* Aligned the hardest tilted-ramp approach with its first ramp, removing an
  abrupt heading retarget without weakening the obstacle geometry.

0.1.0 (2026-06-16)
~~~~~~~~~~~~~~~~~~

Added
^^^^^

* Created the initial Parkour Lab project based on Isaac Lab.
