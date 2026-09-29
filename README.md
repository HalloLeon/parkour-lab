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
| `python -m parkour_lab analyze RUN` | Print a run's recorded result; no automatic qualification |

`--config FILE` accepts explicit settings, not an old experiment manifest.
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
procedural, steps and traversal. These
are development fixtures, **not the approved mixed-world qualification bank**.
Traversal is evaluation-only. Current step fixtures use their measured three-row
range; procedural settings expose difficulty range and one, three or five rows.

Training produces `checkpoint_NNNNNN.plab` and streaming `metrics.jsonl`.
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

Evaluation/play accept either `--tape FILE` or `--command VX VY WZ --steps N`.
The latter includes a final one-second stop (all stop when N <= 50).
Successful playback records the commands actually delivered in `commands.json`;
replay preserves their sequence, not stochastic physics. Physical keyboard,
joystick and Unitree transport belong in the deployment application.

## Ownership and boundaries

| Package/module | Responsibility |
| --- | --- |
| `config.py`, `__main__.py`, `experiment.py` | Typed settings, five public commands, lifecycle and training orchestration |
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
An additional terminal noise sample is isolated from the ordinary Torch RNG
stream so it does not change reset or survivor noise draws.

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

The model uses 45-D proprioceptive frames. The optional/default-on contact
conditioning is twelve privileged teacher force labels, **not** causal foot-contact
indicators. The sensing requirements in `ACCEPTANCE.md` are qualification targets,
not a description of this observation format.

## Acceptance

[ACCEPTANCE.md](ACCEPTANCE.md) freezes v2 revision 5: stairs with realized risers
4/8/12/16 cm, ramps/hills at 10/15/20 degrees, multiscale unevenness over both,
and one causal policy satisfying traversal and flat-command thresholds. No gaps,
steep backward traversal or sim-to-real qualification is required.
These are ROA-informed project targets, not official ROA benchmark claims.

[PROJECT_STAGES.md](PROJECT_STAGES.md) records the implementation plan, evidence
and outstanding validation. Diagnostic reward, tracking, exports and local tests
cannot establish behavioral qualification or replace the independent overall
score of at least 18/20.
