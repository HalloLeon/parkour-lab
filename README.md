# Parkour Lab

An Isaac Lab / Unitree Go2 locomotion research framework with an ROA-like
privileged teacher and causal history student. A configuration-driven workflow
covers training, export and command playback. **Successful training or playback
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
| `python -m parkour_lab train` | Fresh ROA training; optional current learning snapshot continuation |
| `python -m parkour_lab export CHECKPOINT ACTOR` | Export causal motor and history estimator, without the teacher/critic |
| `python -m parkour_lab evaluate ACTOR` | Frozen headless command playback and diagnostic tracking |
| `python -m parkour_lab play ACTOR` | The same playback with a visible simulator, one environment by default |
| `python -m parkour_lab analyze RUN` | Print a run's recorded result; no automatic qualification |

`--config FILE` accepts explicit settings, not an old experiment manifest.
Unknown fields are errors. CLI overrides include `--device`, `--num-envs`,
`--seed`, and training `--updates`. For example:

```json
{
  "task": {"terrain": "procedural", "num_envs": 160, "seed": 42, "device": "cuda:0"},
  "roa": {"history_interval": 5, "regularization_coef": 0.1},
  "updates": 1000,
  "save_interval": 100
}
```

Defaults and validation live in `config.py`; every run records its fully resolved
configuration. Terrain choices are flat, procedural, steps and traversal. These
are development fixtures, **not the approved mixed-world qualification bank**.
Traversal is evaluation-only. Current step fixtures use their measured three-row
range; procedural settings expose difficulty range and one, three or five rows.

Training produces `checkpoint_NNNNNN.pt` and streaming `metrics.jsonl`.
`train --checkpoint FILE --updates N` restores the current policy and optimizers
and performs N additional updates. It starts a fresh simulator, history and seeded
RNG; it is not exact interrupted-rollout resumption. Method settings must match.
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
| `methods/roa/` | ROA observation adapter, teacher/student model, PPO and history-fitting schedule/state, causal controller |
| `methods/base.py`, `methods/models.py` | Small method contract and shared neural components |
| `control/` | Portable command tapes, sensor/action contracts and controller session semantics |
| `runtime/` | Native sensor/motor binding and cleanup |
| `artifacts.py`, `provenance.py` | Current tensor artifacts and source/run receipts |
| `evaluation/` | Frozen command playback and diagnostic metrics |
| `tests/` | Component-organized regression, contract, geometry and CPU integration checks; see [tests/README.md](tests/README.md) |

There is no legacy task registration, gap curriculum, waypoint navigation, old
teacher pipeline, experiment-specific executable or compatibility alias.
Package imports do not launch Isaac Lab. Inference does not import PPO.

The host supports ROA only; a method interface alone does not demonstrate
interchangeability with other learners. ROA owns its learning schedule and
optimizer state. Pre-reset final observations and termination/truncation semantics
must be preserved by any learner integration, rather than assuming every learner
uses PPO. There is no placeholder plugin or unused external dependency.

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

[ACCEPTANCE.md](ACCEPTANCE.md) freezes v2 revision 4: stairs with realized risers
4/8/12/16 cm, ramps/hills at 10/15/20 degrees, multiscale unevenness over both,
and one causal policy satisfying traversal and flat-command thresholds. No gaps,
steep backward traversal or sim-to-real qualification is required.
These are ROA-informed project targets, not official ROA benchmark claims.

[PROJECT_STAGES.md](PROJECT_STAGES.md) records the implementation plan, evidence
and outstanding validation. Diagnostic reward, tracking, exports and local tests
cannot establish behavioral qualification or replace the independent overall
score of at least 18/20.
