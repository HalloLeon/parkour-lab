# Tests

The suite mirrors the production packages. Modules use `test_<component>.py`;
shared fixtures live in `support/`, never in another test module.

| Location | Protected behavior |
| --- | --- |
| `benchmark/` | Public task API, actor/GT separation, spaces, accounting and continuation |
| `control/` | Command tapes, causal frames, loaded policies, reset masks and motor contracts |
| `environments/` | Owned outgoing finals, RNG isolation, reward formulas, dynamics and scans |
| `environments/commands/` | Episode/operator sampling, native command resets and curriculum eligibility |
| `environments/terrains/` | Physical family levels, fixed geometry and curriculum |
| `evaluation/` | Fixed cases, analytic tracking/stopping/failure scores, frozen playback and offline evidence validation |
| `runtime/` | Native sensing, bounded action delivery, accounting and ordered cleanup |
| Root `test_*.py` | Offline CLI, provenance and installed-package dependency boundaries |

Keep tests for supported behavior, numerical contracts and meaningful failure
modes. Do not preserve removed private names merely for historical compatibility.
Reject inconsistent evidence and never overwrite previous captures.

## Local execution

Use Python 3.10 or 3.11 with the package dependencies and SciPy for native
heightfield-source tests. From the repository root:

```bash
python -m pip install -e source/parkour_lab
python -m pip install scipy
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -t . -v
```

Run a component independently, for example:

```bash
python -m unittest discover -s tests/control -t . -v
python -m unittest discover -s tests/evaluation -t . -v
```

Do not add `tests/` to `PYTHONPATH`. See the
[package README](../source/parkour_lab/docs/README.md#verification) for formatting,
distribution and bounded server commands. Git is needed for provenance tests.
No training library is needed to exercise the common task or offline scorer.

## Evidence limits

CPU tests use real tensor/mesh calculations and small native API doubles.
Selected checks read Isaac Lab 2.3.2 source from a sibling `IsaacLab2.3.2` or
`IsaacLab` checkout; they skip explicitly when it is absent. A skip is not
compatibility evidence. The CPU suite does not launch Isaac Sim or PhysX.

The public-task tests cover the fixed 352-value training packet, physical frames,
named motor mapping, scan validity, reset masking, owned buffers and unchanged
RNG state. `scripts/check_public_task.py` separately verifies native readbacks,
role-specific observations, outgoing rows and a fresh task-progress restart.
That checkpoint token is a fixture, not a restored learner.

Geometry checks call native terrain-generator source. Curriculum checks cover
translation eligibility, turning/standing holds, stable families, level boundaries
and progress restoration. `scripts/check_terrain.py` separately records real
colliders, rays, nonzero contacts and controlled native curriculum/reset cases.
These establish their stated plumbing and sampled geometry, not learnability.

Analytic traces distinguish correct/wrong turns, unwanted translation, stationary
arc spinning, residual stopping motion, early endings and missing phases.
Frozen playback checks preserve applied-command/outgoing-state alignment and
first-attempt accounting. `scripts/check_evaluation.py` separately loads a tiny
caller-format policy, captures native cases and invokes the offline CLI in a
fresh process. Its near-default policy is an engineering fixture, not a trained
baseline or proof of successful terrain locomotion.

Native case-context checks protect fresh-stage creation, ray-mesh cache isolation
and refusal to replace another active environment. Source topology and mocked
contact checks do not replace actual cooked geometry, physical support or robot
behavior evidence.
