# Tests

The suite mirrors the production packages. Test modules use `test_<component>.py`;
classes name the component or contract, and methods describe observable behavior.
Shared fixtures live in `support/`, never in another test module.

| Location | What the checks protect |
| --- | --- |
| `control/` | Command leases and tapes, frame ordering, sensor units and motor contracts |
| `environments/` | Command sampling/reset behavior, owned terminal observations and RNG isolation, reward configuration/formulas and scan preprocessing |
| `environments/terrains/` | Procedural and connected terrain geometry, traversal fixtures and roughness contracts |
| `evaluation/` | Flat profiles and terrain crossing/tracking/finish rules, first-attempt failure retention, and read-only archive-integrity checks |
| `methods/` | Real ROA updates, causal routing, timeout semantics, optimizer state, history, export parity and installed backend discovery |
| `runtime/` | Native sensor/action binding, signed reward accounting and pre-reset diagnostics, external command ownership and cleanup |
| Root `test_*.py` | CLI/config validation, artifacts, provenance, package boundaries and the train/export/evaluate workflow |

Keep a test when it guards a supported behavior, numerical contract or meaningful
failure mode. Prefer parameterized cases to repeated checks. Do not assert that
removed private functions remain absent or preserve historical implementation
names. Unsupported artifact formats must still be rejected, and existing evidence
must never be overwritten. Those are data-safety contracts, not legacy support.

## Local execution

Use a CPU Python environment with NumPy, PyTorch, TensorDict, RSL-RL and trimesh.
Install these through `python -m pip install -e 'source/parkour_lab[roa]'` in
Python 3.10 or 3.11. Tooling, package-build and bounded server smoke commands are
in the [package README](../source/parkour_lab/docs/README.md#verification).
The project uses RSL-RL 3.1.2; learner tests execute real PPO and history-fitting
updates, not mock optimizers. Git is needed for provenance tests. From the repository root:

```bash
export PYTHONPATH="$PWD/source/parkour_lab${PYTHONPATH:+:$PYTHONPATH}"
python -m unittest discover -s tests -t . -v
```

Run a component independently, for example:

```bash
python -m unittest discover -s tests/control -t . -v
python -m unittest tests.methods.roa.test_training -v
```

Do not add `tests/` to `PYTHONPATH`. Imports use the `tests` package and its shared
fixtures. These commands are for local development; server runs use the package
CLI and do not require test files.

## Evidence limits

`test_action_boundary.py` exercises actual PPO sample/likelihood storage alongside
bounded native delivery, previous-action/reset feedback, the action-rate formula,
frozen inference parity and explicit artifact conversion. `test_gradients.py`
checks ordinary clipping agreement, large finite norms and true nonfinite
rejection. These portable tests do not depend on downloaded run archives.

Most checks use real CPU tensor/mesh code with small native API doubles. Selected
adapter tests also load Isaac Lab 2.3.2 source from a sibling `IsaacLab2.3.2` or
`IsaacLab` checkout. They explicitly skip when that checkout is absent; a skipped
check is not evidence of compatibility. No test launches Isaac Sim or PhysX.

Timing checks cover 200/400 Hz configuration with unchanged 50 Hz actions, the
SDK's four/eight-substep sampling/reset order, exact cross-rate rejection, and
explicit actor-export weight preservation. The actual SDK lazy-ray scheduling
regression reproduces missed updates and checks fresh cached control reads over
30 seconds at both rates, including partial reset. These do not establish physical
equivalence, flat retention or sim-to-real transfer at the new rate.

Connected-world tests measure source and local float32 topology, feature
dimensions, common roughness and prospective command-tape margins across all
terrain tiers/directions and shape boundaries. Ordinary geometry checks allow
small angle rounding; strict cap/RMS and no-redraw behavior remain covered.
These checks do not qualify native meshes or robot traversal. Mocked native calls
and SDK source checks do not substitute for native execution. Diagnostic-only
geometry/contact runners are not part of the package or test suite; scoped native
checks use direct server commands when needed.

Single-world playback checks additionally cover importer wiring, supported rotated
resets, shared dynamics/noise, first-ending tape preservation, and aligned outgoing
forces/rays. These protect the operational `connected` terrain path, not a
diagnostic runner or a claim of physical support on the server.

Terrain scoring tests use analytic traces for rotated ascent/descent/hill crossings,
bypass and side-edge failures, first-entry tracking, exact timing windows, stable
finish, falls, stalls and map bounds. Offline loader tests reject incomplete or
inconsistent archives and invalid delivery/cleanup receipts. These checks do not
certify cooked contacts or replace a preassigned native qualification bank.

Terrain assignment checks cover development-only IDs, full-SHA/PCG64 mappings,
collision rejection, per-stratum dimension/tread/yaw coverage, deterministic
serialization and unchanged existing randomization. Cheap nominal-margin checks
do not construct all assigned meshes or establish native feasibility. The native
connected evaluator can now consume one explicitly selected development group and
global index. Binding checks cover IDs 0/49/50/99 across all 20 groups, native API
application of seeded mass/friction/gains/reset/noise, the fixed 1500-control tape,
first-ending preservation and rejection of altered saved assignments/physical
receipts/draws even when file hashes are recomputed. Existing flat/train/diagnostic
outputs retain exact parity. These are software/API-double checks; native assigned
cases and bank qualification remain unverified.

The two traversal-layout classes intentionally apply the same adapter checks to
different layouts. The end-to-end CPU test verifies software data flow, artifact
round trips and frozen playback—not physical locomotion, contact generation or
robot qualification. Those require separate native execution and behavioral evidence.

The installed entry-point fixture aliases ROA, and the artifact payload probe only
serializes data. They verify loader/configuration/persistence contracts, not a second
learner or cross-framework interoperability. No alternative learner dependency is
needed to run the suite.
