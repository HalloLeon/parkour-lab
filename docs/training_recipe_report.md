# Inspect the saved training recipe

Run the simulator-free, read-only inspector on the machine where the run and
checkpoint exist (including an HPC `/u/...` path). It requires only Python and
PyYAML, not Isaac or torch.

```bash
python scripts/rsl_rl/training_recipe_report.py /path/to/run/model_22099.pt --verify
python scripts/rsl_rl/training_recipe_report.py /path/to/run/model_22099.pt --metrics /path/to/metrics.json
```

Alternatively pass the run directory and `--checkpoint model_22099.pt`. A run
directory alone reports saved settings but leaves the checkpoint UNKNOWN; the
inspector never silently picks the latest checkpoint. Add `--json` for a structured
report on stdout. No files are written, and no remote paths are fetched.

The report reads actual `params/env.yaml` and `params/agent.yaml`, including the
pivot reward, command probability/durations, resolved reset ranges, DR stage,
history rollout interval, configured learning rate/cap and entropy schedule.
The optional target-overflow cost's weight, normalization and action-term name
are also reported; missing values in older runs stay UNKNOWN, not today's zero default.
Python-specific YAML tags are inert: BaseLoader extracts strings, sequences and
mappings without executing constructors. Checkpoints are streamed into SHA-256,
never deserialized. Missing fields remain UNKNOWN, including old files predating
the yaw-objective setting; current code defaults are not historical evidence.

`--metrics` requires exact SHA-256 matches for the checkpoint and both saved
configuration files. Original remote paths need not equal downloaded local paths.
It deliberately ignores the metrics file's **evaluation** reward configuration.
`--verify` without metrics requires those three files to be readable and metadata
to parse; this is not a hash comparison against an independent reference. Missing,
malformed or mismatched verification inputs exit with status 2. Malformed existing
metadata also fails in report-only mode. No expected repair preset is silently
assumed, and hash matches are not a performance or recipe-correctness verdict.

Optional `git/provenance.json`, `params/resume.json`, `params/warm_start.json` and
`params/curriculum_restart.json` are hashed and reported when present. Missing
sidecars remain MISSING, not a failure of the three-file identity check. Ordinary
resume currently need not create a dedicated sidecar; `agent.resume` and
`load_checkpoint` establish saved intent, not independently verified execution.
The inspector does not load checkpoint-internal metadata or follow source paths.

Saved settings cannot prove the number of completed updates, actual command
exposure, restored optimizer state, effective adaptive learning rate, or executed
entropy schedule. Those require corresponding runtime evidence. A reset profile
label is only inferred when the complete resolved ranges match canonical/jitter;
the actual ranges are always shown. DR `off` does not imply canonical resets.
