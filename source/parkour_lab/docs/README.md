# Parkour Lab

An Isaac Lab / Unitree Go2 locomotion research package with an ROA-like teacher
and causal history student. Run `python -m parkour_lab --help` for training,
export, evaluation, playback and analysis commands.

`bound-actions CHECKPOINT COPY --report REPORT` creates an explicit learner
revision with joint-target bounds derived from native readback. Training and
inference share bounded delivery/feedback while PPO stores the original sample.
The learning payload is copied unchanged; native frozen-policy evaluation is
still needed. See [bounded action delivery](../../../README.md#bounded-action-delivery).

For the ROA numerical-failure investigation, `train --diagnose-training` records
the first extreme raw-action transition, its next PPO rollout and any terminating
exception under `numerical_diagnostic/`. It leaves learning and action delivery
unchanged. The tensor captures are diagnostic evidence, not resumable checkpoints;
an exception capture includes the current policy tensors and gradients, without
optimizer or simulator state. A run without a captured event does not prove the
original failure is resolved.

See the repository's [README](../../../README.md) for setup, package boundaries
and the [acceptance summary](../../../README.md#acceptance).
