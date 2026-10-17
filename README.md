# Parkour Lab

A learner-independent Unitree Go2 task and evaluation package for Isaac Lab.
It provides blind 49D actor observations, a separate shared training-information
packet, five terrain families and checkpoint-linked interaction accounting.
External projects own their learners, inference memory and model files.

See the [package README](source/parkour_lab/docs/README.md) for installation,
the public API, physical terrain levels, frozen evaluation and executable checks.

- [Public task API](source/parkour_lab/docs/README.md#public-task-api): Flat,
  Uneven, Slopes, Stairs, Hills and their common training mixture.
- [Actor and training information](source/parkour_lab/docs/README.md#actor-and-training-information):
  causal deployment inputs and permitted training-only physical fields.
- [Frozen evaluation](source/parkour_lab/docs/README.md#frozen-evaluation):
  caller-loaded policies, fixed commands and offline scoring.
- [Verification](source/parkour_lab/docs/README.md#verification): CPU tests,
  distribution checks and bounded native capture commands.
- [Test coverage](tests/README.md): protected contracts and evidence limits.

Training uses 200 Hz physics and 50 Hz actions. Difficulty progresses within a
stable terrain family; fixed evaluation does not advance it. Reward changes use
ordinary Isaac Lab reward terms and must be identified separately from the
common-reward comparison. Successful software checks do not establish physical
locomotion, learnability or a calibrated method ranking.
