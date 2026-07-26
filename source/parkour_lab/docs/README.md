# Parkour Lab

Parkour Lab is an Isaac Lab reinforcement-learning environment for training a
Unitree A1 with a balanced obstacle-family by difficulty curriculum. It also
provides fixed-cell, seeded evaluation with numerical metrics and optional
video recording. The playback script can sweep the complete family/difficulty
matrix with ``--all_courses`` or evaluate one explicitly selected cell.

The registered Gym tasks are:

- `Parkour-Lab-v0` for adaptive training
- `Parkour-Lab-Play-v0` for reproducible evaluation and video

See the repository's top-level `README.md` for installation, training, and
evaluation commands.
