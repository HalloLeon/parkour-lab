"""Pure helpers for mapping generated terrain difficulty to discrete levels."""

from __future__ import annotations

import math


def difficulty_to_level(difficulty: float, num_levels: int) -> int:
    """Map a normalized difficulty to an equally sized discrete level bin.

    Isaac Lab generates row ``r`` with a difficulty in
    ``[r / num_rows, (r + 1) / num_rows)``. When ``num_rows`` equals the
    number of logical levels, this mapping makes the physical terrain row and
    the task level identical. The upper endpoint is assigned to the last bin.
    """

    if (
        isinstance(num_levels, bool)
        or not isinstance(num_levels, int)
        or num_levels <= 0
    ):
        raise ValueError("num_levels must be a positive integer.")

    difficulty = float(difficulty)
    if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
        raise ValueError("difficulty must be finite and in [0, 1].")

    return min(int(difficulty * num_levels), num_levels - 1)
