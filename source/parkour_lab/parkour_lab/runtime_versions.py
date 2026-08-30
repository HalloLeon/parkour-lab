# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Authoritative, simulator-independent Parkour Lab runtime requirements."""

from __future__ import annotations

import importlib.metadata

REQUIRED_RUNTIME_VERSIONS = {
    "isaaclab": "2.3.2.post1",
    "isaacsim": "5.1.0.0",
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "rsl-rl-lib": "3.1.2",
}


def require_runtime_versions() -> None:
    """Fail unless every installed distribution matches the tested stack."""

    problems: list[str] = []
    for distribution, required in REQUIRED_RUNTIME_VERSIONS.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"{distribution}=={required} is not installed")
            continue
        # Compare the complete normalized distribution metadata, including
        # PyTorch's CUDA local-version suffix.
        if installed != required:
            problems.append(
                f"{distribution}=={required} is required (installed: {installed})"
            )
    if problems:
        raise RuntimeError(
            "Unsupported Parkour Lab runtime: " + "; ".join(problems) + "."
        )
