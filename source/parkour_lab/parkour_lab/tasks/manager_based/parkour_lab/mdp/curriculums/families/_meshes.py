# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lazy mesh factories that keep course definitions dependency-light."""

from __future__ import annotations


def box_mesh(*, extents: tuple[float, float, float]) -> object:
    """Create a Trimesh box without importing Trimesh while courses are loaded."""

    from trimesh.creation import box

    return box(extents=extents)


setattr(box_mesh, "_parkour_metadata_name", "trimesh.creation:box")
