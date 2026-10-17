"""Add one seeded native height layer to existing supporting surface meshes."""

import numpy as np


def _refine_box(mesh, spacing):
    """Tessellate native axis-aligned stair boxes without changing their surfaces.

    A uniform triangle subdivision wastes vertices on the long, narrow treads.
    Rectangular face grids resolve each axis independently, including risers, so
    adjacent face edges receive the same displacement without opening cracks.
    """
    import trimesh

    vertices, faces, offset = [], [], 0
    for facet, normal in zip(mesh.facets, mesh.facets_normal, strict=True):
        axis = int(np.argmax(np.abs(normal)))
        if not np.isclose(abs(normal[axis]), 1.0):
            raise ValueError("Composition requires native axis-aligned stair boxes")
        points = mesh.vertices[np.unique(mesh.faces[facet])]
        lower, upper = points.min(axis=0), points.max(axis=0)
        first, second = [index for index in range(3) if index != axis]
        coordinates = [
            np.linspace(
                lower[index],
                upper[index],
                max(1, int(np.ceil((upper[index] - lower[index]) / spacing))) + 1,
            )
            for index in (first, second)
        ]
        a, b = np.meshgrid(*coordinates, indexing="ij")
        grid = np.empty((a.size, 3))
        grid[:, axis], grid[:, first], grid[:, second] = (
            lower[axis],
            a.ravel(),
            b.ravel(),
        )
        width = len(coordinates[1])
        row, column = np.meshgrid(
            np.arange(len(coordinates[0]) - 1), np.arange(width - 1), indexing="ij"
        )
        start = (row * width + column).ravel()
        triangles = np.concatenate(
            (
                np.column_stack((start, start + width, start + 1)),
                np.column_stack((start + 1, start + width, start + width + 1)),
            )
        )
        triangle = grid[triangles[0]]
        if (
            np.dot(
                np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0]), normal
            )
            < 0
        ):
            triangles = triangles[:, ::-1]
        vertices.append(grid)
        faces.append(triangles + offset)
        offset += len(grid)
    return trimesh.Trimesh(
        vertices=np.concatenate(vertices), faces=np.concatenate(faces)
    )


def compose_surface(pieces, rough_mesh, *, family, size):
    """Vertically displace the same native surface, retaining its central start.

    The native Uneven mesh already contains quantization, cubic interpolation
    and its implicit border. Bilinear sampling transfers that layer to refined
    stair faces; slopes/hills retain their original fine heightfield triangles.
    Components remain ID parameters, while their combined local grades need not.
    """
    x, xi = np.unique(rough_mesh.vertices[:, 0], return_inverse=True)
    y, yi = np.unique(rough_mesh.vertices[:, 1], return_inverse=True)
    if len(x) * len(y) != len(rough_mesh.vertices):
        raise ValueError("Require the native rectangular Uneven height grid")
    heights = np.empty((len(x), len(y)))
    heights[xi, yi] = rough_mesh.vertices[:, 2]
    result = []
    for original in pieces:
        mesh = _refine_box(original, 0.1) if family == "Stairs" else original.copy()
        xy = mesh.vertices[:, :2]
        radius = np.max(np.abs(xy - np.asarray(size) / 2), axis=1)
        edge = np.min(np.column_stack((xy, np.asarray(size) - xy)), axis=1)
        weight = np.clip((radius - 1.0) / 0.5, 0, 1) * np.clip(edge / 0.3, 0, 1)
        clipped = np.clip(xy, [x[0], y[0]], [x[-1], y[-1]])
        mesh.vertices[:, 2] += _bilinear(x, y, heights, clipped) * weight
        result.append(mesh)
    return result


def _bilinear(x, y, heights, xy):
    """Sample a rectangular native height grid without another runtime dependency."""
    i = np.clip(np.searchsorted(x, xy[:, 0], side="right") - 1, 0, len(x) - 2)
    j = np.clip(np.searchsorted(y, xy[:, 1], side="right") - 1, 0, len(y) - 2)
    u = (xy[:, 0] - x[i]) / (x[i + 1] - x[i])
    v = (xy[:, 1] - y[j]) / (y[j + 1] - y[j])
    return (
        (1 - u) * (1 - v) * heights[i, j]
        + u * (1 - v) * heights[i + 1, j]
        + (1 - u) * v * heights[i, j + 1]
        + u * v * heights[i + 1, j + 1]
    )
