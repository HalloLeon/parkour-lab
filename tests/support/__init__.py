"""Shared test data and paths; importing support does not import a test suite."""

import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "source/parkour_lab"


@contextmanager
def installed_backends(entries):
    """Exercise real entry-point discovery with temporary distribution metadata."""
    with (
        tempfile.TemporaryDirectory() as folder,
        patch("sys.path", [folder, *sys.path]),
    ):
        name = f"adapter_{Path(folder).name}"
        metadata = Path(folder) / f"{name}-1.0.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(f"Name: {name}\nVersion: 1.0\n")
        (metadata / "entry_points.txt").write_text(
            "[parkour_lab.methods]\n"
            + "".join(f"{name} = {module}\n" for name, module in entries.items())
        )
        yield
