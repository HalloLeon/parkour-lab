"""Shared test data and paths; importing support does not import a test suite."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "source/parkour_lab"
