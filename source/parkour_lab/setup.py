# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'parkour_lab' python package."""

from pathlib import Path

import toml
from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = Path(__file__).resolve().parent
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(EXTENSION_PATH / "config" / "extension.toml")

# Installation operation
setup(
    name="parkour_lab",
    packages=find_packages(),
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    long_description=(EXTENSION_PATH / "docs" / "README.md").read_text(
        encoding="utf-8"
    ),
    long_description_content_type="text/markdown",
    author=EXTENSION_TOML_DATA["package"]["author"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    license_files=[],
    python_requires=">=3.10,<3.12",
    install_requires=[
        "numpy<2",
        "torch>=2.5",
        "PyYAML>=6",
        "trimesh>=4",
        "gymnasium==1.2.1",
    ],
    extras_require={"roa": ["rsl-rl-lib==3.1.2", "tensordict"]},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
)
