# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'parkour_lab' python package."""

import os

import toml
from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(
    os.path.join(EXTENSION_PATH, "config", "extension.toml")
)

# Installation operation
setup(
    name="parkour_lab",
    packages=find_packages(),
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    license="BSD-3-Clause",
    python_requires=">=3.10,<3.12",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
)
