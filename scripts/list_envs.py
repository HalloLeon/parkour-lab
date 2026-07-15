# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.

All the environments are registered in the ``parkour_lab`` extension. Their task
IDs start with ``Parkour-``.
"""

# Launch Isaac Sim before importing modules that depend on it.

from isaaclab.app import AppLauncher

# Launch the Omniverse application in headless mode.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# The remaining imports require the running simulation application.

import gymnasium as gym
import parkour_lab.tasks  # noqa: F401
from prettytable import PrettyTable


def main() -> None:
    """Print all environments registered in ``parkour_lab`` extension."""

    # Create and configure the table used to display registered tasks.
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # Collect every registered Parkour Lab environment in registry order.
    index = 0
    for task_spec in gym.registry.values():
        if task_spec.id.startswith("Parkour-"):
            table.add_row(
                [
                    index + 1,
                    task_spec.id,
                    task_spec.entry_point,
                    task_spec.kwargs["env_cfg_entry_point"],
                ]
            )
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
