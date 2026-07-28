# Copyright (c) 2026, Leon Yi Bai
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Framework-independent privileged-teacher models."""

from .model import (
    PrivilegedScanEncoder,
    PrivilegedTeacherActor,
    PrivilegedTeacherModelCfg,
    PrivilegedTeacherPolicy,
)

__all__ = [
    "PrivilegedScanEncoder",
    "PrivilegedTeacherActor",
    "PrivilegedTeacherModelCfg",
    "PrivilegedTeacherPolicy",
]
