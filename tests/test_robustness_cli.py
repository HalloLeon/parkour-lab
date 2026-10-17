"""Portable host checks: fail closed, preserve receipts and verify paired controls."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import check_robustness as check
from scripts._robustness_checks import geometry, motor, physics


def surfaces():
    axis = np.linspace(-5.4, 5.4, 73)
    x, y = np.meshgrid(axis, axis)
    base = np.zeros((2, 73, 73))
    noise = np.where((abs(x) > 1.3) | (abs(y) > 1.3), 0.01, 0.0)
    samples = {}
    for condition in ("id", "composition", "terrain_ood"):
        samples[condition] = dict(
            axis_m=axis,
            height_w_m=base + (noise if condition == "composition" else 0),
            env_origin_w_m=np.zeros((2, 3)),
            root_position_w_m=np.array([[0, 0, 0.4], [12, 0, 0.4]]),
            foot_position_w_m=np.zeros((2, 4, 3)),
            root_support_w_m=np.zeros((2, 3)),
            foot_support_w_m=np.zeros((2, 4, 3)),
        )
    definitions = dict(
        terrain_ood=dict(
            fixed_level=3,
            columns=[dict(family="Slopes")],
            levels={3: dict(slope_rise_run=0.4)},
        )
    )
    return samples, definitions


class RobustnessCheckTests(unittest.TestCase):
    def test_help_imports_no_native_or_learning_runtime(self):
        program = """
import importlib.abc, runpy, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in
               ('torch', 'stable_baselines3', 'isaaclab', 'isaacsim', 'omni', 'pxr')):
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, Guard())
sys.argv = ['check_robustness', '--help']
runpy.run_module('scripts.check_robustness', run_name='__main__')
"""
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--check {geometry,motor}", result.stdout)
        self.assertIn("--factor {0,0.5,1}", result.stdout)

    def test_invalid_seed_and_existing_output_fail_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "new"
            with self.assertRaises(SystemExit):
                check.main(
                    ["--check", "motor", "--seed", "-1", "--output", str(missing)]
                )
            self.assertFalse(missing.exists())
            with self.assertRaises(FileExistsError):
                check.main(["--check", "motor", "--output", tmp])

    def test_native_failure_retains_counts_and_closes_environment_then_app(self):
        order = []
        app = SimpleNamespace(close=lambda: order.append("app"))
        task = SimpleNamespace(
            metadata={},
            interactions=dict(session=4, cumulative=4, uncertain=0),
            close=lambda: order.append("task"),
        )
        launcher = SimpleNamespace(AppLauncher=lambda **_: SimpleNamespace(app=app))
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(sys.modules, {"isaaclab.app": launcher}),
            patch.object(check, "write_run_provenance"),
            patch.object(check, "dependency_identity", return_value={}),
            patch.object(check, "_task", return_value=task),
            patch.object(motor, "direct_motor_checks", return_value={}),
            patch.object(
                physics,
                "check_physics",
                side_effect=RuntimeError("injected physical failure"),
            ),
            patch.object(check, "exit_native_process") as exit_process,
        ):
            output = Path(tmp) / "new"
            code = check.main(["--check", "motor", "--output", str(output)])
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(code, 2)
            self.assertIn("injected physical failure", report["error"])
            self.assertEqual(report["interactions"]["session"], 4)
            self.assertEqual(
                report["contexts"]["motor"]["cleanup"]["environment"], "complete"
            )
            self.assertEqual(report["cleanup"]["application"], "complete")
            self.assertEqual(order, ["task", "app"])
            exit_process.assert_called_once_with(2)

    def test_surface_context_failure_is_durable_and_does_not_run_next_condition(self):
        order, count = [], 0

        def task(args, folder, **kwargs):
            nonlocal count
            count += 1
            return SimpleNamespace(
                metadata={},
                interactions=dict(session=0, cumulative=0, uncertain=0),
                _env=SimpleNamespace(cfg=SimpleNamespace(benchmark_terrain={})),
                close=lambda: order.append(kwargs["condition"]),
            )

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(check, "_task", side_effect=task),
            patch.object(
                geometry,
                "sample_surface",
                side_effect=[({}, {}), RuntimeError("ray miss")],
            ),
        ):
            output = Path(tmp)
            report = dict(contexts={})
            with self.assertRaisesRegex(RuntimeError, "ray miss"):
                check._geometry(None, output, report, lambda: None, None)
            self.assertEqual(count, 2)
            self.assertEqual(order, ["id", "composition"])
            receipt = json.loads((output / "composition" / "report.json").read_text())
            self.assertEqual(receipt["status"], "ERROR")
            self.assertEqual(receipt["cleanup"]["environment"], "complete")
            self.assertFalse((output / "terrain_ood").exists())

    def test_pairing_guards_reject_changed_starts_flat_composition_and_id_severity(
        self,
    ):
        sample, definition = surfaces()
        result = geometry.compare_surfaces(sample, definition)
        self.assertTrue(result["paired_starts_and_support"])
        self.assertFalse(result["traversal_validated"])
        sample["composition"]["root_position_w_m"][0, 2] += 0.01
        with self.assertRaises(AssertionError):
            geometry.compare_surfaces(sample, definition)
        sample, definition = surfaces()
        sample["composition"]["height_w_m"][:] = 0
        with self.assertRaises(AssertionError):
            geometry.compare_surfaces(sample, definition)
        sample, definition = surfaces()
        definition["terrain_ood"]["levels"][3]["slope_rise_run"] = 0.3
        with self.assertRaises(AssertionError):
            geometry.compare_surfaces(sample, definition)


if __name__ == "__main__":
    unittest.main()
