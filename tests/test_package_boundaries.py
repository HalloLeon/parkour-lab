"""Dependency boundaries, checked in isolated processes outside the checkout."""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import textwrap
import unittest

from tests.support import SOURCE_ROOT


class PackageBoundaryTests(unittest.TestCase):
    def test_shared_implementations_do_not_import_experiment_scripts(self):
        package = SOURCE_ROOT / "parkour_lab"
        for path in package.rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                )
                for name in names:
                    self.assertFalse(
                        name == "scripts" or name.startswith(("scripts.", "operator_")),
                        f"{path}:{node.lineno}: {name}",
                    )

    def run_isolated(self, code, blocked):
        guard = textwrap.dedent(
            f"""
            import importlib.abc, sys
            sys.path.insert(0, {str(SOURCE_ROOT)!r})
            blocked = {tuple(blocked)!r}
            class ImportGuard(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if any(fullname == p or fullname.startswith(p + '.') for p in blocked):
                        raise AssertionError('Forbidden dependency: ' + fullname)
            sys.meta_path.insert(0, ImportGuard())
        """
        )
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-c", guard + textwrap.dedent(code)],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_portable_commands_and_contracts_work_without_numeric_or_native_dependencies(
        self,
    ):
        self.run_isolated(
            """
            from parkour_lab.control.command_source import BodyTwistLease
            from parkour_lab.control.command_tape import TapeBuilder
            from parkour_lab.control.motor_contract import binding_sha256
            assert BodyTwistLease().resolve(0).command == (0., 0., 0.)
            builder = TapeBuilder({})
            builder.append(0, (0., 0., 0.))
            assert builder.finish(completed=True)['complete']
            assert len(binding_sha256({})) == 64
            """,
            (
                "torch",
                "numpy",
                "isaaclab",
                "omni",
                "pxr",
                "warp",
                "scripts",
                "parkour_lab.methods",
            ),
        )

    def test_saved_terrain_analysis_does_not_import_live_execution_or_learning(self):
        self.run_isolated(
            """
            from parkour_lab.evaluation.terrain_analysis import analyze_terrain_attempt
            try:
                analyze_terrain_attempt('missing-terrain-run')
            except FileNotFoundError:
                pass
            else:
                raise AssertionError('Missing evidence did not fail')
            """,
            (
                "parkour_lab.evaluation.runner",
                "parkour_lab.runtime",
                "parkour_lab.methods",
                "isaaclab",
                "omni",
                "pxr",
                "warp",
            ),
        )

    def test_cli_help_and_portable_dispatch_need_no_numeric_or_native_imports(self):
        self.run_isolated(
            """
            import contextlib, io, json, tempfile
            from pathlib import Path
            from unittest.mock import patch
            from parkour_lab.__main__ import main
            for prefix in ([], ['train'], ['evaluate'], ['play'], ['export'], ['analyze']):
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        main([*prefix, '--help'])
                    except SystemExit as error:
                        assert error.code == 0
                    else:
                        raise AssertionError('Help did not exit')
            with tempfile.TemporaryDirectory() as folder:
                report = Path(folder) / 'report.json'
                report.write_text('{"status": "recorded"}')
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    assert main(['analyze', folder], standalone=True) == 0
                assert json.loads(output.getvalue()) == {'status': 'recorded'}
            # Dispatch itself is portable; tensor decoding belongs to export_actor.
            with patch('parkour_lab.artifacts.export_actor', return_value={'exported': True}) as export:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    assert main(['export', 'checkpoint.plab', 'actor.plab'], standalone=True) == 0
                assert json.loads(output.getvalue()) == {'exported': True}
                export.assert_called_once_with(Path('checkpoint.plab'), Path('actor.plab'), physics_hz=None)
            """,
            (
                "numpy",
                "torch",
                "isaaclab",
                "isaaclab_tasks",
                "omni",
                "pxr",
                "warp",
                "scripts",
                "parkour_lab.runtime",
                "parkour_lab.methods.roa",
            ),
        )

    def test_imports_respect_package_layers_without_simulator_startup(self):
        cases = {
            "config": (
                "from parkour_lab.config import MethodConfig, TaskConfig\n"
                "assert MethodConfig().name == 'roa'",
                ("torch",),
            ),
            "runtime.native": (
                "from parkour_lab.runtime.native import NativeControllerSession",
                ("parkour_lab.methods",),
            ),
            "runtime.session": (
                "import parkour_lab.runtime.session",
                ("parkour_lab.methods",),
            ),
            "environments": (
                "from parkour_lab.environments import configuration\n"
                "from parkour_lab.environments.terrains import procedural",
                ("parkour_lab.methods",),
            ),
            "environments.worlds": (
                "from parkour_lab.environments.terrains.worlds import build_world",
                ("torch", "parkour_lab.methods"),
            ),
            "methods": (
                "from parkour_lab.methods import base, models\nfrom parkour_lab.methods.roa import model",
                (),
            ),
            "inference": (
                "import parkour_lab.methods.roa.runtime",
                ("parkour_lab.methods.roa.training",),
            ),
            "evaluation": (
                "import parkour_lab.evaluation.runner",
                ("parkour_lab.methods",),
            ),
            "artifacts": (
                "import parkour_lab.artifacts",
                ("torch", "parkour_lab.methods.roa"),
            ),
        }
        for name, (code, blocked) in cases.items():
            with self.subTest(module=name):
                self.run_isolated(
                    code,
                    (
                        *blocked,
                        "isaaclab",
                        "isaaclab_tasks",
                        "omni",
                        "pxr",
                        "rsl_rl",
                        "scripts",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
