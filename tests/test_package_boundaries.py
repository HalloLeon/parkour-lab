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
            from parkour_lab.control.command_tape import TapeBuilder
            from parkour_lab.control.motor_contract import binding_sha256
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

    def test_saved_capture_scoring_does_not_import_live_execution_or_learning(self):
        self.run_isolated(
            """
            from parkour_lab.evaluation import score_capture
            try:
                score_capture('missing-capture')
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

    def test_cli_help_needs_no_numeric_native_or_learner_imports(self):
        self.run_isolated(
            """
            import contextlib, io
            from parkour_lab.__main__ import main
            for prefix in ([], ['score'], ['cases']):
                with contextlib.redirect_stdout(io.StringIO()):
                    try:
                        main([*prefix, '--help'])
                    except SystemExit as error:
                        assert error.code == 0
                    else:
                        raise AssertionError('Help did not exit')
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
                "parkour_lab.methods",
                "rsl_rl",
            ),
        )

    def test_retired_controller_workflows_are_absent(self):
        for relative in (
            "methods",
            "artifacts.py",
            "experiment.py",
            "config.py",
            "runtime/training.py",
            "runtime/metrics.py",
            "environments/task_config.py",
            "environments/commands/native.py",
            "environments/commands/sampling.py",
            "environments/commands/sequences.py",
            "evaluation/flat.py",
            "evaluation/terrain.py",
            "evaluation/terrain_analysis.py",
            "evaluation/native.py",
        ):
            self.assertFalse(
                (SOURCE_ROOT / "parkour_lab" / relative).exists(), relative
            )

    def test_imports_respect_package_layers_without_simulator_startup(self):
        cases = {
            "task_settings": (
                "from parkour_lab.benchmark import Go2Config\n"
                "assert Go2Config().num_envs == 160",
                ("torch", "parkour_lab.methods"),
            ),
            "benchmark": (
                "from parkour_lab.benchmark import Go2Config, TASK_IDS\n"
                "from parkour_lab.benchmark.task import Go2Task\n"
                "assert len(TASK_IDS) == 6\n"
                "assert Go2Config().task_id == 'Go2-Flat-v1'",
                ("parkour_lab.methods", "parkour_lab.config", "parkour_lab.artifacts"),
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
                "from parkour_lab.environments import configuration",
                ("parkour_lab.methods",),
            ),
            "environments.terrains": (
                "from parkour_lab.environments.terrains import specification",
                ("torch", "parkour_lab.methods"),
            ),
            "evaluation": (
                "from parkour_lab.evaluation import evaluate, score_capture, case_bank\n"
                "assert len(case_bank('validation')) == 114",
                ("parkour_lab.methods",),
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
