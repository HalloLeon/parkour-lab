"""Real temporary Git repositories protect run identity and dirty-tree logging."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from parkour_lab import provenance


@unittest.skipUnless(shutil.which("git"), "Git is required.")
class RunProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "parkour_lab"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Provenance Test")
        self.git("config", "user.email", "provenance@example.invalid")
        self.git("config", "commit.gpgsign", "false")
        (self.repo / "source.py").write_text("initial = 1\n")
        (self.repo / "binary.dat").write_bytes(b"\x00original\x00")
        self.git("add", "source.py", "binary.dat")
        self.git("commit", "-q", "-m", "Initial test code")
        self.commit = self.git("rev-parse", "HEAD").decode().strip()

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-C", str(self.repo), *args], stderr=subprocess.PIPE
        )

    def test_clean_run_always_records_full_commit_and_header(self):
        log_dir = self.root / "logs"
        result = provenance.write_run_provenance(log_dir, self.repo / "source.py")
        self.assertEqual(result["commit_sha"], self.commit)
        self.assertFalse(result["dirty"])
        self.assertFalse(result["detached_head"])
        self.assertEqual(result["diff_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(
            json.loads((log_dir / "git/provenance.json").read_text()), result
        )
        archive = (log_dir / "git/parkour_lab.diff").read_text()
        self.assertTrue(archive.startswith(f"--- git commit ---\n{self.commit}\n"))
        self.assertIn("--- git status ---", archive)
        self.assertIn("--- git diff ---", archive)

    def test_literal_untracked_data_directory_does_not_hide_commit(self):
        data = self.repo / "${data}"
        data.mkdir()
        payload = data / "result.txt"
        payload.write_text("must not archive arbitrary data contents")
        result, archive = provenance.collect_run_provenance(self.repo)
        self.assertEqual(result["commit_sha"], self.commit)
        self.assertFalse(result["tracked_dirty"])
        self.assertTrue(result["dirty"])
        self.assertEqual(result["untracked_paths"], ["${data}/result.txt"])
        self.assertNotIn(payload.read_bytes(), archive)
        self.assertTrue(payload.exists())

    def test_diff_includes_staged_unstaged_and_binary_changes(self):
        (self.repo / "source.py").write_text("initial = 2\n")
        self.git("add", "source.py")
        (self.repo / "source.py").write_text("initial = 2\nunstaged = 3\n")
        (self.repo / "binary.dat").write_bytes(b"\x00changed\x00")
        result, archive = provenance.collect_run_provenance(self.repo)
        self.assertTrue(result["tracked_dirty"])
        self.assertIn(b"+initial = 2", archive)
        self.assertIn(b"+unstaged = 3", archive)
        self.assertIn(b"GIT binary patch", archive)
        raw_diff = archive.split(b"--- git diff ---\n", 1)[1]
        self.assertEqual(hashlib.sha256(raw_diff).hexdigest(), result["diff_sha256"])

    def test_detached_head_keeps_identity_without_inventing_branch(self):
        self.git("checkout", "--detach", "-q", self.commit)
        result, _ = provenance.collect_run_provenance(self.repo)
        self.assertEqual(result["commit_sha"], self.commit)
        self.assertIsNone(result["branch"])
        self.assertTrue(result["detached_head"])

    def test_git_worktree_and_subdirectory_resolve_correct_root(self):
        worktree = self.root / "linked checkout"
        self.git("worktree", "add", "--detach", "-q", str(worktree), self.commit)
        result, _ = provenance.collect_run_provenance(worktree / "source.py")
        self.assertEqual(result["repository_root"], str(worktree.resolve()))
        self.assertEqual(result["commit_sha"], self.commit)

    def test_renames_and_newline_filenames_do_not_corrupt_untracked_list(self):
        self.git("mv", "source.py", "renamed source.py")
        (self.repo / "new\nfile.py").write_text("x\n")
        result, _ = provenance.collect_run_provenance(self.repo)
        self.assertTrue(result["tracked_dirty"])
        self.assertEqual(result["untracked_paths"], ["new\nfile.py"])

    def test_existing_run_provenance_is_never_overwritten(self):
        log_dir = self.root / "logs"
        provenance.write_run_provenance(log_dir, self.repo)
        before = (log_dir / "git/provenance.json").read_bytes()
        with self.assertRaises(FileExistsError):
            provenance.write_run_provenance(log_dir, self.repo)
        self.assertEqual(before, (log_dir / "git/provenance.json").read_bytes())

    def test_existing_diff_is_not_relabelled_as_current_head(self):
        log_dir = self.root / "partial"
        (log_dir / "git").mkdir(parents=True)
        existing = log_dir / "git/parkour_lab.diff"
        existing.write_text("--- git status ---\nold run, unknown commit\n")
        with self.assertRaises(FileExistsError):
            provenance.write_run_provenance(log_dir, self.repo)
        self.assertNotIn(self.commit, existing.read_text())
        self.assertFalse((log_dir / "git/provenance.json").exists())

    def test_not_a_repository_fails_instead_of_logging_unknown_identity(self):
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            provenance.collect_run_provenance(self.root)

    def test_nested_editable_dependency_records_checkout_and_dirty_diff(self):
        module = self.repo / "source/sdk/sdk/__init__.py"
        module.parent.mkdir(parents=True)
        module.write_text('"""Nested SDK fixture."""\n')
        self.git("add", "source")
        self.git("commit", "-q", "-m", "Add nested dependency")
        commit = self.git("rev-parse", "HEAD").decode().strip()
        with (
            patch("importlib.metadata.version", return_value="2.3.2"),
            patch(
                "importlib.import_module",
                return_value=SimpleNamespace(__file__=str(module)),
            ),
        ):
            clean = provenance.dependency_identity("sdk", "sdk")
            module.write_text('"""Modified SDK fixture."""\n')
            dirty = provenance.dependency_identity("sdk", "sdk")
        self.assertEqual(clean["commit"], commit)
        self.assertFalse(clean["dirty"])
        self.assertEqual(clean["diff_sha256"], hashlib.sha256(b"").hexdigest())
        self.assertEqual(dirty["commit"], commit)
        self.assertTrue(dirty["dirty"])
        self.assertNotEqual(dirty["source_sha256"], clean["source_sha256"])
        self.assertNotEqual(dirty["diff_sha256"], clean["diff_sha256"])

    def test_dependency_installed_inside_repository_does_not_inherit_its_commit(self):
        module = self.repo / ".venv/lib/python3.11/site-packages/sdk/__init__.py"
        module.parent.mkdir(parents=True)
        module.write_text('"""Installed SDK fixture."""\n')
        with (
            patch("importlib.metadata.version", return_value="2.3.2"),
            patch(
                "importlib.import_module",
                return_value=SimpleNamespace(__file__=str(module)),
            ),
        ):
            receipt = provenance.dependency_identity("sdk", "sdk")
        self.assertEqual(receipt["version"], "2.3.2")
        self.assertIn("source_sha256", receipt)
        self.assertNotIn("commit", receipt)
        self.assertNotIn("dirty", receipt)
        self.assertNotIn("diff_sha256", receipt)


class DependencyIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "sdk" / "__init__.py"
        self.path.parent.mkdir()
        self.path.write_text('"""Importable component."""\n')
        self.module = SimpleNamespace(__file__=str(self.path))

    def test_distribution_and_module_versions_remain_distinct(self):
        self.module.__version__ = "2.3.2"
        with (
            patch("importlib.metadata.version", return_value="2.3.2.post1"),
            patch("importlib.import_module", return_value=self.module),
        ):
            receipt = provenance.dependency_identity("sdk", "sdk-distribution")
        self.assertEqual(receipt["requested_distribution"], "sdk-distribution")
        self.assertEqual(receipt["version"], "2.3.2.post1")
        self.assertEqual(receipt["module_version"], "2.3.2")

    def test_importable_component_without_metadata_retains_source_evidence(self):
        self.module.__version__ = "2.3.2"
        with (
            patch(
                "importlib.metadata.version", side_effect=PackageNotFoundError("sdk")
            ) as lookup,
            patch("importlib.import_module", return_value=self.module),
        ):
            receipt = provenance.dependency_identity("sdk", "sdk")
        lookup.assert_called_once_with("sdk")
        self.assertEqual(receipt["requested_distribution"], "sdk")
        self.assertIsNone(receipt["version"])
        self.assertEqual(receipt["module_version"], "2.3.2")
        self.assertEqual(receipt["module_path"], str(self.path.resolve().parent))
        expected = hashlib.sha256(b"__init__.py" + self.path.read_bytes()).hexdigest()
        self.assertEqual(receipt["source_sha256"], expected)

    def test_missing_or_nonstring_module_version_is_not_invented(self):
        for module_version in (None, "", 232):
            with self.subTest(module_version=module_version):
                self.module.__version__ = module_version
                with (
                    patch(
                        "importlib.metadata.version",
                        side_effect=PackageNotFoundError("sdk"),
                    ),
                    patch("importlib.import_module", return_value=self.module),
                ):
                    receipt = provenance.dependency_identity("sdk", "sdk")
                self.assertIsNone(receipt["version"])
                self.assertNotIn("module_version", receipt)
                json.dumps(receipt, allow_nan=False)

    def test_missing_dependency_import_still_fails(self):
        with (
            patch("importlib.metadata.version") as lookup,
            patch("importlib.import_module", side_effect=ModuleNotFoundError("sdk")),
        ):
            with self.assertRaises(ModuleNotFoundError):
                provenance.dependency_identity("sdk", "sdk")
        lookup.assert_not_called()

    def test_unexpected_metadata_error_still_fails(self):
        with (
            patch("importlib.metadata.version", side_effect=OSError("unreadable")),
            patch("importlib.import_module", return_value=self.module),
        ):
            with self.assertRaisesRegex(OSError, "unreadable"):
                provenance.dependency_identity("sdk", "sdk")


class PackageSourceIdentityTests(unittest.TestCase):
    def test_dependency_receipt_describes_imported_core_dependency(self):
        from importlib.metadata import version

        import gymnasium

        receipt = provenance.dependency_identity("gymnasium", "gymnasium")
        self.assertEqual(receipt["version"], version("gymnasium"))
        self.assertEqual(
            Path(receipt["module_path"]), Path(gymnasium.__file__).resolve().parent
        )
        self.assertEqual(len(receipt["source_sha256"]), 64)
        if (Path(receipt["module_path"]).parent / ".git").exists():
            self.assertEqual(len(receipt["commit"]), 40)
            self.assertIs(type(receipt["dirty"]), bool)

    def test_inventory_covers_split_owners_and_is_independent_of_machine_root(self):
        current = provenance.package_source_identity()
        self.assertTrue(
            {
                "environments/configuration.py",
                "environments/commands/episode.py",
                "environments/terrains/generation.py",
                "runtime/native.py",
                "benchmark/config.py",
                "evaluation/runner.py",
                "evaluation/scoring/case.py",
                "provenance.py",
            }.issubset(current)
        )
        self.assertEqual(list(current), sorted(current))
        self.assertTrue(all(not Path(name).is_absolute() for name in current))
        self.assertTrue(all(len(value) == 64 for value in current.values()))
        with tempfile.TemporaryDirectory() as directory:
            inventories = []
            for machine in ("machine_a", "machine_b"):
                root = Path(directory) / machine
                root.mkdir()
                (root / "provenance.py").write_text("identity = 1\n")
                (root / "nested").mkdir()
                (root / "nested/owner.py").write_text("value = 2\n")
                (root / "nested/README.md").write_text("not runtime code\n")
                with patch.object(provenance, "__file__", str(root / "provenance.py")):
                    inventories.append(provenance.package_source_identity())
            self.assertEqual(inventories[0], inventories[1])
            self.assertEqual(
                inventories[0],
                {
                    "nested/owner.py": hashlib.sha256(b"value = 2\n").hexdigest(),
                    "provenance.py": hashlib.sha256(b"identity = 1\n").hexdigest(),
                },
            )
