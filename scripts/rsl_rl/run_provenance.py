"""Explicit Git identity for training runs, independent of RSL-RL's diff logger.

Only local Git metadata is recorded. Remote URLs and environment variables are
deliberately excluded. Untracked paths are reported, not archived or deleted.
Running this module prints the current checkout's identity without writing logs;
it must not be used to attribute today's HEAD to a historical run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git(directory: Path, *arguments: str, allow_detached: bool = False) -> bytes:
    command = ["git", "-c", "color.ui=false", "-C", str(directory), *arguments]
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except FileNotFoundError as error:
        raise RuntimeError("Training provenance requires Git.") from error
    if result.returncode:
        if allow_detached and result.returncode == 1:
            return b""
        raise RuntimeError(
            "Cannot record training Git provenance: "
            + result.stderr.decode("utf-8", errors="replace").strip()
        )
    return result.stdout


def _status_paths(status: bytes) -> tuple[bool, list[str]]:
    """Parse NUL-delimited porcelain status, including two-path renames."""
    records = status.decode("utf-8", errors="surrogateescape").split("\0")
    tracked_dirty = False
    untracked = []
    index = 0
    while index < len(records) and records[index]:
        record = records[index]
        code = record[:2]
        if code == "??":
            untracked.append(record[3:])
        else:
            tracked_dirty = True
        index += 2 if "R" in code or "C" in code else 1
    return tracked_dirty, untracked


def collect_run_provenance(repository_path: str | Path) -> tuple[dict, bytes]:
    """Capture HEAD, branch, dirty status, and a combined staged/worktree diff."""
    path = Path(repository_path).resolve()
    directory = path.parent if path.is_file() else path
    root = Path(_git(directory, "rev-parse", "--show-toplevel").decode().strip())
    commit = _git(root, "rev-parse", "--verify", "HEAD").decode().strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode().strip()
    branch = (
        _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", allow_detached=True)
        .decode()
        .strip()
    )
    subject = (
        _git(root, "show", "-s", "--format=%s", "HEAD")
        .decode("utf-8", errors="replace")
        .strip()
    )
    tracked_dirty, untracked = _status_paths(
        _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    )
    status = _git(root, "status", "--untracked-files=normal")
    # Plain `git diff` omits staged changes. HEAD includes both tracked layers,
    # and --binary preserves binary edits in the archived patch as well.
    diff = _git(
        root, "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--"
    )
    if commit != _git(root, "rev-parse", "--verify", "HEAD").decode().strip():
        raise RuntimeError(
            "Git HEAD changed while recording provenance; retry with a stable checkout."
        )
    metadata = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(root),
        "commit_sha": commit,
        "commit_tree_sha": tree,
        "commit_subject": subject,
        "branch": branch or None,
        "detached_head": not bool(branch),
        "tracked_dirty": tracked_dirty,
        "untracked_paths": untracked,
        "dirty": tracked_dirty or bool(untracked),
        "diff_file": f"{root.name}.diff",
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_base": "HEAD; includes staged and unstaged tracked changes",
        "untracked_contents_archived": False,
    }
    header = (
        f"--- git commit ---\n{commit}\n"
        f"branch: {branch or '(detached HEAD)'}\n"
        f"tracked_dirty: {str(tracked_dirty).lower()}\n"
        f"untracked_files: {len(untracked)}\n\n"
    ).encode()
    archive = header + b"--- git status ---\n" + status + b"\n--- git diff ---\n" + diff
    return metadata, archive


def write_run_provenance(log_dir: str | Path, repository_path: str | Path) -> dict:
    """Write once, before environment construction; never overwrite an old run."""
    metadata, archive = collect_run_provenance(repository_path)
    git_dir = Path(log_dir) / "git"
    git_dir.mkdir(parents=True, exist_ok=True)
    manifest = git_dir / "provenance.json"
    patch = git_dir / metadata["diff_file"]
    if manifest.exists() or patch.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run provenance in {git_dir}"
        )
    with patch.open("xb") as stream:
        stream.write(archive)
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", default=__file__, help="Git checkout or a file inside it."
    )
    args = parser.parse_args()
    metadata, _ = collect_run_provenance(args.repo)
    print(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
