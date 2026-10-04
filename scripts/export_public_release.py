"""Export reviewed source and paper figures without Git history or private data.

The allowlist selects paths, not safe content: review the resulting snapshot
and scan it for secrets before publishing. No files are staged or committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tomllib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def select_files(root: Path, patterns: list[str]) -> list[Path]:
    """Select regular files, rejecting missing entries and symlink traversal."""
    selected: set[Path] = set()
    for pattern in patterns:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError(f"Release pattern must stay inside the repo: {pattern}")
        matches = list(root.glob(pattern))
        if not matches:
            raise ValueError(f"Release pattern matched no files: {pattern}")
        for path in matches:
            relative = path.relative_to(root)
            if any(
                (root / part).is_symlink() for part in (relative, *relative.parents)
            ):
                raise ValueError(f"Release selection contains a symlink: {relative}")
            if not path.is_file():
                raise ValueError(f"Release selection is not a regular file: {relative}")
            if path.name.startswith(".env") and relative != Path(
                "agentprivarena/.env.example"
            ):
                raise ValueError(
                    f"Release selection contains local settings: {relative}"
                )
            if ".git" in relative.parts:
                raise ValueError(f"Release selection contains Git metadata: {relative}")
            selected.add(relative)
    if not selected:
        raise ValueError("Release selection is empty")
    return sorted(selected)


def export_snapshot(root: Path, files: list[Path], destination: Path) -> None:
    """Copy selected files into a new directory and record their SHA-256 hashes."""
    # exist_ok=False prevents mixing a release with any previous contents.
    destination.mkdir(parents=True, exist_ok=False)
    hashes: dict[str, str] = {}
    for relative in files:
        source = root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o755 if source.stat().st_mode & stat.S_IXUSR else 0o644)
        hashes[relative.as_posix()] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest = {
        "format_version": 1,
        "source": "working-tree snapshot; Git history is not included",
        "sha256": hashes,
    }
    (destination / "PUBLIC_RELEASE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Create a new snapshot directory; never overwrite"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all selected paths for review"
    )
    args = parser.parse_args()
    with (REPO_ROOT / "public-release.toml").open("rb") as stream:
        config = tomllib.load(stream)
    patterns = config.get("include")
    if not isinstance(patterns, list) or not all(
        isinstance(pattern, str) for pattern in patterns
    ):
        parser.error(
            "public-release.toml must contain an include list of path patterns"
        )
    try:
        files = select_files(REPO_ROOT, patterns)
        if args.list:
            for relative in files:
                print(relative.as_posix())
        if args.output is None:
            print(f"Selected {len(files)} files; dry run (no files copied).")
        else:
            export_snapshot(REPO_ROOT, files, args.output)
            print(f"Exported {len(files)} files to {args.output}.")
            print("Review content and secret-scan the snapshot before publishing.")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
