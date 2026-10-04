"""Check the boundary between source snapshots and private working files."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_public_release import export_snapshot, select_files


def test_snapshot_copies_only_selected_content(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("VALUE = 1\n")
    (source / ".env").write_text("PRIVATE_KEY=local-only\n")
    (source / "responses.json").write_text('{"private": true}')
    (source / ".git").mkdir()
    (source / ".git" / "config").write_text("private remote")
    destination = tmp_path / "public"

    files = select_files(source, ["*.py"])
    export_snapshot(source, files, destination)

    assert sorted(path.name for path in destination.iterdir()) == [
        "PUBLIC_RELEASE_MANIFEST.json",
        "module.py",
    ]
    manifest = json.loads((destination / "PUBLIC_RELEASE_MANIFEST.json").read_text())
    assert manifest["sha256"] == {
        "module.py": hashlib.sha256(b"VALUE = 1\n").hexdigest()
    }
    with pytest.raises(FileExistsError):
        export_snapshot(source, files, destination)
    assert (destination / "module.py").read_text() == "VALUE = 1\n"


@pytest.mark.parametrize("pattern", ["../private.py", "/tmp/private.py", "missing.py"])
def test_invalid_or_missing_selection_fails(tmp_path: Path, pattern: str) -> None:
    with pytest.raises(ValueError):
        select_files(tmp_path, [pattern])


@pytest.mark.parametrize("path", [".env", ".env.production", ".git/config"])
def test_private_settings_cannot_be_selected(tmp_path: Path, path: str) -> None:
    private = tmp_path / path
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text("private")
    with pytest.raises(ValueError):
        select_files(tmp_path, [path])


def test_symlinked_file_or_parent_is_rejected(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    (private / "settings.py").write_text("PRIVATE = True\n")
    source = tmp_path / "source"
    source.mkdir()
    (source / "settings.py").symlink_to(private / "settings.py")
    (source / "nested").symlink_to(private, target_is_directory=True)
    for pattern in ("settings.py", "nested/settings.py"):
        with pytest.raises(ValueError, match="symlink"):
            select_files(source, [pattern])


def test_reviewed_env_example_is_included(tmp_path: Path) -> None:
    example = tmp_path / "agentprivarena" / ".env.example"
    example.parent.mkdir()
    example.write_text("LLM_API_KEY=\n")
    assert select_files(tmp_path, ["agentprivarena/.env.example"]) == [
        Path("agentprivarena/.env.example")
    ]
