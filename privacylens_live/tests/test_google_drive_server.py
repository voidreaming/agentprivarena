"""Unit tests for the local Google Drive MCP artifact server."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import pytest
from fastmcp.exceptions import ToolError
from mcp.types import ImageContent

from privacylens_live.mcp_servers import google_drive_server as server


def test_search_metadata_and_image_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_bytes = b"fake jpeg bytes"
    _write_drive_index(tmp_path, image_bytes=image_bytes)
    monkeypatch.setenv("GOOGLE_DRIVE_ARTIFACT_ROOT", str(tmp_path))

    search = asyncio.run(
        server.search_files(
            query="Pride parade LGBTQ inclusivity advocacy",
            mime_type="image/jpeg",
        )
    )
    metadata = asyncio.run(server.get_file_metadata(file_id="file_1"))
    image_result = asyncio.run(server.get_file_image(file_id="file_1"))

    assert search["files"] == [
        {
            "file_id": "file_1",
            "name": "pride_parade.jpg",
            "drive_path": "My Drive/Photos/Pride/pride_parade.jpg",
            "mime_type": "image/jpeg",
            "size_bytes": len(image_bytes),
            "available": True,
        }
    ]
    assert metadata["name"] == "pride_parade.jpg"
    assert "local_path" not in metadata
    assert len(image_result) == 2
    assert isinstance(image_result[1], ImageContent)
    assert image_result[1].mimeType == "image/jpeg"


def test_empty_search_is_not_an_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_drive_index(tmp_path)
    monkeypatch.setenv("GOOGLE_DRIVE_ARTIFACT_ROOT", str(tmp_path))

    assert asyncio.run(server.search_files(query="missing")) == {"files": []}


def test_search_without_query_lists_seeded_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_drive_index(tmp_path)
    monkeypatch.setenv("GOOGLE_DRIVE_ARTIFACT_ROOT", str(tmp_path))

    result = asyncio.run(server.search_files())

    assert [file["file_id"] for file in result["files"]] == ["file_1"]


def test_invalid_file_id_raises_tool_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_drive_index(tmp_path)
    monkeypatch.setenv("GOOGLE_DRIVE_ARTIFACT_ROOT", str(tmp_path))

    with pytest.raises(ToolError, match="Use search_files"):
        asyncio.run(server.get_file_metadata(file_id="missing"))


def test_path_traversal_in_index_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "drive_index.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "file_id": "bad",
                        "name": "bad.jpg",
                        "drive_path": "My Drive/bad.jpg",
                        "mime_type": "image/jpeg",
                        "artifact_path": "../bad.jpg",
                    }
                ]
            }
        )
    )
    monkeypatch.setenv("GOOGLE_DRIVE_ARTIFACT_ROOT", str(tmp_path))

    with pytest.raises(ToolError, match="unsafe artifact path"):
        asyncio.run(server.search_files(query=""))


def test_describe_image_uses_injected_describer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_drive_index(tmp_path, image_bytes=b"image bytes")
    monkeypatch.setenv("GOOGLE_DRIVE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("GOOGLE_DRIVE_VISION_MODEL", "vision-model")
    monkeypatch.setenv("GOOGLE_DRIVE_VISION_BASE_URL", "http://vision.example")
    monkeypatch.setenv("GOOGLE_DRIVE_VISION_API_KEY", "test-key")
    calls: list[tuple[bytes, str, str | None, str]] = []

    async def fake_describe(
        image_bytes: bytes,
        *,
        mime_type: str,
        question: str | None,
        detail: Literal["brief", "standard", "detailed"],
    ) -> str:
        calls.append((image_bytes, mime_type, question, detail))
        return "A visible parade scene with flags."

    monkeypatch.setattr(server, "_describe_image_bytes", fake_describe)

    result = asyncio.run(
        server.describe_image(
            file_id="file_1",
            question="What is visible?",
            detail="brief",
        )
    )

    assert result["description"] == "A visible parade scene with flags."
    assert result["description_available"] is True
    assert result["visible_text"] == []
    assert "local_path" not in result
    assert calls == [(b"image bytes", "image/jpeg", "What is visible?", "brief")]


def test_describe_image_reports_unavailable_without_vision_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_drive_index(tmp_path)
    monkeypatch.setenv("GOOGLE_DRIVE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.delenv("GOOGLE_DRIVE_VISION_MODEL", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE_VISION_BASE_URL", raising=False)
    monkeypatch.delenv("GOOGLE_DRIVE_VISION_API_KEY", raising=False)

    result = asyncio.run(server.describe_image(file_id="file_1"))

    assert result["description"] == ""
    assert result["description_available"] is False
    assert "Vision model is not configured" in result["unavailable_reason"]
    assert "local_path" not in result


def test_describe_image_reports_unavailable_when_vision_request_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_drive_index(tmp_path)
    monkeypatch.setenv("GOOGLE_DRIVE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("GOOGLE_DRIVE_VISION_MODEL", "vision-model")
    monkeypatch.setenv("GOOGLE_DRIVE_VISION_BASE_URL", "http://vision.example")
    monkeypatch.setenv("GOOGLE_DRIVE_VISION_API_KEY", "test-key")

    async def failing_describe(
        image_bytes: bytes,
        *,
        mime_type: str,
        question: str | None,
        detail: Literal["brief", "standard", "detailed"],
    ) -> str:
        _ = (image_bytes, mime_type, question, detail)
        raise ToolError("Vision model request failed: content_policy_violation")

    monkeypatch.setattr(server, "_describe_image_bytes", failing_describe)

    result = asyncio.run(server.describe_image(file_id="file_1"))

    assert result["description"] == ""
    assert result["description_available"] is False
    assert "content_policy_violation" in result["unavailable_reason"]
    assert "local_path" not in result


def _write_drive_index(
    root: Path,
    *,
    image_bytes: bytes = b"fake image",
) -> None:
    files_dir = root / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "pride_parade.jpg").write_bytes(image_bytes)
    (root / "drive_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "files": [
                    {
                        "file_id": "file_1",
                        "name": "pride_parade.jpg",
                        "drive_path": "My Drive/Photos/Pride/pride_parade.jpg",
                        "mime_type": "image/jpeg",
                        "artifact_path": "files/pride_parade.jpg",
                    }
                ],
            }
        )
    )
