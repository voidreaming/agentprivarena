"""Tests for runner MCP config construction."""

from __future__ import annotations

import pytest

from privacylens_live.config import Config
from privacylens_live.runner.mcp_config_builder import build_mcp_config


def test_build_mcp_config_keeps_requested_service_urls() -> None:
    config = Config()

    assert build_mcp_config(["mailpit", "google_drive"], config) == {
        "mcpServers": {
            "mailpit": {"url": "http://mailpit-mcp:8080/mcp"},
            "google_drive": {"url": "http://google-drive-mcp:8080/mcp"},
        }
    }


def test_build_mcp_config_fails_fast_for_unknown_dependency() -> None:
    with pytest.raises(ValueError, match="unknown_service"):
        build_mcp_config(["mailpit", "unknown_service"], Config())


def test_build_mcp_config_fails_fast_for_missing_known_service_url() -> None:
    config = Config(mcp_server_urls={"mailpit": "http://mailpit-mcp:8080/mcp"})

    with pytest.raises(ValueError, match="google_drive"):
        build_mcp_config(["mailpit", "google_drive"], config)


def test_build_mcp_config_can_preserve_legacy_lenient_mode() -> None:
    assert build_mcp_config(["mailpit", "unknown_service"], Config(), strict=False) == {
        "mcpServers": {
            "mailpit": {"url": "http://mailpit-mcp:8080/mcp"},
        }
    }
