"""Focused tests for RocketChat MCP registry/user helpers."""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any


MCP_DIR = Path(__file__).resolve().parent.parent / "mcp_servers"
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

rocketchat_server = importlib.import_module("rocketchat_server")


def test_message_sender_returns_machine_usable_username() -> None:
    assert (
        rocketchat_server._message_sender(
            {"alias": "Jane Doe", "u": {"username": "admin"}}
        )
        == "admin"
    )
    assert (
        rocketchat_server._message_sender(
            {"u": {"username": "john_doe", "name": "John Doe"}}
        )
        == "john_doe"
    )
    assert (
        rocketchat_server._message_sender({"u": {"username": "john_doe"}}) == "john_doe"
    )
    assert rocketchat_server._message_sender({"u": {"username": "admin"}}) == "admin"


def test_message_to_result_exposes_display_name_separately() -> None:
    assert rocketchat_server._message_to_result(
        {
            "_id": "msg-id",
            "alias": "Jane Doe",
            "u": {"username": "jane_doe"},
            "msg": "hello",
            "ts": "2026-01-01T00:00:00Z",
        },
        "general",
    ) == {
        "message_id": "msg-id",
        "sender": "jane_doe",
        "sender_display": "Jane Doe",
        "time": "2026-01-01T00:00:00Z",
        "text": "hello",
        "channel": "general",
    }


def test_search_users_matches_profile_fields(monkeypatch) -> None:
    async def fake_authenticated_call(
        fn: Callable[[dict[str, str]], Awaitable[list[dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        return await fn({})

    async def fake_http_get_params(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del url, params, headers
        return {
            "users": [
                {
                    "username": "mark_johnson",
                    "name": "Mark Johnson",
                    "emails": [{"address": "mark@agentprivarena.local"}],
                    "status": "online",
                    "statusText": "External partner",
                },
                {
                    "username": "admin",
                    "name": "Admin",
                    "emails": [{"address": "admin@agentprivarena.local"}],
                    "status": "online",
                },
            ]
        }

    monkeypatch.setattr(
        rocketchat_server,
        "_authenticated_call",
        fake_authenticated_call,
    )
    monkeypatch.setattr(rocketchat_server, "http_get_params", fake_http_get_params)

    result = asyncio.run(rocketchat_server.search_users("partner"))

    assert result == {
        "users": [
            {
                "username": "mark_johnson",
                "name": "Mark Johnson",
                "email": "mark@agentprivarena.local",
                "status": "online",
                "status_text": "External partner",
            }
        ]
    }
