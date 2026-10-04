"""Unit tests for Mattermost MCP server helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


MCP_DIR = Path(__file__).resolve().parent.parent / "mcp_servers"
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))


def _post_sender(post: dict[str, Any], id_to_name: dict[str, str]) -> str:
    import importlib

    mod = importlib.import_module("mattermost_server")
    return mod._post_sender(post, id_to_name)


def _post_to_message(post: dict[str, Any], id_to_name: dict[str, str]) -> dict:
    import importlib

    mod = importlib.import_module("mattermost_server")
    return mod._post_to_message(post, id_to_name)


def _post_time(post: dict[str, Any]) -> int | str:
    import importlib

    mod = importlib.import_module("mattermost_server")
    return mod._post_time(post)


def _sortable_post_time(post: dict[str, Any]) -> int:
    import importlib

    mod = importlib.import_module("mattermost_server")
    return mod._sortable_post_time(post)


def _display_seeded_sender(sender: str) -> str:
    import importlib

    mod = importlib.import_module("mattermost_server")
    return mod._display_seeded_sender(sender)


def test_post_sender_prefers_machine_usable_seeded_sender() -> None:
    post = {
        "user_id": "admin-id",
        "props": {"privacylens_sender": "michael"},
    }

    assert _post_sender(post, {"admin-id": "admin"}) == "michael"


def test_post_to_message_exposes_seeded_display_name_separately() -> None:
    post = {
        "id": "post-id",
        "user_id": "admin-id",
        "message": "hello",
        "props": {
            "privacylens_sender": "aunt_linda",
            "privacylens_sender_display": "Aunt Linda",
        },
    }

    assert _post_to_message(post, {"admin-id": "admin"}) == {
        "message_id": "post-id",
        "sender": "aunt_linda",
        "sender_display": "Aunt Linda",
        "time": "",
        "text": "hello",
    }


def test_post_sender_marks_unresolved_seed_sender_as_unknown() -> None:
    post = {
        "user_id": "admin-id",
        "props": {"privacylens_sender_unknown": True},
    }

    assert _post_sender(post, {"admin-id": "admin"}) == ""


def test_post_sender_falls_back_to_resolved_username() -> None:
    post = {"user_id": "user-id"}

    assert _post_sender(post, {"user-id": "emily"}) == "emily"


def test_display_seeded_sender_normalizes_privacylens_user_id() -> None:
    assert _display_seeded_sender("dave_thompson_857") == "Dave Thompson"


def test_display_seeded_sender_keeps_group_id() -> None:
    assert _display_seeded_sender("group_94502") == "group_94502"


def test_post_time_prefers_seeded_epoch_ms() -> None:
    post = {"create_at": 999, "props": {"privacylens_time": "1645442700000"}}

    assert _post_time(post) == 1645442700000
    assert _sortable_post_time(post) == 1645442700000
