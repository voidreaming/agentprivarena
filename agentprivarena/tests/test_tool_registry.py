"""Tests for centralized service/tool metadata."""

from __future__ import annotations

from agentprivarena.runner.tool_registry import (
    SERVICE_TOOL_SPECS,
    is_agent_read_tool,
    is_final_action_tool,
    is_multimodal_tool,
    is_read_tool,
    is_write_tool,
    known_service_names,
    service_spec,
)


def test_registry_lists_current_mcp_services() -> None:
    assert known_service_names() == (
        "bookstack",
        "mattermost",
        "rocketchat",
        "mailpit",
        "gotosocial",
        "radicale",
        "google_drive",
    )
    assert set(SERVICE_TOOL_SPECS) == set(known_service_names())


def test_registry_exposes_prefixed_tool_roles() -> None:
    assert is_read_tool("mailpit_read_email")
    assert is_read_tool("bookstack_read_page")
    assert is_write_tool("bookstack_update_page")
    assert is_final_action_tool("mailpit_send_email")
    assert not is_final_action_tool("bookstack_update_page")
    assert not is_agent_read_tool("bookstack_update_page", forced_read=False)
    assert not is_read_tool("bookstack_get_page")
    assert not is_read_tool("mattermost_list_messages")
    assert not is_agent_read_tool("bookstack_read_page", forced_read=True)
    assert is_agent_read_tool("bookstack_read_page", forced_read=False)


def test_registry_marks_google_drive_multimodal_tools() -> None:
    spec = service_spec("google_drive")

    assert spec.multimodal_tools == frozenset({"get_file_image", "describe_image"})
    assert spec.supports_forced_read is True
    assert is_multimodal_tool("google_drive_get_file_image")
    assert is_multimodal_tool("describe_image")
