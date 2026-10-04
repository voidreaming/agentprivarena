"""Tests for AgentPrivArena privacy tool semantics registration."""

from __future__ import annotations

import pytest

import agentprivarena.runner.privacy_tool_semantics as pl_tool_semantics
import openhands.sdk.privacy.tool_semantics as sdk_tool_semantics
from agentprivarena.config import Config
from agentprivarena.runner.agent_profile import AgentProfile
from agentprivarena.runner.privacy_tool_semantics import (
    register_agentprivarena_tool_privacy_semantics,
)
from agentprivarena.runner.task_spec import TaskSpec
from openhands.sdk.event import ActionEvent
from openhands.sdk.llm import MessageToolCall
from openhands.sdk.mcp.definition import MCPToolAction
from openhands.sdk.privacy.tool_context import (
    derive_recipient_descriptor,
    extract_write_content,
    should_extract_information_flows,
)


@pytest.fixture(autouse=True)
def isolate_tool_privacy_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sdk_tool_semantics, "_REGISTERED_RULES", [])
    monkeypatch.setattr(pl_tool_semantics, "_REGISTERED", False)


def _action_event(tool_name: str, data: dict) -> ActionEvent:
    return ActionEvent(
        thought=[],
        action=MCPToolAction(data=data),
        tool_name=tool_name,
        tool_call_id=f"call-{tool_name}",
        tool_call=MessageToolCall(
            id=f"call-{tool_name}",
            name=tool_name,
            arguments="{}",
            origin="completion",
        ),
        llm_response_id=f"response-{tool_name}",
    )


def test_agentprivarena_mailpit_semantics() -> None:
    register_agentprivarena_tool_privacy_semantics()
    action_event = _action_event(
        "mailpit_send_email",
        {
            "to": "bob@example.com",
            "cc": "carol@example.com",
            "subject": "Update",
            "body": "Alice has flu.",
        },
    )

    assert extract_write_content(action_event) == "Alice has flu."
    assert derive_recipient_descriptor(action_event) == (
        "private email recipients via mailpit_send_email "
        "(to=bob@example.com, cc=carol@example.com)"
    )


def test_agentprivarena_channel_semantics_distinguish_public_and_private_tools() -> (
    None
):
    register_agentprivarena_tool_privacy_semantics()
    mattermost_event = _action_event(
        "mattermost_send_message",
        {"recipient": "dana", "message": "hello"},
    )
    rocketchat_event = _action_event(
        "rocketchat_send_channel_message",
        {"channel": "advertiser", "message": "hello"},
    )

    assert derive_recipient_descriptor(mattermost_event) == (
        "private Mattermost direct message (visible only to the named recipient) "
        "via mattermost_send_message (recipient=dana)"
    )
    assert derive_recipient_descriptor(rocketchat_event) == (
        "RocketChat channel or direct-message target named by `channel`; "
        "visibility depends on whether the target is a channel or username "
        "via rocketchat_send_channel_message (channel=advertiser)"
    )


def test_agentprivarena_shared_page_and_social_post_semantics() -> None:
    register_agentprivarena_tool_privacy_semantics()
    bookstack_event = _action_event(
        "bookstack_create_page",
        {"name": "Policy Note", "markdown": "Alice has flu."},
    )
    gotosocial_event = _action_event(
        "gotosocial_create_post",
        {"content": "Alice has flu."},
    )

    assert extract_write_content(bookstack_event) == "Alice has flu."
    assert derive_recipient_descriptor(bookstack_event) == (
        "shared BookStack wiki page (visible to all wiki users) "
        "via bookstack_create_page (name=Policy Note)"
    )
    assert derive_recipient_descriptor(gotosocial_event) == (
        "PUBLIC GoToSocial post "
        "(visible to followers and potentially the open internet) "
        "via gotosocial_create_post"
    )


def test_agentprivarena_google_drive_semantics_extract_image_asset_metadata() -> None:
    register_agentprivarena_tool_privacy_semantics()

    assert not should_extract_information_flows("google_drive_search_files")
    assert should_extract_information_flows("google_drive_get_file_metadata")
    assert not should_extract_information_flows("google_drive_get_file_image")
    assert should_extract_information_flows("google_drive_describe_image")


def test_agentprivarena_extracts_only_content_access_observations() -> None:
    register_agentprivarena_tool_privacy_semantics()

    assert not should_extract_information_flows("bookstack_search_pages")
    assert should_extract_information_flows("bookstack_read_page")
    assert not should_extract_information_flows("mattermost_search_messages")
    assert not should_extract_information_flows("mattermost_list_users")
    assert should_extract_information_flows("mattermost_read_messages")
    assert not should_extract_information_flows("radicale_search_events")
    assert should_extract_information_flows("radicale_read_event")
    assert not should_extract_information_flows("rocketchat_search_messages")
    assert should_extract_information_flows("rocketchat_read_channel_history")
    assert should_extract_information_flows("mailpit_read_email")


def test_current_content_read_tool_names_are_extraction_targets() -> None:
    """The MCP servers renamed their content reads after the paper runs
    (``read_page`` -> ``get_page`` etc.). Only the legacy names were registered,
    and the SDK's built-in rules treat any ``get``/``list``/``search`` tool as
    NON-extracting -- so the read boundary silently stopped extracting from real
    reads and built its inventory from whatever defaulted to True (``finish``,
    ``think``, write tools), i.e. after the write it exists to gate.
    """
    register_agentprivarena_tool_privacy_semantics()

    for tool in (
        "bookstack_get_page",
        "mattermost_list_messages",
        "rocketchat_get_channel_history",
        "radicale_get_event",
        "gotosocial_get_profile",
        "gotosocial_list_user_posts",
        "mailpit_read_email",
    ):
        assert should_extract_information_flows(tool), tool
        # single-server tasks get the unprefixed form
        assert should_extract_information_flows(tool.split("_", 1)[1]), tool

    # Metadata/index reads must still be excluded: they return snippets and
    # would flood the inventory with duplicates of the real reads.
    for tool in (
        "bookstack_search_pages",
        "bookstack_list_pages",
        "mattermost_search_messages",
        "mattermost_list_users",
        "mailpit_search_emails",
        "gotosocial_search_posts",
        "radicale_search_events",
        "google_drive_search_files",
    ):
        assert not should_extract_information_flows(tool), tool


def test_privacy_agent_profile_registers_tool_semantics(monkeypatch) -> None:
    calls = []

    def fake_register() -> None:
        calls.append("registered")

    monkeypatch.setattr(
        "agentprivarena.runner.privacy_tool_semantics."
        "register_agentprivarena_tool_privacy_semantics",
        fake_register,
    )
    profile = AgentProfile(prompt_variant="ci_audit", enable_privacy_analyzer=True)
    task = TaskSpec(
        name="sample",
        user_instruction="Send an email",
        dependencies=["mailpit"],
        user_name="Jane",
    )

    agent_kwargs = profile.build_agent_kwargs(
        task=task,
        config=Config(),
        mcp_config={},
    )

    assert calls == ["registered"]
    assert agent_kwargs["privacy_principal"] == "Jane"
    assert agent_kwargs["privacy_task_purpose"] == "Send an email"
