from openhands.sdk.event import ActionEvent
from openhands.sdk.llm import MessageToolCall
from openhands.sdk.mcp.definition import MCPToolAction
from openhands.sdk.privacy.tool_context import (
    derive_recipient_descriptor,
    derive_transmission_context,
    extract_write_content,
    should_extract_information_flows,
)
from openhands.sdk.privacy.tool_semantics import (
    DEFAULT_TOOL_PRIVACY_SEMANTICS,
    ToolPrivacySemantics,
    ToolPrivacySemanticsRule,
    register_tool_privacy_semantics,
    resolve_tool_privacy_semantics,
)


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


def test_unknown_tool_uses_default_privacy_semantics():
    semantics = resolve_tool_privacy_semantics("custom_send")

    assert semantics == DEFAULT_TOOL_PRIVACY_SEMANTICS


def test_builtin_discovery_tools_skip_flow_extraction():
    assert not should_extract_information_flows("bookstack_search_pages")
    assert not should_extract_information_flows("bookstack_list_pages")
    assert not should_extract_information_flows("rocketchat_list_channels")
    assert should_extract_information_flows("bookstack_read_page")
    # `get_page` is NOT a discovery tool: it returns the page body, and it is
    # what BookStack renamed `read_page` to. Treating it as discovery left the
    # read boundary with nothing to extract -- see
    # test_get_and_list_named_content_reads_still_extract.
    assert should_extract_information_flows("bookstack_get_page")


def test_builtin_public_audience_hint_is_preserved():
    action_event = _action_event(
        "gotosocial_create_post",
        {"content": "hello"},
    )

    descriptor = derive_recipient_descriptor(action_event)

    assert descriptor == (
        "PUBLIC social-media post "
        "(visible to all followers and potentially the open internet) "
        "via gotosocial_create_post"
    )


def test_registered_semantics_override_content_and_recipient_fields():
    register_tool_privacy_semantics(
        ToolPrivacySemanticsRule(
            matcher="unit_private_note_tool",
            match="exact",
            semantics=ToolPrivacySemantics(
                content_fields=("text",),
                recipient_fields=("audience",),
                audience_hint="private note",
            ),
        )
    )
    action_event = _action_event(
        "unit_private_note_tool",
        {
            "message": "wrong field",
            "text": "right field",
            "audience": "Dana",
        },
    )

    assert extract_write_content(action_event) == "right field"
    assert derive_recipient_descriptor(action_event) == (
        "private note via unit_private_note_tool (audience=Dana)"
    )


def test_transmission_context_includes_task_purpose_and_channel():
    register_tool_privacy_semantics(
        ToolPrivacySemanticsRule(
            matcher="unit_private_note_tool_context",
            match="exact",
            semantics=ToolPrivacySemantics(
                content_fields=("text",),
                recipient_fields=("audience",),
                audience_hint="private note",
            ),
        )
    )
    action_event = _action_event(
        "unit_private_note_tool_context",
        {"audience": "Dana", "text": "right field"},
    )

    context = derive_transmission_context(
        action_event,
        principal="Jane",
        task_purpose="Send Dana the update.",
        recipient_role="colleague",
    )

    assert context.data_recipient == "audience=Dana"
    assert context.transmission_channel == "private note"
    assert context.principal == "Jane"
    assert context.task_purpose == "Send Dana the update."
    assert context.recipient_role == "colleague"
    assert context.recipient_descriptor == (
        "private note via unit_private_note_tool_context (audience=Dana)"
    )


def test_get_and_list_named_content_reads_still_extract():
    """A content read named with a get_/list_ verb must remain an extraction
    target, with NO app-specific registration.

    This is the container's view: the audit runs inside the agent-server image,
    which has only the built-in rules. When the MCP servers renamed their
    content reads (``read_page`` -> ``get_page``, ``read_messages`` ->
    ``list_messages``), the blunt ``get``/``list`` operation rules started
    excluding them, so the read boundary extracted nothing from real reads and
    the inventory got built from a finish/think observation -- after the write it
    exists to gate.
    """
    for tool in (
        "bookstack_get_page",
        "mattermost_list_messages",
        "rocketchat_get_channel_history",
        "radicale_get_event",
        "gotosocial_get_profile",
        "gotosocial_list_user_posts",
        "google_drive_get_file_metadata",
        # legacy names, still used by archived runs
        "bookstack_read_page",
        "mattermost_read_messages",
        "mailpit_read_email",
    ):
        assert should_extract_information_flows(tool), tool
        assert should_extract_information_flows(tool.split("_", 1)[1]), tool

    # Index / metadata reads return snippets and must stay excluded, or the
    # inventory fills with duplicates of the real reads.
    for tool in (
        "bookstack_search_pages",
        "bookstack_list_pages",
        "mattermost_list_users",
        "mattermost_search_messages",
        "mailpit_search_emails",
        "mailpit_list_contacts",
        "rocketchat_list_channels",
        "rocketchat_get_user_info",
        "radicale_list_events",
        "gotosocial_search_posts",
        "google_drive_search_files",
        "google_drive_get_file_image",
    ):
        assert not should_extract_information_flows(tool), tool
