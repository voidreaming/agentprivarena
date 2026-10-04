"""Privacy semantics registration for PrivacyLens-Live MCP tools."""

from __future__ import annotations

from openhands.sdk.privacy.tool_semantics import (
    ToolPrivacySemantics,
    ToolPrivacySemanticsRule,
    register_tool_privacy_semantics,
)


_REGISTERED = False


def _exact_rules(
    service: str,
    tool: str,
    semantics: ToolPrivacySemantics,
) -> tuple[ToolPrivacySemanticsRule, ToolPrivacySemanticsRule]:
    return (
        ToolPrivacySemanticsRule(
            matcher=tool,
            semantics=semantics,
            match="exact",
        ),
        ToolPrivacySemanticsRule(
            matcher=f"{service}_{tool}",
            semantics=semantics,
            match="exact",
        ),
    )


def _exact_rules_for_tools(
    service: str,
    tools: tuple[str, ...],
    semantics: ToolPrivacySemantics,
) -> tuple[ToolPrivacySemanticsRule, ...]:
    return tuple(
        rule for tool in tools for rule in _exact_rules(service, tool, semantics)
    )


_NO_EXTRACT = ToolPrivacySemantics(extract_from_observation=False)
_EXTRACT = ToolPrivacySemantics(extract_from_observation=True)

_RULES: tuple[ToolPrivacySemanticsRule, ...] = (
    *_exact_rules(
        "mailpit",
        "send_email",
        ToolPrivacySemantics(
            content_fields=("body",),
            recipient_fields=("to", "cc", "bcc"),
            audience_hint="private email recipients",
        ),
    ),
    *_exact_rules(
        "mattermost",
        "send_message",
        ToolPrivacySemantics(
            content_fields=("message",),
            recipient_fields=("recipient",),
            audience_hint=(
                "private Mattermost direct message "
                "(visible only to the named recipient)"
            ),
        ),
    ),
    *_exact_rules(
        "rocketchat",
        "send_channel_message",
        ToolPrivacySemantics(
            content_fields=("message",),
            recipient_fields=("channel",),
            audience_hint=(
                "RocketChat channel or direct-message target named by `channel`; "
                "visibility depends on whether the target is a channel or username"
            ),
        ),
    ),
    *_exact_rules(
        "gotosocial",
        "create_post",
        ToolPrivacySemantics(
            content_fields=("content",),
            recipient_fields=(),
            audience_hint=(
                "PUBLIC GoToSocial post "
                "(visible to followers and potentially the open internet)"
            ),
        ),
    ),
    *_exact_rules(
        "bookstack",
        "create_page",
        ToolPrivacySemantics(
            content_fields=("markdown",),
            recipient_fields=("name",),
            audience_hint="shared BookStack wiki page (visible to all wiki users)",
        ),
    ),
    *_exact_rules(
        "bookstack",
        "update_page",
        ToolPrivacySemantics(
            content_fields=("markdown",),
            recipient_fields=("page_id",),
            audience_hint="shared BookStack wiki page (visible to all wiki users)",
        ),
    ),
    *_exact_rules_for_tools(
        "bookstack",
        ("search_pages", "list_pages"),
        _NO_EXTRACT,
    ),
    *_exact_rules_for_tools(
        "bookstack",
        ("read_page",),
        _EXTRACT,
    ),
    *_exact_rules_for_tools(
        "mattermost",
        ("search_messages", "list_users"),
        _NO_EXTRACT,
    ),
    *_exact_rules_for_tools(
        "mattermost",
        ("read_messages",),
        _EXTRACT,
    ),
    *_exact_rules_for_tools(
        "rocketchat",
        (
            "search_messages",
            "list_channels",
            "search_users",
            "get_user_info",
        ),
        _NO_EXTRACT,
    ),
    *_exact_rules_for_tools(
        "rocketchat",
        ("read_channel_history",),
        _EXTRACT,
    ),
    *_exact_rules_for_tools(
        "mailpit",
        ("search_emails", "list_contacts"),
        _NO_EXTRACT,
    ),
    *_exact_rules_for_tools(
        "mailpit",
        ("read_email",),
        _EXTRACT,
    ),
    *_exact_rules_for_tools(
        "gotosocial",
        ("search_users", "search_posts"),
        _NO_EXTRACT,
    ),
    *_exact_rules_for_tools(
        "gotosocial",
        ("read_profile", "read_user_posts"),
        _EXTRACT,
    ),
    *_exact_rules_for_tools(
        "radicale",
        ("search_events", "list_events"),
        _NO_EXTRACT,
    ),
    *_exact_rules_for_tools(
        "radicale",
        ("read_event",),
        _EXTRACT,
    ),
    *_exact_rules_for_tools(
        "google_drive",
        ("search_files", "get_file_image"),
        _NO_EXTRACT,
    ),
    *_exact_rules_for_tools(
        "google_drive",
        ("get_file_metadata", "describe_image"),
        _EXTRACT,
    ),
    # The MCP servers renamed their content-read tools after the paper runs
    # (`read_page` -> `get_page`, `read_messages` -> `list_messages`,
    # `read_channel_history` -> `get_channel_history`, `read_event` ->
    # `get_event`, `read_profile` -> `get_profile`, `read_user_posts` ->
    # `list_user_posts`). Only the legacy names were registered as extraction
    # targets, and the SDK's built-in rules treat any tool whose name contains
    # the operation token `get`/`list`/`search` as NON-extracting -- so under the
    # new names the read boundary extracted nothing from real content reads, and
    # the inventory got built from whatever defaulted to True (`finish`,
    # `think`, write tools), i.e. after the write it was supposed to gate.
    # Registered last so these exact matches win over the built-in operation
    # rules; the legacy names above stay registered so replaying old runs is
    # unaffected. `mailpit_read_email` never changed.
    *_exact_rules_for_tools("bookstack", ("get_page",), _EXTRACT),
    *_exact_rules_for_tools("mattermost", ("list_messages",), _EXTRACT),
    *_exact_rules_for_tools("rocketchat", ("get_channel_history",), _EXTRACT),
    *_exact_rules_for_tools("radicale", ("get_event",), _EXTRACT),
    *_exact_rules_for_tools(
        "gotosocial",
        ("get_profile", "list_user_posts"),
        _EXTRACT,
    ),
)


def register_privacylens_tool_privacy_semantics() -> None:
    """Register PrivacyLens-Live tool semantics for privacy audit runs."""
    global _REGISTERED
    if _REGISTERED:
        return
    for rule in _RULES:
        register_tool_privacy_semantics(rule)
    _REGISTERED = True
