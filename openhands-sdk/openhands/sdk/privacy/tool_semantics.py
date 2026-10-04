from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DEFAULT_WRITE_CONTENT_FIELDS = ("message", "body", "content", "markdown")
DEFAULT_RECIPIENT_ARG_FIELDS = (
    "recipient",
    "to",
    "cc",
    "bcc",
    "channel",
    "user",
    "username",
)


@dataclass(frozen=True, slots=True)
class ToolPrivacySemantics:
    """Privacy-relevant semantics for a tool.

    This keeps policy-neutral tool facts out of the Agent loop: where outgoing
    content lives, which arguments identify the audience, whether a read-only
    result is content-access evidence worth extracting from, and any
    channel-visibility hint.
    """

    content_fields: tuple[str, ...] = DEFAULT_WRITE_CONTENT_FIELDS
    recipient_fields: tuple[str, ...] = DEFAULT_RECIPIENT_ARG_FIELDS
    audience_hint: str | None = None
    extract_from_observation: bool = True


ToolPrivacyMatchMode = Literal["exact", "fragment", "operation", "suffix"]


@dataclass(frozen=True, slots=True)
class ToolPrivacySemanticsRule:
    """A matcher rule for resolving privacy semantics from a tool name."""

    matcher: str
    semantics: ToolPrivacySemantics
    match: ToolPrivacyMatchMode = "fragment"

    def matches(self, tool_name: str) -> bool:
        if self.match == "exact":
            return tool_name == self.matcher
        if self.match == "operation":
            return self.matcher in tool_name.split("_")
        if self.match == "suffix":
            return tool_name.endswith(self.matcher)
        return self.matcher in tool_name


DEFAULT_TOOL_PRIVACY_SEMANTICS = ToolPrivacySemantics()

_BUILT_IN_RULES: tuple[ToolPrivacySemanticsRule, ...] = (
    # Content reads that happen to be named with a get_/list_ verb. These MUST
    # come before the blunt search/list/get operation rules below, because the
    # first matching rule wins.
    #
    # The operation rules exist to skip index/metadata listings, which return
    # snippets and would flood the inventory with duplicates. But a tool called
    # `get_page` or `list_messages` returns the record body -- it is the primary
    # content read, and excluding it leaves the read boundary with nothing to
    # extract, so the inventory ends up built from whatever defaults to
    # extracting (a finish/think observation, or the write itself) *after* the
    # write it exists to gate. Matched by suffix so both the server-prefixed
    # (`bookstack_get_page`) and bare (`get_page`) forms resolve identically;
    # the sibling metadata tools (`list_pages`, `list_users`, `list_contacts`,
    # `list_channels`, `list_events`, `get_user_info`, `get_file_image`) do not
    # share these suffixes and stay excluded.
    *(
        ToolPrivacySemanticsRule(
            matcher=suffix,
            semantics=ToolPrivacySemantics(extract_from_observation=True),
            match="suffix",
        )
        for suffix in (
            "get_page",
            "get_event",
            "get_channel_history",
            "get_profile",
            "get_file_metadata",
            "list_messages",
            "list_user_posts",
        )
    ),
    ToolPrivacySemanticsRule(
        matcher="search",
        semantics=ToolPrivacySemantics(extract_from_observation=False),
        match="operation",
    ),
    ToolPrivacySemanticsRule(
        matcher="list",
        semantics=ToolPrivacySemantics(extract_from_observation=False),
        match="operation",
    ),
    ToolPrivacySemanticsRule(
        matcher="get",
        semantics=ToolPrivacySemantics(extract_from_observation=False),
        match="operation",
    ),
    ToolPrivacySemanticsRule(
        matcher="create_post",
        semantics=ToolPrivacySemantics(
            audience_hint=(
                "PUBLIC social-media post "
                "(visible to all followers and potentially the open internet)"
            ),
        ),
    ),
    ToolPrivacySemanticsRule(
        matcher="send_channel_message",
        semantics=ToolPrivacySemantics(
            audience_hint="PUBLIC team channel (visible to all channel members)",
        ),
    ),
    ToolPrivacySemanticsRule(
        matcher="create_page",
        semantics=ToolPrivacySemantics(
            audience_hint="shared wiki page (visible to all wiki users)",
        ),
    ),
    ToolPrivacySemanticsRule(
        matcher="update_page",
        semantics=ToolPrivacySemantics(
            audience_hint="shared wiki page (visible to all wiki users)",
        ),
    ),
)
_REGISTERED_RULES: list[ToolPrivacySemanticsRule] = []


def register_tool_privacy_semantics(rule: ToolPrivacySemanticsRule) -> None:
    """Register tool semantics for plugin- or app-specific tools.

    Later registrations take precedence over earlier custom registrations and
    over built-in defaults.
    """
    _REGISTERED_RULES.append(rule)


def resolve_tool_privacy_semantics(tool_name: str) -> ToolPrivacySemantics:
    """Resolve the privacy semantics for a tool name."""
    for rule in reversed(_REGISTERED_RULES):
        if rule.matches(tool_name):
            return rule.semantics
    for rule in _BUILT_IN_RULES:
        if rule.matches(tool_name):
            return rule.semantics
    return DEFAULT_TOOL_PRIVACY_SEMANTICS
