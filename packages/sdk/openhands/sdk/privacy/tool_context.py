from __future__ import annotations

import json
from typing import Any

from openhands.sdk.event import ActionEvent
from openhands.sdk.privacy.flow import TransmissionContext
from openhands.sdk.privacy.tool_semantics import (
    DEFAULT_RECIPIENT_ARG_FIELDS,
    DEFAULT_WRITE_CONTENT_FIELDS,
    resolve_tool_privacy_semantics,
)
from openhands.sdk.tool.tool import PRIVACY_CONTEXT_FIELDS


WRITE_CONTENT_FIELDS = DEFAULT_WRITE_CONTENT_FIELDS
RECIPIENT_ARG_FIELDS = DEFAULT_RECIPIENT_ARG_FIELDS


def extract_privacy_context(
    arguments: dict[str, Any],
    read_only_tool: bool,
) -> dict[str, str | None]:
    """Extract and remove CI privacy context fields from tool arguments."""
    result: dict[str, str | None] = {}
    for field_name in PRIVACY_CONTEXT_FIELDS:
        value = arguments.pop(field_name, None)
        result[field_name] = None if value is None else str(value)
    if read_only_tool:
        return {k: None for k in PRIVACY_CONTEXT_FIELDS}
    return result


def should_extract_information_flows(tool_name: str) -> bool:
    """Return whether a read-only result is content evidence for extraction."""
    return resolve_tool_privacy_semantics(tool_name).extract_from_observation


def derive_recipient_descriptor(action_event: ActionEvent) -> str:
    """Build a short audience descriptor from the write tool and arguments.

    The privacy judge uses this to reason about whether the outgoing flow is
    appropriate for its audience. This is derived from public tool fields only;
    no seed-data or benchmark ground truth is injected.
    """
    tool_name = action_event.tool_name
    semantics = resolve_tool_privacy_semantics(tool_name)
    action = action_event.action
    data = getattr(action, "data", None) if action is not None else None
    parts: list[str] = []
    if isinstance(data, dict):
        for field_name in semantics.recipient_fields:
            value = data.get(field_name)
            if value:
                parts.append(f"{field_name}={value}")

    if semantics.audience_hint:
        if parts:
            return f"{semantics.audience_hint} via {tool_name} ({', '.join(parts)})"
        return f"{semantics.audience_hint} via {tool_name}"
    if parts:
        return f"{tool_name} ({', '.join(parts)})"
    return tool_name


def derive_transmission_context(
    action_event: ActionEvent,
    *,
    principal: str | None = None,
    task_purpose: str | None = None,
    recipient_role: str | None = None,
) -> TransmissionContext:
    """Build structured CI context from a write-tool action.

    The context is derived from public tool call fields plus caller-supplied
    task metadata. It does not inspect benchmark oracle labels or seed ground
    truth.
    """
    tool_name = action_event.tool_name
    semantics = resolve_tool_privacy_semantics(tool_name)
    action = action_event.action
    data = getattr(action, "data", None) if action is not None else None
    recipients: list[str] = []
    if isinstance(data, dict):
        for field_name in semantics.recipient_fields:
            value = data.get(field_name)
            if value:
                recipients.append(f"{field_name}={value}")

    channel = semantics.audience_hint or tool_name
    return TransmissionContext(
        data_recipient=", ".join(recipients) if recipients else channel,
        transmission_channel=channel,
        principal=principal or "",
        task_purpose=task_purpose or "",
        recipient_role=recipient_role,
        tool_name=tool_name,
    )


def extract_write_content(action_event: ActionEvent) -> str:
    """Extract outgoing message content from action arguments.

    MCP tools store args in ``action.data`` dict. The write-content field varies
    by tool. Try common content fields first; fall back to serializing all args.
    """
    action = action_event.action
    if action is None:
        return ""
    data = getattr(action, "data", None)
    if isinstance(data, dict):
        semantics = resolve_tool_privacy_semantics(action_event.tool_name)
        for field in semantics.content_fields:
            if field in data and isinstance(data[field], str):
                return data[field]
        return json.dumps(data, ensure_ascii=False)
    return json.dumps(action.model_dump(), ensure_ascii=False)
