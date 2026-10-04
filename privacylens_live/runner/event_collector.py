"""Collect events from agent conversations and extract structured results.

The OpenHands SDK streams ``ActionEvent`` / ``ObservationEvent`` /
``MessageEvent`` / ``AgentErrorEvent`` / ``ConversationErrorEvent`` to a
callback. This module collects them and exposes typed extractors that
build the structured ``ScenarioResult`` the runner persists to disk.

The MCP observation payload is wrapped twice: an SDK ``ObservationEvent``
envelope around an MCP ``MCPToolObservation`` envelope around the actual
JSON the tool returned. ``_parse_observation`` peels both layers so the
result file shows the real tool output, not framework noise.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from openhands.sdk.privacy.flow import information_flow_key
from privacylens_live.runner.tool_registry import (
    NON_READ_TOOLS,
    SEND_ACTIONS,
    SEND_PREFIXED,
    is_agent_read_tool,
    is_final_action_tool,
)


logger = logging.getLogger("event_collector")

__all__ = [
    "EventCollector",
    "NON_READ_TOOLS",
    "SEND_ACTIONS",
    "SEND_PREFIXED",
    "ScenarioResult",
    "classify_error_category",
    "render_action_string",
    "render_sanitized_trajectory_string",
    "render_trajectory_string",
]


@dataclass
class ScenarioResult:
    """Structured result of running a single PrivacyLens scenario.

    The fields here are what the result JSON file holds. Raw events go
    in a sidecar file (``<name>.events.json``) so this stays readable.
    """

    name: str
    status: str = "no_action"  # "ok" | "no_action" | "error"
    final_action: dict | None = None
    expected_final_action_type: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    # Clarification turns: each entry is
    # ``{round, request, response}`` capturing a time the agent ended
    # on a text question and the runner auto-replied to push it to
    # commit. Empty for runs where the agent committed on the first
    # turn.
    clarifications: list[dict] = field(default_factory=list)
    final_message: str | None = None
    sensitive_info_items: list[str] = field(default_factory=list)
    # Privacy judgments collected at write-tool pre-execution, one per write
    # attempt the judge saw. Each entry carries the tool name, the action-
    # level decision (pass / abstract / block), the rationale, and the
    # transmitted flows with per-flow disposition. Empty unless the L3
    # ``ci_audit`` privacy analyzer was enabled for the run.
    privacy_judgments: list[dict] = field(default_factory=list)
    # Privacy flows extracted from read-tool observations. These are the
    # inventory items later passed to the write-action judge.
    privacy_flows: list[dict] = field(default_factory=list)
    # Audit instructions injected proactively by the privacy auditor at the
    # end of read batches. Each entry is the raw steering text plus the
    # preceding/following tool_call_ids so analysis scripts can place it in
    # the trace. Empty unless the L3 ``ci_audit`` analyzer was active.
    audit_instructions: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    read_policy: str = "natural"
    forced_read_success: bool | None = None
    error_category: str | None = None
    error: str | None = None
    # Raw events — populated in memory, written to a sidecar file by
    # the runner, never inlined into the main result file.
    events: list[dict] = field(default_factory=list)


def _parse_observation(obs_event: dict) -> tuple[Any, bool]:
    """Pull the actual tool output out of an MCPToolObservation envelope.

    The wire shape from FastMCP is::

        {
          "kind": "ObservationEvent",
          "observation": {
            "kind": "MCPToolObservation",
            "is_error": false,
            "content": [
              {"type": "text", "text": "[Tool 'X' executed.]"},
              {"type": "text", "text": "{\"pages\": [...]}"}
            ]
          }
        }

    Returns ``(parsed_value, is_error)`` where ``parsed_value`` is the
    JSON-decoded tool output if it parses, otherwise the raw text. The
    ``[Tool 'X' executed.]`` framing line is skipped.
    """
    if obs_event.get("kind") == "UserRejectObservation":
        return obs_event.get("rejection_reason", "Tool action rejected."), True
    obs = obs_event.get("observation")
    if not isinstance(obs, dict):
        return obs, False

    is_error = bool(obs.get("is_error", False))
    content = obs.get("content")
    if not isinstance(content, list):
        return obs, is_error

    # Walk content items and grab the first text payload that isn't the
    # framing line. Fall through to the last text item if everything
    # looks like framing.
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = item.get("text", "")
        if not text or text.startswith("[Tool '"):
            continue
        try:
            return json.loads(text), is_error
        except json.JSONDecodeError:
            return text, is_error

    for item in reversed(content):
        if isinstance(item, dict) and item.get("type") == "text":
            return item.get("text", ""), is_error

    return obs, is_error


def render_action_string(action: dict | None) -> str:
    """Format a structured final_action dict as the legacy PrivacyLens string.

    The PrivacyLens evaluator's prompts and rule-based scanner both
    expect the ``Action: X\\nAction Input: {...}`` form. This builds
    it from the structured representation so we have a single source
    of truth.
    """
    if not action:
        return ""
    return (
        f"Action: {action['tool']}\nAction Input: {json.dumps(action.get('input', {}))}"
    )


def render_trajectory_string(tool_calls: list[dict]) -> str:
    """Format a structured tool_calls list as the legacy trajectory string."""
    parts = []
    for call in tool_calls:
        obs = call["observation"]
        obs_text = json.dumps(obs) if isinstance(obs, (dict, list)) else str(obs)
        parts.append(
            f"Action: {call['tool']}\n"
            f"Action Input: {json.dumps(call['input'])}\n"
            f"Observation: {obs_text}"
        )
    return "\n\n".join(parts)


# Fields whose values contain long sensitive text (email bodies, page
# content, message text) that trigger Azure content-policy filters.
# Stripping these from observations keeps enough structure for the
# helpfulness judge while avoiding filter hits.
_SENSITIVE_FIELDS = frozenset(
    {
        "markdown",
        "body",
        "html",
        "content",
        "text",
        "message",
        "description",
        "note",
        "bio",
    }
)


def _sanitize_obs(obs: Any) -> Any:
    """Strip long text fields from an observation, keeping structure.

    The helpfulness judge needs to know *what* was found (titles, names,
    subjects, senders, IDs) but not the full body text. This avoids
    triggering Azure content-policy filters while preserving the
    structural cues needed to evaluate whether the agent's final action
    matches what it found.
    """
    if isinstance(obs, dict):
        sanitized = {}
        for k, v in obs.items():
            if k in _SENSITIVE_FIELDS and isinstance(v, str) and len(v) > 80:
                sanitized[k] = f"[{len(v)} chars]"
            elif isinstance(v, (dict, list)):
                sanitized[k] = _sanitize_obs(v)
            else:
                sanitized[k] = v
        return sanitized
    if isinstance(obs, list):
        return [_sanitize_obs(item) for item in obs]
    return obs


def render_sanitized_trajectory_string(tool_calls: list[dict]) -> str:
    """Like render_trajectory_string but with sensitive text stripped.

    Replaces long body/markdown/content fields in observations with
    ``[N chars]`` placeholders. Keeps tool names, inputs, structural
    fields (IDs, names, subjects, senders), and short values intact.
    """
    parts = []
    for call in tool_calls:
        obs = _sanitize_obs(call["observation"])
        obs_text = json.dumps(obs) if isinstance(obs, (dict, list)) else str(obs)
        parts.append(
            f"Action: {call['tool']}\n"
            f"Action Input: {json.dumps(call['input'])}\n"
            f"Observation: {obs_text}"
        )
    return "\n\n".join(parts)


def _extract_text_content(message_obj: Any) -> str:
    """Pull plain text out of an llm_message ``content`` field.

    The SDK serializes message content as either a plain string or a
    list of ``{"type": "text", "text": "..."}`` blocks. This collapses
    both shapes into a single string.
    """
    if isinstance(message_obj, str):
        return message_obj
    if not isinstance(message_obj, dict):
        return ""
    content = message_obj.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    return ""


def classify_error_category(error_text: str | None) -> str | None:
    """Return a stable coarse category for known platform/runtime failures."""
    if not error_text:
        return None
    normalized = error_text.lower()
    if (
        "content_filter" in normalized
        or "content management policy" in normalized
        or "jailbreak" in normalized
    ):
        return "provider_content_filter"
    return None


@runtime_checkable
class _ModelDumpEvent(Protocol):
    """Minimal protocol for SDK/Pydantic event objects."""

    def model_dump(self) -> dict[str, Any]: ...


def _event_to_dict(event: Any) -> dict[str, Any]:
    """Convert SDK event objects and legacy dict events to plain dicts."""
    if isinstance(event, dict):
        return event
    if isinstance(event, _ModelDumpEvent):
        return event.model_dump()
    return {"type": type(event).__name__, "data": str(event)}


def _event_identity(event: dict[str, Any]) -> tuple[str, str]:
    """Stable identity used when merging callback and REST event streams."""
    event_id = event.get("id")
    if event_id is not None:
        return ("id", str(event_id))
    return ("payload", json.dumps(event, sort_keys=True, default=str))


class EventCollector:
    """Collects conversation events and exposes structured extractors."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def on_event(self, event: Any) -> None:
        """Callback for conversation events."""
        try:
            self.events.append(_event_to_dict(event))
        except Exception as e:
            logger.warning(f"Failed to collect event: {e}")

    def reconcile_from_events(self, events: Iterable[Any]) -> int:
        """Merge authoritative conversation events into the callback stream.

        Remote conversations normally deliver events through WebSocket callbacks.
        If that stream disconnects, the SDK reconciles its own
        ``conversation.state.events`` via REST, but user callbacks are not
        backfilled. Pulling from the authoritative state here keeps result files
        complete without changing the runner's persisted schema.
        """
        known = {_event_identity(event) for event in self.events}
        added = 0
        for raw_event in events:
            try:
                event = _event_to_dict(raw_event)
            except Exception as e:
                logger.warning(f"Failed to reconcile event: {e}")
                continue
            identity = _event_identity(event)
            if identity in known:
                continue
            self.events.append(event)
            known.add(identity)
            added += 1

        if added and all(
            isinstance(event.get("timestamp"), str) for event in self.events
        ):
            self.events.sort(key=lambda event: event["timestamp"])
        return added

    # ── Error surfacing ──

    def extract_error_detail(self) -> str | None:
        """Pull the most recent error event's content, if any.

        The agent server's WebSocket stream emits ``ConversationErrorEvent``
        when the LLM call (or anything else inside the run loop) raises.
        The default visualizer skips it, so without this helper the runner
        only sees ``"Remote conversation ended with error"`` with no body.
        """
        for event in reversed(self.events):
            kind = event.get("kind", "")
            etype = event.get("type", "")
            if "Error" in kind or "error" in etype.lower():
                for key in ("error", "message", "detail", "exception"):
                    if event.get(key):
                        return f"{kind or etype}: {event[key]}"
                return json.dumps(event)[:500]
        return None

    def extract_error_category(self, error_text: str | None = None) -> str | None:
        """Classify known run failures without changing control flow."""
        explicit_category = classify_error_category(error_text)
        if explicit_category is not None:
            return explicit_category
        for event in reversed(self.events):
            category = classify_error_category(json.dumps(event, default=str))
            if category is not None:
                return category
        return None

    # ── Action argument extraction ──

    def _extract_action_args(self, event: dict) -> dict:
        """Pull the tool-call arguments out of an ActionEvent.

        Tries the SDK's current shape (``event.action.data``) first,
        then a few legacy/alternative locations.
        """
        action_obj = event.get("action")
        if isinstance(action_obj, dict):
            data = action_obj.get("data")
            if isinstance(data, dict):
                return data
        for key in ("arguments", "action_input", "tool_input"):
            value = event.get(key)
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
        return {}

    def _action_call_id(self, event: dict) -> str:
        """Best-effort tool_call_id for matching ActionEvent ↔ ObservationEvent."""
        action_obj = event.get("action") or {}
        return action_obj.get("tool_call_id") or event.get("tool_call_id", "")

    # ── Structured extraction ──

    def extract_tool_calls(self) -> list[dict]:
        """Build the structured tool-call timeline.

        Each entry is::

            {
              "step": 1,
              "tool": "bookstack_read_page",
              "input": {"page_id": 6},
              "observation": <parsed JSON or string>,
              "is_error": false
            }

        ActionEvents that have a matching AgentErrorEvent (the SDK's
        signal for validation/exec failures the agent then retried)
        get ``is_error: true`` and the error message in ``observation``,
        so the trace stays honest about what happened.
        """
        # Index observations and errors by tool_call_id for O(1) lookup.
        obs_by_call: dict[str, dict] = {}
        err_by_call: dict[str, dict] = {}
        for event in self.events:
            kind = event.get("kind")
            cid = event.get("tool_call_id", "")
            if kind in ("ObservationEvent", "UserRejectObservation") and cid:
                obs_by_call[cid] = event
            elif kind == "AgentErrorEvent" and cid:
                err_by_call[cid] = event

        calls: list[dict] = []
        step = 0
        for event in self.events:
            if event.get("kind") != "ActionEvent":
                continue
            tool = event.get("tool_name", "")
            if not tool:
                continue
            step += 1
            cid = self._action_call_id(event)
            args = self._extract_action_args(event)
            forced_read = bool(event.get("forced_read", False))

            if cid in err_by_call:
                err_msg = err_by_call[cid].get("error", "validation/exec error")
                calls.append(
                    {
                        "step": step,
                        "tool": tool,
                        "input": args,
                        "observation": err_msg,
                        "is_error": True,
                        "forced_read": forced_read,
                    }
                )
                continue

            obs_event = obs_by_call.get(cid, {})
            obs_value, obs_is_error = _parse_observation(obs_event)
            calls.append(
                {
                    "step": step,
                    "tool": tool,
                    "input": args,
                    "observation": obs_value,
                    "is_error": obs_is_error,
                    "forced_read": forced_read,
                }
            )
            if obs_event.get("kind") == "UserRejectObservation":
                calls[-1]["rejected"] = True
        return calls

    def extract_final_action(self) -> dict | None:
        """Find the last successful send-action call as a structured dict.

        Returns ``None`` if the agent never called a send action (e.g.
        it ended with a clarification question instead). The dict has
        ``tool``, ``input``, ``result``, ``is_error`` so the evaluator
        can read the post body / email body / etc. directly.
        """
        for call in reversed(self.extract_tool_calls()):
            tool = call["tool"]
            if not is_final_action_tool(tool):
                continue
            if call["is_error"] or call["observation"] is None:
                continue
            return {
                "tool": tool,
                "input": call["input"],
                "result": call["observation"],
                "is_error": False,
            }
        # Fallback: if every send attempt failed, surface the last failed
        # one so the result file shows what the agent tried.
        for call in reversed(self.extract_tool_calls()):
            tool = call["tool"]
            if not is_final_action_tool(tool):
                continue
            return {
                "tool": tool,
                "input": call["input"],
                "result": call["observation"],
                "is_error": True,
            }
        return None

    def extract_final_message(self) -> str | None:
        """Return the assistant's final natural-language message, if any.

        This is what the agent says to the user after (or instead of)
        calling its last tool. Useful for diagnosing ``no_action`` runs
        where the agent ended on a clarification question, and for
        seeing the agent's final summary on a successful run.

        The SDK does NOT populate a top-level ``sender`` field on
        ``MessageEvent`` — the role lives at ``llm_message.role``
        ("user" / "assistant"). Filter on that, otherwise our own
        auto-confirmation user message will be returned as "the
        agent's final reply".
        """
        for event in reversed(self.events):
            if event.get("kind") != "MessageEvent":
                continue
            msg = event.get("llm_message", {})
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            text = _extract_text_content(msg)
            if text:
                return text.strip()
        for event in reversed(self.events):
            if event.get("kind") != "ObservationEvent":
                continue
            if event.get("tool_name") != "finish":
                continue
            obs_value, obs_is_error = _parse_observation(event)
            if obs_is_error:
                continue
            if isinstance(obs_value, str) and obs_value.strip():
                return obs_value.strip()
        return None

    def extract_stats(self) -> dict:
        """Aggregate counters for the run."""
        tool_calls = self.extract_tool_calls()
        send_attempts = sum(1 for c in tool_calls if is_final_action_tool(c["tool"]))
        errors_recovered = sum(
            1 for c in tool_calls if c["is_error"] and not c.get("rejected")
        )
        forced_read_tool_call_count = sum(1 for c in tool_calls if c.get("forced_read"))
        forced_read_error_count = sum(
            1 for c in tool_calls if c.get("forced_read") and c["is_error"]
        )
        agent_read_tool_call_count = sum(
            1
            for c in tool_calls
            if is_agent_read_tool(c["tool"], forced_read=bool(c.get("forced_read")))
        )
        judgments = self.extract_privacy_judgments()
        privacy_flows = self.extract_privacy_flows()
        audit_instructions = self.extract_audit_instructions()
        provider_content_filter_errors = (
            1 if self.extract_error_category() == "provider_content_filter" else 0
        )
        judge_interventions = sum(
            1 for j in judgments if j.get("decision") not in (None, "pass")
        )
        return {
            "tool_call_count": len(tool_calls),
            "errors_recovered": errors_recovered,
            "send_action_attempts": send_attempts,
            "forced_read_tool_call_count": forced_read_tool_call_count,
            "forced_read_error_count": forced_read_error_count,
            "agent_read_tool_call_count": agent_read_tool_call_count,
            "privacy_flow_count": len(privacy_flows),
            "judge_interventions": judge_interventions,
            "audit_instructions_emitted": len(audit_instructions),
            "provider_content_filter_errors": provider_content_filter_errors,
            "audit_errors": len(self.extract_audit_errors()),
        }

    def extract_audit_errors(self) -> list[str]:
        """Return reasons the privacy audit failed open during this run.

        The audit degrades to a no-op on exception, so without this a crashed
        audit is indistinguishable from one that approved every action.
        """
        return [
            error
            for event in self.events
            if event.get("kind") == "ObservationEvent"
            and isinstance(error := event.get("privacy_audit_error"), str)
            and error
        ]

    def extract_privacy_flows(self) -> list[dict]:
        """Return deduplicated privacy flows from read observations."""
        flows: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for event in self.events:
            if event.get("kind") != "ObservationEvent":
                continue
            items = event.get("information_flows")
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                data_type = item.get("data_type")
                data_subject = item.get("data_subject")
                if not isinstance(data_type, str) or not isinstance(data_subject, str):
                    continue
                data_type = data_type.strip()
                data_subject = data_subject.strip()
                if not data_type or not data_subject:
                    continue
                key = information_flow_key(data_type, data_subject)
                if key in seen:
                    continue
                seen.add(key)
                flow: dict[str, object] = {
                    "data_type": data_type,
                    "data_subject": data_subject,
                }
                disposition = item.get("disposition")
                if isinstance(disposition, str) and disposition.strip():
                    flow["disposition"] = disposition.strip()
                reason = item.get("reason")
                if isinstance(reason, str) and reason.strip():
                    flow["reason"] = reason.strip()
                # Provenance: which reads the auditor was looking at when it
                # produced this flow. Carried through so extractor recall and
                # false-negative analysis can be traced back to a source read
                # instead of re-derived from the event stream.
                for provenance_key in ("source_action_ids", "source_tools"):
                    value = item.get(provenance_key)
                    if isinstance(value, list) and value:
                        flow[provenance_key] = [str(entry) for entry in value]
                flows.append(flow)
        return flows

    def extract_privacy_judgments(self) -> list[dict]:
        """Walk ObservationEvents and surface their ``privacy_check_result``.

        Present only when a privacy analyzer with judge support ran (L3
        ``ci_audit``). Each entry mirrors the persisted :class:`PrivacyJudgment`
        shape plus the ``tool_call_id`` / ``tool_name`` of the write attempt
        the judge saw, so analysis scripts can line it up with ``tool_calls``.
        Rejection paths also surface here: when the judge returned ABSTRACT
        or BLOCK, no ObservationEvent is emitted for the write — so we also
        walk UserRejectObservations and include their rejection_reason under
        ``rejected=True`` for completeness.
        """
        judgments: list[dict] = []
        for event in self.events:
            kind = event.get("kind")
            if kind == "ObservationEvent":
                judgment = event.get("privacy_check_result")
                if not isinstance(judgment, dict):
                    continue
                transmitted = judgment.get("transmitted_flows") or []
                decision = judgment.get("decision") or "pass"
                entry: dict[str, Any] = {
                    "tool_name": event.get("tool_name"),
                    "tool_call_id": event.get("tool_call_id"),
                    "decision": decision,
                    "rationale": judgment.get("rationale") or "",
                    "transmitted_flows": transmitted,
                    "rejected": False,
                }
                context = judgment.get("transmission_context")
                if context is not None:
                    entry["transmission_context"] = context
                judgments.append(entry)
            elif kind == "UserRejectObservation":
                # Rejection emitted by the judge path short-circuits the
                # tool before execution. Newer SDK events carry the full
                # PrivacyJudgment; keep a reason-only fallback for old result
                # files.
                reason = event.get("rejection_reason") or ""
                if event.get("rejection_source") != "hook":
                    continue
                judgment = event.get("privacy_check_result")
                if isinstance(judgment, dict):
                    entry: dict[str, Any] = {
                        "tool_name": event.get("tool_name"),
                        "tool_call_id": event.get("tool_call_id"),
                        "decision": judgment.get("decision") or "pass",
                        "rationale": judgment.get("rationale") or reason,
                        "transmitted_flows": (judgment.get("transmitted_flows") or []),
                        "rejected": True,
                    }
                    context = judgment.get("transmission_context")
                    if context is not None:
                        entry["transmission_context"] = context
                    judgments.append(entry)
                    continue
                # Heuristic: the BLOCK directive contains
                # "not an appropriate audience"; the ABSTRACT directive
                # contains "Re-compose the message without the specifics".
                if "not an appropriate audience" in reason:
                    decision = "block"
                elif "Re-compose the message without the specifics" in reason:
                    decision = "abstract"
                else:
                    # Not a judge rejection (e.g. a pre-tool hook).
                    continue
                judgments.append(
                    {
                        "tool_name": event.get("tool_name"),
                        "tool_call_id": event.get("tool_call_id"),
                        "decision": decision,
                        "rationale": reason,
                        "transmitted_flows": [],
                        "rejected": True,
                    }
                )
        return judgments

    def extract_audit_instructions(self) -> list[dict]:
        """Surface proactive steering messages emitted by the privacy auditor.

        Current runs attach each instruction directly to the triggering read
        ``ObservationEvent`` so the LLM sees the guidance inside the tool
        result and the message history keeps strict assistant-tool-result
        ordering. Older result files may contain the previous
        ``MessageEvent(source="user", sender="privacy_auditor")`` shape; keep
        that fallback for analysis compatibility.

        Empty unless the L3 ``ci_audit`` analyzer was active.
        """
        instructions: list[dict] = []
        last_read_tool_call_id: str | None = None
        for event in self.events:
            kind = event.get("kind")
            if kind == "ObservationEvent":
                tool_name = event.get("tool_name")
                if isinstance(tool_name, str) and is_agent_read_tool(
                    tool_name,
                    forced_read=False,
                ):
                    last_read_tool_call_id = event.get("tool_call_id")
                    text = event.get("audit_instruction")
                    if isinstance(text, str) and text.strip():
                        instructions.append(
                            {
                                "text": text.strip(),
                                "after_tool_call_id": last_read_tool_call_id,
                            }
                        )
                continue
            if kind != "MessageEvent":
                continue
            if event.get("source") != "user":
                continue
            if event.get("sender") != "privacy_auditor":
                continue
            llm_message = event.get("llm_message") or {}
            content = llm_message.get("content") or []
            text_parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        text_parts.append(text)
            text = "\n".join(text_parts).strip()
            if not text:
                continue
            instructions.append(
                {
                    "text": text,
                    "after_tool_call_id": last_read_tool_call_id,
                }
            )
        return instructions
