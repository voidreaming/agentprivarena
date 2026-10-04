from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from openhands.sdk.event import ActionEvent, Event, MessageEvent, ObservationEvent
from openhands.sdk.tool.schema import Observation


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation
    from openhands.sdk.privacy.flow import InformationFlow


@dataclass(frozen=True, slots=True)
class ToolExecutionAuditResult:
    """Result returned by a tool-execution audit extension point.

    ``replacement_events`` short-circuits normal tool execution when populated.
    ``observation_event_updates`` carries metadata to attach to the eventual
    ObservationEvent when execution proceeds.

    ``audit_error`` records that the audit degraded to a no-op because it
    raised. Audits here are fail-open by design, which means a crashed audit
    otherwise looks exactly like an audit that approved the action; recording
    the reason keeps that difference visible in persisted results.
    """

    replacement_events: list[Event] | None = None
    observation_event_updates: dict[str, object] = field(default_factory=dict)
    audit_error: str | None = None


@dataclass(frozen=True, slots=True)
class ReadBatchAuditResult:
    """Result returned by :meth:`ToolExecutionAuditBase.after_read_batch`.

    ``information_flows_by_action_id`` maps the ``ActionEvent.id`` of each
    read in the batch to the information flows the auditor attributes to
    that read's observation. Callers apply these via ``model_copy`` before
    emitting the ObservationEvent, so persistence is unchanged.

    ``instruction_event`` is an optional proactive steering message. The
    default agent loop attaches its text to the triggering read observation so
    the next planning turn sees it inside a tool result, preserving strict
    assistant-tool-result message ordering for OpenAI-compatible providers.

    ``hide_information_flows_from_llm`` keeps extracted flows available to
    later audit hooks and persisted results without rendering them into the
    execution model's tool observation.

    ``audit_error`` records that the audit degraded to a no-op because it
    raised, so a crashed read-boundary audit is distinguishable from one that
    found nothing to flag. It is metadata only and is never shown to the
    execution model.
    """

    information_flows_by_action_id: dict[str, list[InformationFlow]] = field(
        default_factory=dict
    )
    instruction_event: MessageEvent | None = None
    hide_information_flows_from_llm: bool = False
    audit_error: str | None = None


class ToolExecutionAuditBase(ABC):
    """Extension point for auditing tool execution.

    Implementations may inspect actions before execution and observations
    after execution. They should keep domain-specific policy outside the
    Agent loop.
    """

    def before_tool_execution(
        self,
        _conversation: LocalConversation,
        _action_event: ActionEvent,
        _is_read_only: bool,
    ) -> ToolExecutionAuditResult:
        """Run before a tool executes."""
        return ToolExecutionAuditResult()

    def after_tool_execution(
        self,
        _conversation: LocalConversation,
        _action_event: ActionEvent,
        _observation: Observation,
        _is_read_only: bool,
    ) -> ToolExecutionAuditResult:
        """Run after a tool executes."""
        return ToolExecutionAuditResult()

    def after_read_batch(
        self,
        _conversation: LocalConversation,
        _read_pairs: list[tuple[ActionEvent, ObservationEvent]],
    ) -> ReadBatchAuditResult:
        """Run once after a batch of read-only tools has completed.

        Called between ``_ActionBatch.prepare`` and ``_ActionBatch.emit`` so
        implementations can enrich the not-yet-emitted ObservationEvents and
        optionally return a steering ``MessageEvent`` to inject after the
        batch is emitted.
        """
        return ReadBatchAuditResult()
