from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from openhands.sdk.agent.tool_audit import (
    ReadBatchAuditResult,
    ToolExecutionAuditBase,
    ToolExecutionAuditResult,
)
from openhands.sdk.event import (
    ActionEvent,
    MessageEvent,
    ObservationEvent,
    UserRejectObservation,
)
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.logger import get_logger
from openhands.sdk.privacy.analyzer import PrivacyAnalyzerBase
from openhands.sdk.privacy.config import PrivacyAuditGuidanceMode, PrivacyAuditMode
from openhands.sdk.privacy.flow import (
    AuditDecision,
    FlowDisposition,
    InformationFlow,
    JudgmentDecision,
    PrivacyJudgment,
    TransmissionContext,
    TransmittedFlow,
    information_flow_key,
)
from openhands.sdk.privacy.tool_context import (
    derive_recipient_descriptor,
    derive_transmission_context,
    extract_write_content,
    should_extract_information_flows,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


logger = get_logger(__name__)


# Tag attached to MessageEvent.sender for auditor-injected steering messages.
# The auditor never speaks as the human user; ``sender="privacy_auditor"``
# lets downstream consumers (event collectors, stuck-loop detectors) tell
# auditor messages apart from genuine user input without inspecting text.
PRIVACY_AUDITOR_SENDER = "privacy_auditor"


# How many audit rejections one write tool may accumulate before the gate
# stops inviting another rewrite. The gate stays closed either way; past this
# many attempts the directive switches to "remove it or ask the user", since
# repeated rewording has demonstrably not resolved the flagged content.
MAX_RECOMPOSE_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class PrivacyAuditController(ToolExecutionAuditBase):
    """Connect a privacy analyzer to the agent's tool-execution loop.

    The controller has two responsibilities:

    1. **Decision-boundary audit** (after a read batch finishes): call
       :meth:`PrivacyAnalyzerBase.audit_decision` once per batch, attach the
       inventory delta to the batch's last content-bearing ObservationEvent,
       and surface proactive steering guidance when the analyzer returns a
       non-empty instruction. The agent loop attaches that guidance to the
       triggering read observation before exposing it to the execution model.
    2. **Write-time judge** (before a write tool executes): call
       :meth:`PrivacyAnalyzerBase.judge_write_action_with_context` and short-
       circuit non-PASS decisions with a ``UserRejectObservation``.

    ``audit_mode`` supports component ablations without duplicating the
    controller. Read-steer-only records write judgments without enforcing
    them; write-enforce-only keeps read extraction for the judge's inventory
    but withholds read-boundary guidance from the execution model.
    """

    analyzer: PrivacyAnalyzerBase
    principal: str | None = None
    task_purpose: str | None = None
    expected_recipient: str | None = None
    expected_channel: str | None = None
    recipient_role: str | None = None
    guidance_mode: PrivacyAuditGuidanceMode = PrivacyAuditGuidanceMode.INSTRUCTION_ONLY
    audit_mode: PrivacyAuditMode = PrivacyAuditMode.FULL

    def before_tool_execution(
        self,
        conversation: LocalConversation,
        action_event: ActionEvent,
        is_read_only: bool,
    ) -> ToolExecutionAuditResult:
        """Judge write actions before execution. Reads are deferred to the
        batch-level hook ``after_read_batch``.
        """
        if is_read_only:
            return ToolExecutionAuditResult()
        judgment, error = self._judge_write_action_with_error(
            conversation, action_event
        )
        if judgment is None:
            # ``error`` distinguishes "the gate crashed" (fail open, recorded)
            # from "there was nothing in the inventory to judge".
            return ToolExecutionAuditResult(audit_error=error)
        if self.audit_mode is PrivacyAuditMode.READ_STEER_ONLY:
            return ToolExecutionAuditResult(
                observation_event_updates={"privacy_check_result": judgment}
            )
        if judgment.decision is not JudgmentDecision.PASS:
            exhausted = (
                self.count_prior_rejections(conversation, action_event.tool_name)
                >= MAX_RECOMPOSE_ATTEMPTS
            )
            if exhausted:
                judgment = judgment.model_copy(update={"recompose_exhausted": True})
            return ToolExecutionAuditResult(
                replacement_events=[
                    self.build_judgment_rejection(action_event, judgment)
                ]
            )
        return ToolExecutionAuditResult(
            observation_event_updates={"privacy_check_result": judgment}
        )

    def after_read_batch(
        self,
        conversation: LocalConversation,
        read_pairs: list[tuple[ActionEvent, ObservationEvent]],
    ) -> ReadBatchAuditResult:
        """Run one auditor pass over the read batch's observations.

        Attaches the resulting inventory delta to the last read whose tool is
        a content-extraction target (per
        :func:`should_extract_information_flows`), and returns optional auditor
        steering text. The agent loop attaches that text to the triggering read
        ObservationEvent so it lands in context before the next planning turn
        without breaking tool-call/tool-result ordering.
        """
        if not read_pairs:
            return ReadBatchAuditResult()

        buffered: list[tuple] = []
        source_action_ids: list[str] = []
        source_tools: list[str] = []
        for action_event, obs_event in read_pairs:
            tool_name = action_event.tool_name
            if not should_extract_information_flows(tool_name):
                continue
            buffered.append((obs_event.observation, tool_name))
            source_action_ids.append(action_event.id)
            source_tools.append(tool_name)
        if not buffered:
            return ReadBatchAuditResult()

        accumulated = self.gather_accumulated_flows(conversation)
        context = TransmissionContext(
            data_recipient=self.expected_recipient or "",
            transmission_channel=self.expected_channel or "(no pending write)",
            principal=self.principal or "",
            task_purpose=self.task_purpose or "",
            recipient_role=self.recipient_role,
            tool_name="",
        )

        try:
            decision = self.analyzer.audit_decision(
                buffered_reads=buffered,
                accumulated_flows=accumulated,
                transmission_context=context,
            )
        except Exception as exc:
            logger.warning("Privacy auditor failed at read-batch boundary: %s", exc)
            # Fail open, but on the record: an audit that crashed must not be
            # indistinguishable from an audit that passed.
            return ReadBatchAuditResult(audit_error=f"read_boundary: {exc!r}")

        # Persist the entire batch's inventory delta on the last eligible
        # ObservationEvent. Downstream consumers
        # (``gather_accumulated_flows`` / ``extract_privacy_flows``) walk all
        # ObservationEvents and union their ``information_flows`` field, so
        # attaching the delta once per batch keeps the conversation-wide
        # inventory correct without inventing per-observation attribution
        # the auditor never produced. For the same reason the provenance
        # recorded on each flow is the whole eligible batch, not a single read.
        flows_by_id: dict[str, list[InformationFlow]] = {}
        if decision.inventory_delta:
            flows_by_id[source_action_ids[-1]] = [
                flow.model_copy(
                    update={
                        "source_action_ids": list(source_action_ids),
                        "source_tools": list(source_tools),
                    }
                )
                for flow in decision.inventory_delta
            ]

        instruction = ""
        if self.audit_mode is not PrivacyAuditMode.WRITE_ENFORCE_ONLY:
            instruction = _render_read_boundary_instruction(
                decision=decision,
                guidance_mode=self.guidance_mode,
            )
        instruction_event: MessageEvent | None = None
        if instruction:
            instruction_event = MessageEvent(
                source="user",
                sender=PRIVACY_AUDITOR_SENDER,
                llm_message=Message(
                    role="user",
                    content=[TextContent(text=instruction)],
                ),
            )

        return ReadBatchAuditResult(
            information_flows_by_action_id=flows_by_id,
            instruction_event=instruction_event,
            hide_information_flows_from_llm=(
                self.audit_mode is PrivacyAuditMode.WRITE_ENFORCE_ONLY
            ),
        )

    def count_prior_rejections(
        self,
        conversation: LocalConversation,
        tool_name: str,
    ) -> int:
        """Count audit rejections already issued for ``tool_name`` in this task."""
        return sum(
            1
            for event in conversation.state.events
            if isinstance(event, UserRejectObservation)
            and event.rejection_source == "hook"
            and event.tool_name == tool_name
            and event.privacy_check_result is not None
        )

    def gather_accumulated_flows(
        self,
        conversation: LocalConversation,
    ) -> list[InformationFlow]:
        """Collect deduplicated information flows from prior observations."""
        seen: set[tuple[str, str]] = set()
        flows: list[InformationFlow] = []
        for event in conversation.state.events:
            if isinstance(event, ObservationEvent) and event.information_flows:
                for flow in event.information_flows:
                    key = information_flow_key(flow.data_type, flow.data_subject)
                    if key not in seen:
                        seen.add(key)
                        flows.append(flow)
        return flows

    def judge_write_action(
        self,
        conversation: LocalConversation,
        action_event: ActionEvent,
    ) -> PrivacyJudgment | None:
        """Run the privacy judge for a write action when inventory exists."""
        judgment, _ = self._judge_write_action_with_error(conversation, action_event)
        return judgment

    def _judge_write_action_with_error(
        self,
        conversation: LocalConversation,
        action_event: ActionEvent,
    ) -> tuple[PrivacyJudgment | None, str | None]:
        """Judge a write, also reporting why no judgment was produced.

        Returns ``(judgment, None)`` on success, ``(None, None)`` when there is
        no inventory to judge, and ``(None, reason)`` when the gate failed open.
        The caller needs the third case on the record: a crashed gate and a
        passing gate otherwise look identical in the results.
        """
        accumulated = self.gather_accumulated_flows(conversation)
        if not accumulated:
            return None, None
        write_content = extract_write_content(action_event)
        context = derive_transmission_context(
            action_event,
            principal=self.principal,
            task_purpose=self.task_purpose,
            recipient_role=self.recipient_role,
        )
        previously_rejected = gather_previously_rejected_items(
            conversation, action_event.tool_name
        )
        if previously_rejected:
            context = context.model_copy(
                update={"previously_rejected": previously_rejected}
            )
        try:
            judgment = self.analyzer.judge_write_action_with_context(
                write_content=write_content,
                accumulated_flows=accumulated,
                transmission_context=context,
            )
            if judgment.transmission_context is None:
                judgment = judgment.model_copy(update={"transmission_context": context})
            return judgment, None
        except TypeError:
            # Backward-compatibility fallback for analyzers that override the
            # adapter method with the old signature.
            judgment = self.analyzer.judge_write_action(
                write_content=write_content,
                accumulated_flows=accumulated,
                recipient_descriptor=derive_recipient_descriptor(action_event),
                principal=self.principal or "",
            )
            if judgment.transmission_context is None:
                judgment = judgment.model_copy(update={"transmission_context": context})
            return judgment, None
        except Exception as exc:
            logger.warning(
                "Privacy judge failed for %s: %s",
                action_event.tool_name,
                exc,
            )
            return None, f"write_boundary({action_event.tool_name}): {exc!r}"

    def build_judgment_rejection(
        self,
        action_event: ActionEvent,
        judgment: PrivacyJudgment,
    ) -> UserRejectObservation:
        """Translate an ABSTRACT or BLOCK judgment into a tool rejection."""
        if judgment.recompose_exhausted:
            # Repeated rewrites have not resolved the flagged content. Keep the
            # gate closed, but stop inviting another rewrite: an unbounded
            # reject/recompose loop burns the step budget without converging.
            directive = (
                "Earlier attempts at this message were rejected for the same "
                "flagged content, and it is still present. Do not simply "
                "reword it again: either send the message with that content "
                "removed entirely, or ask the user how to proceed."
            )
        elif judgment.decision is JudgmentDecision.BLOCK:
            directive = (
                "This recipient is not an appropriate audience for the "
                "content you were about to send. Consider whether this "
                "information should be sent at all, or drop the flagged "
                "details before retrying."
            )
        else:
            directive = (
                "Re-compose the message without the specifics flagged by "
                "the auditor. Keep the overall shape and intent; drop or "
                "generalize sensitive details before retrying."
            )
        reason = _render_structured_rejection_reason(
            judgment=judgment,
            directive=directive,
        )
        return UserRejectObservation(
            tool_name=action_event.tool_name,
            tool_call_id=action_event.tool_call.id,
            action_id=action_event.id,
            rejection_reason=reason,
            rejection_source="hook",
            privacy_check_result=judgment,
        )


def gather_previously_rejected_items(
    conversation: LocalConversation,
    tool_name: str,
) -> list[TransmittedFlow]:
    """Items the most recent audit rejection for ``tool_name`` flagged.

    The controller is rebuilt per hook call and holds no state, so the prior
    rejection is recovered from the event stream -- which already persists it:
    :meth:`PrivacyAuditController.build_judgment_rejection` attaches the full
    judgment to the ``UserRejectObservation``.
    """
    for event in reversed(conversation.state.events):
        if not isinstance(event, UserRejectObservation):
            continue
        if event.rejection_source != "hook" or event.tool_name != tool_name:
            continue
        judgment = event.privacy_check_result
        if judgment is None:
            continue
        return [
            flow
            for flow in judgment.transmitted_flows
            if flow.disposition is not FlowDisposition.PASS
        ]
    return []


def _render_read_boundary_instruction(
    *,
    decision: AuditDecision,
    guidance_mode: PrivacyAuditGuidanceMode,
) -> str:
    """Render read-boundary guidance for the execution model."""
    base_instruction = (decision.instruction or "").strip()
    if guidance_mode is PrivacyAuditGuidanceMode.INSTRUCTION_ONLY:
        return base_instruction

    sections: list[str] = []
    if base_instruction:
        sections.extend(
            [
                "<AUDITOR_STEERING_NOTE>",
                base_instruction,
                "</AUDITOR_STEERING_NOTE>",
            ]
        )

    if decision.inventory_delta:
        if guidance_mode is PrivacyAuditGuidanceMode.CONTEXTUAL_SHARE_DECISIONS:
            sections.append(
                _render_contextual_share_decisions(decision.inventory_delta)
            )
        else:
            sections.append(_render_annotated_flows(decision.inventory_delta))

    if guidance_mode is PrivacyAuditGuidanceMode.ANNOTATED_FLOWS_WITH_PLAN:
        sections.extend(
            [
                "<WRITE_PRIVACY_PLAN_REQUIRED>",
                (
                    "Before any write, send, publish, create-post, or message "
                    "tool call, first call the think tool with a "
                    "WRITE_PRIVACY_PLAN. In that plan, list which audited "
                    "flows will be used as-is, which will be generalized, and "
                    "which will be omitted. Then compose the write action to "
                    "match that plan."
                ),
                "</WRITE_PRIVACY_PLAN_REQUIRED>",
            ]
        )

    return "\n".join(sections).strip()


def _render_contextual_share_decisions(flows: list[InformationFlow]) -> str:
    lines = [
        "<CONTEXTUAL_SHARE_DECISIONS>",
        (
            "These are read-time contextual share decisions from the privacy "
            "auditor, based on the expected or auditor-inferred semantic "
            "recipient, channel, and task purpose known before the write tool "
            "call."
        ),
        (
            "For PASS decisions, use exact facts only when needed. For "
            "ABSTRACT decisions, use the safe rendering or an equally "
            "high-level version. For BLOCK decisions, omit the fact."
        ),
    ]
    for idx, flow in enumerate(flows, start=1):
        lines.extend(
            [
                f"Decision {idx}:",
                f"- question: {_question_for_flow(flow)}",
            ]
        )
        if flow.semantic_recipient:
            lines.append(f"- inferred_recipient: {flow.semantic_recipient}")
        if flow.semantic_channel:
            lines.append(f"- inferred_channel: {flow.semantic_channel}")
        lines.extend(
            [
                f"- subject: {flow.data_subject}",
                f"- fact: {flow.data_type}",
                f"- decision: {flow.disposition.value}",
                f"- safe_rendering: {_safe_rendering_for_flow(flow)}",
            ]
        )
        if flow.semantic_recipient_role:
            lines.append(f"- recipient_role: {flow.semantic_recipient_role}")
        if flow.context_confidence:
            lines.append(f"- context_confidence: {flow.context_confidence}")
        if flow.context_evidence:
            lines.append(f"- context_evidence: {flow.context_evidence}")
        reason = flow.reason.strip()
        if reason:
            lines.append(f"- reason: {reason}")
    lines.append("</CONTEXTUAL_SHARE_DECISIONS>")
    return "\n".join(lines)


def _question_for_flow(flow: InformationFlow) -> str:
    question = flow.share_question.strip()
    if question:
        return question
    return f"Is it appropriate to share this exact fact about {flow.data_subject}?"


def _safe_rendering_for_flow(flow: InformationFlow) -> str:
    explicit = flow.safe_rendering.strip() or flow.safe_use.strip()
    if explicit:
        return explicit
    if flow.disposition is FlowDisposition.PASS:
        return "Use the exact fact only if it is needed for the task."
    if flow.disposition in (
        FlowDisposition.ABSTRACT,
        FlowDisposition.GENERALIZE,
    ):
        return "Use a higher-level summary; avoid exact details."
    return "Omit this fact from outgoing content."


def _render_annotated_flows(flows: list[InformationFlow]) -> str:
    lines = [
        "<AUDITED_INFORMATION_FLOWS>",
        (
            "These are read-boundary audit annotations for the information "
            "you just observed. Treat them as task-specific guidance for "
            "the likely recipient and channel."
        ),
        (
            "Use PASS as-is only if needed for the task. Use ABSTRACT or "
            "GENERALIZE only at a higher level. Omit BLOCK or OMIT flows "
            "from outgoing content."
        ),
    ]
    for idx, flow in enumerate(flows, start=1):
        lines.extend(
            [
                f"Flow {idx}:",
                f"- subject: {flow.data_subject}",
                f"- fact: {flow.data_type}",
                f"- disposition: {flow.disposition.value}",
                f"- safe_use: {_safe_use_for_flow(flow)}",
            ]
        )
        reason = flow.reason.strip()
        if reason:
            lines.append(f"- reason: {reason}")
    lines.append("</AUDITED_INFORMATION_FLOWS>")
    return "\n".join(lines)


def _safe_use_for_flow(flow: InformationFlow) -> str:
    explicit = flow.safe_use.strip()
    if explicit:
        return explicit
    if flow.disposition is FlowDisposition.PASS:
        return "May use as-is when it is necessary to complete the task."
    if flow.disposition in (
        FlowDisposition.ABSTRACT,
        FlowDisposition.GENERALIZE,
    ):
        return "Use only as a higher-level summary; avoid exact details."
    return "Do not include this flow in outgoing content."


def _render_structured_rejection_reason(
    *,
    judgment: PrivacyJudgment,
    directive: str,
) -> str:
    """Render a concise, structured instruction for the execution model."""
    lines: list[str] = []
    rationale = (judgment.rationale or "").strip()
    if rationale:
        lines.extend(["Audit rationale:", rationale, ""])

    context = judgment.transmission_context
    if context is not None:
        context_lines = [
            f"- recipient: {context.data_recipient or 'unknown'}",
            f"- channel: {context.transmission_channel or 'unknown'}",
        ]
        if context.task_purpose:
            context_lines.append(f"- task purpose: {context.task_purpose}")
        lines.extend(["Transmission context:", *context_lines, ""])

    grouped = _flows_by_disposition(judgment)
    if any(grouped.values()):
        lines.append("Rewrite constraints:")
        _append_flow_group(
            lines,
            "Allowed to keep",
            grouped["pass"],
        )
        _append_flow_group(
            lines,
            "Generalize",
            grouped["generalize"],
        )
        _append_flow_group(
            lines,
            "Omit",
            grouped["omit"],
        )
        _append_flow_group(
            lines,
            "Block",
            grouped["block"],
        )
        lines.append("")

    lines.extend(["Audit directive:", directive])
    return "\n".join(lines)


def _flows_by_disposition(
    judgment: PrivacyJudgment,
) -> dict[str, list[str]]:
    grouped = {
        "pass": [],
        "generalize": [],
        "omit": [],
        "block": [],
    }
    for flow in judgment.transmitted_flows:
        label = f"{flow.data_subject}: {flow.data_type}"
        key = flow.disposition.value
        if key == "abstract":
            key = "generalize"
        grouped[key].append(label)
    return grouped


def _append_flow_group(
    lines: list[str],
    label: str,
    flows: list[str],
) -> None:
    if not flows:
        return
    lines.append(f"- {label}:")
    for flow in flows:
        lines.append(f"  - {flow}")
