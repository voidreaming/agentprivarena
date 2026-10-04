"""Tests for the v2 decision-boundary privacy audit pipeline.

Verifies that:
1. ``audit_decision()`` fires ONCE per read batch, not per-read.
2. ``audit_decision()`` does NOT fire on pure-write batches.
3. When the auditor returns an ``instruction``, the agent loop attaches it to
   the read ObservationEvent so the next planning turn sees it inside the tool
   result without breaking strict tool-call/tool-result message ordering.
4. When ``instruction`` is empty / None, no auditor guidance is attached.
5. ``ObservationEvent.information_flows`` is populated on the last read
   from the auditor's ``inventory_delta``.
6. Accumulated flows are deduplicated before the write-time judge call.
7. Flow-annotated observations include ``[INFORMATION INVENTORY]`` in LLM
   messages.
8. Judge PASS executes the tool; ABSTRACT / BLOCK short-circuit with a
   ``UserRejectObservation`` carrying an audience-focused rationale.
9. ``_extract_write_content`` extracts outgoing message from action arguments.
"""

import dataclasses
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self
from unittest.mock import patch

import pytest
from litellm import ChatCompletionMessageToolCall
from litellm.types.utils import (
    Choices,
    Function,
    Message as LiteLLMMessage,
    ModelResponse,
)
from pydantic import PrivateAttr, SecretStr

from openhands.sdk.agent import Agent
from openhands.sdk.conversation import Conversation
from openhands.sdk.event import (
    ActionEvent,
    LLMConvertibleEvent,
    MessageEvent,
    ObservationEvent,
)
from openhands.sdk.event.llm_convertible.observation import UserRejectObservation
from openhands.sdk.llm import LLM, LLMResponse, Message, MessageToolCall, TextContent
from openhands.sdk.llm.streaming import TokenCallbackType
from openhands.sdk.mcp.definition import MCPToolAction
from openhands.sdk.privacy.analyzer import PrivacyAnalyzerBase
from openhands.sdk.privacy.audit import PRIVACY_AUDITOR_SENDER
from openhands.sdk.privacy.config import (
    AuditPolicy,
    PrivacyAuditMode,
    PrivacyReadAuditMode,
    PrivacySchemaMode,
    PrivacyStrictness,
)
from openhands.sdk.privacy.flow import (
    AuditDecision,
    FlowDisposition,
    InformationFlow,
    JudgmentDecision,
    PrivacyJudgment,
    TransmissionContext,
    TransmittedFlow,
)
from openhands.sdk.privacy.llm_analyzer import (
    _JUDGE_OBJECTIVE_CI,
    _PII_SPEC,
    _POLICY_SPECS,
    AUDIT_DECISION_PROMPT,
    CONTEXTUAL_SHARE_AUDIT_DECISION_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    STRICTNESS_POLICY,
    LLMPrivacyAnalyzer,
    PolicySpec,
    _check_policy_registry,
    _enforce_recompose_verification,
    _is_valid_extracted_data_type,
    _is_valid_extracted_subject,
    _parse_judgment,
    _policy_spec,
    _render_judge_inventory_item,
    _replace_once,
)
from openhands.sdk.testing import TestLLM
from openhands.sdk.tool import Action, Observation, Tool, ToolExecutor, register_tool
from openhands.sdk.tool.tool import ToolAnnotations, ToolDefinition


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState


# ── Recording privacy analyzer ──


class RecordingPrivacyAnalyzer(PrivacyAnalyzerBase):
    """Privacy analyzer that records calls instead of calling an LLM.

    ``audit_result`` lets each test inject the exact :class:`AuditDecision`
    returned from ``audit_decision`` -- including the optional steering
    ``instruction`` -- so control-flow branches around the auditor's
    proactive MessageEvent are exercised without any LLM calls.

    ``judge_result`` lets each test inject the exact :class:`PrivacyJudgment`
    returned from ``judge_write_action`` -- PASS, ABSTRACT, or BLOCK -- so
    write-time branches are exercised without any LLM calls.
    """

    def __init__(
        self,
        audit_result: AuditDecision | None = None,
        judge_result: PrivacyJudgment | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._audit_calls: list[tuple[int, int, TransmissionContext]] = []
        self._judge_calls: list[tuple[str, int, str, str]] = []
        self._audit_result = audit_result or AuditDecision(
            inventory_delta=[
                InformationFlow(
                    data_type="medical diagnosis",
                    data_subject="Alice",
                )
            ]
        )
        self._judge_result = judge_result or PrivacyJudgment()

    def audit_decision(
        self,
        buffered_reads,
        accumulated_flows,
        transmission_context,
    ) -> AuditDecision:
        self._audit_calls.append(
            (
                len(buffered_reads),
                len(accumulated_flows),
                transmission_context,
            )
        )
        return self._audit_result

    def judge_write_action(
        self,
        write_content: str,
        accumulated_flows: list[InformationFlow],
        recipient_descriptor: str,
        principal: str,
    ) -> PrivacyJudgment:
        self._judge_calls.append(
            (
                write_content,
                len(accumulated_flows),
                recipient_descriptor,
                principal,
            )
        )
        return self._judge_result


# ── Simple read-only tool ──


class ReadAction(Action):
    query: str = ""


class ReadObservation(Observation):
    content: list = [  # noqa: RUF012
        TextContent(text="Alice has a medical diagnosis of flu.")
    ]


class ReadExecutor(ToolExecutor[ReadAction, ReadObservation]):
    def __call__(self, action: ReadAction, conversation=None) -> ReadObservation:
        return ReadObservation()


class ReadTool(ToolDefinition[ReadAction, ReadObservation]):
    name = "read_tool"

    @classmethod
    def create(cls, conv_state: "ConversationState | None" = None) -> Sequence[Self]:
        return [
            cls(
                description="A read-only tool",
                action_type=ReadAction,
                observation_type=ReadObservation,
                executor=ReadExecutor(),
                annotations=ToolAnnotations(readOnlyHint=True),
            )
        ]


register_tool("ReadTool", ReadTool)


# ── Simple write tool ──


class WriteAction(Action):
    message: str = ""


class WriteObservation(Observation):
    @property
    def text(self) -> str:
        return "Message sent successfully."


class WriteExecutor(ToolExecutor[WriteAction, WriteObservation]):
    def __call__(self, action: WriteAction, conversation=None) -> WriteObservation:
        return WriteObservation()


class WriteTool(ToolDefinition[WriteAction, WriteObservation]):
    name = "write_tool"

    @classmethod
    def create(cls, conv_state: "ConversationState | None" = None) -> Sequence[Self]:
        return [
            cls(
                description="A write tool",
                action_type=WriteAction,
                observation_type=WriteObservation,
                executor=WriteExecutor(),
            )
        ]


register_tool("WriteTool", WriteTool)


# ── Helpers ──


def _make_tool_call_response(tool_name: str, arguments: str) -> ModelResponse:
    return ModelResponse(
        id="mock-response",
        choices=[
            Choices(
                index=0,
                message=LiteLLMMessage(
                    role="assistant",
                    content="Calling tool.",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_1",
                            type="function",
                            function=Function(
                                name=tool_name,
                                arguments=arguments,
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def _message_text(message: Message) -> str:
    return "\n".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )


class ContextAwarePrivacyLLM(TestLLM):
    """A deterministic LLM whose write changes only when audit context is visible."""

    _captured_messages: list[list[Message]] = PrivateAttr(default_factory=list)
    _captured_kwargs: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    @property
    def captured_messages(self) -> list[list[Message]]:
        return self._captured_messages

    @property
    def captured_kwargs(self) -> list[dict[str, Any]]:
        return self._captured_kwargs

    def completion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        _return_metrics: bool = False,
        add_security_risk_prediction: bool = False,
        add_privacy_context: bool = False,
        on_token: TokenCallbackType | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._captured_messages.append(list(messages))
        self._captured_kwargs.append(dict(kwargs))
        self._call_count += 1

        if self._call_count == 1:
            response = Message(
                role="assistant",
                content=[TextContent(text="I need to read first.")],
                tool_calls=[
                    MessageToolCall(
                        id="ctx_read",
                        name="read_tool",
                        arguments='{"query": "Alice"}',
                        origin="completion",
                    )
                ],
            )
        elif self._call_count == 2:
            context = "\n".join(_message_text(message) for message in messages)
            saw_inventory = (
                "[INFORMATION INVENTORY]" in context
                and "medical diagnosis" in context
                and "Alice" in context
            )
            saw_instruction = (
                "Keep the write scoped to scheduling and omit medical context."
                in context
            )
            message_text = (
                "Scheduling update only: Alice is available tomorrow."
                if saw_inventory and saw_instruction
                else "Alice has a medical diagnosis of flu."
            )
            response = Message(
                role="assistant",
                content=[TextContent(text="I can now write the update.")],
                tool_calls=[
                    MessageToolCall(
                        id="ctx_write",
                        name="write_tool",
                        arguments=json.dumps({"message": message_text}),
                        origin="completion",
                    )
                ],
            )
        else:
            response = Message(role="assistant", content=[TextContent(text="Done.")])

        return self._response_from_message(response)

    def _response_from_message(self, message: Message) -> LLMResponse:
        return LLMResponse(
            message=message,
            metrics=self._zero_metrics(),
            raw_response=self._create_model_response(message),
        )


class CapturingAuditLLM(TestLLM):
    """A scripted audit LLM that records the prompt it received."""

    _captured_messages: list[list[Message]] = PrivateAttr(default_factory=list)

    @property
    def captured_messages(self) -> list[list[Message]]:
        return self._captured_messages

    def completion(
        self,
        messages: list[Message],
        tools: Sequence[ToolDefinition] | None = None,
        _return_metrics: bool = False,
        add_security_risk_prediction: bool = False,
        add_privacy_context: bool = False,
        on_token: TokenCallbackType | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        self._captured_messages.append(list(messages))
        return super().completion(
            messages=messages,
            tools=tools,
            _return_metrics=_return_metrics,
            add_security_risk_prediction=add_security_risk_prediction,
            add_privacy_context=add_privacy_context,
            on_token=on_token,
            **kwargs,
        )


# ── Tests ──


def test_llm_privacy_extractor_subject_quality_filter():
    """Extractor output must identify a concrete subject, not an account label."""
    accepted = [
        "Alice",
        "Dave Thompson",
        "Michael's father",
        "Quantum Innovations",
        "Project Alpha",
        "Quantum Innovations Project Alpha",
        "Project Alpha and Project Beta",
        "northern region cultural group",
        "parade performers",
        "performers and families",
    ]
    rejected = [
        "",
        "admin",
        "admin's father",
        "the sender",
        "the team",
        "community members",
    ]

    for subject in accepted:
        assert _is_valid_extracted_subject(subject)
    for subject in rejected:
        assert not _is_valid_extracted_subject(subject)


def test_llm_privacy_extractor_data_type_quality_filter():
    """Extractor output must name a concrete fact, not a generic category."""
    accepted = [
        "divorce",
        "hospitalization (heart attack)",
        "debt",
        "white-and-green feathered costume design used in the regional parade",
        "Saturday afternoon break near the main square",
        "grade improved from B to A in Quantum Computing",
    ]
    rejected = [
        "",
        "information",
        "details",
        "health information",
        "visual context",
        "personal challenges",
        "tough times",
        "rough week",
        "private matters",
    ]

    for data_type in accepted:
        assert _is_valid_extracted_data_type(data_type)
    for data_type in rejected:
        assert not _is_valid_extracted_data_type(data_type)


def test_audit_decision_prompt_requests_annotated_flows_and_steering():
    """The v2 audit prompt asks for per-flow guidance plus steering."""
    required_phrases = [
        "two-fold",
        "expected recipient/audience and channel",
        '"flows"',
        '"disposition"',
        '"safe_use"',
        '"reason"',
        '"instruction"',
        "Decompose the read content into concrete information units",
        "data_type is the specific shareable attribute or fact",
        "disposition must be one of: pass, abstract, block",
        "reason is a short audience-focused explanation",
        "safe_use is a short instruction",
        "audience-focused",
        "Do NOT name regulations",
        "Do NOT use the phrase 'contextual integrity'",
        "Do NOT invent facts the read content didn't establish",
        'Emit an EMPTY STRING ("")',
    ]
    for phrase in required_phrases:
        assert phrase in AUDIT_DECISION_PROMPT, (
            f"Audit prompt missing required phrase: {phrase!r}"
        )


def test_llm_privacy_analyzer_parses_audited_flow_annotations():
    """Audit JSON supports per-flow disposition and reason."""
    llm = CapturingAuditLLM.from_messages(
        [
            Message(
                role="assistant",
                content=[
                    TextContent(
                        text=json.dumps(
                            {
                                "inferred_context": {
                                    "semantic_recipient": "Mike",
                                    "semantic_channel": "Mattermost DM",
                                    "recipient_role": "scheduling recipient",
                                    "confidence": "high",
                                    "evidence": (
                                        "The task asks Jane to message Mike "
                                        "about scheduling."
                                    ),
                                },
                                "flows": [
                                    {
                                        "data_subject": "Alice",
                                        "data_type": "medical diagnosis",
                                        "disposition": "abstract",
                                        "reason": (
                                            "The likely audience needs "
                                            "availability, not diagnosis details."
                                        ),
                                        "safe_use": (
                                            "Mention availability only; "
                                            "omit the diagnosis."
                                        ),
                                        "question": (
                                            "Is it appropriate for Jane to "
                                            "share Alice's diagnosis with Mike?"
                                        ),
                                        "safe_rendering": (
                                            "Alice is unavailable for the meeting."
                                        ),
                                    }
                                ],
                                "instruction": "Keep the write scoped to scheduling.",
                            }
                        )
                    )
                ],
            )
        ]
    )
    analyzer = LLMPrivacyAnalyzer(llm=llm)

    decision = analyzer.audit_decision(
        buffered_reads=[(ReadObservation(), "read_tool")],
        accumulated_flows=[],
        transmission_context=TransmissionContext(
            principal="Jane",
            task_purpose="Send a scheduling update.",
            data_recipient="Mike",
            transmission_channel="Mattermost DM",
        ),
    )

    assert decision.instruction == "Keep the write scoped to scheduling."
    assert len(decision.inventory_delta) == 1
    flow = decision.inventory_delta[0]
    assert flow.data_subject == "Alice"
    assert flow.data_type == "medical diagnosis"
    assert flow.disposition is FlowDisposition.ABSTRACT
    assert "availability" in flow.reason
    assert "omit the diagnosis" in flow.safe_use
    assert "Is it appropriate for Jane" in flow.share_question
    assert "Alice is unavailable" in flow.safe_rendering
    assert flow.semantic_recipient == "Mike"
    assert flow.semantic_channel == "Mattermost DM"
    assert flow.semantic_recipient_role == "scheduling recipient"
    assert flow.context_confidence == "high"
    assert "message Mike" in flow.context_evidence

    assert isinstance(llm, CapturingAuditLLM)
    user_text = _message_text(llm.captured_messages[0][1])
    assert "Expected recipient / audience before write: Mike" in user_text
    assert "Expected transmission channel before write: Mattermost DM" in user_text


def test_llm_privacy_analyzer_contextual_read_mode_uses_share_prompt():
    llm = CapturingAuditLLM.from_messages(
        [
            Message(
                role="assistant",
                content=[
                    TextContent(
                        text=json.dumps(
                            {
                                "inferred_context": {
                                    "semantic_recipient": "Mary",
                                    "semantic_channel": "email",
                                    "recipient_role": "treatment update recipient",
                                    "confidence": "high",
                                    "evidence": (
                                        "The task purpose says to email Mary "
                                        "a treatment update."
                                    ),
                                },
                                "flows": [
                                    {
                                        "data_subject": "David Smith",
                                        "data_type": "Lisinopril 10mg daily",
                                        "decision": "abstract",
                                        "question": (
                                            "Is it appropriate for Jane to "
                                            "share David Smith's medication "
                                            "dosage with Mary by email?"
                                        ),
                                        "safe_rendering": (
                                            "David's treatment plan has been updated."
                                        ),
                                        "reason": (
                                            "Exact dosage is too specific for "
                                            "the expected update."
                                        ),
                                    }
                                ],
                                "instruction": "Use the safe rendering.",
                            }
                        )
                    )
                ],
            )
        ]
    )
    analyzer = LLMPrivacyAnalyzer(
        llm=llm,
        read_audit_mode=PrivacyReadAuditMode.CONTEXTUAL_SHARE,
    )

    decision = analyzer.audit_decision(
        buffered_reads=[(ReadObservation(), "read_tool")],
        accumulated_flows=[],
        transmission_context=TransmissionContext(
            principal="Jane",
            task_purpose="Email Mary a treatment update.",
            data_recipient="Mary",
            transmission_channel="email",
        ),
    )

    assert len(decision.inventory_delta) == 1
    flow = decision.inventory_delta[0]
    assert flow.disposition is FlowDisposition.ABSTRACT
    assert "David's treatment plan" in flow.safe_rendering
    assert "Is it appropriate for Jane" in flow.share_question
    assert flow.semantic_recipient == "Mary"
    assert flow.semantic_channel == "email"
    assert flow.semantic_recipient_role == "treatment update recipient"
    assert flow.context_confidence == "high"
    assert "email Mary" in flow.context_evidence

    assert isinstance(llm, CapturingAuditLLM)
    system_text = _message_text(llm.captured_messages[0][0])
    assert CONTEXTUAL_SHARE_AUDIT_DECISION_PROMPT in system_text
    assert "Is it appropriate for {principal} to share" in system_text
    assert '"inferred_context"' in system_text
    assert '"semantic_recipient"' in system_text


def test_llm_privacy_judge_prompt_balances_semantic_generalizations():
    """Judge prompt should catch clear leaks without over-flagging generic text."""
    required_phrases = [
        "read-boundary dispositions",
        "safe-use guidance",
        "can this data_subject's data_type be shared",
        "recipient / audience and transmission channel",
        "task purpose / user instruction",
        "prescribed medication routine",
        "recovery journey",
        "communicates a concrete non-public fact",
        "Do not infer transmission from generic business-update language alone",
        "General business updates can be appropriate for external audiences",
        "ordinary task context can establish a legitimate reason to share",
        "outgoing message obeys it",
        "relevant context, but not a blank check",
        "prefer abstract/generalize over block",
        "Image assets are information flows too",
        "task purpose and recipient/channel may make the asset appropriate",
        "do not flag an inventory item merely because it exists in the source material",
        (
            "First-person wording does not automatically make "
            "third-party source facts safe"
        ),
        "Never return abstract or block with an empty transmitted_flows list",
    ]
    forbidden_phrases = [
        "Judge only from explicit evidence",
        "Recipient authorization must be clear",
        "does not by itself justify sending grades",
    ]

    for phrase in required_phrases:
        assert phrase in JUDGE_SYSTEM_PROMPT
    for phrase in forbidden_phrases:
        assert phrase not in JUDGE_SYSTEM_PROMPT


def test_strictness_injects_ambiguous_tiebreak_into_both_prompts():
    """Strictness appends an ambiguous-flow tie-break to BOTH decision prompts.

    BALANCED injects nothing, so its prompts are byte-identical to the base
    constants (guaranteeing no behavior change vs prior CI-Audit).
    """
    llm = CapturingAuditLLM.from_messages(
        [Message(role="assistant", content=[TextContent(text="{}")])]
    )

    balanced = LLMPrivacyAnalyzer(llm=llm, strictness=PrivacyStrictness.BALANCED)
    assert STRICTNESS_POLICY[PrivacyStrictness.BALANCED] == ""
    assert balanced._audit_decision_prompt() == AUDIT_DECISION_PROMPT
    assert balanced._judge_system_prompt() == JUDGE_SYSTEM_PROMPT

    lenient = LLMPrivacyAnalyzer(llm=llm, strictness=PrivacyStrictness.LENIENT)
    strict = LLMPrivacyAnalyzer(llm=llm, strictness=PrivacyStrictness.STRICT)
    for analyzer in (lenient, strict):
        read_prompt = analyzer._audit_decision_prompt()
        judge_prompt = analyzer._judge_system_prompt()
        # The base prompt is preserved and the block is appended to both boundaries.
        assert read_prompt.startswith(AUDIT_DECISION_PROMPT)
        assert judge_prompt.startswith(JUDGE_SYSTEM_PROMPT)
        assert "STRICTNESS POLICY" in read_prompt
        assert "STRICTNESS POLICY" in judge_prompt
        assert "GENUINELY AMBIGUOUS" in judge_prompt

    # The two non-default levels steer ambiguity in opposite directions.
    assert "resolve toward PASS" in lenient._judge_system_prompt()
    assert "resolve toward the strongest intervention" in strict._judge_system_prompt()

    # Contextual-share read mode also receives the block.
    strict_ctx = LLMPrivacyAnalyzer(
        llm=llm,
        read_audit_mode=PrivacyReadAuditMode.CONTEXTUAL_SHARE,
        strictness=PrivacyStrictness.STRICT,
    )
    ctx_prompt = strict_ctx._audit_decision_prompt()
    assert ctx_prompt.startswith(CONTEXTUAL_SHARE_AUDIT_DECISION_PROMPT)
    assert "STRICTNESS POLICY" in ctx_prompt


def test_audit_policy_swaps_criterion_in_both_prompts():
    """Swapping the audit policy replaces the decision criterion at BOTH
    boundaries (read-time steering drives most of the effect, so a write-only
    swap would barely change behaviour). CI is the no-op default."""
    llm = CapturingAuditLLM.from_messages(
        [Message(role="assistant", content=[TextContent(text="{}")])]
    )
    # CI substitutes nothing: both prompts stay byte-identical to the validated
    # text, so the default condition is unaffected by the policy machinery.
    ci = LLMPrivacyAnalyzer(llm=llm, audit_policy=AuditPolicy.CONTEXTUAL_INTEGRITY)
    assert ci._audit_decision_prompt() == AUDIT_DECISION_PROMPT
    assert ci._judge_system_prompt() == JUDGE_SYSTEM_PROMPT
    ci_ctx = LLMPrivacyAnalyzer(
        llm=llm,
        audit_policy=AuditPolicy.CONTEXTUAL_INTEGRITY,
        read_audit_mode=PrivacyReadAuditMode.CONTEXTUAL_SHARE,
    )
    assert ci_ctx._audit_decision_prompt() == CONTEXTUAL_SHARE_AUDIT_DECISION_PROMPT

    for policy, marker in [
        (AuditPolicy.PII, "PERSONALLY IDENTIFIABLE INFORMATION"),
        (AuditPolicy.DATA_MINIMIZATION, "TASK NECESSITY"),
    ]:
        judge = LLMPrivacyAnalyzer(llm=llm, audit_policy=policy)._judge_system_prompt()
        read_plain = LLMPrivacyAnalyzer(
            llm=llm, audit_policy=policy
        )._audit_decision_prompt()
        read_ctx = LLMPrivacyAnalyzer(
            llm=llm,
            audit_policy=policy,
            read_audit_mode=PrivacyReadAuditMode.CONTEXTUAL_SHARE,
        )._audit_decision_prompt()

        # the policy criterion is substituted in at BOTH boundaries ...
        assert marker in judge
        assert "STEP 2 — AUDIENCE APPROPRIATENESS." not in judge
        assert _JUDGE_OBJECTIVE_CI not in judge
        # ... and CI's own criterion is genuinely GONE, not merely overridden.
        # (Appending it left the CI text in place and was empirically ignored.)
        assert "Is it appropriate for {principal} to share" not in read_ctx
        assert "the exact fact is clearly appropriate and needed" not in read_ctx
        assert "appears appropriate for the likely recipient" not in read_plain
        # shared scaffolding is untouched
        assert "STEP 1 — DETECT TRANSMITTED FACTS." in judge
        assert "Assign a disposition to each transmitted item:" in judge


def test_policy_registry_fails_loudly_instead_of_degrading_to_base(monkeypatch):
    """An unregistered or half-registered policy must raise, never fall back to
    the base criterion. A silent fallback makes a cross-policy comparison
    measure nothing, and that failure is invisible in the results."""
    # The base policy substitutes nothing, by design.
    assert _policy_spec(AuditPolicy.CONTEXTUAL_INTEGRITY) is None
    assert _policy_spec(AuditPolicy.PII) is _PII_SPEC

    # A policy present in the enum but absent from the registry.
    monkeypatch.delitem(_POLICY_SPECS, AuditPolicy.PII)
    with pytest.raises(ValueError, match="no PolicySpec registered"):
        _policy_spec(AuditPolicy.PII)
    with pytest.raises(ValueError, match="no PolicySpec registered"):
        _check_policy_registry()

    # A registered policy with a blank span.
    blank = dataclasses.replace(_PII_SPEC, judge_step2="   ")
    monkeypatch.setitem(_POLICY_SPECS, AuditPolicy.PII, blank)
    with pytest.raises(ValueError, match="empty criterion spans: judge_step2"):
        _check_policy_registry()


def test_policy_spec_requires_every_span():
    """Every span is a required field, so a half-written spec fails at
    construction rather than inheriting the base criterion for the rest."""
    with pytest.raises(TypeError):
        PolicySpec(read_intro="only one span")  # type: ignore[call-arg]


def test_replace_once_raises_on_missing_marker():
    """``str.replace`` would leave the base criterion in place and return
    silently; the substitution must fail loudly if a prompt edit moved a
    marker."""
    assert _replace_once("keep AAA tail", "AAA", "BBB") == "keep BBB tail"
    with pytest.raises(ValueError, match="prompt marker not found"):
        _replace_once("no marker here", "ABSENT", "x")


def test_customizable_policy_injected_into_both_prompts():
    """A user-authored privacy policy is appended to both audit prompts; empty is
    a no-op, and it composes with the strictness block (strictness first)."""
    llm = CapturingAuditLLM.from_messages(
        [Message(role="assistant", content=[TextContent(text="{}")])]
    )
    # Empty policy (default) = byte-identical to pure CI inference.
    plain = LLMPrivacyAnalyzer(llm=llm)
    assert plain._audit_decision_prompt() == AUDIT_DECISION_PROMPT
    assert plain._judge_system_prompt() == JUDGE_SYSTEM_PROMPT

    policy = "Never share the principal's home address or any photo with anyone."
    an = LLMPrivacyAnalyzer(llm=llm, privacy_policy=policy)
    for prompt in (an._audit_decision_prompt(), an._judge_system_prompt()):
        assert "USER PRIVACY POLICY" in prompt
        assert policy in prompt

    both = LLMPrivacyAnalyzer(
        llm=llm, strictness=PrivacyStrictness.STRICT, privacy_policy=policy
    )
    jp = both._judge_system_prompt()
    assert jp.startswith(JUDGE_SYSTEM_PROMPT)
    assert "STRICTNESS POLICY" in jp and "USER PRIVACY POLICY" in jp
    assert jp.index("STRICTNESS POLICY") < jp.index("USER PRIVACY POLICY")


def test_parse_judgment_recovers_missing_flow_attribution_for_block():
    """A non-pass judge decision should retain a concrete offending flow."""
    result = _parse_judgment(
        {
            "decision": "block",
            "rationale": (
                "The message reveals patent-pending projects at "
                "Quantum Innovations to an external partner."
            ),
            "transmitted_flows": [],
        },
        fallback_flows=[
            InformationFlow(
                data_type="patent-pending projects",
                data_subject="Quantum Innovations",
            ),
            InformationFlow(
                data_type="prototype phase and pending patent approval",
                data_subject="Quantum Innovations Project Alpha",
            ),
        ],
        fallback_recipient="external partner",
    )

    assert result.decision is JudgmentDecision.BLOCK
    assert len(result.transmitted_flows) == 1
    flow = result.transmitted_flows[0]
    assert flow.data_type == "patent-pending projects"
    assert flow.data_subject == "Quantum Innovations"
    assert flow.data_recipient == "external partner"
    assert flow.disposition is FlowDisposition.BLOCK


def test_audit_decision_fires_once_per_read_batch():
    """The auditor fires exactly once per read batch and the inventory
    delta lands on the batch's last ObservationEvent."""
    analyzer = RecordingPrivacyAnalyzer()

    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(
        llm=llm,
        tools=[Tool(name="ReadTool")],
        privacy_analyzer=analyzer,
    )

    collected_events: list = []
    conversation = Conversation(agent=agent, callbacks=[collected_events.append])

    with patch(
        "openhands.sdk.llm.llm.litellm_completion",
        side_effect=lambda *a, **kw: _make_tool_call_response(
            "read_tool", '{"query": "test"}'
        ),
    ):
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text="Search for pages.")],
            )
        )
        agent.step(conversation, on_event=conversation._on_event)

    # audit_decision fired exactly once for the single-read batch.
    assert len(analyzer._audit_calls) == 1
    n_reads, n_accum, ctx = analyzer._audit_calls[0]
    assert n_reads == 1
    assert n_accum == 0
    assert ctx.data_recipient == ""
    # judge_write_action should NOT have fired on a read-only batch
    assert len(analyzer._judge_calls) == 0

    # The single ObservationEvent should carry information_flows from
    # the auditor's inventory_delta.
    obs_events = [e for e in collected_events if isinstance(e, ObservationEvent)]
    assert len(obs_events) == 1
    assert obs_events[0].information_flows is not None
    assert len(obs_events[0].information_flows) == 1
    assert obs_events[0].information_flows[0].data_type == "medical diagnosis"
    assert obs_events[0].information_flows[0].data_subject == "Alice"


def test_read_boundary_audit_receives_expected_recipient_and_channel():
    """Planning-time audience context reaches the read-boundary auditor."""
    analyzer = RecordingPrivacyAnalyzer()

    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(
        llm=llm,
        tools=[Tool(name="ReadTool")],
        privacy_analyzer=analyzer,
        privacy_expected_recipient="Mike",
        privacy_expected_channel="Mattermost DM",
    )

    conversation = Conversation(agent=agent, callbacks=[lambda _event: None])

    with patch(
        "openhands.sdk.llm.llm.litellm_completion",
        side_effect=lambda *a, **kw: _make_tool_call_response(
            "read_tool", '{"query": "test"}'
        ),
    ):
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text="Search for pages.")],
            )
        )
        agent.step(conversation, on_event=conversation._on_event)

    assert len(analyzer._audit_calls) == 1
    _n_reads, _n_accum, ctx = analyzer._audit_calls[0]
    assert ctx.data_recipient == "Mike"
    assert ctx.transmission_channel == "Mattermost DM"


def test_audit_decision_skips_pure_write_batch():
    """The auditor does NOT fire when the batch contains only writes."""
    analyzer = RecordingPrivacyAnalyzer()

    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(
        llm=llm,
        tools=[Tool(name="WriteTool")],
        privacy_analyzer=analyzer,
    )

    collected_events: list = []
    conversation = Conversation(agent=agent, callbacks=[collected_events.append])

    with patch(
        "openhands.sdk.llm.llm.litellm_completion",
        side_effect=lambda *a, **kw: _make_tool_call_response(
            "write_tool", '{"message": "hello"}'
        ),
    ):
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text="Send a message.")],
            )
        )
        agent.step(conversation, on_event=conversation._on_event)

    # audit_decision should NOT have been called -- nothing to audit.
    assert analyzer._audit_calls == []

    # ObservationEvent (if the judge passed an empty inventory) should
    # have information_flows = None because the audit hook didn't fire.
    obs_events = [e for e in collected_events if isinstance(e, ObservationEvent)]
    assert len(obs_events) == 1
    assert obs_events[0].information_flows is None


def test_audit_instruction_attached_to_read_observation():
    """When the auditor returns an instruction, it is attached to the read
    ObservationEvent instead of emitted as a user-role MessageEvent."""
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[
                InformationFlow(
                    data_type="recent cancer diagnosis",
                    data_subject="Emily",
                )
            ],
            instruction=(
                "Mike has asked only about tomorrow's schedule; keep your "
                "reply scoped to scheduling and omit unrelated medical context."
            ),
        )
    )

    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(
        llm=llm,
        tools=[Tool(name="ReadTool")],
        privacy_analyzer=analyzer,
    )

    collected_events: list = []
    conversation = Conversation(agent=agent, callbacks=[collected_events.append])

    with patch(
        "openhands.sdk.llm.llm.litellm_completion",
        side_effect=lambda *a, **kw: _make_tool_call_response(
            "read_tool", '{"query": "Emily"}'
        ),
    ):
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text="Find Emily's notes.")],
            )
        )
        agent.step(conversation, on_event=conversation._on_event)

    auditor_msgs = [
        e
        for e in collected_events
        if isinstance(e, MessageEvent) and e.sender == PRIVACY_AUDITOR_SENDER
    ]
    assert auditor_msgs == []
    obs_events = [e for e in collected_events if isinstance(e, ObservationEvent)]
    assert len(obs_events) == 1
    assert obs_events[0].audit_instruction is not None
    assert "scheduling" in obs_events[0].audit_instruction


def test_write_enforce_only_keeps_read_flows_internal():
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[
                InformationFlow(
                    data_type="recent cancer diagnosis",
                    data_subject="Emily",
                    disposition=FlowDisposition.BLOCK,
                )
            ],
            instruction="Omit Emily's diagnosis from the reply.",
        )
    )
    agent = Agent(
        llm=LLM(
            usage_id="test-llm",
            model="test-model",
            api_key=SecretStr("test-key"),
            base_url="http://test",
        ),
        tools=[Tool(name="ReadTool")],
        privacy_analyzer=analyzer,
        privacy_audit_mode=PrivacyAuditMode.WRITE_ENFORCE_ONLY,
    )
    collected_events: list = []
    conversation = Conversation(agent=agent, callbacks=[collected_events.append])

    with patch(
        "openhands.sdk.llm.llm.litellm_completion",
        side_effect=lambda *a, **kw: _make_tool_call_response(
            "read_tool", '{"query": "Emily"}'
        ),
    ):
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text="Find Emily's notes.")],
            )
        )
        agent.step(conversation, on_event=conversation._on_event)

    obs_events = [e for e in collected_events if isinstance(e, ObservationEvent)]
    assert len(obs_events) == 1
    read_observation = obs_events[0]
    assert read_observation.information_flows is not None
    assert read_observation.information_flows[0].data_subject == "Emily"
    assert read_observation.information_flows_visible_to_llm is False
    assert read_observation.audit_instruction is None
    rendered = _message_text(read_observation.to_llm_message())
    assert "Alice has a medical diagnosis of flu." in rendered
    assert "[INFORMATION INVENTORY]" not in rendered
    assert "recent cancer diagnosis" not in rendered


def test_audit_context_reaches_next_planning_turn_and_changes_write():
    """The next planning turn sees read inventory + auditor instruction.

    The test LLM intentionally writes a leaky message unless both audit signals
    are visible in its second prompt. This verifies the intermediate context
    that can affect the subsequent write action.
    """
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[
                InformationFlow(
                    data_type="medical diagnosis",
                    data_subject="Alice",
                    disposition=FlowDisposition.BLOCK,
                    reason="The recipient only needs scheduling information.",
                )
            ],
            instruction=(
                "Keep the write scoped to scheduling and omit medical context."
            ),
        ),
        judge_result=PrivacyJudgment(decision=JudgmentDecision.PASS),
    )
    llm = ContextAwarePrivacyLLM(model="test-model", usage_id="test-llm")
    agent = Agent(
        llm=llm,
        tools=[Tool(name="ReadTool"), Tool(name="WriteTool")],
        privacy_analyzer=analyzer,
    )

    collected_events: list = []
    conversation = Conversation(agent=agent, callbacks=[collected_events.append])
    conversation.send_message(
        Message(
            role="user",
            content=[TextContent(text="Read Alice's note and send an update.")],
        )
    )

    agent.step(conversation, on_event=conversation._on_event)
    agent.step(conversation, on_event=conversation._on_event)

    assert llm.call_count == 2
    assert llm.captured_kwargs[0]["parallel_tool_calls"] is False
    assert llm.captured_kwargs[1]["parallel_tool_calls"] is False

    second_messages = llm.captured_messages[1]
    read_tool_idx = next(
        i
        for i, message in enumerate(second_messages)
        if message.role == "tool" and message.name == "read_tool"
    )
    read_tool_text = _message_text(second_messages[read_tool_idx])
    assert "Alice has a medical diagnosis of flu." in read_tool_text
    assert "[INFORMATION INVENTORY]" in read_tool_text
    assert "medical diagnosis" in read_tool_text
    assert "subject: Alice" in read_tool_text
    assert "disposition: block" in read_tool_text
    assert "only needs scheduling information" in read_tool_text

    assert "[PRIVACY AUDITOR GUIDANCE]" in read_tool_text
    assert "omit medical context" in read_tool_text
    assert not any(
        message.role == "user" and "omit medical context" in _message_text(message)
        for message in second_messages
    )

    assert len(analyzer._audit_calls) == 1
    assert len(analyzer._judge_calls) == 1
    write_content, n_flows, _recipient, _principal = analyzer._judge_calls[0]
    assert n_flows == 1
    assert "Scheduling update only" in write_content
    assert "medical diagnosis" not in write_content
    assert "flu" not in write_content

    write_obs = [
        event
        for event in collected_events
        if isinstance(event, ObservationEvent) and event.tool_name == "write_tool"
    ]
    assert len(write_obs) == 1


def test_audit_no_instruction_means_no_user_message():
    """When the auditor returns instruction=None, no auditor MessageEvent
    is emitted but ObservationEvents still get information_flows."""
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[
                InformationFlow(
                    data_type="own sprint progress",
                    data_subject="John",
                )
            ],
            instruction=None,
        )
    )

    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(
        llm=llm,
        tools=[Tool(name="ReadTool")],
        privacy_analyzer=analyzer,
    )

    collected_events: list = []
    conversation = Conversation(agent=agent, callbacks=[collected_events.append])

    with patch(
        "openhands.sdk.llm.llm.litellm_completion",
        side_effect=lambda *a, **kw: _make_tool_call_response(
            "read_tool", '{"query": "own notes"}'
        ),
    ):
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text="Read my own notes.")],
            )
        )
        agent.step(conversation, on_event=conversation._on_event)

    # No auditor MessageEvent and no attached auditor instruction.
    auditor_msgs = [
        e
        for e in collected_events
        if isinstance(e, MessageEvent) and e.sender == PRIVACY_AUDITOR_SENDER
    ]
    assert auditor_msgs == []

    # But the ObservationEvent still carries the inventory delta.
    obs_events = [e for e in collected_events if isinstance(e, ObservationEvent)]
    assert len(obs_events) == 1
    assert obs_events[0].information_flows is not None
    assert obs_events[0].audit_instruction is None
    assert obs_events[0].information_flows[0].data_subject == "John"


def test_no_privacy_analyzer_skips_all_privacy_logic():
    """Without a privacy_analyzer, no audit_decision call happens."""
    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(
        llm=llm,
        tools=[Tool(name="ReadTool")],
        # No privacy_analyzer
    )

    collected_events: list = []
    conversation = Conversation(agent=agent, callbacks=[collected_events.append])

    with patch(
        "openhands.sdk.llm.llm.litellm_completion",
        side_effect=lambda *a, **kw: _make_tool_call_response(
            "read_tool", '{"query": "test"}'
        ),
    ):
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text="Search for pages.")],
            )
        )
        agent.step(conversation, on_event=conversation._on_event)

    # ObservationEvent should have information_flows = None
    obs_events = [e for e in collected_events if isinstance(e, ObservationEvent)]
    assert len(obs_events) == 1
    assert obs_events[0].information_flows is None
    # No auditor MessageEvent either.
    auditor_msgs = [
        e
        for e in collected_events
        if isinstance(e, MessageEvent) and e.sender == PRIVACY_AUDITOR_SENDER
    ]
    assert auditor_msgs == []


def test_privacy_schema_self_report_mode_preserves_legacy_tool_fields():
    """When enabled, privacy analyzer still adds CI self-report schema fields."""
    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(llm=llm, tools=[], privacy_analyzer=RecordingPrivacyAnalyzer())

    assert agent._should_add_privacy_context_to_tools()


def test_privacy_schema_none_mode_disables_self_report_tool_fields():
    """External audit can run without asking the agent to self-report CI fields."""
    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(
        llm=llm,
        tools=[],
        privacy_analyzer=RecordingPrivacyAnalyzer(),
        privacy_schema_mode=PrivacySchemaMode.NONE,
    )

    assert not agent._should_add_privacy_context_to_tools()


def test_privacy_sequential_tool_calls_requests_single_provider_call():
    """L3 asks the provider to avoid returning batched tool calls."""
    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(
        llm=llm,
        tools=[Tool(name="ReadTool")],
        privacy_analyzer=RecordingPrivacyAnalyzer(),
    )

    captured_kwargs = {}
    collected_events: list = []
    conversation = Conversation(agent=agent, callbacks=[collected_events.append])

    def fake_completion(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return _make_tool_call_response("read_tool", '{"query": "test"}')

    with patch(
        "openhands.sdk.llm.llm.litellm_completion",
        side_effect=fake_completion,
    ):
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text="Search for pages.")],
            )
        )
        agent.step(conversation, on_event=conversation._on_event)

    assert captured_kwargs["parallel_tool_calls"] is False


def test_accumulated_flows_are_deduplicated():
    """Duplicate flows from overlapping searches are merged before write check.

    Directly tests ``_gather_accumulated_flows`` by injecting
    ObservationEvents with overlapping information_flows into the
    conversation state.
    """
    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    analyzer = RecordingPrivacyAnalyzer()
    agent = Agent(
        llm=llm,
        tools=[Tool(name="ReadTool")],
        privacy_analyzer=analyzer,
    )

    conversation = Conversation(agent=agent)

    # Simulate two read observations with overlapping flows
    obs1 = ObservationEvent(
        observation=ReadObservation(),
        action_id="act-1",
        tool_name="read_tool",
        tool_call_id="call-1",
        information_flows=[
            InformationFlow(
                data_type="medical diagnosis",
                data_subject="Alice",
                disposition=FlowDisposition.GENERALIZE,
                reason="Share availability without diagnosis details.",
            ),
            InformationFlow(data_type="salary", data_subject="Bob"),
        ],
    )
    obs2 = ObservationEvent(
        observation=ReadObservation(),
        action_id="act-2",
        tool_name="read_tool",
        tool_call_id="call-2",
        information_flows=[
            # Duplicate of obs1
            InformationFlow(data_type="medical diagnosis", data_subject="Alice"),
            # Same fact, different label ordering
            InformationFlow(
                data_type="diagnosis (medical)",
                data_subject="Alice",
            ),
            # New unique flow
            InformationFlow(data_type="home address", data_subject="Carol"),
        ],
    )

    # Inject events into conversation state
    conversation._state._events.append(obs1)
    conversation._state._events.append(obs2)

    accumulated = agent._gather_accumulated_flows(conversation)

    # Should have 3 unique flows, not 5 raw
    assert len(accumulated) == 3
    types = {(f.data_type, f.data_subject) for f in accumulated}
    assert types == {
        ("medical diagnosis", "Alice"),
        ("salary", "Bob"),
        ("home address", "Carol"),
    }


# ── Tests for flow-annotated observations ──


def test_flow_annotated_observation_message():
    """to_llm_message() includes INFORMATION INVENTORY annotation."""
    obs_event = ObservationEvent(
        observation=ReadObservation(),
        action_id="act-1",
        tool_name="read_tool",
        tool_call_id="call-1",
        information_flows=[
            InformationFlow(
                data_type="medical diagnosis",
                data_subject="Alice",
                disposition=FlowDisposition.GENERALIZE,
                reason="Share availability without diagnosis details.",
            ),
            InformationFlow(data_type="salary", data_subject="Bob"),
        ],
    )
    msg = obs_event.to_llm_message()

    # Should have 2 content blocks: observation + annotation
    assert len(msg.content) == 2
    annotation = msg.content[1]
    assert isinstance(annotation, TextContent)
    assert "[INFORMATION INVENTORY]" in annotation.text
    assert "medical diagnosis (subject: Alice)" in annotation.text
    assert "disposition: abstract" in annotation.text
    assert "Share availability without diagnosis details." in annotation.text
    assert "salary (subject: Bob)" in annotation.text
    assert "disposition: pass" in annotation.text


def test_observation_message_includes_attached_audit_instruction():
    obs_event = ObservationEvent(
        observation=ReadObservation(),
        action_id="act-1",
        tool_name="read_tool",
        tool_call_id="call-1",
        audit_instruction="Keep the reply scoped to scheduling.",
    )
    msg = obs_event.to_llm_message()

    assert len(msg.content) == 2
    instruction = msg.content[1]
    assert isinstance(instruction, TextContent)
    assert "[PRIVACY AUDITOR GUIDANCE]" in instruction.text
    assert "Keep the reply scoped to scheduling." in instruction.text


def test_write_judgment_metadata_is_not_exposed_to_execution_model():
    obs_event = ObservationEvent(
        observation=ReadObservation(),
        action_id="act-1",
        tool_name="write_tool",
        tool_call_id="call-1",
        privacy_check_result=PrivacyJudgment(
            decision=JudgmentDecision.BLOCK,
            rationale="Do not expose this internal audit decision.",
        ),
    )

    msg = obs_event.to_llm_message()

    assert len(msg.content) == 1
    content = msg.content[0]
    assert isinstance(content, TextContent)
    assert "internal audit decision" not in content.text


def test_no_annotation_when_no_flows():
    """No annotation when flows are empty or None."""
    obs_none = ObservationEvent(
        observation=ReadObservation(),
        action_id="act-1",
        tool_name="read_tool",
        tool_call_id="call-1",
        information_flows=None,
    )
    msg_none = obs_none.to_llm_message()
    assert len(msg_none.content) == 1  # just the observation

    obs_empty = ObservationEvent(
        observation=ReadObservation(),
        action_id="act-2",
        tool_name="read_tool",
        tool_call_id="call-2",
        information_flows=[],
    )
    msg_empty = obs_empty.to_llm_message()
    assert len(msg_empty.content) == 1  # just the observation


# ── Helpers for ActionEvent construction ──


def _make_action_event(tool_name: str, action: Action) -> ActionEvent:
    """Build an ActionEvent with required fields for testing."""
    return ActionEvent(
        thought=[],
        action=action,
        tool_name=tool_name,
        tool_call_id="call-1",
        tool_call=MessageToolCall(
            id="call-1",
            name=tool_name,
            arguments="{}",
            origin="completion",
        ),
        llm_response_id="resp-1",
    )


# ── Tests for _extract_write_content ──


def test_extract_write_content_message_field():
    """_extract_write_content extracts the 'message' field from MCP action data."""
    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    analyzer = RecordingPrivacyAnalyzer()
    agent = Agent(llm=llm, tools=[], privacy_analyzer=analyzer)

    action = MCPToolAction(data={"recipient": "alice", "message": "Hello Alice"})
    ae = _make_action_event("send_message", action)
    result = agent._extract_write_content(ae)
    assert result == "Hello Alice"


def test_extract_write_content_body_field():
    """_extract_write_content extracts the 'body' field for email tools."""
    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    analyzer = RecordingPrivacyAnalyzer()
    agent = Agent(llm=llm, tools=[], privacy_analyzer=analyzer)

    action = MCPToolAction(
        data={"to": "bob@example.com", "subject": "Hi", "body": "Dear Bob..."}
    )
    ae = _make_action_event("send_email", action)
    result = agent._extract_write_content(ae)
    assert result == "Dear Bob..."


def test_extract_write_content_fallback_serializes_all():
    """Falls back to serializing all data when no known field."""
    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    analyzer = RecordingPrivacyAnalyzer()
    agent = Agent(llm=llm, tools=[], privacy_analyzer=analyzer)

    action = MCPToolAction(data={"channel": "general", "text_payload": "hello"})
    ae = _make_action_event("custom_tool", action)
    result = agent._extract_write_content(ae)
    assert "text_payload" in result
    assert "hello" in result


# ── Tests for write-time judgment control flow (mixed batches) ──


def _mixed_read_write_response() -> ModelResponse:
    """Build an LLM response batching one read and one write tool call."""
    return ModelResponse(
        id="mock-response",
        choices=[
            Choices(
                index=0,
                message=LiteLLMMessage(
                    role="assistant",
                    content="Reading and writing.",
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_read",
                            type="function",
                            function=Function(
                                name="read_tool",
                                arguments='{"query": "test"}',
                            ),
                        ),
                        ChatCompletionMessageToolCall(
                            id="call_write",
                            type="function",
                            function=Function(
                                name="write_tool",
                                arguments='{"message": "hello"}',
                            ),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def _run_mixed_batch(
    analyzer: RecordingPrivacyAnalyzer,
    *,
    privacy_sequential_tool_calls: bool = False,
) -> list:
    """Drive a mixed read+write batch through one agent step.

    Returns the list of events emitted, so tests can look for the
    ObservationEvent or UserRejectObservation produced for the write.
    """
    llm = LLM(
        usage_id="test-llm",
        model="test-model",
        api_key=SecretStr("test-key"),
        base_url="http://test",
    )
    agent = Agent(
        llm=llm,
        tools=[Tool(name="ReadTool"), Tool(name="WriteTool")],
        privacy_analyzer=analyzer,
        privacy_sequential_tool_calls=privacy_sequential_tool_calls,
    )

    collected_events = []
    conversation = Conversation(agent=agent, callbacks=[collected_events.append])

    with patch(
        "openhands.sdk.llm.llm.litellm_completion",
        side_effect=lambda *a, **kw: _mixed_read_write_response(),
    ):
        conversation.send_message(
            Message(
                role="user",
                content=[TextContent(text="Read and send.")],
            )
        )
        agent.step(conversation, on_event=conversation._on_event)

    return collected_events


def test_privacy_sequential_tool_calls_defers_extra_batch_calls():
    """If a provider still returns a batch, L3 executes only the first call."""
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[
                InformationFlow(
                    data_type="medical diagnosis",
                    data_subject="Alice",
                )
            ],
            instruction="Keep the reply scoped to the user's request.",
        ),
        judge_result=PrivacyJudgment(decision=JudgmentDecision.PASS),
    )

    events = _run_mixed_batch(analyzer, privacy_sequential_tool_calls=True)

    assert len(analyzer._audit_calls) == 1
    assert analyzer._judge_calls == []

    read_obs_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, ObservationEvent) and e.tool_name == "read_tool"
    )
    reject_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, UserRejectObservation) and e.tool_name == "write_tool"
    )
    read_obs = events[read_obs_idx]
    assert isinstance(read_obs, ObservationEvent)
    assert read_obs.audit_instruction is not None
    assert "user's request" in read_obs.audit_instruction
    assert read_obs_idx < reject_idx
    assert not any(
        isinstance(e, MessageEvent) and e.sender == PRIVACY_AUDITOR_SENDER
        for e in events
    )

    write_obs = [
        e
        for e in events
        if isinstance(e, ObservationEvent) and e.tool_name == "write_tool"
    ]
    assert write_obs == []

    rejects = [
        e
        for e in events
        if isinstance(e, UserRejectObservation) and e.tool_name == "write_tool"
    ]
    assert len(rejects) == 1
    assert "deferred" in rejects[0].rejection_reason
    assert rejects[0].rejection_source == "hook"

    messages = LLMConvertibleEvent.events_to_messages(
        [e for e in events if isinstance(e, LLMConvertibleEvent)]
    )
    assistant_idx = next(i for i, m in enumerate(messages) if m.tool_calls)
    assert messages[assistant_idx + 1].role == "tool"
    assert messages[assistant_idx + 2].role == "tool"
    assert "[PRIVACY AUDITOR GUIDANCE]" in _message_text(messages[assistant_idx + 1])
    assert not any(
        message.role == "user" and "user's request" in _message_text(message)
        for message in messages[assistant_idx + 1 :]
    )


def test_judge_pass_executes_write():
    """When the judge returns PASS, the write tool executes normally."""
    analyzer = RecordingPrivacyAnalyzer(
        judge_result=PrivacyJudgment(decision=JudgmentDecision.PASS)
    )

    events = _run_mixed_batch(analyzer)

    # The decision-boundary auditor fired exactly once for the read batch.
    assert len(analyzer._audit_calls) == 1

    # Judge fired once for the write, after the auditor accumulated flows.
    assert len(analyzer._judge_calls) == 1
    write_content, n_flows, recipient, principal = analyzer._judge_calls[0]
    # WriteTool here is not an MCP tool, so _extract_write_content falls
    # back to a full model_dump of the action. The message text must be in
    # there somewhere.
    assert "hello" in write_content
    assert n_flows >= 1  # at least the one flow returned by the auditor
    assert recipient  # defaulted to tool name when no recipient field
    # No principal configured on the agent -> empty string
    assert principal == ""

    # The write tool should have executed -> ObservationEvent for write_tool.
    write_obs = [
        e
        for e in events
        if isinstance(e, ObservationEvent) and e.tool_name == "write_tool"
    ]
    assert len(write_obs) == 1
    # The judgment is persisted on the observation for post-hoc analysis.
    assert write_obs[0].privacy_check_result is not None
    assert write_obs[0].privacy_check_result.decision is JudgmentDecision.PASS

    # No reject for the write.
    rejects = [
        e
        for e in events
        if isinstance(e, UserRejectObservation) and e.tool_name == "write_tool"
    ]
    assert rejects == []


def test_mixed_batch_instruction_attaches_to_read_tool_response():
    """A mixed read+write assistant message keeps only tool responses after
    the assistant tool call and carries auditor guidance in the read result.
    """
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[
                InformationFlow(
                    data_type="medical diagnosis",
                    data_subject="Alice",
                )
            ],
            instruction="Keep the reply scoped to the user's request.",
        ),
        judge_result=PrivacyJudgment(decision=JudgmentDecision.PASS),
    )

    events = _run_mixed_batch(analyzer)

    read_obs_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, ObservationEvent) and e.tool_name == "read_tool"
    )
    write_obs_idx = next(
        i
        for i, e in enumerate(events)
        if isinstance(e, ObservationEvent) and e.tool_name == "write_tool"
    )
    read_obs = events[read_obs_idx]
    assert isinstance(read_obs, ObservationEvent)
    assert read_obs.audit_instruction is not None
    assert "user's request" in read_obs.audit_instruction
    assert read_obs_idx < write_obs_idx
    assert not any(
        isinstance(e, MessageEvent) and e.sender == PRIVACY_AUDITOR_SENDER
        for e in events
    )

    messages = LLMConvertibleEvent.events_to_messages(
        [e for e in events if isinstance(e, LLMConvertibleEvent)]
    )
    assistant_idx = next(i for i, m in enumerate(messages) if m.tool_calls)
    assert messages[assistant_idx + 1].role == "tool"
    assert messages[assistant_idx + 2].role == "tool"
    assert "[PRIVACY AUDITOR GUIDANCE]" in _message_text(messages[assistant_idx + 1])
    assert not any(
        message.role == "user" and "user's request" in _message_text(message)
        for message in messages[assistant_idx + 1 :]
    )


def test_judge_abstract_short_circuits_with_rationale():
    """ABSTRACT: the write is rejected with the judge's rationale."""
    judgment = PrivacyJudgment(
        transmitted_flows=[
            TransmittedFlow(
                data_type="medical diagnosis",
                data_subject="Alice",
                data_recipient="public post",
                disposition=FlowDisposition.GENERALIZE,
            )
        ],
        decision=JudgmentDecision.ABSTRACT,
        rationale="A public post is a broader audience than a medical fact warrants.",
    )
    analyzer = RecordingPrivacyAnalyzer(judge_result=judgment)

    events = _run_mixed_batch(analyzer)

    # Judge was consulted.
    assert len(analyzer._judge_calls) == 1

    # No ObservationEvent was produced for the write -- tool did not run.
    write_obs = [
        e
        for e in events
        if isinstance(e, ObservationEvent) and e.tool_name == "write_tool"
    ]
    assert write_obs == []

    # UserRejectObservation with rationale + re-compose directive.
    rejects = [
        e
        for e in events
        if isinstance(e, UserRejectObservation) and e.tool_name == "write_tool"
    ]
    assert len(rejects) == 1
    reason = rejects[0].rejection_reason
    assert "broader audience" in reason  # the judge's rationale
    assert "Re-compose" in reason  # the ABSTRACT directive
    assert "Rewrite constraints:" in reason
    assert "Generalize:" in reason
    assert "Alice: medical diagnosis" in reason
    assert rejects[0].rejection_source == "hook"
    assert rejects[0].privacy_check_result is not None
    assert rejects[0].privacy_check_result.decision is JudgmentDecision.ABSTRACT
    assert rejects[0].privacy_check_result.transmitted_flows == (
        judgment.transmitted_flows
    )


def test_judge_block_short_circuits_with_stronger_directive():
    """BLOCK: the write is rejected with the 'not an appropriate audience' text."""
    judgment = PrivacyJudgment(
        transmitted_flows=[
            TransmittedFlow(
                data_type="medical diagnosis",
                data_subject="Alice",
                data_recipient="public post",
                disposition=FlowDisposition.BLOCK,
            )
        ],
        decision=JudgmentDecision.BLOCK,
        rationale="Alice has not consented to public disclosure of her diagnosis.",
    )
    analyzer = RecordingPrivacyAnalyzer(judge_result=judgment)

    events = _run_mixed_batch(analyzer)

    # No tool execution.
    write_obs = [
        e
        for e in events
        if isinstance(e, ObservationEvent) and e.tool_name == "write_tool"
    ]
    assert write_obs == []

    rejects = [
        e
        for e in events
        if isinstance(e, UserRejectObservation) and e.tool_name == "write_tool"
    ]
    assert len(rejects) == 1
    reason = rejects[0].rejection_reason
    assert "Alice" in reason  # from the rationale
    assert "not an appropriate audience" in reason  # the BLOCK directive
    assert "Rewrite constraints:" in reason
    assert "Block:" in reason
    # BLOCK uses a stronger directive, not the ABSTRACT re-compose one.
    assert "Re-compose" not in reason
    assert rejects[0].privacy_check_result is not None
    assert rejects[0].privacy_check_result.decision is JudgmentDecision.BLOCK
    assert rejects[0].privacy_check_result.transmitted_flows == (
        judgment.transmitted_flows
    )


def test_judge_blinding_removes_read_disposition_and_its_anchor():
    """The write gate can only be shown to add something if it is not handed the
    read boundary's answer plus a note calling it 'a strong signal'."""
    llm = CapturingAuditLLM.from_messages(
        [Message(role="assistant", content=[TextContent(text="{}")])]
    )
    default = LLMPrivacyAnalyzer(llm=llm)
    assert default._judge_system_prompt() == JUDGE_SYSTEM_PROMPT

    # blinding composes with every policy, not just the default
    for policy in AuditPolicy:
        blind = LLMPrivacyAnalyzer(
            llm=llm, audit_policy=policy, judge_blind_to_read_disposition=True
        )._judge_system_prompt()
        assert "strong signal" not in blind
        assert "read-boundary dispositions" not in blind
        assert "No prior disposition" in blind
        # shared scaffolding survives
        assert "STEP 1 — DETECT TRANSMITTED FACTS." in blind
        assert "Assign a disposition to each transmitted item:" in blind

    # the annotations must also leave the rendered inventory, not just the prompt
    flow = InformationFlow(
        data_type="HIV diagnosis",
        data_subject="Bob",
        disposition=FlowDisposition.ABSTRACT,
        reason="not appropriate for this audience",
        safe_rendering="a health matter",
    )
    assert "abstract" in _render_judge_inventory_item(flow)
    blinded = _render_judge_inventory_item(flow, blind=True)
    assert blinded == "- data_type: HIV diagnosis\n  data_subject: Bob"


def test_verify_recompose_block_is_opt_in():
    llm = CapturingAuditLLM.from_messages(
        [Message(role="assistant", content=[TextContent(text="{}")])]
    )
    assert LLMPrivacyAnalyzer(llm=llm)._judge_system_prompt() == JUDGE_SYSTEM_PROMPT
    on = LLMPrivacyAnalyzer(llm=llm, verify_recompose=True)._judge_system_prompt()
    assert on.startswith(JUDGE_SYSTEM_PROMPT)
    assert "RE-COMPOSITION VERIFICATION." in on
    assert '"resolved_items"' in on


def test_recompose_verification_overrides_a_model_pass():
    """The observed failure mode is a retry that passes because the criterion was
    re-applied loosely to a reworded message. Enforcement is in code, so the
    model's own PASS does not decide the outcome."""
    rejected = [
        TransmittedFlow(
            data_type="HIV diagnosis",
            data_subject="Bob",
            data_recipient="team channel",
            disposition=FlowDisposition.OMIT,
        )
    ]
    passing = PrivacyJudgment(decision=JudgmentDecision.PASS, rationale="Looks fine.")

    # model claims pass but reports the item unresolved
    overridden = _enforce_recompose_verification(
        passing,
        {
            "resolved_items": [
                {"data_type": "HIV diagnosis", "data_subject": "Bob", "resolved": False}
            ]
        },
        rejected,
    )
    assert overridden.decision is JudgmentDecision.ABSTRACT
    assert [f.data_type for f in overridden.unresolved_items] == ["HIV diagnosis"]
    assert "Still unresolved" in overridden.rationale

    # silence is not consent: an omitted item counts as unresolved
    silent = _enforce_recompose_verification(passing, {}, rejected)
    assert silent.decision is JudgmentDecision.ABSTRACT
    assert len(silent.unresolved_items) == 1

    # a genuinely resolved item lets the model's decision stand
    cleared = _enforce_recompose_verification(
        passing,
        {
            "resolved_items": [
                {"data_type": "HIV  DIAGNOSIS", "data_subject": "bob", "resolved": True}
            ]
        },
        rejected,
    )
    assert cleared.decision is JudgmentDecision.PASS
    assert cleared.unresolved_items == []
