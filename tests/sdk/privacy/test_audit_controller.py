from types import SimpleNamespace
from typing import cast

from openhands.sdk.conversation import LocalConversation
from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent
from openhands.sdk.event.llm_convertible.observation import UserRejectObservation
from openhands.sdk.llm import MessageToolCall
from openhands.sdk.mcp.definition import MCPToolAction
from openhands.sdk.privacy.analyzer import PrivacyAnalyzerBase
from openhands.sdk.privacy.audit import (
    PRIVACY_AUDITOR_SENDER,
    PrivacyAuditController,
    gather_previously_rejected_items,
)
from openhands.sdk.privacy.config import PrivacyAuditGuidanceMode, PrivacyAuditMode
from openhands.sdk.privacy.flow import (
    AuditDecision,
    FlowDisposition,
    InformationFlow,
    JudgmentDecision,
    PrivacyJudgment,
    TransmissionContext,
    TransmittedFlow,
)
from openhands.sdk.tool.schema import Observation


class DummyObservation(Observation):
    pass


class RecordingPrivacyAnalyzer(PrivacyAnalyzerBase):
    def __init__(
        self,
        audit_result: AuditDecision | None = None,
        judgment: PrivacyJudgment | None = None,
    ):
        super().__init__()
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
        self._judgment = judgment or PrivacyJudgment()

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
        return self._judgment


class ContextRecordingPrivacyAnalyzer(RecordingPrivacyAnalyzer):
    def __init__(self):
        super().__init__()
        self._context_calls: list[TransmissionContext] = []

    def judge_write_action_with_context(
        self,
        write_content: str,
        accumulated_flows: list[InformationFlow],
        transmission_context: TransmissionContext,
    ) -> PrivacyJudgment:
        self._context_calls.append(transmission_context)
        return PrivacyJudgment()


def _conversation_with_events(events: list[object]) -> LocalConversation:
    return cast(
        LocalConversation, SimpleNamespace(state=SimpleNamespace(events=events))
    )


def _observation_event(
    action_id: str,
    flows: list[InformationFlow] | None,
    *,
    tool_name: str = "read_tool",
) -> ObservationEvent:
    return ObservationEvent(
        observation=DummyObservation.from_text("ok"),
        action_id=action_id,
        tool_name=tool_name,
        tool_call_id=f"call-{action_id}",
        information_flows=flows,
    )


def _read_action_event(action_id: str, tool_name: str = "read_tool") -> ActionEvent:
    return ActionEvent(
        thought=[],
        action=MCPToolAction(data={}),
        tool_name=tool_name,
        tool_call_id=f"call-{action_id}",
        tool_call=MessageToolCall(
            id=f"call-{action_id}",
            name=tool_name,
            arguments="{}",
            origin="completion",
        ),
        llm_response_id=f"response-{action_id}",
    )


def _action_event(tool_name: str, action: MCPToolAction) -> ActionEvent:
    return ActionEvent(
        thought=[],
        action=action,
        tool_name=tool_name,
        tool_call_id="call-write",
        tool_call=MessageToolCall(
            id="call-write",
            name=tool_name,
            arguments="{}",
            origin="completion",
        ),
        llm_response_id="response-write",
    )


def test_audit_controller_gathers_deduplicated_flows():
    analyzer = RecordingPrivacyAnalyzer()
    controller = PrivacyAuditController(analyzer=analyzer)
    conversation = _conversation_with_events(
        [
            _observation_event(
                "1",
                [
                    InformationFlow(
                        data_type="medical diagnosis",
                        data_subject="Alice",
                    ),
                    InformationFlow(data_type="salary", data_subject="Bob"),
                ],
            ),
            _observation_event(
                "2",
                [
                    InformationFlow(
                        data_type="diagnosis (medical)",
                        data_subject="Alice",
                    ),
                    InformationFlow(data_type="home address", data_subject="Carol"),
                ],
            ),
        ]
    )

    flows = controller.gather_accumulated_flows(conversation)

    assert {(f.data_type, f.data_subject) for f in flows} == {
        ("medical diagnosis", "Alice"),
        ("salary", "Bob"),
        ("home address", "Carol"),
    }


def test_after_read_batch_skips_metadata_listing_tools():
    analyzer = RecordingPrivacyAnalyzer()
    controller = PrivacyAuditController(analyzer=analyzer)
    conversation = _conversation_with_events([])

    # Build a fake batch where the only read is a listing tool — by tool
    # semantics this is not a content-extraction target, so the auditor
    # should be skipped entirely (no LLM call).
    action = _read_action_event("1", tool_name="bookstack_list_pages")
    obs = _observation_event(
        "1",
        flows=None,
        tool_name="bookstack_list_pages",
    )
    result = controller.after_read_batch(conversation, [(action, obs)])

    assert analyzer._audit_calls == []
    assert result.information_flows_by_action_id == {}
    assert result.instruction_event is None


def test_after_read_batch_attributes_delta_to_last_eligible_read():
    flow = InformationFlow(data_type="therapy progress", data_subject="Ethan")
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(inventory_delta=[flow], instruction=None)
    )
    controller = PrivacyAuditController(
        analyzer=analyzer,
        principal="Jane",
        task_purpose="Daily check-in",
    )
    conversation = _conversation_with_events([])

    actions_obs: list[tuple[ActionEvent, ObservationEvent]] = []
    for i in (1, 2, 3):
        ae = _read_action_event(str(i), tool_name="mailpit_read_email")
        oe = _observation_event(str(i), flows=None, tool_name="mailpit_read_email")
        actions_obs.append((ae, oe))

    result = controller.after_read_batch(conversation, actions_obs)

    assert len(analyzer._audit_calls) == 1
    n_reads, n_accum, ctx = analyzer._audit_calls[0]
    assert n_reads == 3
    assert n_accum == 0
    assert ctx.principal == "Jane"
    assert ctx.task_purpose == "Daily check-in"
    assert ctx.data_recipient == ""

    last_action_id = actions_obs[-1][0].id
    assert list(result.information_flows_by_action_id.keys()) == [last_action_id]
    attributed = result.information_flows_by_action_id[last_action_id]
    assert [(f.data_type, f.data_subject) for f in attributed] == [
        (flow.data_type, flow.data_subject)
    ]
    # Provenance spans the whole eligible batch: the auditor is handed the
    # batch's observations together and never attributes a flow to one read, so
    # narrowing this to a single read would be inventing attribution.
    assert attributed[0].source_action_ids == [ae.id for ae, _ in actions_obs]
    assert attributed[0].source_tools == ["mailpit_read_email"] * 3
    assert result.instruction_event is None


def test_after_read_batch_passes_expected_recipient_and_channel():
    analyzer = RecordingPrivacyAnalyzer()
    controller = PrivacyAuditController(
        analyzer=analyzer,
        principal="Jane",
        task_purpose="Message Mike about tomorrow's schedule.",
        expected_recipient="Mike",
        expected_channel="Mattermost DM",
    )
    conversation = _conversation_with_events([])

    action = _read_action_event("1", tool_name="mailpit_read_email")
    obs = _observation_event("1", flows=None, tool_name="mailpit_read_email")
    controller.after_read_batch(conversation, [(action, obs)])

    assert len(analyzer._audit_calls) == 1
    _n_reads, _n_accum, ctx = analyzer._audit_calls[0]
    assert ctx.principal == "Jane"
    assert ctx.task_purpose == "Message Mike about tomorrow's schedule."
    assert ctx.data_recipient == "Mike"
    assert ctx.transmission_channel == "Mattermost DM"
    assert ctx.tool_name == ""


def test_after_read_batch_emits_instruction_message_when_present():
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[],
            instruction="Keep your reply scoped to the scheduling question.",
        )
    )
    controller = PrivacyAuditController(analyzer=analyzer, principal="Jane")
    conversation = _conversation_with_events([])

    action = _read_action_event("1", tool_name="mailpit_read_email")
    obs = _observation_event("1", flows=None, tool_name="mailpit_read_email")
    result = controller.after_read_batch(conversation, [(action, obs)])

    assert isinstance(result.instruction_event, MessageEvent)
    assert result.instruction_event.source == "user"
    assert result.instruction_event.sender == PRIVACY_AUDITOR_SENDER
    content_block = result.instruction_event.llm_message.content[0]
    from openhands.sdk.llm import TextContent

    assert isinstance(content_block, TextContent)
    assert "scheduling question" in content_block.text


def test_after_read_batch_strips_blank_instruction():
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[],
            instruction="   \n  ",
        )
    )
    controller = PrivacyAuditController(analyzer=analyzer, principal="Jane")
    conversation = _conversation_with_events([])

    action = _read_action_event("1", tool_name="mailpit_read_email")
    obs = _observation_event("1", flows=None, tool_name="mailpit_read_email")
    result = controller.after_read_batch(conversation, [(action, obs)])

    assert result.instruction_event is None


def test_write_enforce_only_extracts_flows_without_read_steering():
    flow = InformationFlow(data_type="medical diagnosis", data_subject="Alice")
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[flow],
            instruction="Do not share the diagnosis.",
        )
    )
    controller = PrivacyAuditController(
        analyzer=analyzer,
        audit_mode=PrivacyAuditMode.WRITE_ENFORCE_ONLY,
    )
    conversation = _conversation_with_events([])
    action = _read_action_event("1", tool_name="mailpit_read_email")
    obs = _observation_event("1", flows=None, tool_name="mailpit_read_email")

    result = controller.after_read_batch(conversation, [(action, obs)])

    assert list(result.information_flows_by_action_id) == [action.id]
    (attributed,) = result.information_flows_by_action_id[action.id]
    assert (attributed.data_type, attributed.data_subject) == (
        flow.data_type,
        flow.data_subject,
    )
    # One eligible read in the batch, so batch-level provenance is exact -- the
    # usual case, since the default sequential mode runs one tool per turn.
    assert attributed.source_action_ids == [action.id]
    assert attributed.source_tools == ["mailpit_read_email"]
    assert result.instruction_event is None
    assert result.hide_information_flows_from_llm is True


def test_after_read_batch_annotated_flow_mode_injects_structured_flows():
    flow = InformationFlow(
        data_type="detailed class schedule with room numbers",
        data_subject="Timothy",
        disposition=FlowDisposition.ABSTRACT,
        safe_use="Mention school involvement generally; omit rooms and times.",
        reason="Detailed minor schedule is not needed for this audience.",
    )
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[flow],
            instruction="Keep the reply scoped to school involvement.",
        )
    )
    controller = PrivacyAuditController(
        analyzer=analyzer,
        guidance_mode=PrivacyAuditGuidanceMode.ANNOTATED_FLOWS,
    )
    conversation = _conversation_with_events([])

    action = _read_action_event("1", tool_name="mailpit_read_email")
    obs = _observation_event("1", flows=None, tool_name="mailpit_read_email")
    result = controller.after_read_batch(conversation, [(action, obs)])

    assert result.instruction_event is not None
    content_block = result.instruction_event.llm_message.content[0]
    from openhands.sdk.llm import TextContent

    assert isinstance(content_block, TextContent)
    assert "<AUDITOR_STEERING_NOTE>" in content_block.text
    assert "<AUDITED_INFORMATION_FLOWS>" in content_block.text
    assert "disposition: abstract" in content_block.text
    assert "omit rooms and times" in content_block.text


def test_after_read_batch_plan_mode_requires_write_privacy_plan():
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[
                InformationFlow(
                    data_type="account password",
                    data_subject="Emily",
                    disposition=FlowDisposition.BLOCK,
                )
            ],
            instruction=None,
        )
    )
    controller = PrivacyAuditController(
        analyzer=analyzer,
        guidance_mode=PrivacyAuditGuidanceMode.ANNOTATED_FLOWS_WITH_PLAN,
    )
    conversation = _conversation_with_events([])

    action = _read_action_event("1", tool_name="mailpit_read_email")
    obs = _observation_event("1", flows=None, tool_name="mailpit_read_email")
    result = controller.after_read_batch(conversation, [(action, obs)])

    assert result.instruction_event is not None
    content_block = result.instruction_event.llm_message.content[0]
    from openhands.sdk.llm import TextContent

    assert isinstance(content_block, TextContent)
    assert "WRITE_PRIVACY_PLAN" in content_block.text
    assert "Do not include this flow" in content_block.text


def test_after_read_batch_contextual_mode_injects_share_decisions():
    analyzer = RecordingPrivacyAnalyzer(
        audit_result=AuditDecision(
            inventory_delta=[
                InformationFlow(
                    data_type="Lisinopril 10mg daily",
                    data_subject="David Smith",
                    disposition=FlowDisposition.ABSTRACT,
                    share_question=(
                        "Is it appropriate for Jane to share David Smith's "
                        "exact medication dosage with Mary by email?"
                    ),
                    safe_rendering=(
                        "David's treatment plan has been updated; "
                        "medication-specific details should go through "
                        "the care team."
                    ),
                    reason="The expected audience needs an update, not dosage.",
                    semantic_recipient="Mary",
                    semantic_channel="email",
                    semantic_recipient_role="treatment update recipient",
                    context_confidence="high",
                    context_evidence="The task says to email Mary.",
                )
            ],
            instruction="Use safe renderings for medical details.",
        )
    )
    controller = PrivacyAuditController(
        analyzer=analyzer,
        guidance_mode=PrivacyAuditGuidanceMode.CONTEXTUAL_SHARE_DECISIONS,
    )
    conversation = _conversation_with_events([])

    action = _read_action_event("1", tool_name="mailpit_read_email")
    obs = _observation_event("1", flows=None, tool_name="mailpit_read_email")
    result = controller.after_read_batch(conversation, [(action, obs)])

    assert result.instruction_event is not None
    content_block = result.instruction_event.llm_message.content[0]
    from openhands.sdk.llm import TextContent

    assert isinstance(content_block, TextContent)
    assert "<CONTEXTUAL_SHARE_DECISIONS>" in content_block.text
    assert "Is it appropriate for Jane" in content_block.text
    assert "inferred_recipient: Mary" in content_block.text
    assert "inferred_channel: email" in content_block.text
    assert "recipient_role: treatment update recipient" in content_block.text
    assert "context_confidence: high" in content_block.text
    assert "context_evidence: The task says to email Mary." in content_block.text
    assert "decision: abstract" in content_block.text
    assert "David's treatment plan has been updated" in content_block.text


def test_audit_controller_passes_write_context_to_judge():
    analyzer = RecordingPrivacyAnalyzer()
    controller = PrivacyAuditController(
        analyzer=analyzer,
        principal="Jane",
        task_purpose="Email Bob an update.",
    )
    conversation = _conversation_with_events(
        [
            _observation_event(
                "1",
                [
                    InformationFlow(
                        data_type="medical diagnosis",
                        data_subject="Alice",
                    )
                ],
            )
        ]
    )
    action_event = _action_event(
        "mailpit_send_email",
        MCPToolAction(data={"to": "bob@example.com", "body": "Alice has flu."}),
    )

    judgment = controller.judge_write_action(conversation, action_event)

    assert judgment is not None
    assert judgment.decision == PrivacyJudgment().decision
    assert judgment.transmission_context is not None
    assert judgment.transmission_context.task_purpose == "Email Bob an update."
    assert analyzer._judge_calls == [
        ("Alice has flu.", 1, "mailpit_send_email (to=bob@example.com)", "Jane")
    ]


def test_audit_controller_prefers_structured_transmission_context():
    analyzer = ContextRecordingPrivacyAnalyzer()
    controller = PrivacyAuditController(
        analyzer=analyzer,
        principal="Jane",
        task_purpose="Send the caregiver a care update.",
    )
    conversation = _conversation_with_events(
        [
            _observation_event(
                "1",
                [
                    InformationFlow(
                        data_type="therapy progress",
                        data_subject="Ethan",
                    )
                ],
            )
        ]
    )
    action_event = _action_event(
        "mailpit_send_email",
        MCPToolAction(data={"to": "lisa@example.com", "body": "Ethan improved."}),
    )

    judgment = controller.judge_write_action(conversation, action_event)

    assert judgment is not None
    assert len(analyzer._context_calls) == 1
    context = analyzer._context_calls[0]
    assert context.data_recipient == "to=lisa@example.com"
    assert context.transmission_channel == "mailpit_send_email"
    assert context.principal == "Jane"
    assert context.task_purpose == "Send the caregiver a care update."
    assert judgment.transmission_context == context


def test_read_steer_only_records_non_pass_judgment_without_rejecting_write():
    judgment = PrivacyJudgment(
        decision=JudgmentDecision.ABSTRACT,
        rationale="Use a higher-level health update.",
    )
    analyzer = RecordingPrivacyAnalyzer(judgment=judgment)
    controller = PrivacyAuditController(
        analyzer=analyzer,
        audit_mode=PrivacyAuditMode.READ_STEER_ONLY,
    )
    conversation = _conversation_with_events(
        [
            _observation_event(
                "1",
                [
                    InformationFlow(
                        data_type="medical diagnosis",
                        data_subject="Alice",
                    )
                ],
            )
        ]
    )
    action_event = _action_event(
        "mailpit_send_email",
        MCPToolAction(data={"to": "bob@example.com", "body": "Alice has flu."}),
    )

    result = controller.before_tool_execution(
        conversation,
        action_event,
        is_read_only=False,
    )

    assert result.replacement_events is None
    recorded = cast(
        PrivacyJudgment,
        result.observation_event_updates["privacy_check_result"],
    )
    assert recorded.decision is JudgmentDecision.ABSTRACT
    assert recorded.rationale == judgment.rationale
    assert recorded.transmission_context is not None
    assert recorded.transmission_context.data_recipient == "to=bob@example.com"
    assert len(analyzer._judge_calls) == 1


def test_full_audit_rejects_non_pass_write_judgment():
    analyzer = RecordingPrivacyAnalyzer(
        judgment=PrivacyJudgment(
            decision=JudgmentDecision.BLOCK,
            rationale="The recipient is not an appropriate audience.",
        )
    )
    controller = PrivacyAuditController(analyzer=analyzer)
    conversation = _conversation_with_events(
        [
            _observation_event(
                "1",
                [InformationFlow(data_type="password", data_subject="Alice")],
            )
        ]
    )
    action_event = _action_event(
        "mailpit_send_email",
        MCPToolAction(data={"to": "bob@example.com", "body": "password123"}),
    )

    result = controller.before_tool_execution(
        conversation,
        action_event,
        is_read_only=False,
    )

    assert result.replacement_events is not None
    assert len(result.replacement_events) == 1


class ExplodingPrivacyAnalyzer(RecordingPrivacyAnalyzer):
    """Analyzer that raises at both boundaries, to exercise the fail-open path."""

    def audit_decision(self, buffered_reads, accumulated_flows, transmission_context):
        raise RuntimeError("auditor exploded")

    def judge_write_action_with_context(
        self,
        write_content: str,
        accumulated_flows: list[InformationFlow],
        transmission_context: TransmissionContext,
    ) -> PrivacyJudgment:
        raise RuntimeError("judge exploded")


def test_read_boundary_failure_fails_open_but_is_recorded():
    """The audit degrades to a no-op when the auditor raises. Without recording
    it, a crashed audit is indistinguishable from one that found nothing --
    which reads as a clean result."""
    controller = PrivacyAuditController(analyzer=ExplodingPrivacyAnalyzer())
    conversation = _conversation_with_events([])
    action = _read_action_event("1", tool_name="mailpit_read_email")
    obs = _observation_event("1", flows=None, tool_name="mailpit_read_email")

    result = controller.after_read_batch(conversation, [(action, obs)])

    # fails open: no flows, no steering, execution is untouched
    assert result.information_flows_by_action_id == {}
    assert result.instruction_event is None
    # ... but on the record
    assert result.audit_error is not None
    assert "read_boundary" in result.audit_error
    assert "auditor exploded" in result.audit_error


def test_write_boundary_failure_fails_open_but_is_recorded():
    controller = PrivacyAuditController(analyzer=ExplodingPrivacyAnalyzer())
    conversation = _conversation_with_events(
        [
            _observation_event(
                "1",
                [InformationFlow(data_type="password", data_subject="Alice")],
            )
        ]
    )
    action_event = _action_event(
        "mailpit_send_email",
        MCPToolAction(data={"to": "bob@example.com", "body": "password123"}),
    )

    result = controller.before_tool_execution(
        conversation, action_event, is_read_only=False
    )

    # fails open: the write is NOT blocked
    assert result.replacement_events is None
    assert result.audit_error is not None
    assert "write_boundary(mailpit_send_email)" in result.audit_error


def test_empty_inventory_is_not_reported_as_an_audit_error():
    """'Nothing to judge' must stay distinguishable from 'the gate crashed'."""
    controller = PrivacyAuditController(analyzer=RecordingPrivacyAnalyzer())
    conversation = _conversation_with_events([])
    action_event = _action_event(
        "mailpit_send_email",
        MCPToolAction(data={"to": "bob@example.com", "body": "hello"}),
    )

    result = controller.before_tool_execution(
        conversation, action_event, is_read_only=False
    )

    assert result.replacement_events is None
    assert result.audit_error is None


def _hook_rejection(
    tool_name: str,
    judgment: PrivacyJudgment,
    *,
    action_id: str = "1",
) -> UserRejectObservation:
    return UserRejectObservation(
        tool_name=tool_name,
        tool_call_id=f"call-{action_id}",
        action_id=action_id,
        rejection_reason="rejected by audit",
        rejection_source="hook",
        privacy_check_result=judgment,
    )


def test_previously_rejected_items_recovered_from_the_event_stream():
    """The controller is rebuilt per hook call and holds no state, so the prior
    rejection must be read back from the events -- which already carry it."""
    flagged = TransmittedFlow(
        data_type="HIV diagnosis",
        data_subject="Bob",
        data_recipient="team channel",
        disposition=FlowDisposition.OMIT,
    )
    benign = TransmittedFlow(
        data_type="meeting time",
        data_subject="Bob",
        data_recipient="team channel",
        disposition=FlowDisposition.PASS,
    )
    conversation = _conversation_with_events(
        [
            _hook_rejection(
                "rocketchat_send_channel_message",
                PrivacyJudgment(
                    decision=JudgmentDecision.ABSTRACT,
                    transmitted_flows=[flagged, benign],
                ),
            )
        ]
    )

    items = gather_previously_rejected_items(
        conversation, "rocketchat_send_channel_message"
    )
    # only the offending items carry forward; PASS items were never the problem
    assert [f.data_type for f in items] == ["HIV diagnosis"]

    # a different write tool is a different transmission, not a retry
    assert gather_previously_rejected_items(conversation, "mailpit_send_email") == []
    # and a first attempt has nothing to verify
    assert gather_previously_rejected_items(_conversation_with_events([]), "x") == []


def test_rejected_items_reach_the_judge_as_write_context():
    analyzer = ContextRecordingPrivacyAnalyzer()
    controller = PrivacyAuditController(analyzer=analyzer)
    flagged = TransmittedFlow(
        data_type="HIV diagnosis",
        data_subject="Bob",
        data_recipient="team channel",
        disposition=FlowDisposition.OMIT,
    )
    conversation = _conversation_with_events(
        [
            _observation_event(
                "1", [InformationFlow(data_type="HIV diagnosis", data_subject="Bob")]
            ),
            _hook_rejection(
                "mailpit_send_email",
                PrivacyJudgment(
                    decision=JudgmentDecision.ABSTRACT, transmitted_flows=[flagged]
                ),
            ),
        ]
    )
    action_event = _action_event(
        "mailpit_send_email",
        MCPToolAction(data={"to": "team@example.com", "body": "Bob is unwell"}),
    )

    controller.before_tool_execution(conversation, action_event, is_read_only=False)

    (ctx,) = analyzer._context_calls
    assert [f.data_type for f in ctx.previously_rejected] == ["HIV diagnosis"]


def test_recompose_budget_switches_the_directive_instead_of_looping():
    """Past the retry budget the gate stays closed, but stops inviting another
    rewrite -- an unbounded reject/recompose loop burns the step budget."""
    judgment = PrivacyJudgment(
        decision=JudgmentDecision.ABSTRACT,
        rationale="Discloses a diagnosis to the wrong audience.",
        transmitted_flows=[
            TransmittedFlow(
                data_type="HIV diagnosis",
                data_subject="Bob",
                data_recipient="team channel",
                disposition=FlowDisposition.OMIT,
            )
        ],
    )
    analyzer = RecordingPrivacyAnalyzer(judgment=judgment)
    controller = PrivacyAuditController(analyzer=analyzer)
    inventory = _observation_event(
        "1", [InformationFlow(data_type="HIV diagnosis", data_subject="Bob")]
    )
    action_event = _action_event(
        "mailpit_send_email",
        MCPToolAction(data={"to": "team@example.com", "body": "Bob is unwell"}),
    )

    def reject_for(n_prior: int) -> UserRejectObservation:
        events: list[object] = [inventory]
        events += [
            _hook_rejection("mailpit_send_email", judgment, action_id=str(i))
            for i in range(n_prior)
        ]
        result = controller.before_tool_execution(
            _conversation_with_events(events), action_event, is_read_only=False
        )
        assert result.replacement_events is not None
        (rejection,) = result.replacement_events
        assert isinstance(rejection, UserRejectObservation)
        return rejection

    # within budget: the usual "re-compose without the specifics" directive
    early = reject_for(1)
    assert early.privacy_check_result is not None
    assert early.privacy_check_result.recompose_exhausted is False
    assert "Re-compose the message" in early.rejection_reason

    # budget exhausted: still rejected, but no longer inviting a reword
    late = reject_for(3)
    assert late.privacy_check_result is not None
    assert late.privacy_check_result.recompose_exhausted is True
    assert "Do not simply" in late.rejection_reason
    assert "ask the user how to proceed" in late.rejection_reason
