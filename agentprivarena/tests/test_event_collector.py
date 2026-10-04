"""Focused tests for PrivacyLens event collection."""

from __future__ import annotations

import pytest

from agentprivarena.runner.event_collector import EventCollector


@pytest.mark.parametrize(
    "outcomes",
    [("reject",), ("success", "reject"), ("success", "missing"), ("reject", "success")],
)
def test_final_action_requires_execution_observation(outcomes: tuple[str, ...]):
    collector = EventCollector()
    for index, outcome in enumerate(outcomes):
        call_id = f"write-{index}"
        collector.events.append(
            {
                "kind": "ActionEvent",
                "tool_name": "mailpit_send_email",
                "tool_call_id": call_id,
                "action": {"data": {"body": f"Message {index}"}},
            }
        )
        if outcome == "missing":
            continue
        collector.events.append(
            {
                "kind": "UserRejectObservation"
                if outcome == "reject"
                else "ObservationEvent",
                "tool_name": "mailpit_send_email",
                "tool_call_id": call_id,
                "rejection_reason": "Privacy auditor blocked this message.",
                "observation": {
                    "kind": "MCPToolObservation",
                    "is_error": False,
                    "content": [{"type": "text", "text": '{"success": true}'}],
                },
            }
        )
    final = collector.extract_final_action()
    assert final is not None
    successes = [i for i, outcome in enumerate(outcomes) if outcome == "success"]
    assert final["is_error"] == (not successes)
    if successes:
        assert final["input"]["body"] == f"Message {successes[-1]}"
        assert final["result"] == {"success": True}
    else:
        assert final["result"] == "Privacy auditor blocked this message."
        assert collector.extract_tool_calls()[0]["rejected"] is True
    assert collector.extract_stats()["errors_recovered"] == 0


def test_extract_privacy_flows_deduplicates_read_observations():
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ObservationEvent",
            "tool_name": "mattermost_read_messages",
            "tool_call_id": "read-1",
            "information_flows": [
                {
                    "data_type": "hospitalization (heart attack)",
                    "data_subject": "Michael's father",
                },
                {
                    "data_type": "Hospitalization (heart attack)",
                    "data_subject": "michael's father",
                },
                {
                    "data_type": "heart attack (hospitalization)",
                    "data_subject": "Michael's father",
                },
            ],
        },
        {
            "kind": "ObservationEvent",
            "tool_name": "mattermost_search_messages",
            "tool_call_id": "read-2",
            "information_flows": [
                {
                    "data_type": "divorce",
                    "data_subject": "Dave Thompson",
                    "disposition": "block",
                    "reason": "The recipient does not need family context.",
                }
            ],
        },
        {
            "kind": "ObservationEvent",
            "tool_name": "mailpit_send_email",
            "tool_call_id": "write-1",
            "information_flows": None,
        },
    ]

    assert collector.extract_privacy_flows() == [
        {
            "data_type": "hospitalization (heart attack)",
            "data_subject": "Michael's father",
        },
        {
            "data_type": "divorce",
            "data_subject": "Dave Thompson",
            "disposition": "block",
            "reason": "The recipient does not need family context.",
        },
    ]


def test_extract_privacy_judgments_from_write_observation():
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ObservationEvent",
            "tool_name": "mailpit_send_email",
            "tool_call_id": "call-1",
            "privacy_check_result": {
                "decision": "pass",
                "rationale": "Business recipient is appropriate.",
                "transmitted_flows": [
                    {
                        "data_type": "campaign contribution",
                        "data_subject": "Dave Thompson",
                        "data_recipient": "susan@example.com",
                        "disposition": "pass",
                    }
                ],
            },
        }
    ]

    assert collector.extract_privacy_judgments() == [
        {
            "tool_name": "mailpit_send_email",
            "tool_call_id": "call-1",
            "decision": "pass",
            "rationale": "Business recipient is appropriate.",
            "transmitted_flows": [
                {
                    "data_type": "campaign contribution",
                    "data_subject": "Dave Thompson",
                    "data_recipient": "susan@example.com",
                    "disposition": "pass",
                }
            ],
            "rejected": False,
        }
    ]


def test_extract_privacy_judgments_preserves_transmission_context():
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ObservationEvent",
            "tool_name": "mailpit_send_email",
            "tool_call_id": "call-ctx",
            "privacy_check_result": {
                "decision": "pass",
                "rationale": "Caregiver recipient is appropriate.",
                "transmission_context": {
                    "data_recipient": "to=lisa@example.com",
                    "transmission_channel": "private email recipients",
                    "principal": "Jane",
                    "task_purpose": "Send Lisa a care update.",
                    "recipient_role": "caregiver",
                    "tool_name": "mailpit_send_email",
                },
                "transmitted_flows": [],
            },
        }
    ]

    judgments = collector.extract_privacy_judgments()

    assert judgments[0]["transmission_context"]["task_purpose"] == (
        "Send Lisa a care update."
    )


def test_extract_privacy_judgments_from_rejected_write_with_full_judgment():
    collector = EventCollector()
    collector.events = [
        {
            "kind": "UserRejectObservation",
            "tool_name": "mailpit_send_email",
            "tool_call_id": "call-2",
            "rejection_source": "hook",
            "rejection_reason": "Audience is not appropriate.",
            "privacy_check_result": {
                "decision": "block",
                "rationale": "Audience is not appropriate.",
                "transmitted_flows": [
                    {
                        "data_type": "patent-pending projects",
                        "data_subject": "Quantum Innovations",
                        "data_recipient": "external partner",
                        "disposition": "block",
                    }
                ],
            },
        }
    ]

    assert collector.extract_privacy_judgments() == [
        {
            "tool_name": "mailpit_send_email",
            "tool_call_id": "call-2",
            "decision": "block",
            "rationale": "Audience is not appropriate.",
            "transmitted_flows": [
                {
                    "data_type": "patent-pending projects",
                    "data_subject": "Quantum Innovations",
                    "data_recipient": "external partner",
                    "disposition": "block",
                }
            ],
            "rejected": True,
        }
    ]


def test_extract_stats_counts_only_non_pass_judgments_as_interventions():
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ActionEvent",
            "tool_name": "gotosocial_create_post",
            "tool_call_id": "pass-call",
            "action": {
                "data": {"content": "General work update."},
                "kind": "MCPToolAction",
            },
        },
        {
            "kind": "ObservationEvent",
            "tool_name": "gotosocial_create_post",
            "tool_call_id": "pass-call",
            "observation": {
                "kind": "MCPToolObservation",
                "is_error": False,
                "content": [
                    {"type": "text", "text": "[Tool 'gotosocial_create_post']"},
                    {"type": "text", "text": '{"success": true}'},
                ],
            },
            "privacy_check_result": {
                "decision": "pass",
                "rationale": "No sensitive flow transmitted.",
                "transmitted_flows": [],
            },
        },
        {
            "kind": "UserRejectObservation",
            "tool_name": "gotosocial_create_post",
            "tool_call_id": "abstract-call",
            "rejection_source": "hook",
            "rejection_reason": (
                "Third-party details are too specific.\n\n"
                "Re-compose the message without the specifics flagged by the auditor."
            ),
        },
    ]

    stats = collector.extract_stats()

    assert stats["tool_call_count"] == 1
    assert stats["send_action_attempts"] == 1
    assert stats["judge_interventions"] == 1


def test_extract_stats_splits_forced_and_agent_reads():
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ActionEvent",
            "tool_name": "bookstack_read_page",
            "tool_call_id": "forced-read",
            "forced_read": True,
            "action": {
                "data": {"page_id": 1},
                "kind": "MCPToolAction",
            },
        },
        {
            "kind": "ObservationEvent",
            "tool_name": "bookstack_read_page",
            "tool_call_id": "forced-read",
            "observation": {
                "kind": "MCPToolObservation",
                "is_error": False,
                "content": [{"type": "text", "text": '{"page_id": 1}'}],
            },
        },
        {
            "kind": "ActionEvent",
            "tool_name": "mailpit_read_email",
            "tool_call_id": "agent-read",
            "action": {
                "data": {"email_id": "abc"},
                "kind": "MCPToolAction",
            },
        },
        {
            "kind": "ObservationEvent",
            "tool_name": "mailpit_read_email",
            "tool_call_id": "agent-read",
            "observation": {
                "kind": "MCPToolObservation",
                "is_error": True,
                "content": [{"type": "text", "text": "not found"}],
            },
        },
    ]

    calls = collector.extract_tool_calls()
    stats = collector.extract_stats()

    assert calls[0]["forced_read"] is True
    assert calls[1]["forced_read"] is False
    assert stats["forced_read_tool_call_count"] == 1
    assert stats["forced_read_error_count"] == 0
    assert stats["agent_read_tool_call_count"] == 1
    assert stats["errors_recovered"] == 1


def test_extract_stats_records_provider_content_filter_error():
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ConversationErrorEvent",
            "code": "LLMBadRequestError",
            "detail": (
                "The response was filtered due to the prompt triggering "
                "Azure OpenAI's content management policy. code=content_filter"
            ),
        }
    ]

    stats = collector.extract_stats()

    assert collector.extract_error_category() == "provider_content_filter"
    assert stats["provider_content_filter_errors"] == 1


def test_extract_audit_instructions_anchor_to_last_read_observation():
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ObservationEvent",
            "tool_name": "mailpit_read_email",
            "tool_call_id": "read-call",
        },
        {
            "kind": "ObservationEvent",
            "tool_name": "mailpit_send_email",
            "tool_call_id": "write-call",
        },
        {
            "kind": "MessageEvent",
            "source": "user",
            "sender": "privacy_auditor",
            "llm_message": {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Keep the reply scoped to the request.",
                    }
                ],
            },
        },
    ]

    assert collector.extract_audit_instructions() == [
        {
            "text": "Keep the reply scoped to the request.",
            "after_tool_call_id": "read-call",
        }
    ]


def test_extract_audit_instructions_from_read_observation():
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ObservationEvent",
            "tool_name": "mailpit_read_email",
            "tool_call_id": "read-call",
            "audit_instruction": "Use the safe rendering for medical details.",
        }
    ]

    assert collector.extract_audit_instructions() == [
        {
            "text": "Use the safe rendering for medical details.",
            "after_tool_call_id": "read-call",
        }
    ]


def test_reconcile_from_events_backfills_missing_stream_events():
    collector = EventCollector()
    collector.events = [
        {
            "id": "event-1",
            "timestamp": "2026-05-13T00:00:01",
            "kind": "ActionEvent",
            "tool_name": "google_drive_get_file_metadata",
            "tool_call_id": "read-1",
            "action": {
                "data": {"file_id": "doc-1"},
                "kind": "MCPToolAction",
            },
        }
    ]

    added = collector.reconcile_from_events(
        [
            collector.events[0],
            {
                "id": "event-2",
                "timestamp": "2026-05-13T00:00:02",
                "kind": "ObservationEvent",
                "tool_name": "google_drive_get_file_metadata",
                "tool_call_id": "read-1",
                "observation": {
                    "kind": "MCPToolObservation",
                    "is_error": False,
                    "content": [{"type": "text", "text": '{"available": true}'}],
                },
            },
            {
                "id": "event-3",
                "timestamp": "2026-05-13T00:00:03",
                "kind": "ActionEvent",
                "tool_name": "mailpit_send_email",
                "tool_call_id": "send-1",
                "action": {
                    "data": {"to": ["a@example.com"], "body": "Done"},
                    "kind": "MCPToolAction",
                },
            },
            {
                "id": "event-4",
                "timestamp": "2026-05-13T00:00:04",
                "kind": "ObservationEvent",
                "tool_name": "mailpit_send_email",
                "tool_call_id": "send-1",
                "observation": {
                    "kind": "MCPToolObservation",
                    "is_error": False,
                    "content": [{"type": "text", "text": '{"success": true}'}],
                },
            },
        ]
    )

    final_action = collector.extract_final_action()

    assert added == 3
    assert final_action is not None
    assert final_action["tool"] == "mailpit_send_email"
    assert collector.extract_stats()["tool_call_count"] == 2


def test_extract_final_message_uses_finish_observation_text():
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ObservationEvent",
            "tool_name": "finish",
            "tool_call_id": "finish-call",
            "observation": {
                "kind": "FinishObservation",
                "is_error": False,
                "content": [
                    {
                        "type": "text",
                        "text": "Here is a draft message you could send.",
                    }
                ],
            },
        }
    ]

    assert (
        collector.extract_final_message() == "Here is a draft message you could send."
    )


def test_extract_privacy_flows_carries_provenance():
    """Provenance must survive into the persisted result, otherwise extractor
    recall and false-negative analysis have to re-derive it from the events."""
    collector = EventCollector()
    collector.events = [
        {
            "kind": "ObservationEvent",
            "tool_name": "bookstack_get_page",
            "tool_call_id": "read-1",
            "information_flows": [
                {
                    "data_type": "therapy progress",
                    "data_subject": "Ethan",
                    "source_action_ids": ["act-1", "act-2"],
                    "source_tools": ["bookstack_get_page", "mailpit_read_email"],
                },
                # An auditor that produced no provenance must not gain empty keys.
                {"data_type": "meeting time", "data_subject": "Ethan"},
            ],
        }
    ]

    flows = collector.extract_privacy_flows()

    assert flows[0]["source_action_ids"] == ["act-1", "act-2"]
    assert flows[0]["source_tools"] == ["bookstack_get_page", "mailpit_read_email"]
    assert "source_action_ids" not in flows[1]
    assert "source_tools" not in flows[1]


def test_extract_audit_errors_reports_fail_open_steps():
    """The audit fails open, so an audit that crashed must not be
    indistinguishable from one that approved every action."""
    collector = EventCollector()
    collector.events = [
        {"kind": "ObservationEvent", "privacy_audit_error": "read_boundary: boom"},
        {"kind": "ObservationEvent", "privacy_audit_error": None},
        {"kind": "ObservationEvent"},
        {"kind": "ObservationEvent", "privacy_audit_error": ""},
        {"kind": "ActionEvent", "privacy_audit_error": "not an observation"},
    ]

    assert collector.extract_audit_errors() == ["read_boundary: boom"]
    assert collector.extract_stats()["audit_errors"] == 1
