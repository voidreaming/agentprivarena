"""Tests for runner conversation execution control."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from privacylens_live.runner.conversation_executor import (
    AUTO_CONFIRMATION,
    ConversationExecutor,
)
from privacylens_live.runner.event_collector import EventCollector


class FakeState:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []


class FakeConversation:
    def __init__(
        self,
        *,
        collector: EventCollector,
        run_events: Sequence[list[dict[str, Any]]],
    ) -> None:
        self.collector = collector
        self.run_events = list(run_events)
        self.messages: list[str] = []
        self.run_count = 0
        self.state = FakeState()

    def send_message(self, message: str) -> None:
        self.messages.append(message)

    def run(self) -> None:
        if self.run_count < len(self.run_events):
            self.state.events.extend(self.run_events[self.run_count])
        self.run_count += 1


def test_executor_stops_after_successful_initial_send() -> None:
    collector = EventCollector()
    conversation = FakeConversation(
        collector=collector,
        run_events=[_send_success_events()],
    )

    result = ConversationExecutor().run(
        conversation=conversation,
        initial_message="Send the update.",
        collector=collector,
        task_name="case1",
    )

    assert result.clarifications == []
    assert conversation.messages == ["Send the update."]
    assert conversation.run_count == 1


def test_executor_auto_confirms_once_then_stops_after_send() -> None:
    collector = EventCollector()
    conversation = FakeConversation(
        collector=collector,
        run_events=[
            [_assistant_message("Who should receive it?")],
            _send_success_events(),
        ],
    )

    result = ConversationExecutor().run(
        conversation=conversation,
        initial_message="Send the update.",
        collector=collector,
        task_name="case2",
    )

    assert result.clarifications == [
        {
            "round": 1,
            "request": "Who should receive it?",
            "response": AUTO_CONFIRMATION,
        }
    ]
    assert conversation.messages == ["Send the update.", AUTO_CONFIRMATION]
    assert conversation.run_count == 2


def test_executor_caps_clarification_rounds() -> None:
    collector = EventCollector()
    conversation = FakeConversation(
        collector=collector,
        run_events=[
            [_assistant_message("Which channel?")],
            [_assistant_message("What tone?")],
            [_assistant_message("Anything else?")],
        ],
    )

    result = ConversationExecutor(max_clarification_rounds=2).run(
        conversation=conversation,
        initial_message="Post the update.",
        collector=collector,
        task_name="case3",
    )

    assert [item["request"] for item in result.clarifications] == [
        "Which channel?",
        "What tone?",
    ]
    assert conversation.messages == [
        "Post the update.",
        AUTO_CONFIRMATION,
        AUTO_CONFIRMATION,
    ]
    assert conversation.run_count == 3


def test_executor_stops_without_final_message() -> None:
    collector = EventCollector()
    conversation = FakeConversation(collector=collector, run_events=[[]])

    result = ConversationExecutor().run(
        conversation=conversation,
        initial_message="Post the update.",
        collector=collector,
        task_name="case4",
    )

    assert result.clarifications == []
    assert conversation.messages == ["Post the update."]
    assert conversation.run_count == 1


def test_executor_reconciles_events_from_conversation_state() -> None:
    collector = EventCollector()
    conversation = FakeConversation(
        collector=collector,
        run_events=[_send_success_events()],
    )

    ConversationExecutor().run(
        conversation=conversation,
        initial_message="Send the update.",
        collector=collector,
        task_name="case5",
    )

    final_action = collector.extract_final_action()

    assert final_action is not None
    assert final_action["tool"] == "mailpit_send_email"


def _assistant_message(text: str) -> dict[str, Any]:
    return {
        "kind": "MessageEvent",
        "llm_message": {
            "role": "assistant",
            "content": text,
        },
    }


def _send_success_events() -> list[dict[str, Any]]:
    return [
        {
            "kind": "ActionEvent",
            "tool_name": "mailpit_send_email",
            "tool_call_id": "send-1",
            "action": {
                "data": {"to": "a@example.com", "body": "Done"},
                "kind": "MCPToolAction",
            },
        },
        {
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
