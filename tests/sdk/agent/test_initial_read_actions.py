"""Tests for programmatic initial read actions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

import pytest

from openhands.sdk.agent import Agent
from openhands.sdk.agent.base import InitialReadAction
from openhands.sdk.conversation import Conversation
from openhands.sdk.event import ActionEvent, MessageEvent, ObservationEvent
from openhands.sdk.llm import Message, TextContent
from openhands.sdk.testing import TestLLM
from openhands.sdk.tool import Action, Observation, Tool, ToolExecutor, register_tool
from openhands.sdk.tool.tool import ToolAnnotations, ToolDefinition


if TYPE_CHECKING:
    from openhands.sdk.conversation.base import BaseConversation
    from openhands.sdk.conversation.state import ConversationState


READ_EXECUTIONS: list[str] = []


class StartupReadAction(Action):
    query: str


class StartupReadObservation(Observation):
    pass


class StartupReadExecutor(ToolExecutor[StartupReadAction, StartupReadObservation]):
    def __call__(
        self,
        action: StartupReadAction,
        conversation: BaseConversation | None = None,
    ) -> StartupReadObservation:
        READ_EXECUTIONS.append(action.query)
        return StartupReadObservation.from_text(f"read:{action.query}")


class StartupReadTool(ToolDefinition[StartupReadAction, StartupReadObservation]):
    name = "startup_read"

    @classmethod
    def create(cls, conv_state: ConversationState | None = None) -> Sequence[Self]:
        return [
            cls(
                description="Read startup data.",
                action_type=StartupReadAction,
                observation_type=StartupReadObservation,
                annotations=ToolAnnotations(readOnlyHint=True),
                executor=StartupReadExecutor(),
            )
        ]


class StartupWriteAction(Action):
    message: str


class StartupWriteObservation(Observation):
    pass


class StartupWriteExecutor(ToolExecutor[StartupWriteAction, StartupWriteObservation]):
    def __call__(
        self,
        action: StartupWriteAction,
        conversation: BaseConversation | None = None,
    ) -> StartupWriteObservation:
        return StartupWriteObservation.from_text(f"write:{action.message}")


class StartupWriteTool(ToolDefinition[StartupWriteAction, StartupWriteObservation]):
    name = "startup_write"

    @classmethod
    def create(cls, conv_state: ConversationState | None = None) -> Sequence[Self]:
        return [
            cls(
                description="Write startup data.",
                action_type=StartupWriteAction,
                observation_type=StartupWriteObservation,
                executor=StartupWriteExecutor(),
            )
        ]


register_tool("StartupReadTool", StartupReadTool)
register_tool("StartupWriteTool", StartupWriteTool)


def _assistant_text(text: str) -> Message:
    return Message(role="assistant", content=[TextContent(text=text)])


def test_initial_read_actions_execute_once_before_first_llm_message():
    READ_EXECUTIONS.clear()
    llm = TestLLM.from_messages(
        [
            _assistant_text("Done"),
            _assistant_text("Done again"),
        ]
    )
    agent = Agent(
        llm=llm,
        tools=[Tool(name="StartupReadTool")],
        initial_read_actions=[
            InitialReadAction(
                tool_name="startup_read",
                arguments={"query": "seed"},
                summary="Forced startup read.",
            )
        ],
    )
    events = []
    conversation = Conversation(agent=agent, callbacks=[events.append])

    conversation.send_message("Go")
    conversation.run()

    action_events = [e for e in events if isinstance(e, ActionEvent)]
    obs_events = [e for e in events if isinstance(e, ObservationEvent)]
    assert READ_EXECUTIONS == ["seed"]
    assert len(action_events) == 1
    assert action_events[0].tool_name == "startup_read"
    assert action_events[0].forced_read is True
    assert action_events[0].summary == "Forced startup read."
    assert len(obs_events) == 1

    forced_action_index = events.index(action_events[0])
    first_agent_message_index = next(
        i
        for i, event in enumerate(events)
        if isinstance(event, MessageEvent) and event.source == "agent"
    )
    assert forced_action_index < first_agent_message_index

    conversation.send_message("Next task")
    conversation.run()

    assert READ_EXECUTIONS == ["seed"]
    assert (
        sum(
            1
            for event in events
            if isinstance(event, MessageEvent) and event.source == "agent"
        )
        == 2
    )


def test_initial_read_actions_reject_non_read_only_tools():
    llm = TestLLM.from_messages([_assistant_text("Done")])
    agent = Agent(
        llm=llm,
        tools=[Tool(name="StartupWriteTool")],
        initial_read_actions=[
            InitialReadAction(
                tool_name="startup_write",
                arguments={"message": "unsafe"},
            )
        ],
    )
    conversation = Conversation(agent=agent, callbacks=[])
    conversation.send_message("Go")

    with pytest.raises(ValueError, match="read-only"):
        agent.step(conversation, on_event=lambda _event: None)
