"""Focused tests for Radicale MCP server edge cases."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp.exceptions import ToolError


MCP_DIR = Path(__file__).resolve().parent.parent / "mcp_servers"
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

radicale_server: Any = importlib.import_module("radicale_server")


class _Component:
    def __init__(self, value: str, name: str = "COMPONENT"):
        self.value = value
        self.name = name

    def __str__(self) -> str:
        return f"<{self.name}{{}}{self.value}>"


class _VEvent:
    def __init__(
        self,
        uid: str = "event-1",
        summary: str = "Trip planning",
        description: str = "Discuss travel details",
    ):
        self.uid = _Component(uid, "UID")
        self.summary = _Component(summary, "SUMMARY")
        self.description = _Component(description, "DESCRIPTION")
        self.dtstart = _Component("2026-04-15 10:00:00+00:00", "DTSTART")
        self.dtend = _Component("2026-04-15 11:00:00+00:00", "DTEND")
        self.location = _Component("Terminal 1", "LOCATION")
        self.attendee = _Component("mailto:John Doe", "ATTENDEE")


class _VObject:
    def __init__(self, vevent: _VEvent | None):
        self.vevent = vevent


class _CalendarEvent:
    def __init__(self, vevent: _VEvent | None):
        self.vobject_instance = _VObject(vevent)


class _Calendar:
    def __init__(self, events: list[_CalendarEvent]):
        self._events = events
        self.search_queries: list[str] = []

    def events(self) -> list[_CalendarEvent]:
        return self._events

    def search(self, query: str) -> list[_CalendarEvent]:
        self.search_queries.append(query)
        return self._events


def test_list_events_skips_malformed_caldav_events(monkeypatch):
    """A single malformed event should not break the whole list call."""
    calendar = _Calendar(
        [
            _CalendarEvent(_VEvent(uid="good", summary="Good event")),
            _CalendarEvent(None),
        ]
    )
    monkeypatch.setattr(radicale_server, "_calendar_or_raise", lambda: calendar)

    result = asyncio.run(radicale_server.list_events())

    assert result == {
        "events": [
            {
                "event_id": "good",
                "name": "Good event",
                "start": "2026-04-15 10:00:00+00:00",
            }
        ]
    }


def test_search_events_accepts_summary_alias(monkeypatch):
    """Recover when the model puts the search term in summary instead of query."""
    calendar = _Calendar([_CalendarEvent(_VEvent(uid="trip", summary="Trip"))])
    monkeypatch.setattr(radicale_server, "_calendar_or_raise", lambda: calendar)

    result = asyncio.run(radicale_server.search_events(summary="trip"))

    assert calendar.search_queries == ["trip"]
    assert result == {"events": [{"event_id": "trip", "name": "Trip"}]}


def test_search_events_requires_non_empty_query(monkeypatch):
    monkeypatch.setattr(radicale_server, "_calendar_or_raise", lambda: _Calendar([]))

    with pytest.raises(ToolError, match="requires a non-empty `query`"):
        asyncio.run(radicale_server.search_events())


def test_parse_event_uses_vobject_component_values():
    """Tool output should expose clean values, not vobject repr strings."""
    result = radicale_server._parse_event(_VEvent(uid="trip", summary="Paris trip"))

    assert result == {
        "event_id": "trip",
        "name": "Paris trip",
        "description": "Discuss travel details",
        "start": "2026-04-15 10:00:00+00:00",
        "end": "2026-04-15 11:00:00+00:00",
        "location": "Terminal 1",
        "attendees": ["John Doe"],
    }
