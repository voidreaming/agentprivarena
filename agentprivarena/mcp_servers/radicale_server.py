# pyright: reportMissingImports=false, reportCallIssue=false, reportGeneralTypeIssues=false
#
# The ``caldav`` library has no type stubs, so pyright resolves every
# caldav symbol as ``object``. That trips ``reportCallIssue``
# (``DAVClient(...)`` is "not callable") and ``reportGeneralTypeIssues``
# (``caldav.Calendar`` annotations are "not a class"). The library is
# correct at runtime; suppressing these three rules at the file level
# is the project convention. ``reportMissingImports`` covers fresh
# checkouts where ``uv sync --dev`` hasn't run yet.
"""Radicale MCP server — calendar tools.

Provides event search and read via Radicale CalDAV.

Error handling
--------------
Radicale is the only server in this set that does not speak plain HTTP+JSON;
it speaks CalDAV via the ``caldav`` Python library, so the shared
``base.http_*`` helpers do not apply. Instead, every tool wraps its caldav
calls in :func:`_translate_caldav_error`, which converts:

- ``NotFoundError`` → ``HTTPToolError(status_code=404)`` (so business-logic
  callers can branch on it the same way other servers do)
- ``AuthorizationError`` → ``HTTPToolError(status_code=401)`` with a hint to
  check ``RADICALE_USER`` / ``RADICALE_PASSWORD``
- Any other ``DAVError`` → generic ``ToolError`` carrying the exception class
  and message
- Transport-level errors from underlying ``requests`` / ``urllib3`` → generic
  ``ToolError`` so the agent still sees an actionable failure rather than
  a raw stack trace

The :func:`search_events` tool keeps a deliberate fast-path → client-side
fallback: Radicale's CalDAV REPORT search is unreliable, so a search-specific
failure logs a warning and tries listing all events client-side. Auth /
connectivity failures (from ``_get_calendar``) are *not* tolerated; only the
``calendar.search()`` call itself is wrapped by the fallback.
"""

from __future__ import annotations

from typing import Annotated, Any, NoReturn

import caldav
from base import HTTPToolError, get_env, logger
from caldav.lib.error import (
    AuthorizationError,
    DAVError,
    NotFoundError,
)
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field


RADICALE_URL = get_env("RADICALE_URL", "http://radicale:5232")
RADICALE_USER = get_env("RADICALE_USER", "admin")
RADICALE_PASSWORD = get_env("RADICALE_PASSWORD", "admin")

mcp = FastMCP("radicale")


# ── Error translation ──


def _translate_caldav_error(operation: str, exc: BaseException) -> NoReturn:
    """Translate a caldav (or transport) exception into a ToolError. Always raises."""
    if isinstance(exc, NotFoundError):
        msg = f"{operation}: not found in Radicale: {exc}"
        logger.warning(msg)
        raise HTTPToolError(msg, status_code=404) from exc
    if isinstance(exc, AuthorizationError):
        msg = (
            f"{operation}: unauthorized — check RADICALE_USER / "
            f"RADICALE_PASSWORD ({exc})"
        )
        logger.warning(msg)
        raise HTTPToolError(msg, status_code=401) from exc
    if isinstance(exc, DAVError):
        msg = f"{operation} failed: {type(exc).__name__}: {exc}"
        logger.warning(msg)
        raise ToolError(msg) from exc
    # Transport / socket / DNS errors from the underlying requests stack.
    msg = f"{operation} failed: {type(exc).__name__}: {exc}"
    logger.warning(msg)
    raise ToolError(msg) from exc


# ── Calendar lookup ──


def _get_calendar() -> caldav.Calendar:
    """Connect to Radicale and return the default calendar.

    Calls into here may raise ``DAVError`` or transport errors; callers must
    wrap with :func:`_translate_caldav_error`.
    """
    client = caldav.DAVClient(
        url=RADICALE_URL,
        username=RADICALE_USER,
        password=RADICALE_PASSWORD,
    )
    principal = client.principal()
    calendars = principal.calendars()
    if calendars:
        return calendars[0]
    return principal.make_calendar(name="default")


def _calendar_or_raise() -> caldav.Calendar:
    """Get the default calendar; translates caldav errors to ToolError."""
    try:
        return _get_calendar()
    except Exception as exc:
        _translate_caldav_error("calendar lookup", exc)


def _component_value(component: Any, default: str = "") -> str:
    """Return the user-facing value from a vobject component."""
    if component is None:
        return default
    value = getattr(component, "value", component)
    return str(value)


def _parse_event(vevent: Any) -> dict[str, Any]:
    """Extract event details from a vobject vevent."""
    attendees: list[str] = []
    if hasattr(vevent, "attendee"):
        att = vevent.attendee
        if not isinstance(att, list):
            att = [att]
        attendees = [_component_value(a).replace("mailto:", "") for a in att]

    return {
        "event_id": _component_value(vevent.uid),
        "name": _component_value(getattr(vevent, "summary", None)),
        "description": _component_value(getattr(vevent, "description", None)),
        "start": _component_value(getattr(vevent, "dtstart", None)),
        "end": _component_value(getattr(vevent, "dtend", None)),
        "location": _component_value(getattr(vevent, "location", None)),
        "attendees": attendees,
    }


def _event_vevent(event: Any) -> Any | None:
    """Return an event's VEVENT payload, or None for malformed CalDAV objects."""
    try:
        vobject_instance = event.vobject_instance
    except Exception as exc:
        logger.warning(
            "Skipping Radicale event with unreadable vobject_instance: %s: %s",
            type(exc).__name__,
            exc,
        )
        return None

    vevent = getattr(vobject_instance, "vevent", None)
    if vevent is None:
        logger.warning("Skipping Radicale event without a VEVENT payload.")
        return None
    return vevent


def _summary_only(event: Any) -> dict[str, Any] | None:
    vevent = _event_vevent(event)
    if vevent is None:
        return None
    return {
        "event_id": _component_value(vevent.uid),
        "name": _component_value(getattr(vevent, "summary", None)),
    }


# ── Tools ──


@mcp.tool(annotations={"readOnlyHint": True})
async def search_events(
    query: Annotated[
        str | None,
        Field(
            description=(
                "Required search terms to match against event titles and "
                "descriptions. Put the terms in `query`; do not put them "
                "only in the action `summary`. Parameter name is `query` "
                "(not `keyword` or `summary`)."
            )
        ),
    ] = None,
    summary: Annotated[
        str | None,
        Field(
            description=(
                "Compatibility alias for `query` when a client mistakenly "
                "uses the action-summary field as the search term. Prefer "
                "`query` for new calls."
            )
        ),
    ] = None,
) -> dict:
    """Search calendar events by keyword.

    The search-term parameter is named ``query`` (not ``keyword``).
    ``summary`` is accepted only as a compatibility alias for clients
    that confuse the SDK action-summary field with the search term.
    Search results are discovery summaries; use event_id to call read_event
    before relying on full event details in a final action. For summary
    or upcoming-plans tasks over a compact set of plausible events, read
    each plausible event before deciding whether to include or omit it.
    """
    search_query = (query or summary or "").strip()
    if not search_query:
        raise ToolError(
            "search_events requires a non-empty `query`. "
            "Example: search_events(query='trip')."
        )

    calendar = _calendar_or_raise()
    # Fast path: ask Radicale to search via CalDAV REPORT.
    try:
        events = calendar.search(search_query)
    except Exception as exc:
        # Radicale's REPORT search is unreliable; fall back to client-side
        # filtering. This branch is the *only* tolerated caldav failure in
        # this module — it always involves a follow-up call that, if it
        # fails, will surface as a hard error.
        logger.warning(
            f"search_events: fast-path search failed "
            f"({type(exc).__name__}: {exc}); "
            f"falling back to client-side filter over all events"
        )
        try:
            all_events = calendar.events()
        except Exception as exc2:
            _translate_caldav_error("search_events fallback list", exc2)
        q = search_query.lower()
        results: list[dict[str, Any]] = []
        for event in all_events:
            vevent = _event_vevent(event)
            if vevent is None:
                continue
            summary = _component_value(getattr(vevent, "summary", None))
            desc = _component_value(getattr(vevent, "description", None))
            if q in summary.lower() or q in desc.lower():
                results.append(
                    {"event_id": _component_value(vevent.uid), "name": summary}
                )
        return {"events": results}

    # Fast path succeeded.
    return {"events": [item for e in events if (item := _summary_only(e))]}


def _event_id_field_description() -> str:
    return (
        "iCalendar UID, as returned by search_events or list_events "
        "(the `event_id` field)."
    )


async def _read_event_payload(event_id: str) -> dict:
    calendar = _calendar_or_raise()
    try:
        event = calendar.event_by_uid(event_id)
    except NotFoundError as exc:
        raise ToolError(
            f"Event '{event_id}' not found in Radicale. "
            "Use list_events or search_events to discover valid event IDs."
        ) from exc
    except Exception as exc:
        _translate_caldav_error(f"read_event({event_id})", exc)
    vevent = _event_vevent(event)
    if vevent is None:
        raise ToolError(
            f"Event '{event_id}' in Radicale has no readable VEVENT payload. "
            "Use list_events or search_events to discover another valid event ID."
        )
    return _parse_event(vevent)


@mcp.tool(annotations={"readOnlyHint": True})
async def read_event(
    event_id: Annotated[
        str,
        Field(description=_event_id_field_description()),
    ],
) -> dict:
    """Read detailed information for a calendar event.

    Pass the event_id from search_events or list_events.
    """
    return await _read_event_payload(event_id)


@mcp.tool(annotations={"readOnlyHint": True})
async def list_events() -> dict:
    """List all calendar events.

    List results are discovery summaries. Use event_id to call read_event
    before relying on full event details in a final action. For summary
    or upcoming-plans tasks over a compact set of plausible events, read
    each plausible event before deciding whether to include or omit it.

    `start` is a Python datetime string representation
    (e.g. '2026-04-15 10:00:00+00:00'), not a strict ISO 8601 string.
    """
    calendar = _calendar_or_raise()
    try:
        all_events = calendar.events()
    except Exception as exc:
        _translate_caldav_error("list_events", exc)
    events = []
    for event in all_events:
        vevent = _event_vevent(event)
        if vevent is None:
            continue
        events.append(
            {
                "event_id": _component_value(vevent.uid),
                "name": _component_value(getattr(vevent, "summary", None)),
                "start": _component_value(getattr(vevent, "dtstart", None)),
            }
        )
    return {"events": events}


if __name__ == "__main__":
    logger.info(f"Starting Radicale MCP server (CalDAV at {RADICALE_URL})")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
