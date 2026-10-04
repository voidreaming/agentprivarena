"""Mailpit MCP server — email tools.

Provides email send (SMTP), search, read, and contact lookup via Mailpit API.

Error handling
--------------
HTTP-side tools (search/read/list_contacts) go through the shared
``base.http_*`` helpers, which translate every httpx failure into a
``HTTPToolError`` carrying the upstream status code. There is no auth on
Mailpit's REST API, so no login flow or 401 retry is needed.

The :func:`send_email` tool uses ``smtplib`` rather than HTTP, so its error
translation is inline and Mailpit-specific: SMTP-level failures (connection
refused, server rejection, transient errors) are converted into ``ToolError``
with the SMTP response code and message attached so the agent sees actionable
context instead of an opaque traceback.

Business-logic failures (e.g. an unknown ``email_id``) raise :class:`ToolError`
directly with an actionable recovery hint that names the tool the agent can
use to discover valid inputs.
"""

from __future__ import annotations

import smtplib
from email.mime.text import MIMEText
from email.utils import parseaddr
from typing import Annotated, Any

from base import (
    HTTPToolError,
    get_env,
    http_get_params,
    logger,
)
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field


MAILPIT_API = get_env("MAILPIT_API_URL", "http://mailpit:8025")
SMTP_HOST = get_env("MAILPIT_SMTP_HOST", "mailpit")
SMTP_PORT = int(get_env("MAILPIT_SMTP_PORT", "1025"))

# Hardcoded persona for the seed environment. Tasks rewrite recipients but
# always send "from" the test user.
_FROM_ADDRESS = "john.doe@gmail.com"
_CONTACT_REGISTRY_MARKER = "PRIVACYLENS_CONTACT"
_CONTACT_REGISTRY_SUBJECT_PREFIX = "PrivacyLens Contact:"

mcp = FastMCP("mailpit")


def _address_parts(field: Any) -> tuple[str, str]:
    """Return ``(email, display_name)`` for Mailpit address fields."""
    if isinstance(field, dict):
        return field.get("Address", ""), field.get("Name", "")
    if field:
        name, email = parseaddr(str(field))
        return email, name
    return "", ""


def _format_address(field: Any) -> str:
    """Mailpit returns addresses as either ``{Address, Name}`` dicts or strings."""
    email, _ = _address_parts(field)
    return email


def _recipient_values(value: str | list[str]) -> list[str]:
    raw_values = value if isinstance(value, list) else [value]
    recipients: list[str] = []
    for raw_value in raw_values:
        recipients.extend(
            address.strip() for address in str(raw_value).split(",") if address.strip()
        )
    return recipients


def _is_contact_registry_summary(msg: dict[str, Any]) -> bool:
    subject = str(msg.get("Subject") or "")
    return subject.startswith(_CONTACT_REGISTRY_SUBJECT_PREFIX)


def _parse_contact_registry_text(text: str) -> dict[str, str] | None:
    if _CONTACT_REGISTRY_MARKER not in text:
        return None

    fields: dict[str, str] = {}
    key_map = {
        "name": "name",
        "email": "email",
        "remark": "remark",
        "info": "info",
        "mailing address": "mailing_address",
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        raw_key, raw_value = line.split(":", 1)
        key = key_map.get(raw_key.strip().lower())
        if key:
            fields[key] = raw_value.strip()

    if not fields.get("email") and not fields.get("name"):
        return None
    return fields


def _contact_matches(contact: dict[str, str], needle: str) -> bool:
    searchable = " ".join(contact.values()).lower()
    return needle in searchable


async def _contact_registry_from_message(msg: dict[str, Any]) -> dict[str, str] | None:
    message_id = msg.get("ID", "")
    if not message_id:
        return None
    try:
        detail = await http_get_params(f"{MAILPIT_API}/api/v1/message/{message_id}")
    except HTTPToolError as exc:
        logger.warning(
            f"list_contacts: skipping contact registry message {message_id}: {exc}"
        )
        return None
    return _parse_contact_registry_text(str(detail.get("Text") or ""))


# ── Tools ──


@mcp.tool()
async def send_email(
    to: Annotated[
        str | list[str],
        Field(
            description=(
                "Recipient email address, or a list of recipient email addresses. "
                "Use cc/bcc for additional recipients when possible."
            )
        ),
    ],
    subject: Annotated[str, Field(description="Email subject line.")],
    body: Annotated[
        str,
        Field(
            description=(
                "Plain-text email body. Parameter name is `body` "
                "(not `text`, not `content`)."
            )
        ),
    ],
    cc: Annotated[
        str | list[str],
        Field(
            description=(
                "Optional CC recipients as a comma-separated string or list of "
                "email addresses (e.g. 'a@x.com,b@x.com'). Empty for no CC."
            )
        ),
    ] = "",
    bcc: Annotated[
        str | list[str],
        Field(
            description=(
                "Optional BCC recipients as a comma-separated string or list of "
                "email addresses. Empty for no BCC."
            )
        ),
    ] = "",
) -> dict:
    """Send an email to the specified recipient.

    The body parameter is named ``body`` (not ``text`` or ``content``).
    There is no ``from`` / ``from_email`` parameter — the sender address
    is fixed server-side; do not pass one.
    """
    to_recipients = _recipient_values(to)
    if not to_recipients:
        raise ToolError("send_email requires at least one recipient in `to`.")
    cc_recipients = _recipient_values(cc) if cc else []
    bcc_recipients = _recipient_values(bcc) if bcc else []
    to_display = ", ".join(to_recipients)

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = _FROM_ADDRESS
    msg["To"] = to_display
    if cc_recipients:
        msg["Cc"] = ", ".join(cc_recipients)

    recipients = [*to_recipients, *cc_recipients, *bcc_recipients]

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
            smtp.sendmail(_FROM_ADDRESS, recipients, msg.as_string())
    except smtplib.SMTPResponseException as exc:
        # SMTPResponseException covers most rejection paths and carries
        # both the SMTP code and the server's textual reason.
        raw = exc.smtp_error
        detail = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
        msg_text = (
            f"SMTP send to {to_display} rejected by {SMTP_HOST}:{SMTP_PORT} "
            f"(code {exc.smtp_code}): {detail}"
        )
        logger.warning(msg_text)
        raise ToolError(msg_text) from exc
    except smtplib.SMTPException as exc:
        # Generic SMTP-protocol errors (no response code attached).
        msg_text = (
            f"SMTP send to {to_display} failed against {SMTP_HOST}:{SMTP_PORT}: "
            f"{type(exc).__name__}: {exc}"
        )
        logger.warning(msg_text)
        raise ToolError(msg_text) from exc
    except OSError as exc:
        # Connection refused / DNS / socket errors before SMTP handshake.
        msg_text = (
            f"Cannot reach SMTP server {SMTP_HOST}:{SMTP_PORT}: "
            f"{type(exc).__name__}: {exc}"
        )
        logger.warning(msg_text)
        raise ToolError(msg_text) from exc

    return {"success": True, "to": to_display, "subject": subject}


@mcp.tool(annotations={"readOnlyHint": True})
async def search_emails(
    query: Annotated[
        str,
        Field(
            description=(
                "Required search terms to match against email subjects and "
                "bodies. Put the terms in `query`; do not put them only in "
                "the action `summary`. Parameter name is `query` "
                "(not `keyword`)."
            )
        ),
    ],
) -> dict:
    """Search emails by keyword in subject or body.

    The search-term parameter is named ``query`` (not ``keyword``).
    Search results are discovery summaries; use the email_id field to
    read_email before relying on an email's full body in a final action.

    `time` is an ISO 8601 timestamp string. `from_email` is the sender's
    email address (matches the `email` field returned by list_contacts and
    is suitable for passing as send_email's `to`).
    """
    data = await http_get_params(
        f"{MAILPIT_API}/api/v1/search",
        params={"query": query, "limit": 20},
    )
    emails = [
        {
            "email_id": msg.get("ID", ""),
            "subject": msg.get("Subject", ""),
            "from_email": _format_address(msg.get("From")),
            "time": msg.get("Created", ""),
            "snippet": msg.get("Snippet", ""),
        }
        for msg in data.get("messages", [])
        if not _is_contact_registry_summary(msg)
    ]
    return {"emails": emails}


@mcp.tool(annotations={"readOnlyHint": True})
async def read_email(
    email_id: Annotated[
        str,
        Field(
            description=(
                "Mailpit message ID, as returned by search_emails (the "
                "`email_id` field). Pass the identifier exactly; do not "
                "describe the email in `summary`."
            )
        ),
    ],
) -> dict:
    """Read a specific email.

    Pass the email_id from search_emails results.

    `time` is an ISO 8601 timestamp string. `from_email` and entries in
    the `to` list are email addresses, suitable for passing as send_email's
    `to`, `cc`, or `bcc` arguments.
    """
    try:
        msg = await http_get_params(
            f"{MAILPIT_API}/api/v1/message/{email_id}",
        )
    except HTTPToolError as exc:
        if exc.status_code == 404:
            raise ToolError(
                f"Email '{email_id}' not found in Mailpit. "
                "Use search_emails to discover valid email IDs."
            ) from exc
        raise
    return {
        "email_id": msg.get("ID", ""),
        "subject": msg.get("Subject", ""),
        "from_email": _format_address(msg.get("From")),
        "to": [_format_address(r) for r in msg.get("To", [])],
        "time": msg.get("Created", ""),
        "body": msg.get("Text", ""),
    }


@mcp.tool(annotations={"readOnlyHint": True})
async def list_contacts(
    name: Annotated[
        str,
        Field(
            description=(
                "Required name or email-address fragment to match against the "
                "From/To headers of all known emails. Partial matches are "
                "supported. Put the fragment in `name`; do not put it only "
                "in the action `summary`. Parameter name is `name` "
                "(not `query`)."
            )
        ),
    ],
) -> dict:
    """Search for contacts by name. Derives contacts from email headers.

    The search-term parameter is named ``name`` (not ``query``).

    The `email` field on each result is what to pass as send_email's `to`,
    `cc`, or `bcc` parameter.
    """
    data = await http_get_params(
        f"{MAILPIT_API}/api/v1/search",
        params={"query": name, "limit": 50},
    )
    contacts: dict[str, dict[str, str]] = {}
    needle = name.lower()
    for msg in data.get("messages", []):
        if _is_contact_registry_summary(msg):
            registry_contact = await _contact_registry_from_message(msg)
            if registry_contact and _contact_matches(registry_contact, needle):
                key = registry_contact.get("email") or registry_contact.get("name", "")
                contacts[key] = registry_contact
            continue

        # "From" field
        email, contact_name = _address_parts(msg.get("From"))
        if email and needle in (email + contact_name).lower():
            contacts[email] = {"name": contact_name, "email": email}
        # "To" field (list of dicts)
        for to in msg.get("To", []):
            email, contact_name = _address_parts(to)
            if email and needle in (email + contact_name).lower():
                contacts[email] = {
                    "name": contact_name,
                    "email": email,
                }
    return {"contacts": list(contacts.values())}


if __name__ == "__main__":
    logger.info(f"Starting Mailpit MCP server (API at {MAILPIT_API})")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
