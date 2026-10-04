"""Focused tests for Mailpit MCP contact registry behavior."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any


MCP_DIR = Path(__file__).resolve().parent.parent / "mcp_servers"
if str(MCP_DIR) not in sys.path:
    sys.path.insert(0, str(MCP_DIR))

mailpit_server = importlib.import_module("mailpit_server")


def test_send_email_accepts_list_recipients(monkeypatch) -> None:
    sent: list[tuple[str, list[str], str]] = []

    class FakeSMTP:
        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port

        def __enter__(self) -> FakeSMTP:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def sendmail(self, from_addr: str, recipients: list[str], message: str) -> None:
            sent.append((from_addr, recipients, message))

    monkeypatch.setattr(mailpit_server.smtplib, "SMTP", FakeSMTP)

    result = asyncio.run(
        mailpit_server.send_email(
            ["sarah@example.com"],
            "Project Update",
            "Body",
            cc=["team@example.com"],
            bcc="rob@example.com, laura@example.com",
        )
    )

    assert result == {
        "success": True,
        "to": "sarah@example.com",
        "subject": "Project Update",
    }
    assert sent[0][0] == "john.doe@gmail.com"
    assert sent[0][1] == [
        "sarah@example.com",
        "team@example.com",
        "rob@example.com",
        "laura@example.com",
    ]
    assert "To: sarah@example.com" in sent[0][2]
    assert "Cc: team@example.com" in sent[0][2]


def test_parse_contact_registry_text_keeps_profile_fields() -> None:
    contact = mailpit_server._parse_contact_registry_text(
        "\n".join(
            [
                "AGENTPRIVARENA_CONTACT",
                "Name: Sarah Connor",
                "Email: sarah@example.com",
                "Remark: Investor",
                "Info: Met at a conference.",
                "Mailing Address: 123 Main St",
            ]
        )
    )

    assert contact == {
        "name": "Sarah Connor",
        "email": "sarah@example.com",
        "remark": "Investor",
        "info": "Met at a conference.",
        "mailing_address": "123 Main St",
    }


def test_search_emails_hides_contact_registry(monkeypatch) -> None:
    async def fake_http_get_params(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del url, params, headers
        return {
            "messages": [
                {"ID": "c1", "Subject": "AgentPrivArena Contact: Sarah Connor"},
                {
                    "ID": "e1",
                    "Subject": "Project Update",
                    "From": {"Address": "sarah@example.com", "Name": "Sarah"},
                    "Created": "2022-02-21T10:00:00Z",
                    "Snippet": "Normal email",
                },
            ]
        }

    monkeypatch.setattr(mailpit_server, "http_get_params", fake_http_get_params)

    result = asyncio.run(mailpit_server.search_emails("Sarah"))

    assert result == {
        "emails": [
            {
                "email_id": "e1",
                "subject": "Project Update",
                "from_email": "sarah@example.com",
                "time": "2022-02-21T10:00:00Z",
                "snippet": "Normal email",
            }
        ]
    }


def test_list_contacts_reads_registry_entries(monkeypatch) -> None:
    async def fake_http_get_params(
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        del params, headers
        if url.endswith("/api/v1/search"):
            return {
                "messages": [
                    {"ID": "c1", "Subject": "AgentPrivArena Contact: Sarah Connor"}
                ]
            }
        return {
            "ID": "c1",
            "Text": (
                "AGENTPRIVARENA_CONTACT\n"
                "Name: Sarah Connor\n"
                "Email: sarah@example.com\n"
                "Remark: Investor\n"
                "Info: Met at a conference."
            ),
        }

    monkeypatch.setattr(mailpit_server, "http_get_params", fake_http_get_params)

    result = asyncio.run(mailpit_server.list_contacts("Investor"))

    assert result == {
        "contacts": [
            {
                "name": "Sarah Connor",
                "email": "sarah@example.com",
                "remark": "Investor",
                "info": "Met at a conference.",
            }
        ]
    }
