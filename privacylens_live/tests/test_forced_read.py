"""Tests for forced-read planning."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from privacylens_live.base.seeder import CleanupHandle, SeedResult
from privacylens_live.config import Config
from privacylens_live.runner.forced_read import (
    ForcedReadResolver,
    build_initial_read_actions,
    validate_read_policy,
)


class FakeResolver(ForcedReadResolver):
    def __init__(self) -> None:
        pass

    async def mailpit_email_id(self, record: dict[str, Any]) -> str:
        return f"email-{record['subject']}"

    async def gotosocial_user_id(self) -> str:
        return "acct-123"


def _write_seed(seed_dir: Path, service: str, records: list[dict[str, Any]]) -> None:
    (seed_dir / f"{service}.json").write_text(json.dumps(records))


def test_forced_read_planner_builds_service_specific_read_actions(tmp_path: Path):
    task_dir = tmp_path / "main"
    seed_dir = task_dir / "seed_data"
    seed_dir.mkdir(parents=True)
    _write_seed(
        seed_dir,
        "bookstack",
        [
            {"id": "a", "title": "Alpha", "content": "A"},
            {"id": "b", "title": "Beta", "content": "B"},
        ],
    )
    _write_seed(
        seed_dir,
        "radicale",
        [{"event_id": "evt-1", "event_name": "School", "content": "Drop off"}],
    )
    _write_seed(
        seed_dir,
        "mailpit",
        [
            {"subject": "Roadmap", "body": "Ship it"},
            {"type": "contact", "name": "Lisa", "email": "lisa@example.com"},
        ],
    )
    _write_seed(seed_dir, "mattermost", [{"message": "hello", "sender_id": "ann"}])
    _write_seed(
        seed_dir,
        "rocketchat",
        [
            {"message": "ops note", "channel": "#Ops"},
            {"message": "dm note", "channel": "Direct Message"},
        ],
    )
    _write_seed(seed_dir, "gotosocial", [{"content": "status update"}])
    seed_result = SeedResult(
        handles=[CleanupHandle(service="bookstack", record_ids=["101", "102"])]
    )

    actions = asyncio.run(
        build_initial_read_actions(
            task_dir=task_dir,
            seed_result=seed_result,
            config=Config(),
            resolver=FakeResolver(),
        )
    )

    assert [(a["tool_name"], a["arguments"]) for a in actions] == [
        ("bookstack_read_page", {"page_id": 101}),
        ("bookstack_read_page", {"page_id": 102}),
        ("gotosocial_read_user_posts", {"user_id": "acct-123"}),
        ("mailpit_read_email", {"email_id": "email-Roadmap"}),
        ("mattermost_read_messages", {"max_count": 200}),
        ("radicale_read_event", {"event_id": "evt-1"}),
        ("rocketchat_read_channel_history", {"channel": "general", "count": 100}),
        ("rocketchat_read_channel_history", {"channel": "ops", "count": 100}),
    ]


def test_forced_read_planner_skips_registry_and_contact_records(tmp_path: Path):
    task_dir = tmp_path / "main"
    seed_dir = task_dir / "seed_data"
    seed_dir.mkdir(parents=True)
    _write_seed(
        seed_dir,
        "mailpit",
        [{"type": "contact", "name": "Lisa", "email": "lisa@example.com"}],
    )
    _write_seed(
        seed_dir,
        "mattermost",
        [{"type": "user", "name": "Ann", "username": "ann", "email": "a@x"}],
    )
    _write_seed(seed_dir, "rocketchat", [{"type": "channel", "channel": "#ops"}])

    actions = asyncio.run(
        build_initial_read_actions(
            task_dir=task_dir,
            seed_result=SeedResult(),
            config=Config(),
            resolver=FakeResolver(),
        )
    )

    assert actions == []


def test_forced_read_planner_builds_google_drive_image_actions(tmp_path: Path):
    task_dir = tmp_path / "main"
    drive_dir = task_dir / "seed_data" / "google_drive"
    files_dir = drive_dir / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "campaign.jpg").write_bytes(b"fake image")
    (drive_dir / "drive_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "files": [
                    {
                        "file_id": "file_1",
                        "name": "campaign.jpg",
                        "drive_path": "My Drive/campaign.jpg",
                        "mime_type": "image/jpeg",
                        "artifact_path": "files/campaign.jpg",
                    },
                    {
                        "file_id": "file_2",
                        "name": "missing.jpg",
                        "drive_path": "My Drive/missing.jpg",
                        "mime_type": "image/jpeg",
                        "artifact_path": "files/missing.jpg",
                    },
                    {
                        "file_id": "file_3",
                        "name": "notes.txt",
                        "drive_path": "My Drive/notes.txt",
                        "mime_type": "text/plain",
                        "artifact_path": "files/notes.txt",
                    },
                ],
            }
        )
    )

    actions = asyncio.run(
        build_initial_read_actions(
            task_dir=task_dir,
            seed_result=SeedResult(),
            config=Config(),
            resolver=FakeResolver(),
        )
    )

    assert [(a["tool_name"], a["arguments"]) for a in actions] == [
        ("google_drive_get_file_metadata", {"file_id": "file_1"}),
        ("google_drive_get_file_image", {"file_id": "file_1"}),
        (
            "google_drive_describe_image",
            {"file_id": "file_1", "detail": "standard"},
        ),
        ("google_drive_get_file_metadata", {"file_id": "file_2"}),
        ("google_drive_get_file_metadata", {"file_id": "file_3"}),
    ]


def test_validate_read_policy():
    assert validate_read_policy("natural") == "natural"
    assert validate_read_policy("forced_oracle") == "forced_oracle"
