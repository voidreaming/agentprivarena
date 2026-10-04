"""Focused tests for PrivacyLens service seeding orchestration."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from agentprivarena.base.seeder import (
    CleanupHandle,
    GoogleDriveArtifactHandler,
    Seeder,
    ServiceHandler,
    _contact_registry_body,
    _mattermost_registry_user,
    _mattermost_sender_username,
    _rocketchat_channel_name,
    _rocketchat_message_payload,
    _rocketchat_registry_channel,
    _rocketchat_registry_user,
)


class _RecordingHandler(ServiceHandler):
    def __init__(self, service_name: str, calls: list[str]):
        self.service_name = service_name
        self.calls = calls

    async def reset(self) -> None:
        self.calls.append(self.service_name)

    async def seed(self, records: list[dict]) -> CleanupHandle:
        del records
        return CleanupHandle(service=self.service_name)


class _FailingResetHandler(ServiceHandler):
    service_name = "failing"

    async def reset(self) -> None:
        raise RuntimeError("reset failed")

    async def seed(self, records: list[dict]) -> CleanupHandle:
        del records
        return CleanupHandle(service=self.service_name)


def test_reset_services_resets_each_dependency_once_in_order():
    calls: list[str] = []
    seeder = Seeder()
    seeder.handlers = {
        "mattermost": _RecordingHandler("mattermost", calls),
        "bookstack": _RecordingHandler("bookstack", calls),
    }

    asyncio.run(
        seeder.reset_services(
            ["mattermost", "bookstack", "mattermost", "unknown-service"]
        )
    )

    assert calls == ["mattermost", "bookstack"]


def test_reset_services_continues_after_service_reset_failure():
    calls: list[str] = []
    seeder = Seeder()
    seeder.handlers = {
        "failing": _FailingResetHandler(),
        "mattermost": _RecordingHandler("mattermost", calls),
    }

    asyncio.run(seeder.reset_services(["failing", "mattermost"]))

    assert calls == ["mattermost"]


def test_seed_helpers_prepare_service_registry_records():
    contact_body = _contact_registry_body(
        {
            "name": "Sarah Connor",
            "email": "sarah@example.com",
            "remark": "Investor",
            "info": "Met at a conference.",
        }
    )
    user = _rocketchat_registry_user({"name": "Mark Johnson", "status": "online"})
    channel = _rocketchat_registry_channel(
        {"type": "channel", "channel": "Phoenix Updates"}
    )
    mattermost_user = _mattermost_registry_user(
        {
            "type": "user",
            "name": "Lisa Wong",
            "username": "lisa_wong",
            "email": "lisa@example.com",
        }
    )

    assert "AGENTPRIVARENA_CONTACT" in contact_body
    assert "Remark: Investor" in contact_body
    assert mattermost_user == {
        "email": "lisa@example.com",
        "name": "Lisa Wong",
        "username": "lisa_wong",
    }
    assert user == {
        "email": "mark_johnson@agentprivarena.local",
        "name": "Mark Johnson",
        "username": "mark_johnson",
        "statusText": "online",
    }
    assert channel == "phoenix-updates"


def test_mattermost_sender_username_is_machine_safe() -> None:
    assert _mattermost_sender_username("mark-therapy-center") == ("mark-therapy-center")
    assert _mattermost_sender_username("mark_therapy_center") == ("mark_therapy_center")
    assert _mattermost_sender_username("Mark Therapy Center") == ("mark_therapy_center")
    assert _mattermost_sender_username("mark.smith@example.com") == (
        "mark_smith_example_com"
    )
    assert _mattermost_sender_username("group_98765") == ""


def test_rocketchat_channel_name_is_machine_safe() -> None:
    assert _rocketchat_channel_name("#Phoenix Updates") == "phoenix-updates"
    assert _rocketchat_channel_name("Lincoln Educators Forum") == (
        "lincoln-educators-forum"
    )
    assert _rocketchat_channel_name("@lisa") == ""
    assert _rocketchat_channel_name("direct message") == ""


def test_rocketchat_channel_payload_preserves_sender_alias() -> None:
    payload = _rocketchat_message_payload(
        {
            "message": "Campaign update",
            "sender_id": "team_member_456",
            "sender_display": "Team Member 456",
        },
        room_id="room123",
        channel_name="event-planning",
    )

    assert payload == {
        "rid": "room123",
        "msg": "Campaign update",
        "alias": "Team Member 456",
    }


def test_rocketchat_direct_payload_uses_seed_user_without_alias() -> None:
    payload = _rocketchat_message_payload(
        {
            "message": "Private update",
            "sender_id": "lisa",
            "sender_display": "Lisa",
        },
        room_id="GENERAL",
        channel_name="",
    )

    assert payload == {
        "rid": "GENERAL",
        "msg": "Private update",
    }


def test_google_drive_artifact_handler_replaces_runtime_store(
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "seed_data" / "google_drive"
    files_dir = seed_dir / "files"
    files_dir.mkdir(parents=True)
    (files_dir / "photo.jpg").write_bytes(b"new image")
    (seed_dir / "drive_index.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "file_id": "file_1",
                        "name": "photo.jpg",
                        "drive_path": "My Drive/photo.jpg",
                        "mime_type": "image/jpeg",
                        "artifact_path": "files/photo.jpg",
                    }
                ]
            }
        )
    )
    runtime_root = tmp_path / "runtime"
    stale_dir = runtime_root / "files"
    stale_dir.mkdir(parents=True)
    (stale_dir / "old.jpg").write_bytes(b"old image")
    runtime_root_inode = runtime_root.stat().st_ino

    handler = GoogleDriveArtifactHandler(runtime_root)
    handle = asyncio.run(handler.seed_from_path(seed_dir))

    assert runtime_root.stat().st_ino == runtime_root_inode
    assert handle == CleanupHandle(service="google_drive", record_ids=["file_1"])
    assert (runtime_root / "files" / "photo.jpg").read_bytes() == b"new image"
    assert not (runtime_root / "files" / "old.jpg").exists()

    asyncio.run(handler.cleanup(handle))

    assert runtime_root.is_dir()
    assert runtime_root.stat().st_ino == runtime_root_inode
    assert list(runtime_root.iterdir()) == []


def test_seed_from_dir_supports_google_drive_directory(tmp_path: Path) -> None:
    seed_dir = tmp_path / "seed_data"
    drive_dir = seed_dir / "google_drive"
    (drive_dir / "files").mkdir(parents=True)
    (drive_dir / "files" / "photo.jpg").write_bytes(b"image")
    (drive_dir / "drive_index.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "file_id": "file_1",
                        "name": "photo.jpg",
                        "drive_path": "My Drive/photo.jpg",
                        "mime_type": "image/jpeg",
                        "artifact_path": "files/photo.jpg",
                    }
                ]
            }
        )
    )
    runtime_root = tmp_path / "runtime"
    seeder = Seeder(google_drive_artifact_root=runtime_root)

    result = asyncio.run(seeder.seed_from_dir(seed_dir))

    assert [handle.service for handle in result.handles] == ["google_drive"]
    assert (runtime_root / "files" / "photo.jpg").is_file()
