"""Forced-read planner for the oracle retrieval ablation.

This module deliberately stays outside prompt construction. It converts seeded
live-service records into read-only SDK startup actions, using only seed_data and
runtime service IDs needed to address the live records.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal, cast

import httpx

from agentprivarena.base.seeder import SeedResult
from agentprivarena.base.trajectory_evaluator import is_coverage_target_record
from agentprivarena.config import Config
from agentprivarena.runner.tool_registry import SERVICE_TOOL_SPECS


ReadPolicy = Literal["natural", "forced_oracle"]
VALID_READ_POLICIES: tuple[ReadPolicy, ...] = ("natural", "forced_oracle")


class ForcedReadResolver:
    """Resolve runtime IDs that are assigned by backing services."""

    def __init__(self, config: Config) -> None:
        self.config = config

    async def mailpit_email_id(self, record: dict[str, Any]) -> str:
        """Resolve a seeded Mailpit email to its runtime message ID."""
        subject = str(record.get("subject") or "").strip()
        if not subject:
            raise ValueError("Cannot resolve Mailpit email without a subject.")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.config.mailpit_api_url}/api/v1/search",
                params={"query": subject, "limit": 50},
            )
            response.raise_for_status()

        messages = response.json().get("messages", [])
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("Subject") or "") != subject:
                continue
            email_id = str(message.get("ID") or "").strip()
            if email_id:
                return email_id

        raise ValueError(f"Seeded Mailpit email not found by subject: {subject!r}")

    async def gotosocial_user_id(self) -> str:
        """Return the configured local GoToSocial account ID."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.config.gotosocial_url}/api/v1/accounts/verify_credentials",
                headers={"Authorization": f"Bearer {self.config.gotosocial_token}"},
            )
            response.raise_for_status()

        user_id = str(response.json().get("id") or "").strip()
        if not user_id:
            raise ValueError("GoToSocial verify_credentials returned no account id.")
        return user_id


async def build_initial_read_actions(
    *,
    task_dir: Path,
    seed_result: SeedResult | None,
    config: Config,
    resolver: ForcedReadResolver | None = None,
) -> list[dict[str, Any]]:
    """Build SDK ``initial_read_actions`` for all coverage-target seed records."""
    seed_dir = task_dir / "seed_data"
    if not seed_dir.exists():
        return []

    resolver = resolver or ForcedReadResolver(config)
    actions: list[dict[str, Any]] = []

    for seed_path in sorted(seed_dir.iterdir(), key=lambda path: path.name):
        service = _seed_service_name(seed_path)
        if service is None:
            continue
        spec = SERVICE_TOOL_SPECS.get(service)
        if spec is None or not spec.supports_forced_read:
            continue

        if service == "google_drive":
            if seed_path.is_dir():
                actions.extend(_google_drive_actions(seed_path))
            continue

        if not seed_path.is_file() or seed_path.suffix != ".json":
            continue
        records = _load_target_records(seed_path)
        if not records:
            continue

        actions.extend(
            await _json_service_actions(
                service,
                records=records,
                seed_result=seed_result,
                resolver=resolver,
            )
        )

    return actions


def _seed_service_name(seed_path: Path) -> str | None:
    if seed_path.is_file() and seed_path.suffix == ".json":
        return seed_path.stem
    if seed_path.is_dir():
        return seed_path.name
    return None


async def _json_service_actions(
    service: str,
    *,
    records: list[dict[str, Any]],
    seed_result: SeedResult | None,
    resolver: ForcedReadResolver,
) -> list[dict[str, Any]]:
    if service == "bookstack":
        return _bookstack_actions(records, seed_result)
    if service == "radicale":
        return _radicale_actions(records)
    if service == "mailpit":
        return await _mailpit_actions(records, resolver)
    if service == "mattermost":
        return [
            _action(
                "mattermost_read_messages",
                {"max_count": 200},
                "Forced oracle read of seeded Mattermost messages.",
            )
        ]
    if service == "rocketchat":
        return _rocketchat_actions(records)
    if service == "gotosocial":
        user_id = await resolver.gotosocial_user_id()
        return [
            _action(
                "gotosocial_read_user_posts",
                {"user_id": user_id},
                "Forced oracle read of seeded GoToSocial posts.",
            )
        ]
    return []


def validate_read_policy(read_policy: str) -> ReadPolicy:
    if read_policy not in VALID_READ_POLICIES:
        expected = ", ".join(VALID_READ_POLICIES)
        raise ValueError(f"Unknown read_policy {read_policy!r}; expected {expected}")
    return cast(ReadPolicy, read_policy)


def _load_target_records(seed_file: Path) -> list[dict[str, Any]]:
    loaded = json.loads(seed_file.read_text())
    if not isinstance(loaded, list):
        return []
    return [
        record
        for record in loaded
        if isinstance(record, dict) and is_coverage_target_record(record)
    ]


def _handle_ids(seed_result: SeedResult | None, service: str) -> list[str]:
    if seed_result is None:
        return []
    for handle in seed_result.handles:
        if handle.service == service:
            return list(handle.record_ids)
    return []


def _bookstack_actions(
    records: list[dict[str, Any]],
    seed_result: SeedResult | None,
) -> list[dict[str, Any]]:
    page_ids = _handle_ids(seed_result, "bookstack")
    if len(page_ids) < len(records):
        raise ValueError(
            "BookStack forced-read planning needs one runtime page id per "
            f"coverage record; got {len(page_ids)} ids for {len(records)} records."
        )

    actions: list[dict[str, Any]] = []
    for page_id in page_ids[: len(records)]:
        actions.append(
            _action(
                "bookstack_read_page",
                {"page_id": int(page_id)},
                f"Forced oracle read of BookStack page {page_id}.",
            )
        )
    return actions


def _radicale_actions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for record in records:
        event_id = str(record.get("event_id") or "").strip()
        if not event_id:
            raise ValueError("Radicale forced-read record is missing event_id.")
        actions.append(
            _action(
                "radicale_read_event",
                {"event_id": event_id},
                f"Forced oracle read of Radicale event {event_id}.",
            )
        )
    return actions


async def _mailpit_actions(
    records: list[dict[str, Any]],
    resolver: ForcedReadResolver,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for record in records:
        email_id = await resolver.mailpit_email_id(record)
        actions.append(
            _action(
                "mailpit_read_email",
                {"email_id": email_id},
                f"Forced oracle read of Mailpit email {email_id}.",
            )
        )
    return actions


def _rocketchat_actions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    channels = sorted(
        {
            _rocketchat_channel_name(record.get("channel")) or "general"
            for record in records
        }
    )
    return [
        _action(
            "rocketchat_read_channel_history",
            {"channel": channel, "count": 100},
            f"Forced oracle read of RocketChat channel {channel}.",
        )
        for channel in channels
    ]


def _google_drive_actions(seed_path: Path) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for file in _load_google_drive_files(seed_path):
        file_id = str(file.get("file_id") or "").strip()
        if not file_id:
            raise ValueError(
                f"Google Drive forced-read file in {seed_path} is missing file_id."
            )
        actions.append(
            _action(
                "google_drive_get_file_metadata",
                {"file_id": file_id},
                f"Forced oracle read of Google Drive metadata for {file_id}.",
            )
        )
        if not _google_drive_file_is_available_image(seed_path, file):
            continue
        actions.extend(
            [
                _action(
                    "google_drive_get_file_image",
                    {"file_id": file_id},
                    f"Forced oracle read of Google Drive image {file_id}.",
                ),
                _action(
                    "google_drive_describe_image",
                    {"file_id": file_id, "detail": "standard"},
                    (
                        "Forced oracle visual description of "
                        f"Google Drive image {file_id}."
                    ),
                ),
            ]
        )
    return actions


def _load_google_drive_files(seed_path: Path) -> list[dict[str, Any]]:
    index_path = seed_path / "drive_index.json"
    if not index_path.is_file():
        raise ValueError(f"Google Drive forced-read index not found: {index_path}")
    loaded = json.loads(index_path.read_text())
    files = loaded.get("files") if isinstance(loaded, dict) else None
    if not isinstance(files, list):
        raise ValueError(
            f"Google Drive forced-read index has no files list: {index_path}"
        )
    out: list[dict[str, Any]] = []
    for index, file in enumerate(files):
        if not isinstance(file, dict):
            raise ValueError(
                f"Google Drive forced-read file entry {index} is not an object."
            )
        out.append(file)
    return out


def _google_drive_file_is_available_image(
    seed_path: Path, file: dict[str, Any]
) -> bool:
    mime_type = str(file.get("mime_type") or "").casefold()
    if not mime_type.startswith("image/"):
        return False
    artifact_path = str(file.get("artifact_path") or "").strip()
    if not artifact_path:
        return False
    candidate = Path(artifact_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(
            "Google Drive forced-read file "
            f"{file.get('file_id')!r} uses unsafe artifact_path."
        )
    root = seed_path.resolve()
    resolved = (seed_path / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(
            "Google Drive forced-read file "
            f"{file.get('file_id')!r} resolves outside seed directory."
        )
    return resolved.is_file()


def _rocketchat_channel_name(value: Any) -> str:
    channel = str(value or "").strip()
    if not channel:
        return ""
    lowered = channel.casefold()
    if lowered in {"direct message", "dm"} or channel.startswith("@"):
        return ""
    channel = channel.removeprefix("#").strip()
    tokens = re.findall(r"[a-z0-9]+", channel.casefold())
    return "-".join(tokens)


def _action(tool_name: str, arguments: dict[str, Any], summary: str) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "arguments": arguments,
        "summary": summary,
    }
