"""Prepare generated seed records before writing live-service fixtures.

The raw PrivacyLens trajectories were produced by a mock tool engine. Some
observations are not directly seedable into real services: search results may
contain only IDs, chat search results may omit sender metadata, and a failed
mock read may appear as an ``{"exception": ...}`` payload. This module is a
small repair layer between trajectory parsing and service-specific seed files.
It keeps the generated live fixtures honest while preserving repair decisions in
``seed_manifest.json`` for auditability.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any


MANIFEST_FILENAME = "seed_manifest.json"
DROPPED_TASKS_FILENAME = "dropped_tasks.json"
_DEFAULT_TIME_BASE = datetime(2022, 1, 1, tzinfo=UTC)
_SENDER_SUFFIX_RE = re.compile(r"\s[-–—]\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*$")
_SENDER_SELF_INTRO_RE = re.compile(
    r"\b(?:this is|it's|it is)\s+"
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})"
    r"(?:\s+from|\s+here|[,.!]|$)"
)
_INVALID_INFERRED_SENDERS = {
    "John",
    "Jane",
    "Meeting",
    "Project",
    "Reminder",
    "Team",
    "Update",
}


@dataclass
class PreparedSeed:
    """Structured result of the pre-seed repair pass."""

    service_records: dict[str, list[dict[str, Any]]]
    manifest: dict[str, Any]


def prepare_seed_records(
    task_name: str,
    service_records: dict[str, list[dict[str, Any]]],
) -> PreparedSeed:
    """Repair raw converted records and return seedable records plus manifest."""
    prepared: dict[str, list[dict[str, Any]]] = {}
    manifest: dict[str, Any] = {
        "version": 1,
        "task": task_name,
        "services": {},
        "drop_recommended": False,
    }

    for service, records in sorted(service_records.items()):
        cloned_records = copy.deepcopy(records)
        if service == "bookstack":
            out, service_manifest = _prepare_bookstack(cloned_records)
        elif service == "mattermost":
            out, service_manifest = _prepare_mattermost(cloned_records)
        elif service == "radicale":
            out, service_manifest = _prepare_radicale(cloned_records)
        elif service == "mailpit":
            out, service_manifest = _prepare_mailpit(cloned_records)
        elif service == "rocketchat":
            out, service_manifest = _prepare_rocketchat(cloned_records)
        elif service == "gotosocial":
            out, service_manifest = _prepare_gotosocial(cloned_records)
        else:
            out, service_manifest = _prepare_passthrough(cloned_records)

        if cloned_records and not out and not service_manifest.get("drop_recommended"):
            _record_warning(
                service_manifest,
                "all_records_dropped_no_seedable_content",
                {"records_in": len(cloned_records)},
            )
            service_manifest["drop_recommended"] = True

        prepared[service] = out
        manifest["services"][service] = service_manifest
        if service_manifest.get("drop_recommended"):
            manifest["drop_recommended"] = True

    return PreparedSeed(service_records=prepared, manifest=manifest)


def _base_manifest(records_in: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records_in": len(records_in),
        "records_out": 0,
        "repairs": {},
        "warnings": [],
        "dropped_records": [],
        "drop_recommended": False,
    }


def _count_repair(manifest: dict[str, Any], key: str) -> None:
    repairs = manifest["repairs"]
    repairs[key] = repairs.get(key, 0) + 1


def _record_warning(manifest: dict[str, Any], key: str, detail: dict[str, Any]) -> None:
    manifest["warnings"].append({"kind": key, **detail})


def _drop_record(
    manifest: dict[str, Any],
    key: str,
    record: dict[str, Any],
    *,
    detail: dict[str, Any] | None = None,
) -> None:
    item = {"kind": key, "record": record}
    if detail:
        item.update(detail)
    manifest["dropped_records"].append(item)


def _drop_raw_observations(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        if "raw_observation" not in record:
            out.append(record)
            continue
        _drop_record(manifest, "raw_observation_not_seedable", record)
        _record_warning(
            manifest,
            "raw_observation_not_seedable",
            {"action": record.get("action", "")},
        )
        manifest["drop_recommended"] = True
    return out


def _prepare_passthrough(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _base_manifest(records)
    records = _drop_raw_observations(records, manifest)
    manifest["records_out"] = len(records)
    return records, manifest


def _prepare_bookstack(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop read-page duplicates when a search result already has the body."""
    manifest = _base_manifest(records)
    records = _drop_raw_observations(records, manifest)
    out: list[dict[str, Any]] = []
    seen_content: set[str] = set()

    for record in records:
        prepared = copy.deepcopy(record)
        title = str(prepared.get("title") or "").strip()
        if title:
            prepared["title"] = title

        content_key = _content_key(prepared.get("content"))
        is_synthesized_read_page = not prepared.get("id")
        if content_key and is_synthesized_read_page and content_key in seen_content:
            _count_repair(manifest, "merged_duplicate_read_page")
            _drop_record(
                manifest,
                "duplicate_read_page_covered_by_search_result",
                prepared,
                detail={"title": title},
            )
            continue

        if content_key:
            seen_content.add(content_key)
        out.append(prepared)

    manifest["records_out"] = len(out)
    return out, manifest


def _prepare_radicale(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop ID-only search markers when detailed events are present."""
    manifest = _base_manifest(records)
    records = _drop_raw_observations(records, manifest)
    detailed_ids = {
        record.get("event_id")
        for record in records
        if record.get("event_id") and record.get("event_name")
    }

    out: list[dict[str, Any]] = []
    for record in records:
        if set(record.keys()) == {"event_id"}:
            event_id = record["event_id"]
            if event_id in detailed_ids:
                _count_repair(manifest, "merged_id_only_search_result")
                _drop_record(
                    manifest,
                    "id_only_search_result_covered_by_detail",
                    record,
                )
                continue
            _record_warning(
                manifest,
                "unmatched_id_only_event",
                {"event_id": event_id},
            )
            manifest["drop_recommended"] = True
            continue
        out.append(record)

    manifest["records_out"] = len(out)
    return out, manifest


def _prepare_mailpit(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep read-level email content and drop search-only email headers."""
    manifest = _base_manifest(records)
    records = _drop_raw_observations(records, manifest)
    out = copy.deepcopy(records)
    dropped: set[int] = set()

    full_by_subject: dict[str, int] = {}
    for index, record in enumerate(out):
        if _has_email_body(record):
            subject = str(record.get("subject") or "")
            if subject and subject not in full_by_subject:
                full_by_subject[subject] = index

    for index, record in enumerate(out):
        if "exception" in record:
            _count_repair(manifest, "moved_exception_to_negative_observation")
            _drop_record(manifest, "negative_observation", record)
            dropped.add(index)
            continue

        if record.get("type") == "contact":
            _count_repair(manifest, "prepared_contact_registry_record")
            continue

        if not _is_header_only_email(record):
            continue

        target_index = full_by_subject.get(str(record.get("subject") or ""))
        if target_index is None or target_index == index:
            _drop_record(
                manifest,
                "search_result_only_email_without_read_observation",
                record,
            )
            dropped.add(index)
            continue

        target = out[target_index]
        for key, value in record.items():
            if key not in target or target.get(key) in ("", [], None):
                target[key] = value
        _count_repair(manifest, "merged_header_only_email")
        _drop_record(
            manifest,
            "header_only_email_covered_by_detail",
            record,
            detail={"merged_into_subject": target.get("subject", "")},
        )
        dropped.add(index)

    records_out = [record for index, record in enumerate(out) if index not in dropped]
    manifest["records_out"] = len(records_out)
    return records_out, manifest


def _prepare_mattermost(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fill deterministic time metadata and conservative sender hints."""
    manifest = _base_manifest(records)
    records = _drop_raw_observations(records, manifest)
    total = len(records)
    out: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        if "message" not in record:
            if record.get("type") == "user":
                _count_repair(manifest, "prepared_user_registry_record")
            out.append(record)
            continue

        prepared = copy.deepcopy(record)
        original_time = str(prepared.get("time") or "").strip()
        if original_time:
            prepared["seed_time_ms"] = _to_epoch_ms(original_time)
        else:
            prepared["seed_time_ms"] = _deterministic_message_time_ms(index, total)
            prepared["time"] = _iso_from_epoch_ms(prepared["seed_time_ms"])
            _count_repair(manifest, "filled_missing_time")

        sender_id = str(prepared.get("sender_id") or "").strip()
        if sender_id:
            prepared["sender_display"] = _display_from_sender_id(sender_id)
        else:
            inferred = infer_sender_from_message(str(prepared.get("message") or ""))
            if inferred:
                prepared["sender_id"] = _sender_id_from_display(inferred)
                prepared["sender_display"] = inferred
                prepared["sender_inferred"] = True
                _count_repair(manifest, "inferred_sender_from_message_text")
            else:
                prepared["sender_unknown"] = True
                _record_warning(
                    manifest,
                    "missing_sender_unresolved",
                    {
                        "message_id": prepared.get("message_id", ""),
                        "message_preview": str(prepared.get("message", ""))[:160],
                    },
                )
                manifest["drop_recommended"] = True
        out.append(prepared)

    manifest["records_out"] = len(out)
    return out, manifest


def _prepare_rocketchat(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _base_manifest(records)
    records = _drop_raw_observations(records, manifest)
    out: list[dict[str, Any]] = []
    for record in records:
        prepared = copy.deepcopy(record)
        if "message" in prepared:
            sender_id = str(prepared.get("sender_id") or "").strip()
            if sender_id:
                sender_display = _display_from_chat_handle(sender_id)
                prepared["sender_id"] = _sender_id_from_display(sender_display)
                prepared["sender_display"] = sender_display
            out.append(prepared)
        else:
            if prepared.get("type") == "channel":
                _count_repair(manifest, "prepared_channel_registry_record")
            elif prepared.get("type") == "user_profile":
                _count_repair(manifest, "prepared_user_registry_record")
                _record_warning(
                    manifest,
                    "user_profile_extra_fields_need_service_support",
                    {
                        "email": (prepared.get("profile") or {}).get("email", ""),
                        "fields": sorted((prepared.get("profile") or {}).keys()),
                    },
                )
                manifest["drop_recommended"] = True
            else:
                _count_repair(manifest, "prepared_user_registry_record")
            out.append(prepared)
    manifest["records_out"] = len(out)
    return out, manifest


def _prepare_gotosocial(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _base_manifest(records)
    records = _drop_raw_observations(records, manifest)
    for record in records:
        if record.get("type") == "profile":
            _record_warning(
                manifest,
                "profile_record_not_seedable_without_account_registry",
                {
                    "user_id": record.get("user_id", ""),
                    "name": record.get("name", ""),
                },
            )
            manifest["drop_recommended"] = True
    manifest["records_out"] = len(records)
    return records, manifest


def _has_email_body(record: dict[str, Any]) -> bool:
    return bool(record.get("body") or record.get("content") or record.get("snippet"))


def _content_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _is_header_only_email(record: dict[str, Any]) -> bool:
    required = {"from", "to", "subject"}
    return required.issubset(record) and not _has_email_body(record)


def infer_sender_from_message(message: str) -> str | None:
    """Infer a sender only from explicit author markers in message text."""
    for pattern in (_SENDER_SUFFIX_RE, _SENDER_SELF_INTRO_RE):
        match = pattern.search(message)
        if not match:
            continue
        name = match.group(1).strip()
        if name not in _INVALID_INFERRED_SENDERS:
            return name
    return None


def _sender_id_from_display(display: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", display.casefold())
    return "_".join(tokens) or "unknown_sender"


def _display_from_sender_id(sender_id: str) -> str:
    sender_id = sender_id.strip()
    if not sender_id:
        return sender_id
    if sender_id.lower().startswith("group_"):
        return sender_id
    without_suffix = re.sub(r"[_-]\d+$", "", sender_id)
    tokens = [token for token in re.split(r"[_\-.]+", without_suffix) if token]
    if not tokens or not all(token.isalpha() for token in tokens):
        return sender_id
    return " ".join(token.capitalize() for token in tokens)


def _display_from_chat_handle(sender_id: str) -> str:
    sender_id = sender_id.strip()
    if not sender_id:
        return sender_id
    handle = sender_id.lstrip("@")
    tokens = [token for token in re.split(r"[_\-.]+", handle) if token]
    if tokens and all(token.isalpha() for token in tokens):
        return " ".join(token.capitalize() for token in tokens)
    return handle or sender_id


def _deterministic_message_time_ms(index: int, total: int) -> int:
    # Earlier records in benchmark observations should sort as newer.
    dt = _DEFAULT_TIME_BASE + timedelta(seconds=max(total - index, 0))
    return int(dt.timestamp() * 1000)


def _iso_from_epoch_ms(value: int) -> str:
    return (
        datetime.fromtimestamp(value / 1000, tz=UTC).isoformat().replace("+00:00", "Z")
    )


def _to_epoch_ms(value: str) -> int:
    value = value.strip()
    if not value:
        return int(_DEFAULT_TIME_BASE.timestamp() * 1000)
    if value.isdigit():
        number = int(value)
        return number if number > 10_000_000_000 else number * 1000
    normalized = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return int(_DEFAULT_TIME_BASE.timestamp() * 1000)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)
