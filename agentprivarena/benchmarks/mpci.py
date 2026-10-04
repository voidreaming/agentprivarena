"""MPCI-Bench adapter for AgentPrivArena task directories.

This module converts MPCI-Bench records into the runtime contract already used
by the live platform:

``task.json``
    User-facing task metadata consumed by :class:`AgentPrivArenaRunner`.
``seed_data/*.json``
    Per-service records consumed by the live seeder.
``seed_manifest.json``
    Audit trail for conversion and artifact availability.
``oracle.json``
    Benchmark-only metadata that should not be injected into the agent prompt
    or service seed data.

The adapter deliberately preserves VISPR image references as artifacts without
transcribing labels into seed data. The current live stack can run the textual
service portion immediately; the artifact section is the stable hook for a
multimodal runner or image-capable MCP service.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from agentprivarena.base.trajectory_parser import (
    TrajectoryStep,
    get_seed_observations,
    parse_trajectory,
)
from agentprivarena.config import ACTION_TO_SERVICE
from agentprivarena.tasks.generate import (
    _channel_from_instruction,
    _channel_registry_record,
    _has_channel_identity,
    _has_target_identity,
    _observation_to_seed_records,
    _rewrite_instruction,
    _slug_identifier,
    _target_from_instruction,
    _target_registry_record,
)
from agentprivarena.tasks.structured_seed import (
    DROPPED_TASKS_FILENAME,
    MANIFEST_FILENAME,
    PreparedSeed,
    prepare_seed_records,
)


logger = logging.getLogger(__name__)

BENCHMARK_NAME = "mpci_bench"
ORACLE_FILENAME = "oracle.json"

REQUIRED_TOP_LEVEL_KEYS = {"name", "seed", "story", "trace", "img_metadata"}
REQUIRED_TRACE_KEYS = {
    "user_instruction",
    "toolkits",
    "executable_trajectory",
    "final_action",
}

MPCI_ACTION_TO_SERVICE: dict[str, str] = {
    **ACTION_TO_SERVICE,
    "GoogleDriveSearchFiles": "google_drive",
    "GoogleDriveReadFile": "google_drive",
    "GoogleDriveUploadFile": "google_drive",
    "GoogleDriveShareFile": "google_drive",
}

MPCI_TOOLKIT_TO_MCP: dict[str, str] = {
    "NotionManager": "bookstack",
    "GoogleDriveManager": "google_drive",
    "ZoomManager": "bookstack",
    "GoogleFormFiller": "bookstack",
    "Messenger": "mattermost",
    "Slack": "rocketchat",
    "Gmail": "mailpit",
    "FacebookManager": "gotosocial",
    "GoogleCalendar": "radicale",
}

GOOGLE_DRIVE_ACTIONS = {
    "GoogleDriveSearchFiles",
    "GoogleDriveReadFile",
    "GoogleDriveUploadFile",
    "GoogleDriveShareFile",
}
_ADDRESSABLE_TARGET_SERVICES = {"mailpit", "mattermost", "rocketchat"}
MISSING_IMAGE_ARTIFACT = "missing_image_artifact"
PAIR_INCOMPLETE_AFTER_QUALITY_FILTER = "pair_incomplete_after_quality_filter"
SEED_MANIFEST_DROP_RECOMMENDED = "seed_manifest_drop_recommended"
_CHAT_HANDLE_TARGET_RE = re.compile(
    r"\b(?P<name>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})"
    r"\s*\(@(?P<username>[A-Za-z0-9][A-Za-z0-9_.-]{1,80})\)"
)
_CHAT_HANDLE_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])@(?P<username>[A-Za-z0-9][A-Za-z0-9_-]{1,80})"
    r"(?![A-Za-z0-9_.-])"
)
_EMAIL_ADDRESS_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ATTACHMENT_REQUEST_RE = re.compile(
    r"\b(?:attach(?:ed|es|ing)?|attachment|attachments)\b",
    re.IGNORECASE,
)
_SURVEILLANCE_OR_PROFILING_RE = re.compile(
    r"\b(?:surveillance|profiling)\b",
    re.IGNORECASE,
)
_DIRECTED_DEMONSTRATION_MESSAGE_RE = re.compile(
    r"\b(?:public\s+demonstration|demonstration\s+event)\b",
    re.IGNORECASE,
)
_DEMONSTRATION_SENSITIVE_RECIPIENT_RE = re.compile(
    r"\b(?:employer|human\s+rights)\b",
    re.IGNORECASE,
)
_BACKGROUND_SCREENING_ACTIVIST_RE = re.compile(
    r"\bbackground\s+screening\b.*\bactivist\b|"
    r"\bactivist\b.*\bbackground\s+screening\b",
    re.IGNORECASE,
)
_CHAT_SERVICE_INSTRUCTION_RE: dict[str, re.Pattern[str]] = {
    "mattermost": re.compile(r"\bMattermost\b", re.IGNORECASE),
    "rocketchat": re.compile(r"\bRocketChat\b", re.IGNORECASE),
}
_CHAT_SERVICE_DISPLAY_NAMES = frozenset({"mattermost", "rocketchat"})
_REPLY_EMAIL_SENDER_RE = re.compile(
    r"\breply\s+to\s+(?:the\s+)?email\s+from\s+(?:the\s+)?"
    r"(?P<sender>[^.?!,\n]+)",
    re.IGNORECASE,
)
_REPLY_REQUEST_RE = re.compile(r"\breply\s+to\b", re.IGNORECASE)
_REPLY_SENDER_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "email",
        "from",
        "message",
        "sender",
        "the",
    }
)


def load_mpci_entries(data_path: Path) -> list[dict[str, Any]]:
    """Load and validate the released MPCI-Bench JSON list."""
    data = json.loads(data_path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected MPCI-Bench JSON list in {data_path}")
    for index, entry in enumerate(data):
        _validate_entry(entry, index=index)
    return data


def generate_mpci_tasks(
    data_path: Path,
    output_dir: Path,
    *,
    image_root: Path | None = None,
    names: Iterable[str] | None = None,
    limit: int | None = None,
    quality_filter: bool = False,
    require_images: bool = False,
    paired_only: bool = False,
) -> dict[str, Any]:
    """Generate runner-compatible task directories from MPCI-Bench.

    Args:
        data_path: Path to ``mpci_bench.json``.
        output_dir: Directory where task subdirectories will be written.
        image_root: Optional VISPR/MPCI root used to copy image artifacts.
        names: Optional scenario-name allowlist.
        limit: Optional maximum number of selected entries to generate.
        quality_filter: Drop entries whose seed manifest recommends dropping.
        require_images: Drop entries with unavailable image artifacts.
        paired_only: Keep only pair ids whose positive and negative entries both
            survive the other quality filters.

    Returns:
        Summary metadata written in the same spirit as
        ``dropped_tasks.json`` for existing PrivacyLens task generation.
    """
    entries = _select_entries(load_mpci_entries(data_path), names=names, limit=limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    kept: list[str] = []
    dropped: dict[str, list[str]] = {}
    missing_images: list[dict[str, str]] = []
    for entry in entries:
        task_dir = _write_mpci_task(entry, output_dir, image_root=image_root)
        manifest = json.loads((task_dir / MANIFEST_FILENAME).read_text())
        reasons: list[str] = []
        for artifact in manifest.get("image_artifacts", []):
            if not artifact.get("available"):
                missing_images.append(
                    {
                        "name": entry["name"],
                        "source_path": artifact.get("source_path", ""),
                    }
                )
                if require_images:
                    _append_reason(reasons, MISSING_IMAGE_ARTIFACT)
        if quality_filter and manifest.get("drop_recommended"):
            for reason in _manifest_drop_reasons(manifest):
                _append_reason(reasons, reason)
        if reasons:
            dropped[str(entry["name"])] = reasons
        else:
            kept.append(task_dir.name)

    if paired_only:
        paired_drops = _paired_only_drops(kept)
        for name in paired_drops:
            _append_reason(
                dropped.setdefault(name, []),
                PAIR_INCOMPLETE_AFTER_QUALITY_FILTER,
            )
        kept = [name for name in kept if name not in paired_drops]

    for name in dropped:
        task_dir = output_dir / name
        if task_dir.exists():
            shutil.rmtree(task_dir)

    dropped_tasks = [
        {"name": name, "reasons": dropped[name]}
        for name in [str(entry["name"]) for entry in entries]
        if name in dropped
    ]

    summary = {
        "version": 1,
        "benchmark": BENCHMARK_NAME,
        "source": str(data_path),
        "total_entries": len(entries),
        "generated_tasks": len(kept),
        "generated_names": kept,
        "missing_image_artifacts": missing_images,
        "dropped_tasks": dropped_tasks,
        "quality_filter": {
            "enabled": quality_filter,
            "require_images": require_images,
            "paired_only": paired_only,
        },
    }
    (output_dir / DROPPED_TASKS_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _manifest_drop_reasons(manifest: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for service, service_manifest in manifest.get("services", {}).items():
        if not service_manifest.get("drop_recommended"):
            continue
        warnings = service_manifest.get("warnings", [])
        for warning in warnings:
            if isinstance(warning, dict):
                kind = warning.get("kind", SEED_MANIFEST_DROP_RECOMMENDED)
                _append_reason(reasons, f"{service}:{kind}")
        if not warnings:
            _append_reason(reasons, f"{service}:{SEED_MANIFEST_DROP_RECOMMENDED}")
    return reasons or [SEED_MANIFEST_DROP_RECOMMENDED]


def _paired_only_drops(names: Sequence[str]) -> set[str]:
    by_pair: dict[str, set[str]] = defaultdict(set)
    for name in names:
        by_pair[_pair_id(name)].add(_appropriateness(name))
    complete_pairs = {
        pair_id
        for pair_id, labels in by_pair.items()
        if {"appropriate", "inappropriate"}.issubset(labels)
    }
    return {name for name in names if _pair_id(name) not in complete_pairs}


def _validate_entry(entry: Any, *, index: int) -> None:
    if not isinstance(entry, dict):
        raise ValueError(f"MPCI entry {index} is not an object")
    missing = REQUIRED_TOP_LEVEL_KEYS - set(entry)
    if missing:
        name = entry.get("name", f"entry[{index}]")
        raise ValueError(f"{name}: missing MPCI keys {sorted(missing)}")
    trace = entry.get("trace")
    if not isinstance(trace, dict):
        raise ValueError(f"{entry['name']}: trace is not an object")
    missing_trace = REQUIRED_TRACE_KEYS - set(trace)
    if missing_trace:
        raise ValueError(f"{entry['name']}: missing trace keys {sorted(missing_trace)}")


def _select_entries(
    entries: Sequence[dict[str, Any]],
    *,
    names: Iterable[str] | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    selected = list(entries)
    if names is not None:
        name_set = {name for name in names if name}
        selected = [entry for entry in selected if entry.get("name") in name_set]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _write_mpci_task(
    entry: dict[str, Any],
    output_dir: Path,
    *,
    image_root: Path | None,
) -> Path:
    name = str(entry["name"])
    task_dir = output_dir / name
    seed_dir = task_dir / "seed_data"
    seed_dir.mkdir(parents=True, exist_ok=True)

    image_artifacts = _copy_image_artifacts(entry, task_dir, image_root=image_root)
    google_drive_files = _write_google_drive_seed(
        entry,
        seed_dir,
        image_artifacts=image_artifacts,
    )
    prepared = _prepare_mpci_entry(entry, image_artifacts=image_artifacts)
    prepared.manifest["benchmark"] = BENCHMARK_NAME
    prepared.manifest["image_artifacts"] = image_artifacts
    prepared.manifest["google_drive_files"] = google_drive_files

    for service, records in prepared.service_records.items():
        seed_file = seed_dir / f"{service}.json"
        seed_file.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    (task_dir / MANIFEST_FILENAME).write_text(
        json.dumps(prepared.manifest, indent=2, ensure_ascii=False)
    )
    (task_dir / "task.json").write_text(
        json.dumps(
            _build_task_spec(entry, image_artifacts=image_artifacts),
            indent=2,
            ensure_ascii=False,
        )
    )
    (task_dir / ORACLE_FILENAME).write_text(
        json.dumps(_build_oracle(entry), indent=2, ensure_ascii=False)
    )
    return task_dir


def _prepare_mpci_entry(
    entry: dict[str, Any],
    *,
    image_artifacts: list[dict[str, Any]],
) -> PreparedSeed:
    trace = entry["trace"]
    steps = parse_trajectory(str(trace["executable_trajectory"]))
    service_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in get_seed_observations(steps):
        if step.action_name in GOOGLE_DRIVE_ACTIONS:
            continue
        service = MPCI_ACTION_TO_SERVICE.get(step.action_name)
        if service is None:
            logger.warning(
                "Unknown MPCI action %s in %s", step.action_name, entry["name"]
            )
            continue
        records = _mpci_observation_to_seed_records(
            step,
            service,
            image_artifacts=image_artifacts,
        )
        service_records[service].extend(records)
    _add_mpci_target_registry_record(entry, service_records)
    _add_instruction_chat_target_registry_records(entry, service_records)

    prepared = prepare_seed_records(str(entry["name"]), dict(service_records))
    _apply_mpci_quality_checks(entry, prepared)
    return prepared


def _mpci_observation_to_seed_records(
    step: TrajectoryStep,
    service: str,
    *,
    image_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    del image_artifacts
    return _observation_to_seed_records(step, service)


def _add_mpci_target_registry_record(
    entry: dict[str, Any],
    service_records: dict[str, list[dict[str, Any]]],
) -> None:
    trace = entry["trace"]
    service = MPCI_ACTION_TO_SERVICE.get(str(trace.get("final_action") or ""))
    if service not in _ADDRESSABLE_TARGET_SERVICES:
        return

    instruction = _rewrite_mpci_instruction(str(trace.get("user_instruction") or ""))
    if service == "rocketchat":
        channel = _channel_from_instruction(instruction)
        if channel and not _has_channel_identity(service_records, service, channel):
            service_records[service].append(_channel_registry_record(channel))
            return

    handle_record = _chat_handle_registry_record(service, instruction)
    if handle_record is not None:
        username = str(handle_record["username"])
        if not _has_username_identity(service_records, service, username):
            service_records[service].append(handle_record)
        return

    target = _target_from_instruction(instruction)
    if target is None or _has_target_identity(service_records, service, target):
        return

    record = _target_registry_record(service, target)
    if record is not None:
        service_records[service].append(record)


def _add_instruction_chat_target_registry_records(
    entry: dict[str, Any],
    service_records: dict[str, list[dict[str, Any]]],
) -> None:
    trace = entry["trace"]
    instruction = _rewrite_mpci_instruction(str(trace.get("user_instruction") or ""))
    dependencies = set(_dependencies(trace))
    for service, pattern in _CHAT_SERVICE_INSTRUCTION_RE.items():
        if service not in dependencies or not pattern.search(instruction):
            continue
        _add_chat_target_registry_record(service, instruction, service_records)


def _add_chat_target_registry_record(
    service: str,
    instruction: str,
    service_records: dict[str, list[dict[str, Any]]],
) -> None:
    if service == "rocketchat":
        channel = _channel_from_instruction(instruction)
        if channel and not _has_channel_identity(service_records, service, channel):
            service_records[service].append(_channel_registry_record(channel))
            return

    handle_record = _chat_handle_registry_record(service, instruction)
    if handle_record is None:
        return

    username = str(handle_record["username"])
    if not _has_username_identity(service_records, service, username):
        service_records[service].append(handle_record)


def _chat_handle_registry_record(
    service: str,
    instruction: str,
) -> dict[str, str] | None:
    if service not in {"mattermost", "rocketchat"}:
        return None

    match = _CHAT_HANDLE_TARGET_RE.search(instruction)
    if match is not None:
        name = match.group("name").strip()
        username = _slug_identifier(match.group("username").lstrip("@"))
        if _slug_identifier(name) in _CHAT_SERVICE_DISPLAY_NAMES:
            name = _display_name_from_handle(username)
    else:
        match = _CHAT_HANDLE_RE.search(instruction)
        if match is None:
            return None
        username = _slug_identifier(match.group("username").lstrip("@"))
        name = _display_name_from_handle(username)

    if not username:
        return None

    if service == "rocketchat":
        return {
            "type": "user",
            "name": name,
            "username": username,
            "email": f"{username}@agentprivarena.local",
            "status": "Task target recipient",
        }
    return {
        "type": "user",
        "name": name,
        "username": username,
        "email": f"{username}@agentprivarena.local",
    }


def _display_name_from_handle(username: str) -> str:
    parts = [part for part in re.split(r"[_-]+", username) if part]
    return " ".join(part.capitalize() for part in parts) or username


def _has_username_identity(
    service_records: dict[str, list[dict[str, Any]]],
    service: str,
    username: str,
) -> bool:
    expected = _slug_identifier(username)
    return any(
        _slug_identifier(str(record.get("username") or record.get("sender_id") or ""))
        == expected
        for record in service_records.get(service, [])
    )


def _apply_mpci_quality_checks(entry: dict[str, Any], prepared: PreparedSeed) -> None:
    _mark_unsupported_mailpit_attachment(entry, prepared)
    _mark_unresolved_mailpit_recipient(entry, prepared)
    _mark_unresolved_mailpit_reply_sender(entry, prepared)
    _mark_agent_actionability_risk(entry, prepared)


def _mark_agent_actionability_risk(
    entry: dict[str, Any],
    prepared: PreparedSeed,
) -> None:
    trace = entry["trace"]
    service = MPCI_ACTION_TO_SERVICE.get(str(trace.get("final_action") or ""))
    if service is None:
        return

    instruction = _rewrite_mpci_instruction(str(trace.get("user_instruction") or ""))
    reason = _agent_actionability_risk_reason(instruction)
    if reason is None:
        return

    _mark_service_drop(
        prepared,
        service,
        {
            "kind": "agent_actionability_risk",
            "reason": reason,
        },
    )


def _agent_actionability_risk_reason(instruction: str) -> str | None:
    if _SURVEILLANCE_OR_PROFILING_RE.search(instruction):
        return "surveillance_or_profiling"
    if _DIRECTED_DEMONSTRATION_MESSAGE_RE.search(
        instruction
    ) and _DEMONSTRATION_SENSITIVE_RECIPIENT_RE.search(instruction):
        return "directed_demonstration_message"
    if _BACKGROUND_SCREENING_ACTIVIST_RE.search(instruction):
        return "background_screening_activist"
    return None


def _mark_unsupported_mailpit_attachment(
    entry: dict[str, Any],
    prepared: PreparedSeed,
) -> None:
    trace = entry["trace"]
    service = MPCI_ACTION_TO_SERVICE.get(str(trace.get("final_action") or ""))
    if service != "mailpit":
        return

    instruction = _rewrite_mpci_instruction(str(trace.get("user_instruction") or ""))
    if not _ATTACHMENT_REQUEST_RE.search(instruction):
        return

    _mark_service_drop(
        prepared,
        "mailpit",
        {
            "kind": "attachment_unsupported",
            "tool": "mailpit_send_email",
        },
    )


def _mark_unresolved_mailpit_recipient(
    entry: dict[str, Any],
    prepared: PreparedSeed,
) -> None:
    trace = entry["trace"]
    service = MPCI_ACTION_TO_SERVICE.get(str(trace.get("final_action") or ""))
    if service != "mailpit":
        return

    instruction = _rewrite_mpci_instruction(str(trace.get("user_instruction") or ""))
    if _EMAIL_ADDRESS_RE.search(instruction):
        return
    if _is_email_reply_instruction(instruction):
        return

    _mark_service_drop(
        prepared,
        "mailpit",
        {
            "kind": "recipient_unresolved",
        },
    )


def _mark_unresolved_mailpit_reply_sender(
    entry: dict[str, Any],
    prepared: PreparedSeed,
) -> None:
    trace = entry["trace"]
    service = MPCI_ACTION_TO_SERVICE.get(str(trace.get("final_action") or ""))
    if service != "mailpit":
        return

    instruction = _rewrite_mpci_instruction(str(trace.get("user_instruction") or ""))
    if _EMAIL_ADDRESS_RE.search(instruction):
        return

    if not _is_email_reply_instruction(instruction):
        return

    sender = _reply_sender_from_instruction(instruction)
    mailpit_records = prepared.service_records.get("mailpit", [])
    if sender is None and _records_contain_real_email(mailpit_records):
        return

    if sender is not None and _mailpit_records_include_sender(mailpit_records, sender):
        return

    _mark_service_drop(
        prepared,
        "mailpit",
        {
            "kind": "reply_sender_unresolved",
            "sender": sender or "reply target",
        },
    )


def _mark_service_drop(
    prepared: PreparedSeed,
    service: str,
    warning: dict[str, Any],
) -> None:
    service_manifest = prepared.manifest["services"].setdefault(
        service,
        {
            "records_in": 0,
            "records_out": 0,
            "repairs": {},
            "warnings": [],
            "dropped_records": [],
            "drop_recommended": False,
        },
    )
    service_manifest["warnings"].append(warning)
    service_manifest["drop_recommended"] = True
    prepared.manifest["drop_recommended"] = True


def _records_contain_real_email(
    records: list[dict[str, Any]],
) -> bool:
    text = json.dumps(records, ensure_ascii=False)
    return any(
        not match.group(0).casefold().endswith("@agentprivarena.local")
        for match in _EMAIL_ADDRESS_RE.finditer(text)
    )


def _is_email_reply_instruction(instruction: str) -> bool:
    return _REPLY_REQUEST_RE.search(instruction) is not None


def _reply_sender_from_instruction(instruction: str) -> str | None:
    match = _REPLY_EMAIL_SENDER_RE.search(instruction)
    if match is None:
        return None
    sender = match.group("sender")
    sender = re.split(
        r"\b(?:about|regarding|with|for)\b",
        sender,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    sender = re.sub(r"\s+", " ", sender).strip(" :;")
    return sender or None


def _mailpit_records_include_sender(
    records: list[dict[str, Any]],
    sender: str,
) -> bool:
    sender_tokens = _identity_tokens(sender)
    if not sender_tokens:
        return True
    for record in records:
        searchable = " ".join(
            str(record.get(key) or "")
            for key in ("from", "from_email", "sender", "email", "name")
        )
        record_tokens = _identity_tokens(searchable)
        if sender_tokens.issubset(record_tokens):
            return True
    return False


def _identity_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if token not in _REPLY_SENDER_STOPWORDS
    }


def _write_google_drive_seed(
    entry: dict[str, Any],
    seed_dir: Path,
    *,
    image_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    files = _google_drive_index_files(entry, image_artifacts=image_artifacts)
    if not files:
        return []

    drive_dir = seed_dir / "google_drive"
    files_dir = drive_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        source_task_path = str(file.pop("_source_task_path", ""))
        if not source_task_path:
            continue
        source = seed_dir.parent / source_task_path
        if source.is_file():
            shutil.copy2(source, drive_dir / file["artifact_path"])

    (drive_dir / "drive_index.json").write_text(
        json.dumps({"version": 1, "files": files}, indent=2, ensure_ascii=False)
    )
    return files


def _google_drive_index_files(
    entry: dict[str, Any],
    *,
    image_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trace_files = _google_drive_trace_files(entry)
    if not trace_files and image_artifacts:
        trace_files = [
            {
                "name": Path(str(artifact["task_path"])).name,
                "path": "",
            }
            for artifact in image_artifacts
        ]

    files: list[dict[str, Any]] = []
    for index, file_obj in enumerate(trace_files):
        artifact = image_artifacts[index] if index < len(image_artifacts) else None
        filename = _drive_filename(file_obj, artifact=artifact, index=index)
        file_id = _drive_file_id(entry["name"], file_obj, index=index)
        drive_path = str(
            file_obj.get("path")
            or file_obj.get("file_path")
            or file_obj.get("drive_path")
            or ""
        )
        item = {
            "file_id": file_id,
            "name": filename,
            "drive_path": drive_path,
            "mime_type": (
                str(artifact.get("mime_type"))
                if artifact is not None
                else _mime_type_for(filename)
            ),
            "artifact_path": f"files/{filename}",
            "available": bool(artifact and artifact.get("available")),
        }
        if artifact is not None:
            item["_source_task_path"] = str(artifact["task_path"])
        files.append(item)
    return files


def _google_drive_trace_files(entry: dict[str, Any]) -> list[dict[str, Any]]:
    trace = entry["trace"]
    steps = parse_trajectory(str(trace["executable_trajectory"]))
    files: list[dict[str, Any]] = []
    for step in get_seed_observations(steps):
        if step.action_name not in GOOGLE_DRIVE_ACTIONS:
            continue
        obs = step.observation
        if not isinstance(obs, dict):
            files.append({"name": str(obs)})
            continue
        observed_files = obs.get("files")
        if isinstance(observed_files, list):
            for item in observed_files:
                files.append(item if isinstance(item, dict) else {"name": str(item)})
            continue
        files.append(obs)
    return _dedupe_drive_files(files)


def _dedupe_drive_files(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for file in files:
        key = (
            str(file.get("id") or file.get("file_id") or ""),
            str(file.get("name") or file.get("title") or file.get("path") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(file)
    return out


def _drive_filename(
    file_obj: dict[str, Any],
    *,
    artifact: dict[str, Any] | None,
    index: int,
) -> str:
    name = str(file_obj.get("name") or file_obj.get("title") or "").strip()
    if name:
        return Path(name).name
    if artifact is not None:
        return Path(str(artifact["task_path"])).name
    return f"google_drive_file_{index}.jpg"


def _drive_file_id(
    task_name: Any,
    file_obj: dict[str, Any],
    *,
    index: int,
) -> str:
    raw = str(file_obj.get("id") or file_obj.get("file_id") or "").strip()
    if not raw:
        raw = f"mpci_{task_name}_{index}"
    tokens = re.findall(r"[a-zA-Z0-9_-]+", raw)
    return "_".join(tokens) or f"mpci_file_{index}"


def _copy_image_artifacts(
    entry: dict[str, Any],
    task_dir: Path,
    *,
    image_root: Path | None,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    metadata = entry.get("img_metadata")
    if not isinstance(metadata, list):
        return artifacts

    artifact_dir = task_dir / "artifacts" / "images"
    for index, item in enumerate(metadata):
        if not isinstance(item, dict):
            continue
        source_path = str(
            item.get("path") or item.get("image_path") or item.get("actual_path") or ""
        )
        filename = Path(source_path).name or f"image_{index}.jpg"
        task_relative_path = Path("artifacts") / "images" / filename
        source = _resolve_image_source(source_path, image_root=image_root)
        artifact = {
            "source_path": source_path,
            "task_path": task_relative_path.as_posix(),
            "mime_type": _mime_type_for(filename),
            "available": source is not None,
        }
        if source is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, task_dir / task_relative_path)
        artifacts.append(artifact)
    return artifacts


def _resolve_image_source(source_path: str, *, image_root: Path | None) -> Path | None:
    if image_root is None or not source_path:
        return None
    filename = Path(source_path).name
    candidates = [
        image_root / source_path,
        image_root / "bench" / "vispr" / "train2017" / filename,
        image_root / "vispr" / "train2017" / filename,
        image_root / "train2017" / filename,
        image_root / filename,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _mime_type_for(filename: str) -> str:
    suffix = Path(filename).suffix.casefold()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _build_task_spec(
    entry: dict[str, Any],
    *,
    image_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    trace = entry["trace"]
    seed = entry["seed"]
    return {
        "name": entry["name"],
        "benchmark": BENCHMARK_NAME,
        "benchmark_case": {
            "source_name": entry["name"],
            "pair_id": _pair_id(str(entry["name"])),
            "appropriateness": _appropriateness(str(entry["name"])),
        },
        "user_instruction": _rewrite_mpci_instruction(str(trace["user_instruction"])),
        "user_name": trace.get("user_name", "John Doe"),
        "user_email": trace.get("user_email", "john.doe@gmail.com"),
        "toolkits": trace.get("toolkits", []),
        "dependencies": _dependencies(trace),
        "final_action_type": trace["final_action"],
        "sensitive_info_items": _sensitive_info_items(seed),
        "artifacts": {"images": image_artifacts},
    }


def _build_oracle(entry: dict[str, Any]) -> dict[str, Any]:
    name = str(entry["name"])
    return {
        "benchmark": BENCHMARK_NAME,
        "source_name": name,
        "pair_id": _pair_id(name),
        "appropriateness": _appropriateness(name),
        "seed": entry["seed"],
        "story": entry["story"],
        "img_metadata": entry["img_metadata"],
    }


def _dependencies(trace: dict[str, Any]) -> list[str]:
    services = {
        MPCI_TOOLKIT_TO_MCP[toolkit]
        for toolkit in trace.get("toolkits", [])
        if toolkit in MPCI_TOOLKIT_TO_MCP
    }
    final_action_service = MPCI_ACTION_TO_SERVICE.get(str(trace.get("final_action")))
    if final_action_service is not None:
        services.add(final_action_service)
    return sorted(services)


def _rewrite_mpci_instruction(instruction: str) -> str:
    placeholder = "__MPCI_GOOGLE_DRIVE__"
    instruction = instruction.replace("GoogleDrive", "Google Drive")
    instruction = instruction.replace("Google Drive", placeholder)
    rewritten = _rewrite_instruction(instruction)
    return rewritten.replace(placeholder, "Google Drive")


def _sensitive_info_items(seed: Any) -> list[str]:
    if not isinstance(seed, dict):
        return []
    data_type = str(seed.get("data_type") or "").strip()
    data_subject = str(seed.get("data_subject") or "").strip()
    if data_type and data_subject:
        return [f"{data_subject}: {data_type}"]
    if data_type:
        return [data_type]
    return []


def _appropriateness(name: str) -> str:
    lower = name.casefold()
    if lower.endswith("_neg") or lower.endswith("_inappropriate"):
        return "inappropriate"
    if lower.endswith("_pos") or lower.endswith("_appropriate"):
        return "appropriate"
    return "unknown"


def _pair_id(name: str) -> str:
    lower = name.casefold()
    for suffix in ("_neg", "_pos", "_inappropriate", "_appropriate"):
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate AgentPrivArena task dirs from MPCI-Bench"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--names", default="")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Drop entries whose seed manifest recommends dropping.",
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help="Drop entries whose image artifacts are unavailable.",
    )
    parser.add_argument(
        "--paired-only",
        action="store_true",
        help="Keep only pair ids whose positive and negative entries both pass.",
    )
    args = parser.parse_args()

    names = [name.strip() for name in args.names.split(",") if name.strip()] or None
    summary = generate_mpci_tasks(
        args.data,
        args.output,
        image_root=args.image_root,
        names=names,
        limit=args.limit,
        quality_filter=args.quality_filter,
        require_images=args.require_images,
        paired_only=args.paired_only,
    )
    print(
        f"Generated {summary['generated_tasks']} MPCI task directories in {args.output}"
    )
    if summary["missing_image_artifacts"]:
        print(
            f"Warning: {len(summary['missing_image_artifacts'])} image artifact(s) "
            "were not found."
        )


if __name__ == "__main__":
    main()
