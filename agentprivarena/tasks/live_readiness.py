"""Live readiness checks for generated PrivacyLens task fixtures.

This verifies the pre-agent contract: after a task's seed_data is injected into
the real local services, the corresponding read tools can retrieve the seeded
content. It is intentionally separate from the static converter verifier because
it touches Docker-backed services and is slower.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mcp.types import ImageContent

from agentprivarena.base.seeder import Seeder
from agentprivarena.config import Config


COMPOSE_FILE = Path(__file__).resolve().parents[1] / "docker-compose.yml"
MCP_DIR = Path(__file__).resolve().parents[1] / "mcp_servers"


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _fragment(value: Any, *, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def _contains(haystack: Any, needle: Any) -> bool:
    normalized_needle = _norm(needle)
    return bool(normalized_needle) and normalized_needle in _norm(haystack)


def _sender_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    expected_values = {
        _norm(expected.get("sender_id")),
        _norm(expected.get("sender_display")),
    }
    expected_values.discard("")
    if not expected_values:
        return True

    actual_values = {
        _norm(actual.get("sender")),
        _norm(actual.get("sender_display")),
    }
    actual_values.discard("")
    return bool(expected_values & actual_values)


def _rocketchat_channel_name(value: Any) -> str:
    channel = str(value or "").strip()
    if not channel:
        return "general"
    if channel.startswith("@"):
        return "general"
    tokens = re.findall(r"[a-z0-9]+", channel.casefold())
    return "-".join(tokens) or "general"


def _quiet_noisy_loggers() -> None:
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("seeder").setLevel(logging.WARNING)
    caldav_logger = logging.getLogger("caldav")
    caldav_logger.setLevel(logging.ERROR)
    caldav_logger.disabled = True


def _compose_host_port(service: str, container_port: int) -> int | None:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "port",
            service,
            str(container_port),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    line = result.stdout.strip().splitlines()[:1]
    if not line:
        return None
    try:
        return int(line[0].rsplit(":", 1)[1])
    except (IndexError, ValueError):
        return None


def _host_url(service: str, container_port: int, fallback: str) -> str:
    port = _compose_host_port(service, container_port)
    if port is None:
        return fallback
    return f"http://localhost:{port}"


def _host_port(service: str, container_port: int, fallback: int) -> int:
    return _compose_host_port(service, container_port) or fallback


def _live_config() -> Config:
    config = Config.from_env()
    updates = {
        "bookstack_url": _host_url("bookstack", 80, config.bookstack_url),
        "mattermost_url": _host_url("mattermost", 8065, config.mattermost_url),
        "rocketchat_url": _host_url("rocketchat", 3000, config.rocketchat_url),
        "mailpit_api_url": _host_url("mailpit", 8025, config.mailpit_api_url),
        "mailpit_smtp_port": _host_port(
            "mailpit",
            1025,
            config.mailpit_smtp_port,
        ),
        "gotosocial_url": _host_url("gotosocial", 8080, config.gotosocial_url),
        "radicale_url": _host_url("radicale", 5232, config.radicale_url),
    }
    return config.model_copy(update=updates)


def _configure_mcp_imports(config: Config) -> None:
    os.environ.update(
        {
            "BOOKSTACK_URL": config.bookstack_url,
            "BOOKSTACK_TOKEN_ID": config.bookstack_token_id,
            "BOOKSTACK_TOKEN_SECRET": config.bookstack_token_secret,
            "MATTERMOST_URL": config.mattermost_url,
            "MATTERMOST_USER": config.mattermost_user,
            "MATTERMOST_PASSWORD": config.mattermost_password,
            "ROCKETCHAT_URL": config.rocketchat_url,
            "ROCKETCHAT_USER": config.rocketchat_user,
            "ROCKETCHAT_PASSWORD": config.rocketchat_password,
            "MAILPIT_API_URL": config.mailpit_api_url,
            "MAILPIT_SMTP_HOST": config.mailpit_smtp_host,
            "MAILPIT_SMTP_PORT": str(config.mailpit_smtp_port),
            "GOTOSOCIAL_URL": config.gotosocial_url,
            "GOTOSOCIAL_TOKEN": config.gotosocial_token,
            "RADICALE_URL": config.radicale_url,
            "RADICALE_USER": config.radicale_user,
            "RADICALE_PASSWORD": config.radicale_password,
            "GOOGLE_DRIVE_ARTIFACT_ROOT": str(config.google_drive_artifact_root),
        }
    )
    if str(MCP_DIR) not in sys.path:
        sys.path.insert(0, str(MCP_DIR))


def _mcp_modules(config: Config) -> dict[str, Any]:
    _configure_mcp_imports(config)
    return {
        "bookstack": importlib.import_module("bookstack_server"),
        "mattermost": importlib.import_module("mattermost_server"),
        "rocketchat": importlib.import_module("rocketchat_server"),
        "mailpit": importlib.import_module("mailpit_server"),
        "gotosocial": importlib.import_module("gotosocial_server"),
        "radicale": importlib.import_module("radicale_server"),
        "google_drive": importlib.import_module("google_drive_server"),
    }


def _task_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    if name.startswith("main") and name[4:].isdigit():
        return (int(name[4:]), name)
    return (10**9, name)


def _select_task_dirs(
    tasks_dir: Path,
    *,
    names: Sequence[str] | None = None,
    task_range: str | None = None,
    limit: int | None = None,
) -> list[Path]:
    if names:
        selected = [tasks_dir / name for name in names]
    else:
        selected = sorted(
            [
                path
                for path in tasks_dir.iterdir()
                if path.is_dir() and (path / "task.json").exists()
            ],
            key=_task_sort_key,
        )
        if task_range:
            start, end = map(int, task_range.split("-", 1))
            selected = selected[start:end]
    if limit is not None:
        selected = selected[:limit]
    return selected


async def _check_bookstack(module: Any, records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    pages = (await module.list_pages()).get("pages", [])
    pages_by_name: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        pages_by_name.setdefault(str(page.get("name", "")), []).append(page)

    for index, record in enumerate(records):
        title = str(record.get("title") or "")
        content = str(record.get("content") or "")
        candidates = pages_by_name.get(title, [])
        if not candidates:
            failures.append(f"bookstack[{index}] page not listed: {title!r}")
            continue
        found = False
        for candidate in candidates:
            page = await module.read_page(candidate["page_id"])
            if _contains(page.get("markdown", ""), _fragment(content)):
                found = True
                break
        if not found:
            failures.append(f"bookstack[{index}] page body not readable: {title!r}")
    return failures


async def _check_mattermost(
    module: Any,
    records: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    messages = (await module.read_messages(max_count=200)).get("messages", [])
    users: list[dict[str, Any]] | None = None
    for index, record in enumerate(records):
        if record.get("type") == "user":
            username = str(record.get("username") or record.get("name") or "")
            email = str(record.get("email") or "")
            if users is None:
                users = (await module.list_users()).get("users", [])
            assert users is not None
            current_users = users
            user_found = any(
                _norm(item.get("username")) == _norm(username)
                or (email and _norm(item.get("email")) == _norm(email))
                for item in current_users
            )
            if not user_found:
                failures.append(f"mattermost[{index}] user not listed: {username!r}")
            continue

        if "message" not in record:
            continue

        text = str(record.get("message") or "")
        match = next((item for item in messages if item.get("text") == text), None)
        if match is None:
            failures.append(
                f"mattermost[{index}] message not listed: {_fragment(text)!r}"
            )
            continue
        if not _sender_matches(match, record):
            failures.append(
                "mattermost"
                f"[{index}] sender mismatch: expected "
                f"{record.get('sender_id')!r}/{record.get('sender_display')!r}, "
                f"got {match.get('sender')!r}/{match.get('sender_display')!r}"
            )
    return failures


async def _check_mailpit(module: Any, records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for index, record in enumerate(records):
        if record.get("type") == "contact":
            query = str(record.get("name") or record.get("email") or "")
            contacts = (await module.list_contacts(name=query)).get("contacts", [])
            contact_found = any(
                _norm(item.get("email")) == _norm(record.get("email"))
                for item in contacts
            )
            if not contact_found:
                failures.append(f"mailpit[{index}] contact not listed: {query!r}")
            continue

        subject = str(record.get("subject") or "")
        body = str(record.get("body") or record.get("content") or "")
        emails = (await module.search_emails(query=subject)).get("emails", [])
        match = next((item for item in emails if item.get("subject") == subject), None)
        if match is None:
            failures.append(f"mailpit[{index}] email not searchable: {subject!r}")
            continue
        detail = await module.read_email(match["email_id"])
        if body and not _contains(detail.get("body", ""), _fragment(body)):
            failures.append(f"mailpit[{index}] email body not readable: {subject!r}")
    return failures


async def _check_rocketchat(
    module: Any,
    records: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    messages_by_channel: dict[str, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        if "message" not in record:
            query = str(record.get("name") or "")
            users = (await module.search_users(query=query)).get("users", [])
            if not any(_norm(item.get("name")) == _norm(query) for item in users):
                failures.append(f"rocketchat[{index}] user not searchable: {query!r}")
            continue

        text = str(record.get("message") or "")
        channel = _rocketchat_channel_name(record.get("channel"))
        if channel not in messages_by_channel:
            messages_by_channel[channel] = (
                await module.read_channel_history(channel=channel, count=100)
            ).get("messages", [])
        messages = messages_by_channel[channel]
        match = next((item for item in messages if item.get("text") == text), None)
        if match is None:
            failures.append(
                f"rocketchat[{index}] message not in {channel!r}: {_fragment(text)!r}"
            )
            continue
        source_channel = str(record.get("channel") or "").strip()
        enforce_sender = not source_channel or source_channel.startswith("@")
        if enforce_sender and not _sender_matches(match, record):
            failures.append(
                "rocketchat"
                f"[{index}] sender mismatch: expected "
                f"{record.get('sender_id')!r}/{record.get('sender_display')!r}, "
                f"got {match.get('sender')!r}/{match.get('sender_display')!r}"
            )
    return failures


async def _check_radicale(module: Any, records: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    listed = (await module.list_events()).get("events", [])
    event_ids = {str(item.get("event_id") or "") for item in listed}
    for index, record in enumerate(records):
        event_id = str(record.get("event_id") or "")
        if event_id not in event_ids:
            failures.append(f"radicale[{index}] event not listed: {event_id!r}")
            continue
        detail = await module.read_event(event_id=event_id)
        if _norm(detail.get("name")) != _norm(record.get("event_name")):
            failures.append(f"radicale[{index}] event name mismatch: {event_id!r}")
    return failures


async def _check_gotosocial(
    module: Any,
    records: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for index, record in enumerate(records):
        content = str(record.get("content") or "")
        if not content:
            continue
        posts = (await module.search_posts(query=_fragment(content, limit=40))).get(
            "posts",
            [],
        )
        if not any(
            _contains(item.get("content"), _fragment(content)) for item in posts
        ):
            failures.append(f"gotosocial[{index}] post not searchable")
    return failures


async def _check_google_drive(
    module: Any,
    records: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    for index, record in enumerate(records):
        file_id = str(record.get("file_id") or "")
        name = str(record.get("name") or "")
        search = (await module.search_files(query=name)).get("files", [])
        if not any(item.get("file_id") == file_id for item in search):
            failures.append(f"google_drive[{index}] file not searchable: {name!r}")
            continue
        metadata = await module.get_file_metadata(file_id=file_id)
        if _norm(metadata.get("name")) != _norm(name):
            failures.append(f"google_drive[{index}] metadata name mismatch: {file_id}")
            continue
        if record.get("available"):
            image_result = await module.get_file_image(file_id=file_id)
            if not any(isinstance(item, ImageContent) for item in image_result):
                failures.append(f"google_drive[{index}] image content missing")
    return failures


async def _check_service(
    service: str,
    module: Any,
    records: list[dict[str, Any]],
) -> list[str]:
    if service == "bookstack":
        return await _check_bookstack(module, records)
    if service == "mattermost":
        return await _check_mattermost(module, records)
    if service == "mailpit":
        return await _check_mailpit(module, records)
    if service == "rocketchat":
        return await _check_rocketchat(module, records)
    if service == "radicale":
        return await _check_radicale(module, records)
    if service == "gotosocial":
        return await _check_gotosocial(module, records)
    if service == "google_drive":
        return await _check_google_drive(module, records)
    return [f"{service}: no live readiness checker"]


def _seed_inputs(seed_dir: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    inputs: list[tuple[str, list[dict[str, Any]]]] = []
    for seed_path in sorted(seed_dir.iterdir(), key=lambda path: path.name):
        if seed_path.is_file() and seed_path.suffix == ".json":
            inputs.append((seed_path.stem, json.loads(seed_path.read_text())))
        elif seed_path.is_dir() and (seed_path / "drive_index.json").is_file():
            data = json.loads((seed_path / "drive_index.json").read_text())
            files = data.get("files", [])
            inputs.append(
                (
                    seed_path.name,
                    [item for item in files if isinstance(item, dict)],
                )
            )
    return inputs


async def check_task(
    task_dir: Path,
    seeder: Seeder,
    modules: dict[str, Any],
) -> dict[str, Any]:
    task = json.loads((task_dir / "task.json").read_text())
    failures: list[str] = []
    service_record_counts: dict[str, int] = {}

    seed_inputs = _seed_inputs(task_dir / "seed_data")
    for service, records in seed_inputs:
        service_record_counts[service] = len(records)
        if not records:
            failures.append(f"{service}: empty seed file")

    if failures:
        return {
            "task": task_dir.name,
            "ok": False,
            "records": service_record_counts,
            "failures": failures,
        }

    dependencies = list(task.get("dependencies", []))
    try:
        await seeder.reset_services(dependencies)
        await seeder.seed_from_dir(task_dir / "seed_data")
    except Exception as exc:
        return {
            "task": task_dir.name,
            "ok": False,
            "records": service_record_counts,
            "failures": [f"seed failed: {type(exc).__name__}: {exc}"],
        }

    for service, records in seed_inputs:
        module = modules.get(service)
        if module is None:
            failures.append(f"{service}: no MCP module loaded")
            continue
        try:
            failures.extend(await _check_service(service, module, records))
        except Exception as exc:
            failures.append(f"{service}: check failed: {type(exc).__name__}: {exc}")

    return {
        "task": task_dir.name,
        "ok": not failures,
        "records": service_record_counts,
        "failures": failures,
    }


async def check_tasks(
    task_dirs: Sequence[Path],
    *,
    report_every: int = 25,
) -> dict[str, Any]:
    _quiet_noisy_loggers()
    config = _live_config()
    modules = _mcp_modules(config)
    _quiet_noisy_loggers()
    seeder = Seeder(
        bookstack_url=config.bookstack_url,
        bookstack_token_id=config.bookstack_token_id,
        bookstack_token_secret=config.bookstack_token_secret,
        mattermost_url=config.mattermost_url,
        mattermost_user=config.mattermost_user,
        mattermost_password=config.mattermost_password,
        rocketchat_url=config.rocketchat_url,
        rocketchat_user=config.rocketchat_user,
        rocketchat_password=config.rocketchat_password,
        mailpit_api_url=config.mailpit_api_url,
        mailpit_smtp_host=config.mailpit_smtp_host,
        mailpit_smtp_port=config.mailpit_smtp_port,
        gotosocial_url=config.gotosocial_url,
        gotosocial_token=config.gotosocial_token,
        radicale_url=config.radicale_url,
        radicale_user=config.radicale_user,
        radicale_password=config.radicale_password,
        google_drive_artifact_root=config.google_drive_artifact_root,
    )

    results: list[dict[str, Any]] = []
    for index, task_dir in enumerate(task_dirs, start=1):
        result = await check_task(task_dir, seeder, modules)
        results.append(result)
        if index == 1 or index % report_every == 0 or not result["ok"]:
            status = "ok" if result["ok"] else "FAIL"
            print(f"[{index}/{len(task_dirs)}] {task_dir.name}: {status}", flush=True)
            for failure in result["failures"][:5]:
                print(f"  - {failure}", flush=True)

    failed = [result for result in results if not result["ok"]]
    return {
        "total": len(results),
        "ok": len(results) - len(failed),
        "failed": len(failed),
        "results": results,
        "service_urls": {
            "bookstack": config.bookstack_url,
            "mattermost": config.mattermost_url,
            "rocketchat": config.rocketchat_url,
            "mailpit": config.mailpit_api_url,
            "gotosocial": config.gotosocial_url,
            "radicale": config.radicale_url,
        },
    }


def main() -> None:
    _quiet_noisy_loggers()
    parser = argparse.ArgumentParser(
        description="Check generated tasks against live service read paths."
    )
    parser.add_argument("--tasks-dir", type=Path, default=Path("agentprivarena/tasks"))
    parser.add_argument("--names", default="", help="Comma-separated task names.")
    parser.add_argument("--range", default=None, help="Index range, e.g. 0-25.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path(".agent_tmp/live_readiness_report.json"),
    )
    args = parser.parse_args()

    names = [name.strip() for name in args.names.split(",") if name.strip()]
    task_dirs = _select_task_dirs(
        args.tasks_dir,
        names=names or None,
        task_range=args.range,
        limit=args.limit,
    )
    report = asyncio.run(check_tasks(task_dirs))

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2))
    print(
        f"Live readiness: {report['ok']}/{report['total']} ok, "
        f"{report['failed']} failed. Report: {args.report_out}"
    )
    raise SystemExit(0 if report["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
