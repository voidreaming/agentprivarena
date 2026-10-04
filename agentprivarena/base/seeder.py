# pyright: reportMissingImports=false, reportCallIssue=false, reportGeneralTypeIssues=false
#
# The Radicale handler uses the ``caldav`` library, which has no type
# stubs. Pyright resolves every caldav symbol as ``object``, which then
# trips ``reportCallIssue`` (``DAVClient(...)`` is "not callable") and
# ``reportGeneralTypeIssues`` (``caldav.Calendar`` annotations are
# "not a class"). The library is correct at runtime; suppressing these
# three rules at the file level is the project convention (mirrors
# ``mcp_servers/radicale_server.py``). ``reportMissingImports`` is
# kept for fresh checkouts where ``uv sync --dev`` hasn't run yet.
"""Data seeder — injects seed_data/ files into real services.

Reads per-service JSON files from a task's seed_data/ directory and
creates the corresponding records in each service via REST APIs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import smtplib
import time
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.utils import formataddr
from pathlib import Path
from typing import Any

import httpx


logger = logging.getLogger("seeder")
_ROCKETCHAT_SEED_PASSWORD = "password123"
_MATTERMOST_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}$")


def _ical_dt(s: str) -> str:
    """Strip ISO timestamp punctuation for an iCalendar DTSTART/DTEND."""
    return s.replace("-", "").replace(":", "").replace("+", "").split(".")[0]


def _slug_identifier(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    return "_".join(tokens) or "agentprivarena_user"


def _header_value(value: Any) -> str:
    return str(value or "").replace("\n", " ").replace("\r", " ").strip()


def _rate_limit_delay(resp: httpx.Response, attempt: int) -> float:
    retry_after = resp.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return float(retry_after)
    if reset_at := resp.headers.get("X-RateLimit-Reset"):
        try:
            return max((float(reset_at) / 1000) - time.time() + 0.5, 1.0)
        except ValueError:
            pass
    match = re.search(r"wait\s+(\d+)\s+seconds?", resp.text)
    if match:
        return float(match.group(1))
    return float(2 * (attempt + 1))


def _contact_registry_body(record: dict[str, Any]) -> str:
    lines = [
        "AGENTPRIVARENA_CONTACT",
        f"Name: {record.get('name', '')}",
        f"Email: {record.get('email', '')}",
        f"Remark: {record.get('remark', '')}",
        f"Info: {record.get('info', '')}",
    ]
    mailing_address = record.get("mailing_address")
    if mailing_address:
        lines.append(f"Mailing Address: {mailing_address}")
    return "\n".join(lines)


def _rocketchat_registry_user(record: dict[str, Any]) -> dict[str, str] | None:
    if record.get("type") == "channel":
        return None
    if record.get("type") == "user_profile":
        profile = record.get("profile") or {}
        first_name = str(profile.get("first_name") or "").strip()
        last_name = str(profile.get("last_name") or "").strip()
        name = f"{first_name} {last_name}".strip()
        email = str(profile.get("email") or "").strip()
        username_source = email.split("@", 1)[0] if email else name
        username = _slug_identifier(username_source)
        return {
            "email": email or f"{username}@agentprivarena.local",
            "name": name or username,
            "username": username,
            "statusText": str(profile.get("title") or "").strip(),
        }

    name = str(record.get("name") or "").strip()
    if not name:
        return None
    email = str(record.get("email") or "").strip()
    username_source = str(record.get("username") or "").strip()
    if not username_source and email:
        username_source = email.split("@", 1)[0]
    username = _slug_identifier(username_source or name)
    return {
        "email": email or f"{username}@agentprivarena.local",
        "name": name,
        "username": username,
        "statusText": str(record.get("status") or "").strip(),
    }


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


def _rocketchat_registry_channel(record: dict[str, Any]) -> str:
    if record.get("type") != "channel":
        return ""
    return _rocketchat_channel_name(record.get("channel"))


def _rocketchat_message_payload(
    record: dict[str, Any],
    *,
    room_id: str,
    channel_name: str,
) -> dict[str, str]:
    payload = {
        "rid": room_id,
        "msg": str(record.get("message", "")),
    }
    if channel_name:
        alias = str(record.get("sender_display") or record.get("sender_id") or "")
        if alias:
            payload["alias"] = alias
    return payload


def _mattermost_sender_username(sender_id: str) -> str:
    sender = sender_id.strip()
    if not sender or sender.casefold().startswith("group_"):
        return ""
    if (
        "@" not in sender
        and sender == sender.casefold()
        and _MATTERMOST_USERNAME_RE.fullmatch(sender)
    ):
        return sender
    return _slug_identifier(sender)


def _mattermost_registry_user(record: dict[str, Any]) -> dict[str, str] | None:
    if record.get("type") != "user":
        return None
    username_source = (
        str(record.get("username") or "")
        or str(record.get("email") or "").split("@", 1)[0]
        or str(record.get("name") or "")
    )
    username = _mattermost_sender_username(username_source)
    if not username:
        return None
    return {
        "email": str(record.get("email") or f"{username}@agentprivarena.local"),
        "name": str(record.get("name") or username),
        "username": username,
    }


@dataclass
class CleanupHandle:
    """Tracks seeded records for cleanup after a task run."""

    service: str
    record_ids: list[str] = field(default_factory=list)


@dataclass
class SeedResult:
    """Result of seeding a task's data."""

    handles: list[CleanupHandle] = field(default_factory=list)
    id_map: dict[str, str] = field(default_factory=dict)


class ServiceHandler:
    """Base class for per-service seeding logic."""

    service_name: str = ""

    async def reset(self) -> None:
        """Remove prior task content from the backing service."""
        pass

    async def seed(self, records: list[dict]) -> CleanupHandle:
        raise NotImplementedError

    async def seed_from_path(self, seed_path: Path) -> CleanupHandle:
        records = json.loads(seed_path.read_text())
        return await self.seed(records)

    async def cleanup(self, handle: CleanupHandle) -> None:
        pass


class BookStackHandler(ServiceHandler):
    """Seed BookStack pages via REST API."""

    service_name = "bookstack"

    def __init__(self, url: str, token_id: str = "", token_secret: str = ""):
        self.url = url
        self.token_id = token_id
        self.token_secret = token_secret
        self._book_id: int | None = None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": (f"Token {self.token_id}:{self.token_secret}"),
            "Content-Type": "application/json",
        }

    async def _ensure_book(self) -> int:
        """Get or create the workspace book."""
        if self._book_id is not None:
            return self._book_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Check if workspace book exists
            resp = await client.get(
                f"{self.url}/api/books",
                params={"count": 100},
                headers=self._headers(),
            )
            if resp.status_code == 200:
                for book in resp.json().get("data", []):
                    if book.get("name") == "Workspace":
                        book_id = int(book["id"])
                        self._book_id = book_id
                        return book_id

            # Create it
            resp = await client.post(
                f"{self.url}/api/books",
                json={
                    "name": "Workspace",
                    "description": "AgentPrivArena workspace",
                },
                headers=self._headers(),
            )
            resp.raise_for_status()
            book_id = int(resp.json()["id"])
            self._book_id = book_id
            logger.info(f"Created workspace book id={book_id}")
            return book_id

    async def seed(self, records: list[dict]) -> CleanupHandle:
        handle = CleanupHandle(service="bookstack")
        book_id = await self._ensure_book()

        async with httpx.AsyncClient(timeout=30.0) as client:
            for record in records:
                title = record.get("title", "Untitled")
                content = record.get("content", "")
                orig_id = record.get("id", "")
                tags = [{"name": t} for t in record.get("tags", [])]
                if orig_id:
                    tags.append({"name": "agentprivarena_id", "value": orig_id})

                resp = await client.post(
                    f"{self.url}/api/pages",
                    json={
                        "book_id": book_id,
                        "name": title,
                        "markdown": content,
                        "tags": tags,
                    },
                    headers=self._headers(),
                )
                if resp.status_code in (200, 201):
                    page_id = resp.json()["id"]
                    handle.record_ids.append(str(page_id))
                    logger.info(f"Seeded BookStack page: {title} (id={page_id})")
                else:
                    logger.warning(
                        f"Failed to seed page '{title}': "
                        f"{resp.status_code} {resp.text[:200]}"
                    )

        return handle

    async def reset(self) -> None:
        """Remove all pages from the dedicated workspace book."""
        book_id = await self._ensure_book()
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    f"{self.url}/api/pages",
                    params={"count": 500},
                    headers=self._headers(),
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(f"Failed to list BookStack pages for reset: {exc}")
                return

            page_ids: list[int] = []
            for page in resp.json().get("data", []):
                page_book_id = page.get("book_id")
                if page_book_id is not None and int(page_book_id) != book_id:
                    continue
                page_id = page.get("id")
                if page_id is not None:
                    page_ids.append(int(page_id))

            await self._delete_pages(client, page_ids)
            await self._purge_recycle_bin(client, set(page_ids))

    async def cleanup(self, handle: CleanupHandle) -> None:
        """Permanently delete pages we created.

        BookStack's ``DELETE /api/pages/{id}`` is a *soft* delete: it
        moves the page to the recycle bin and returns 200. Without a
        follow-up step the recycle bin grows unbounded across runs.
        We:

        1. Soft-delete each tracked page (and actually check the
           response status — httpx doesn't raise on 4xx by default).
        2. Query the recycle bin once and find the entries with
           ``deletable.id`` matching one of our tracked page ids.
        3. ``DELETE /api/recycle-bin/{entry_id}`` for each match to
           permanently remove them. Targeted — we never touch any
           recycle bin contents we didn't just create.
        """
        if not handle.record_ids:
            return
        target_ids = {int(pid) for pid in handle.record_ids}
        async with httpx.AsyncClient(timeout=30.0) as client:
            await self._delete_pages(client, list(target_ids))
            await self._purge_recycle_bin(client, target_ids)

    async def _delete_pages(
        self,
        client: httpx.AsyncClient,
        page_ids: list[int],
    ) -> None:
        for page_id in page_ids:
            try:
                resp = await client.delete(
                    f"{self.url}/api/pages/{page_id}",
                    headers=self._headers(),
                )
            except Exception as e:
                logger.warning(f"Failed to soft-delete page {page_id}: {e}")
                continue
            if resp.status_code not in (200, 204, 404):
                logger.warning(
                    f"Soft-delete failed for page {page_id}: "
                    f"HTTP {resp.status_code} {resp.text[:200]}"
                )

    async def _purge_recycle_bin(
        self,
        client: httpx.AsyncClient,
        target_ids: set[int],
    ) -> None:
        if not target_ids:
            return
        try:
            resp = await client.get(
                f"{self.url}/api/recycle-bin",
                params={"count": 500},
                headers=self._headers(),
            )
            resp.raise_for_status()
            entries = resp.json().get("data", [])
        except httpx.HTTPError as exc:
            logger.warning(f"Failed to query BookStack recycle bin: {exc}")
            return

        for entry in entries:
            deletable = entry.get("deletable", {})
            if deletable.get("type") != "page" or deletable.get("id") not in target_ids:
                continue
            entry_id = entry.get("id")
            try:
                del_resp = await client.delete(
                    f"{self.url}/api/recycle-bin/{entry_id}",
                    headers=self._headers(),
                )
                del_resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(
                    f"Failed to permanently delete recycle bin "
                    f"entry {entry_id} (page {deletable.get('id')}): {exc}"
                )


class MattermostHandler(ServiceHandler):
    """Seed Mattermost DMs via REST API v4."""

    service_name = "mattermost"

    def __init__(self, url: str, user: str, password: str):
        self.url = url
        self.user = user
        self.password = password
        self._token: str = ""
        self._user_id: str = ""

    async def _login(self) -> None:
        if self._token:
            return
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self.url}/api/v4/users/login",
                json={
                    "login_id": self.user,
                    "password": self.password,
                },
            )
            resp.raise_for_status()
            self._token = resp.headers["Token"]
            self._user_id = resp.json()["id"]

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"}

    async def reset(self) -> None:
        """Delete visible messages and seed users from prior runs."""
        await self._login()
        async with httpx.AsyncClient(timeout=30.0) as client:
            for channel in await self._channels_for_cleanup(client):
                channel_id = channel.get("id")
                if channel_id:
                    await self._delete_channel_posts(client, channel_id)
            await self._delete_seeded_users(client)

    async def seed(self, records: list[dict]) -> CleanupHandle:
        handle = CleanupHandle(service="mattermost")
        await self._login()

        async with httpx.AsyncClient(timeout=30.0) as client:
            # Create users mentioned in records. The original PrivacyLens
            # sender_id may be an email/display-ish identifier; Mattermost
            # usernames need a safe machine identifier that tools can reuse.
            sender_usernames: dict[str, str] = {}
            seed_users: dict[str, dict[str, str]] = {}
            for record in records:
                sender_id = str(record.get("sender_id") or "").strip()
                if sender_id:
                    username = _mattermost_sender_username(sender_id)
                    sender_usernames[sender_id] = username
                    if username:
                        seed_users[username] = {
                            "email": f"{username}@agentprivarena.local",
                            "name": str(record.get("sender_display") or username),
                            "username": username,
                        }

                registry_user = _mattermost_registry_user(record)
                if registry_user is not None:
                    seed_users[registry_user["username"]] = registry_user

            for user in seed_users.values():
                await self._create_seed_user(client, user)

            # Post messages
            for record in records:
                message = record.get("message", "")
                if not message:
                    continue
                try:
                    # Get team for posting
                    resp = await client.get(
                        f"{self.url}/api/v4/users/me/teams",
                        headers=self._headers(),
                    )
                    teams = resp.json()
                    if not teams:
                        continue
                    # Post to town-square as seed data
                    resp = await client.get(
                        f"{self.url}/api/v4/teams/{teams[0]['id']}"
                        "/channels/name/town-square",
                        headers=self._headers(),
                    )
                    if resp.status_code != 200:
                        continue
                    channel_id = resp.json()["id"]
                    await self._ensure_channel_member(client, channel_id)

                    resp = await client.post(
                        f"{self.url}/api/v4/posts",
                        # Store the machine-usable username separately from
                        # the display name so read tools do not induce sends
                        # to invalid targets such as "Mark Therapy Center".
                        json={
                            "channel_id": channel_id,
                            "message": message,
                            "props": {
                                "agentprivarena_sender": sender_usernames.get(
                                    str(record.get("sender_id") or "").strip(),
                                    "",
                                )
                                or record.get("sender_id", ""),
                                "agentprivarena_sender_display": record.get(
                                    "sender_display", ""
                                ),
                                "agentprivarena_sender_unknown": bool(
                                    record.get("sender_unknown", False)
                                ),
                                "agentprivarena_time": record.get("seed_time_ms", ""),
                            },
                        },
                        headers=self._headers(),
                    )
                    if resp.status_code in (200, 201):
                        post_id = resp.json()["id"]
                        handle.record_ids.append(post_id)
                except Exception as e:
                    logger.warning(f"Failed to seed Mattermost message: {e}")

        return handle

    async def _channels_for_cleanup(
        self, client: httpx.AsyncClient
    ) -> list[dict[str, Any]]:
        resp = await client.get(
            f"{self.url}/api/v4/users/me/channels",
            headers=self._headers(),
        )
        resp.raise_for_status()
        by_id = {
            channel["id"]: channel
            for channel in resp.json()
            if isinstance(channel, dict) and channel.get("id")
        }

        teams_resp = await client.get(
            f"{self.url}/api/v4/users/me/teams",
            headers=self._headers(),
        )
        teams_resp.raise_for_status()
        for team in teams_resp.json():
            if not isinstance(team, dict) or not team.get("id"):
                continue
            channel_resp = await client.get(
                f"{self.url}/api/v4/teams/{team['id']}/channels/name/town-square",
                headers=self._headers(),
            )
            if channel_resp.status_code != 200:
                continue
            channel = channel_resp.json()
            channel_id = channel.get("id")
            if not channel_id:
                continue
            await self._ensure_channel_member(client, channel_id)
            by_id[channel_id] = channel

        return list(by_id.values())

    async def _ensure_channel_member(
        self,
        client: httpx.AsyncClient,
        channel_id: str,
    ) -> None:
        try:
            resp = await client.post(
                f"{self.url}/api/v4/channels/{channel_id}/members",
                json={"user_id": self._user_id},
                headers=self._headers(),
            )
        except Exception as exc:
            logger.warning(f"Failed to join Mattermost channel {channel_id}: {exc}")
            return
        if resp.status_code not in (200, 201, 400, 409):
            logger.warning(
                f"Failed to join Mattermost channel {channel_id}: "
                f"HTTP {resp.status_code} {resp.text[:200]}"
            )

    async def _create_seed_user(
        self,
        client: httpx.AsyncClient,
        user: dict[str, str],
    ) -> None:
        username = user["username"]
        if username == self.user:
            return
        name_parts = user["name"].split(maxsplit=1)
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        try:
            resp = await client.post(
                f"{self.url}/api/v4/users",
                json={
                    "email": user["email"],
                    "username": username,
                    "first_name": first_name,
                    "last_name": last_name,
                    "password": "Password123!",
                },
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            logger.warning(f"Failed to create Mattermost seed user '{username}': {exc}")
            return
        if resp.status_code in (200, 201):
            return
        if resp.status_code in (400, 409) and "already" in resp.text.lower():
            return
        logger.warning(
            f"Failed to create Mattermost seed user '{username}': "
            f"HTTP {resp.status_code} {resp.text[:200]}"
        )

    async def cleanup(self, handle: CleanupHandle) -> None:
        await self._login()
        async with httpx.AsyncClient(timeout=30.0) as client:
            await self._delete_posts(client, handle.record_ids)

    async def _delete_seeded_users(self, client: httpx.AsyncClient) -> None:
        current_user = self.user.casefold()
        for user in await self._list_users(client):
            if user.get("is_bot"):
                continue
            username = str(user.get("username") or "")
            if not username or username.casefold() == current_user:
                continue
            user_id = user.get("id")
            if not user_id:
                continue
            try:
                resp = await client.delete(
                    f"{self.url}/api/v4/users/{user_id}",
                    params={"permanent": "true"},
                    headers=self._headers(),
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    f"Failed to delete Mattermost seed user '{username}': {exc}"
                )
                continue
            if resp.status_code in (200, 204, 404):
                continue
            if resp.status_code == 401 and "EnableAPIUserDeletion" in resp.text:
                logger.warning(
                    "Mattermost permanent user deletion is disabled; "
                    "seed users will accumulate until "
                    "MM_SERVICESETTINGS_ENABLEAPIUSERDELETION=true is active."
                )
                return
            logger.warning(
                f"Failed to delete Mattermost seed user '{username}': "
                f"HTTP {resp.status_code} {resp.text[:200]}"
            )

    async def _list_users(self, client: httpx.AsyncClient) -> list[dict[str, Any]]:
        users: list[dict[str, Any]] = []
        page = 0
        while True:
            resp = await client.get(
                f"{self.url}/api/v4/users",
                params={"page": page, "per_page": 200},
                headers=self._headers(),
            )
            if resp.status_code != 200:
                logger.warning(
                    f"Failed to list Mattermost users for reset: "
                    f"HTTP {resp.status_code} {resp.text[:200]}"
                )
                return users
            batch = resp.json()
            users.extend(user for user in batch if isinstance(user, dict))
            if len(batch) < 200:
                return users
            page += 1

    async def _delete_channel_posts(
        self,
        client: httpx.AsyncClient,
        channel_id: str,
    ) -> None:
        for _ in range(20):
            resp = await client.get(
                f"{self.url}/api/v4/channels/{channel_id}/posts",
                params={"per_page": 200},
                headers=self._headers(),
            )
            resp.raise_for_status()
            post_ids = resp.json().get("order", [])
            if not post_ids:
                return
            deleted = await self._delete_posts(client, post_ids)
            if deleted == 0 or len(post_ids) < 200:
                return

    async def _delete_posts(
        self,
        client: httpx.AsyncClient,
        post_ids: list[str],
    ) -> int:
        deleted = 0
        for post_id in post_ids:
            try:
                resp = await client.delete(
                    f"{self.url}/api/v4/posts/{post_id}",
                    headers=self._headers(),
                )
                if resp.status_code in (200, 204):
                    deleted += 1
                elif resp.status_code != 404:
                    logger.warning(
                        f"Failed to delete Mattermost post {post_id}: "
                        f"HTTP {resp.status_code} {resp.text[:200]}"
                    )
            except httpx.HTTPError as e:
                logger.warning(f"Failed to delete Mattermost post {post_id}: {e}")
        return deleted


class RocketChatHandler(ServiceHandler):
    """Seed RocketChat channel messages for Slack."""

    service_name = "rocketchat"

    def __init__(self, url: str, user: str, password: str):
        self.url = url
        self.user = user
        self.password = password
        self._auth: dict[str, str] = {}

    async def _login(self) -> dict[str, str]:
        if self._auth:
            return self._auth
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self.url}/api/v1/login",
                json={"user": self.user, "password": self.password},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
        self._auth = {
            "X-Auth-Token": data["authToken"],
            "X-User-Id": data["userId"],
        }
        return self._auth

    async def reset(self) -> None:
        """Delete visible channel messages and seeded users from prior runs."""
        auth = await self._login()
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{self.url}/api/v1/channels.list",
                params={"count": 100},
                headers=auth,
            )
            resp.raise_for_status()
            for channel in resp.json().get("channels", []):
                room_id = channel.get("_id")
                channel_name = str(channel.get("name") or "")
                if not room_id:
                    continue
                if channel_name == "general":
                    await self._delete_room_messages(client, auth, room_id)
                else:
                    await self._delete_seeded_channel(client, auth, room_id)
            await self._delete_seeded_users(client, auth)

    async def seed(self, records: list[dict]) -> CleanupHandle:
        handle = CleanupHandle(service="rocketchat")
        auth = await self._login()

        async with httpx.AsyncClient(timeout=30.0) as client:
            seed_auth_cache: dict[str, dict[str, str]] = {}
            sender_users: dict[str, dict[str, str]] = {}
            for record in records:
                sender = str(record.get("sender_id") or "").strip()
                if not sender or sender.startswith("group_"):
                    continue
                sender_users[sender] = {
                    "email": f"{sender}@agentprivarena.local",
                    "name": str(record.get("sender_display") or sender),
                    "username": sender,
                    "statusText": "",
                }

            for user in sender_users.values():
                await self._create_seed_user(client, auth, user)

            registry_users = [
                user
                for record in records
                if "message" not in record
                for user in [_rocketchat_registry_user(record)]
                if user is not None
            ]
            for user in registry_users:
                await self._create_seed_user(client, auth, user)

            channel_names = {
                name
                for record in records
                for name in (
                    _rocketchat_channel_name(record.get("channel")),
                    _rocketchat_registry_channel(record),
                )
                if name and name != "general"
            }
            channel_ids = {
                name: await self._ensure_channel(client, auth, name)
                for name in sorted(channel_names)
            }

            # Post messages
            for record in records:
                message = record.get("message", "")
                sender = record.get("sender_id", "")
                if not message:
                    continue
                message_headers = auth
                channel_name = _rocketchat_channel_name(record.get("channel"))
                room_id = channel_ids.get(channel_name, "GENERAL")
                post_as_seed_user = (
                    not channel_name and sender and not str(sender).startswith("group_")
                )
                if post_as_seed_user:
                    try:
                        sender_key = str(sender)
                        if sender_key not in seed_auth_cache:
                            seed_auth_cache[sender_key] = await self._login_seed_user(
                                client,
                                sender_key,
                            )
                        message_headers = seed_auth_cache[sender_key]
                    except httpx.HTTPError as exc:
                        logger.warning(
                            f"Failed to log in RocketChat seed user '{sender}': {exc}"
                        )
                        continue
                payload = _rocketchat_message_payload(
                    record,
                    room_id=room_id,
                    channel_name=channel_name,
                )
                try:
                    result = await self._post_with_rate_limit_retry(
                        client,
                        f"{self.url}/api/v1/chat.sendMessage",
                        json={"message": payload},
                        headers=message_headers,
                    )
                    if (
                        result.status_code == 400
                        and "not enough permission" in result.text.lower()
                        and "alias" in payload
                    ):
                        payload = {k: v for k, v in payload.items() if k != "alias"}
                        result = await self._post_with_rate_limit_retry(
                            client,
                            f"{self.url}/api/v1/chat.sendMessage",
                            json={"message": payload},
                            headers=message_headers,
                        )
                    if result.status_code not in (200, 201):
                        logger.warning(
                            f"Failed to seed RC message: HTTP "
                            f"{result.status_code} {result.text[:200]}"
                        )
                        continue
                    msg_id = result.json().get("message", {}).get("_id", "")
                    if msg_id and channel_name in ("", "general"):
                        handle.record_ids.append(msg_id)
                except httpx.HTTPError as e:
                    logger.warning(f"Failed to seed RC message: {e}")

        return handle

    async def _create_seed_user(
        self,
        client: httpx.AsyncClient,
        auth: dict[str, str],
        user: dict[str, str],
    ) -> None:
        username = user["username"]
        if username == self.user:
            return
        resp = await client.post(
            f"{self.url}/api/v1/users.create",
            json={
                "email": user["email"],
                "name": user["name"],
                "username": username,
                "password": _ROCKETCHAT_SEED_PASSWORD,
                "statusText": user["statusText"],
            },
            headers=auth,
        )
        if resp.status_code in (200, 201):
            return
        if resp.status_code == 400 and "already" in resp.text.lower():
            return
        logger.warning(
            f"Failed to create RocketChat seed user '{username}': "
            f"HTTP {resp.status_code} {resp.text[:200]}"
        )

    async def _ensure_channel(
        self,
        client: httpx.AsyncClient,
        auth: dict[str, str],
        name: str,
    ) -> str:
        if name == "general":
            return "GENERAL"

        info_resp = await client.get(
            f"{self.url}/api/v1/channels.info",
            params={"roomName": name},
            headers=auth,
        )
        if info_resp.status_code == 200:
            room_id = (info_resp.json().get("channel") or {}).get("_id", "")
            if room_id:
                return room_id

        resp = await client.post(
            f"{self.url}/api/v1/channels.create",
            json={"name": name},
            headers=auth,
        )
        if resp.status_code in (200, 201):
            room_id = (resp.json().get("channel") or {}).get("_id", "")
            if room_id:
                return room_id

        logger.warning(
            f"Failed to create RocketChat channel '{name}': "
            f"HTTP {resp.status_code} {resp.text[:200]}"
        )
        return "GENERAL"

    async def _login_seed_user(
        self,
        client: httpx.AsyncClient,
        username: str,
    ) -> dict[str, str]:
        resp: httpx.Response | None = None
        for attempt in range(5):
            resp = await client.post(
                f"{self.url}/api/v1/login",
                json={"user": username, "password": _ROCKETCHAT_SEED_PASSWORD},
            )
            if resp.status_code != 429:
                break
            await asyncio.sleep(_rate_limit_delay(resp, attempt))
        assert resp is not None
        resp.raise_for_status()
        data = resp.json()["data"]
        return {
            "X-Auth-Token": data["authToken"],
            "X-User-Id": data["userId"],
        }

    async def _post_with_rate_limit_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        resp: httpx.Response | None = None
        for attempt in range(5):
            resp = await client.post(url, json=json, headers=headers)
            if resp.status_code != 429:
                break
            await asyncio.sleep(_rate_limit_delay(resp, attempt))
        assert resp is not None
        return resp

    async def _delete_seeded_users(
        self,
        client: httpx.AsyncClient,
        auth: dict[str, str],
    ) -> None:
        resp = await client.get(
            f"{self.url}/api/v1/users.list",
            params={"count": 200},
            headers=auth,
        )
        resp.raise_for_status()
        current_user = self.user.casefold()
        for user in resp.json().get("users", []):
            if user.get("type") == "bot":
                continue
            username = str(user.get("username") or "")
            if not username or username.casefold() == current_user:
                continue
            user_id = user.get("_id")
            if not user_id:
                continue
            delete_resp = await client.post(
                f"{self.url}/api/v1/users.delete",
                json={"userId": user_id},
                headers=auth,
            )
            if delete_resp.status_code not in (200, 404):
                logger.warning(
                    f"Failed to delete RocketChat seed user '{username}': "
                    f"HTTP {delete_resp.status_code} {delete_resp.text[:200]}"
                )

    async def _delete_seeded_channel(
        self,
        client: httpx.AsyncClient,
        auth: dict[str, str],
        room_id: str,
    ) -> None:
        resp = await client.post(
            f"{self.url}/api/v1/channels.delete",
            json={"roomId": room_id},
            headers=auth,
        )
        if resp.status_code not in (200, 404):
            logger.warning(
                f"Failed to delete RocketChat channel {room_id}: "
                f"HTTP {resp.status_code} {resp.text[:200]}"
            )

    async def cleanup(self, handle: CleanupHandle) -> None:
        if not handle.record_ids:
            auth = await self._login()
            async with httpx.AsyncClient(timeout=30.0) as client:
                await self._delete_all_seeded_channels(client, auth)
            return
        auth = await self._login()
        async with httpx.AsyncClient(timeout=30.0) as client:
            await self._delete_rocketchat_messages(
                client,
                auth,
                "GENERAL",
                handle.record_ids,
            )
            await self._delete_all_seeded_channels(client, auth)

    async def _delete_all_seeded_channels(
        self,
        client: httpx.AsyncClient,
        auth: dict[str, str],
    ) -> None:
        resp = await client.get(
            f"{self.url}/api/v1/channels.list",
            params={"count": 100},
            headers=auth,
        )
        resp.raise_for_status()
        for channel in resp.json().get("channels", []):
            if channel.get("name") == "general":
                continue
            room_id = channel.get("_id")
            if room_id:
                await self._delete_seeded_channel(client, auth, room_id)

    async def _delete_room_messages(
        self,
        client: httpx.AsyncClient,
        auth: dict[str, str],
        room_id: str,
    ) -> None:
        for _ in range(20):
            resp = await client.get(
                f"{self.url}/api/v1/channels.history",
                params={"roomId": room_id, "count": 100},
                headers=auth,
            )
            resp.raise_for_status()
            message_ids = [
                msg["_id"] for msg in resp.json().get("messages", []) if msg.get("_id")
            ]
            if not message_ids:
                return
            deleted = await self._delete_rocketchat_messages(
                client,
                auth,
                room_id,
                message_ids,
            )
            if deleted == 0 or len(message_ids) < 100:
                return

    async def _delete_rocketchat_messages(
        self,
        client: httpx.AsyncClient,
        auth: dict[str, str],
        room_id: str,
        message_ids: list[str],
    ) -> int:
        deleted = 0
        for msg_id in message_ids:
            try:
                resp = await client.post(
                    f"{self.url}/api/v1/chat.delete",
                    json={"roomId": room_id, "msgId": msg_id},
                    headers=auth,
                )
                if resp.status_code in (200, 204):
                    deleted += 1
                elif resp.status_code != 404:
                    logger.warning(
                        f"Failed to delete RC message {msg_id}: "
                        f"HTTP {resp.status_code} {resp.text[:200]}"
                    )
            except httpx.HTTPError as e:
                logger.warning(f"Failed to delete RC message {msg_id}: {e}")
        return deleted


class MailpitHandler(ServiceHandler):
    """Seed Mailpit emails via SMTP."""

    service_name = "mailpit"

    def __init__(self, api_url: str, smtp_host: str, smtp_port: int):
        self.api_url = api_url
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    async def reset(self) -> None:
        await self._wipe_messages()

    async def seed(self, records: list[dict]) -> CleanupHandle:
        handle = CleanupHandle(service="mailpit")
        for record in records:
            if record.get("type") == "contact":
                subject = f"AgentPrivArena Contact: {record.get('name', '')}"
                body = _contact_registry_body(record)
                sender = record.get("email") or "contact@agentprivarena.local"
                to = "john.doe@gmail.com"
                msg = MIMEText(body)
                msg["Subject"] = subject
                msg["From"] = formataddr((str(record.get("name", "")), sender))
                msg["To"] = to
                msg["X-AgentPrivArena-Record-Type"] = "contact"
                msg["X-AgentPrivArena-Contact-ID"] = _header_value(
                    record.get("contact_id", "")
                )
                msg["X-AgentPrivArena-Contact-Name"] = _header_value(record.get("name"))
                msg["X-AgentPrivArena-Contact-Email"] = _header_value(
                    record.get("email")
                )
                msg["X-AgentPrivArena-Contact-Remark"] = _header_value(
                    record.get("remark")
                )
                try:
                    with smtplib.SMTP(self.smtp_host, self.smtp_port) as smtp:
                        smtp.sendmail(sender, [to], msg.as_string())
                    logger.info(f"Seeded contact registry entry: {record.get('name')}")
                    handle.record_ids.append(subject)
                except Exception as e:
                    logger.warning(
                        f"Failed to seed contact '{record.get('name', '')}': {e}"
                    )
                continue

            subject = record.get("subject", "Seeded Email")
            body = record.get(
                "body",
                record.get("content", record.get("snippet", "")),
            )
            sender = record.get(
                "sender",
                record.get("from", "sender@agentprivarena.local"),
            )
            to = record.get("to", "john.doe@gmail.com")

            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = to if isinstance(to, str) else ", ".join(to)

            try:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as smtp:
                    smtp.sendmail(
                        sender,
                        [to] if isinstance(to, str) else to,
                        msg.as_string(),
                    )
                logger.info(f"Seeded email: {subject}")
                handle.record_ids.append(subject)
            except Exception as e:
                logger.warning(f"Failed to seed email '{subject}': {e}")

        return handle

    async def cleanup(self, handle: CleanupHandle) -> None:
        # Mailpit is a dedicated test inbox — wipe it wholesale.
        # SMTP gives us no per-message ID we could DELETE, so the
        # only honest cleanup is to DELETE /api/v1/messages, which
        # empties the inbox. Safe because nothing else writes here.
        if not handle.record_ids:
            return
        await self._wipe_messages()

    async def _wipe_messages(self) -> None:
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.delete(f"{self.api_url}/api/v1/messages")
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.warning(f"Failed to wipe Mailpit inbox: {e}")


class GoToSocialHandler(ServiceHandler):
    """Seed GoToSocial posts via the Mastodon-compatible API.

    Posts the agentprivarena user's own statuses for each post-shaped
    record. Profile records are skipped — GoToSocial does not let
    you provision arbitrary other-user accounts via the public API,
    so the agent will only ever see the agentprivarena account.
    """

    service_name = "gotosocial"

    def __init__(self, url: str, token: str = ""):
        self.url = url
        self.token = token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}

    async def reset(self) -> None:
        """Delete posts from the configured local test account."""
        if not self.token:
            return
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                resp = await client.get(
                    f"{self.url}/api/v1/accounts/verify_credentials",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                account_id = resp.json().get("id")
                if not account_id:
                    return
                await self._delete_account_statuses(client, account_id)
            except httpx.HTTPError as exc:
                logger.warning(f"Failed to reset GoToSocial posts: {exc}")

    async def seed(self, records: list[dict]) -> CleanupHandle:
        handle = CleanupHandle(service="gotosocial")
        if not self.token:
            logger.warning(
                "GoToSocial token not configured; skipping %d records",
                len(records),
            )
            return handle

        async with httpx.AsyncClient(timeout=30.0) as client:
            for record in records:
                # Profile records can't be seeded — only the agentprivarena
                # user exists. Log once per record and move on.
                if record.get("type") == "profile":
                    logger.warning(
                        "GoToSocial profile records cannot be seeded; "
                        "agent will see only the agentprivarena user"
                    )
                    continue

                content = record.get("content", "")
                if not content:
                    continue
                try:
                    resp = await client.post(
                        f"{self.url}/api/v1/statuses",
                        data={"status": content},
                        headers=self._headers(),
                    )
                    if resp.status_code in (200, 201):
                        status_id = resp.json().get("id", "")
                        if status_id:
                            handle.record_ids.append(status_id)
                            logger.info(f"Seeded GoToSocial status id={status_id}")
                    else:
                        logger.warning(
                            f"Failed to seed GoToSocial status: "
                            f"{resp.status_code} {resp.text[:200]}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to seed GoToSocial status: {e}")

        return handle

    async def cleanup(self, handle: CleanupHandle) -> None:
        if not self.token or not handle.record_ids:
            return
        async with httpx.AsyncClient(timeout=30.0) as client:
            await self._delete_statuses(client, handle.record_ids)

    async def _delete_account_statuses(
        self,
        client: httpx.AsyncClient,
        account_id: str,
    ) -> None:
        for _ in range(20):
            resp = await client.get(
                f"{self.url}/api/v1/accounts/{account_id}/statuses",
                params={"limit": 40},
                headers=self._headers(),
            )
            resp.raise_for_status()
            statuses = resp.json()
            status_ids: list[str] = []
            for status in statuses:
                if not isinstance(status, dict):
                    continue
                status_id = status.get("id")
                if isinstance(status_id, str):
                    status_ids.append(status_id)
            if not status_ids:
                return
            deleted = await self._delete_statuses(client, status_ids)
            if deleted == 0 or len(status_ids) < 40:
                return

    async def _delete_statuses(
        self,
        client: httpx.AsyncClient,
        status_ids: list[str],
    ) -> int:
        deleted = 0
        for status_id in status_ids:
            try:
                resp = await client.delete(
                    f"{self.url}/api/v1/statuses/{status_id}",
                    headers=self._headers(),
                )
                if resp.status_code in (200, 204):
                    deleted += 1
                elif resp.status_code not in (404, 410):
                    logger.warning(
                        f"Failed to delete GoToSocial status {status_id}: "
                        f"HTTP {resp.status_code} {resp.text[:200]}"
                    )
            except httpx.HTTPError as e:
                logger.warning(f"Failed to delete GoToSocial status {status_id}: {e}")
        return deleted


class RadicaleHandler(ServiceHandler):
    """Seed Radicale calendar events via CalDAV."""

    service_name = "radicale"

    def __init__(self, url: str, user: str, password: str):
        self.url = url
        self.user = user
        self.password = password

    async def reset(self) -> None:
        import caldav

        try:
            client = caldav.DAVClient(
                url=self.url,
                username=self.user,
                password=self.password,
            )
            principal = client.principal()
            calendars = principal.calendars()
            if not calendars:
                return
            calendar = calendars[0]
        except Exception as e:
            logger.warning(f"Failed to connect to Radicale for reset: {e}")
            return

        for event in calendar.events():
            try:
                event.delete()
            except Exception as e:
                logger.warning(f"Failed to delete calendar event during reset: {e}")

    async def seed(self, records: list[dict]) -> CleanupHandle:
        import caldav

        handle = CleanupHandle(service="radicale")
        client = caldav.DAVClient(
            url=self.url,
            username=self.user,
            password=self.password,
        )
        try:
            principal = client.principal()
            calendars = principal.calendars()
            calendar = (
                calendars[0] if calendars else principal.make_calendar(name="default")
            )
        except Exception as e:
            logger.warning(f"Failed to connect to Radicale: {e}")
            return handle

        for record in records:
            event_id = record.get("event_id", record.get("event_name", ""))
            if isinstance(event_id, str) and not record.get("event_name"):
                continue

            name = record.get("event_name", "Event")
            content = record.get("content", "")
            start = record.get("start_time", "2022-02-25T10:00:00")
            end = record.get("end_time", "2022-02-25T11:00:00")
            location = record.get("location", "")
            attendees = record.get("attendees", [])

            vcal = (
                "BEGIN:VCALENDAR\nVERSION:2.0\nBEGIN:VEVENT\n"
                f"UID:{event_id}\n"
                f"SUMMARY:{name}\n"
                f"DESCRIPTION:{content}\n"
                f"DTSTART:{_ical_dt(start)}\n"
                f"DTEND:{_ical_dt(end)}\n"
                f"LOCATION:{location}\n"
            )
            for attendee in attendees:
                vcal += f"ATTENDEE:mailto:{attendee}\n"
            vcal += "END:VEVENT\nEND:VCALENDAR"

            try:
                calendar.save_event(vcal)
                handle.record_ids.append(event_id)
                logger.info(f"Seeded calendar event: {name}")
            except Exception as e:
                logger.warning(f"Failed to seed event '{name}': {e}")

        return handle

    async def cleanup(self, handle: CleanupHandle) -> None:
        if not handle.record_ids:
            return
        import caldav

        try:
            client = caldav.DAVClient(
                url=self.url,
                username=self.user,
                password=self.password,
            )
            principal = client.principal()
            calendars = principal.calendars()
            if not calendars:
                return
            calendar = calendars[0]
        except Exception as e:
            logger.warning(f"Failed to connect to Radicale for cleanup: {e}")
            return

        for event_uid in handle.record_ids:
            try:
                event = calendar.event_by_uid(event_uid)
                event.delete()
            except Exception as e:
                logger.warning(f"Failed to delete calendar event {event_uid}: {e}")


class GoogleDriveArtifactHandler(ServiceHandler):
    """Seed local Google Drive artifacts into the MCP-mounted runtime store."""

    service_name = "google_drive"

    def __init__(self, artifact_root: Path | str):
        self.artifact_root = Path(artifact_root)

    async def reset(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        for child in self.artifact_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()

    async def seed(self, records: list[dict]) -> CleanupHandle:
        del records
        raise RuntimeError(
            "google_drive seed data must be a directory containing drive_index.json"
        )

    async def seed_from_path(self, seed_path: Path) -> CleanupHandle:
        index_path = seed_path / "drive_index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Google Drive seed index not found: {index_path}")
        await self.reset()
        shutil.copytree(seed_path, self.artifact_root, dirs_exist_ok=True)
        files = json.loads(index_path.read_text()).get("files", [])
        record_ids = [
            str(item.get("file_id"))
            for item in files
            if isinstance(item, dict) and item.get("file_id")
        ]
        return CleanupHandle(service=self.service_name, record_ids=record_ids)

    async def cleanup(self, handle: CleanupHandle) -> None:
        del handle
        await self.reset()


class Seeder:
    """Orchestrates seeding from a task's seed_data/ directory."""

    def __init__(
        self,
        bookstack_url: str = "http://localhost:3000",
        bookstack_token_id: str = "",
        bookstack_token_secret: str = "",
        mattermost_url: str = "http://localhost:8065",
        mattermost_user: str = "admin",
        mattermost_password: str = "admin",
        rocketchat_url: str = "http://localhost:3100",
        rocketchat_user: str = "admin",
        rocketchat_password: str = "admin",
        mailpit_api_url: str = "http://localhost:8025",
        mailpit_smtp_host: str = "localhost",
        mailpit_smtp_port: int = 1025,
        gotosocial_url: str = "http://localhost:4000",
        gotosocial_token: str = "",
        radicale_url: str = "http://localhost:5232",
        radicale_user: str = "admin",
        radicale_password: str = "admin",
        google_drive_artifact_root: Path | str = (
            ".agent_tmp/agentprivarena_artifacts/google_drive"
        ),
    ):
        self.handlers: dict[str, ServiceHandler] = {
            "bookstack": BookStackHandler(
                bookstack_url,
                bookstack_token_id,
                bookstack_token_secret,
            ),
            "mattermost": MattermostHandler(
                mattermost_url,
                mattermost_user,
                mattermost_password,
            ),
            "rocketchat": RocketChatHandler(
                rocketchat_url,
                rocketchat_user,
                rocketchat_password,
            ),
            "mailpit": MailpitHandler(
                mailpit_api_url,
                mailpit_smtp_host,
                mailpit_smtp_port,
            ),
            "gotosocial": GoToSocialHandler(
                gotosocial_url,
                gotosocial_token,
            ),
            "radicale": RadicaleHandler(
                radicale_url,
                radicale_user,
                radicale_password,
            ),
            "google_drive": GoogleDriveArtifactHandler(google_drive_artifact_root),
        }

    async def seed_from_dir(self, seed_dir: Path) -> SeedResult:
        """Read seed data files and inject into real services."""
        result = SeedResult()
        for seed_path in sorted(seed_dir.iterdir(), key=lambda path: path.name):
            if seed_path.is_file() and seed_path.suffix == ".json":
                service = seed_path.stem
            elif seed_path.is_dir():
                service = seed_path.name
            else:
                continue
            handler = self.handlers.get(service)
            if handler is None:
                logger.warning(f"No handler for service: {service}")
                continue
            handle = await handler.seed_from_path(seed_path)
            result.handles.append(handle)
        return result

    async def reset_services(self, services: list[str]) -> None:
        """Clear task-visible service content before seeding a scenario."""
        seen: set[str] = set()
        for service in services:
            if service in seen:
                continue
            seen.add(service)
            handler = self.handlers.get(service)
            if handler is None:
                logger.warning(f"No handler for service reset: {service}")
                continue
            try:
                await handler.reset()
            except Exception as exc:
                logger.warning(f"Failed to reset service '{service}': {exc}")

    async def cleanup(self, result: SeedResult) -> None:
        """Remove all seeded data."""
        for handle in result.handles:
            handler = self.handlers.get(handle.service)
            if handler:
                await handler.cleanup(handle)
