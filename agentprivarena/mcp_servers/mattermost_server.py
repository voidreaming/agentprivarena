"""Mattermost MCP server — direct messaging tools.

Provides DM send/receive, message search, and user listing via Mattermost
REST API v4.

Error handling
--------------
All upstream HTTP calls go through the shared ``base.http_*`` helpers, which
translate every httpx failure into a ``HTTPToolError`` carrying the upstream
status code. Tools wrap their work in :func:`_authenticated_call`, which:

- supplies fresh auth headers from the cached login
- on a confirmed ``HTTP 401`` response, clears the auth cache, re-logs in, and
  retries the call exactly once before propagating the error

Business-logic failures (e.g. an unknown recipient) raise :class:`ToolError`
directly with an actionable recovery hint that names the tool the agent can
use to discover valid inputs.
"""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from typing import Annotated, Any

from base import (
    HTTPToolError,
    http_get,
    http_get_params,
    http_post,
    http_post_for_login,
    logger,
)
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import Field


MATTERMOST_URL = os.environ.get("MATTERMOST_URL", "http://mattermost:8065")
MM_USER = os.environ.get("MATTERMOST_USER", "admin")
MM_PASSWORD = os.environ.get("MATTERMOST_PASSWORD", "admin")

API = f"{MATTERMOST_URL}/api/v4"
SEEDED_SENDER_PROP = "agentprivarena_sender"
SEEDED_SENDER_DISPLAY_PROP = "agentprivarena_sender_display"
SEEDED_SENDER_UNKNOWN_PROP = "agentprivarena_sender_unknown"
SEEDED_TIME_PROP = "agentprivarena_time"
GROUP_SENDER_PREFIX = "group_"

mcp = FastMCP("mattermost")

# ── Auth cache ──
# Module-level so the helpers below can clear it on a 401 retry.
_auth: dict[str, str] = {}  # {"token": ..., "user_id": ...}


# ── Auth helpers ──


async def _login() -> dict[str, str]:
    """Log in (or return cached creds) and populate ``_auth``.

    Mattermost returns the bearer token in the ``Token`` *response header*
    rather than the JSON body, so we use ``http_post_for_login`` to get both.
    """
    if _auth.get("token"):
        return _auth
    body, headers = await http_post_for_login(
        f"{API}/users/login",
        json_data={"login_id": MM_USER, "password": MM_PASSWORD},
    )
    token = headers.get("token") or headers.get("Token")
    if not token:
        raise ToolError(
            "Mattermost login response missing 'Token' header. "
            "Check MATTERMOST_USER / MATTERMOST_PASSWORD credentials."
        )
    _auth["token"] = token
    _auth["user_id"] = body["id"]
    return _auth


def _auth_headers(auth: dict[str, str]) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth['token']}"}


async def _authenticated_call[T](
    fn: Callable[[dict[str, str]], Awaitable[T]],
) -> T:
    """Run ``fn(headers)`` with cached auth, refreshing on a single 401.

    The retry is intentionally narrow: we only refresh when the upstream
    explicitly responds with HTTP 401 (a confirmed auth rejection). Network
    errors, 5xx, and other 4xx propagate as-is. The retry happens at most
    once; a second 401 surfaces normally so the agent can see the failure.
    """
    auth = await _login()
    try:
        return await fn(_auth_headers(auth))
    except HTTPToolError as exc:
        if exc.status_code != 401:
            raise
        logger.info(
            "Mattermost auth token rejected (HTTP 401); "
            "clearing cache and retrying once."
        )
        _auth.clear()
        auth = await _login()
        return await fn(_auth_headers(auth))


async def _resolve_user_id(username: str, headers: dict[str, str]) -> str:
    """Resolve a Mattermost username to a user ID, with a friendly miss message."""
    try:
        data = await http_get(f"{API}/users/username/{username}", headers=headers)
    except HTTPToolError as exc:
        if exc.status_code == 404:
            raise ToolError(
                f"User '{username}' not found in Mattermost. "
                "Use list_users to discover valid usernames."
            ) from exc
        raise
    return data["id"]


async def _get_or_create_dm(
    user_id: str, other_user_id: str, headers: dict[str, str]
) -> str:
    """Get or create a direct-message channel between two users."""
    dm = await http_post(
        f"{API}/channels/direct",
        json_data=[user_id, other_user_id],
        headers=headers,
    )
    return dm["id"]


async def _resolve_user_ids_to_usernames(
    user_ids: list[str], headers: dict[str, str]
) -> dict[str, str]:
    """Batch-resolve Mattermost user IDs to usernames via /api/v4/users/ids.

    Returns a ``{user_id: username}`` map. Callers should fall back to the
    raw user_id on a miss via ``id_to_name.get(uid, uid)``.

    Failure handling: HTTP 401 propagates so the outer ``_authenticated_call``
    can refresh the auth cache and retry. All other failures are tolerated
    and logged at warning level — the caller will fall through to raw IDs
    rather than turning a perf optimization into a tool failure.
    """
    if not user_ids:
        return {}
    try:
        users = await http_post(
            f"{API}/users/ids",
            json_data=user_ids,
            headers=headers,
        )
    except HTTPToolError as exc:
        if exc.status_code == 401:
            raise
        logger.warning(
            f"_resolve_user_ids_to_usernames: failed to resolve "
            f"{len(user_ids)} ids ({exc}); falling back to raw IDs"
        )
        return {}
    return {
        u["id"]: u.get("username", u["id"])
        for u in (users or [])
        if isinstance(u, dict) and "id" in u
    }


async def _channels_for_listing(headers: dict[str, str]) -> list[dict[str, Any]]:
    """Return visible channels plus seeded town-square channels."""
    channels = await http_get(f"{API}/users/me/channels", headers=headers)
    by_id = {
        channel["id"]: channel
        for channel in channels
        if isinstance(channel, dict) and channel.get("id")
    }

    teams = await http_get(f"{API}/users/me/teams", headers=headers)
    for team in teams:
        if not isinstance(team, dict) or not team.get("id"):
            continue
        try:
            channel = await http_get(
                f"{API}/teams/{team['id']}/channels/name/town-square",
                headers=headers,
            )
        except HTTPToolError as exc:
            if exc.status_code == 401:
                raise
            logger.warning(
                "Mattermost read_messages: failed to resolve town-square "
                f"for team {team['id']} ({exc}); continuing"
            )
            continue
        if isinstance(channel, dict) and channel.get("id"):
            by_id[channel["id"]] = channel

    return list(by_id.values())


def _display_seeded_sender(sender: str) -> str:
    """Convert PrivacyLens sender ids into readable person names when possible."""
    sender = sender.strip()
    if not sender:
        return sender
    if sender.lower().startswith(GROUP_SENDER_PREFIX):
        return sender

    without_suffix = re.sub(r"[_-]\d+$", "", sender)
    tokens = [token for token in re.split(r"[_\-.]+", without_suffix) if token]
    if not tokens or not all(token.isalpha() for token in tokens):
        return sender
    return " ".join(token.capitalize() for token in tokens)


def _post_sender_identity(
    post: dict[str, Any],
    id_to_name: dict[str, str],
) -> tuple[str, str]:
    """Return ``(username, display_name)`` for a Mattermost post.

    PrivacyLens seed data is posted by the admin account but carries the
    original benchmark sender in post props. Tool outputs must keep ``sender``
    machine-usable because agents pass it to ``send_message``. Human-readable
    seeded names are therefore exposed separately as ``sender_display``.
    """
    props = post.get("props")
    if isinstance(props, dict):
        if props.get(SEEDED_SENDER_UNKNOWN_PROP):
            return "", "unknown sender"
        seeded_display = props.get(SEEDED_SENDER_DISPLAY_PROP)
        seeded_sender = props.get(SEEDED_SENDER_PROP)
        if isinstance(seeded_sender, str) and seeded_sender:
            display = (
                seeded_display
                if isinstance(seeded_display, str) and seeded_display
                else _display_seeded_sender(seeded_sender)
            )
            if seeded_sender.lower().startswith(GROUP_SENDER_PREFIX):
                return "", display
            return seeded_sender, display

    user_id = str(post.get("user_id", ""))
    username = id_to_name.get(user_id, user_id)
    return username, username


def _post_sender(post: dict[str, Any], id_to_name: dict[str, str]) -> str:
    """Return the username suitable for ``send_message`` when available."""
    return _post_sender_identity(post, id_to_name)[0]


def _post_time(post: dict[str, Any]) -> int | str:
    """Return seeded benchmark time when available, else Mattermost create time."""
    props = post.get("props")
    if isinstance(props, dict):
        seeded_time = props.get(SEEDED_TIME_PROP)
        if isinstance(seeded_time, int):
            return seeded_time
        if isinstance(seeded_time, str) and seeded_time.isdigit():
            return int(seeded_time)
    return post.get("create_at", "")


def _sortable_post_time(post: dict[str, Any]) -> int:
    value = _post_time(post)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _post_to_message(
    post: dict[str, Any],
    id_to_name: dict[str, str],
    channel_name: str | None = None,
) -> dict[str, Any]:
    sender, sender_display = _post_sender_identity(post, id_to_name)
    message = {
        "message_id": post["id"],
        "sender": sender,
        "time": _post_time(post),
        "text": post["message"],
    }
    if sender_display and sender_display != sender:
        message["sender_display"] = sender_display
    if channel_name:
        message["conversation"] = channel_name
    return message


# ── Tools ──


@mcp.tool()
async def send_message(
    recipient: Annotated[
        str,
        Field(
            description=(
                "Mattermost username (no '@' prefix), as returned by list_users. "
                "Not a display name and not an email address."
            )
        ),
    ],
    message: Annotated[
        str,
        Field(
            description=(
                "Plain-text message body. Parameter name is `message` (not `text`)."
            )
        ),
    ],
) -> dict:
    """Send a private direct message to one user. Only the recipient will see it.

    Pass the recipient username from list_users results. The message-body
    parameter is named ``message`` (not ``text`` as in some other chat APIs).
    """

    async def _do(headers: dict[str, str]) -> dict[str, Any]:
        other_id = await _resolve_user_id(recipient, headers)
        channel_id = await _get_or_create_dm(_auth["user_id"], other_id, headers)
        return await http_post(
            f"{API}/posts",
            json_data={"channel_id": channel_id, "message": message},
            headers=headers,
        )

    post = await _authenticated_call(_do)
    return {
        "success": True,
        "message_id": post["id"],
        "recipient": recipient,
    }


async def _read_messages_payload(max_count: int) -> dict:
    async def _do(headers: dict[str, str]) -> list[dict[str, Any]]:
        channels = await _channels_for_listing(headers)
        raw_posts: list[dict[str, Any]] = []
        channel_by_id: dict[str, str] = {}
        for ch in channels:
            channel_id = ch.get("id")
            if not channel_id:
                continue
            channel_by_id[channel_id] = ch.get("display_name") or ch.get("name", "")
            try:
                data = await http_get_params(
                    f"{API}/channels/{channel_id}/posts",
                    params={"per_page": max_count},
                    headers=headers,
                )
            except HTTPToolError as exc:
                if exc.status_code == 401:
                    raise
                logger.warning(
                    "Mattermost read_messages: failed to read channel "
                    f"{channel_id} ({exc}); continuing"
                )
                continue
            order = data.get("order", [])
            posts = data.get("posts", {})
            for post_id in order[:max_count]:
                post = posts.get(post_id, {})
                if post.get("message"):
                    raw_posts.append(post)

        raw_posts.sort(key=lambda p: _sortable_post_time(p), reverse=True)
        user_ids = list({p.get("user_id", "") for p in raw_posts if p.get("user_id")})
        id_to_name = await _resolve_user_ids_to_usernames(user_ids, headers)

        return [
            _post_to_message(p, id_to_name, channel_by_id.get(p.get("channel_id", "")))
            for p in raw_posts[:max_count]
        ]

    messages = await _authenticated_call(_do)
    return {"messages": messages}


@mcp.tool(annotations={"readOnlyHint": True})
async def read_messages(
    max_count: Annotated[
        int,
        Field(
            description=(
                "Maximum number of recent messages to read across visible "
                "channel and DM conversations. Parameter name is `max_count` "
                "(not `limit`, not `count`, not `per_page`)."
            ),
            ge=1,
            le=200,
        ),
    ] = 20,
) -> dict:
    """Read recent messages across visible channel and DM conversations."""
    return await _read_messages_payload(max_count)


@mcp.tool(annotations={"readOnlyHint": True})
async def search_messages(
    query: Annotated[
        str,
        Field(
            description=(
                "Required search terms to match against message text. Put the "
                "terms in `query`; do not put them only in the action "
                "`summary`. Space-separated terms are OR'd together. "
                "Parameter name is `query` (not `keyword`)."
            )
        ),
    ],
) -> dict:
    """Search messages by keyword across all conversations.

    The search-term parameter is named ``query`` (not ``keyword``).

    `sender` is a Mattermost username, suitable for passing to send_message
    as `recipient` when non-empty. `sender_display` is the human-readable
    logical sender for seeded messages. `time` is a Mattermost
    epoch-milliseconds integer.
    """

    async def _do(headers: dict[str, str]) -> list[dict[str, Any]]:
        teams = await http_get(f"{API}/users/me/teams", headers=headers)
        if not teams:
            raise ToolError(
                "No team found for the authenticated Mattermost user. "
                "Create a team first via the admin API "
                "(POST /api/v4/teams) before searching messages."
            )
        team_id = teams[0]["id"]
        data = await http_post(
            f"{API}/teams/{team_id}/posts/search",
            json_data={"terms": query, "is_or_search": True},
            headers=headers,
        )

        order = data.get("order", [])
        posts = data.get("posts", {})
        raw_posts = [
            posts.get(pid, {})
            for pid in order[:20]
            if posts.get(pid, {}).get("message")
        ]

        user_ids = list({p.get("user_id", "") for p in raw_posts if p.get("user_id")})
        id_to_name = await _resolve_user_ids_to_usernames(user_ids, headers)

        return [_post_to_message(p, id_to_name) for p in raw_posts]

    messages = await _authenticated_call(_do)
    return {"messages": messages}


@mcp.tool(annotations={"readOnlyHint": True})
async def list_users() -> dict:
    """List all users on the server.

    Use the username field as the recipient for send_message.
    Bot accounts are filtered out.

    This tool takes no parameters. Do not pass ``page``, ``per_page``,
    ``limit``, or any other pagination argument — there is no pagination
    on this tool.
    """

    async def _do(headers: dict[str, str]) -> list[dict[str, Any]]:
        return await http_get_params(
            f"{API}/users",
            params={"per_page": 50},
            headers=headers,
        )

    data = await _authenticated_call(_do)
    users = [
        {
            "username": u["username"],
            "name": (f"{u.get('first_name', '')} {u.get('last_name', '')}").strip(),
            "email": u.get("email", ""),
        }
        for u in data
        if not u.get("is_bot")
    ]
    return {"users": users}


if __name__ == "__main__":
    logger.info(f"Starting Mattermost MCP server (url={MATTERMOST_URL})")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8080)
