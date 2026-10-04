"""Generate per-task directories from main_data.json.

Reads the PrivacyLens dataset, parses each entry's executable_trajectory,
and writes per-task directories with task.json and seed_data/*.json files.

Usage:
    python -m privacylens_live.tasks.generate \\
        --data main_data.json --output privacylens_live/tasks
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from privacylens_live.base.trajectory_parser import (
    TrajectoryStep,
    get_seed_observations,
    parse_trajectory,
)
from privacylens_live.config import ACTION_TO_SERVICE
from privacylens_live.tasks.structured_seed import (
    DROPPED_TASKS_FILENAME,
    MANIFEST_FILENAME,
    PreparedSeed,
    prepare_seed_records,
)


logger = logging.getLogger("generate")
MATTERMOST_GROUP_CHAT_UNSUPPORTED = "mattermost_group_chat_unsupported"
REFERENCED_SOURCE_WITHOUT_SEED_OBSERVATION = (
    "referenced_source_without_seed_observation"
)
PROVIDER_CONTENT_FILTER_UNSTABLE = "provider_content_filter_unstable"
# These tasks are valid PrivacyLens examples, but Azure OpenAI consistently
# rejects the initial prompt as a content-filter false positive before the
# agent can take any action. Keep them out of live-platform batches so the
# eval measures agent behavior rather than provider policy instability.
PROVIDER_CONTENT_FILTER_DROP_TASKS = frozenset({"main221", "main245", "main308"})

_MATTERMOST_GROUP_TARGET_RE = re.compile(
    r"(?:"
    r"group chat|"
    r"whatsapp group|"
    r"\bgroup\s+in\s+[\"“]|"
    r"\bto\s+(?:the|our|my)\s+[^.]{0,80}\bgroup\b"
    r")",
    re.IGNORECASE,
)
_MATTERMOST_GROUP_FALSE_POSITIVE_RE = re.compile(
    r"\b(?:group project|support group member)\b",
    re.IGNORECASE,
)
_CHANNEL_PATTERNS = [
    re.compile(
        r"[\"“](?P<channel>[^\"”]{2,80})[\"”]\s+"
        r"(?:(?:RocketChat|Slack)\s+)?channel\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:in|on|to)\s+(?:the\s+)?"
        r"#(?P<channel>[A-Za-z0-9][A-Za-z0-9_-]{1,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:in|on|to)\s+(?:the\s+)?"
        r"(?P<channel>[A-Z][A-Za-z0-9& _-]{1,80}?)\s+"
        r"(?:(?:RocketChat|Slack)\s+)?channel\b",
        re.IGNORECASE,
    ),
]
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PAREN_EMAIL_RE = re.compile(
    r"(?P<name>[^?!\n()]{1,160}?)\s*\((?P<email>"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\)"
)
_PAREN_PHONE_RE = re.compile(
    r"(?P<name>[^?!\n()]{1,160}?)\s*\((?P<phone>"
    r"\+?[0-9][0-9 .()_-]{6,}[0-9])\)"
)
_PAREN_NAME_RE = re.compile(
    r"\((?P<name>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})\)"
)
_TARGET_PATTERNS = [
    re.compile(
        r"\b(?P<target>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})"
        r"(?:'s|’s|\s+has)\s+been\s+asking\b"
        r"[\s\S]{0,220}?\b(?:update|message|reply\s+to)\s+"
        r"(?:him|her|them)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:help\s+me\s+write|write|draft|compose)\s+"
        r"(?:an?\s+)?(?:text\s+message|message|email)\s+to\s+"
        r"(?P<target>[^.?!\n]{1,120}?)"
        r"(?:\s+(?:to|about|asking|explaining|regarding|concerning|on|via|for)\b|[.?!]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:respond|reply)\s+to\s+(?P<target>[^.?!\n]{1,120}?)"
        r"(?:'s|’s)\s+(?:latest\s+)?"
        r"(?:(?:Mattermost|Messenger|RocketChat|Slack|Mailpit|Gmail)\s+)?"
        r"(?:message|email|mail|dm|direct message|inquiry|request|question)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:send\s+a\s+text\s+back\s+to|text\s+back|message\s+back)\s+"
        r"(?P<target>[^.?!\n]{1,120}?)"
        r"(?:\s+(?:on|via|about|regarding|concerning)\b|[.?!]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:send|compose)\s+(?:an?\s+)?(?:email|message)\s+to\s+"
        r"(?P<target>[^.?!\n]{1,120}?)"
        r"(?:\s+(?:with|about|asking|explaining|regarding|concerning|on|via|for)\b|[.?!]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:send|message|update)\s+"
        r"(?P<target>(?:(?:my|our|the)\s+)?"
        r"(?:(?:coworker|co-worker|colleague|friend|manager|partner|"
        r"client|husband|wife|sibling|son|daughter|child|kid|step-father|"
        r"stepfather|step-mother|stepmother)\s+)?"
        r"[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})"
        r"\s+(?:an?\s+)?(?:message|update|dm|direct message)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:message|update)\s+"
        r"(?P<target>(?:(?:my|our|the)\s+)?"
        r"(?:(?:coworker|co-worker|colleague|friend|manager|partner|"
        r"client|husband|wife|sibling)\s+)?"
        r"[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})"
        r"\b[^.?!\n]{0,120}\b(?:via|on)\s+"
        r"(?:RocketChat|Mattermost|Messenger|Slack)\s+"
        r"(?:direct\s+)?message\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b[Ss]end\s+(?P<target>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})"
        r"\s+(?:an?\s+|some\s+|my\s+|our\s+|the\s+)?(?:update|message|email|reply|note)\b",
    ),
    re.compile(
        r"\b[Ss]end\s+"
        r"(?P<target>(?:(?:my|our|the)\s+)?"
        r"(?:(?:son|daughter|child|kid|step-father|stepfather|step-mother|"
        r"stepmother|coworker|co-worker|colleague|friend|manager|partner|"
        r"client|husband|wife|sibling)\s+)?"
        r"[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})"
        r"\s+(?:some\s+|an?\s+|my\s+|our\s+|the\s+)?"
        r"(?:insights|information|details|notes|summary|tips|advice|update|"
        r"message|email|reply|note)\b",
    ),
    re.compile(
        r"\b[Ll]et\s+(?P<target>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,3})\s+know\b",
    ),
]
_TARGET_PREFIX_RE = re.compile(
    r"^(?:"
    r"my|our|the|a|an|his|her|their|"
    r"friend|close friend|coworker|co-worker|colleague|client|manager|"
    r"third-party contractor|contractor|business partner|"
    r"husband|wife|sibling|son|daughter|child|kid|"
    r"step-father|stepfather|step-mother|stepmother|"
    r"dad|father|mom|mother|parent|parents"
    r")\s+",
    re.IGNORECASE,
)
_TARGET_STOP_RE = re.compile(
    r"\s+"
    r"(?:on|in|via|using|from|at|about|with|for|asking|explaining|"
    r"regarding|concerning|that|who|if|to)"
    r"\b.*$",
    re.IGNORECASE,
)
_NON_TARGET_NAMES = {
    "BookStack",
    "Gmail",
    "Mailpit",
    "Mattermost",
    "Messenger",
    "Notion",
    "RocketChat",
    "Slack",
}
_SOURCE_REFERENCE_TERMS: dict[str, tuple[str, ...]] = {
    "bookstack": ("BookStackManager", "BookStack", "Notion", "notebook", "notebooks"),
    "mailpit": ("Mailpit", "Gmail", "email", "emails", "mail", "mails"),
    "mattermost": ("Mattermost", "Messenger"),
    "rocketchat": ("RocketChat", "Slack"),
    "gotosocial": ("GoToSocial", "Facebook"),
    "radicale": ("Radicale", "Google Calendar", "calendar"),
}
_SOURCE_REFERENCE_VERB_RE = re.compile(
    r"\b(?:check(?:\s+out)?|look(?:\s+at|\s+through)?|review|read|use)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TargetRecipient:
    name: str
    email: str = ""


def _normalize_radicale_event(event: dict) -> dict:
    """Normalize a calendar event into the seeder's expected shape.

    Source data uses several shapes for the same event:
    - flat: ``{event_name, start_time, end_time, content, location, ...}``
    - structured: ``{summary, description, start: {dateTime}, end: {dateTime}}``

    The seeder only reads the flat shape, so anything else is silently
    dropped without normalization. This collapses both into the flat
    form.
    """
    name = (
        event.get("event_name") or event.get("summary") or event.get("title") or "Event"
    )
    start = event.get("start_time", "")
    if not start and isinstance(event.get("start"), dict):
        start = event["start"].get("dateTime") or event["start"].get("date") or ""
    end = event.get("end_time", "")
    if not end and isinstance(event.get("end"), dict):
        end = event["end"].get("dateTime") or event["end"].get("date") or ""
    return {
        "event_id": str(event.get("event_id") or event.get("id") or name),
        "event_name": name,
        "content": event.get("content") or event.get("description", ""),
        "start_time": start,
        "end_time": end,
        "location": event.get("location", ""),
        "attendees": event.get("attendees", []),
    }


def _warn_unknown_shape(step: TrajectoryStep, service: str) -> None:
    """Log a warning when an observation falls into the raw fallback path.

    This is the hint that the source data has a shape the extractor
    doesn't recognize and the resulting seed file will contain a
    record the seeder cannot consume. The verify command surfaces
    these counts; this warning makes them visible during ``generate``.
    """
    obs = step.observation
    if isinstance(obs, dict):
        shape = f"keys={list(obs.keys())}"
    else:
        shape = f"type={type(obs).__name__}"
    logger.warning(
        "Unknown observation shape for %s → %s (%s); falling back to raw_observation",
        step.action_name,
        service,
        shape,
    )


# PrivacyLens toolkit name → MCP server name mapping.
# Used only during task generation to compute dependencies.
_TOOLKIT_TO_MCP: dict[str, str] = {
    "NotionManager": "bookstack",
    "Messenger": "mattermost",
    "Slack": "rocketchat",
    "Gmail": "mailpit",
    "FacebookManager": "gotosocial",
    "GoogleCalendar": "radicale",
    "ZoomManager": "bookstack",
    "GoogleFormFiller": "bookstack",
}


def _observation_to_seed_records(step: TrajectoryStep, service: str) -> list[dict]:
    """Convert a trajectory step's observation into seed records.

    Extracts the actual data items (pages, messages, emails, events)
    from the observation JSON and returns them as a list of records
    ready to be injected into the service.
    """
    obs = step.observation
    if isinstance(obs, str):
        # Generic string-shaped fallback: use the same sentinel
        # key as the per-service fallbacks so verify counts it.
        return [{"raw_observation": obs, "action": step.action_name}]

    records = []

    if service == "bookstack":
        # Multiple shapes here:
        #   NotionManagerSearchContent → {"results": [{id,title,content}]}
        #   NotionManagerReadPage      → {"page_content": "<plain text>"}
        #   ZoomManagerSearchTranscript→ {"search_results": ["snippet", ...]}
        if "results" in obs:
            for item in obs["results"]:
                records.append(
                    {
                        "id": str(item.get("id", "")),
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "tags": item.get("tags", []),
                    }
                )
        elif "page_content" in obs:
            content = obs["page_content"]
            if not isinstance(content, str):
                content = json.dumps(content)
            # Source has no title — synthesize from first line so the
            # page is discoverable via list_pages / search_pages.
            first_line = content.split("\n", 1)[0].strip()
            title = first_line[:80].strip() if first_line else "Untitled Page"
            records.append(
                {
                    "id": "",
                    "title": title,
                    "content": content,
                    "tags": [],
                }
            )
        elif "search_results" in obs:
            snippets = obs["search_results"]
            if isinstance(snippets, list):
                content = "\n\n".join(str(s) for s in snippets)
                records.append(
                    {
                        "id": "",
                        "title": "Meeting transcript search results",
                        "content": content,
                        "tags": [],
                    }
                )
            else:
                _warn_unknown_shape(step, service)
                records.append({"raw_observation": obs, "action": step.action_name})
        elif "transcript" in obs:
            # ZoomManagerGetMeetingTranscript returns a single transcript.
            transcript = obs["transcript"]
            if isinstance(transcript, list):
                content = "\n".join(str(s) for s in transcript)
            else:
                content = str(transcript)
            records.append(
                {
                    "id": "",
                    "title": "Meeting transcript",
                    "content": content,
                    "tags": [],
                }
            )
        else:
            _warn_unknown_shape(step, service)
            records.append({"raw_observation": obs, "action": step.action_name})

    elif service == "mattermost":
        # MessengerReceiveMessage has {"messages": [...]} with full
        # sender_id/time/message records. MessengerSearchInChat has
        # {"results": [...]} with only message_id+message — sender
        # and time aren't carried in the source data.
        if "messages" in obs:
            inner = obs["messages"]
            # Source occasionally double-wraps: {"messages": [{"messages": [...]}]}
            # (e.g. main324). Unwrap one level when we see it.
            if (
                len(inner) == 1
                and isinstance(inner[0], dict)
                and "messages" in inner[0]
            ):
                inner = inner[0]["messages"]
            for msg in inner:
                records.append(
                    {
                        "message_id": msg.get("message_id", ""),
                        "sender_id": msg.get("sender_id", ""),
                        "time": msg.get("time", ""),
                        "message": msg.get("message", ""),
                    }
                )
        elif "results" in obs:
            for msg in obs["results"]:
                records.append(
                    {
                        "message_id": msg.get("message_id", ""),
                        "sender_id": "",
                        "time": "",
                        "message": msg.get("message", ""),
                    }
                )
        else:
            _warn_unknown_shape(step, service)
            records.append({"raw_observation": obs, "action": step.action_name})

    elif service == "rocketchat":
        # SlackSearchMessage has {"messages": [...]}. Slack uses
        # `from`/`content`/`in`/`timestamp` as the canonical keys —
        # the older names are kept as fallbacks for hand-authored
        # fixtures, but real data hits the Slack-native names first.
        if "messages" in obs:
            for msg in obs["messages"]:
                records.append(
                    {
                        "message_id": msg.get("message_id", ""),
                        "sender_id": msg.get(
                            "from", msg.get("sender_id", msg.get("user_id", ""))
                        ),
                        "time": msg.get("timestamp", msg.get("time", "")),
                        "message": msg.get(
                            "content", msg.get("message", msg.get("text", ""))
                        ),
                        "channel": msg.get("in", msg.get("channel", "")),
                    }
                )
        elif "profile" in obs:
            records.append(
                {
                    "type": "user_profile",
                    "profile": obs["profile"],
                }
            )
        elif "results" in obs:
            for item in obs["results"]:
                records.append(item)
        else:
            _warn_unknown_shape(step, service)
            records.append({"raw_observation": obs, "action": step.action_name})

    elif service == "mailpit":
        # GmailSearchEmails has {"emails": [...]} or {"results": [...]}
        if "emails" in obs:
            for email in obs["emails"]:
                records.append(email)
        elif "results" in obs:
            for item in obs["results"]:
                records.append(item)
        elif "contacts" in obs:
            for contact in obs["contacts"]:
                records.append({"type": "contact", **contact})
        else:
            # Single email read
            records.append(obs)

    elif service == "radicale":
        # GoogleCalendarSearchEvents has {"events": [...]}
        if "events" in obs:
            for event in obs["events"]:
                if isinstance(event, str):
                    records.append({"event_id": event})
                else:
                    records.append(_normalize_radicale_event(event))
        elif "event_details" in obs:
            for event in obs["event_details"]:
                records.append(_normalize_radicale_event(event))
        else:
            _warn_unknown_shape(step, service)
            records.append({"raw_observation": obs, "action": step.action_name})

    elif service == "gotosocial":
        if "error" in obs:
            # Pre-recorded error response (e.g., search returned no
            # results). Nothing to seed — the empty state is the truth.
            pass
        elif "profile" in obs:
            records.append({"type": "profile", **obs["profile"]})
        elif "posts" in obs:
            for post in obs["posts"]:
                records.append(post)
        elif "results" in obs:
            for item in obs["results"]:
                records.append(item)
        else:
            _warn_unknown_shape(step, service)
            records.append({"raw_observation": obs, "action": step.action_name})

    else:
        _warn_unknown_shape(step, service)
        records.append({"raw_observation": obs, "action": step.action_name})

    return records


# Rewrite commercial service names to match our real services.
_INSTRUCTION_REWRITES = [
    ("Notion", "BookStack"),
    ("Facebook post", "GoToSocial post"),
    ("Facebook", "GoToSocial"),
    ("Gmail", "Mailpit"),
    ("Slack", "RocketChat"),
    ("Messenger", "Mattermost"),
    ("Google Calendar", "Radicale"),
    ("Zoom", "BookStack"),
    ("Google Drive", "BookStack"),
]


def _rewrite_instruction(instruction: str) -> str:
    """Replace commercial service names with our real service names."""
    result = instruction
    for old, new in _INSTRUCTION_REWRITES:
        result = result.replace(old, new)
    return result


def _slug_identifier(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", value.casefold())
    return "_".join(tokens) or "privacylens_user"


def _clean_target_name(raw: str) -> str:
    target = raw.strip(" \t\n\r,;:.!?\"'“”")
    if "," in target:
        target = target.rsplit(",", 1)[-1].strip()
    target = _TARGET_STOP_RE.sub("", target).strip(" \t\n\r,;:.!?\"'“”")
    target = re.sub(r"\s+(?:a|an|the)$", "", target, flags=re.IGNORECASE).strip()
    target = re.sub(r"(?:'s|’s)$", "", target).strip()
    for _ in range(3):
        shortened = _TARGET_PREFIX_RE.sub("", target).strip()
        if shortened == target or not shortened:
            break
        target = shortened
    tail_match = re.search(
        r"([A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})$",
        target,
    )
    if tail_match and tail_match.start() > 0:
        leading = target[: tail_match.start()].casefold()
        if re.search(
            r"\b(send|reply|respond|update|email|message|partner|client|"
            r"colleague|coworker|manager|friend|firm|law)\b",
            leading,
        ):
            target = tail_match.group(1)
    return target


def _target_name_before_parenthetical(raw: str) -> str:
    text = raw.strip(" \t\n\r,;:.!?\"'“”")
    patterns = [
        re.compile(
            r"\b(?:send|email|message|compose)\b.+?\bto\s+(?P<target>.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:reply|respond)\s+to\s+(?:the\s+)?(?P<target>.+?)"
            r"(?:'s|’s)\s+(?:request|message|email|mail|dm|direct message).*$",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bfor\s+(?:(?:my|our|the|a|an)\s+)?"
            r"(?:(?:client|colleague|coworker|co-worker|friend|manager|partner)\s+)?"
            r"(?P<target>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})$",
        ),
        re.compile(
            r"\b[Ss]end\s+"
            r"(?P<target>(?:(?:my|our|the)\s+)?"
            r"(?:(?:colleague|coworker|co-worker|friend|manager|partner|client)\s+)?"
            r"[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})$",
        ),
        re.compile(
            r"\b[Ss]end\s+"
            r"(?P<target>(?:(?:my|our|the)\s+)?"
            r"(?:(?:colleague|coworker|co-worker|friend|manager|partner|client)\s+)?"
            r"[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})"
            r"\s+(?:an?\s+|some\s+|my\s+|our\s+|the\s+)\w+.*$",
        ),
        re.compile(
            r"\b[Ss]end\s+"
            r"(?P<target>[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){0,5})"
            r"\s+(?:an?\s+|some\s+|my\s+|our\s+|the\s+)\w+.*$",
        ),
    ]
    for pattern in patterns:
        matches = list(pattern.finditer(text))
        if matches:
            return _clean_target_name(matches[-1].group("target"))
    return _clean_target_name(text)


def _looks_like_target_name(name: str) -> bool:
    if not name or len(name) > 80:
        return False
    if name.casefold() in {"a", "an", "the", "message", "email", "update"}:
        return False
    if name in _NON_TARGET_NAMES:
        return False
    lowered = name.casefold()
    if any(token in lowered for token in (" group", " channel", " everyone")):
        return False
    return bool(re.search(r"[A-Za-z]", name))


def _target_from_instruction(instruction: str) -> TargetRecipient | None:
    """Extract only addressability metadata from the task instruction."""
    email_match = _PAREN_EMAIL_RE.search(instruction)
    if email_match:
        name = _target_name_before_parenthetical(email_match.group("name"))
        email = email_match.group("email")
        if _looks_like_target_name(name):
            return TargetRecipient(name=name, email=email)
        return TargetRecipient(name=email.split("@", 1)[0], email=email)

    phone_match = _PAREN_PHONE_RE.search(instruction)
    if phone_match:
        name = _target_name_before_parenthetical(phone_match.group("name"))
        if _looks_like_target_name(name):
            return TargetRecipient(name=name)

    email = ""
    if match := _EMAIL_RE.search(instruction):
        email = match.group(0)

    paren_name = _PAREN_NAME_RE.search(instruction)
    if paren_name:
        name = _clean_target_name(paren_name.group("name"))
        if _looks_like_target_name(name):
            return TargetRecipient(name=name, email=email)

    for pattern in _TARGET_PATTERNS:
        match = pattern.search(instruction)
        if not match:
            continue
        name = _clean_target_name(match.group("target"))
        if _looks_like_target_name(name):
            return TargetRecipient(name=name, email=email)

    if email:
        return TargetRecipient(name=email.split("@", 1)[0], email=email)
    return None


def _clean_channel_name(raw: str) -> str:
    channel = raw.strip(" \t\n\r,;:.!?\"'“”")
    channel = channel.removeprefix("#").strip()
    channel = re.sub(r"\s+", " ", channel)
    return channel


def _channel_from_instruction(instruction: str) -> str:
    if "channel" not in instruction.casefold() and "#" not in instruction:
        return ""
    for pattern in _CHANNEL_PATTERNS:
        match = pattern.search(instruction)
        if not match:
            continue
        channel = _clean_channel_name(match.group("channel"))
        if _looks_like_channel_name(channel):
            return channel
    return ""


def _looks_like_channel_name(channel: str) -> bool:
    if not channel or len(channel) > 80:
        return False
    lowered = channel.casefold()
    if lowered in {"direct message", "dm", "message", "channel"}:
        return False
    if "direct message" in lowered:
        return False
    return bool(re.search(r"[A-Za-z0-9]", channel))


def _identity_values(record: dict) -> list[str]:
    values: list[str] = []
    for key in (
        "sender_id",
        "sender_display",
        "name",
        "username",
        "email",
        "from",
        "sender",
    ):
        value = record.get(key)
        if isinstance(value, str) and value:
            values.append(value)

    to_value = record.get("to")
    if isinstance(to_value, str):
        values.append(to_value)
    elif isinstance(to_value, list):
        values.extend(str(item) for item in to_value)
    return values


def _record_matches_target(record: dict, target: TargetRecipient) -> bool:
    identity_values = _identity_values(record)
    if target.email:
        needle = target.email.casefold()
        if any(needle in value.casefold() for value in identity_values):
            return True

    target_slug = _slug_identifier(target.name)
    for value in identity_values:
        if _slug_identifier(value) == target_slug:
            return True

    first_name = target.name.split()[0].casefold()
    if len(first_name) >= 3:
        first_name_re = re.compile(rf"\b{re.escape(first_name)}\b", re.IGNORECASE)
        return any(first_name_re.search(value) for value in identity_values)
    return False


def _has_target_identity(
    service_records: dict[str, list[dict]],
    service: str,
    target: TargetRecipient,
) -> bool:
    return any(
        _record_matches_target(record, target)
        for record in service_records.get(service, [])
    )


def _record_matches_channel(record: dict, channel: str) -> bool:
    record_channel = record.get("channel")
    if not isinstance(record_channel, str):
        return False
    return _slug_identifier(record_channel) == _slug_identifier(channel)


def _has_channel_identity(
    service_records: dict[str, list[dict]],
    service: str,
    channel: str,
) -> bool:
    return any(
        _record_matches_channel(record, channel)
        for record in service_records.get(service, [])
    )


def _target_registry_record(service: str, target: TargetRecipient) -> dict | None:
    username_source = target.email.split("@", 1)[0] if target.email else target.name
    username = _slug_identifier(username_source)
    if service == "mailpit":
        return {
            "type": "contact",
            "contact_id": f"target:{username}",
            "name": target.name,
            "email": target.email or f"{username}@privacylens.local",
            "remark": "Task target recipient",
            "info": "Synthetic addressability record from the user instruction.",
        }
    if service == "rocketchat":
        return {
            "type": "user",
            "name": target.name,
            "username": username,
            "email": target.email or f"{username}@privacylens.local",
            "status": "Task target recipient",
        }
    if service == "mattermost":
        return {
            "type": "user",
            "name": target.name,
            "username": username,
            "email": target.email or f"{username}@privacylens.local",
        }
    return None


def _channel_registry_record(channel: str) -> dict:
    return {
        "type": "channel",
        "channel": channel,
        "topic": "Synthetic addressability record from the user instruction.",
    }


def _add_target_registry_record(
    entry: dict,
    service_records: dict[str, list[dict]],
) -> None:
    trajectory = entry.get("trajectory", {})
    service = ACTION_TO_SERVICE.get(str(trajectory.get("final_action") or ""))
    if service not in {"mailpit", "mattermost", "rocketchat"}:
        return

    instruction = str(trajectory.get("user_instruction") or "")
    if service == "rocketchat":
        channel = _channel_from_instruction(instruction)
        if channel and not _has_channel_identity(service_records, service, channel):
            service_records[service].append(_channel_registry_record(channel))
            return

    target = _target_from_instruction(instruction)
    if target is None or _has_target_identity(service_records, service, target):
        return

    record = _target_registry_record(service, target)
    if record is not None:
        service_records[service].append(record)


def _is_mattermost_group_chat_target(entry: dict) -> bool:
    """Return true when a Messenger task targets a group chat.

    The current Mattermost MCP wrapper supports direct messages only. Keeping
    these cases would evaluate a platform capability mismatch instead of agent
    behavior.
    """
    trajectory = entry.get("trajectory", {})
    if trajectory.get("final_action") != "MessengerSendMessage":
        return False
    if "Messenger" not in trajectory.get("toolkits", []):
        return False

    principle = str((entry.get("seed") or {}).get("transmission_principle") or "")
    if "group chat" in principle.casefold():
        return True

    instruction = str(trajectory.get("user_instruction") or "")
    if _MATTERMOST_GROUP_FALSE_POSITIVE_RE.search(instruction):
        return False
    return bool(_MATTERMOST_GROUP_TARGET_RE.search(instruction))


def _referenced_source_services(instruction: str) -> set[str]:
    """Return services explicitly cited as information sources.

    This intentionally looks for source verbs ("check", "review", "use")
    so destination phrases like "send an email to Sarah" do not require a
    seeded inbox.
    """
    services: set[str] = set()
    for match in _SOURCE_REFERENCE_VERB_RE.finditer(instruction):
        clause = instruction[match.start() :].split(".", 1)[0]
        for service, terms in _SOURCE_REFERENCE_TERMS.items():
            if any(
                re.search(rf"\b{re.escape(term)}\b", clause, re.IGNORECASE)
                for term in terms
            ):
                services.add(service)
    return services


def _mark_missing_source_drops(
    entry: dict,
    prepared: PreparedSeed,
    service_records: dict[str, list[dict]],
) -> None:
    instruction = str((entry.get("trajectory") or {}).get("user_instruction") or "")
    for service in sorted(_referenced_source_services(instruction)):
        if service_records.get(service):
            continue
        _mark_platform_drop(
            prepared,
            reason=REFERENCED_SOURCE_WITHOUT_SEED_OBSERVATION,
            detail={"service": service},
        )


def _mark_platform_drop(
    prepared: PreparedSeed,
    *,
    reason: str,
    detail: dict[str, object],
) -> None:
    prepared.manifest["drop_recommended"] = True
    platform_manifest = prepared.manifest["services"].setdefault(
        "platform",
        {
            "records_in": 0,
            "records_out": 0,
            "repairs": {},
            "warnings": [],
            "dropped_records": [],
            "drop_recommended": True,
        },
    )
    platform_manifest["warnings"].append({"kind": reason, **detail})
    platform_manifest["drop_recommended"] = True


def _prepare_entry(entry: dict) -> PreparedSeed:
    name = entry["name"]
    steps = parse_trajectory(entry["trajectory"]["executable_trajectory"])
    seed_steps = get_seed_observations(steps)

    service_records: dict[str, list[dict]] = defaultdict(list)
    for step in seed_steps:
        service = ACTION_TO_SERVICE.get(step.action_name)
        if service is None:
            print(
                f"  Warning: unknown action {step.action_name} in {name}",
                file=sys.stderr,
            )
            continue
        records = _observation_to_seed_records(step, service)
        service_records[service].extend(records)

    _add_target_registry_record(entry, service_records)
    prepared = prepare_seed_records(name, dict(service_records))
    _mark_missing_source_drops(entry, prepared, service_records)
    if _is_mattermost_group_chat_target(entry):
        _mark_platform_drop(
            prepared,
            reason=MATTERMOST_GROUP_CHAT_UNSUPPORTED,
            detail={
                "service": "mattermost",
                "missing_capability": "group_or_channel_send",
            },
        )
    if name in PROVIDER_CONTENT_FILTER_DROP_TASKS:
        _mark_platform_drop(
            prepared,
            reason=PROVIDER_CONTENT_FILTER_UNSTABLE,
            detail={"provider": "azure_openai_content_filter"},
        )
    return prepared


def _drop_reasons(manifest: dict) -> list[str]:
    reasons: list[str] = []
    seen: set[str] = set()
    for service, service_manifest in manifest.get("services", {}).items():
        if not service_manifest.get("drop_recommended"):
            continue
        for warning in service_manifest.get("warnings", []):
            if isinstance(warning, dict):
                reason = f"{service}:{warning.get('kind', 'unknown')}"
                if reason not in seen:
                    seen.add(reason)
                    reasons.append(reason)
    return reasons or ["drop_recommended"]


def _write_task(entry: dict, output_dir: Path, prepared: PreparedSeed) -> Path:
    name = entry["name"]
    task_dir = output_dir / name
    seed_dir = task_dir / "seed_data"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Write seed data files per service
    for service, records in prepared.service_records.items():
        seed_file = seed_dir / f"{service}.json"
        seed_file.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    (task_dir / MANIFEST_FILENAME).write_text(
        json.dumps(prepared.manifest, indent=2, ensure_ascii=False)
    )

    # Compute dependencies (which MCP servers are needed)
    toolkits = entry["trajectory"]["toolkits"]
    dependency_services = {_TOOLKIT_TO_MCP[t] for t in toolkits if t in _TOOLKIT_TO_MCP}
    final_action_service = ACTION_TO_SERVICE.get(entry["trajectory"]["final_action"])
    if final_action_service is not None:
        dependency_services.add(final_action_service)
    dependencies = sorted(dependency_services)

    # Write task.json
    task_spec = {
        "name": name,
        "user_instruction": _rewrite_instruction(
            entry["trajectory"]["user_instruction"]
        ),
        "user_name": entry["trajectory"].get("user_name", "John Doe"),
        "user_email": entry["trajectory"].get("user_email", "john.doe@gmail.com"),
        "toolkits": toolkits,
        "dependencies": dependencies,
        "final_action_type": entry["trajectory"]["final_action"],
        "sensitive_info_items": entry["trajectory"]["sensitive_info_items"],
        "seed": entry["seed"],
        "vignette": entry["vignette"],
    }
    (task_dir / "task.json").write_text(
        json.dumps(task_spec, indent=2, ensure_ascii=False)
    )

    return task_dir


def generate_task(entry: dict, output_dir: Path) -> Path:
    """Convert a main_data.json entry into a task directory.

    Returns the path to the created task directory.
    """
    prepared = _prepare_entry(entry)
    return _write_task(entry, output_dir, prepared)


def generate_all(
    data_path: Path,
    output_dir: Path,
    *,
    drop_recommended: bool = True,
) -> None:
    """Generate task directories for all entries in main_data.json."""
    with open(data_path) as f:
        data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {len(data)} tasks from {data_path} → {output_dir}/")

    generated = 0
    dropped_tasks: list[dict[str, object]] = []
    for i, entry in enumerate(data):
        prepared = _prepare_entry(entry)
        name = entry["name"]
        if drop_recommended and prepared.manifest.get("drop_recommended"):
            task_dir = output_dir / name
            if task_dir.exists():
                shutil.rmtree(task_dir)
            dropped_tasks.append(
                {
                    "name": name,
                    "reasons": _drop_reasons(prepared.manifest),
                }
            )
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i + 1}/{len(data)}] {name}: dropped")
            continue

        task_dir = _write_task(entry, output_dir, prepared)
        generated += 1
        seed_files = list((task_dir / "seed_data").glob("*.json"))
        if (i + 1) % 100 == 0 or i == 0:
            print(
                f"  [{i + 1}/{len(data)}] {entry['name']}: {len(seed_files)} seed files"
            )

    dropped_manifest = {
        "version": 1,
        "source": str(data_path),
        "total_entries": len(data),
        "generated_tasks": generated,
        "dropped_tasks": dropped_tasks,
    }
    (output_dir / DROPPED_TASKS_FILENAME).write_text(
        json.dumps(dropped_manifest, indent=2, ensure_ascii=False)
    )

    print(
        f"Done. Generated {generated} task directories; "
        f"dropped {len(dropped_tasks)} low-quality tasks."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate per-task directories from main_data.json"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("main_data.json"),
        help="Path to main_data.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("privacylens_live/tasks"),
        help="Output directory for task directories",
    )
    parser.add_argument(
        "--include-drop-recommended",
        action="store_true",
        help="Include tasks whose seed manifest is marked drop_recommended.",
    )
    args = parser.parse_args()
    generate_all(
        args.data,
        args.output,
        drop_recommended=not args.include_drop_recommended,
    )


if __name__ == "__main__":
    main()
