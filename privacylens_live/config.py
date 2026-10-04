"""Central configuration for PrivacyLens-Live.

Single source of truth for service credentials, URLs, and the LLM endpoint.

Token management
----------------
Local demonstration credentials are defined here as Pydantic field defaults.
Provisioned tokens belong in environment variables or an ignored ``.env`` file.
Service settings are written to ``privacylens_live/.env`` by
:meth:`Config.write_env_file` so that ``docker compose`` and the host-side
seeder/runner share one authoritative copy. ``Config.from_env`` reads ``.env``
(if present) plus the process environment.

Tokens that must be provisioned at runtime against a fresh service instance
(currently ``gotosocial_token``) start as empty strings and are filled by
:mod:`privacylens_live.bootstrap` after the first ``docker compose up``. The
bootstrap step then re-writes ``.env`` so subsequent runs are zero-touch.

LLM API keys have no defaults. They can be supplied through the process
environment or an ignored ``.env`` file. ``write_env_file`` preserves existing
user settings but never copies LLM API keys from the process into that file.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


logger = logging.getLogger("config")


# Providers litellm reaches directly. A base_url is meaningful for these only
# when it is that provider's own endpoint, so a gateway URL is a
# misconfiguration rather than an override. Module-level rather than a class
# attribute: pydantic turns a leading-underscore class attribute into a
# ModelPrivateAttr, which is not the tuple the check needs.
NATIVE_PROVIDERS = ("gemini", "vertex_ai", "anthropic", "bedrock")


class Config(BaseModel):
    """Configuration for PrivacyLens-Live platform."""

    # ── BookStack (wiki / knowledge base) ──
    bookstack_url: str = "http://localhost:3000"
    # Create an API token in your local BookStack instance and set these in
    # .env or the process environment. Tokens are tied to that instance's DB.
    bookstack_token_id: str = ""
    bookstack_token_secret: str = ""

    # ── Mattermost (direct messaging) ──
    mattermost_url: str = "http://localhost:8065"
    mattermost_user: str = "admin"
    mattermost_password: str = "Admin123!"
    mattermost_email: str = "admin@privacylens.local"
    mattermost_team: str = "privacylens"

    # ── RocketChat (team chat / channels) ──
    rocketchat_url: str = "http://localhost:3100"
    rocketchat_user: str = "admin"
    rocketchat_password: str = "admin"

    # ── Mailpit (email) ──
    mailpit_api_url: str = "http://localhost:8025"
    mailpit_smtp_host: str = "localhost"
    mailpit_smtp_port: int = 1025

    # ── GoToSocial (social media) ──
    gotosocial_url: str = "http://localhost:4000"
    gotosocial_user: str = "privacylens"
    # GoToSocial enforces a 100% strength check on passwords; the
    # earlier ``Admin123!`` was only 80% and got rejected by the
    # admin CLI on first-time provisioning.
    gotosocial_password: str = "PrivacyLens-Admin-2026!@#"
    gotosocial_email: str = "privacylens@privacylens.local"
    # Empty by default — populated by bootstrap_gotosocial against a fresh
    # gotosocial-data volume, then persisted to .env.
    gotosocial_token: str = ""

    # ── Radicale (calendar) ──
    radicale_url: str = "http://localhost:5232"
    radicale_user: str = "admin"
    radicale_password: str = "admin"

    # ── Google Drive artifact backend (host-side path for the seeder) ──
    google_drive_artifact_root: Path = Path(
        ".agent_tmp/privacylens_artifacts/google_drive"
    )

    # ── MCP server URLs (Docker-internal, for agent container) ──
    mcp_server_urls: dict[str, str] = Field(
        default_factory=lambda: {
            "bookstack": "http://bookstack-mcp:8080/mcp",
            "mattermost": "http://mattermost-mcp:8080/mcp",
            "rocketchat": "http://rocketchat-mcp:8080/mcp",
            "mailpit": "http://mailpit-mcp:8080/mcp",
            "gotosocial": "http://gotosocial-mcp:8080/mcp",
            "radicale": "http://radicale-mcp:8080/mcp",
            "google_drive": "http://google-drive-mcp:8080/mcp",
        }
    )

    # ── Agent LLM config ──
    llm_model: str = "openai/gpt-5.4"
    llm_api_key: str = ""
    # Empty URLs use the selected provider's default endpoint.
    llm_base_url: str = ""
    llm_api_version: str = "2025-03-01-preview"
    max_iterations: int = 20

    # ── Extraction LLM — Grok-4-20-non-reasoning for privacy flow extraction ──
    extraction_llm_model: str = "openai/grok-4-20-non-reasoning"
    extraction_llm_api_key: str = ""
    extraction_llm_base_url: str = ""

    # ── Evaluation LLM (same model by default) ──
    eval_model: str = "openai/gpt-5.4"

    # ── Paths ──
    data_path: Path = Path("main_data.json")
    tasks_dir: Path = Path("privacylens_live/tasks")
    results_dir: Path = Path("results")

    # ── Docker ──
    docker_network: str = "privacylens_live_privacylens-net"
    agent_server_image: str = "privacylens-agent-server:local"

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Config:
        """Create config from environment variables, layered over defaults.

        Resolution order, lowest to highest precedence:

        1. Pydantic field defaults (this class).
        2. The ``privacylens_live/.env`` file (if it exists), parsed and
           merged into ``os.environ``. This is where bootstrap-provisioned
           tokens and any user overrides live.
        3. The process environment (``os.environ``), so an explicit
           ``BOOKSTACK_TOKEN_ID=foo python -m privacylens_live run`` still
           wins.

        LLM API keys have no fallback defaults. Existing keys explicitly
        stored in the ignored ``.env`` file follow the same precedence.
        """
        if env_file is None:
            env_file = ENV_FILE_PATH
        if env_file.exists():
            load_env_file(env_file)

        # Build kwargs only for env-overridable fields. Anything not present
        # in the environment falls through to the class default. Typed as
        # Any because Pydantic field types are heterogeneous (str, int) and
        # pyright can't track them through a dict construction.
        overrides: dict[str, Any] = {}

        def _str(field: str, env: str) -> None:
            val = os.environ.get(env)
            if val is not None and val != "":
                overrides[field] = val

        _str("bookstack_url", "BOOKSTACK_URL")
        _str("bookstack_token_id", "BOOKSTACK_TOKEN_ID")
        _str("bookstack_token_secret", "BOOKSTACK_TOKEN_SECRET")
        _str("mattermost_url", "MATTERMOST_URL")
        _str("mattermost_user", "MATTERMOST_USER")
        _str("mattermost_password", "MATTERMOST_PASSWORD")
        _str("mattermost_email", "MATTERMOST_EMAIL")
        _str("mattermost_team", "MATTERMOST_TEAM")
        _str("rocketchat_url", "ROCKETCHAT_URL")
        _str("rocketchat_user", "ROCKETCHAT_USER")
        _str("rocketchat_password", "ROCKETCHAT_PASSWORD")
        _str("mailpit_api_url", "MAILPIT_API_URL")
        _str("mailpit_smtp_host", "MAILPIT_SMTP_HOST")
        port = os.environ.get("MAILPIT_SMTP_PORT")
        if port:
            overrides["mailpit_smtp_port"] = int(port)
        _str("gotosocial_url", "GOTOSOCIAL_URL")
        _str("gotosocial_user", "GOTOSOCIAL_USER")
        _str("gotosocial_password", "GOTOSOCIAL_PASSWORD")
        _str("gotosocial_email", "GOTOSOCIAL_EMAIL")
        _str("gotosocial_token", "GOTOSOCIAL_TOKEN")
        _str("radicale_url", "RADICALE_URL")
        _str("radicale_user", "RADICALE_USER")
        _str("radicale_password", "RADICALE_PASSWORD")
        _str("google_drive_artifact_root", "GOOGLE_DRIVE_ARTIFACT_ROOT")
        _str("docker_network", "DOCKER_NETWORK")
        _str("agent_server_image", "AGENT_SERVER_IMAGE")

        # Prefer PrivacyLens-specific names, with OPENAI_API_KEY as a fallback.
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        overrides["llm_api_key"] = os.environ.get("LLM_API_KEY") or openai_api_key

        # Both LLMs accept the existing aliases for a provider's own endpoint.
        for field in ("llm_base_url", "extraction_llm_base_url"):
            base_url = os.environ.get(field.upper())
            if base_url is not None:
                overrides[field] = (
                    ""
                    if base_url.strip().lower() in {"none", "-", "direct"}
                    else base_url
                )
        llm_model = os.environ.get("LLM_MODEL")
        if llm_model:
            # Accept either bare model name (e.g. ``gpt-5.4``) or a
            # litellm-style prefixed form. The Foundry /openai/v1 endpoint
            # is OpenAI-compatible, so bare names should use the openai
            # provider there; legacy Azure endpoints still use azure.
            if "/" not in llm_model:
                base_url = overrides.get(
                    "llm_base_url", cls.model_fields["llm_base_url"].default
                )
                provider = (
                    "openai"
                    if not base_url or str(base_url).rstrip("/").endswith("/v1")
                    else "azure"
                )
                llm_model = f"{provider}/{llm_model}"
            overrides["llm_model"] = llm_model
        llm_api_version = os.environ.get("LLM_API_VERSION")
        if llm_api_version:
            overrides["llm_api_version"] = llm_api_version

        # Extraction LLM (privacy analyzer) — separate credentials
        extraction_key = (
            os.environ.get("EXTRACTION_LLM_API_KEY") or overrides["llm_api_key"]
        )
        if extraction_key:
            overrides["extraction_llm_api_key"] = extraction_key
        extraction_model = os.environ.get("EXTRACTION_LLM_MODEL")
        if extraction_model:
            overrides["extraction_llm_model"] = extraction_model

        cls._check_routing(overrides)
        return cls(**overrides)

    @classmethod
    def _check_routing(cls, overrides: dict[str, object]) -> None:
        """Refuse to send a natively-routed model's traffic to a gateway.

        Silently misrouting is the failure this guards: the request reaches the
        gateway, fails or is served by a different model entirely, and the run
        completes with results attributed to a model that never ran. Because
        every result records only the model *name*, that would be invisible
        afterwards.
        """
        for model_key, base_key, label in (
            ("llm_model", "llm_base_url", "execution"),
            ("extraction_llm_model", "extraction_llm_base_url", "audit"),
        ):
            model = str(overrides.get(model_key) or cls.model_fields[model_key].default)
            base = overrides.get(base_key)
            if base is None:
                base = cls.model_fields[base_key].default
            base = str(base)
            provider = model.split("/", 1)[0].lower()
            if provider not in NATIVE_PROVIDERS or not base:
                continue
            if provider not in base.lower():
                raise ValueError(
                    f"{label} model {model!r} is routed natively by litellm, but "
                    f"{base_key.upper()} is set to {base!r}, which would send its "
                    f"traffic to that gateway. Set the matching *_BASE_URL to "
                    f"'none' to use the provider's own endpoint."
                )

    def write_env_file(self, path: Path | None = None) -> Path:
        """Write the docker-compose ``.env`` file from this config.

        The .env file is the bridge between Python-side state (this class)
        and docker-compose-side state (the MCP server containers' env vars).
        Anything that needs to flow from one to the other goes through here.

        Existing settings outside the generated service fields are preserved,
        including LLM configuration explicitly stored by the user. LLM keys
        from the process environment are never added to the file.
        """
        if path is None:
            path = ENV_FILE_PATH
        drive_root = self.google_drive_artifact_root
        drive_host_root = (
            drive_root if drive_root.is_absolute() else Path("..") / drive_root
        )
        lines = [
            "# Service settings generated by Config.write_env_file.",
            "# Edit .env or export environment variables to override defaults.",
            "# Run `python -m privacylens_live setup` to apply changes.",
            "#",
            "# The *_URL values here are HOST-side (localhost:port) and used by",
            "# the Python seeder + runner. The MCP server containers use",
            "# Docker-internal URLs hardcoded in docker-compose.yml — those are",
            "# NOT pulled from this file.",
            "",
            f"BOOKSTACK_URL={self.bookstack_url}",
            f"BOOKSTACK_TOKEN_ID={self.bookstack_token_id}",
            f"BOOKSTACK_TOKEN_SECRET={self.bookstack_token_secret}",
            "",
            f"MATTERMOST_URL={self.mattermost_url}",
            f"MATTERMOST_USER={self.mattermost_user}",
            f"MATTERMOST_PASSWORD={self.mattermost_password}",
            f"MATTERMOST_EMAIL={self.mattermost_email}",
            f"MATTERMOST_TEAM={self.mattermost_team}",
            "",
            f"ROCKETCHAT_URL={self.rocketchat_url}",
            f"ROCKETCHAT_USER={self.rocketchat_user}",
            f"ROCKETCHAT_PASSWORD={self.rocketchat_password}",
            "",
            f"MAILPIT_API_URL={self.mailpit_api_url}",
            f"MAILPIT_SMTP_HOST={self.mailpit_smtp_host}",
            f"MAILPIT_SMTP_PORT={self.mailpit_smtp_port}",
            "",
            f"GOTOSOCIAL_URL={self.gotosocial_url}",
            f"GOTOSOCIAL_USER={self.gotosocial_user}",
            f"GOTOSOCIAL_PASSWORD={self.gotosocial_password}",
            f"GOTOSOCIAL_EMAIL={self.gotosocial_email}",
            f"GOTOSOCIAL_TOKEN={self.gotosocial_token}",
            "",
            f"RADICALE_URL={self.radicale_url}",
            f"RADICALE_USER={self.radicale_user}",
            f"RADICALE_PASSWORD={self.radicale_password}",
            "",
            f"GOOGLE_DRIVE_ARTIFACT_ROOT={drive_root}",
            f"GOOGLE_DRIVE_ARTIFACT_HOST_ROOT={drive_host_root}",
            "",
        ]
        if path.exists():
            generated_keys = {line.partition("=")[0] for line in lines if "=" in line}
            preserved = [
                line
                for line in path.read_text().splitlines()
                if "=" in line
                and not line.lstrip().startswith("#")
                and line.partition("=")[0].strip() not in generated_keys
            ]
            if preserved:
                lines.extend(["# Additional user settings.", *preserved, ""])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))
        logger.info("Wrote %s", path)
        return path


# ── Module-level helpers ──

ENV_FILE_PATH = Path(__file__).parent / ".env"


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a ``.env`` file and merge it into ``os.environ``.

    Tiny dependency-free parser: ``KEY=VALUE`` per line, ``#`` starts a
    comment, blank lines ignored, no escaping / interpolation. Surrounding
    single or double quotes on the value are stripped (matching the
    convention used by python-dotenv and docker-compose) so lines like
    ``LLM_API_KEY = "abc123"`` parse to ``LLM_API_KEY=abc123``.
    Existing entries in ``os.environ`` win, so the process environment
    overrides ``.env`` (matching docker-compose's own behavior).
    """
    parsed: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Strip a single layer of matching surrounding quotes.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        parsed[key] = value
        if key not in os.environ:
            os.environ[key] = value
    return parsed


# ── PrivacyLens action name → service mapping ──
# Used ONLY by task generator to route seed data to service-specific files.
# Maps original PrivacyLens action names to seed data file stems.
ACTION_TO_SERVICE: dict[str, str] = {
    "NotionManagerSearchContent": "bookstack",
    "NotionManagerReadPage": "bookstack",
    "NotionManagerCreatePage": "bookstack",
    "NotionManagerEditPage": "bookstack",
    "NotionManagerGetAllContent": "bookstack",
    "NotionManagerSharePage": "bookstack",
    "MessengerReceiveMessage": "mattermost",
    "MessengerSearchInChat": "mattermost",
    "MessengerSendMessage": "mattermost",
    "MessengerSend": "mattermost",
    "GmailSendEmail": "mailpit",
    "GmailSearchEmails": "mailpit",
    "GmailReadEmail": "mailpit",
    "GmailSearchContacts": "mailpit",
    "SlackSendMessage": "rocketchat",
    "SlackSearchMessage": "rocketchat",
    "SlackSearchChannelByUser": "rocketchat",
    "SlackSearchChannelOrUser": "rocketchat",
    "SlackGetUserDetails": "rocketchat",
    "FacebookManagerCreatePost": "gotosocial",
    "FacebookManagerGetUserProfile": "gotosocial",
    "FacebookManagerGetUserPosts": "gotosocial",
    "FacebookManagerSearchFriends": "gotosocial",
    "FacebookManagerSearchPosts": "gotosocial",
    "FacebookManagerGetUserFriends": "gotosocial",
    "FacebookManagerUpdateStatus": "gotosocial",
    "GoogleCalendarSearchEvents": "radicale",
    "GoogleCalendarReadEvents": "radicale",
    "ZoomManagerGetMeetingTranscript": "bookstack",
    "ZoomManagerSearchMeetings": "bookstack",
    "ZoomManagerSearchTranscript": "bookstack",
    "ZoomManagerGetParticipants": "bookstack",
    "ZoomManagerSearchMeetingTranscript": "bookstack",
    "ZoomManagerGetMeetingTranscripts": "bookstack",
}
