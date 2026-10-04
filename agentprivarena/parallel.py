"""Parallel launcher for isolated AgentPrivArena evaluation stacks.

Each selected privacy mitigation level gets its own docker-compose project
(separate containers, volumes, networks) so evaluation batches can run in
parallel without seed-data cross-contamination.

Usage:
    # Start selected stacks, bootstrap, and run evaluations:
    uv run python -m agentprivarena.parallel

    # Skip L1 and avoid default stack ports:
    uv run python -m agentprivarena.parallel --levels l0,l2,l3 --base-offset 2000

    # Teardown selected stacks:
    uv run python -m agentprivarena.parallel \
        --levels l0,l2,l3 --base-offset 2000 --teardown
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

from agentprivarena.config import Config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("parallel")

COMPOSE_FILE = Path(__file__).parent / "docker-compose.yml"

# Level definitions: variant_name -> stable level index.
LEVEL_INDICES = {
    "baseline": 0,
    "privacy_enhanced": 1,
    "ci_reasoning": 2,
    "ci_audit": 3,
}

LEVEL_LABELS = {
    "baseline": "L0_baseline",
    "privacy_enhanced": "L1_privacy",
    "ci_reasoning": "L2_ci_reasoning",
    "ci_audit": "L3_ci_audit",
}

LEVEL_ALIASES = {
    "l0": "baseline",
    "l1": "privacy_enhanced",
    "l2": "ci_reasoning",
    "l3": "ci_audit",
}

DEFAULT_LEVELS = tuple(LEVEL_INDICES)

# Services that must be healthy before bootstrap can run.
INFRA_SERVICES = [
    "bookstack-db",
    "bookstack",
    "mattermost-db",
    "mattermost",
    "mongo",
    "rocketchat",
    "mailpit",
    "gotosocial",
    "radicale",
]

MCP_SERVICES = [
    "bookstack-mcp",
    "mattermost-mcp",
    "rocketchat-mcp",
    "mailpit-mcp",
    "gotosocial-mcp",
    "radicale-mcp",
    "google-drive-mcp",
]

# Base ports for the default (stack 0) configuration.
BASE_PORTS = {
    "BOOKSTACK_HOST_PORT": 3000,
    "MATTERMOST_HOST_PORT": 8065,
    "ROCKETCHAT_HOST_PORT": 3100,
    "MAILPIT_HTTP_HOST_PORT": 8025,
    "MAILPIT_SMTP_HOST_PORT": 1025,
    "GOTOSOCIAL_HOST_PORT": 4000,
    "RADICALE_HOST_PORT": 5232,
    "BOOKSTACK_MCP_HOST_PORT": 9001,
    "MATTERMOST_MCP_HOST_PORT": 9002,
    "ROCKETCHAT_MCP_HOST_PORT": 9003,
    "MAILPIT_MCP_HOST_PORT": 9004,
    "GOTOSOCIAL_MCP_HOST_PORT": 9005,
    "RADICALE_MCP_HOST_PORT": 9006,
    "GOOGLE_DRIVE_MCP_HOST_PORT": 9007,
}


def _project_name(variant: str) -> str:
    return f"agentprivarena_{variant}"


def _normalize_level(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("Empty level name")
    normalized = LEVEL_ALIASES.get(normalized, normalized)
    if normalized not in LEVEL_INDICES:
        valid = ", ".join([*LEVEL_ALIASES, *LEVEL_INDICES])
        raise ValueError(f"Unknown level {value!r}; expected one of: {valid}")
    return normalized


def _selected_levels(levels: str, base_offset: int) -> list[tuple[str, int, int]]:
    variants = [_normalize_level(level) for level in levels.split(",")]
    selected: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        index = LEVEL_INDICES[variant]
        selected.append((variant, index, base_offset + index * 10))
    return selected


def _port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


def _check_ports_available(levels: list[tuple[str, int, int]]) -> None:
    assigned_ports: dict[int, str] = {}
    conflicts: list[str] = []

    for variant, _idx, offset in levels:
        for var, base_port in BASE_PORTS.items():
            port = base_port + offset
            label = f"{variant}:{var}"
            previous = assigned_ports.get(port)
            if previous is not None:
                conflicts.append(f"{port} is assigned to both {previous} and {label}")
                continue
            assigned_ports[port] = label

    for port, label in assigned_ports.items():
        if not _port_available(port):
            conflicts.append(f"{port} is already in use for {label}")

    if conflicts:
        details = "\n  - ".join(conflicts)
        raise RuntimeError(
            "Selected stacks have host-port conflicts:\n"
            f"  - {details}\n"
            "Use a larger --base-offset, for example 2000, or tear down the "
            "conflicting stack first."
        )


def _compose(
    variant: str, *args: str, check: bool = True
) -> subprocess.CompletedProcess:
    """Run docker compose with the correct project name and env file."""
    env_file = Path(__file__).parent / f".env.{variant}"
    cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "-p",
        _project_name(variant),
    ]
    if env_file.exists():
        cmd += ["--env-file", str(env_file)]
    cmd += list(args)
    logger.info("  $ %s", " ".join(cmd[-4:]))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 and check:
        logger.error("[%s] compose failed: %s", variant, result.stderr.strip())
        result.check_returncode()
    return result


def _replace_env_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text().splitlines()
    replaced = False
    updated: list[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            updated.append(f"{key}={value}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"{key}={value}")
    path.write_text("\n".join(updated) + "\n")


def _refresh_gotosocial_internal_token(variant: str) -> None:
    """Mint a GoToSocial token that works from MCP containers.

    The normal bootstrap talks to the host-mapped GoToSocial URL. On fresh
    isolated stacks, that token verifies from the host but can be rejected
    when the MCP server calls the Docker-internal ``http://gotosocial:8080``.
    Mint once from inside the stack network, persist it to the stack env file,
    then recreate only the GoToSocial MCP server.
    """
    env_file = Path(__file__).parent / f".env.{variant}"
    container = f"{_project_name(variant)}-gotosocial-mcp-1"
    internal_script = dedent(
        r"""
        import httpx
        import re
        from urllib.parse import parse_qs, urlparse

        base = "http://gotosocial:8080"
        email = "agentprivarena@agentprivarena.local"
        password = "AgentPrivArena-Admin-2026!@#"
        redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

        app_resp = httpx.post(
            base + "/api/v1/apps",
            data={
                "client_name": "agentprivarena-internal",
                "redirect_uris": redirect_uri,
                "scopes": "read write",
            },
            timeout=20.0,
        )
        app_resp.raise_for_status()
        app = app_resp.json()
        auth_params = {
            "response_type": "code",
            "client_id": app["client_id"],
            "redirect_uri": redirect_uri,
            "scope": "read write",
        }

        def inject_cookie(client, response):
            match = re.match(r"([^=]+)=([^;]+)", response.headers.get("set-cookie", ""))
            if match:
                client.cookies.set(match.group(1), match.group(2))

        with httpx.Client(timeout=20.0, follow_redirects=False) as client:
            response = client.get(base + "/oauth/authorize", params=auth_params)
            inject_cookie(client, response)
            response = client.post(
                base + "/auth/sign_in",
                data={"username": email, "password": password},
            )
            inject_cookie(client, response)
            response = client.get(base + "/oauth/authorize", params=auth_params)
            inject_cookie(client, response)
            consent = client.post(base + "/oauth/authorize", data={})
            code = parse_qs(urlparse(consent.headers.get("location", "")).query).get(
                "code", [None]
            )[0]
            if not code:
                raise RuntimeError(
                    "GoToSocial OAuth consent did not return an authorization code."
                )

        token_resp = httpx.post(
            base + "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "client_id": app["client_id"],
                "client_secret": app["client_secret"],
                "redirect_uri": redirect_uri,
            },
            timeout=20.0,
        )
        token_resp.raise_for_status()
        token = token_resp.json().get("access_token", "")
        if not token:
            raise RuntimeError("GoToSocial token exchange returned no access_token.")

        verify = httpx.get(
            base + "/api/v1/accounts/verify_credentials",
            headers={"Authorization": "Bearer " + token},
            timeout=20.0,
        )
        if verify.status_code != 200:
            raise RuntimeError(
                f"GoToSocial internal token rejected: HTTP {verify.status_code}"
            )
        print(token)
        """
    )
    minted = subprocess.run(
        ["docker", "exec", container, "python", "-c", internal_script],
        capture_output=True,
        text=True,
        check=False,
    )
    if minted.returncode != 0:
        raise RuntimeError(
            f"GoToSocial internal token mint failed for {variant}: "
            f"{minted.stderr.strip()}"
        )
    token = minted.stdout.strip()
    if len(token) < 20:
        raise RuntimeError(f"GoToSocial internal token for {variant} is malformed.")

    _replace_env_value(env_file, "GOTOSOCIAL_TOKEN", token)
    _compose(variant, "up", "-d", "--force-recreate", "gotosocial-mcp")


def make_config(variant: str, offset: int) -> Config:
    """Create a Config with port-offset URLs for an isolated stack."""
    return Config(
        bookstack_url=f"http://localhost:{3000 + offset}",
        mattermost_url=f"http://localhost:{8065 + offset}",
        rocketchat_url=f"http://localhost:{3100 + offset}",
        mailpit_api_url=f"http://localhost:{8025 + offset}",
        mailpit_smtp_host="localhost",
        mailpit_smtp_port=1025 + offset,
        gotosocial_url=f"http://localhost:{4000 + offset}",
        radicale_url=f"http://localhost:{5232 + offset}",
        google_drive_artifact_root=Path(
            f".agent_tmp/agentprivarena_artifacts/{variant}/google_drive"
        ),
        docker_network=f"{_project_name(variant)}_agentprivarena-net",
    )


def write_env_file(variant: str, offset: int, config: Config) -> Path:
    """Write per-stack .env file with port variables + service credentials."""
    env_path = Path(__file__).parent / f".env.{variant}"
    lines = [
        f"# Auto-generated for stack: {variant} (offset={offset})",
        "",
        "# Host port mappings (consumed by docker-compose.yml)",
    ]
    for var, base_port in BASE_PORTS.items():
        lines.append(f"{var}={base_port + offset}")

    lines += [
        "",
        "# Service credentials (consumed by MCP server containers)",
        f"BOOKSTACK_TOKEN_ID={config.bookstack_token_id}",
        f"BOOKSTACK_TOKEN_SECRET={config.bookstack_token_secret}",
        f"MATTERMOST_USER={config.mattermost_user}",
        f"MATTERMOST_PASSWORD={config.mattermost_password}",
        f"ROCKETCHAT_USER={config.rocketchat_user}",
        f"ROCKETCHAT_PASSWORD={config.rocketchat_password}",
        f"GOTOSOCIAL_TOKEN={config.gotosocial_token}",
        f"RADICALE_USER={config.radicale_user}",
        f"RADICALE_PASSWORD={config.radicale_password}",
        f"GOOGLE_DRIVE_ARTIFACT_HOST_ROOT=../{config.google_drive_artifact_root}",
        "",
    ]
    env_path.write_text("\n".join(lines))
    logger.info("Wrote %s", env_path)
    return env_path


def _provision_bookstack_token(variant: str, config: Config) -> None:
    """Auto-create the BookStack API token on a fresh database.

    Each isolated stack has its own bookstack-db volume, so the token
    from the original stack doesn't exist. This runs ``php artisan
    tinker`` inside the container to create it non-interactively.
    """
    php = (
        "$u = \\BookStack\\Users\\Models\\User::find(1);"
        "$t = new \\BookStack\\Api\\ApiToken();"
        '$t->name = "agentprivarena";'
        f'$t->token_id = "{config.bookstack_token_id}";'
        f'$t->secret = \\Hash::make("{config.bookstack_token_secret}");'
        '$t->expires_at = "2099-12-31";'
        "$t->user_id = $u->id;"
        "$t->save();"
        'echo "Token created: " . $t->token_id;'
    )
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "-p",
            _project_name(variant),
            "exec",
            "-T",
            "bookstack",
            "bash",
            "-c",
            f"cd /app/www && php artisan tinker --execute='{php}'",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"BookStack token provisioning failed for {variant}: {result.stderr}"
        )
    logger.info("[%s] %s", variant, result.stdout.strip())


def setup_stack(variant: str, offset: int) -> Config:
    """Bring up one isolated stack and bootstrap it."""
    from agentprivarena.bootstrap import bootstrap_all

    logger.info("=== Setting up stack: %s (offset=%d) ===", variant, offset)

    # 1. Create config with offset ports
    config = make_config(variant, offset)

    # 2. Write initial .env (credentials use defaults, tokens empty)
    write_env_file(variant, offset, config)

    # 3. Start infrastructure
    logger.info("[%s] Starting infrastructure...", variant)
    _compose(variant, "up", "-d", *INFRA_SERVICES)

    # 4. Wait for all infrastructure to be healthy.
    #    BookStack and RocketChat have long start_periods (60-90s),
    #    so we allow up to 5 minutes total.
    import json
    import time

    logger.info("[%s] Waiting for services to be healthy...", variant)
    healthy_count = 0
    for attempt in range(60):  # 60 × 5s = 5 minutes max
        result = _compose(variant, "ps", "--format", "json", check=False)
        if result.returncode == 0 and result.stdout.strip():
            try:
                raw = result.stdout.strip()
                services = json.loads(f"[{raw.replace(chr(10), ',')}]")
                healthy_count = sum(1 for s in services if s.get("Health") == "healthy")
                total_infra = len(INFRA_SERVICES)
                if healthy_count >= total_infra:
                    logger.info("[%s] All %d services healthy.", variant, healthy_count)
                    break
            except (json.JSONDecodeError, TypeError):
                pass
        if attempt % 6 == 0:
            logger.info(
                "[%s] Waiting for health... (%d/%d healthy, attempt %d)",
                variant,
                healthy_count,
                len(INFRA_SERVICES),
                attempt,
            )
        time.sleep(5)
    else:
        logger.warning("[%s] Timed out waiting for health. Proceeding anyway.", variant)

    # 5. Auto-provision BookStack API token on the fresh database
    logger.info("[%s] Provisioning BookStack token...", variant)
    _provision_bookstack_token(variant, config)

    # 6. Bootstrap (provision remaining tokens) — needs COMPOSE_PROJECT_NAME for
    #    docker compose exec commands inside bootstrap.py
    logger.info("[%s] Running bootstrap...", variant)
    old_project = os.environ.get("COMPOSE_PROJECT_NAME")
    os.environ["COMPOSE_PROJECT_NAME"] = _project_name(variant)
    # Override host URLs for bootstrap to talk to this stack
    os.environ["BOOKSTACK_URL"] = config.bookstack_url
    os.environ["MATTERMOST_URL"] = config.mattermost_url
    os.environ["ROCKETCHAT_URL"] = config.rocketchat_url
    os.environ["MAILPIT_API_URL"] = config.mailpit_api_url
    os.environ["GOTOSOCIAL_URL"] = config.gotosocial_url
    os.environ["RADICALE_URL"] = config.radicale_url
    os.environ["MAILPIT_SMTP_PORT"] = str(config.mailpit_smtp_port)

    config = bootstrap_all(config, COMPOSE_FILE)

    if old_project is not None:
        os.environ["COMPOSE_PROJECT_NAME"] = old_project
    else:
        os.environ.pop("COMPOSE_PROJECT_NAME", None)

    # 6. Re-write .env with provisioned tokens
    write_env_file(variant, offset, config)

    # 7. Start MCP servers
    logger.info("[%s] Starting MCP servers...", variant)
    _compose(variant, "up", "-d", "--force-recreate", *MCP_SERVICES)
    logger.info("[%s] Refreshing GoToSocial token for internal MCP calls...", variant)
    _refresh_gotosocial_internal_token(variant)

    logger.info("[%s] Stack ready.", variant)
    return config


def run_evaluation(
    variant: str,
    offset: int,
    *,
    results_template: str,
    read_policy: str,
    extra_flags: list[str] | None = None,
) -> subprocess.Popen:
    """Run evaluation for one level as a subprocess."""
    config = make_config(variant, offset)
    results_dir = results_template.format(
        variant=variant,
        level=LEVEL_LABELS[variant],
        read_policy=read_policy,
    )

    env = os.environ.copy()
    # Override service URLs for this stack
    env["BOOKSTACK_URL"] = config.bookstack_url
    env["MATTERMOST_URL"] = config.mattermost_url
    env["ROCKETCHAT_URL"] = config.rocketchat_url
    env["MAILPIT_API_URL"] = config.mailpit_api_url
    env["MAILPIT_SMTP_HOST"] = config.mailpit_smtp_host
    env["MAILPIT_SMTP_PORT"] = str(config.mailpit_smtp_port)
    env["GOTOSOCIAL_URL"] = config.gotosocial_url
    env["RADICALE_URL"] = config.radicale_url
    env["GOOGLE_DRIVE_ARTIFACT_ROOT"] = str(config.google_drive_artifact_root)
    env["DOCKER_NETWORK"] = config.docker_network

    # Point at this stack's .env for token credentials
    env_file = Path(__file__).parent / f".env.{variant}"
    if env_file.exists():
        # Parse and inject credentials into env
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env.setdefault(k, v)

    cmd = [
        sys.executable,
        "-m",
        "agentprivarena",
        "run",
        "--prompt-variant",
        variant,
        "--results-dir",
        results_dir,
        "--read-policy",
        read_policy,
        "--resume",
    ]
    if variant == "ci_audit":
        cmd.append("--enable-privacy-analyzer")
    if extra_flags:
        cmd.extend(extra_flags)

    logger.info("[%s] Starting evaluation → %s", variant, results_dir)
    log_file = Path(results_dir) / "_runner.log"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    logger.info("[%s] PID %d, log: %s", variant, proc.pid, log_file)
    return proc


def teardown_all(levels: list[tuple[str, int, int]]) -> None:
    """Stop and remove selected stacks."""
    for variant, _idx, _offset in levels:
        logger.info("Tearing down stack: %s", variant)
        _compose(variant, "down", "-v", check=False)
    logger.info("Selected stacks torn down.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel AgentPrivArena runner")
    parser.add_argument(
        "--teardown", action="store_true", help="Tear down selected stacks"
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Setup stacks but don't run evaluations",
    )
    parser.add_argument(
        "--run-only", action="store_true", help="Skip setup, just launch evaluations"
    )
    parser.add_argument(
        "--levels",
        default=",".join(DEFAULT_LEVELS),
        help=(
            "Comma-separated levels to run. Accepts l0/l1/l2/l3 or "
            "baseline/privacy_enhanced/ci_reasoning/ci_audit."
        ),
    )
    parser.add_argument(
        "--base-offset",
        type=int,
        default=0,
        help=(
            "Port offset base. Use a non-zero value, e.g. 2000, when the default "
            "agentprivarena stack is already running."
        ),
    )
    parser.add_argument(
        "--results-template",
        default="results/v5_{variant}",
        help=(
            "Results directory template. Supports {variant}, {level}, and "
            "{read_policy}; example: results/live_{level}_{read_policy}."
        ),
    )
    parser.add_argument(
        "--read-policy",
        default="natural",
        choices=["natural", "forced_oracle"],
        help=(
            "Retrieval policy to pass through to agentprivarena run. "
            "Use forced_oracle for the forced-read ablation."
        ),
    )
    args = parser.parse_args()
    levels = _selected_levels(args.levels, args.base_offset)

    if args.teardown:
        teardown_all(levels)
        return

    if not args.run_only:
        _check_ports_available(levels)
        # Setup stacks sequentially (bootstrap can't run in parallel
        # because it modifies os.environ for COMPOSE_PROJECT_NAME)
        for variant, idx, offset in levels:
            setup_stack(variant, offset)

    if args.setup_only:
        logger.info("Selected stacks set up. Run with --run-only to start evaluations.")
        return

    # Launch selected evaluation processes in parallel.
    procs = []
    for variant, idx, offset in levels:
        proc = run_evaluation(
            variant,
            offset,
            results_template=args.results_template,
            read_policy=args.read_policy,
        )
        procs.append((variant, proc))

    logger.info("Selected evaluations launched. Waiting for completion...")
    for variant, proc in procs:
        proc.wait()
        logger.info("[%s] finished with exit code %d", variant, proc.returncode)

    logger.info("All evaluations complete.")


if __name__ == "__main__":
    main()
