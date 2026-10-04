"""Build MCP config dicts for the OpenHands agent from task specs."""

from __future__ import annotations

from agentprivarena.config import Config
from agentprivarena.runner.tool_registry import known_service_names


def build_mcp_config(
    dependencies: list[str],
    config: Config,
    *,
    strict: bool = True,
) -> dict:
    """Build an MCP config dict from task dependencies.

    The MCP config uses Docker-internal URLs so the agent container
    (on agentprivarena-net) can reach the MCP servers by DNS name.

    Args:
        dependencies: List of MCP server names
            (e.g., ["bookstack", "gotosocial"])
        config: Platform configuration with MCP server URLs

    Returns:
        Dict suitable for Agent(mcp_config=...)
    """
    servers = {}
    unknown = []
    missing_url = []
    known = set(known_service_names())
    for mcp_name in dependencies:
        if mcp_name not in known:
            unknown.append(mcp_name)
            continue
        url = config.mcp_server_urls.get(mcp_name)
        if url:
            servers[mcp_name] = {"url": url}
        else:
            missing_url.append(mcp_name)
    if strict and unknown:
        expected = ", ".join(known_service_names())
        raise ValueError(
            f"Unknown MCP task dependencies {unknown}; expected one of: {expected}"
        )
    if strict and missing_url:
        raise ValueError(
            f"No MCP server URL configured for known task dependencies {missing_url}"
        )
    return {"mcpServers": servers}
